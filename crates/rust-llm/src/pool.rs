//! A persistent worker pool, and the two shapes of parallel work the forward
//! pass has.
//!
//! One pool is built at load time and lives as long as the runtime. A job is
//! published once and run by every worker plus the submitting thread; `run`
//! returns only after all of them are finished with it. That is what lets a job
//! borrow the caller's stack and write into disjoint pieces of a caller-owned
//! slice without allocating anything per call.
//!
//! Threads are not spawned per matmul: at 32 cores and a few hundred matmuls
//! per token, spawning would cost more than the work and would allocate a stack
//! every time.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;

use llm_core::quant::{Activations, GgmlType};
use llm_core::{KernelError, Kernels};

/// A job as the workers see it: `(worker index, worker count)`.
type Job = dyn Fn(usize, usize) + Sync;

struct Shared {
    /// Every thread, the submitter included, meets here before the job runs.
    start: Barrier,
    /// And here when it is done, which is what bounds the job's lifetime.
    end: Barrier,
    /// Written by the submitter before `start`, read by the workers after it.
    /// The barrier is the only synchronisation this needs.
    job: UnsafeCell<Option<*const Job>>,
    stop: AtomicBool,
}

// SAFETY: `job` is only written between two barriers that every thread passes
// through, so no two threads ever touch it at the same time.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    threads: usize,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Builds a pool that runs jobs on `threads` threads in total, the
    /// submitting thread being one of them.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            start: Barrier::new(threads),
            end: Barrier::new(threads),
            job: UnsafeCell::new(None),
            stop: AtomicBool::new(false),
        });
        let workers = (1..threads)
            .map(|ith| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("llm-{ith}"))
                    .spawn(move || worker(&shared, ith, threads))
                    .expect("spawn worker thread")
            })
            .collect();
        Self {
            threads,
            shared,
            workers,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `f` on every thread and waits for all of them.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        if self.threads == 1 {
            f(0, 1);
            return;
        }
        let borrowed: *const (dyn Fn(usize, usize) + Sync + '_) = f;
        // SAFETY: the pointer is published while this thread is inside the two
        // barriers below, and this function does not return until every worker
        // has passed `end`, so `f` outlives every use of it. That is what the
        // cast to `'static` stands on; nothing else writes the cell between
        // those barriers.
        unsafe {
            let erased: *const Job = std::mem::transmute(borrowed);
            *self.shared.job.get() = Some(erased);
        }
        self.shared.start.wait();
        f(0, self.threads);
        self.shared.end.wait();
    }

    /// `out[r] = row_r . a` for a `[k, out.len()]` quantized weight tensor,
    /// with the rows split between the threads.
    ///
    /// The kernel is looked up once here rather than inside the loop, and an
    /// unsupported type is refused before any thread starts.
    pub fn matvec(
        &self,
        kernels: &Kernels,
        ty: GgmlType,
        w: &[u8],
        k: usize,
        a: &Activations,
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        kernels.vec_dot(ty)?;
        let stride = ty.bytes_for(k as u64).expect("row size fits in a u64") as usize;
        let rows = out.len();
        let out = Scatter::new(out);
        self.run(&|ith, nth| {
            let (lo, hi) = split(rows, ith, nth);
            if lo == hi {
                return;
            }
            // SAFETY: the ranges `split` hands out are disjoint, so no two
            // threads write the same rows.
            let chunk = unsafe { out.at(lo, hi - lo) };
            kernels
                .matvec(ty, &w[lo * stride..], k, a, chunk)
                .expect("the kernel was resolved before the threads started");
        });
        Ok(())
    }

    /// Splits `out` into chunks of `out_chunk` and `work` into chunks of
    /// `work_chunk`, then runs `f(i, &mut out[i], &mut work[i])` for every
    /// chunk, distributed over the threads.
    ///
    /// This is the per-head shape: one attention head or one delta-net value
    /// head owns one output chunk and one piece of scratch (its score row, or
    /// its recurrent state) and touches nothing else.
    pub fn chunks(
        &self,
        out: &mut [f32],
        out_chunk: usize,
        work: &mut [f32],
        work_chunk: usize,
        f: impl Fn(usize, &mut [f32], &mut [f32]) + Sync,
    ) {
        let n = out.len() / out_chunk;
        assert_eq!(
            out.len(),
            n * out_chunk,
            "output does not divide into chunks"
        );
        assert_eq!(
            work.len(),
            n * work_chunk,
            "scratch does not divide into {n} chunks"
        );
        let out = Scatter::new(out);
        let work = Scatter::new(work);
        self.run(&|ith, nth| {
            let (lo, hi) = split(n, ith, nth);
            for i in lo..hi {
                // SAFETY: chunk `i` belongs to exactly one iteration of one
                // thread, because `split` partitions `0..n`.
                let (o, w) = unsafe {
                    (
                        out.at(i * out_chunk, out_chunk),
                        work.at(i * work_chunk, work_chunk),
                    )
                };
                f(i, o, w);
            }
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if self.workers.is_empty() {
            return;
        }
        self.shared.stop.store(true, Ordering::Relaxed);
        // The workers are parked on `start`; releasing it lets them see `stop`
        // and return. They do not reach `end`, so this thread must not either.
        self.shared.start.wait();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared, ith: usize, nth: usize) {
    loop {
        shared.start.wait();
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: see `Pool::run`. The job was published before this barrier
        // and stays alive until every thread has passed `end`.
        let job = unsafe { (*shared.job.get()).expect("a job was published") };
        unsafe { (*job)(ith, nth) };
        shared.end.wait();
    }
}

/// A `&mut [f32]` several threads may split between them.
///
/// Handing out `&mut` from `&self` is sound only because every caller derives
/// its range from the worker index, so the ranges are disjoint; that invariant
/// lives in the two methods above and nowhere else.
#[derive(Clone, Copy)]
struct Scatter {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: see the type's documentation; the ranges are disjoint by
// construction.
unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

impl Scatter {
    fn new(s: &mut [f32]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// # Safety
    /// `start..start + len` must not overlap any other live slice from the
    /// same `Scatter`.
    // Handing out `&mut` from `&self` is the whole point of the type, and the
    // disjointness that makes it sound is the caller's contract above.
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, start: usize, len: usize) -> &mut [f32] {
        debug_assert!(start + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

/// Worker `ith` of `nth` takes this slice of `0..n`, the remainder spread over
/// the first few workers rather than piled onto the last one.
fn split(n: usize, ith: usize, nth: usize) -> (usize, usize) {
    let per = n / nth;
    let rem = n % nth;
    let lo = ith * per + ith.min(rem);
    let hi = lo + per + usize::from(ith < rem);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn split_partitions_the_range() {
        for n in 0..40usize {
            for nth in 1..9usize {
                let mut at = 0;
                for ith in 0..nth {
                    let (lo, hi) = split(n, ith, nth);
                    assert_eq!(lo, at, "n={n} nth={nth} ith={ith}");
                    assert!(hi >= lo);
                    at = hi;
                }
                assert_eq!(at, n, "n={n} nth={nth}");
            }
        }
    }
}
