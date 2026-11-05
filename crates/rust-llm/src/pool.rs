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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use llm_core::quant::{Activations, GgmlType};
use llm_core::{KernelError, Kernels};

/// A job as the workers see it: `(worker index, worker count)`.
type Job = dyn Fn(usize, usize) + Sync;

/// How long a thread spins before it gives the CPU up.
///
/// `spin_loop` is tens of cycles, so this is roughly twenty microseconds:
/// longer than the serial stretch between two jobs, which is what a worker is
/// usually waiting through, and short enough that a thread which has to wait
/// for a real reason parks instead of burning quota that the thread it waits
/// for needs. The forward pass runs 161 jobs per token, so this is the knob
/// that decides whether the threads stay hot between them.
const SPINS: u32 = 2000;

struct Shared {
    /// Incremented once per published job. A worker waits for it to change,
    /// which is both the hand-out and the release of the job pointer below.
    seq: AtomicU64,
    /// Workers still inside the current job. The submitter waits for zero.
    pending: AtomicUsize,
    /// Written by the submitter before `seq` is bumped, read by the workers
    /// after they observe the bump, which is what orders the two.
    job: UnsafeCell<Option<*const Job>>,
    stop: AtomicBool,
    /// Set by a worker whose job panicked, read by the submitter once
    /// `pending` is zero.
    failed: AtomicBool,
}

// SAFETY: `job` is written while every worker is waiting for `seq` to change,
// and read only after that bump has been observed, so no two threads ever
// touch it at the same time.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    /// Hands out a job: release the pointer through `seq`, then make sure a
    /// worker that has already parked comes back.
    ///
    /// `unpark` on a thread that is still spinning is an atomic store and no
    /// syscall, and it leaves a token that makes that thread's next `park`
    /// return at once, which the loop in `wait_for_job` re-checks. That is why
    /// this does not have to know which workers are parked, and why there is
    /// no window in which a worker parks just after being skipped.
    fn publish(&self, workers: &[JoinHandle<()>]) {
        self.seq.fetch_add(1, Ordering::Release);
        for w in workers {
            w.thread().unpark();
        }
    }

    /// Waits for a job newer than `seen` and returns the new sequence number.
    fn wait_for_job(&self, seen: u64) -> u64 {
        for _ in 0..SPINS {
            let now = self.seq.load(Ordering::Acquire);
            if now != seen {
                return now;
            }
            std::hint::spin_loop();
        }
        loop {
            let now = self.seq.load(Ordering::Acquire);
            if now != seen {
                return now;
            }
            std::thread::park();
        }
    }

    /// Waits until every worker is out of the current job.
    ///
    /// The submitter spins and then yields rather than parking: it is waiting
    /// for threads that are running, and under a CPU quota what they need is
    /// the quota back, which `yield_now` gives them while keeping the wake-up
    /// free.
    fn wait_for_workers(&self) {
        for _ in 0..SPINS {
            if self.pending.load(Ordering::Acquire) == 0 {
                return;
            }
            std::hint::spin_loop();
        }
        while self.pending.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }
    }
}

pub struct Pool {
    threads: usize,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    stats: Stats,
}

/// What the submitter spends on getting a job run, as opposed to running it.
///
/// Two numbers, because they say different things. `hand out` is publishing
/// the job and waking whoever parked, which the submitter pays before it
/// starts its own share. `join` is what it waits after finishing that share:
/// the slowest worker, including however long that worker took to wake up.
/// Off they are one relaxed load per job; on they are four `Instant::now()`
/// calls, against a job that is tens of microseconds.
#[derive(Default)]
pub struct Stats {
    on: AtomicBool,
    wake_ns: AtomicU64,
    join_ns: AtomicU64,
    jobs: AtomicU64,
}

impl Stats {
    /// `(jobs, wake nanoseconds, join nanoseconds)`.
    pub fn read(&self) -> (u64, u64, u64) {
        (
            self.jobs.load(Ordering::Relaxed),
            self.wake_ns.load(Ordering::Relaxed),
            self.join_ns.load(Ordering::Relaxed),
        )
    }

    pub fn clear(&self) {
        self.wake_ns.store(0, Ordering::Relaxed);
        self.join_ns.store(0, Ordering::Relaxed);
        self.jobs.store(0, Ordering::Relaxed);
    }
}

impl Pool {
    /// Builds a pool that runs jobs on `threads` threads in total, the
    /// submitting thread being one of them.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            seq: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
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
            stats: Stats::default(),
        }
    }

    /// The barrier accounting, for `rust-llm profile`.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn measure(&self, on: bool) {
        self.stats.on.store(on, Ordering::Relaxed);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `f` on every thread and waits for all of them.
    ///
    /// A panic in `f` is carried out of the job rather than out of this
    /// function directly. If the panic escaped here, the workers would be left
    /// inside a job nobody waits for and `Drop` would join threads that are
    /// still running, so an index error in the forward pass would be a hang
    /// instead of a stack trace. The panic is therefore caught, every thread
    /// is waited for, and only then is it resumed. The buffers the job was
    /// writing are left half finished, which is why this resumes rather than
    /// reporting: the caller's results are not usable either way.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        if self.threads == 1 {
            f(0, 1);
            return;
        }
        let borrowed: *const (dyn Fn(usize, usize) + Sync + '_) = f;
        // SAFETY: the pointer is published after every worker has left the
        // previous job, and this function does not return until every worker
        // has left this one, so `f` outlives every use of it. That is what the
        // cast to `'static` stands on; nothing else writes the cell in
        // between.
        unsafe {
            let erased: *const Job = std::mem::transmute(borrowed);
            *self.shared.job.get() = Some(erased);
        }
        let timed = self.stats.on.load(Ordering::Relaxed);
        let at = timed.then(Instant::now);
        self.shared
            .pending
            .store(self.threads - 1, Ordering::Relaxed);
        self.shared.publish(&self.workers);
        if let Some(at) = at {
            self.stats
                .wake_ns
                .fetch_add(at.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let mine = catch(|| f(0, self.threads));
        let at = timed.then(Instant::now);
        self.shared.wait_for_workers();
        if let Some(at) = at {
            self.stats
                .join_ns
                .fetch_add(at.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.stats.jobs.fetch_add(1, Ordering::Relaxed);
        }
        if let Err(panic) = mine {
            std::panic::resume_unwind(panic);
        }
        // A worker's panic has already printed its own message and location.
        assert!(
            !self.shared.failed.swap(false, Ordering::Relaxed),
            "a worker thread panicked in this job; its panic is logged above"
        );
    }

    /// `out[r] = row_r . a` for a `[k, out.len()]` quantized weight tensor,
    /// with the rows split between the threads.
    pub fn matvec(
        &self,
        kernels: &Kernels,
        ty: GgmlType,
        w: &[u8],
        k: usize,
        a: &Activations,
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        self.matvec_group(kernels, k, a, [Matvec { ty, w, out }])
    }

    /// Several matvecs that read the same activations, as **one** job over the
    /// rows of all of them.
    ///
    /// This is a barrier count, not a kernel change. A job costs two barriers
    /// and, under a CPU quota, the wait for whichever worker the scheduler
    /// gets around to last; the forward pass has groups of projections that
    /// read the same normalised input, and run as separate jobs they pay that
    /// twice, three times or four times for no reason. The rows of the group
    /// are one range that the split divides, so a group of a 8192-row tensor
    /// and two 32-row ones also balances, which three separate jobs could not.
    ///
    /// The result does not depend on the grouping or on the thread count: an
    /// output row is still computed by exactly one worker from start to
    /// finish, over the same activations, which are quantized once here
    /// instead of once per tensor.
    ///
    /// The kernels are looked up before any thread starts, so an unsupported
    /// type is an error rather than a panic inside a job.
    ///
    /// With a batch of `t` activation rows in `a`, every output is `t` rows of
    /// `rows` values, token-major, and a claimed block of weight rows is run
    /// against all `t` of them before the next block is touched. That is where
    /// the batch's win comes from: the block is sized to sit in cache, so its
    /// weight bytes cross the memory bus once for the whole batch instead of
    /// once per token. It is also why the result is the same as `t` separate
    /// calls, bit for bit: an output is still one weight row dotted with one
    /// activation row, in the one order.
    pub fn matvec_group<const N: usize>(
        &self,
        kernels: &Kernels,
        k: usize,
        a: &Activations,
        group: [Matvec<'_>; N],
    ) -> Result<(), KernelError> {
        for m in &group {
            kernels.vec_dot(m.ty)?;
        }
        let t = a.t();
        let parts = group.map(|m| {
            assert_eq!(
                m.out.len() % t,
                0,
                "an output of {} values is not {t} rows",
                m.out.len()
            );
            Part {
                ty: m.ty,
                w: m.w,
                stride: m.ty.bytes_for(k as u64).expect("row size fits in a u64") as usize,
                rows: m.out.len() / t,
                out: Scatter::new(m.out),
            }
        });
        let total: usize = parts.iter().map(|p| p.rows).sum();
        let parts = &parts;
        // The block the workers claim. Small enough that the last one does
        // not decide the job's length, large enough that the weight stream
        // stays sequential and the atomic is noise: 64 rows of this model's
        // largest tensor are 90 KiB of weights, tens of microseconds.
        let block = (total / (self.threads * 4)).clamp(1, 64);
        let claimed = AtomicUsize::new(0);
        self.run(&|_ith, _nth| {
            loop {
                let from = claimed.fetch_add(block, Ordering::Relaxed);
                if from >= total {
                    return;
                }
                let to = (from + block).min(total);
                let mut base = 0;
                for p in parts {
                    let (from, to) = (from.max(base), to.min(base + p.rows));
                    if from < to {
                        p.run(kernels, k, a, from - base, to - from);
                    }
                    base += p.rows;
                }
            }
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
        let claimed = AtomicUsize::new(0);
        self.run(&|_ith, _nth| {
            loop {
                let i = claimed.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return;
                }
                // SAFETY: chunk `i` is handed to exactly one worker, because
                // `claimed` is advanced atomically.
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
        // The workers are waiting for the next job; one more bump of `seq`
        // lets them see `stop` and return. They never look at the job
        // pointer, and nothing waits for `pending` here.
        self.shared.publish(&self.workers);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Runs `f`, returning its panic instead of unwinding.
///
/// The job is behind a shared reference and holds raw pointers, so it is not
/// `UnwindSafe` by the compiler's reckoning. What that lint protects against is
/// observing half-written state after catching a panic; here the panic is
/// resumed or turned into one as soon as the barrier is passed, and nothing
/// reads those buffers in between.
fn catch<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
}

fn worker(shared: &Shared, ith: usize, nth: usize) {
    let mut seen = 0;
    loop {
        seen = shared.wait_for_job(seen);
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: see `Pool::run`. The job was published before the bump of
        // `seq` this thread has just observed, and stays alive until every
        // thread has left it, which is what `pending` below reports.
        let job = unsafe { (*shared.job.get()).expect("a job was published") };
        if catch(|| unsafe { (*job)(ith, nth) }).is_err() {
            // The submitter turns this into its own panic once every worker is
            // out. This thread must stay alive and report itself out, or the
            // submitter waits on it forever.
            shared.failed.store(true, Ordering::Relaxed);
        }
        // Release, so that whatever the job wrote, and the flag above, are
        // visible to the submitter as soon as it sees this.
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

/// One matvec of a group: a quantized weight tensor and where its rows go.
pub struct Matvec<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub out: &'a mut [f32],
}

/// A [`Matvec`] as the workers see it, with the output handed out by range.
struct Part<'a> {
    ty: GgmlType,
    w: &'a [u8],
    stride: usize,
    rows: usize,
    out: Scatter,
}

impl Part<'_> {
    /// Computes rows `at..at + len` of this part, for every row of the batch.
    ///
    /// One token goes straight through [`Kernels::matvec`], which is the path
    /// generation has always taken. A batch runs one pass per token over the
    /// same block of weight rows, which is where its win comes from: the block
    /// is small enough to stay in cache, so its bytes cross the memory bus
    /// once for the whole batch.
    fn run(&self, kernels: &Kernels, k: usize, a: &Activations, at: usize, len: usize) {
        let t = a.t();
        if t == 1 {
            // SAFETY: a row block is handed to exactly one worker, because
            // `claimed` is advanced atomically, and every part has its own
            // output slice.
            let chunk = unsafe { self.out.at(at, len) };
            kernels
                .matvec(self.ty, &self.w[at * self.stride..], k, a, chunk)
                .expect("the kernel was resolved before the threads started");
            return;
        }
        for tt in 0..t {
            // SAFETY: as above, and the `t` ranges here are one row block
            // in `t` disjoint rows of the output.
            let chunk = unsafe { self.out.at(tt * self.rows + at, len) };
            kernels
                .matvec_row(self.ty, &self.w[at * self.stride..], k, a, tt, chunk)
                .expect("the kernel was resolved before the threads started");
        }
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

#[cfg(test)]
mod tests {
    use super::Pool;

    /// A failing job has to come out as a panic on the submitting thread, from
    /// either side of the pool, and leave the pool usable.
    ///
    /// Before this the pool deadlocked instead: the panic skipped the second
    /// barrier, every other thread waited on it, and `Drop` then waited on the
    /// first one. A test that hangs is the symptom, which is why this test is
    /// worth its noise.
    #[test]
    fn a_panicking_job_panics_the_submitter_instead_of_hanging() {
        let pool = Pool::new(4);
        // The panics below are expected, so their messages are not printed.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        for culprit in [0usize, 1, 3] {
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.run(&|ith, _| assert!(ith != culprit, "worker {ith} is the culprit"));
            }));
            assert!(out.is_err(), "a panic on thread {culprit} was swallowed");
        }

        std::panic::set_hook(hook);

        // And the pool still works, which is what says the failed jobs left no
        // thread behind a barrier.
        let mut seen = vec![0.0f32; 4];
        let mut scratch = vec![0.0f32; 4];
        pool.chunks(&mut seen, 1, &mut scratch, 1, |i, o, _| {
            o[0] = i as f32 + 1.0
        });
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Every chunk is claimed by exactly one worker, whatever the thread
    /// count. That is the property the static split used to have by
    /// construction and the claim counter now has to be tested for: it is
    /// what keeps one head, and one output row, the work of a single worker
    /// from start to finish, and so what keeps the result independent of the
    /// thread count.
    #[test]
    fn every_chunk_is_claimed_exactly_once() {
        for nth in [1usize, 3, 4, 8] {
            let pool = Pool::new(nth);
            for n in [1usize, 7, 37] {
                let mut seen = vec![0.0f32; n];
                let mut by = vec![-1.0f32; n];
                pool.chunks(&mut seen, 1, &mut by, 1, |i, o, w| {
                    o[0] += 1.0;
                    w[0] = i as f32;
                });
                assert!(
                    seen.iter().all(|&v| v == 1.0),
                    "nth={nth} n={n}: a chunk was run twice or not at all: {seen:?}"
                );
                let want: Vec<f32> = (0..n).map(|i| i as f32).collect();
                assert_eq!(by, want, "nth={nth} n={n}: a chunk got the wrong index");
            }
        }
    }
}
