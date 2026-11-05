//! What this box can actually stream from memory, at 1 to 8 threads.
//!
//! Batch-1 generation reads every weight byte once per token, so the
//! generation rate has a hard ceiling at `bandwidth / weight bytes` and no
//! amount of kernel work can pass it. This measures the numerator. It needs no
//! model and no thread pool, which is why it is an example here rather than a
//! mode of the runner.
//!
//! What is measured is a sequential read of a buffer far larger than the 36
//! MiB of shared L3, with each thread streaming its own contiguous region of
//! one allocation, which is the access pattern a matvec over a weight tensor
//! has. Reads are accumulated into eight independent partial sums so the
//! loop is not limited by a dependency chain, and the result is consumed, so
//! nothing is optimised away.
//!
//! ```sh
//! cargo run --release -p llm-core --example bench_bandwidth
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Total footprint, the same at every thread count so the comparison is about
/// the threads and not about the working set. 1 GiB is 28 times the L3.
const BYTES: usize = 1 << 30;
/// Each thread streams for this long. Seconds of work, so that a scheduler
/// hiccup is a per cent and not a factor.
const WINDOW: Duration = Duration::from_secs(2);
const REPEATS: usize = 3;
/// What the Q4_K_M model's weights come to, from `docs/benchmarks.md`: the
/// mapped file is 2.52 GiB and generating one token reads all of it.
const WEIGHT_GIB: f64 = 2.52;

fn main() {
    let threads: Vec<usize> = std::env::args()
        .nth(1)
        .map(|a| {
            a.split(',')
                .map(|n| n.trim().parse().expect("thread count"))
                .collect()
        })
        .unwrap_or_else(|| (1..=8).collect());

    // One allocation, touched once here so the pages are resident before
    // anything is timed.
    let mut buf = vec![0u64; BYTES / 8];
    for (i, v) in buf.iter_mut().enumerate() {
        *v = i as u64;
    }

    println!(
        "{} MiB buffer, {:.0}s window, median of {REPEATS}, weights {WEIGHT_GIB} GiB",
        BYTES >> 20,
        WINDOW.as_secs_f64(),
    );
    println!("threads  GiB/s   (min .. max)        implied generation tok/s");
    for nth in threads {
        let mut rates: Vec<f64> = (0..REPEATS).map(|_| stream(&buf, nth)).collect();
        rates.sort_by(f64::total_cmp);
        let mid = rates[rates.len() / 2];
        println!(
            "{nth:>7}  {mid:6.2}  ({:.2} .. {:.2})        {:.2}",
            rates[0],
            rates[rates.len() - 1],
            mid / WEIGHT_GIB,
        );
    }
}

/// Reads `buf` with `nth` threads for one window and returns GiB/s.
fn stream(buf: &[u64], nth: usize) -> f64 {
    let read = AtomicU64::new(0);
    let per = buf.len() / nth;
    let start = Instant::now();
    let deadline = start + WINDOW;
    std::thread::scope(|s| {
        for ith in 0..nth {
            let region = &buf[ith * per..(ith + 1) * per];
            let read = &read;
            s.spawn(move || {
                let mut bytes = 0u64;
                let mut sink = 0u64;
                while Instant::now() < deadline {
                    sink = sink.wrapping_add(sum(region));
                    bytes += (region.len() * 8) as u64;
                }
                read.fetch_add(bytes, Ordering::Relaxed);
                // Consumed so the loop above cannot be optimised away.
                std::hint::black_box(sink);
            });
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    read.load(Ordering::Relaxed) as f64 / elapsed / (1u64 << 30) as f64
}

/// Eight independent accumulators, so the loop is limited by the loads and not
/// by the latency of one add chain.
fn sum(region: &[u64]) -> u64 {
    let mut acc = [0u64; 8];
    for block in region.as_chunks::<8>().0 {
        for (a, &v) in acc.iter_mut().zip(block) {
            *a = a.wrapping_add(v);
        }
    }
    acc.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}
