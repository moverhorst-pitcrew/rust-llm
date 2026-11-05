//! `rust-llm bench`: prompt and generation throughput, repeatably enough to
//! compare two commits.
//!
//! This is a subcommand rather than a `cargo bench` harness because the
//! measurement needs the model file and a loaded [`Runner`], which is exactly
//! what the CLI already builds, and because a bench crate would be a new
//! dependency for shape that is already here.
//!
//! Three things make a number from this comparable to another one:
//!
//! - the prompt is a fixed text, truncated to an exact token count, so the
//!   measured work does not depend on how a tokenizer splits a sentence;
//! - generation runs a fixed number of tokens through `forward` and ignores
//!   stop tokens, so it does not depend on what the model decides to say;
//! - a warm-up run is thrown away and the repeats are reported as a median
//!   with their spread. On this box the numbers move 10 to 20 per cent with
//!   whatever else is running, so a single number without its spread is not a
//!   measurement.

use std::time::{Duration, Instant};

use crate::forward::Runner;
use crate::model::Model;
use crate::sample::argmax;

/// The text every bench prompt is taken from, truncated to the requested
/// token count. Fixed on purpose: a prompt that changes changes the numbers.
pub const FILLER: &str = "Memory bandwidth, not arithmetic, is what limits a \
single stream of tokens on a CPU. A four billion parameter model quantized to \
four bits is about two and a half gibibytes of weights, and generating one \
token reads every one of those bytes exactly once, because a batch of one has \
no other row to share them with. The arithmetic is a few gigaflops against a \
machine that can do hundreds, so the processor spends its time waiting for \
cache lines. Threads help until the memory controller is saturated and then \
they stop helping, and past that point they cost more than they earn, because \
every thread that joins the pool also joins the barrier at the end of each \
job. Reading the prompt is a different shape of work: every token still reads \
every weight, but the tokens are known in advance, so the same bytes could be \
shared between several of them. That is why a prompt should be faster per \
token than generation, and why it is worth measuring the two separately \
instead of quoting one number for both.";

pub struct Options {
    pub context: usize,
    pub generate: usize,
    pub repeats: usize,
    /// Tokens per batch in the prompt phase. The generation phase is batch-1
    /// whatever this says, because there is no second token to batch with.
    pub batch: usize,
}

/// One timed pass: the prompt, then a fixed number of generated tokens.
struct Timed {
    prompt: Duration,
    generate: Duration,
}

/// Runs the benchmark for every thread count in `threads`, over one loaded
/// model, and prints one block per configuration.
///
/// The thread counts share the model and the memory map, so the load and the
/// tokenizer verification are paid once; each gets its own [`Runner`], which
/// is what owns the pool and the caches.
pub fn sweep(model: &Model<'_>, prompt: &[u32], threads: &[usize], opts: &Options) {
    let quota = cpu_quota().unwrap_or_else(|| "unknown".to_string());
    for &nth in threads {
        let mut runner = Runner::with_batch(model, opts.context, nth, opts.batch);
        // Not counted: the first pass faults in the pages this configuration
        // touches and starts the pool's threads.
        once(&mut runner, prompt, opts.generate);
        let load_before = load_average().unwrap_or(f64::NAN);
        let runs: Vec<Timed> = (0..opts.repeats)
            .map(|_| once(&mut runner, prompt, opts.generate))
            .collect();
        let load_after = load_average().unwrap_or(f64::NAN);

        println!(
            "threads {nth}  context {}  prompt {} tok  batch {}  generate {} tok  tier {}  quota {quota}  repeats {}  load {load_before:.2} -> {load_after:.2}",
            opts.context,
            prompt.len(),
            runner.max_batch(),
            opts.generate,
            runner.kernels().tier().as_str(),
            opts.repeats,
        );
        let prompt_rate: Vec<f64> = runs.iter().map(|r| rate(prompt.len(), r.prompt)).collect();
        let generate_rate: Vec<f64> = runs
            .iter()
            .map(|r| rate(opts.generate, r.generate))
            .collect();
        println!("  prompt      {}", summary(&prompt_rate));
        println!("  generation  {}", summary(&generate_rate));
    }
    // Printed once, not per configuration: `VmHWM` is a high-water mark for
    // the whole process, so every configuration after the first would be
    // reported with the peak of the ones before it. A per-configuration
    // number comes from a bench with one thread count and one context.
    if let Some(peak) = crate::peak_rss_kib() {
        println!(
            "peak RSS {peak} KiB ({:.2} GiB), for the whole process",
            peak as f64 / (1 << 20) as f64
        );
    }
}

/// One untimed pass, for a caller that wants the pages faulted in and the
/// pool's threads running before it measures anything.
pub fn warm_up(runner: &mut Runner<'_>, prompt: &[u32], generate: usize) {
    once(runner, prompt, generate);
}

fn once(runner: &mut Runner<'_>, prompt: &[u32], generate: usize) -> Timed {
    runner.reset();
    let start = Instant::now();
    let mut next = 0u32;
    let mut read = 0;
    while read < prompt.len() {
        let take = (prompt.len() - read).min(runner.max_batch());
        next = argmax(runner.forward_batch(&prompt[read..read + take]));
        read += take;
    }
    let prompt_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..generate {
        // Stop tokens are deliberately ignored: the point is a fixed amount
        // of work, not a readable answer.
        next = argmax(runner.forward(next));
    }
    Timed {
        prompt: prompt_time,
        generate: start.elapsed(),
    }
}

fn rate(n: usize, d: Duration) -> f64 {
    if d.is_zero() {
        return 0.0;
    }
    n as f64 / d.as_secs_f64()
}

/// `median tok/s  (n runs, min .. max, spread x%)`, the spread relative to the
/// median because that is what says whether two medians can be compared.
fn summary(rates: &[f64]) -> String {
    let (lo, hi, mid) = (min(rates), max(rates), median(rates));
    let spread = if mid > 0.0 {
        (hi - lo) / mid * 100.0
    } else {
        0.0
    };
    format!(
        "{mid:6.2} tok/s  ({} runs, {lo:.2} .. {hi:.2}, spread {spread:.1}%)",
        rates.len()
    )
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => 0.0,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The one-minute load average, which is the other half of what a number here
/// means: this box is shared, and a run that competes with three other busy
/// processes for the same eight-CPU quota is slower for reasons that have
/// nothing to do with the commit under test. Reported before and after the
/// repeats so a reader can see whether the two halves of a comparison ran
/// under the same conditions.
pub fn load_average() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// The cgroup CPU quota, as CPUs and as the raw pair.
///
/// A thread count without its quota is not a measurement: on this box
/// `cpu.max` is `800000/100000`, so eight CPUs of the thirty-two the kernel
/// reports, and that is why `available_parallelism` says 8.
pub fn cpu_quota() -> Option<String> {
    let raw = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let raw = raw.trim();
    let (quota, period) = raw.split_once(' ')?;
    if quota == "max" {
        return Some(format!("none ({raw})"));
    }
    let cpus = quota.parse::<f64>().ok()? / period.parse::<f64>().ok()?;
    Some(format!("{cpus:.2} CPUs ({quota}/{period})"))
}

#[cfg(test)]
mod tests {
    use super::{median, summary};

    #[test]
    fn the_median_and_the_spread_are_the_reported_numbers() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        // 5 of 10 is the spread the summary has to quote, against the median
        // and not against the minimum.
        let s = summary(&[9.0, 10.0, 14.0]);
        assert!(s.contains(" 10.00 tok/s"), "{s}");
        assert!(s.contains("9.00 .. 14.00"), "{s}");
        assert!(s.contains("spread 50.0%"), "{s}");
    }
}
