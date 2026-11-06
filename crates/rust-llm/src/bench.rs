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
use crate::generate::{DraftSource, Speculated, Stop, generate};
use crate::model::Model;
use crate::sample::{Sampler, argmax};

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
    /// Drafted tokens per pass. Above 0, every block also reports what a
    /// verification row costs; see [`verify_rows`].
    pub speculate: usize,
    /// Where those drafts come from.
    pub draft_source: DraftSource,
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
        if opts.speculate > 0 {
            let token = median(&generate_rate).recip() * 1e3;
            verify_rows(model, prompt, nth, opts, token);
            if opts.draft_source == DraftSource::Mtp {
                mtp_cost(model, prompt, nth, opts, token);
            }
            speculated(model, prompt, nth, opts);
        }
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

/// What a verification batch costs, per row count, and what that makes the
/// break-even acceptance rate.
///
/// This is the number speculative decoding stands or falls by, so it is
/// measured rather than argued from the per-operator table. A verification
/// batch runs the output head on **every** row, which no other path does: the
/// prompt computes it once per batch and generation once per token. So the
/// marginal row of a verification batch is the marginal row of a prompt batch
/// plus whatever the head's extra rows cost, and the point of this is to say how
/// much that is.
///
/// A draft is only worth running if the tokens it saves cost more than the rows
/// it adds, which at `m` ms per marginal row against `token` ms per sequentially
/// generated token is `accepted drafts per drafted token > m / token`. That
/// ratio is the last column.
///
/// Each row count is timed on its own runner, `repeats` times, with one row
/// committed per call so the position advances as it would in a real run. The
/// batch's tokens are taken from the prompt: what they are cannot change the
/// time, only the drafts' acceptance, which is not what this measures.
fn verify_rows(model: &Model<'_>, prompt: &[u32], threads: usize, opts: &Options, token: f64) {
    let mut runner =
        Runner::with_speculation(model, opts.context, threads, opts.batch, opts.speculate);
    println!(
        "  verify      the output head on every row, {} repeats, {token:.2} ms per \
         sequential token",
        opts.repeats
    );
    println!(
        "    {:>5}{:>12}{:>14}{:>22}",
        "rows", "ms/batch", "ms/extra row", "break-even acceptance"
    );
    let mut first = f64::NAN;
    for rows in 1..=opts.speculate + 1 {
        let ms = batch_of(&mut runner, prompt, rows, opts.repeats);
        if rows == 1 {
            first = ms;
            println!("    {rows:>5}{ms:>12.2}{:>14}{:>22}", "--", "--");
        } else {
            let per = (ms - first) / (rows - 1) as f64;
            println!("    {rows:>5}{ms:>12.2}{per:>14.2}{:>22.2}", per / token);
        }
    }
    // The pair that says what having the ring at all costs a pass the drafter
    // found nothing for: the same call, the same row count, the same window,
    // against a runner whose state is one buffer updated in place. The
    // arithmetic is identical, so whatever this gap is, it is the memory: a
    // pass writes a ring slot it has not touched for `draft` passes and has to
    // fetch it first, where an in-place update is already resident.
    let mut off = Runner::with_batch(model, opts.context, threads, opts.batch);
    let plain = batch_of(&mut off, prompt, 1, opts.repeats);
    println!(
        "    one row against a single state buffer: {plain:.2} ms, so the ring costs \
         {:+.2} ms ({:+.1}%) on a pass with no draft",
        first - plain,
        (first - plain) / plain * 100.0,
    );
}

/// What the MTP head costs, and the break-even acceptance that follows from it.
///
/// Prompt lookup drafts for free, so [`verify_rows`] is the whole cost there
/// and its last column is the whole break-even. A head is not free: a draft
/// step is one pass over the MTP block and the output head, and the steps are
/// sequential, so `d` drafts cost `d` of them. Keeping the block caught up
/// costs a row per committed token on top of that. Both have to be measured
/// before an acceptance rate means anything, which is why this is here and not
/// an argument from the operator table.
///
/// So speculation pays at draft length `d` with `a` of those drafts accepted
/// when
///
/// ```text
/// verify(d + 1) + d * step + (a + 1) * catch_up  <  (a + 1) * token
/// ```
///
/// which solves to `a > (verify(d + 1) + d * step) / (token - catch_up) - 1`.
/// The last column is that bound as drafts accepted per drafted token, `a / d`,
/// which is what [`Speculated::acceptance`] reports and what the acceptance
/// floor test holds the head to.
///
/// `step` is timed on `draft_mtp` asking for one token from a fixed position.
/// It writes the block's cache in the draft region and advances nothing, so
/// calling it repeatedly from the same position is the same work every time;
/// the confidence floor cannot shorten it either, because the block and the
/// head run before it is consulted.
///
/// `catch_up` is the difference between a plain `forward` on a runner with the
/// head enabled and one without. That is the catch-up row plus the extra
/// `rms_norm` the non-verifying path now does, which is the honest pair: it is
/// what a committed token costs more with the feature on, whether or not a
/// draft was accepted.
fn mtp_cost(model: &Model<'_>, prompt: &[u32], threads: usize, opts: &Options, token: f64) {
    let mut on = Runner::with_speculation(model, opts.context, threads, opts.batch, opts.speculate);
    assert!(on.enable_mtp(), "--draft-source mtp needs the block");
    let mut off = Runner::with_speculation(model, opts.context, threads, opts.batch, opts.speculate);

    // Both runners read the prompt, so the block's cache is caught up and the
    // attention walks a real number of positions rather than one.
    let mut history: Vec<u32> = Vec::new();
    for runner in [&mut on, &mut off] {
        runner.reset();
        let mut read = 0;
        while read < prompt.len() {
            let take = (prompt.len() - read).min(runner.max_batch());
            runner.forward_batch(&prompt[read..read + take]);
            read += take;
        }
    }
    history.extend_from_slice(prompt);

    let mut draft = [0u32; 1];
    // Thrown away: the first call faults in the block's cache and its scratch.
    on.draft_mtp(&history, &mut draft);
    let step = median_ms(opts.repeats, || {
        on.draft_mtp(&history, &mut draft);
    });
    // The pair for a committed token. One token each, from the same position,
    // so the only difference is the block.
    let last = *history.last().expect("a prompt");
    let with = median_ms(opts.repeats, || {
        on.forward(last);
    });
    let without = median_ms(opts.repeats, || {
        off.forward(last);
    });
    let catch_up = with - without;

    println!(
        "  mtp         the head's own cost, {} repeats, {token:.2} ms per sequential token",
        opts.repeats
    );
    println!(
        "    one draft step {step:.2} ms, catch-up per committed token {catch_up:+.2} ms          ({with:.2} against {without:.2})"
    );
    println!(
        "    {:>5}{:>12}{:>14}{:>22}",
        "draft", "ms/batch", "ms/draft", "break-even acceptance"
    );
    for d in 1..=opts.speculate {
        let batch = batch_of(&mut on, prompt, d + 1, opts.repeats);
        // What the drafts of one pass cost on top of its batch.
        let drafts = d as f64 * step;
        let a = (batch + drafts) / (token - catch_up) - 1.0;
        println!(
            "    {d:>5}{batch:>12.2}{drafts:>14.2}{:>22.2}",
            a / d as f64
        );
    }
}

/// The median of `repeats` calls of `f`, in milliseconds.
fn median_ms(repeats: usize, mut f: impl FnMut()) -> f64 {
    let ms: Vec<f64> = (0..repeats)
        .map(|_| {
            let at = Instant::now();
            f();
            at.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    median(&ms)
}

/// Generation through the accept loop with the drafter on, against the same
/// loop with it off, in one window and on one runner.
///
/// The pair is what a speed-up is: both rows run `generate` on the same runner,
/// so the ring is allocated and written in both and the only difference between
/// them is whether drafts are offered. What having the ring at all costs is the
/// separate pair at the end of [`verify_rows`], and what the whole feature
/// costs against a build without it is the `generation` row above, which is
/// today's loop on a runner with a single state buffer.
///
/// Greedy, because the acceptance of a draft is a property of the model and the
/// text and not of a seed, and stop tokens are honoured rather than ignored, so
/// a run that ends early is reported with the tokens it produced. The histogram
/// is summed over the repeats.
fn speculated(model: &Model<'_>, prompt: &[u32], threads: usize, opts: &Options) {
    let mut runner =
        Runner::with_speculation(model, opts.context, threads, opts.batch, opts.speculate);
    for (label, draft) in [("sequential", 0), ("speculated", opts.speculate)] {
        assert!(runner.set_draft(draft), "the ring was sized for this draft");
        // The MTP block is enabled for the drafting arm only, so `sequential`
        // stays the loop without the feature rather than the loop still paying
        // to keep the block caught up. That catch-up is therefore inside the
        // speculated row, which is where a speed-up has to carry it.
        if draft > 0 && opts.draft_source == DraftSource::Mtp {
            runner.reset();
            assert!(runner.enable_mtp(), "the file has an MTP block");
        }
        // Not counted, and once per draft length: the first run faults in the
        // slots this draft length is the first to touch.
        let _ = generated(&mut runner, model, prompt, opts.generate);
        let mut rates = Vec::new();
        let mut tally = Speculated::default();
        let mut produced = 0;
        let mut stopped = Stop::MaxTokens;
        for _ in 0..opts.repeats {
            let (n, secs, stop, spec) = generated(&mut runner, model, prompt, opts.generate);
            rates.push(rate(n, Duration::from_secs_f64(secs)));
            tally.plain += spec.plain;
            tally.offered += spec.offered;
            for (a, b) in tally.accepted.iter_mut().zip(spec.accepted) {
                *a += b;
            }
            (produced, stopped) = (n, stop);
        }
        println!("  {label}  {}", summary(&rates));
        if draft > 0 {
            println!(
                "    {produced} tokens from {}, {stopped:?}, {}",
                opts.draft_source.as_str(),
                tally.report()
            );
        }
    }
}

/// One generation run through [`generate`], from a fresh cache and the prompt.
fn generated(
    runner: &mut Runner<'_>,
    model: &Model<'_>,
    prompt: &[u32],
    tokens: usize,
) -> (usize, f64, Stop, Speculated) {
    runner.reset();
    let mut sampler = Sampler::greedy(model.vocab());
    let (ids, stop, timings) = generate(runner, &mut sampler, prompt, tokens, |_| true);
    (
        ids.len(),
        timings.generate_time.as_secs_f64(),
        stop,
        timings.speculation,
    )
}

/// The median of `repeats` verification batches of `rows` rows, from a fresh
/// cache and the prompt.
///
/// One row is committed per call, so the position advances as it would in a run
/// rather than standing still, and the first call of each width is thrown away:
/// it is what faults in the ring slots nothing has touched yet. The batch's
/// tokens are taken from the prompt, because what they are cannot change the
/// time, only a draft's acceptance, which is not what this measures.
fn batch_of(runner: &mut Runner<'_>, prompt: &[u32], rows: usize, repeats: usize) -> f64 {
    runner.reset();
    let mut read = 0;
    while read < prompt.len() {
        let take = (prompt.len() - read).min(runner.max_batch());
        runner.forward_batch(&prompt[read..read + take]);
        read += take;
    }
    let batch: Vec<u32> = (0..rows).map(|i| prompt[i % prompt.len()]).collect();
    runner.verify(&batch);
    runner.accept(1);
    let times: Vec<f64> = (0..repeats)
        .map(|_| {
            let at = Instant::now();
            runner.verify(&batch);
            let ms = at.elapsed().as_secs_f64() * 1e3;
            runner.accept(1);
            ms
        })
        .collect();
    median(&times)
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
