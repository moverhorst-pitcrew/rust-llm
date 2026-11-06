//! Generation over [`Runner`], and the timings that go with it.
//!
//! One function with a [`Sampler`] argument rather than a greedy function next
//! to a sampling one: a greedy run is [`crate::sample::Sampling::GREEDY`], not
//! a separate code path. Speculative decoding is wired the same way, and for a
//! stronger reason: there is one loop, and with `--speculate 0` it is the loop
//! it always was, so there is no second generation path to keep in step.
//!
//! # Where a draft comes from
//!
//! Two sources, and the loop below cannot tell them apart. [`prompt_lookup`]
//! guesses from the text so far and needs no weights at all;
//! [`Runner::draft_mtp`] runs the model's own multi-token-prediction block,
//! which needs the head, its cache and the hidden state the trunk produced, so
//! the draft callback is handed the runner. Whichever it is, it fills a buffer
//! with tokens and says how many, and everything below this line is the same.
//!
//! # How a draft is accepted
//!
//! The draft source guesses the next few tokens,
//! [`Runner::verify`] runs them as one batch and produces the logits of every
//! row, and the loop below then does exactly what it does without speculation:
//! one `sampler.sample` per token, in order. The only difference is where the
//! logits come from. A drafted token is "accepted" when the sampler's own pick
//! happens to be the token the batch already ran at the next row, which means
//! its logits are there to be read instead of costing a pass; otherwise the
//! batch's remaining rows are dropped by committing fewer of them.
//!
//! That is why output with speculation on is **bit-equal** to output with it
//! off, and not merely equal in distribution: the sampler is called once per
//! emitted token, in the same order, so the RNG stream and the recent-token
//! window see the same thing. This is deliberately not Leviathan-style
//! speculative sampling, where a draft is accepted against the drafter's own
//! probability; a prompt-lookup draft has no probability to accept against, and
//! an accept rule that drew from a residual distribution would consume the RNG
//! differently and cost the equivalence test.

use std::time::{Duration, Instant};

use crate::forward::{MAX_DRAFT, Runner};
use crate::sample::Sampler;

/// Longest suffix of the text so far that [`prompt_lookup`] tries to match.
///
/// A longer match is a better guess and a rarer one. Eight is where the
/// acceptance histogram in `docs/benchmarks.md` stops gaining: past it the
/// matches that exist are matches a shorter suffix already found at the same
/// place, and the scan is longer for nothing.
pub const MAX_MATCH: usize = 8;

/// Shortest suffix [`prompt_lookup`] will take a match on.
///
/// One token is not a match, it is a coincidence: on a 248320-token vocabulary
/// a common token occurs all over the history and the token after an arbitrary
/// one of its occurrences predicts nothing. Each wrong draft costs a
/// verification row, so the floor is two.
pub const MIN_MATCH: usize = 2;

/// Where [`generate`] gets its drafts.
///
/// A selector rather than a function the caller passes, because it comes off a
/// command line and out of an HTTP request. What it actually does is decide
/// whether [`Runner::enable_mtp`] is called; the loop itself asks the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftSource {
    /// Guesses from the prompt and the text generated so far. Needs no weights
    /// and works on any file, which is why it is the default.
    #[default]
    PromptLookup,
    /// The model's own multi-token-prediction block. Needs a conversion that
    /// kept it; see `docs/models.md`.
    Mtp,
}

impl DraftSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PromptLookup => "prompt-lookup",
            Self::Mtp => "mtp",
        }
    }
}

impl std::str::FromStr for DraftSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "prompt-lookup" => Ok(Self::PromptLookup),
            "mtp" => Ok(Self::Mtp),
            other => Err(format!(
                "unknown draft source {other:?}; it is prompt-lookup or mtp"
            )),
        }
    }
}

/// What a run cost, split the way the two phases differ: the prompt is read
/// with the caches growing from nothing, generation reads all the weights per
/// token.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub prompt_tokens: usize,
    pub prompt_time: Duration,
    pub generated_tokens: usize,
    pub generate_time: Duration,
    pub speculation: Speculated,
}

impl Timings {
    pub fn prompt_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prompt_time)
    }

    pub fn generate_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.generate_time)
    }
}

/// What speculation did, all zero when it is off.
///
/// The histogram is the point of this and not the acceptance rate it averages
/// to: what a longer draft is worth depends on how the accepted lengths are
/// distributed, because a pass that accepts none still pays for every row it
/// ran. It is also the measurement any later draft source, an MTP head
/// included, has to beat.
#[derive(Debug, Clone, Copy, Default)]
pub struct Speculated {
    /// Passes that ran a plain [`Runner::forward`], because the drafter found
    /// nothing to guess from or the context had no room for a batch.
    pub plain: usize,
    /// Drafted tokens offered, which is the denominator of [`Self::acceptance`].
    pub offered: usize,
    /// Verification passes by how many of their drafts were accepted:
    /// `accepted[j]` ran a batch and accepted exactly `j` of the drafts in it.
    pub accepted: [usize; MAX_DRAFT + 1],
}

impl Speculated {
    /// Verification passes, which is one per batch that ran.
    pub fn passes(&self) -> usize {
        self.accepted.iter().sum()
    }

    /// Drafted tokens accepted, which is the number of forward passes
    /// speculation saved.
    pub fn saved(&self) -> usize {
        self.accepted.iter().enumerate().map(|(j, n)| j * n).sum()
    }

    /// Accepted drafts per drafted token, which is the number
    /// `docs/benchmarks.md` compares against the break-even acceptance.
    pub fn acceptance(&self) -> f64 {
        if self.offered == 0 {
            return 0.0;
        }
        self.saved() as f64 / self.offered as f64
    }

    /// One line: the passes, the acceptance and the histogram.
    pub fn report(&self) -> String {
        let top = self
            .accepted
            .iter()
            .rposition(|&n| n > 0)
            .unwrap_or_default();
        let hist: Vec<String> = self.accepted[..=top]
            .iter()
            .enumerate()
            .map(|(j, n)| format!("{j}:{n}"))
            .collect();
        format!(
            "{} passes, {} plain, {} of {} drafts accepted ({:.2} per draft), accepted \
             lengths {}",
            self.passes(),
            self.plain,
            self.saved(),
            self.offered,
            self.acceptance(),
            hist.join(" "),
        )
    }
}

fn rate(n: usize, d: Duration) -> f64 {
    if d.is_zero() {
        return 0.0;
    }
    n as f64 / d.as_secs_f64()
}

/// Why generation stopped, which the caller usually wants to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// A token from the model's end-of-turn set.
    Eos(u32),
    MaxTokens,
    /// The context filled up; the caller asked for more than it allocated.
    ContextFull,
    /// `on_token` asked to stop. The HTTP surface uses it when the client it
    /// was streaming to went away: every further token costs every core and
    /// nobody will read it.
    Cancelled,
}

/// The next few tokens, guessed from the text so far.
///
/// Takes the longest suffix of `history`, between [`MIN_MATCH`] and
/// [`MAX_MATCH`] tokens, that occurs earlier in `history`, and writes what
/// followed the **last** such occurrence into `out`. Returns how many tokens
/// that was, which is fewer than `out.len()` when the occurrence is near the
/// end, and `0` when no suffix matches anywhere: then the caller runs a plain
/// forward pass and text with nothing to guess from pays nothing.
///
/// The longest suffix first, because a longer match is a better guess; the last
/// occurrence of it, because the most recent context that looked like this one
/// is the best guide to what follows it now.
///
/// The scan is a backward walk and nothing is indexed or cached. At a context
/// of 4096 the worst case is seven passes over the history, under 0.3 ms, and
/// what one hit saves is a 75 ms pass. An index would be a structure to keep in
/// step with the history for a gain below this function's own noise. It
/// allocates nothing, which is what lets it sit in the generation loop.
pub fn prompt_lookup(history: &[u32], out: &mut [u32]) -> usize {
    // A match needs a suffix and at least one position before it.
    if out.is_empty() || history.len() <= MIN_MATCH {
        return 0;
    }
    let longest = MAX_MATCH.min(history.len() - 1);
    for len in (MIN_MATCH..=longest).rev() {
        let suffix = &history[history.len() - len..];
        for start in (0..history.len() - len).rev() {
            if &history[start..start + len] != suffix {
                continue;
            }
            // At least one token follows, because `start + len` is inside the
            // history; fewer than asked for when the occurrence is near the end.
            let take = out.len().min(history.len() - start - len);
            out[..take].copy_from_slice(&history[start + len..][..take]);
            return take;
        }
    }
    0
}

/// Reads the prompt, then generates until a stop token, `max_tokens` or the
/// end of the context.
///
/// `on_token` sees every generated token as it is produced, before the next
/// forward pass, so a caller can stream. Stop tokens are not passed to it. It
/// returns whether to keep going: `false` ends the run with [`Stop::Cancelled`]
/// and the token it just saw is the last one in the output.
///
/// The prompt goes through in batches of up to [`Runner::max_batch`] tokens,
/// which is where a prompt is faster per token than generation. The prompt's
/// last token is still the one generation starts from: its logits are the first
/// prediction, which is also why nothing re-reads the prompt and the delta-net
/// state generation continues from is exactly the state the prompt left behind.
///
/// Every token that goes through the model enters the sampler's recent-token
/// window exactly once, the prompt's included and in order: the window is what
/// the repetition penalty reads, and a penalty that only saw the generated
/// tokens would not notice the prompt repeating itself.
///
/// With `runner.draft()` above zero the tokens come out in bursts of up to
/// `draft + 1` per pass, which `on_token` sees one at a time and in order
/// exactly as it would without speculation; a burst that is cut short by
/// `on_token`, a stop token, `max_tokens` or the end of the context stops where
/// a sequential run would have stopped, with the same token last and the caches
/// holding the same positions. See the module documentation for why the tokens
/// themselves are identical.
pub fn generate(
    runner: &mut Runner<'_>,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_tokens: usize,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Stop, Timings) {
    // Which source drafts is the runner's: whether the MTP block was enabled is
    // what `--draft-source` decided, and a runner without one drafts from the
    // prompt. One `if` rather than a parameter here, because every caller of
    // `generate` would otherwise have to pass the same thing twice.
    match runner.drafts_from_mtp() {
        true => generate_drafting(
            runner,
            sampler,
            prompt,
            max_tokens,
            |r, history, out| r.draft_mtp(history, out),
            on_token,
        ),
        false => generate_drafting(
            runner,
            sampler,
            prompt,
            max_tokens,
            |_, history, out| prompt_lookup(history, out),
            on_token,
        ),
    }
}

/// [`generate`] with the drafts coming from somewhere other than
/// [`prompt_lookup`].
///
/// `draft_with` is handed the runner, the prompt plus everything generated so
/// far, and a buffer of `runner.draft()` tokens to fill; it returns how many it
/// wrote. It is called once per pass and must not allocate, because it sits in
/// the generation loop.
///
/// The runner is in that signature for [`Runner::draft_mtp`], which cannot
/// guess from the text: it needs the MTP block, its cache and the hidden state
/// the trunk left behind. [`prompt_lookup`] ignores it.
///
/// This exists for two reasons and is not a strategy type: a draft source is a
/// different function passed here and nothing else, and the tests need a draft
/// source whose acceptance they choose. On the tiny fixture's random weights a
/// natural match is never accepted, so the burst boundaries, a stop token,
/// `max_tokens`, a cancelling `on_token` and the end of the context inside one
/// batch, would otherwise only be reachable with the 2.8 GB model.
pub fn generate_drafting(
    runner: &mut Runner<'_>,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_tokens: usize,
    mut draft_with: impl FnMut(&mut Runner<'_>, &[u32], &mut [u32]) -> usize,
    mut on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Stop, Timings) {
    assert!(!prompt.is_empty(), "the prompt is empty");
    let eos = runner.config().eos.clone();
    let mut timings = Timings::default();
    // The context is the ceiling on prompt plus generated tokens, so no run can
    // produce more than that many and reserving for a larger `max_tokens` is
    // space that could never be used. It is also what keeps the reservation
    // bounded: `max_tokens` arrives from a command line or an HTTP request, and
    // `with_capacity` of a number that size is a panic, or an abort where the
    // allocation is refused.
    let mut out = Vec::with_capacity(max_tokens.min(runner.context()));

    let start = Instant::now();
    let mut next = 0u32;
    let mut read = 0;
    while read < prompt.len() {
        // The batch cannot be longer than the positions the caches hold, so
        // the room left in the context caps it next to the scratch's own
        // maximum.
        let room = runner.context() - runner.pos();
        if room == 0 {
            // What went through the model, which is not the whole prompt here:
            // the caller handed over more of it than the context holds, and a
            // count of 0 would read as a model with nothing to say.
            timings.prompt_tokens = read;
            timings.prompt_time = start.elapsed();
            return (out, Stop::ContextFull, timings);
        }
        let take = (prompt.len() - read).min(runner.max_batch()).min(room);
        let batch = &prompt[read..read + take];
        let logits = runner.forward_batch(batch);
        // Every token of the batch, in order, exactly as the single-token path
        // remembered them one at a time.
        for &t in batch {
            sampler.remember(t);
        }
        read += take;
        if read == prompt.len() {
            next = sampler.sample(logits);
        }
    }
    timings.prompt_tokens = prompt.len();
    timings.prompt_time = start.elapsed();

    let draft = runner.draft();
    // All three are sized once here and never grow, and all three are empty
    // with speculation off, where `Vec::new` and a zero-length `vec!` allocate
    // nothing at all: the off path's allocation count is what it was.
    let mut history = Vec::with_capacity(if draft > 0 {
        prompt.len() + out.capacity()
    } else {
        0
    });
    history.extend_from_slice(if draft > 0 { prompt } else { &[] });
    let mut drafts = vec![0u32; draft];
    let mut rows: Vec<u32> = Vec::with_capacity(if draft > 0 { draft + 1 } else { 0 });
    // Rows of the batch in flight that have been committed. Zero exactly when
    // `rows` is empty, which is when there is no batch in flight.
    let mut taken = 0usize;

    let start = Instant::now();
    let stop = loop {
        if eos.contains(&next) {
            break Stop::Eos(next);
        }
        out.push(next);
        if draft > 0 {
            history.push(next);
        }
        if !on_token(next) {
            break Stop::Cancelled;
        }
        if out.len() >= max_tokens {
            break Stop::MaxTokens;
        }
        if runner.pos() == runner.context() {
            break Stop::ContextFull;
        }

        // Everything above is what decides whether `next` may be fed to the
        // model at all, and it is the same five questions in the same order
        // whether the token came out of a batch or out of a pass of its own.
        // Below is the only thing speculation changes: where the logits that
        // follow `next` come from.
        if taken < rows.len() && rows[taken] == next {
            // The batch in flight already ran this token, at row `taken`: the
            // draft is accepted, the row is committed and its logits are the
            // prediction a pass would have produced.
            runner.accept(1);
            next = sampler.sample(runner.logit_row(taken));
            taken += 1;
            continue;
        }
        // Any batch in flight ends here, either because the sampler did not
        // pick its next draft or because its rows ran out. Its uncommitted
        // rows stay in the caches past `pos`, where nothing reads them and the
        // next batch overwrites them.
        if !rows.is_empty() {
            timings.speculation.accepted[taken - 1] += 1;
            rows.clear();
            taken = 0;
        }

        // A batch has to fit the context, so the drafts are capped by the room
        // left as well as by `draft`.
        let room = runner.context() - runner.pos();
        let n = if draft == 0 || room < 2 {
            0
        } else {
            draft_with(runner, &history, &mut drafts[..draft.min(room - 1)])
        };
        if n == 0 {
            timings.speculation.plain += usize::from(draft > 0);
            next = sampler.sample(runner.forward(next));
            continue;
        }
        rows.push(next);
        rows.extend_from_slice(&drafts[..n]);
        timings.speculation.offered += n;
        runner.verify(&rows);
        runner.accept(1);
        next = sampler.sample(runner.logit_row(0));
        taken = 1;
    };
    if !rows.is_empty() {
        timings.speculation.accepted[taken - 1] += 1;
    }
    timings.generated_tokens = out.len();
    timings.generate_time = start.elapsed();

    (out, stop, timings)
}

#[cfg(test)]
mod tests {
    use super::{MAX_MATCH, MIN_MATCH, prompt_lookup};

    /// What the drafter does, by hand, on histories small enough to read.
    ///
    /// The cases are the ones that can be got wrong silently: the longest
    /// suffix winning over a shorter one, the *last* occurrence winning over an
    /// earlier one, a match at the very start of the history, a match that
    /// leaves fewer tokens than were asked for, and the floor and the cap on the
    /// match length.
    #[test]
    fn the_drafter_guesses_from_the_last_longest_match() {
        let mut out = [0u32; 4];

        // "1 2 3" occurs once before, and the draft is what followed it there.
        let h = [1u32, 2, 3, 9, 8, 7, 1, 2, 3];
        assert_eq!(prompt_lookup(&h, &mut out), 4);
        assert_eq!(out, [9, 8, 7, 1]);

        // Two occurrences of "1 2": the last one wins, so the draft is 5 and
        // not 9, which is what an earlier occurrence would have given.
        let h = [1u32, 2, 9, 0, 1, 2, 5, 6, 1, 2];
        assert_eq!(prompt_lookup(&h, &mut out), 4);
        assert_eq!(&out[..4], &[5, 6, 1, 2]);

        // A longer match points somewhere else than a shorter one, and the
        // longer one decides: "0 1 2" is followed by 7, "1 2" also occurs
        // later and is followed by 4.
        let h = [0u32, 1, 2, 7, 7, 1, 2, 4, 4, 0, 1, 2];
        assert_eq!(prompt_lookup(&h, &mut out), 4);
        assert_eq!(&out[..4], &[7, 7, 1, 2]);

        // A match at position 0, which is the edge of the backward scan.
        let h = [4u32, 5, 6, 4, 5];
        assert_eq!(prompt_lookup(&h, &mut out), 3);
        assert_eq!(&out[..3], &[6, 4, 5]);

        // A draft that runs past the end of the history: "1 2" occurs at 0 and
        // only two tokens follow it there, so two are drafted where four were
        // asked for.
        let h = [1u32, 2, 1, 2];
        assert_eq!(prompt_lookup(&h, &mut out), 2);
        assert_eq!(&out[..2], &[1, 2]);
    }

    #[test]
    fn the_drafter_offers_nothing_when_there_is_nothing_to_match() {
        let mut out = [0u32; 4];
        // Nothing repeats.
        assert_eq!(prompt_lookup(&[1, 2, 3, 4, 5], &mut out), 0);
        // Shorter than the minimum match plus the position it needs before it.
        for len in 0..=MIN_MATCH {
            let h: Vec<u32> = vec![7; len];
            assert_eq!(prompt_lookup(&h, &mut out), 0, "history of {len}");
        }
        // Asked for no drafts at all.
        assert_eq!(prompt_lookup(&[1, 2, 1, 2, 1, 2], &mut []), 0);
    }

    /// A single repeated token is not a match: with MIN_MATCH at 2 the suffix
    /// has to be two tokens, so a lone 9 recurring somewhere predicts nothing.
    #[test]
    fn one_token_is_not_a_match() {
        let mut out = [0u32; 2];
        assert_eq!(MIN_MATCH, 2);
        // 9 occurs earlier, followed by 4; the suffix "3 9" does not occur.
        assert_eq!(prompt_lookup(&[9, 4, 1, 2, 3, 9], &mut out), 0);
        // Add the second token of the suffix and the same history matches.
        assert_eq!(prompt_lookup(&[3, 9, 4, 1, 2, 3, 9], &mut out), 2);
        assert_eq!(out, [4, 1]);
    }

    /// The cap: a suffix longer than [`MAX_MATCH`] is never looked for, so a
    /// history whose only repeat is longer than the cap still matches on its
    /// last [`MAX_MATCH`] tokens, and the draft is what follows *that*.
    #[test]
    fn the_match_length_is_capped() {
        // 0..=9 twice. The whole ten-token suffix repeats, but the drafter
        // matches on the last MAX_MATCH of it, which occurs at index 2 as well
        // as at index 12, and the last one is the one it takes.
        let h: Vec<u32> = (0..10).chain(0..10).collect();
        let mut out = [0u32; 3];
        assert_eq!(MAX_MATCH, 8);
        // The suffix 2..=9 occurs at index 2 and at index 12, which is the
        // suffix itself, so the last earlier one is index 2 and 0 follows it.
        assert_eq!(prompt_lookup(&h, &mut out), 3);
        assert_eq!(out, [0, 1, 2]);
    }
}
