//! Greedy generation over [`Runner`], and the timings that go with it.
//!
//! Sampling is argmax and nothing else. A temperature, top-k or top-p surface
//! is a separate change; putting the hooks in now would be an interface with
//! one implementation.

use std::time::{Duration, Instant};

use crate::forward::Runner;

/// What a run cost, split the way the two phases differ: the prompt is read
/// with the caches growing from nothing, generation reads all the weights per
/// token.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub prompt_tokens: usize,
    pub prompt_time: Duration,
    pub generated_tokens: usize,
    pub generate_time: Duration,
}

impl Timings {
    pub fn prompt_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prompt_time)
    }

    pub fn generate_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.generate_time)
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
}

/// Reads the prompt, then generates greedily until a stop token, `max_tokens`
/// or the end of the context.
///
/// `on_token` sees every generated token as it is produced, before the next
/// forward pass, so a caller can stream. Stop tokens are not passed to it.
///
/// The prompt's last token is the one generation starts from: its logits are
/// the first prediction, which is also why nothing re-reads the prompt and the
/// delta-net state generation continues from is exactly the state the prompt
/// left behind.
pub fn generate(
    runner: &mut Runner<'_>,
    prompt: &[u32],
    max_tokens: usize,
    mut on_token: impl FnMut(u32),
) -> (Vec<u32>, Stop, Timings) {
    assert!(!prompt.is_empty(), "the prompt is empty");
    let eos = runner.config().eos.clone();
    let mut timings = Timings::default();
    let mut out = Vec::with_capacity(max_tokens);

    let start = Instant::now();
    let mut next = 0u32;
    for (i, &t) in prompt.iter().enumerate() {
        if runner.pos() == runner.context() {
            return (out, Stop::ContextFull, timings);
        }
        let logits = runner.forward(t);
        if i + 1 == prompt.len() {
            next = argmax(logits);
        }
    }
    timings.prompt_tokens = prompt.len();
    timings.prompt_time = start.elapsed();

    let start = Instant::now();
    let stop = loop {
        if eos.contains(&next) {
            break Stop::Eos(next);
        }
        out.push(next);
        on_token(next);
        if out.len() >= max_tokens {
            break Stop::MaxTokens;
        }
        if runner.pos() == runner.context() {
            break Stop::ContextFull;
        }
        next = argmax(runner.forward(next));
    };
    timings.generated_tokens = out.len();
    timings.generate_time = start.elapsed();

    (out, stop, timings)
}

/// The highest logit, lowest token id winning a tie.
///
/// The tie rule only matters for reproducibility; on this model exact ties
/// between two logits do not happen in practice.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_takes_the_first_of_a_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-5.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    }
}
