//! Generation over [`Runner`], and the timings that go with it.
//!
//! One function with a [`Sampler`] argument rather than a greedy function next
//! to a sampling one: a greedy run is [`crate::sample::Sampling::GREEDY`], not
//! a separate code path.

use std::time::{Duration, Instant};

use crate::forward::Runner;
use crate::sample::Sampler;

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

/// Reads the prompt, then generates until a stop token, `max_tokens` or the
/// end of the context.
///
/// `on_token` sees every generated token as it is produced, before the next
/// forward pass, so a caller can stream. Stop tokens are not passed to it.
///
/// The prompt's last token is the one generation starts from: its logits are
/// the first prediction, which is also why nothing re-reads the prompt and the
/// delta-net state generation continues from is exactly the state the prompt
/// left behind.
///
/// Every token that goes through the model enters the sampler's recent-token
/// window exactly once, the prompt's included: the window is what the
/// repetition penalty reads, and a penalty that only saw the generated tokens
/// would not notice the prompt repeating itself.
pub fn generate(
    runner: &mut Runner<'_>,
    sampler: &mut Sampler,
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
        sampler.remember(t);
        if i + 1 == prompt.len() {
            next = sampler.sample(logits);
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
        next = sampler.sample(runner.forward(next));
    };
    timings.generated_tokens = out.len();
    timings.generate_time = start.elapsed();

    (out, stop, timings)
}
