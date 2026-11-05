//! Sampling: the settings, the buffers the per-token work runs on, and the
//! draw.
//!
//! The vocabulary is 248320 entries and this runs once per generated token, on
//! a model that produces about ten tokens a second. A full sort of the logit
//! vector, or an allocation per token, is a measurable share of a token, so
//! [`Sampler`] owns every buffer it needs, sized from the vocabulary in
//! [`Sampler::new`], and [`Sampler::sample`] allocates nothing at all.
//!
//! # Order of operations
//!
//! 1. **Repetition penalty**, over the window of recently seen tokens. It
//!    iterates the window (64 entries by default), never the vocabulary, and
//!    follows llama.cpp's convention: a positive logit is divided by the
//!    penalty and a non-positive one multiplied, so a penalty above 1 only ever
//!    lowers a logit. Applied once per distinct token in the window, however
//!    often it occurs, which is also what llama.cpp does.
//! 2. **Temperature**, folded into the softmax exponent rather than applied as
//!    a pass of its own: `exp((l - max) / t)` is the same function as
//!    `softmax(l / t)` and costs one pass over the vocabulary less.
//! 3. **top-k**: the `k` largest logits, by partial selection over a
//!    preallocated index buffer. No full sort.
//! 4. **top-p**: the shortest prefix of the candidates, in descending order,
//!    whose mass covers `top_p`. The mass is normalised over the top-k
//!    candidates, which is what llama.cpp's default chain does too: its top-p
//!    sampler softmaxes whatever its top-k sampler left it.
//! 5. **min-p**: a threshold against the largest probability. Exact in one
//!    pass over the candidates and needing no normalisation at all, because the
//!    maximum is subtracted before the exponential, so the largest
//!    unnormalised probability is exactly 1.
//! 6. **The draw**, from the renormalised survivors.
//!
//! `temperature == 0` is greedy and takes exactly [`argmax`], including its
//! lowest-token-id tie rule, so a greedy run is byte-identical to the one the
//! correctness gate recorded.

/// Sampler settings.
///
/// Plain and `Copy`. A [`Sampler`] is built from one and never changes it,
/// because its buffers are sized from these numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// `0` is greedy: exactly [`argmax`], with no draw at all.
    pub temperature: f32,
    /// Keep at most this many candidates. `0` means the whole vocabulary.
    pub top_k: usize,
    /// Keep the shortest prefix of the candidates covering this much mass.
    /// `1.0` is off.
    pub top_p: f32,
    /// Drop candidates below this fraction of the largest probability. `0` is
    /// off.
    pub min_p: f32,
    /// `1.0` is off. Above 1 discourages repetition, below 1 encourages it.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty looks at.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Sampling {
    /// Argmax and nothing else, which is what the correctness gate runs.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: 0,
    };

    /// `true` where the penalty would change a logit.
    fn penalises(&self) -> bool {
        self.repeat_penalty != 1.0 && self.repeat_last_n > 0
    }
}

/// Qwen3.5's own recommended settings for this model, see `docs/decisions.md`.
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

/// A seed taken from the operating system's entropy.
///
/// `RandomState` is seeded by the standard library from the OS once per process
/// and stepped per instance, so this needs neither a dependency nor a clock.
/// Whoever uses it has to print the seed, or the run cannot be replayed.
pub fn entropy_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
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

/// Every buffer the per-token work needs, plus the RNG and the recent-token
/// window.
pub struct Sampler {
    set: Sampling,
    vocab: usize,
    /// The logits with the penalty applied, so the stages below have one
    /// mutable vector to work from. Empty while the penalty is off, which is
    /// the default, and that is also when the greedy path reads the forward
    /// pass's own logits untouched.
    penalized: Vec<f32>,
    /// Candidate token ids. The first [`Sampler::kept`] are the survivors of
    /// the last call.
    idx: Vec<u32>,
    /// Their probabilities, normalised over the survivors.
    probs: Vec<f32>,
    /// The recent tokens, oldest overwritten first.
    recent: Vec<u32>,
    at: usize,
    filled: usize,
    kept: usize,
    rng: Rng,
}

impl Sampler {
    /// Allocates for a `vocab`-entry logit vector. Nothing below this
    /// allocates again.
    pub fn new(vocab: usize, set: Sampling) -> Self {
        // The greedy path touches neither, but `candidates` still reports the
        // token it picked, so there is one slot for it.
        let candidates = if set.temperature > 0.0 { vocab } else { 1 };
        Self {
            set,
            vocab,
            penalized: vec![0.0; if set.penalises() { vocab } else { 0 }],
            idx: vec![0; candidates],
            probs: vec![0.0; candidates],
            recent: vec![0; set.repeat_last_n],
            at: 0,
            filled: 0,
            kept: 0,
            rng: Rng::seeded(set.seed),
        }
    }

    /// A greedy sampler, which is what the correctness gate and the trace path
    /// use.
    pub fn greedy(vocab: usize) -> Self {
        Self::new(vocab, Sampling::GREEDY)
    }

    pub fn settings(&self) -> Sampling {
        self.set
    }

    /// The survivors of the last [`Sampler::sample`] call and their
    /// probabilities, in descending order of logit where the pipeline needed
    /// that order. Written for the tests that compare the pipeline against a
    /// naive reference, and useful for explaining a draw.
    pub fn candidates(&self) -> (&[u32], &[f32]) {
        (&self.idx[..self.kept], &self.probs[..self.kept])
    }

    /// Forgets the recent-token window. The RNG stream carries on, so a seeded
    /// session stays reproducible across a reset.
    pub fn reset(&mut self) {
        self.at = 0;
        self.filled = 0;
        self.kept = 0;
    }

    /// Adds a token to the recent-token window.
    ///
    /// [`Sampler::sample`] does this for the token it draws, so a caller only
    /// has to call it for tokens that went through the model without being
    /// sampled, which is every token of the prompt.
    pub fn remember(&mut self, token: u32) {
        if self.recent.is_empty() {
            return;
        }
        self.recent[self.at] = token;
        self.at = (self.at + 1) % self.recent.len();
        self.filled = (self.filled + 1).min(self.recent.len());
    }

    /// Draws the next token and remembers it. Allocates nothing.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let token = self.pick(logits);
        self.remember(token);
        token
    }

    fn pick(&mut self, raw: &[f32]) -> u32 {
        assert_eq!(
            raw.len(),
            self.vocab,
            "this sampler was built for a different vocabulary"
        );
        let Self {
            set,
            penalized,
            idx,
            probs,
            recent,
            filled,
            kept,
            rng,
            ..
        } = self;

        let penalty = set.penalises() && *filled > 0;
        if penalty {
            penalized.copy_from_slice(raw);
            for (i, &t) in recent[..*filled].iter().enumerate() {
                // llama.cpp applies the penalty once per distinct token in the
                // window, however often it occurs there.
                if recent[..i].contains(&t) {
                    continue;
                }
                if let Some(v) = penalized.get_mut(t as usize) {
                    *v = if *v > 0.0 {
                        *v / set.repeat_penalty
                    } else {
                        *v * set.repeat_penalty
                    };
                }
            }
        }
        let logits: &[f32] = if penalty { penalized.as_slice() } else { raw };

        if set.temperature <= 0.0 {
            let token = argmax(logits);
            idx[0] = token;
            probs[0] = 1.0;
            *kept = 1;
            return token;
        }

        // A vector with no finite maximum has no distribution to draw from:
        // every entry is -inf, or one is +inf and carries all the mass. Both
        // come out as the argmax, which is also what the greedy path would say.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            let token = argmax(logits);
            idx[0] = token;
            probs[0] = 1.0;
            *kept = 1;
            return token;
        }

        // Descending by logit, lowest token id first on a tie. A total order,
        // so which tokens the partial selection below keeps is determined even
        // where logits are equal, and a naive full sort keeps the same ones.
        let order = |a: &u32, b: &u32| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        };

        let n = logits.len();
        let k = if set.top_k == 0 { n } else { set.top_k.min(n) };
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = i as u32;
        }
        if k < n {
            idx.select_nth_unstable_by(k - 1, &order);
        }
        let cand = &mut idx[..k];
        // The candidates only have to be ordered for top-p; min-p is a
        // threshold and the draw does not care, so with top-p off this is
        // where the full sort would otherwise have been.
        if set.top_p < 1.0 {
            cand.sort_unstable_by(&order);
        }

        let mut sum = 0.0f64;
        for (p, &id) in probs[..k].iter_mut().zip(cand.iter()) {
            *p = ((logits[id as usize] - max) / set.temperature).exp();
            sum += *p as f64;
        }

        let mut m = k;
        if set.top_p < 1.0 {
            let target = set.top_p as f64 * sum;
            let mut cum = 0.0f64;
            for (j, &p) in probs[..k].iter().enumerate() {
                cum += p as f64;
                if cum >= target {
                    m = j + 1;
                    break;
                }
            }
        }

        if set.min_p > 0.0 {
            // The largest unnormalised probability is exactly `exp(0)`, so the
            // threshold is `min_p` itself and the maximum always survives.
            let mut w = 0;
            for j in 0..m {
                if probs[j] >= set.min_p {
                    cand[w] = cand[j];
                    probs[w] = probs[j];
                    w += 1;
                }
            }
            // Only reachable with min_p above 1, which the CLI refuses; the
            // first candidate is still the one the compaction left in place.
            m = w.max(1);
        }

        let total: f64 = probs[..m].iter().map(|&p| p as f64).sum();
        let inv = (1.0 / total) as f32;
        for p in probs[..m].iter_mut() {
            *p *= inv;
        }
        *kept = m;

        let u = rng.next_f32();
        let mut cum = 0.0f32;
        for (j, &p) in probs[..m].iter().enumerate() {
            cum += p;
            if cum > u {
                return cand[j];
            }
        }
        // f32 rounding can leave the cumulative sum a hair below `u`.
        cand[m - 1]
    }
}

/// xoshiro256++, seeded through splitmix64.
///
/// Thirty lines and no dependency. The generator is not cryptographic and does
/// not need to be: what it has to be is reproducible from a `u64` seed and
/// well distributed over the candidate set, and this is the standard choice for
/// exactly that.
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn seeded(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            v ^ (v >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in `[0, 1)`, at f32's 24 bits of mantissa.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{Rng, Sampler, Sampling, argmax};

    #[test]
    fn argmax_takes_the_first_of_a_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-5.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    }

    #[test]
    fn the_generator_stays_in_the_unit_interval_and_covers_it() {
        let mut rng = Rng::seeded(7);
        let mut buckets = [0usize; 8];
        for _ in 0..80_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} is outside [0, 1)");
            buckets[(v * 8.0) as usize] += 1;
        }
        for (i, &n) in buckets.iter().enumerate() {
            assert!(n > 8_000, "bucket {i} got {n} of 80000 draws");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_stream_and_another_seed_does_not() {
        let one: Vec<u64> = (0..8).map(|_| Rng::seeded(42).next_u64()).collect();
        let again: Vec<u64> = (0..8).map(|_| Rng::seeded(42).next_u64()).collect();
        assert_eq!(one, again);
        assert_ne!(one[0], Rng::seeded(43).next_u64());
    }

    /// The window is a ring, so the penalty has to see the last `repeat_last_n`
    /// tokens and not the first ones.
    #[test]
    fn the_window_keeps_the_most_recent_tokens() {
        let set = Sampling {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 3,
            ..Sampling::GREEDY
        };
        let mut s = Sampler::new(5, set);
        for t in [0u32, 1, 2, 3] {
            s.remember(t);
        }
        // 0 has fallen out of the window, 3 is in it, so 3 is halved and loses
        // to 0 even though its logit started higher.
        let logits = [1.0f32, 0.0, 0.0, 1.5, 0.0];
        assert_eq!(s.sample(&logits), 0);
        s.reset();
        // With nothing in the window the raw logits decide.
        assert_eq!(s.sample(&logits), 3);
    }
}
