//! The sampler against a naive reference, and its edge cases.
//!
//! None of this needs the weights: a logit vector is a logit vector, and what
//! is under test is the pipeline, not the model.

use rust_llm::sample::{Sampler, Sampling, argmax};

/// The same pipeline, naively: a full sort of the whole logit vector and f64
/// softmax. Returns the survivors, descending, with their normalised
/// probabilities.
///
/// The penalty is applied in f32 here as well, because it is elementwise and
/// has no accumulation to be naive about; doing it in f64 would only move the
/// last bit of a logit and could reorder a tie for no gain in what this
/// reference is for.
fn reference(logits: &[f32], set: Sampling, recent: &[u32]) -> Vec<(u32, f64)> {
    let mut l = logits.to_vec();
    if set.repeat_penalty != 1.0 && set.repeat_last_n > 0 {
        let window = &recent[recent.len().saturating_sub(set.repeat_last_n)..];
        let mut seen: Vec<u32> = Vec::new();
        for &t in window {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let v = &mut l[t as usize];
            *v = if *v > 0.0 {
                *v / set.repeat_penalty
            } else {
                *v * set.repeat_penalty
            };
        }
    }

    if set.temperature <= 0.0 {
        return vec![(argmax(&l), 1.0)];
    }

    let mut ids: Vec<u32> = (0..l.len() as u32).collect();
    ids.sort_by(|a, b| l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b)));

    let k = if set.top_k == 0 {
        ids.len()
    } else {
        set.top_k.min(ids.len())
    };
    ids.truncate(k);

    let max = l[ids[0] as usize] as f64;
    let mut probs: Vec<f64> = ids
        .iter()
        .map(|&id| ((l[id as usize] as f64 - max) / set.temperature as f64).exp())
        .collect();

    if set.top_p < 1.0 {
        let total: f64 = probs.iter().sum();
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (j, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= set.top_p as f64 * total {
                keep = j + 1;
                break;
            }
        }
        ids.truncate(keep);
        probs.truncate(keep);
    }

    if set.min_p > 0.0 {
        let p_max = probs.iter().copied().fold(0.0f64, f64::max);
        let keep: Vec<usize> = (0..probs.len())
            .filter(|&j| probs[j] >= set.min_p as f64 * p_max)
            .collect();
        ids = keep.iter().map(|&j| ids[j]).collect();
        probs = keep.iter().map(|&j| probs[j]).collect();
    }

    let total: f64 = probs.iter().sum();
    ids.into_iter()
        .zip(probs.iter().map(|p| p / total))
        .collect()
}

/// A deterministic logit vector, so a failure is reproducible from the seed in
/// the message.
fn logits(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Uniform in [-spread, spread).
            ((s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * spread
        })
        .collect()
}

/// Compares the sampler's surviving candidate set against the reference's,
/// ignoring the order the two happen to hold them in.
fn same_candidates(what: &str, got: (&[u32], &[f32]), want: &[(u32, f64)]) {
    let (ids, probs) = got;
    let mut mine: Vec<(u32, f32)> = ids.iter().copied().zip(probs.iter().copied()).collect();
    mine.sort_by_key(|&(id, _)| id);
    let mut theirs = want.to_vec();
    theirs.sort_by_key(|&(id, _)| id);

    let my_ids: Vec<u32> = mine.iter().map(|&(id, _)| id).collect();
    let their_ids: Vec<u32> = theirs.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        my_ids, their_ids,
        "{what}: the surviving candidates differ from the reference's"
    );
    for (&(id, mine), &(_, theirs)) in mine.iter().zip(&theirs) {
        let d = (mine as f64 - theirs).abs();
        assert!(
            d <= 1e-5 * theirs.max(1e-6),
            "{what}: token {id} has probability {mine} here and {theirs} in the \
             reference, off by {d}"
        );
    }
}

/// Every parameter on its own, and all of them together, against the naive
/// reference.
#[test]
fn the_pipeline_matches_a_naive_reference() {
    let cases: &[(&str, Sampling)] = &[
        (
            "temperature only",
            Sampling {
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                seed: 1,
            },
        ),
        (
            "a cold temperature",
            Sampling {
                temperature: 0.05,
                top_k: 0,
                top_p: 1.0,
                ..Sampling::default()
            },
        ),
        (
            "a hot temperature",
            Sampling {
                temperature: 4.0,
                top_k: 0,
                top_p: 1.0,
                ..Sampling::default()
            },
        ),
        (
            "top-k only",
            Sampling {
                temperature: 1.0,
                top_k: 17,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                seed: 2,
            },
        ),
        (
            "top-p only",
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.6,
                min_p: 0.0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                seed: 3,
            },
        ),
        (
            "min-p only",
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.3,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                seed: 4,
            },
        ),
        ("the defaults", Sampling::default()),
        (
            "everything at once",
            Sampling {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.02,
                repeat_penalty: 1.3,
                repeat_last_n: 16,
                seed: 5,
            },
        ),
    ];

    for &(what, set) in cases {
        for seed in [1u64, 2, 3] {
            for spread in [0.5f32, 4.0, 30.0] {
                let l = logits(2048, seed, spread);
                let mut sampler = Sampler::new(l.len(), set);
                // A window with a repeat in it, so the dedup is exercised too.
                let recent: Vec<u32> = [3u32, 11, 400, 11, 7, 1999, 0, 3].to_vec();
                for &t in &recent {
                    sampler.remember(t);
                }
                let token = sampler.sample(&l);
                let want = reference(&l, set, &recent);
                let label = format!("{what}, logit seed {seed}, spread {spread}");
                same_candidates(&label, sampler.candidates(), &want);
                assert!(
                    want.iter().any(|&(id, _)| id == token),
                    "{label}: drew {token}, which is not a candidate"
                );
            }
        }
    }
}

#[test]
fn temperature_zero_is_argmax() {
    for seed in [1u64, 9, 77] {
        let l = logits(4096, seed, 6.0);
        let mut sampler = Sampler::new(l.len(), Sampling::GREEDY);
        assert_eq!(sampler.sample(&l), argmax(&l), "logit seed {seed}");
    }

    // Exact ties go to the lowest id, and a vector with no finite entry has no
    // distribution at all; both have to come out as argmax does.
    let mut ties = vec![-1.0f32; 64];
    ties[7] = 5.0;
    ties[40] = 5.0;
    let mut sampler = Sampler::new(ties.len(), Sampling::GREEDY);
    assert_eq!(sampler.sample(&ties), 7);

    let dead = vec![f32::NEG_INFINITY; 64];
    let mut sampler = Sampler::new(dead.len(), Sampling::GREEDY);
    assert_eq!(sampler.sample(&dead), 0);
    let mut sampler = Sampler::new(
        dead.len(),
        Sampling {
            temperature: 0.8,
            ..Sampling::default()
        },
    );
    assert_eq!(sampler.sample(&dead), 0, "a drawing sampler on all -inf");
}

/// Greedy has to be byte-identical to what the gate recorded, so the penalty
/// being off must leave the logits completely untouched.
#[test]
fn a_penalty_of_one_changes_nothing() {
    let l = logits(1024, 5, 3.0);
    let set = Sampling {
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        ..Sampling::GREEDY
    };
    let mut sampler = Sampler::new(l.len(), set);
    for t in 0..64u32 {
        sampler.remember(t);
    }
    assert_eq!(sampler.sample(&l), argmax(&l));
}

#[test]
fn the_same_seed_gives_the_same_tokens_and_another_seed_does_not() {
    let l = logits(4096, 12, 2.0);
    let run = |seed: u64| -> Vec<u32> {
        let mut s = Sampler::new(
            l.len(),
            Sampling {
                seed,
                ..Sampling::default()
            },
        );
        (0..32).map(|_| s.sample(&l)).collect()
    };
    let a = run(7);
    assert_eq!(a, run(7), "the same seed gave a different sequence");
    let b = run(8);
    assert_ne!(a, b, "two seeds gave the same 32 tokens");
}

#[test]
fn the_edges_of_every_parameter() {
    let l = logits(1024, 21, 3.0);
    let base = Sampling {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: 99,
    };
    let top = argmax(&l);

    let edges: &[(&str, Sampling, usize)] = &[
        ("top_k 1", Sampling { top_k: 1, ..base }, 1),
        (
            "top_k at the vocabulary",
            Sampling {
                top_k: l.len(),
                ..base
            },
            l.len(),
        ),
        (
            "top_k past the vocabulary",
            Sampling {
                top_k: l.len() * 3,
                ..base
            },
            l.len(),
        ),
        ("top_p 0", Sampling { top_p: 0.0, ..base }, 1),
        ("min_p 1", Sampling { min_p: 1.0, ..base }, 1),
    ];
    for &(what, set, want) in edges {
        let mut s = Sampler::new(l.len(), set);
        let token = s.sample(&l);
        let (ids, probs) = s.candidates();
        assert_eq!(ids.len(), want, "{what}: kept {} candidates", ids.len());
        same_candidates(what, (ids, probs), &reference(&l, set, &[]));
        if want == 1 {
            assert_eq!(
                token, top,
                "{what}: a single candidate has to be the argmax"
            );
            assert_eq!(
                probs[0], 1.0,
                "{what}: a single candidate carries all the mass"
            );
        }
    }

    // top_p 1 and min_p 0 are off, so they leave the whole vocabulary.
    for (what, set) in [
        ("top_p 1", Sampling { top_p: 1.0, ..base }),
        ("min_p 0", Sampling { min_p: 0.0, ..base }),
    ] {
        let mut s = Sampler::new(l.len(), set);
        s.sample(&l);
        assert_eq!(s.candidates().0.len(), l.len(), "{what} cut something");
    }
}

/// The penalty divides a positive logit and multiplies a negative one, and a
/// token outside the window keeps the logit the forward pass produced.
#[test]
fn the_repetition_penalty_moves_both_signs_towards_zero() {
    let l = [2.0f32, -2.0, 1.0, -1.0, 0.0];
    let set = Sampling {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 2.0,
        repeat_last_n: 4,
        seed: 1,
    };
    let mut s = Sampler::new(l.len(), set);
    for t in [0u32, 1] {
        s.remember(t);
    }
    s.sample(&l);
    let (ids, probs) = s.candidates();
    let p: std::collections::HashMap<u32, f32> =
        ids.iter().copied().zip(probs.iter().copied()).collect();

    // 2.0 / 2 = 1.0, so token 0 ends up at token 2's untouched logit;
    // -2.0 * 2 = -4.0, so token 1 drops below token 3's untouched -1.0.
    assert!(
        (p[&0] - p[&2]).abs() < 1e-6,
        "a penalised +2 should land on an untouched +1: {} vs {}",
        p[&0],
        p[&2]
    );
    assert!(
        p[&1] < p[&3],
        "a penalised -2 should sit below an untouched -1"
    );
    // Tokens 2, 3 and 4 were never in the window, so their logits are still
    // 1, -1 and 0 and the ratios between them are `exp` of the differences.
    // The ratios are what survives the renormalisation the penalty forces.
    for (id, other, want) in [(2u32, 4u32, 1.0f32.exp()), (4, 3, 1.0f32.exp())] {
        let got = p[&id] / p[&other];
        assert!(
            (got - want).abs() < 1e-5,
            "tokens {id} and {other} were outside the window, but their ratio is \
             {got} instead of {want}"
        );
    }
}

/// Over many draws the sampler has to produce the distribution it reports.
#[test]
fn the_draw_follows_the_reported_probabilities() {
    let l = logits(512, 31, 1.0);
    let set = Sampling {
        temperature: 1.0,
        top_k: 8,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: 4,
    };
    let mut s = Sampler::new(l.len(), set);
    s.sample(&l);
    let want: Vec<(u32, f32)> = {
        let (ids, probs) = s.candidates();
        ids.iter().copied().zip(probs.iter().copied()).collect()
    };

    let draws = 50_000;
    let mut counts = std::collections::HashMap::new();
    for _ in 0..draws {
        *counts.entry(s.sample(&l)).or_insert(0usize) += 1;
    }
    for (id, p) in want {
        let got = *counts.get(&id).unwrap_or(&0) as f64 / draws as f64;
        // Three standard errors, with a floor so a rare token is not judged on
        // noise alone.
        let se = (p as f64 * (1.0 - p as f64) / draws as f64).sqrt();
        assert!(
            (got - p as f64).abs() < 4.0 * se + 0.003,
            "token {id} was drawn {got:.4} of the time, reported {p:.4}"
        );
    }
    let ids: Vec<u32> = s.candidates().0.to_vec();
    for drawn in counts.keys() {
        assert!(
            ids.contains(drawn),
            "token {drawn} was drawn but is not one of the candidates {ids:?}"
        );
    }
}
