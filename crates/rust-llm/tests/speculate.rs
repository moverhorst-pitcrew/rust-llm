//! Speculative decoding against the tiny fixture: what a verification batch
//! produces, and what a burst of accepted tokens does at every boundary the
//! sequential loop has.
//!
//! The fixture's weights are noise, so its *acceptance* is meaningless and
//! nothing here measures one. What is not meaningless is everything else: the
//! logits of a row, the rollback after a rejected draft, and where a run stops.
//! Those are properties of the pass and the loop, not of the weights, and this
//! is where they run in CI without the 2.7 GB file.
//!
//! The drafts in the first half are **injected**, not drafted. On random
//! weights a natural match would be an accident, and what has to be checked is
//! the two interesting cases on purpose: a batch whose drafts are all right and
//! one whose drafts are all wrong.

use llm_core::Gguf;
use rust_llm::forward::Runner;
use rust_llm::generate::generate;
use rust_llm::model::Model;
use rust_llm::sample::{Sampler, Sampling, argmax};
use rust_llm::{Stop, Tokenizer, generate as gen_mod};

mod support;

/// Positions to allocate. Larger than anything here needs, except where a test
/// asks for a smaller one on purpose.
const CONTEXT: usize = 128;
const THREADS: usize = 2;
/// Drafts per pass in the tests that do not vary it.
const DRAFT: usize = 4;

/// A prompt that repeats itself, so the drafter finds something to guess from
/// however meaningless the fixture's answer is. Without a repeat it would
/// never draft and the burst tests would all be testing the plain path.
const REPETITIVE: &str = "one two three one two three one two three one two three ";

struct Fixture {
    gguf: Gguf,
    tokenizer: Tokenizer,
}

fn fixture(name: &str, eos: u32) -> Fixture {
    let tiny = support::write(name, eos);
    Fixture {
        gguf: Gguf::open(&tiny.model).expect("open the fixture"),
        tokenizer: Tokenizer::from_file(&tiny.tokenizer).expect("the fixture tokenizer"),
    }
}

/// The logits of every row of a verification batch have to be the logits the
/// same tokens produce one pass at a time, **bit for bit**, and the state the
/// batch leaves behind has to be the state those passes would have left.
///
/// Three batches over the same prefix: one whose drafts are the model's own
/// next tokens, so every draft is right; one whose drafts are all wrong; and
/// one that is right up to the middle and wrong after it. All three have to
/// give identical logits for every row, because a row's logits do not depend on
/// whether the row after it was guessed correctly: that is the whole premise of
/// verifying a draft, and a batch that got it wrong would show as a verified
/// draft being accepted on the strength of logits nobody would have produced.
///
/// Then the rollback. After committing `k` rows the runner has to be
/// indistinguishable from one that only ever saw those `k` tokens: the
/// comparison is the next pass's whole logit vector *and* the tokens generated
/// after it. The logits alone would miss a stale recurrent state that only the
/// token after them reads; the tokens alone would miss a wrong state that the
/// decay fades below an argmax before it shows.
#[test]
fn a_verification_batch_is_the_same_as_the_passes_it_replaces() {
    let f = fixture("speculate-rows", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");
    let vocab = model.vocab();

    // The sequential side: the model's own tokens, and the logits of every
    // pass that produced them.
    let mut single = Runner::new(&model, CONTEXT, THREADS);
    let mut rows = Vec::new();
    let mut want: Vec<Vec<f32>> = Vec::new();
    let mut next = read_prompt(&mut single, &prompt);
    for _ in 0..=DRAFT {
        rows.push(next);
        let logits = single.forward(next).to_vec();
        next = argmax(&logits);
        want.push(logits);
    }
    assert_eq!(rows.len(), DRAFT + 1);
    // The tokens after the batch, from the runner that generated it the slow
    // way, per number of committed rows.
    let tail: Vec<Vec<u32>> = (1..=rows.len())
        .map(|keep| {
            let mut r = Runner::new(&model, CONTEXT, THREADS);
            let mut next = read_prompt(&mut r, &prompt);
            for &row in &rows[..keep] {
                assert_eq!(next, row, "the sequential run left the batch's path");
                next = argmax(r.forward(row));
            }
            let mut out = vec![next];
            for _ in 0..4 {
                next = argmax(r.forward(next));
                out.push(next);
            }
            out
        })
        .collect();

    // Wrong drafts: a token the model did not pick. Every row of the fixture's
    // vocabulary is a real row, so this is a token the pass can actually run.
    let wrong = |right: u32| (right + 1) % vocab as u32;
    let batches = [
        ("every draft right", rows.clone()),
        (
            "every draft wrong",
            std::iter::once(rows[0])
                .chain(rows[1..].iter().map(|&t| wrong(t)))
                .collect(),
        ),
        (
            "wrong in the middle",
            rows.iter()
                .enumerate()
                .map(|(i, &t)| if i == 3 { wrong(t) } else { t })
                .collect::<Vec<u32>>(),
        ),
    ];

    for (what, batch) in batches {
        let mut runner = Runner::with_speculation(&model, CONTEXT, THREADS, 16, DRAFT);
        let at = read_prompt(&mut runner, &prompt);
        assert_eq!(at, rows[0], "the prompt's prediction moved");
        let logits = runner.verify(&batch);
        assert_eq!(logits.len(), batch.len() * vocab);

        // How many of the drafts a real accept loop would have taken, which is
        // how far the rows agree with what the model picks.
        let accepted = batch[1..]
            .iter()
            .zip(&rows[1..])
            .take_while(|(a, b)| a == b)
            .count();
        for (r, want) in want.iter().enumerate().take(accepted + 1) {
            assert_eq!(
                runner.logit_row(r),
                &want[..],
                "{what}: row {r}'s logits are not the pass's"
            );
        }

        // The rollback: commit the rows a rejection at `accepted` would commit.
        let keep = accepted + 1;
        runner.accept(keep);
        assert_eq!(runner.pos(), prompt.len() + keep, "{what}: position");
        let mut got = vec![argmax(runner.logit_row(accepted))];
        for _ in 0..4 {
            let last = *got.last().expect("a token");
            got.push(argmax(runner.forward(last)));
        }
        assert_eq!(
            got,
            tail[keep - 1],
            "{what}: after committing {keep} of {} rows the runner does not \
             generate what one that never saw the rest does",
            batch.len()
        );
        // And the logits of the pass after the rollback, whole, not the token
        // they argmax to.
        let mut plain = Runner::new(&model, CONTEXT, THREADS);
        let mut next = read_prompt(&mut plain, &prompt);
        for &row in &rows[..keep] {
            next = argmax(plain.forward(row));
        }
        let fresh = plain.forward(next).to_vec();
        let mut rolled = Runner::with_speculation(&model, CONTEXT, THREADS, 16, DRAFT);
        read_prompt(&mut rolled, &prompt);
        rolled.verify(&batch);
        rolled.accept(keep);
        assert_eq!(
            rolled.forward(next),
            &fresh[..],
            "{what}: the logits after a rollback of {} rows are not a fresh runner's",
            batch.len() - keep
        );
    }
}

/// Speculation must not change a single token, a single stop reason or the
/// position the caches end at, at any of the boundaries the sequential loop
/// has: a stop token, `max_tokens`, a cancelling `on_token` and a context that
/// fills up. All four can land in the middle of a burst, which is the whole
/// risk: the batch has already run tokens the run is not allowed to emit, and
/// one of them must not be streamed, counted, remembered or left in the caches.
///
/// The drafts are the model's own next tokens, taken from a sequential run, so
/// every draft is accepted and every burst is the full `draft + 1` tokens wide.
/// That is on purpose: on the fixture's random weights a drafted token is
/// almost never the one the sampler picks, so a prompt-lookup draft would make
/// all of this test the plain path. The other side, a draft that is never
/// accepted, is the second drafter below.
///
/// Greedy and sampled with a fixed seed, because the sampled side is where the
/// RNG order shows: the equivalence is bit-equality, not equality in
/// distribution, so one draw too many or too few would change every token after
/// it.
#[test]
fn a_burst_stops_exactly_where_a_sequential_run_stops() {
    // An end-of-sequence token the fixture reaches part-way through an answer,
    // found by running it: the weights are noise, so there is no telling in
    // advance. Taken from a run of its own, with a fixture that cannot stop.
    let probe = fixture("speculate-probe", support::UNREACHED_EOS);
    let model = Model::load(&probe.gguf).expect("load the fixture");
    let prompt = probe.tokenizer.encode(REPETITIVE, true).expect("encode");
    let mut runner = Runner::new(&model, CONTEXT, THREADS);
    let (ids, _, _) = generate(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        12,
        |_| true,
    );
    assert!(ids.len() > 6, "the fixture generated {} tokens", ids.len());
    let reachable_eos = ids[5];

    let sampled = Sampling {
        temperature: 0.9,
        top_k: 16,
        top_p: 0.9,
        min_p: 0.0,
        repeat_penalty: 1.2,
        repeat_last_n: 16,
        seed: 0xfeed,
    };

    for (what, eos) in [
        ("no stop token in reach", support::UNREACHED_EOS),
        ("a stop token in the middle", reachable_eos),
    ] {
        let f = fixture(&format!("speculate-{eos}"), eos);
        let model = Model::load(&f.gguf).expect("load the fixture");
        let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");

        // max_tokens, the number of tokens after which `on_token` gives up, and
        // the context. A context of `prompt + 3` fills up inside the first
        // burst; the rest stop at an arbitrary point inside one.
        let cases = [
            ("a long run", 12usize, usize::MAX, CONTEXT),
            ("max_tokens of 1", 1, usize::MAX, CONTEXT),
            ("max_tokens of 3", 3, usize::MAX, CONTEXT),
            ("max_tokens of 5", 5, usize::MAX, CONTEXT),
            ("cancelled after 1", 12, 1, CONTEXT),
            ("cancelled after 2", 12, 2, CONTEXT),
            ("cancelled after 6", 12, 6, CONTEXT),
            ("three spare positions", 12, usize::MAX, prompt.len() + 3),
            ("six spare positions", 12, usize::MAX, prompt.len() + 6),
        ];
        for (case, max_tokens, cancel_after, context) in cases {
            for set in [Sampling::GREEDY, sampled] {
                // The sequential side, which is also where the drafts come
                // from: the whole token history a run with no speculation
                // produces, prompt included.
                let sequential = || {
                    let mut runner = Runner::with_batch(&model, context, THREADS, 8);
                    let mut sampler = Sampler::new(model.vocab(), set);
                    let mut seen = 0usize;
                    let (ids, stop, _) =
                        generate(&mut runner, &mut sampler, &prompt, max_tokens, |_| {
                            seen += 1;
                            seen < cancel_after
                        });
                    (ids, stop, runner.pos())
                };
                let (want, want_stop, want_pos) = sequential();
                // Longer than any run here can be, so the drafter never runs
                // out of tokens to offer: a run that is cut short asks for
                // drafts it then never uses.
                let mut truth: Vec<u32> = prompt.clone();
                truth.extend_from_slice(&want);
                truth.extend(std::iter::repeat_n(want.last().copied().unwrap_or(0), 16));

                for (drafter, draft) in [("right", DRAFT), ("wrong", DRAFT)] {
                    let mut runner = Runner::with_speculation(&model, context, THREADS, 8, draft);
                    let mut sampler = Sampler::new(model.vocab(), set);
                    let mut seen = 0usize;
                    let mut offered = 0usize;
                    let (got, stop, timings) = gen_mod::generate_drafting(
                        &mut runner,
                        &mut sampler,
                        &prompt,
                        max_tokens,
                        |history, out| {
                            offered += 1;
                            let take = out.len().min(truth.len() - history.len());
                            for (i, slot) in out[..take].iter_mut().enumerate() {
                                let right = truth[history.len() + i];
                                // A token the sampler will not pick, so the
                                // first comparison of every batch fails.
                                *slot = if drafter == "right" {
                                    right
                                } else {
                                    (right + 1) % model.vocab() as u32
                                };
                            }
                            take
                        },
                        |_| {
                            seen += 1;
                            seen < cancel_after
                        },
                    );
                    let at = format!(
                        "{what}, {case}, temperature {}, {drafter} drafts: ",
                        set.temperature
                    );
                    assert_eq!(got, want, "{at}different tokens");
                    assert_eq!(stop, want_stop, "{at}different stop reason");
                    assert_eq!(runner.pos(), want_pos, "{at}different position");
                    // The point of the whole test: the paths above only mean
                    // something if a batch really ran.
                    if want.len() > 1 {
                        assert!(
                            offered > 0 && timings.speculation.passes() > 0,
                            "{at}no verification batch ran: {}",
                            timings.speculation.report()
                        );
                    }
                    let spec = timings.speculation;
                    if drafter == "right" && case == "a long run" {
                        assert!(
                            spec.saved() > 0,
                            "{at}no draft was accepted, so no burst happened: {}",
                            spec.report()
                        );
                    }
                    if drafter == "wrong" {
                        assert_eq!(
                            spec.saved(),
                            0,
                            "{at}a draft the sampler cannot pick was accepted: {}",
                            spec.report()
                        );
                    }
                }
            }
        }
    }
}

/// The ring rotates across batches, so the live buffer after a commit is
/// wherever that batch ended rather than a fixed slot. A run long enough to
/// wrap it many times has to come out the same as one with no ring at all.
///
/// Every draft length from 1 to 6, because which slot a row writes is
/// arithmetic modulo the ring size and an off-by-one there would show at some
/// lengths and not others. The drafts are the model's own tokens again, so
/// every batch is full width and the ring turns a whole slot per pass.
#[test]
fn every_draft_length_generates_what_no_speculation_generates() {
    let f = fixture("speculate-lengths", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");

    let mut runner = Runner::new(&model, CONTEXT, THREADS);
    let mut sampler = Sampler::greedy(model.vocab());
    let (want, want_stop, _) = generate(&mut runner, &mut sampler, &prompt, 48, |_| true);
    let want_pos = runner.pos();
    assert!(want.len() > 24, "only {} tokens generated", want.len());
    let mut truth: Vec<u32> = prompt.clone();
    truth.extend_from_slice(&want);
    truth.extend(std::iter::repeat_n(0, 16));

    for draft in 1..=6 {
        let mut runner = Runner::with_speculation(&model, CONTEXT, THREADS, 16, draft);
        let mut sampler = Sampler::greedy(model.vocab());
        let (got, stop, timings) = gen_mod::generate_drafting(
            &mut runner,
            &mut sampler,
            &prompt,
            48,
            |history, out| {
                let take = out.len().min(truth.len() - history.len());
                out[..take].copy_from_slice(&truth[history.len()..][..take]);
                take
            },
            |_| true,
        );
        let spec = timings.speculation;
        assert_eq!(got, want, "draft {draft}: different tokens");
        assert_eq!(stop, want_stop, "draft {draft}: different stop reason");
        assert_eq!(runner.pos(), want_pos, "draft {draft}: different position");
        assert!(
            spec.passes() > 2,
            "draft {draft}: {} verification batches is too few to have wrapped \
             the ring: {}",
            spec.passes(),
            spec.report()
        );
        println!("draft {draft}: {}", spec.report());
    }
}

/// A draft longer than the ring is a caller mistake, and `set_draft` says so
/// rather than overrunning a buffer that was never allocated.
#[test]
fn a_draft_longer_than_the_ring_is_refused() {
    let f = fixture("speculate-cap", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let mut runner = Runner::with_speculation(&model, CONTEXT, THREADS, 16, 3);
    assert_eq!(runner.draft(), 3);
    assert!(runner.set_draft(0));
    assert_eq!(runner.draft(), 0);
    assert!(runner.set_draft(3));
    assert!(!runner.set_draft(4), "a draft of 4 fits a ring of 4 slots");
    assert_eq!(runner.draft(), 3, "a refused draft changed the runner");
}

/// The histogram has to count what happened, because it is the number a draft
/// length is tuned on and a miscount there is invisible.
///
/// Every generated token came out of exactly one verification row or one plain
/// pass, so the three counters have to add up to the tokens, and no pass can
/// have accepted more drafts than the draft length.
#[test]
fn the_acceptance_histogram_adds_up() {
    let f = fixture("speculate-histogram", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");

    let mut runner = Runner::new(&model, CONTEXT, THREADS);
    let mut sampler = Sampler::greedy(model.vocab());
    let (want, _, _) = generate(&mut runner, &mut sampler, &prompt, 40, |_| true);
    let mut truth: Vec<u32> = prompt.clone();
    truth.extend_from_slice(&want);
    truth.extend(std::iter::repeat_n(0, 16));

    // Drafts that are right two passes out of three, so the histogram has
    // something other than one bar in it.
    let mut pass = 0usize;
    let mut runner = Runner::with_speculation(&model, CONTEXT, THREADS, 16, DRAFT);
    let mut sampler = Sampler::greedy(model.vocab());
    let (ids, stop, timings) = gen_mod::generate_drafting(
        &mut runner,
        &mut sampler,
        &prompt,
        40,
        |history, out| {
            pass += 1;
            let take = out.len().min(truth.len() - history.len());
            for (i, slot) in out[..take].iter_mut().enumerate() {
                let right = truth[history.len() + i];
                // Right up to position `pass % (DRAFT + 1)` and wrong after it,
                // so the accepted lengths spread over the histogram.
                *slot = if i < pass % (DRAFT + 1) {
                    right
                } else {
                    (right + 1) % model.vocab() as u32
                };
            }
            take
        },
        |_| true,
    );
    let s = timings.speculation;

    // Every token the loop produced came out of exactly one of three places: a
    // verification batch's first row, a draft accepted inside one, or a plain
    // pass. Two tokens are not among them, which is what the correction is: the
    // first token of the run came out of the prompt's own logits, and a run that
    // ended on a stop token produced one token it did not emit, the stop token.
    let produced = s.passes() + s.saved() + s.plain;
    let emitted = produced + 1 - usize::from(matches!(stop, Stop::Eos(_)));
    assert_eq!(
        emitted,
        ids.len(),
        "{} passes plus {} accepted drafts plus {} plain passes is {emitted} \
         tokens, {stop:?}, against the {} emitted: {}",
        s.passes(),
        s.saved(),
        s.plain,
        ids.len(),
        s.report()
    );
    assert!(s.offered >= s.saved(), "more accepted than offered");
    assert!(
        (0.0..=1.0).contains(&s.acceptance()),
        "acceptance {} is not a fraction",
        s.acceptance()
    );
    assert!(
        s.accepted[DRAFT + 1..].iter().all(|&n| n == 0),
        "a pass accepted more drafts than it offered: {}",
        s.report()
    );
    // The spread the drafter above was built to produce, so this is also the
    // check that a histogram with one bar in it would have failed.
    assert!(
        s.accepted[..=DRAFT].iter().filter(|&&n| n > 0).count() >= 3,
        "the accepted lengths landed in fewer than three buckets: {}",
        s.report()
    );
    println!("{}", s.report());
}

/// Reads the prompt and returns the token it predicts.
fn read_prompt(runner: &mut Runner<'_>, prompt: &[u32]) -> u32 {
    let mut logits = Vec::new();
    for &t in prompt {
        logits = runner.forward(t).to_vec();
    }
    argmax(&logits)
}
