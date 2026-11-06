//! Drafting from the model's own multi-token-prediction block, against the
//! tiny fixture.
//!
//! **The one way an MTP block can be wrong is the way nothing fails.**
//! Verification is exact, so a block that computes nonsense costs acceptance
//! and never a token: the feature quietly becomes plain decode with extra work
//! in front of it. Nothing in the output can show that, so what is checked here
//! is not the output alone.
//!
//! The fixture's weights are noise, so its acceptance is meaningless and
//! nothing here measures one; the acceptance floor that does is the `--ignored`
//! test in `tests/gate.rs`. What is not meaningless is everything else:
//!
//! * the draft path produces the same tokens as speculation off, at every
//!   draft length,
//! * the head drafts the same tokens however the trunk reached the position it
//!   drafts from, which is what says the catch-up covers every position and the
//!   rollback leaves nothing of a rejected draft behind,
//! * a burst of accepted tokens leaves the state a sequential run leaves, with
//!   the block live and being caught up one row at a time,
//! * and a file without the block says so instead of drafting rubbish.
//!
//! What this fixture **cannot** show is whether the first draft step pairs its
//! token with the trunk's hidden state or with something stale: every route
//! here would be wrong in the same way, so every comparison would still pass.
//! That is what the `--ignored` acceptance floor in `tests/gate.rs` is for, and
//! it is why that test exists at all.

use llm_core::Gguf;
use rust_llm::forward::Runner;
use rust_llm::generate::{generate, generate_drafting};
use rust_llm::model::Model;
use rust_llm::sample::{Sampler, Sampling};
use rust_llm::{Stop, Tokenizer};

mod support;

const CONTEXT: usize = 128;
const THREADS: usize = 2;
const DRAFT: usize = 4;

/// A prompt that repeats itself, as in `tests/speculate.rs`: on the fixture's
/// weights it is the only way either drafter finds anything at all, and it
/// keeps the two tests comparable.
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

/// A runner with the block enabled, which has to be done before the first
/// token: the block is caught up as tokens go through and cannot be caught up
/// on positions already past.
fn drafting<'a>(model: &'a Model<'a>, context: usize, batch: usize, draft: usize) -> Runner<'a> {
    let mut runner = Runner::with_speculation(model, context, THREADS, batch, draft);
    assert!(runner.enable_mtp(), "the fixture has an MTP block");
    assert!(runner.drafts_from_mtp());
    runner
}

/// Drafting from the head must not change a token, at any draft length.
///
/// The same comparison `every_draft_length_generates_what_no_speculation_
/// generates` makes for injected drafts, with the head as the source: the
/// drafts are now whatever the block produces, which on these weights is
/// mostly nothing, and the point is that the answer is identical either way.
#[test]
fn drafting_from_the_head_changes_no_token() {
    let f = fixture("mtp-equivalence", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    assert!(model.mtp.is_some(), "the fixture carries blk.2");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");

    let mut off = Runner::new(&model, CONTEXT, THREADS);
    let (want, want_stop, _) = generate(
        &mut off,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        48,
        |_| true,
    );
    let want_pos = off.pos();
    assert!(want.len() > 24, "only {} tokens generated", want.len());

    for draft in 1..=6 {
        let mut runner = drafting(&model, CONTEXT, 16, draft);
        let (got, stop, timings) = generate(
            &mut runner,
            &mut Sampler::greedy(model.vocab()),
            &prompt,
            48,
            |_| true,
        );
        let spec = timings.speculation;
        assert_eq!(got, want, "draft {draft}: different tokens");
        assert_eq!(stop, want_stop, "draft {draft}: different stop reason");
        assert_eq!(runner.pos(), want_pos, "draft {draft}: different position");
        // The head has to have been asked, or this test is the plain path with
        // extra steps. What it offered is not asserted: on noise weights the
        // confidence floor stops most steps, which is the floor working.
        assert!(
            spec.passes() + spec.plain > 0,
            "draft {draft}: the loop never reached the drafter"
        );
        println!("draft {draft}: {}", spec.report());
    }
}

/// The head must draft the same tokens however the trunk reached the position
/// it is drafting from.
///
/// This is the test for the catch-up and the rollback, and it is the one that
/// fails when they are wrong. Three runners end on the same committed tokens at
/// the same position by three different routes:
///
/// * token by token, which is a batch of one per position;
/// * in batches of eight, so the catch-up runs several rows at once and each of
///   them has to be paired with the trunk's hidden state of the row in front of
///   it rather than with the batch's last;
/// * through `verify` and `accept(1)`, with four deliberately wrong rows after
///   every committed one, so every step leaves a rejected draft's entries in the
///   block's cache for the next catch-up to overwrite.
///
/// A catch-up that skipped a position, paired a token with the wrong hidden
/// state, or left a rejected draft's cache entry readable would make at least
/// one of the three draft something else. Nothing in the generated output would
/// show it.
#[test]
fn the_head_drafts_the_same_however_the_trunk_got_there() {
    let f = fixture("mtp-catch-up", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");
    let vocab = model.vocab();

    // The committed history every route ends on: the prompt plus what greedy
    // generation produces, so the tokens are ones the trunk really predicts.
    let mut plain = Runner::new(&model, CONTEXT, THREADS);
    let (tail, _, _) = generate(&mut plain, &mut Sampler::greedy(vocab), &prompt, 12, |_| {
        true
    });
    let mut history: Vec<u32> = prompt.clone();
    history.extend_from_slice(&tail);

    // The head is confident at some positions and not at others on these
    // weights, and three empty drafts would compare nothing. So the history is
    // cut back to the last prefix the head does draft from, found with a runner
    // of its own. The fixture is deterministic, so that prefix is the same on
    // every machine and every run.
    let history = {
        let mut probe = drafting(&model, CONTEXT, 1, DRAFT);
        let mut at = None;
        for i in 0..history.len() {
            probe.forward(history[i]);
            let mut out = [0u32; DRAFT];
            if i + 1 >= prompt.len() && probe.draft_mtp(&history[..i + 1], &mut out) > 0 {
                at = Some(i + 1);
            }
        }
        let at = at.expect("the head drafts from no position in this history");
        println!("comparing the three routes at {at} committed tokens");
        history[..at].to_vec()
    };

    // Route 1: one token per pass.
    let mut one = drafting(&model, CONTEXT, 1, DRAFT);
    for &t in &history {
        one.forward(t);
    }

    // Route 2: batches of eight.
    let mut batched = drafting(&model, CONTEXT, 8, DRAFT);
    let mut read = 0;
    while read < history.len() {
        let take = (history.len() - read).min(batched.max_batch());
        batched.forward_batch(&history[read..read + take]);
        read += take;
    }

    // Route 3: a verification batch per token, every draft of which is
    // rejected. `accept(1)` commits the first row and rolls the rest back.
    let mut rolled = drafting(&model, CONTEXT, 16, DRAFT);
    for (i, &t) in history.iter().enumerate() {
        let room = CONTEXT - rolled.pos();
        if room <= DRAFT + 1 || i + 1 == history.len() {
            rolled.forward(t);
            continue;
        }
        let mut rows = vec![t];
        // Tokens the trunk did not predict here, so no row after the first is
        // ever the right one; they are real vocabulary rows, so the pass runs.
        rows.extend((0..DRAFT).map(|j| (t + 1 + j as u32) % vocab as u32));
        rolled.verify(&rows);
        rolled.accept(1);
    }

    let mut drafts = [[0u32; DRAFT]; 3];
    let mut counts = [0usize; 3];
    for (slot, runner) in [&mut one, &mut batched, &mut rolled]
        .into_iter()
        .enumerate()
    {
        assert_eq!(runner.pos(), history.len(), "route {slot}: wrong position");
        counts[slot] = runner.draft_mtp(&history, &mut drafts[slot]);
    }
    println!(
        "token by token: {:?}, batched: {:?}, rolled back: {:?}",
        &drafts[0][..counts[0]],
        &drafts[1][..counts[1]],
        &drafts[2][..counts[2]],
    );
    assert_eq!(
        counts[0], counts[1],
        "a batched prompt left the block in a different state than a token-by-token one"
    );
    assert_eq!(
        drafts[0], drafts[1],
        "a batched prompt left the block in a different state than a token-by-token one"
    );
    assert_eq!(
        counts[0], counts[2],
        "a rejected draft left something behind in the block's cache"
    );
    assert_eq!(
        drafts[0], drafts[2],
        "a rejected draft left something behind in the block's cache"
    );
    // A draft of nothing would make the three comparisons above vacuous.
    assert!(
        counts[0] > 0,
        "the head drafted nothing, so the three routes were compared on an empty draft"
    );
}

/// A burst of accepted tokens, with the block live and caught up one row at a
/// time, has to leave exactly what a sequential run leaves.
///
/// The head's own drafts are almost never accepted on noise weights, so the
/// drafter here **runs the head and then offers the sequential run's tokens**.
/// That is deliberate: the block, its cache and its hidden state go through the
/// real path, and the loop then accepts, which is what makes `accept(1)` catch
/// the block up row by row inside a burst. The acceptance is the fixture's, the
/// code path is the product's.
///
/// The boundaries are the ones a burst can be cut short on: `max_tokens` inside
/// a batch, a cancelling `on_token`, and a context with fewer positions left
/// than the batch wants.
#[test]
fn a_burst_with_the_head_live_stops_where_a_sequential_run_stops() {
    let f = fixture("mtp-burst", support::UNREACHED_EOS);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");
    let sampled = Sampling {
        temperature: 0.8,
        top_k: 8,
        top_p: 0.9,
        min_p: 0.0,
        repeat_penalty: 1.1,
        repeat_last_n: 16,
        seed: 0x5eed_2026,
    };

    let cases = [
        ("a long run", 24, usize::MAX, CONTEXT),
        ("max-tokens inside a batch", 7, usize::MAX, CONTEXT),
        ("cancelled after 3", 24, 3, CONTEXT),
        ("four spare positions", 24, usize::MAX, prompt.len() + 4),
    ];
    for (case, max_tokens, cancel_after, context) in cases {
        for set in [Sampling::GREEDY, sampled] {
            let mut off = Runner::with_batch(&model, context, THREADS, 8);
            let mut seen = 0usize;
            let (want, want_stop, _) = generate(
                &mut off,
                &mut Sampler::new(model.vocab(), set),
                &prompt,
                max_tokens,
                |_| {
                    seen += 1;
                    seen < cancel_after
                },
            );
            let want_pos = off.pos();
            // Longer than any run here, so the drafter never runs out: a run
            // cut short asks for drafts it then never uses.
            let mut truth: Vec<u32> = prompt.clone();
            truth.extend_from_slice(&want);
            truth.extend(std::iter::repeat_n(want.last().copied().unwrap_or(0), 16));

            let mut runner = drafting(&model, context, 8, DRAFT);
            let mut sampler = Sampler::new(model.vocab(), set);
            let mut seen = 0usize;
            let mut asked = 0usize;
            let (got, stop, timings) = generate_drafting(
                &mut runner,
                &mut sampler,
                &prompt,
                max_tokens,
                |r, history, out| {
                    // The head really runs: its cache is written in the draft
                    // region and its hidden state is carried from step to step,
                    // which is the path the catch-up then has to overwrite.
                    let from_head = r.draft_mtp(history, out);
                    asked += 1;
                    let take = out.len().min(truth.len() - history.len()).max(from_head);
                    out[..take].copy_from_slice(&truth[history.len()..][..take]);
                    take
                },
                |_| {
                    seen += 1;
                    seen < cancel_after
                },
            );
            let at = format!("{case}, temperature {}: ", set.temperature);
            assert_eq!(got, want, "{at}different tokens");
            assert_eq!(stop, want_stop, "{at}different stop reason");
            assert_eq!(runner.pos(), want_pos, "{at}different position");
            let spec = timings.speculation;
            if want.len() > 1 {
                assert!(asked > 0, "{at}the loop never reached the drafter");
                assert!(
                    spec.passes() > 0,
                    "{at}no verification batch ran: {}",
                    spec.report()
                );
            }
            if case == "a long run" {
                assert!(
                    spec.saved() > 0,
                    "{at}no draft was accepted, so no burst happened: {}",
                    spec.report()
                );
            }
            println!("{at}{}", spec.report());
        }
    }
}

/// A file without the block refuses to draft from it rather than drafting
/// rubbish.
///
/// The conversion this runner started on is such a file (`docs/models.md`), and
/// it still has to load and run: the trunk is identical and only the head is
/// missing. So `enable_mtp` says no, `draft_mtp` offers nothing, and generation
/// is unaffected.
#[test]
fn a_file_without_the_block_cannot_draft_from_it() {
    let tiny = support::write_without_mtp("mtp-absent", support::UNREACHED_EOS);
    let gguf = Gguf::open(&tiny.model).expect("open the fixture");
    let tokenizer = Tokenizer::from_file(&tiny.tokenizer).expect("the fixture tokenizer");
    let model = Model::load(&gguf).expect("a file without the block still loads");
    assert!(model.mtp.is_none());
    assert_eq!(
        model.config.n_blocks,
        support::N_BLOCKS,
        "the trunk is intact"
    );

    let prompt = tokenizer.encode(REPETITIVE, true).expect("encode");
    let mut runner = Runner::with_speculation(&model, CONTEXT, THREADS, 16, DRAFT);
    assert!(!runner.enable_mtp(), "there is no block to enable");
    assert!(!runner.drafts_from_mtp());
    let mut out = [0u32; DRAFT];
    assert_eq!(runner.draft_mtp(&prompt, &mut out), 0);

    // And the file generates, which is the half of this that would be easy to
    // lose: refusing the head must not refuse the model.
    let (ids, _, _) = generate(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        8,
        |_| true,
    );
    assert_eq!(ids.len(), 8);
}

/// A run that stops on a stop token has to stop on it inside a burst too,
/// which is the boundary the fixture can only reach because `eos` is a
/// parameter of the writer.
#[test]
fn a_stop_token_inside_a_burst_ends_the_run() {
    // What a greedy run on this fixture produces first, so there is a stop
    // token the model actually reaches.
    let probe = fixture("mtp-eos-probe", support::UNREACHED_EOS);
    let model = Model::load(&probe.gguf).expect("load the probe fixture");
    let prompt = probe.tokenizer.encode(REPETITIVE, true).expect("encode");
    let mut runner = Runner::new(&model, CONTEXT, THREADS);
    let (ids, _, _) = generate(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        12,
        |_| true,
    );
    let eos = ids[6];

    let f = fixture("mtp-eos", eos);
    let model = Model::load(&f.gguf).expect("load the fixture");
    let prompt = f.tokenizer.encode(REPETITIVE, true).expect("encode");
    let mut off = Runner::new(&model, CONTEXT, THREADS);
    let (want, want_stop, _) = generate(
        &mut off,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        48,
        |_| true,
    );
    assert_eq!(
        want_stop,
        Stop::Eos(eos),
        "the probe's token is not the stop"
    );
    assert_eq!(
        want.len(),
        6,
        "the stop landed somewhere else than expected"
    );

    let mut truth: Vec<u32> = prompt.clone();
    truth.extend_from_slice(&want);
    truth.extend(std::iter::repeat_n(eos, 16));
    let mut runner = drafting(&model, CONTEXT, 8, DRAFT);
    let (got, stop, _) = generate_drafting(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        48,
        |r, history, out| {
            let from_head = r.draft_mtp(history, out);
            let take = out.len().min(truth.len() - history.len()).max(from_head);
            out[..take].copy_from_slice(&truth[history.len()..][..take]);
            take
        },
        |_| true,
    );
    assert_eq!(got, want, "a burst produced different tokens");
    assert_eq!(stop, want_stop, "a burst did not stop on the stop token");
    assert_eq!(
        runner.pos(),
        off.pos(),
        "a burst committed a different count"
    );
}
