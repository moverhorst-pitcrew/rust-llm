//! The correctness gate: greedy generation against llama.cpp on the same GGUF
//! file, over the committed prompt set in `fixtures/gate_prompts.json`.
//!
//! Four tests, all ignored by default because they need the weights.
//! `RUST_LLM_MODEL` is the path to the GGUF file and has no default, so it goes
//! in front of every command below:
//!
//! ```text
//! # correctness: every next-token prediction against the reference  (~2 min)
//! cargo test -p rust-llm --release --test gate -- --ignored --nocapture every_next_token
//!
//! # regression: free-running output against the committed fixture   (~2 min)
//! cargo test -p rust-llm --release --test gate -- --ignored --nocapture greedy_output_is
//!
//! # the caches: reset, and the hand-over from prompt to generation   (~30 s)
//! cargo test -p rust-llm --release --test gate -- --ignored --nocapture resetting
//!
//! # re-derive one prompt of the fixture from a local llama.cpp build (~30 s)
//! RUST_LLM_REFERENCE_BIN=$REF_DIR/llama.cpp/build/bin \
//!   RUST_LLM_GATE_ONLY=capital_no_think \
//!   cargo test -p rust-llm --release --test gate -- --ignored --nocapture the_fixture_is_still
//! ```
//!
//! The first three need only the model, which is what keeps the gate runnable
//! without llama.cpp; the last is what keeps the fixture from rotting. Set
//! `RUST_LLM_WRITE_FIXTURE=1` on it to write the fixture instead of comparing
//! against it. They all run against the dispatched fast path, never a pinned
//! scalar tier, and print which tier that was.
//!
//! `RUST_LLM_GATE_ONLY` is a substring of a case name and holds one command to
//! one prompt, for writing as much as for checking: a case is merged into the
//! fixture by name as soon as it is produced, so a command that is stopped
//! part-way keeps its work and the next one continues. Without it the whole set
//! is about five minutes of reference time, which still fits in one command but
//! leaves nothing to resume from.
//!
//! ## Why there are two gates rather than one
//!
//! Quantizing activations to int8 is a discontinuity. Two implementations that
//! differ by one ulp anywhere flip a quantization level in the next matmul,
//! and a flipped level moves every output of that matmul by about 1e-3
//! relative, correlated across all its rows. Measured here: our matmuls agree
//! with the reference's to 2.5e-7 in block 0 and the two diverge to about 1e-2
//! on the logits by block 3, where it then stays. On a logit scale whose
//! next-token margins are typically 0.25 to 1.2 that is a band of roughly
//! +/-0.1 inside which neither implementation's choice is determined by the
//! model; it is determined by rounding.
//!
//! This is not specific to this runner. llama.cpp disagrees with *itself*
//! across its own two kernel paths on this prompt set: with weight repacking
//! on it answers "Here's a thinking process that leads to the answer", with
//! `--no-repack` it answers "Here's my thought process for solving this word
//! problem", from the same file and the same greedy sampler.
//!
//! So free-running token-for-token agreement is not a property an
//! implementation can have unless its kernels are bit-identical to the
//! reference's, which these deliberately are not. The gate is therefore split:
//!
//! * [`every_next_token_matches_the_reference`] is the correctness gate. It
//!   feeds the reference's own tokens, so every position is an independent
//!   comparison of one forward pass and a flipped near-tie cannot hide the
//!   predictions behind it. Disagreements are allowed only at positions
//!   committed in the fixture, with their measured margin.
//! * [`greedy_output_is_unchanged`] is the regression gate: free-running
//!   output must equal the committed sequence, token for token. For most
//!   prompts that sequence *is* the reference's; the fixture records both and
//!   the test says which.

use std::path::{Path, PathBuf};
use std::process::Command;

use llm_core::Gguf;
use rust_llm::Tokenizer;
use rust_llm::chat::{self, ChatOptions, Message, Role};
use rust_llm::forward::Runner;
use rust_llm::generate::generate;
use rust_llm::model::Model;
use rust_llm::sample::{Sampler, argmax};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
/// Enough for every prompt in the set plus its answer, and small enough that
/// the KV cache stays a rounding error next to the weights.
const CONTEXT: usize = 1024;
/// How far apart the top two logits have to be before a disagreement with the
/// reference counts as a real one rather than as rounding. Chosen from the
/// measured divergence of the two implementations' logits, about 1e-2 relative
/// on a scale of +/-20, with room to spare; typical margins are twenty times
/// this.
const INDETERMINATE: f32 = 0.25;

/// The model file, from `RUST_LLM_MODEL`.
///
/// No default: a path baked into the repository is a path on one machine. The
/// message is what someone who has just cloned this needs to read.
fn model_path() -> String {
    std::env::var("RUST_LLM_MODEL")
        .unwrap_or_else(|_| panic!("set RUST_LLM_MODEL to the GGUF file (see docs/models.md)"))
}

/// One prompt as the gate drives it.
struct Case {
    name: String,
    /// The rendered prompt, which is also exactly what the reference is fed.
    prompt: String,
    max_tokens: usize,
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn fixture() -> serde_json::Value {
    serde_json::from_str(&read(&format!("{FIXTURES}/reference_generations.json")))
        .expect("parse reference_generations.json")
}

/// Renders the committed prompt set. Every test goes through this, so no two
/// of them can be comparing different prompts.
fn cases() -> Vec<Case> {
    let f: serde_json::Value =
        serde_json::from_str(&read(&format!("{FIXTURES}/gate_prompts.json")))
            .expect("parse gate_prompts.json");
    let cases = f["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 8,
        "the gate needs at least 8 prompts, the fixture has {}",
        cases.len()
    );
    let out: Vec<Case> = cases
        .iter()
        .map(|case| {
            let name = case["name"].as_str().expect("name").to_string();
            let max_tokens = case["max_tokens"].as_u64().expect("max_tokens") as usize;
            let prompt = match case["raw"].as_str() {
                Some(raw) => raw.to_string(),
                None => {
                    let messages: Vec<Message> = case["messages"]
                        .as_array()
                        .unwrap_or_else(|| panic!("{name}: messages or raw"))
                        .iter()
                        .map(|m| {
                            let role: Role = m["role"]
                                .as_str()
                                .expect("role")
                                .parse()
                                .unwrap_or_else(|e| panic!("{name}: {e}"));
                            Message::new(role, m["content"].as_str().expect("content"))
                        })
                        .collect();
                    chat::render(
                        &messages,
                        ChatOptions {
                            add_generation_prompt: true,
                            thinking: case["thinking"].as_bool().expect("thinking"),
                        },
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                }
            };
            Case {
                name,
                prompt,
                max_tokens,
            }
        })
        .collect();
    assert!(
        out.iter().all(|c| c.max_tokens >= 32),
        "every case must ask for at least 32 tokens"
    );
    out
}

fn open_model() -> Gguf {
    let path = model_path();
    Gguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"))
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(format!("{FIXTURES}/tokenizer.json")).expect("tokenizer.json")
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// The correctness gate.
///
/// Feeds the prompt and then the reference's own tokens, checking the
/// prediction made at every position against the token the reference chose
/// there. Disagreements are permitted only where the fixture records one, and
/// only when the margin is inside the band the two implementations' arithmetic
/// cannot resolve; see the module comment.
#[test]
#[ignore = "needs the model file"]
fn every_next_token_matches_the_reference() {
    let cases = cases();
    let fixture = fixture();
    let expected = fixture["cases"].as_array().expect("cases");
    assert_eq!(expected.len(), cases.len(), "fixture and prompt set differ");

    let gguf = open_model();
    let model = Model::load(&gguf).expect("load model");
    let threads = threads();
    let mut runner = Runner::new(&model, CONTEXT, threads);
    println!(
        "kernel tier {}, {threads} threads, context {CONTEXT}",
        runner.kernels().tier().as_str()
    );

    let mut failures = Vec::new();
    let (mut positions, mut agreed) = (0usize, 0usize);
    // How far every recorded margin has moved since the fixture was written.
    let mut moved: Vec<f32> = Vec::new();
    for (case, want) in cases.iter().zip(expected) {
        assert_eq!(
            want["name"].as_str(),
            Some(case.name.as_str()),
            "fixture case order changed"
        );
        let prompt = ids_of(&want["prompt_tokens"]);
        let reference = ids_of(&want["reference_tokens"]);
        let allowed = indeterminate_positions(want);
        assert_eq!(
            prompt,
            tokenizer().encode(&case.prompt, true).expect("encode"),
            "{}: the committed prompt tokens are not what the prompt renders to",
            case.name
        );

        runner.reset();
        let mut feed = prompt.clone();
        feed.extend_from_slice(&reference[..reference.len() - 1]);
        let mut tightest = f32::INFINITY;
        let mut disagreed = Vec::new();
        for (i, &token) in feed.iter().enumerate() {
            let logits = runner.forward(token);
            if i + 1 < prompt.len() {
                continue;
            }
            let at = i + 1 - prompt.len();
            let (ours, margin, runner_up) = top2(logits);
            positions += 1;
            tightest = tightest.min(margin);
            if ours == reference[at] {
                agreed += 1;
                continue;
            }
            disagreed.push((at, margin));
            if !allowed.iter().any(|&(p, _)| p == at) {
                failures.push(format!(
                    "{}: position {at} is not recorded as indeterminate, but we predict \
                     {ours} where the reference chose {}; margin {margin:.6}",
                    case.name, reference[at]
                ));
            } else if margin > INDETERMINATE {
                failures.push(format!(
                    "{}: position {at} disagrees with a margin of {margin:.6}, which is \
                     above the {INDETERMINATE} rounding band, so this is a real difference",
                    case.name
                ));
            } else if runner_up != reference[at] {
                failures.push(format!(
                    "{}: position {at} disagrees and the reference's token {} is not even \
                     our runner-up ({runner_up})",
                    case.name, reference[at]
                ));
            }
        }
        for &(at, _) in &allowed {
            if !disagreed.iter().any(|&(p, _)| p == at) {
                failures.push(format!(
                    "{}: position {at} is recorded as indeterminate but now agrees; \
                     regenerate the fixture",
                    case.name
                ));
            }
        }
        let margins: Vec<String> = disagreed
            .iter()
            .map(|&(at, margin)| {
                let was = allowed
                    .iter()
                    .find(|&&(p, _)| p == at)
                    .map_or(f32::NAN, |&(_, m)| m);
                moved.push(margin - was);
                format!("at {at} margin {margin:.5} (fixture {was:.5})")
            })
            .collect();
        println!(
            "{:28} {:3} positions, {} indeterminate, tightest margin {tightest:8.5}{}{}",
            case.name,
            reference.len(),
            allowed.len(),
            if margins.is_empty() { "" } else { ": " },
            margins.join(", "),
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {positions} next-token predictions are wrong:\n{}",
        failures.len(),
        failures.join("\n")
    );
    let widest = moved.iter().copied().fold(0.0f32, |m, d| m.max(d.abs()));
    println!(
        "{agreed} of {positions} next-token predictions identical to the reference, \
         the rest inside the rounding band and recorded in the fixture; the \
         recorded margins moved by at most {widest:.5}"
    );
}

/// The regression gate: free-running greedy generation, token for token
/// against the committed sequence.
#[test]
#[ignore = "needs the model file"]
fn greedy_output_is_unchanged() {
    let cases = cases();
    let fixture = fixture();
    let expected = fixture["cases"].as_array().expect("cases");
    assert_eq!(expected.len(), cases.len(), "fixture and prompt set differ");

    let gguf = open_model();
    let tokenizer = tokenizer();
    let model = Model::load(&gguf).expect("load model");
    let threads = threads();
    let mut runner = Runner::new(&model, CONTEXT, threads);
    println!(
        "kernel tier {}, {threads} threads, context {CONTEXT}",
        runner.kernels().tier().as_str()
    );

    let mut failures = Vec::new();
    let (mut like_reference, mut total) = (0usize, 0usize);
    for (case, want) in cases.iter().zip(expected) {
        let prompt = tokenizer.encode(&case.prompt, true).expect("encode");
        // Every case starts from a clean cache, which is what makes the result
        // independent of the order the cases run in.
        runner.reset();
        let (ids, stop, timings) = generate(
            &mut runner,
            &mut Sampler::greedy(model.vocab()),
            &prompt,
            case.max_tokens,
            |_| true,
        );
        let committed = ids_of(&want["tokens"]);
        let reference = ids_of(&want["reference_tokens"]);
        total += ids.len();
        if committed == reference {
            like_reference += 1;
        }
        println!(
            "{:28} {:3} tokens, {:?}, {:.2} tok/s, {}",
            case.name,
            ids.len(),
            stop,
            timings.generate_per_second(),
            if committed == reference {
                "identical to the reference".to_string()
            } else {
                format!(
                    "reference diverges at {}",
                    committed
                        .iter()
                        .zip(&reference)
                        .position(|(a, b)| a != b)
                        .map_or("?".to_string(), |p| p.to_string())
                )
            }
        );
        if ids != committed {
            let at = ids
                .iter()
                .zip(&committed)
                .position(|(a, b)| a != b)
                .unwrap_or(ids.len().min(committed.len()));
            failures.push(format!(
                "{}: diverges from the committed output at token {at}\n  ours: {:?}\n  want: {:?}\n  ours text: {:?}\n  want text: {:?}",
                case.name,
                &ids[at.min(ids.len())..],
                &committed[at.min(committed.len())..],
                tokenizer.decode(&ids, false).unwrap_or_default(),
                tokenizer.decode(&committed, false).unwrap_or_default(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} prompts changed:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
    println!(
        "{} prompts, {total} tokens unchanged; {like_reference} of {} are the reference's \
         own output for their whole length",
        cases.len(),
        cases.len()
    );
}

/// Generation must not depend on how the work was split over threads.
#[test]
#[ignore = "needs the model file"]
fn the_thread_count_does_not_change_the_output() {
    let case = &cases()[0];
    let gguf = open_model();
    let tokenizer = tokenizer();
    let model = Model::load(&gguf).expect("load model");
    let prompt = tokenizer.encode(&case.prompt, true).expect("encode");

    let mut outputs = Vec::new();
    for threads in [1usize, 3, threads()] {
        let mut runner = Runner::new(&model, CONTEXT, threads);
        let (ids, _, _) = generate(
            &mut runner,
            &mut Sampler::greedy(model.vocab()),
            &prompt,
            24,
            |_| true,
        );
        println!("{threads} threads: {ids:?}");
        outputs.push((threads, ids));
    }
    // Each matmul row and each head is computed by exactly one worker from
    // start to finish, so the split cannot change a single result.
    for (threads, ids) in &outputs[1..] {
        assert_eq!(
            *ids, outputs[0].1,
            "{threads} threads produced different tokens than 1"
        );
    }
}

/// The two ways the caches can be subtly wrong, both of which look like "the
/// first token is right and the rest drifts".
///
/// A reset runner has to be a fresh one: the delta-net state and the
/// convolution history are carried in the same buffers from one run to the
/// next, and a `reset` that left anything behind would carry a previous run
/// into this one.
///
/// The comparison is the whole logit vector, bit for bit, and not the token it
/// argmaxes to. Leftover state is not a small effect in the logits, but the
/// decay makes it a small effect in the *tokens*: the delta-net state is
/// multiplied by a factor below 1 at every step, so after a prompt's worth of
/// tokens a stale state has faded far below what it takes to move an argmax.
/// Measured: dropping the `ssm` buffer from `reset` leaves the generated tokens
/// identical and the first logits different, which is the one bug this test
/// exists to catch.
///
/// The second half, feeding the prompt and all but the last generated token as
/// one long prompt and requiring it to predict that last token, is close to
/// tautological *today*: reading a token and generating one go through the same
/// `forward`, so the prefilled path replays the same tokens through the same
/// positions. It is here because that is exactly the property a batched prompt
/// path would break, and because the hand-over is cheap to keep nailed down
/// while it is still free. What checks the hand-over against something outside
/// this runner is
/// [`every_next_token_matches_the_reference`], which feeds the reference's
/// tokens from the reference's prompt.
#[test]
#[ignore = "needs the model file"]
fn resetting_the_caches_is_the_same_as_a_fresh_runner() {
    let case = &cases()[0];
    let gguf = open_model();
    let tokenizer = tokenizer();
    let model = Model::load(&gguf).expect("load model");
    let prompt = tokenizer.encode(&case.prompt, true).expect("encode");
    let mut runner = Runner::new(&model, CONTEXT, threads());

    // The first token of a fresh run, which is the token that reads the most
    // state and the least context.
    let fresh_first = runner.forward(prompt[0]).to_vec();
    runner.reset();
    let fresh_prompt = {
        let mut logits = Vec::new();
        for &t in &prompt {
            logits = runner.forward(t).to_vec();
        }
        logits
    };
    runner.reset();
    let (first, _, _) = generate(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        24,
        |_| true,
    );

    runner.reset();
    if let Some(d) = difference(runner.forward(prompt[0]), &fresh_first) {
        panic!("after a reset, the first token's logits differ from a fresh runner's: {d}");
    }
    runner.reset();
    let mut reset_prompt = Vec::new();
    for &t in &prompt {
        reset_prompt = runner.forward(t).to_vec();
    }
    if let Some(d) = difference(&reset_prompt, &fresh_prompt) {
        panic!("after a reset, the prompt's last logits differ from a fresh runner's: {d}");
    }
    runner.reset();
    let (second, _, _) = generate(
        &mut runner,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        24,
        |_| true,
    );
    assert_eq!(
        first, second,
        "a reset runner generated different tokens than a fresh one:\n  \
         fresh: {first:?}\n  reset: {second:?}"
    );

    runner.reset();
    let mut last = 0;
    for &t in prompt.iter().chain(&first[..first.len() - 1]) {
        last = argmax(runner.forward(t));
    }
    assert_eq!(
        last,
        *first.last().expect("24 tokens"),
        "reading {} tokens as one prompt predicts {last}, but generating them one \
         at a time ended on {:?}; the state carried into generation is not the \
         state the prompt produced",
        prompt.len() + first.len() - 1,
        first.last()
    );
    println!(
        "{} tokens, identical after a reset and when prefilled",
        first.len()
    );
}

/// A batched prompt has to leave exactly the state one token at a time would
/// have left, and the comparison is **bit-exact**, not within a tolerance.
///
/// That is a property of the design rather than a hope: a batched matmul is the
/// same products accumulated in the same order (one weight row against one
/// activation row, the kernel test
/// `the_batched_kernel_is_bit_identical_to_the_single_one` pins that down), the
/// convolution and the recurrence run token by token inside the batch, and
/// attention stops its score loop at the token's own position. So any
/// difference at all is a bug, and a tolerance would hide the small ones:
/// rounding is what this model's output is discontinuous in, so a difference
/// too small to see in the logits is one flipped quantization level away from a
/// different token.
///
/// What is compared is the whole 248320-entry logit vector after the prompt,
/// *and* the tokens generated after it. The logits alone would miss a stale
/// recurrent state that only the next token reads; the tokens alone would miss
/// a wrong state that the decay fades below an argmax before it shows (see
/// [`resetting_the_caches_is_the_same_as_a_fresh_runner`]).
///
/// Batch sizes: one that divides the prompt, several that do not, and one
/// larger than the whole prompt. Batch 1 is the token-by-token side itself,
/// which is what makes this also the check that the batched path has not
/// changed the single-token path.
#[test]
#[ignore = "needs the model file"]
fn a_batched_prompt_is_the_same_as_one_token_at_a_time() {
    let case = &cases()[0];
    let gguf = open_model();
    let tokenizer = tokenizer();
    let model = Model::load(&gguf).expect("load model");
    let prompt = tokenizer.encode(&case.prompt, true).expect("encode");
    let threads = threads();

    // The token-by-token side: a runner that cannot batch at all.
    let mut single = Runner::with_batch(&model, CONTEXT, threads, 1);
    assert_eq!(single.max_batch(), 1);
    let mut want_logits = Vec::new();
    for &t in &prompt {
        want_logits = single.forward(t).to_vec();
    }
    single.reset();
    let (want_tokens, _, _) = generate(
        &mut single,
        &mut Sampler::greedy(model.vocab()),
        &prompt,
        16,
        |_| true,
    );
    println!(
        "{} prompt tokens, {} generated, tier {}, {threads} threads",
        prompt.len(),
        want_tokens.len(),
        single.kernels().tier().as_str(),
    );

    let sizes = [2usize, 5, 7, prompt.len(), prompt.len() + 9];
    // A prompt divisible by 2, 5 and 7 would be at least 70 tokens long and
    // no prompt in the set is, so both cases are always covered; asserting it
    // keeps that true if the prompt set changes.
    assert!(
        sizes.iter().any(|b| prompt.len().is_multiple_of(*b)),
        "no batch size divides a prompt of {} tokens",
        prompt.len()
    );
    assert!(
        sizes.iter().any(|b| !prompt.len().is_multiple_of(*b)),
        "every batch size divides a prompt of {} tokens",
        prompt.len()
    );

    for batch in sizes {
        let mut runner = Runner::with_batch(&model, CONTEXT, threads, batch);
        let mut read = 0;
        let mut got = Vec::new();
        let mut batches = 0;
        while read < prompt.len() {
            let take = (prompt.len() - read).min(batch);
            got = runner.forward_batch(&prompt[read..read + take]).to_vec();
            read += take;
            batches += 1;
        }
        if let Some(d) = difference(&got, &want_logits) {
            panic!(
                "batch {batch} ({batches} batches over {} tokens): the prompt's last \
                 logits differ from the token-by-token path: {d}",
                prompt.len()
            );
        }
        assert_eq!(runner.pos(), prompt.len(), "batch {batch}: wrong position");

        // The hand-over: generation reads the caches and the recurrent state
        // the prompt left, so this is where a state that is subtly wrong shows.
        runner.reset();
        let (ids, _, _) = generate(
            &mut runner,
            &mut Sampler::greedy(model.vocab()),
            &prompt,
            16,
            |_| true,
        );
        assert_eq!(
            ids, want_tokens,
            "batch {batch}: generation after a batched prompt produced different \
             tokens than after a token-by-token one"
        );
        println!(
            "batch {batch:3}: {batches} batch(es), logits identical, {} tokens identical",
            ids.len()
        );
    }
}

#[test]
#[ignore = "needs the model file and a local reference build"]
fn the_fixture_is_still_what_the_reference_produces() {
    let bin = match std::env::var("RUST_LLM_REFERENCE_BIN") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => panic!(
            "set RUST_LLM_REFERENCE_BIN to the directory holding llama-completion \
             (see docs/decisions.md for the build recipe)"
        ),
    };
    let cases = cases();
    let tokenizer = tokenizer();
    let gguf = open_model();
    let model = Model::load(&gguf).expect("load model");
    let write = std::env::var("RUST_LLM_WRITE_FIXTURE").is_ok();
    // `RUST_LLM_GATE_ONLY` is a substring of the case name, and it works for
    // writing as much as for checking: a case is merged into the fixture by
    // name, so the set can be regenerated one prompt per command.
    let only = std::env::var("RUST_LLM_GATE_ONLY").ok();
    let selected: Vec<&Case> = cases
        .iter()
        .filter(|c| only.as_ref().is_none_or(|want| c.name.contains(want)))
        .collect();
    assert!(
        !selected.is_empty(),
        "RUST_LLM_GATE_ONLY={} matched none of the {} cases",
        only.unwrap_or_default(),
        cases.len()
    );
    let by_piece = pieces_to_ids(&tokenizer, model.vocab());
    let threads = threads();
    let mut runner = Runner::new(&model, CONTEXT, threads);

    let mut out = Vec::new();
    for case in &selected {
        let (reference_prompt, pieces) = reference_run(&bin, &case.prompt, case.max_tokens);
        let prompt_ids = tokenizer.encode(&case.prompt, true).expect("encode");
        assert_eq!(
            prompt_ids, reference_prompt,
            "{}: the reference tokenizes the prompt differently",
            case.name
        );
        let reference: Vec<u32> = pieces
            .iter()
            .map(|p| {
                *by_piece.get(p).unwrap_or_else(|| {
                    panic!(
                        "{}: the reference printed a piece that maps to no single token: {p:?}",
                        case.name
                    )
                })
            })
            .collect();

        // Our own free-running output, and the positions where a next-token
        // prediction from the reference's context lands inside the rounding
        // band, both recorded so the two gates above need neither the
        // reference nor a second run.
        runner.reset();
        let (ours, _, _) = generate(
            &mut runner,
            &mut Sampler::greedy(model.vocab()),
            &prompt_ids,
            case.max_tokens,
            |_| true,
        );

        runner.reset();
        let mut feed = prompt_ids.clone();
        feed.extend_from_slice(&reference[..reference.len() - 1]);
        let mut indeterminate = Vec::new();
        for (i, &token) in feed.iter().enumerate() {
            let logits = runner.forward(token);
            if i + 1 < prompt_ids.len() {
                continue;
            }
            let at = i + 1 - prompt_ids.len();
            let (pick, margin, _) = top2(logits);
            if pick != reference[at] {
                indeterminate.push(serde_json::json!({ "at": at, "margin": margin }));
            }
        }
        println!(
            "{:28} {:4} prompt + {:3} reference tokens, {} indeterminate position(s), \
             free-running {}",
            case.name,
            prompt_ids.len(),
            reference.len(),
            indeterminate.len(),
            if ours == reference {
                "identical".to_string()
            } else {
                format!(
                    "diverges at {}",
                    ours.iter()
                        .zip(&reference)
                        .position(|(a, b)| a != b)
                        .map_or("?".to_string(), |p| p.to_string())
                )
            }
        );
        out.push(serde_json::json!({
            "name": case.name,
            "prompt": case.prompt,
            "prompt_tokens": prompt_ids,
            "reference_tokens": reference,
            "reference_text": tokenizer.decode(&reference, false).unwrap_or_default(),
            "tokens": ours,
            "text": tokenizer.decode(&ours, false).unwrap_or_default(),
            "indeterminate": indeterminate,
        }));
        if write {
            // After every case, not once at the end: a command that is stopped
            // part-way keeps what it produced, and the next one continues with
            // the cases that are still missing.
            merge_into_fixture(&cases, &out, &reference_description(&bin));
            println!("{:28} merged into the fixture", case.name);
        }
    }

    if write {
        return;
    }

    let committed = fixture();
    let committed_cases = committed["cases"].as_array().expect("cases");
    let fresh = &out[..];
    if only.is_none() {
        assert_eq!(committed_cases.len(), fresh.len(), "case count changed");
    }
    // Matched by name, not by position, so checking one prompt is the same
    // comparison as checking all of them.
    for is in fresh {
        let name = is["name"].as_str().unwrap_or("?");
        let was = committed_cases
            .iter()
            .find(|c| c["name"] == is["name"])
            .unwrap_or_else(|| panic!("{name} is not in the committed fixture"));
        assert_eq!(
            ids_of(&was["prompt_tokens"]),
            ids_of(&is["prompt_tokens"]),
            "{name}: the prompt changed"
        );
        assert_eq!(
            ids_of(&was["reference_tokens"]),
            ids_of(&is["reference_tokens"]),
            "{name}: the reference no longer produces the committed output"
        );
    }
    println!(
        "the fixture still matches the reference on {} of its {} prompts",
        fresh.len(),
        committed_cases.len()
    );
}

/// Writes the fixture with `fresh` merged into whatever is committed, matched
/// by name and ordered by the prompt set.
///
/// Merging rather than replacing is what allows one prompt per command. A case
/// that is in neither the fresh batch nor the existing file is left out, and the
/// gates then fail on the case count rather than silently running a short set.
///
/// The top-level `reference` always describes the build of the most recent
/// write. On a partial write against a different build that is worth saying out
/// loud, because the cases that were not regenerated came from the old one.
///
/// `model` and `model_sha256` are read off the file this ran against, not
/// written from a literal: the hash is the only field that ties the fixture to
/// a set of weights, and one that is typed in says nothing about its own
/// origin. A partial write against a different model file is refused outright,
/// because two models' cases in one fixture is exactly what that field exists
/// to rule out.
fn merge_into_fixture(all: &[Case], fresh: &[serde_json::Value], reference: &str) {
    let path = format!("{FIXTURES}/reference_generations.json");
    let existing: Option<serde_json::Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let old = existing
        .as_ref()
        .and_then(|d| d["cases"].as_array())
        .map_or(&[][..], |a| a.as_slice());
    if let Some(was) = existing.as_ref().and_then(|d| d["reference"].as_str())
        && was != reference
        && fresh.len() < all.len()
    {
        println!(
            "note: the {} case(s) already in the fixture were produced by\n  {was}\nand this \
             write records\n  {reference}\nso regenerate the rest before trusting the file",
            old.len()
        );
    }

    let path_of_model = model_path();
    let model_name = std::path::Path::new(&path_of_model)
        .file_name()
        .map_or_else(
            || path_of_model.clone(),
            |n| n.to_string_lossy().into_owned(),
        );
    let sha = sha256_of(&path_of_model);
    if let Some(was) = existing.as_ref().and_then(|d| d["model_sha256"].as_str()) {
        assert!(
            was == sha || fresh.len() == all.len(),
            "the fixture holds cases from a model with sha256 {was} and this run used \
             {sha}; regenerate every case, not one, or the file would describe two files"
        );
    }

    let cases: Vec<serde_json::Value> = all
        .iter()
        .filter_map(|case| {
            fresh
                .iter()
                .chain(old)
                .find(|c| c["name"].as_str() == Some(case.name.as_str()))
                .cloned()
        })
        .collect();
    let document = serde_json::json!({
        "comment": [
            "Generated by the ignored test the_fixture_is_still_what_the_reference_produces",
            "in crates/rust-llm/tests/gate.rs, from the prompts in gate_prompts.json.",
            "Regenerate one prompt at a time with RUST_LLM_GATE_ONLY=<name substring>.",
            "",
            "reference_tokens is llama.cpp's greedy output for the same prompt on the same",
            "GGUF file; tokens is this runner's. They are identical for most prompts. Where",
            "they are not, indeterminate lists the positions at which a next-token",
            "prediction made from the reference's own context differs, with the margin",
            "between the top two logits there. Those margins are inside the band that int8",
            "activation quantization leaves undetermined, which the gate's module comment",
            "explains; the reference disagrees with its own repacked kernels on the same",
            "prompts for the same reason."
        ],
        "model": model_name,
        "model_sha256": sha,
        "reference": reference,
        "cases": cases,
    });
    let rendered = serde_json::to_string_pretty(&document).expect("serialize") + "\n";
    std::fs::write(&path, &rendered).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Describes how two logit vectors differ, or `None` when they are identical.
///
/// A vocabulary of a quarter of a million logits cannot go in an assertion
/// message, so what comes back is the first index that differs, both values
/// there, and how many of them differ in total.
fn difference(a: &[f32], b: &[f32]) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("{} logits against {}", a.len(), b.len()));
    }
    let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let at = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some(format!(
        "{differing} of {} logits differ, the first at {at}: {} against {}",
        a.len(),
        a[at],
        b[at]
    ))
}

fn ids_of(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("token array")
        .iter()
        .map(|i| i.as_u64().expect("token id") as u32)
        .collect()
}

/// The positions the fixture records as indeterminate, with the margin that
/// was measured there when it was written.
///
/// The margin is recorded, and printed next to the one measured now, because
/// what matters about a disagreement is not that it is permitted but how far
/// inside the rounding band it is. A margin that has widened is a change in
/// the arithmetic even when the test still passes, and the only way to see
/// that is to put the two numbers next to each other.
fn indeterminate_positions(case: &serde_json::Value) -> Vec<(usize, f32)> {
    case["indeterminate"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| {
                    (
                        e["at"].as_u64().expect("at") as usize,
                        e["margin"].as_f64().unwrap_or(f64::NAN) as f32,
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The argmax, how far ahead of the runner-up it is, and which token that was.
fn top2(logits: &[f32]) -> (u32, f32, u32) {
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut second = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.0 {
            second = best;
            best = (v, i);
        } else if v > second.0 {
            second = (v, i);
        }
    }
    (best.1 as u32, best.0 - second.0, second.1 as u32)
}

/// Maps a token's printed text back to its id.
///
/// The reference never prints an id for a generated token, only its text, so
/// the whole vocabulary is decoded once to invert it. Collisions are asserted
/// away rather than assumed: this is a byte-level BPE vocabulary, so two
/// tokens with the same text would mean the inversion is not a function and
/// the comparison would be meaningless.
fn pieces_to_ids(tokenizer: &Tokenizer, vocab: usize) -> std::collections::HashMap<String, u32> {
    let mut map = std::collections::HashMap::with_capacity(vocab);
    let mut lossy = 0usize;
    for id in 0..vocab as u32 {
        let Ok(text) = tokenizer.decode(&[id], false) else {
            continue;
        };
        // A token that is half a multi-byte character cannot be inverted this
        // way; the replacement character is what the decoder leaves behind.
        if text.is_empty() || text.contains('\u{fffd}') {
            lossy += 1;
            continue;
        }
        assert!(
            map.insert(text.clone(), id).is_none(),
            "two tokens decode to {text:?}, so a printed piece is ambiguous"
        );
    }
    println!(
        "piece map: {} tokens, {lossy} not invertible on their own",
        map.len()
    );
    map
}

/// Runs the reference on one prompt and returns `(prompt token ids, generated
/// token pieces)`.
///
/// The reference prints each generated token's text as it goes, interleaved
/// with its own verbose log lines, so a token's bytes are exactly the text
/// between one log line and the `n_past` line that follows it. That is the
/// only way to get the reference's ids rather than its prose: it never prints
/// them for generated tokens. One extra token is asked for and dropped, so
/// every token the gate uses is bounded by an `n_past` line on both sides and
/// the newlines the reference prints on exit cannot be mistaken for part of
/// the last one.
fn reference_run(bin: &Path, prompt: &str, max_tokens: usize) -> (Vec<u32>, Vec<String>) {
    let dir = std::env::temp_dir().join(format!("rust-llm-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let prompt_file = dir.join("prompt.txt");
    // `--file` drops one trailing newline, and every chat prompt ends in one,
    // so an extra one goes in for it to eat. Without this the reference is
    // fed a prompt one token shorter than ours.
    std::fs::write(&prompt_file, format!("{prompt}\n")).expect("write prompt");

    // The parser needs the generated text and the verbose log interleaved as
    // the reference writes them, so both streams go to one file. Capturing
    // them as two pipes would lose exactly the ordering that marks the token
    // boundaries.
    let log_file = dir.join("reference.log");
    let log = std::fs::File::create(&log_file).expect("create reference log");
    let status = Command::new(bin.join("llama-completion"))
        .stdout(std::process::Stdio::from(log.try_clone().expect("dup log")))
        .stderr(std::process::Stdio::from(log))
        .env("LD_LIBRARY_PATH", bin)
        .args([
            "-m",
            &model_path(),
            "-f",
            prompt_file.to_str().expect("utf-8 path"),
            "--no-escape",
            "-n",
            &(max_tokens + 1).to_string(),
            "--temp",
            "0",
            "--top-k",
            "1",
            "--seed",
            "1",
            "-t",
            "8",
            "-c",
            "2048",
            "-no-cnv",
            "-sp",
            // An f32 KV cache and a plain softmax, which is what this runner
            // has. The reference defaults to an f16 cache and flash
            // attention: f16 keys are a different model, not a different
            // rounding.
            "-fa",
            "off",
            "-ctk",
            "f32",
            "-ctv",
            "f32",
            // The reference's repacked kernels are a different implementation
            // of the same arithmetic, with the quantization blocks reordered
            // and the f32 block scales accumulated in another order. Its plain
            // Q8_K path is the one this runner's kernels are written against.
            "--no-repack",
            "--verbose-prompt",
            "--no-display-prompt",
            "-v",
        ])
        .status()
        .expect("run llama-completion");
    let merged = std::fs::read(&log_file).expect("read reference log");
    assert!(
        status.success(),
        "llama-completion failed: {}",
        String::from_utf8_lossy(&merged)
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .join("\n")
    );
    parse_reference(&String::from_utf8_lossy(&merged), max_tokens)
}

/// Splits the reference's merged output into the prompt's token ids and the
/// pieces of the first `max_tokens` generated tokens.
fn parse_reference(merged: &str, max_tokens: usize) -> (Vec<u32>, Vec<String>) {
    let mut prompt = Vec::new();
    // `0.01.436.610 I   9419 -> 'Hello'`, one line per prompt token.
    for line in merged.lines() {
        if let Some(rest) = line.split(" I ").nth(1)
            && let Some((id, _)) = rest.trim().split_once(" -> ")
            && let Ok(id) = id.parse::<u32>()
        {
            prompt.push(id);
        }
    }
    assert!(
        !prompt.is_empty(),
        "no prompt tokens in the reference output; was --verbose-prompt passed?"
    );

    let mut pieces = Vec::new();
    let mut gap = String::new();
    let mut rest = merged;
    while let Some(start) = next_log_line(rest) {
        gap.push_str(&rest[..start]);
        let line_end = rest[start..]
            .find('\n')
            .map_or(rest.len(), |i| start + i + 1);
        let line = &rest[start..line_end];
        if line.contains(" D n_past = ") && !gap.is_empty() {
            // The prompt's own eval produces an empty gap.
            pieces.push(std::mem::take(&mut gap));
        }
        gap.clear();
        rest = &rest[line_end..];
    }
    assert!(
        pieces.len() >= max_tokens,
        "the reference produced {} tokens, expected at least {max_tokens}; \
         it may have stopped early",
        pieces.len()
    );
    pieces.truncate(max_tokens);
    (prompt, pieces)
}

/// Offset of the next verbose log line.
///
/// A log line starts with the reference's own timestamp, `h.mm.sss.uuu`,
/// followed by a one-letter level and a space. Generated token text sits
/// directly in front of such a timestamp on the same line, which is why this
/// scans for the pattern anywhere rather than at line starts.
///
/// The hour field is required to be a single digit. It has to be pinned down,
/// because a token whose text ends in a digit butts straight up against the
/// timestamp and `1` followed by `0.01.975.397` would otherwise parse as
/// `10.01.975.397` and eat the token. These runs last seconds, so the hour is
/// always `0`; a reference that ran for ten hours would have to be parsed
/// differently, and would fail loudly here as a short token count rather than
/// quietly.
fn next_log_line(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let field =
        |at: usize, n: usize| at + n <= b.len() && b[at..at + n].iter().all(u8::is_ascii_digit);
    (0..b.len()).find(|&i| {
        let mut j = i;
        if !field(j, 1) {
            return false;
        }
        j += 1;
        for n in [2usize, 3, 3] {
            if j >= b.len() || b[j] != b'.' || !field(j + 1, n) {
                return false;
            }
            j += 1 + n;
        }
        j + 2 < b.len() && b[j] == b' ' && b[j + 1].is_ascii_uppercase() && b[j + 2] == b' '
    })
}

/// The model file's sha256, from `sha256sum`.
///
/// A process rather than a crate: this is the only hash the workspace needs,
/// and it is needed by one ignored test that already shells out to the
/// reference runner. Adding a dependency for it would put it in the product's
/// lock file.
fn sha256_of(path: &str) -> String {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("run sha256sum: {e}"));
    assert!(
        out.status.success(),
        "sha256sum {path} failed: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let hash = text
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("no hash in sha256sum output: {text:?}"));
    assert_eq!(hash.len(), 64, "not a sha256: {hash:?}");
    hash.to_string()
}

fn reference_description(bin: &Path) -> String {
    let version = Command::new(bin.join("llama-completion"))
        .env("LD_LIBRARY_PATH", bin)
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .chain(String::from_utf8_lossy(&o.stdout).lines())
                .find(|l| l.contains("version"))
                .unwrap_or("unknown")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "llama.cpp {version}, llama-completion --temp 0 --top-k 1 --seed 1 -t 8 -c 2048 \
         -no-cnv -sp -fa off -ctk f32 -ctv f32 --no-repack, same GGUF file"
    )
}

#[cfg(test)]
mod parser {
    use super::{next_log_line, parse_reference};

    const SAMPLE: &str = "\
0.01.436.610 I   9419 -> 'Hello'
0.01.436.659 D embd_inp.size(): 1, n_consumed: 0
0.01.511.099 D n_past = 1
0.01.512.816 D n_remain: 1
,0.01.584.635 D n_past = 2
0.01.585.013 D n_remain: 0

0.01.585.100 D n_past = 3
0.01.585.200 I common_perf_print: done
";

    #[test]
    fn finds_log_lines_but_not_token_text() {
        assert_eq!(next_log_line("The0.01.731.242 D n_past = 20\n"), Some(3));
        assert_eq!(next_log_line("plain text\n"), None);
        // A number in generated text must not look like a timestamp.
        assert_eq!(next_log_line("in 2026.10.20 the"), None);
        // A token that is itself a digit must not be swallowed by the
        // timestamp that follows it.
        assert_eq!(next_log_line("10.01.975.397 D n_past = 22\n"), Some(1));
    }

    #[test]
    fn reads_prompt_ids_and_token_pieces() {
        let (prompt, pieces) = parse_reference(SAMPLE, 2);
        assert_eq!(prompt, vec![9419]);
        // The prompt's own eval has no piece; the second token is a newline.
        assert_eq!(pieces, vec![",".to_string(), "\n".to_string()]);
    }
}
