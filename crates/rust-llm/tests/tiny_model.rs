//! The command line, driven end to end against a tiny committed model.
//!
//! Needs no weights, so it runs in CI. What it proves is the surface and not
//! the output: the fixture's weights are noise, so the text is meaningless,
//! but it is produced by the real loader, the real kernels, the real chat
//! template and the real sampler, and it is identical on every run. See
//! `tests/support/mod.rs`.

use std::process::{Command, Stdio};

use llm_core::Gguf;
use rust_llm::chat::{ChatOptions, Message, render};
use rust_llm::forward::Runner;
use rust_llm::model::Model;
use rust_llm::sample::{Sampler, Sampling, argmax};
use rust_llm::{Tokenizer, generate};

mod support;

const BIN: &str = env!("CARGO_BIN_EXE_rust-llm");

/// What one invocation produced.
struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

impl Run {
    fn ok(&self, what: &str) -> &str {
        assert!(
            self.status.success(),
            "{what} failed: {}\n--- stderr ---\n{}",
            self.status,
            self.stderr
        );
        &self.stdout
    }
}

fn run(model: &std::path::Path, args: &[&str], stdin: &str) -> Run {
    let mut child = Command::new(BIN)
        .arg("--model")
        .arg(model)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rust-llm");
    use std::io::Write;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait for rust-llm");
    Run {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The fixture loads through the real loader and the real tokenizer check.
#[test]
fn the_fixture_is_a_model_this_runner_accepts() {
    let tiny = support::write("loads", support::UNREACHED_EOS);
    let gguf = Gguf::open(&tiny.model).expect("open the fixture");
    let tokenizer = Tokenizer::from_file(&tiny.tokenizer).expect("the fixture tokenizer");
    tokenizer
        .verify_against_weights(&gguf)
        .expect("the fixture's tokenizer and weights agree");
    let model = Model::load(&gguf).expect("load the fixture");

    assert_eq!(model.config.n_blocks, support::N_BLOCKS);
    assert_eq!(model.config.n_linear_blocks(), 1);
    assert_eq!(model.config.n_full_blocks(), 1);
    assert_eq!(model.vocab(), support::VOCAB);
    let bytes = std::fs::metadata(&tiny.model).expect("stat").len();
    assert!(
        bytes < 400_000,
        "the fixture is {bytes} bytes; it is supposed to stay a few hundred KB \
         so it costs nothing to write and load in every test that wants one"
    );

    // The quantized layouts the writer produces have to be the ones the reader
    // expects, which a surface test would otherwise never notice: a mis-packed
    // Q6_K super-block is still a valid file full of different numbers.
    let embd = gguf.tensor("token_embd.weight").expect("token_embd");
    let mut row = vec![0.0f32; support::N_EMBD];
    let stride = embd.ty.bytes_for(support::N_EMBD as u64).expect("row size") as usize;
    llm_core::dequantize_row(embd.ty, &embd.data[..stride], &mut row);
    let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        (0.1..=0.25).contains(&max),
        "the embedding row came back with a largest magnitude of {max}, so the \
         Q6_K block the writer produced is not the one the reader reads"
    );
    assert!(
        row.iter().any(|&v| v != row[0]),
        "the whole embedding row is one value"
    );
}

/// `run` loads, streams and stops on `--max-tokens`.
///
/// Greedy, like every run below that asserts on the stop reason: the fixture's
/// end-of-turn tokens are inside the vocabulary it can produce, so a draw
/// reaches one now and then and the run would stop for the other reason.
#[test]
fn run_streams_and_stops_on_max_tokens() {
    let tiny = support::write("run", support::UNREACHED_EOS);
    let out = run(
        &tiny.model,
        &[
            "run",
            "--prompt",
            "hello",
            "--max-tokens",
            "8",
            "--context",
            "128",
            "--temperature",
            "0",
        ],
        "",
    );
    let text = out.ok("run").to_string();
    assert!(!text.trim().is_empty(), "nothing was streamed");
    assert!(
        out.stderr.contains("stopped: MaxTokens"),
        "expected to stop on --max-tokens:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("generate 8 tokens"),
        "expected all 8 tokens:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("2 blocks, 1 full attention, 1 linear"),
        "the header did not describe the fixture:\n{}",
        out.stderr
    );
}

/// The token a greedy run produces first, so a fixture can be built that stops
/// on exactly that token.
fn first_greedy_token(tiny: &support::Tiny, prompt: &str) -> u32 {
    let gguf = Gguf::open(&tiny.model).expect("open the fixture");
    let tokenizer = Tokenizer::from_file(&tiny.tokenizer).expect("the fixture tokenizer");
    let model = Model::load(&gguf).expect("load the fixture");
    let mut runner = Runner::new(&model, 128, 1);
    let rendered = render(
        &[Message::user(prompt)],
        ChatOptions {
            add_generation_prompt: true,
            thinking: true,
        },
    )
    .expect("render");
    let tokens = tokenizer.encode(&rendered, true).expect("encode");
    let mut last = 0;
    for &t in &tokens {
        last = argmax(runner.forward(t));
    }
    last
}

/// Generation stops on a token from the model's end-of-turn set, and the token
/// is not printed.
#[test]
fn run_stops_on_end_of_sequence() {
    let probe = support::write("eos-probe", support::UNREACHED_EOS);
    let first = first_greedy_token(&probe, "hello");
    assert_ne!(
        first,
        support::UNREACHED_EOS,
        "the probe fixture already stops on its own first token"
    );

    // The same weights, with the token this model produces first named as the
    // end of a turn, so the run has to stop before printing anything.
    let tiny = support::write("eos", first);
    let out = run(
        &tiny.model,
        &[
            "run",
            "--prompt",
            "hello",
            "--max-tokens",
            "8",
            "--context",
            "128",
            // Greedy, because `first` is the argmax: a draw would sometimes
            // pick something else and the run would not stop here at all.
            "--temperature",
            "0",
        ],
        "",
    );
    let text = out.ok("run");
    assert_eq!(text, "\n", "an immediate stop printed {text:?}");
    assert!(
        out.stderr.contains(&format!("stopped: Eos({first})")),
        "expected to stop on the end-of-sequence token {first}:\n{}",
        out.stderr
    );
}

/// The same seed gives the same text twice and a different seed does not.
#[test]
fn sampling_is_reproducible_from_the_seed() {
    let tiny = support::write("seed", support::UNREACHED_EOS);
    let sampled = |seed: &str| {
        let out = run(
            &tiny.model,
            &[
                "run",
                "--prompt",
                "hello",
                "--max-tokens",
                "24",
                "--context",
                "128",
                "--temperature",
                "1.0",
                "--top-k",
                "40",
                "--top-p",
                "0.9",
                "--min-p",
                "0.02",
                "--repeat-penalty",
                "1.1",
                "--repeat-last-n",
                "16",
                "--seed",
                seed,
            ],
            "",
        );
        out.ok("a sampled run").to_string()
    };
    let a = sampled("1234");
    assert_eq!(a, sampled("1234"), "the same seed gave different text");
    assert_ne!(a, sampled("5678"), "two seeds gave the same text");

    // Greedy is the other half of the same claim: with no seed at all it still
    // produces the same text, because there is no draw.
    let greedy = |_: ()| {
        run(
            &tiny.model,
            &[
                "run",
                "--prompt",
                "hello",
                "--max-tokens",
                "24",
                "--context",
                "128",
                "--temperature",
                "0",
            ],
            "",
        )
        .ok("a greedy run")
        .to_string()
    };
    assert_eq!(greedy(()), greedy(()));
    assert_ne!(greedy(()), a, "sampling produced the greedy output");
}

/// Without `--seed` the effective seed is printed, and feeding it back
/// reproduces the run. Without that, a run taken from entropy could not be
/// replayed at all.
#[test]
fn the_seed_taken_from_entropy_is_printed_and_replays() {
    let tiny = support::write("entropy-seed", support::UNREACHED_EOS);
    let args = [
        "run",
        "--prompt",
        "hello",
        "--max-tokens",
        "16",
        "--context",
        "128",
        "--temperature",
        "1.0",
    ];
    let out = run(&tiny.model, &args, "");
    let text = out.ok("a run with no seed").to_string();
    let seed = printed_seed(&out);

    let mut replay: Vec<&str> = args.to_vec();
    replay.extend(["--seed", &seed]);
    let again = run(&tiny.model, &replay, "");
    assert_eq!(
        text,
        again.ok("the replay"),
        "replaying with the printed seed {seed} gave different text"
    );

    // Two runs from entropy pick different seeds, or the seed is not random.
    // It has to be the seed and not the whole of stderr: that carries the
    // timings, so it differs between any two runs and would pass with a
    // hard-coded seed, which is the one thing this is here to catch.
    let other = run(&tiny.model, &args, "");
    other.ok("a second run with no seed");
    assert_ne!(
        seed,
        printed_seed(&other),
        "two runs from entropy used the same seed"
    );
}

/// The effective seed off the `sampling:` line, which is the only way to replay
/// a run that took its seed from entropy.
fn printed_seed(out: &Run) -> String {
    out.stderr
        .lines()
        .find_map(|l| l.strip_prefix("sampling: ")?.rsplit_once("seed "))
        .map(|(_, s)| s.to_string())
        .unwrap_or_else(|| panic!("no effective seed on stderr:\n{}", out.stderr))
}

/// Every sampling flag reaches the sampler, and a value that cannot be a
/// setting is an error that names the argument.
#[test]
fn the_sampling_flags_are_honoured_and_checked() {
    let tiny = support::write("flags", support::UNREACHED_EOS);

    // top-k 1 is greedy by another route: one candidate always carries all the
    // mass, whatever the seed.
    let top_k_1 = |seed: &str| {
        run(
            &tiny.model,
            &[
                "run",
                "--prompt",
                "hello",
                "--max-tokens",
                "12",
                "--context",
                "128",
                "--temperature",
                "1.0",
                "--top-k",
                "1",
                "--seed",
                seed,
            ],
            "",
        )
        .ok("top-k 1")
        .to_string()
    };
    assert_eq!(
        top_k_1("1"),
        top_k_1("2"),
        "with one candidate the seed should not matter"
    );

    for (args, wanted) in [
        (vec!["--temperature", "-1"], "--temperature"),
        (vec!["--top-p", "1.5"], "--top-p"),
        (vec!["--top-p", "-0.1"], "--top-p"),
        (vec!["--min-p", "2"], "--min-p"),
        (vec!["--repeat-penalty", "0"], "--repeat-penalty"),
        (vec!["--top-k", "-3"], "--top-k"),
        (vec!["--seed", "banana"], "--seed"),
        (vec!["--temperature", "nonsense"], "--temperature"),
        (vec!["--speculate", "17"], "--speculate"),
        (vec!["--speculate", "-1"], "--speculate"),
        (vec!["--speculate", "some"], "--speculate"),
        (vec!["--draft-source", "head"], "draft source"),
        (vec!["--draft-source", ""], "draft source"),
    ] {
        let mut full = vec!["run", "--prompt", "hello"];
        full.extend(&args);
        let out = run(&tiny.model, &full, "");
        assert!(
            !out.status.success(),
            "{args:?} was accepted:\n{}",
            out.stderr
        );
        assert!(
            out.stderr.contains(wanted),
            "{args:?} failed without naming {wanted}:\n{}",
            out.stderr
        );
    }
}

/// `--speculate` changes the speed and nothing else, which on the command line
/// means the streamed text has to be the same at every draft length.
///
/// Greedy, so the answer is a function of the prompt alone and the three runs
/// can be compared directly. The fixture's weights are noise, so almost no
/// draft is accepted and what this really covers is the surface: the flag
/// parses, the runner allocates its ring, the drafter runs over a real history
/// and the accept loop produces the same stream. Where a draft *is* accepted is
/// `tests/speculate.rs`, and on real weights the gate.
#[test]
fn the_speculate_flag_changes_nothing_but_the_speed() {
    let tiny = support::write("speculate-cli", support::UNREACHED_EOS);
    let answer = |draft: &str| {
        let out = run(
            &tiny.model,
            &[
                "run",
                "--prompt",
                "hello hello hello hello",
                "--max-tokens",
                "16",
                "--context",
                "128",
                "--temperature",
                "0",
                "--speculate",
                draft,
            ],
            "",
        );
        let text = out.ok(&format!("--speculate {draft}")).to_string();
        (text, out.stderr)
    };
    let (want, off) = answer("0");
    assert!(
        !off.contains("speculation:"),
        "--speculate 0 reported speculation it did not do:\n{off}"
    );
    for draft in ["1", "4"] {
        let (got, log) = answer(draft);
        assert_eq!(got, want, "--speculate {draft} streamed different text");
        assert!(
            log.contains(&format!("speculation: {draft} drafted tokens per step")),
            "--speculate {draft} did not say what it was doing:\n{log}"
        );
        assert!(
            log.contains("drafts accepted"),
            "--speculate {draft} printed no acceptance line:\n{log}"
        );
    }
}

/// `--draft-source` picks where the drafts come from, and either source
/// streams the same text.
///
/// The parsing and the refusal are the point. `mtp` on a file without the
/// block has to fail by name rather than quietly falling back to prompt
/// lookup, because the two differ only in speed and a silent fallback would
/// read as a feature that does nothing. On a file that has the block, the
/// streamed answer has to be the one `--speculate 0` gives, which is the
/// command-line end of the equivalence `tests/mtp.rs` asserts in the library.
#[test]
fn the_draft_source_flag_changes_nothing_but_the_speed() {
    let with_mtp = support::write("draft-source-cli", support::UNREACHED_EOS);
    let without = support::write_without_mtp("draft-source-cli-plain", support::UNREACHED_EOS);
    let answer = |model: &std::path::Path, args: &[&str]| {
        let mut full = vec![
            "run",
            "--prompt",
            "hello hello hello hello",
            "--max-tokens",
            "16",
            "--context",
            "128",
            "--temperature",
            "0",
        ];
        full.extend(args);
        let out = run(model, &full, "");
        (out.status.success(), out.stdout, out.stderr)
    };

    let (ok, want, _) = answer(&with_mtp.model, &["--speculate", "0"]);
    assert!(ok, "the baseline run failed");
    for source in ["prompt-lookup", "mtp"] {
        let (ok, got, log) = answer(
            &with_mtp.model,
            &["--speculate", "4", "--draft-source", source],
        );
        assert!(ok, "--draft-source {source} failed:
{log}");
        assert_eq!(got, want, "--draft-source {source} streamed different text");
        assert!(
            log.contains(&format!("4 drafted tokens per step from {source}")),
            "--draft-source {source} did not say which source it used:
{log}"
        );
    }
    // The file the runner started on: it loads and runs, and the flag that
    // needs a head it does not have is refused by name.
    let (ok, _, log) = answer(&without.model, &["--speculate", "4"]);
    assert!(ok, "a file without the block failed to run at all:
{log}");
    assert!(
        log.contains("no MTP block"),
        "a file without the block did not say so:
{log}"
    );
    let (ok, _, log) = answer(
        &without.model,
        &["--speculate", "4", "--draft-source", "mtp"],
    );
    assert!(!ok, "--draft-source mtp was accepted on a file without one");
    assert!(
        log.contains("nextn.eh_proj"),
        "the refusal did not name what is missing:
{log}"
    );
}

/// `chat` keeps the conversation, streams every turn, and handles `/reset`,
/// `/exit` and the end of its input.
///
/// Greedy, so every turn's answer is a function of the conversation in front
/// of it and the three runs below can be compared against each other directly
/// rather than counted.
#[test]
fn chat_holds_a_conversation() {
    let tiny = support::write("chat", support::UNREACHED_EOS);
    let args = [
        "chat",
        "--max-tokens",
        "6",
        "--context",
        "256",
        "--temperature",
        "0",
    ];
    let session = |input: &str| {
        let out = run(&tiny.model, &args, input);
        out.ok("chat").to_string()
    };

    let one = session("one\n/exit\n");
    assert!(!one.trim().is_empty(), "the first turn streamed nothing");

    let two = session("one\ntwo\n/exit\n");
    assert!(
        two.starts_with(&one),
        "the first turn of a two-turn session differs from a one-turn session:\n  \
         one turn:  {one:?}\n  two turns: {two:?}"
    );
    assert_ne!(
        two, one,
        "the second turn answered with nothing, or with the first turn again"
    );

    // After /reset the third turn sees exactly what the first one saw, so it
    // has to answer exactly the same.
    let reset = session("one\ntwo\n/reset\none\n/exit\n");
    assert_eq!(
        reset,
        format!("{two}{one}"),
        "/reset did not put the conversation back to empty"
    );

    // The end of stdin is as clean an exit as /exit, and gets the same answer.
    assert_eq!(
        session("one\n"),
        one,
        "Ctrl-D did not end the session cleanly"
    );

    // A turn that is only whitespace is not a turn.
    assert_eq!(
        session("\n   \none\n/exit\n"),
        one,
        "a blank line was a turn"
    );
}

/// A full context is reported rather than generated into.
#[test]
fn a_full_context_is_reported() {
    let tiny = support::write("context", support::UNREACHED_EOS);
    // The prompt alone is longer than this, because the fixture's tokenizer
    // has no merges and the chat template is tokenized byte by byte.
    let out = run(
        &tiny.model,
        &[
            "chat",
            "--max-tokens",
            "4",
            "--context",
            "8",
            "--temperature",
            "0",
        ],
        "hello\n",
    );
    out.ok("chat with a tiny context");
    assert!(
        out.stderr.contains("the context holds 8"),
        "a conversation that does not fit was not reported:\n{}",
        out.stderr
    );
}

/// `--system` and the thinking toggle change the prompt on both surfaces,
/// which is visible in the token count the header prints and in the output.
#[test]
fn the_system_prompt_and_the_thinking_toggle_reach_the_model() {
    let tiny = support::write("prompt-shape", support::UNREACHED_EOS);
    let base = [
        "run",
        "--prompt",
        "hello",
        "--max-tokens",
        "8",
        "--context",
        "512",
        "--temperature",
        "0",
    ];
    let prompt_tokens = |out: &Run| -> usize {
        out.stderr
            .lines()
            .find_map(|l| l.strip_prefix("loaded in ")?.split_once("prompt is "))
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no prompt token count on stderr:\n{}", out.stderr))
    };

    let plain = run(&tiny.model, &base, "");
    plain.ok("a plain run");

    let mut with_system: Vec<&str> = base.to_vec();
    with_system.extend(["--system", "you are a teapot"]);
    let system = run(&tiny.model, &with_system, "");
    system.ok("a run with a system prompt");
    assert!(
        prompt_tokens(&system) > prompt_tokens(&plain),
        "--system did not lengthen the prompt"
    );

    let mut no_think: Vec<&str> = base.to_vec();
    no_think.push("--no-think");
    let unthinking = run(&tiny.model, &no_think, "");
    unthinking.ok("a run with thinking off");
    assert!(
        prompt_tokens(&unthinking) > prompt_tokens(&plain),
        "--no-think did not change the prompt: with thinking off the template \
         closes the think block itself, which is more tokens, not fewer"
    );

    // And on chat, where the same two flags go through a different path.
    let chat = [
        "chat",
        "--max-tokens",
        "4",
        "--context",
        "512",
        "--temperature",
        "0",
    ];
    let plain_chat = run(&tiny.model, &chat, "hello\n/exit\n")
        .ok("chat with neither flag")
        .to_string();
    for extra in [vec!["--system", "you are a teapot"], vec!["--no-think"]] {
        let mut full: Vec<&str> = chat.to_vec();
        full.extend(&extra);
        let out = run(&tiny.model, &full, "hello\n/exit\n");
        assert_ne!(
            out.ok("chat with the flag"),
            plain_chat,
            "{extra:?} did not reach chat"
        );
    }
}

/// The library's own generation loop against the fixture, so the sampler is
/// exercised over a real logit vector from a real forward pass and not only
/// over synthetic ones.
#[test]
fn the_generation_loop_runs_on_the_fixture() {
    let tiny = support::write("library", support::UNREACHED_EOS);
    let gguf = Gguf::open(&tiny.model).expect("open the fixture");
    let tokenizer = Tokenizer::from_file(&tiny.tokenizer).expect("the fixture tokenizer");
    let model = Model::load(&gguf).expect("load the fixture");
    let mut runner = Runner::new(&model, 128, 2);
    let prompt = tokenizer.encode("hello", true).expect("encode");

    for set in [
        Sampling::GREEDY,
        Sampling::default(),
        Sampling {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.8,
            min_p: 0.05,
            repeat_penalty: 1.4,
            repeat_last_n: 8,
            seed: 3,
        },
    ] {
        runner.reset();
        let mut sampler = Sampler::new(model.vocab(), set);
        let (ids, stop, timings) =
            generate::generate(&mut runner, &mut sampler, &prompt, 16, |_| true);
        assert!(
            !ids.is_empty(),
            "{set:?} generated nothing, stopped {stop:?}"
        );
        assert_eq!(timings.prompt_tokens, prompt.len());
        assert!(
            ids.iter().all(|&id| (id as usize) < model.vocab()),
            "{set:?} produced a token outside the vocabulary: {ids:?}"
        );
        // Decoding has to work on whatever came out, which is what a streaming
        // caller does per token.
        for &id in &ids {
            tokenizer.decode(&[id], false).expect("decode");
        }
    }
}
