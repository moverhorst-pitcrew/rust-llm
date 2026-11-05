//! `rust-llm` — the command line around the runner.
//!
//! Arguments are parsed by hand. They are all `--name value`, there is no
//! subcommand-specific grammar, and a parser crate would be a dependency and a
//! build-time cost for something this shape.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use rust_llm::chat::{ChatOptions, Message};
use rust_llm::forward::Runner;
use rust_llm::model::Model;
use rust_llm::sample::{self, Sampler, Sampling, entropy_seed};
use rust_llm::{Stop, Tokenizer, bench, default_threads, generate, peak_rss_kib};

use llm_core::{Gguf, Kernels, Tier};

const USAGE: &str = "\
rust-llm run --prompt <text> [options]
rust-llm chat [options]
rust-llm serve [options]
rust-llm bench [options]
rust-llm profile [options]
rust-llm trace --prompt <text> [options]

  run     generate from the prompt and stream the result
  chat    an interactive conversation, streamed. /reset starts over,
          /exit or Ctrl-D quits. No line editing and no history: that
          would be a readline dependency for a development surface.
  serve   an OpenAI-compatible HTTP server: POST /v1/chat/completions
          and POST /v1/completions, streaming and not. One request is
          answered at a time; the options below are its defaults.
  bench   prompt and generation tok/s, median of repeated runs, per thread
          count (see docs/benchmarks.md)
  profile where the time goes per operator, one table for the prompt phase
          and one for generation (see docs/benchmarks.md)
  trace   print one per-layer activation sum per line, for comparison
          against a reference runner (see docs/decisions.md)

options
  --model <path>        GGUF file; required, or set $RUST_LLM_MODEL
  --tokenizer <path>    tokenizer.json (default: next to the model)
  --prompt <text>       the user message (required by run and trace)
  --system <text>       a system message, placed first
  --context <n>         positions to allocate for the KV cache (default 4096)
  --threads <n>[,<n>..] worker threads; bench takes a list and runs each
                        (default: all cores)
  --max-tokens <n>      stop after this many generated tokens (default 512,
                        bench 16, and bench generates exactly this many)
  --batch <n>           tokens per batch while reading the prompt (default:
                        see docs/decisions.md; 1 reads it token by token)
  --speculate <n>       draft n tokens per step from the text so far and verify
                        them in one batch; 0 is off (default and why: see
                        docs/benchmarks.md). Costs about 50 MiB per drafted
                        token and never changes what comes out.
  --tier <name>         pin the kernel tier (scalar, avx2+fma, avx2+vnni)
                        instead of taking the strongest the CPU reports
  --think, --no-think   thinking mode, on by default, as the template is
  --raw                 take --prompt as the prompt verbatim, no chat template
  --host <addr>         serve: address to listen on (default 127.0.0.1)
  --port <n>            serve: port to listen on, 0 picks a free one
                        (default 8080)

sampling
  --temperature <f>     0 is greedy argmax (default 0.6)
  --top-k <n>           keep the n highest logits, 0 for all (default 20)
  --top-p <f>           keep the smallest set covering this mass (default 0.95)
  --min-p <f>           drop below this fraction of the top probability
                        (default 0, off)
  --repeat-penalty <f>  1 is off (default 1)
  --repeat-last-n <n>   tokens the penalty looks back over (default 64)
  --seed <n>            default: from entropy, printed to stderr

bench and profile
  --prompt-tokens <n>   prompt length, taken from a fixed text (default 19)

bench only
  --repeats <n>         timed runs per thread count, after one warm-up that
                        is not counted (default 5)
";

struct Args {
    command: String,
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    prompt: String,
    system: Option<String>,
    context: usize,
    /// One for every command but `bench`, which runs them all.
    threads: Vec<usize>,
    /// `None` is the command's own default: 512 for `run`, 16 for `bench`.
    max_tokens: Option<usize>,
    batch: usize,
    speculate: usize,
    tier: Option<String>,
    prompt_tokens: usize,
    repeats: usize,
    thinking: bool,
    raw: bool,
    sampling: Sampling,
    seed: Option<u64>,
    host: String,
    port: u16,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rust-llm: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse(std::env::args().skip(1))? {
        Some(args) => args,
        None => {
            print!("{USAGE}");
            return Ok(());
        }
    };

    let tokenizer_path = args.tokenizer.clone().unwrap_or_else(|| {
        args.model
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("tokenizer.json")
    });

    let load = std::time::Instant::now();
    let gguf = Gguf::open(&args.model)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    tokenizer.verify_against_weights(&gguf)?;
    let model = match &args.tier {
        None => Model::load(&gguf)?,
        Some(name) => {
            let tier = Tier::parse(name).ok_or_else(|| {
                format!(
                    "unknown --tier {name:?}; one of {}",
                    Tier::ALL.map(|t| t.as_str()).join(", ")
                )
            })?;
            let kernels = Kernels::for_tier(tier)
                .ok_or_else(|| format!("this CPU cannot run the {name} kernels"))?;
            Model::with_kernels(&gguf, kernels)?
        }
    };
    let load = load.elapsed();

    // Before the runner: each of these builds its own, one per thread count.
    if args.command == "bench" {
        return bench(&model, &tokenizer, &args, load);
    }
    if args.command == "profile" {
        return profile(&model, &tokenizer, &args);
    }

    let mut runner = Runner::with_speculation(
        &model,
        args.context,
        args.threads[0],
        args.batch,
        args.speculate,
    );

    eprintln!(
        "model {} ({} blocks, {} full attention, {} linear), kernels {}, {} threads, context {}, prompt batch {}",
        args.model.display(),
        model.config.n_blocks,
        model.config.n_full_blocks(),
        model.config.n_linear_blocks(),
        runner.kernels().tier().as_str(),
        runner.threads(),
        runner.context(),
        runner.max_batch(),
    );
    if runner.draft() > 0 {
        eprintln!(
            "speculation: {} drafted tokens per step, verified in one batch of {}",
            runner.draft(),
            runner.draft() + 1,
        );
    }

    if args.command == "trace" {
        let tokens = encode_prompt(&args, &tokenizer)?;
        eprintln!(
            "loaded in {:.2}s, prompt is {} tokens",
            load.as_secs_f64(),
            tokens.len()
        );
        runner.tracer.enable();
        for &t in &tokens {
            runner.tracer.clear();
            runner.forward(t);
        }
        let mut out = std::io::stdout().lock();
        for node in runner.tracer.nodes() {
            let edges: Vec<String> = node.edges.iter().map(|v| format!("{v:.6}")).collect();
            writeln!(out, "{}\t{:.6}\t{}", node.name, node.sum, edges.join(","))?;
        }
        return Ok(());
    }

    // Before the seed is resolved below: the server resolves one per request,
    // so that every answer it gives can be replayed on its own.
    if args.command == "serve" {
        return serve(&args, &tokenizer, model.vocab(), &mut runner, load);
    }

    // The seed is resolved here and printed, because a run taken from entropy
    // is only reproducible if the number it used is on the screen.
    let sampling = Sampling {
        seed: args.seed.unwrap_or_else(entropy_seed),
        ..args.sampling
    };
    let mut sampler = Sampler::new(model.vocab(), sampling);
    if sampling.temperature > 0.0 {
        eprintln!(
            "sampling: temperature {}, top-k {}, top-p {}, min-p {}, repeat penalty {} over {}, seed {}",
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.repeat_last_n,
            sampling.seed,
        );
    } else {
        eprintln!("sampling: greedy (temperature 0)");
    }

    if args.command == "chat" {
        return chat(&args, &tokenizer, &mut runner, &mut sampler, load);
    }

    let tokens = encode_prompt(&args, &tokenizer)?;
    eprintln!(
        "loaded in {:.2}s, prompt is {} tokens",
        load.as_secs_f64(),
        tokens.len()
    );

    let (_, stop, timings) = stream(&mut runner, &mut sampler, &tokenizer, &tokens, &args)?;

    eprintln!(
        "stopped: {stop:?}\nprompt   {} tokens in {:.2}s = {:.2} tok/s\ngenerate {} tokens in {:.2}s = {:.2} tok/s",
        timings.prompt_tokens,
        timings.prompt_time.as_secs_f64(),
        timings.prompt_per_second(),
        timings.generated_tokens,
        timings.generate_time.as_secs_f64(),
        timings.generate_per_second(),
    );
    // Printed for any prompt, because what a draft source is worth depends on
    // the text: this is how to see whether a prompt of your own is one
    // speculation helps with.
    if args.speculate > 0 {
        eprintln!("speculation: {}", timings.speculation.report());
    }
    if let Some(peak) = peak_rss_kib() {
        eprintln!(
            "peak RSS {peak} KiB ({:.2} GiB)",
            peak as f64 / (1 << 20) as f64
        );
    }
    Ok(())
}

/// The prompt as the model sees it: the chat template around `--system` and
/// `--prompt`, or `--prompt` verbatim with `--raw`.
fn encode_prompt(
    args: &Args,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let prompt = if args.raw {
        args.prompt.clone()
    } else {
        let mut messages = Vec::new();
        if let Some(system) = &args.system {
            messages.push(Message::system(system.clone()));
        }
        messages.push(Message::user(args.prompt.clone()));
        render(&messages, args.thinking)?
    };
    let tokens = tokenizer.encode(&prompt, true)?;
    // Only reachable with --raw: the chat template always emits its own
    // tokens, so an empty --prompt still renders to something.
    if tokens.is_empty() {
        return Err("the prompt is empty, so there is nothing to run".into());
    }
    Ok(tokens)
}

fn render(messages: &[Message], thinking: bool) -> Result<String, rust_llm::ChatError> {
    rust_llm::chat::render(
        messages,
        ChatOptions {
            add_generation_prompt: true,
            thinking,
        },
    )
}

/// Generates from `tokens`, streaming each token's text to stdout as it comes
/// out, and returns what was generated.
///
/// A character spread over two tokens cannot be decoded from either of them
/// alone, and the decoder leaves a replacement character where it was. That is
/// a property of what was *streamed*, so it is recorded here rather than looked
/// for in the clean text afterwards, which cannot contain one.
fn stream(
    runner: &mut Runner<'_>,
    sampler: &mut Sampler,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    args: &Args,
) -> Result<(Vec<u32>, Stop, generate::Timings), Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout();
    let mut streamed_a_broken_character = false;
    let max_tokens = args.max_tokens.unwrap_or(512);
    let (ids, stop, timings) = generate::generate(runner, sampler, tokens, max_tokens, |id| {
        if let Ok(text) = tokenizer.decode(&[id], false) {
            streamed_a_broken_character |= text.contains(char::REPLACEMENT_CHARACTER);
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        }
        true
    });
    println!();

    if streamed_a_broken_character {
        let text = tokenizer.decode(&ids, false)?;
        eprintln!("--- decoded as a whole ---");
        println!("{text}");
    }
    Ok((ids, stop, timings))
}

/// The HTTP surface.
///
/// Every option above becomes a per-request default; a request overrides the
/// fields it sends and keeps the rest. See `rust_llm::serve` and
/// `docs/decisions.md`.
fn serve(
    args: &Args,
    tokenizer: &Tokenizer,
    vocab: usize,
    runner: &mut Runner<'_>,
    load: std::time::Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = std::net::TcpListener::bind((args.host.as_str(), args.port))
        .map_err(|e| format!("cannot listen on {}:{}: {e}", args.host, args.port))?;
    let defaults = rust_llm::serve::Defaults {
        model: args
            .model
            .file_name()
            .map_or_else(
                || args.model.as_os_str().to_string_lossy(),
                |name| name.to_string_lossy(),
            )
            .into_owned(),
        system: args.system.clone(),
        max_tokens: args.max_tokens.unwrap_or(512),
        thinking: args.thinking,
        sampling: args.sampling,
        seed: args.seed,
        speculate: args.speculate,
    };
    eprintln!(
        "loaded in {:.2}s. defaults: max-tokens {}, thinking {}, temperature {}, \
         top-k {}, top-p {}, min-p {}, repeat penalty {} over {}, seed {}, \
         speculate {}",
        load.as_secs_f64(),
        defaults.max_tokens,
        defaults.thinking,
        defaults.sampling.temperature,
        defaults.sampling.top_k,
        defaults.sampling.top_p,
        defaults.sampling.min_p,
        defaults.sampling.repeat_penalty,
        defaults.sampling.repeat_last_n,
        match defaults.seed {
            Some(seed) => seed.to_string(),
            None => "from entropy, per request".to_string(),
        },
        defaults.speculate,
    );
    // Last, and with the address the listener actually got, so that `--port 0`
    // is usable and a client can treat this line as "the server is up".
    eprintln!("listening on http://{}", listener.local_addr()?);
    rust_llm::serve::serve(&listener, runner, tokenizer, vocab, &defaults)?;
    Ok(())
}

/// The interactive surface.
///
/// Every turn re-renders the whole conversation and resets the runner. That is
/// not laziness: the chat template drops the reasoning of assistant turns
/// before the last user turn, so the token prefix of turn N+1 is not a prefix
/// of turn N and reusing the KV cache across turns would be wrong rather than
/// merely stale. See `docs/decisions.md`.
fn chat(
    args: &Args,
    tokenizer: &Tokenizer,
    runner: &mut Runner<'_>,
    sampler: &mut Sampler,
    load: std::time::Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "loaded in {:.2}s. /reset starts over, /exit or Ctrl-D quits.",
        load.as_secs_f64()
    );
    let mut messages: Vec<Message> = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message::system(system.clone()));
    }
    let keep = messages.len();

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut line = String::new();
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        line.clear();
        if input.read_line(&mut line)? == 0 {
            // Ctrl-D.
            eprintln!();
            return Ok(());
        }
        let text = line.trim();
        match text {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                messages.truncate(keep);
                runner.reset();
                sampler.reset();
                eprintln!("(conversation cleared)");
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(text.to_string()));
        let prompt = render(&messages, args.thinking)?;
        let tokens = tokenizer.encode(&prompt, true)?;
        if tokens.len() >= runner.context() {
            eprintln!(
                "the conversation is {} tokens and the context holds {}; \
                 /reset, or restart with a larger --context",
                tokens.len(),
                runner.context()
            );
            messages.pop();
            continue;
        }

        runner.reset();
        let (ids, stop, _) = stream(runner, sampler, tokenizer, &tokens, args)?;
        if stop == Stop::ContextFull {
            eprintln!(
                "the context of {} positions filled up part-way through this answer; \
                 /reset, or restart with a larger --context",
                runner.context()
            );
        }
        messages.push(Message::assistant(tokenizer.decode(&ids, false)?));
    }
}

fn parse(args: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let mut out = Args {
        command: String::new(),
        // No default: a path baked in here would be a path on one machine.
        model: std::env::var("RUST_LLM_MODEL")
            .map(PathBuf::from)
            .unwrap_or_default(),
        tokenizer: None,
        prompt: String::new(),
        system: None,
        context: 4096,
        threads: vec![default_threads()],
        max_tokens: None,
        batch: rust_llm::forward::DEFAULT_BATCH,
        // Off unless asked for: see docs/benchmarks.md for the measurement the
        // default comes out of.
        speculate: 0,
        tier: None,
        prompt_tokens: 19,
        repeats: 5,
        thinking: true,
        raw: false,
        sampling: Sampling::default(),
        seed: None,
        // Loopback: this server has no authentication and runs a model that
        // answers whatever it is asked, so it is not something to put on a
        // network by default.
        host: "127.0.0.1".to_string(),
        port: 8080,
    };
    let mut args = args.peekable();
    let mut seen_prompt = false;
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-h" | "--help" | "help" => return Ok(None),
            "--think" => out.thinking = true,
            "--no-think" => out.thinking = false,
            "--raw" => out.raw = true,
            "--model" => out.model = PathBuf::from(value()?),
            "--tokenizer" => out.tokenizer = Some(PathBuf::from(value()?)),
            "--system" => out.system = Some(value()?),
            "--prompt" => {
                out.prompt = value()?;
                seen_prompt = true;
            }
            "--host" => out.host = value()?,
            "--port" => {
                let v = value()?;
                out.port = v
                    .parse()
                    .map_err(|_| format!("--port needs a port number, got {v:?}"))?;
            }
            "--context" => out.context = number(&value()?, "--context")?,
            "--threads" => out.threads = thread_list(&value()?)?,
            "--max-tokens" => out.max_tokens = Some(number(&value()?, "--max-tokens")?),
            "--batch" => out.batch = number(&value()?, "--batch")?.max(1),
            "--speculate" => {
                let v = number(&value()?, "--speculate")?;
                if v > rust_llm::forward::MAX_DRAFT {
                    return Err(format!(
                        "--speculate is at most {}, got {v}; a longer draft is {} MiB of \
                         recurrent state that no measured acceptance rate pays for",
                        rust_llm::forward::MAX_DRAFT,
                        (v + 1) * 50,
                    ));
                }
                out.speculate = v;
            }
            "--tier" => out.tier = Some(value()?),
            "--prompt-tokens" => out.prompt_tokens = number(&value()?, "--prompt-tokens")?.max(1),
            "--repeats" => out.repeats = number(&value()?, "--repeats")?.max(1),
            "--temperature" => {
                out.sampling.temperature = positive(&value()?, "--temperature", true)?
            }
            "--top-k" => out.sampling.top_k = number(&value()?, "--top-k")?,
            "--top-p" => out.sampling.top_p = fraction(&value()?, "--top-p")?,
            "--min-p" => out.sampling.min_p = fraction(&value()?, "--min-p")?,
            // A penalty of zero would zero every repeated negative logit and
            // divide every positive one by nothing; it is not a setting.
            "--repeat-penalty" => {
                out.sampling.repeat_penalty = positive(&value()?, "--repeat-penalty", false)?
            }
            "--repeat-last-n" => out.sampling.repeat_last_n = number(&value()?, "--repeat-last-n")?,
            "--seed" => {
                let v = value()?;
                out.seed = Some(
                    v.parse()
                        .map_err(|_| format!("--seed needs a number, got {v:?}"))?,
                );
            }
            other if out.command.is_empty() && !other.starts_with('-') => {
                out.command = other.to_string();
            }
            other => return Err(format!("unknown argument {other:?}\n\n{USAGE}")),
        }
    }
    if out.command.is_empty() {
        return Ok(None);
    }
    if !matches!(
        out.command.as_str(),
        "run" | "chat" | "serve" | "bench" | "profile" | "trace"
    ) {
        return Err(format!("unknown command {:?}\n\n{USAGE}", out.command));
    }
    if matches!(out.command.as_str(), "run" | "trace") && !seen_prompt {
        return Err(format!(
            "--prompt is required by {}\n\n{USAGE}",
            out.command
        ));
    }
    if out.model.as_os_str().is_empty() {
        return Err(format!(
            "no model: pass --model <path> or set RUST_LLM_MODEL (see docs/models.md)\n\n{USAGE}"
        ));
    }
    Ok(Some(out))
}

fn number(s: &str, what: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("{what} needs a number, got {s:?}"))
}

/// A float argument. Out of range is refused by name rather than clamped: a
/// silently clamped `--top-p 1.5` would look like it worked.
fn float(s: &str, what: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("{what} needs a number, got {s:?}"))?;
    if !v.is_finite() {
        return Err(format!("{what} needs a finite number, got {s:?}"));
    }
    Ok(v)
}

fn fraction(s: &str, what: &str) -> Result<f32, String> {
    let v = float(s, what)?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("{what} must be between 0 and 1, got {s}"));
    }
    Ok(v)
}

fn positive(s: &str, what: &str, zero_allowed: bool) -> Result<f32, String> {
    let v = float(s, what)?;
    if v < 0.0 || (v == 0.0 && !zero_allowed) {
        return Err(if zero_allowed {
            format!("{what} cannot be negative, got {s}")
        } else {
            format!("{what} must be above 0, got {s}")
        });
    }
    Ok(v)
}

/// `rust-llm bench`: the fixed prompt, truncated to an exact token count, run
/// at every thread count the sweep asks for.
///
/// The prompt is truncated in token space rather than in characters, so
/// `--prompt-tokens 19` is 19 tokens and not "about 19": the comparison
/// between two commits is only worth something if the amount of work is the
/// same on both sides.
fn bench(
    model: &Model<'_>,
    tokenizer: &Tokenizer,
    args: &Args,
    load: std::time::Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let generate = args.max_tokens.unwrap_or(16);
    if args.prompt_tokens + generate > args.context {
        return Err(format!(
            "prompt {} plus {generate} generated tokens does not fit a context of {}",
            args.prompt_tokens, args.context
        )
        .into());
    }
    let prompt = bench_prompt(tokenizer, args.prompt_tokens)?;
    let prompt = &prompt[..];

    eprintln!(
        "model {} ({} blocks), loaded in {:.2}s",
        args.model.display(),
        model.config.n_blocks,
        load.as_secs_f64(),
    );
    bench::sweep(
        model,
        prompt,
        &args.threads,
        &bench::Options {
            context: args.context,
            generate,
            repeats: args.repeats,
            batch: args.batch,
            speculate: args.speculate,
        },
    );
    Ok(())
}

/// `rust-llm profile`: one table for the prompt phase and one for generation,
/// over the same fixed text `bench` uses so the two commands describe the same
/// work.
///
/// The warm-up is not in either table: it runs before the timers are cleared.
fn profile(
    model: &Model<'_>,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let generate = args.max_tokens.unwrap_or(16);
    let prompt = bench_prompt(tokenizer, args.prompt_tokens)?;
    let prompt = &prompt[..];
    // The same check `bench` makes, and for the same reason: without it the
    // run ends in the forward pass's assert instead of saying what to change.
    if prompt.len() + generate > args.context {
        return Err(format!(
            "prompt {} plus {generate} generated tokens does not fit a context of {}",
            prompt.len(),
            args.context
        )
        .into());
    }
    let mut runner = Runner::with_batch(model, args.context, args.threads[0], args.batch);

    eprintln!(
        "quota {}, load {:.2}",
        bench::cpu_quota().unwrap_or_else(|| "unknown".into()),
        bench::load_average().unwrap_or(f64::NAN),
    );

    // One pass with the timers off, so page faults and thread start-up are
    // not in the table.
    bench::warm_up(&mut runner, prompt, generate);
    runner.enable_profile();

    runner.clear_profile();
    runner.reset();
    let mut read = 0;
    while read < prompt.len() {
        let take = (prompt.len() - read).min(runner.max_batch());
        runner.forward_batch(&prompt[read..read + take]);
        read += take;
    }
    print!("{}", runner.profile_report("prompt", prompt.len()));

    let mut next = sample::argmax(runner.logits());
    runner.clear_profile();
    for _ in 0..generate {
        next = sample::argmax(runner.forward(next));
    }
    println!();
    print!("{}", runner.profile_report("generation", generate));
    Ok(())
}

/// Exactly `n` tokens of the fixed bench text, repeated if `n` is more than it
/// holds.
///
/// Repeating rather than refusing, because the long-context configurations
/// need a prompt that fills the context and the measurement does not care what
/// the text says: what matters is that two runs read the same `n` tokens, so
/// the work being compared is the same. The copies are joined with a space, so
/// a copy boundary is an ordinary word boundary rather than a token that exists
/// nowhere else in the text.
fn bench_prompt(tokenizer: &Tokenizer, n: usize) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut ids = tokenizer.encode(bench::FILLER, true)?;
    if n > ids.len() {
        let copies = n.div_ceil(ids.len());
        ids = tokenizer.encode(&vec![bench::FILLER; copies].join(" "), true)?;
    }
    assert!(
        ids.len() >= n,
        "{n} tokens asked for, the repeated bench text gives {}",
        ids.len()
    );
    ids.truncate(n);
    Ok(ids)
}

/// `4` or `1,2,3,4`: one thread count for `run`, a sweep for `bench`.
///
/// Always at least one count: `split` yields one empty part for an empty
/// argument, and that fails in [`number`].
fn thread_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|part| number(part.trim(), "--threads").map(|n| n.max(1)))
        .collect()
}
