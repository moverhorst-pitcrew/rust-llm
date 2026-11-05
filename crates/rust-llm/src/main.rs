//! `rust-llm` — the command line around the runner.
//!
//! Arguments are parsed by hand. There are seven of them, they are all
//! `--name value`, and a parser crate would be a dependency and a build-time
//! cost for something this shape.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use rust_llm::chat::{ChatOptions, Message};
use rust_llm::forward::Runner;
use rust_llm::model::Model;
use rust_llm::{Tokenizer, generate};

use llm_core::Gguf;

const USAGE: &str = "\
rust-llm run --prompt <text> [options]
rust-llm trace --prompt <text> [options]

  run     generate greedily from the prompt and stream the result
  trace   print one per-layer activation sum per line, for comparison
          against a reference runner (see docs/decisions.md)

options
  --model <path>        GGUF file; required, or set $RUST_LLM_MODEL
  --tokenizer <path>    tokenizer.json (default: next to the model)
  --prompt <text>       the user message (required)
  --system <text>       a system message, placed first
  --context <n>         positions to allocate for the KV cache (default 4096)
  --threads <n>         worker threads (default: all cores)
  --max-tokens <n>      stop after this many generated tokens (default 512)
  --think, --no-think   thinking mode, on by default, as the template is
  --raw                 take --prompt as the prompt verbatim, no chat template
";

struct Args {
    command: String,
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    prompt: String,
    system: Option<String>,
    context: usize,
    threads: usize,
    max_tokens: usize,
    thinking: bool,
    raw: bool,
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
    let model = Model::load(&gguf)?;
    let mut runner = Runner::new(&model, args.context, args.threads);
    let load = load.elapsed();

    let prompt = if args.raw {
        args.prompt.clone()
    } else {
        let mut messages = Vec::new();
        if let Some(system) = &args.system {
            messages.push(Message::system(system.clone()));
        }
        messages.push(Message::user(args.prompt.clone()));
        rust_llm::chat::render(
            &messages,
            ChatOptions {
                add_generation_prompt: true,
                thinking: args.thinking,
            },
        )?
    };
    let tokens = tokenizer.encode(&prompt, true)?;

    eprintln!(
        "model {} ({} blocks, {} full attention, {} linear), kernels {}, {} threads, context {}",
        args.model.display(),
        model.config.n_blocks,
        model.config.n_full_blocks(),
        model.config.n_linear_blocks(),
        runner.kernels().tier().as_str(),
        runner.threads(),
        runner.context(),
    );
    eprintln!(
        "loaded in {:.2}s, prompt is {} tokens",
        load.as_secs_f64(),
        tokens.len()
    );

    if args.command == "trace" {
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

    let mut stdout = std::io::stdout();
    let (ids, stop, timings) = generate::generate(&mut runner, &tokens, args.max_tokens, |id| {
        // One token at a time, decoded on its own: a multi-byte character
        // split across two tokens would come out as a replacement
        // character, so the whole text is decoded again at the end.
        if let Ok(text) = tokenizer.decode(&[id], false) {
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        }
    });
    println!();

    let text = tokenizer.decode(&ids, false)?;
    if text.as_bytes().contains(&0xef) {
        // The streamed form had a split character in it; print the clean text.
        eprintln!("--- decoded as a whole ---");
        println!("{text}");
    }

    eprintln!(
        "stopped: {stop:?}\nprompt   {} tokens in {:.2}s = {:.2} tok/s\ngenerate {} tokens in {:.2}s = {:.2} tok/s",
        timings.prompt_tokens,
        timings.prompt_time.as_secs_f64(),
        timings.prompt_per_second(),
        timings.generated_tokens,
        timings.generate_time.as_secs_f64(),
        timings.generate_per_second(),
    );
    if let Some(peak) = peak_rss_kib() {
        eprintln!(
            "peak RSS {peak} KiB ({:.2} GiB)",
            peak as f64 / (1 << 20) as f64
        );
    }
    Ok(())
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
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        max_tokens: 512,
        thinking: true,
        raw: false,
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
            "--context" => out.context = number(&value()?, "--context")?,
            "--threads" => out.threads = number(&value()?, "--threads")?.max(1),
            "--max-tokens" => out.max_tokens = number(&value()?, "--max-tokens")?,
            other if out.command.is_empty() && !other.starts_with('-') => {
                out.command = other.to_string();
            }
            other => return Err(format!("unknown argument {other:?}\n\n{USAGE}")),
        }
    }
    if out.command.is_empty() {
        return Ok(None);
    }
    if out.command != "run" && out.command != "trace" {
        return Err(format!("unknown command {:?}\n\n{USAGE}", out.command));
    }
    if !seen_prompt {
        return Err(format!("--prompt is required\n\n{USAGE}"));
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

/// `VmHWM` from `/proc/self/status`, where it exists.
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:")?.split_whitespace().next())
        .and_then(|v| v.parse().ok())
}
