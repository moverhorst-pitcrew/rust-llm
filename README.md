# rust-llm

Native Rust runner for a local, uncensored Qwen3.5-4B (Heretic) GGUF model on CPU.

No llama.cpp, no ggml, no C or C++ anywhere in the build.

## Layout

- `crates/llm-core` — GGUF reader over a memory map, ggml quantization block
  layouts, and the dequant-free quantized matmul kernels with their runtime
  CPU dispatch.
- `crates/rust-llm` — tokenizer, chat template, the model, the HTTP surface and
  the CLI.
- `fixtures/` — committed test fixtures and the script that regenerates them.
- `docs/models.md` — which model, where to get it, and what is inside the file.
- `docs/decisions.md` — every crate and data-layout choice with its reason.
- `docs/benchmarks.md` — the machine, the measured bandwidth ceiling, and every
  performance number with the command that produced it.

## Building

```sh
cargo build --release
```

The binary is `target/release/rust-llm`, written as `rust-llm` below.

The weights are not in the repository. `docs/models.md` says which file to
download and where to put it; everything below takes it from `--model` or from
`$RUST_LLM_MODEL`, and the tokenizer is read from next to the model unless
`--tokenizer` says otherwise.

```sh
export RUST_LLM_MODEL=/path/to/Qwen3.5-4B-heretic-Q4_K_M.gguf
```

## Using it

Three surfaces, all with the same model, context, threading and sampling
options:

```sh
rust-llm run --prompt "Why is the sky blue?"     # generate once, streamed
rust-llm chat                                    # an interactive conversation
rust-llm serve                                   # an HTTP server
rust-llm --help                                  # every option
```

`chat` keeps the conversation until `/reset`, and `/exit` or Ctrl-D ends it.
There is no line editing and no history, which would be a readline dependency
for a development surface.

### Options

| option | default | |
| --- | --- | --- |
| `--model <path>` | `$RUST_LLM_MODEL` | the GGUF file, required |
| `--tokenizer <path>` | next to the model | `tokenizer.json` |
| `--system <text>` | none | a system message, placed first |
| `--context <n>` | 4096 | positions allocated for the KV cache |
| `--threads <n>` | all cores | worker threads |
| `--max-tokens <n>` | 512 | stop after this many generated tokens |
| `--speculate <n>` | 0 (off) | draft n tokens per step and verify them in one batch |
| `--think`, `--no-think` | thinking on | the model's own thinking mode |
| `--raw` | off | `run` only: take `--prompt` verbatim, no chat template |

`--speculate` is speculative decoding by prompt lookup: the next few tokens are
guessed from the text so far, taken from where it last repeated itself, and
verified in one pass over the weights instead of one pass each. **It never
changes what comes out.** Every token is drawn by the same sampler in the same
order, so a fixed seed gives the same answer with it on and off, byte for byte,
and there is a test over the whole prompt set that says so.

Off by default because it is only worth it on text that repeats itself: 10 to
29 per cent faster on a long quoted passage, boilerplate or a diff, and nothing
on ordinary prose, where the drafts are wrong often enough to cost more than
they save. A draft costs about 50 MiB of memory. `docs/benchmarks.md` has the
acceptance rates and the gains per draft length, and the number the default
comes out of; the gains there do not order neatly by draft length, so read the
table rather than reaching for the largest number the memory allows.

Sampling, with Qwen3.5's own recommended values as the defaults. A value that
cannot be a setting is refused by name rather than clamped:

| option | default | |
| --- | --- | --- |
| `--temperature <f>` | 0.6 | `0` is greedy argmax |
| `--top-k <n>` | 20 | keep the n highest logits, `0` for all |
| `--top-p <f>` | 0.95 | keep the smallest set covering this mass |
| `--min-p <f>` | 0 (off) | drop below this fraction of the top probability |
| `--repeat-penalty <f>` | 1 (off) | above 1 discourages repetition |
| `--repeat-last-n <n>` | 64 | tokens the penalty looks back over |
| `--seed <n>` | from entropy | printed to stderr, so a run can be replayed |

## The HTTP server

```sh
rust-llm serve --host 127.0.0.1 --port 8080
```

`POST /v1/chat/completions` and `POST /v1/completions`, OpenAI-shaped, with and
without streaming. Loopback by default: there is no authentication, so this is
not something to put on a network as it stands.

One request is answered at a time. Generation is the `Runner` and its KV cache
and there is one of each, which is also all the hardware has: a single token
already uses every core. A second request waits its turn instead of being
refused.

Every option above is the default for a request that does not send that field;
the fields a request can override are `messages` or `prompt`, `temperature`,
`top_p`, `top_k`, `min_p`, `repetition_penalty`, `max_tokens`, `seed`,
`speculate`, `stream` and `chat_template_kwargs.enable_thinking` for the
thinking toggle. Anything
else in the body is ignored, `stop` and `n` included; `docs/decisions.md` says
why.

`speculate` is capped differently from the rest: the ring of state buffers it
needs is allocated when the server starts, so a request asking for more than
`--speculate` was given is refused with a 400 rather than quietly served with
fewer drafts. Less than that, down to 0, is honoured.

`--context` is the ceiling on the prompt plus the answer, so a `max_tokens`
above it is capped at it: no run could reach the larger number anyway. A prompt
that is `--context` tokens or longer leaves nothing to answer with and is
refused with a 400 naming both numbers, rather than answered with an empty
completion.

A chat completion:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "temperature": 0.6,
    "max_tokens": 256
  }'
```

The same one, streamed, which is what `-N` is for: without it curl buffers and
the point of streaming is lost.

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

```
data: {"choices":[{"delta":{"role":"assistant"},...}],...}
data: {"choices":[{"delta":{"content":"The"},...}],...}
data: {"choices":[{"delta":{},"finish_reason":"stop",...}],...}
data: [DONE]
```

A text completion, whose `prompt` goes to the model verbatim with no chat
template around it:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The three primary colors are", "max_tokens": 32}'
```

## Testing

```sh
cargo test --workspace --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
```

The model weights are not in the repository, so the tests that need them are
`#[ignore]`d and CI is green without them. The surfaces are covered without the
weights by a tiny committed model the tests write themselves. With the real
model on disk (see `docs/models.md`):

```sh
cargo test --workspace --release -- --ignored --nocapture --test-threads=1
```

`--test-threads=1` is not optional for that set. Every test in it loads the
model and starts a worker pool on all cores, so the harness's default of running
them in parallel oversubscribes the machine several times over: it makes the run
slower, it multiplies the resident memory by the number of tests, and any
measurement taken at the same time is meaningless.

One test in that set, `the_fixture_is_still_what_the_reference_produces`, also
needs a built reference runner and fails without one: point
`RUST_LLM_REFERENCE_BIN` at the directory holding `llama-completion`, which is
the only variable the test needs because it passes `LD_LIBRARY_PATH` to the
reference itself. Forgetting it does not cost only that test: the failing test
binary ends the whole run, so the binaries after `tests/gate.rs` never run and
the red is the missing variable rather than a broken gate. The build recipe is
in `docs/decisions.md` under "The reference harness".

Kernel microbenchmark:

```sh
cargo run --release -p llm-core --example bench_kernels
```

## Measuring

Throughput, as a median of repeated runs with the spread reported, per thread
count:

```sh
rust-llm bench --context 1024 --threads 1,2,3,4,5,6,7,8 --prompt-tokens 192 \
  --max-tokens 16 --repeats 3
```

With `--speculate n` the same command also reports what a verification row costs
per row count, and generation with the drafter on against the drafter off on one
runner, with the acceptance histogram:

```sh
rust-llm bench --context 1024 --threads 8 --prompt-tokens 400 \
  --max-tokens 32 --repeats 5 --speculate 4
```

A `--prompt-tokens` above the fixed bench text's own length repeats that text,
which is the case speculation is for. `rust-llm run --speculate n` prints the
same histogram for a prompt of your own.

The prompt is a fixed text truncated to exactly `--prompt-tokens` tokens, or
repeated if more are asked for than its 204, and generation runs exactly
`--max-tokens` tokens through the forward pass and ignores stop tokens, so two
runs measure the same work. The prompt is read in batches of `--batch` tokens,
16 by default, which is what makes it faster per token than generation; a prompt
of fewer tokens than the batch is one batch. The first run of each
thread count is a warm-up and is not counted. Every configuration prints its
thread count, context, prompt length, generated tokens, kernel tier, the cgroup
CPU quota and the load average before and after, because this box is shared and
a number without those is not comparable with another one.

Where the time goes, per operator, for the prompt phase and for generation:

```sh
rust-llm profile --context 1024 --threads 8 --prompt-tokens 192 --max-tokens 8
```

What the machine can stream from memory, which is the ceiling batch-1
generation runs into:

```sh
cargo run --release -p llm-core --example bench_bandwidth
```

`docs/benchmarks.md` holds the numbers these produce on the development box,
with llama.cpp on the same file at matching thread counts.
