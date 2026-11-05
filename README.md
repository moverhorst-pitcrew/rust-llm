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
| `--think`, `--no-think` | thinking on | the model's own thinking mode |
| `--raw` | off | `run` only: take `--prompt` verbatim, no chat template |

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
`top_p`, `top_k`, `min_p`, `repetition_penalty`, `max_tokens`, `seed`, `stream`
and `chat_template_kwargs.enable_thinking` for the thinking toggle. Anything
else in the body is ignored, `stop` and `n` included; `docs/decisions.md` says
why.

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
cargo test --workspace --release -- --ignored --nocapture
```

One test in that set, `the_fixture_is_still_what_the_reference_produces`, also
needs a built reference runner and fails without one:
`RUST_LLM_REFERENCE_BIN` to the directory holding `llama-completion`, with
`LD_LIBRARY_PATH` pointing at the same directory. The build recipe is in
`docs/decisions.md` under "The reference harness".

Kernel microbenchmark:

```sh
cargo run --release -p llm-core --example bench_kernels
```

## Measuring

Throughput, as a median of repeated runs with the spread reported, per thread
count:

```sh
rust-llm bench --context 1024 --threads 1,2,3,4,5,6,8 --prompt-tokens 19 \
  --max-tokens 16 --repeats 5
```

The prompt is a fixed text truncated to exactly `--prompt-tokens` tokens, and
generation runs exactly `--max-tokens` tokens through the forward pass and
ignores stop tokens, so two runs measure the same work. The first run of each
thread count is a warm-up and is not counted. Every configuration prints its
thread count, context, prompt length, generated tokens, kernel tier, the cgroup
CPU quota and the load average before and after, because this box is shared and
a number without those is not comparable with another one.

Where the time goes, per operator, for the prompt phase and for generation:

```sh
rust-llm profile --context 1024 --threads 4 --prompt-tokens 19 --max-tokens 16
```

What the machine can stream from memory, which is the ceiling batch-1
generation runs into:

```sh
cargo run --release -p llm-core --example bench_bandwidth
```

`docs/benchmarks.md` holds the numbers these produce on the development box,
with llama.cpp on the same file at matching thread counts.
