# rust-llm

Native Rust runner for a local, uncensored Qwen3.5-4B (Heretic) GGUF model on CPU.

No llama.cpp, no ggml, no C or C++ anywhere in the build.

## Layout

- `crates/llm-core` — GGUF reader over a memory map, ggml quantization block
  layouts, and the dequant-free quantized matmul kernels with their runtime
  CPU dispatch.
- `crates/rust-llm` — tokenizer, chat template, and the model and CLI.
- `fixtures/` — committed test fixtures and the script that regenerates them.
- `docs/models.md` — which model, where to get it, and what is inside the file.
- `docs/decisions.md` — every crate and data-layout choice with its reason.

## Building and testing

```sh
cargo test --workspace --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
```

The model weights are not in the repository, so the tests that need them are
`#[ignore]`d and CI is green without them. With the model on disk (see
`docs/models.md`):

```sh
cargo test --workspace --release -- --ignored --nocapture
```

Kernel microbenchmark:

```sh
cargo run --release -p llm-core --example bench_kernels
```
