# Model files

The runner targets one model: the Heretic (decensored) build of Qwen3.5-4B,
from [`saidutta69/Qwen3.5-4B-heretic`](https://huggingface.co/saidutta69/Qwen3.5-4B-heretic).

It was picked over the other Heretic and uncensored Qwen3.5-4B builds because
it is the newest one that ships a GGUF *and* the matching bf16 safetensors,
`tokenizer.json` and the Heretic `reproduce/` artifacts in the same repository.
The thing we run and the thing we compare against therefore come from one set
of weights. The fallback, if that repository disappears, is
`HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive`.

Weights are never committed: `.gitignore` excludes `*.gguf` and `/models/`.
`tokenizer.json` is committed, as `fixtures/tokenizer.json`, so the tokenizer
tests run without the weights and without network. It is a byte-identical copy;
the ignored `weights` test asserts that.

## Files

Repository revision `a950550950e71a01114f68b79ac8dfd6a8720a4f` (2026-11-01).

| file | size | sha256 |
| --- | --- | --- |
| `Qwen3.5-4B-heretic-Q4_K_M.gguf` | 2.71 GB | `7141a114b722c2ee90b123c853177773f8e52092f80a296d442f5c6aec43f404` |
| `tokenizer.json` | 20.0 MB | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` |
| `config.json` | 2.9 kB | `f9940e34c56718f5f0d4bd21784086e1cbae3e0d766c14ffced7d00687414ab1` |
| `chat_template.jinja` | 7.8 kB | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

The repository also carries `F16`, `Q5_K_M`, `Q6_K` and `Q8_0` GGUFs of the
same weights. `Q4_K_M` is the one the runner is built and tested against; it is
the only one that fits an 8 GB target with room for the KV cache.

## Download

```sh
mkdir -p models/qwen3.5-4b-heretic && cd models/qwen3.5-4b-heretic
base=https://huggingface.co/saidutta69/Qwen3.5-4B-heretic/resolve/main
for f in Qwen3.5-4B-heretic-Q4_K_M.gguf tokenizer.json config.json chat_template.jinja; do
    curl -fL --retry 3 -o "$f" "$base/$f"
done
sha256sum Qwen3.5-4B-heretic-Q4_K_M.gguf tokenizer.json config.json chat_template.jinja
```

The tests that read the model look for it at
`/paperclip/models/qwen3.5-4b-heretic/Qwen3.5-4B-heretic-Q4_K_M.gguf`, or
wherever `RUST_LLM_MODEL` points. They are `#[ignore]`d, so CI passes without
the weights; run them with:

```sh
cargo test --workspace --release -- --ignored --nocapture
```

## What is in the Q4_K_M file

Read out of the file, not out of `config.json`:

- `general.architecture = qwen35`, GGUF version 3, 426 tensors, 38 metadata keys
- quantization mix: 216 `Q4_K`, 33 `Q6_K`, 177 `F32`
- `block_count = 33`, `nextn_predict_layers = 1`, and only `blk.0..blk.31`
  exist: the MTP block was stripped on conversion. The real block count is 32.
- hybrid attention: blocks 3, 7, 11, 15, 19, 23, 27 and 31 are full attention,
  the other 24 are linear attention (gated delta net). This comes from the
  `qwen35.attention.recurrent_layers` bool array.
- `embedding_length = 2560`, `feed_forward_length = 9216`, `head_count = 16`,
  `head_count_kv = 4`, `key_length = value_length = 256`, `rms_norm_eps = 1e-6`
- `rope.freq_base = 1e7`, `rope.dimension_count = 64` of 256 (partial rotary),
  `rope.dimension_sections = [11, 11, 10, 0]`, `context_length = 262144`
- `ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`,
  `ssm.time_step_rank = 32`, `ssm.inner_size = 4096`
- vocabulary 248320 tokens with 247587 merges, tied embeddings
- two stop tokens: `<|im_end|>` = 248046 (the GGUF `eos_token_id`, and what the
  chat template emits) and `<|endoftext|>` = 248044 (what `config.json` calls
  `eos_token_id`). Generation stops on both.
