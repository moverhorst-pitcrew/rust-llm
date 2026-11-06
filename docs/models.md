# Model files

The runner targets one model: the MTP-preserved Heretic (decensored) build of
Qwen3.5-4B. The weights are
[`RBergBauer/Qwen3.5-4B-MTP-Heretic`](https://huggingface.co/RBergBauer/Qwen3.5-4B-MTP-Heretic)
and the GGUF conversion is
[`mradermacher/Qwen3.5-4B-MTP-Heretic-GGUF`](https://huggingface.co/mradermacher/Qwen3.5-4B-MTP-Heretic-GGUF),
which mirrors that repository's full quant ladder.

It was picked over the Heretic build this runner started on because its GGUF
**keeps the multi-token-prediction block**, which is what drafting from the
model's own head needs; see "The file that did not have the MTP block" below
for what that cost and what it means for the numbers. Both repositories are
needed: the GGUF comes from the conversion, and `tokenizer.json`, `config.json`
and `chat_template.jinja` come from the weights.

Weights are never committed: `.gitignore` excludes `*.gguf` and `/models/`.
`tokenizer.json` is committed, as `fixtures/tokenizer.json`, so the tokenizer
tests run without the weights and without network. It is a byte-identical copy;
the ignored `weights` test asserts that.

## Files

| file | from | size | sha256 |
| --- | --- | --- | --- |
| `Qwen3.5-4B-MTP-Heretic.Q4_K_M.gguf` | the GGUF repo | 2 783 447 456 B (2.59 GiB) | `0d3cdafb8920a939cadd8590a5b0da52f2b0a4eb15df9301cf289721b3db0a21` |
| `tokenizer.json` | the weights | 19 989 325 B | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `config.json` | the weights | 2 914 B | `477550a01d14da52f85299a6f585693d63ced5aa576aed21b8e9bf586aefc875` |
| `chat_template.jinja` | the weights | 7 756 B | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

`Q4_K_M` is the one the runner is built and tested against. It keeps the model
in the same bandwidth class as the file before it, 2.78 GB against 2.71 GB,
which is what makes the decode numbers in `docs/benchmarks.md` comparable across
the swap. The ladder also has `Q6_K`,
`a4b8cf65396a911f28470ae9235f92fc5a4c0e73952d951aabcd8d5d3ee727d2`, 3 563 028 896
bytes; it differs for our purposes in exactly one tensor, `blk.32.nextn.eh_proj`
being `Q6_K` rather than `Q4_K`, and it is only worth downloading if the
measured draft acceptance lands under the measured break-even.

**The safetensors pairing holds, and the dtype is not what the GGUF repository
suggests.** The weights ship as three safetensors shards, `fp16` for the trunk
and `bf16` for the 15 tensors of the MTP block (120 599 552 of 4 659 865 088
parameters). That costs nothing here: the correctness gate's reference is
llama.cpp on the *same GGUF file*, so quantization is out of the comparison
either way.

Those 15 bf16 tensors are also the clearest statement of what the MTP block is.
Their safetensors names map onto the GGUF's one for one:

| safetensors | GGUF |
| --- | --- |
| `mtp.pre_fc_norm_embedding.weight` | `blk.32.nextn.enorm.weight` |
| `mtp.pre_fc_norm_hidden.weight` | `blk.32.nextn.hnorm.weight` |
| `mtp.fc.weight` `[2560, 5120]` | `blk.32.nextn.eh_proj.weight` `[5120, 2560]` |
| `mtp.layers.0.*` | `blk.32.attn_*`, `blk.32.ffn_*` |
| `mtp.norm.weight` | `blk.32.nextn.shared_head_norm.weight` |

The repository is multimodal (it carries a vision tower and the preprocessor
configs). The GGUF conversion is text-only, so none of that is in the file this
runner reads.

## Download

```sh
mkdir -p models/qwen3.5-4b-mtp-heretic && cd models/qwen3.5-4b-mtp-heretic
gguf=https://huggingface.co/mradermacher/Qwen3.5-4B-MTP-Heretic-GGUF/resolve/main
hf=https://huggingface.co/RBergBauer/Qwen3.5-4B-MTP-Heretic/resolve/main
curl -fL --retry 3 -O "$gguf/Qwen3.5-4B-MTP-Heretic.Q4_K_M.gguf"
for f in tokenizer.json tokenizer_config.json config.json \
         generation_config.json chat_template.jinja; do
    curl -fL --retry 3 -o "$f" "$hf/$f"
done
sha256sum Qwen3.5-4B-MTP-Heretic.Q4_K_M.gguf tokenizer.json config.json \
    chat_template.jinja
```

The tests that read the model take its path from `RUST_LLM_MODEL` and have no
default, so they say what to set rather than looking in a directory that only
exists on one machine. They are `#[ignore]`d, so CI passes without the weights;
run them with:

```sh
RUST_LLM_MODEL=<path>/Qwen3.5-4B-MTP-Heretic.Q4_K_M.gguf \
  cargo test --workspace --release -- --ignored --nocapture --test-threads=1
```

## What is in the Q4_K_M file

Read out of the file, not out of `config.json`:

- `general.architecture = qwen35`, GGUF version 3, 441 tensors, 51 metadata keys
- quantization mix: 222 `Q4_K`, 35 `Q6_K`, 184 `F32`
- `block_count = 33` and `nextn_predict_layers = 1`, and `blk.0..blk.32` all
  exist: the trunk is 32 blocks and `blk.32` is the MTP block. See "The MTP
  block" below.
- hybrid attention: blocks 3, 7, 11, 15, 19, 23, 27 and 31 are full attention,
  the other 24 are linear attention (gated delta net). This comes from the
  `qwen35.attention.recurrent_layers` bool array, which has 33 entries, the
  last of them false because the MTP block is full attention too.
- `embedding_length = 2560`, `feed_forward_length = 9216`, `head_count = 16`,
  `head_count_kv = 4`, `key_length = value_length = 256`, `rms_norm_eps = 1e-6`
- `rope.freq_base = 1e7`, `rope.dimension_count = 64` of 256 (partial rotary),
  `rope.dimension_sections = [11, 11, 10, 0]`, `context_length = 262144`
- `ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`,
  `ssm.time_step_rank = 32`, `ssm.inner_size = 4096`
- vocabulary 248320 tokens with 247587 merges, tied embeddings
  (`token_embd.weight` is `[2560, 248320]` `Q6_K` and doubles as the head)
- two stop tokens: `<|im_end|>` = 248046 (the GGUF `eos_token_id`, and what the
  chat template emits) and `<|endoftext|>` = 248044 (what `config.json` calls
  `eos_token_id`). Generation stops on both.

## The MTP block

`blk.32` is complete, 15 tensors: a full-attention Qwen3.5 block
(`attn_norm`, `attn_q` `[2560, 8192]`, `attn_k` and `attn_v` `[2560, 1024]`,
`attn_q_norm` and `attn_k_norm` `[256]`, `attn_output` `[4096, 2560]`,
`post_attention_norm`, `ffn_gate` and `ffn_up` `[2560, 9216]`, `ffn_down`
`[9216, 2560]`) plus the five the head adds: `nextn.enorm`, `nextn.hnorm`,
`nextn.eh_proj` `[5120, 2560]`, `nextn.shared_head_norm`.

There is no `nextn.embed_tokens` and no `nextn.shared_head_head`, so the block
reads tokens through `token_embd` and writes logits through the tied head, the
same two tensors the trunk uses. `eh_proj` being `[2 * n_embd, n_embd]` is the
concatenation of the normed hidden state and the normed embedding along the
embedding axis.

## The tokenizer is not byte-identical to the previous one

Same vocabulary (248044 entries plus 33 added tokens, 248077 in all) and the
same 247587 merges, so every id is the id it was. One thing differs: the
`pre_tokenizer`'s split regex matches `\p{L}+` where the previous file matched
`[\p{L}\p{M}]+`, so a combining mark no longer stays attached to the letter in
front of it.

That is invisible on anything NFC can compose, which is every non-ASCII case the
fixtures had: the NFC normalizer runs first, so no combining mark reaches the
regex. It is visible on scripts whose marks have no composed form. `नमस्ते दुनिया`
is 6 tokens under the old regex and 10 under this one. `fixtures/token_ids.json`
has a `combining_marks` case for exactly that, with ids from the Hugging Face
`tokenizers` package, so the difference is pinned rather than assumed.

## This conversion writes fewer tokenizer metadata keys

51 metadata keys against the previous file's 38, from a different quantizer. 16
are new `general.*` and `mradermacher.*` provenance keys, and three the previous
file had are gone: `tokenizer.ggml.normalizer.nfc`, `.clean_spaces` and
`.add_eos_token`. Nothing in the runner reads them. The NFC normalizer is pinned
on the `tokenizer.json` side instead, by
`the_committed_tokenizer_is_the_shape_the_weights_expect`; what `tests/weights.rs`
asserts about the GGUF is only that it does not claim something *other* than
NFC, which would mean these ids are not these weights' ids.

## The file that did not have the MTP block

The runner started on
[`saidutta69/Qwen3.5-4B-heretic`](https://huggingface.co/saidutta69/Qwen3.5-4B-heretic),
revision `a950550950e71a01114f68b79ac8dfd6a8720a4f`, file
`Qwen3.5-4B-heretic-Q4_K_M.gguf`, 2.71 GB, sha256
`7141a114b722c2ee90b123c853177773f8e52092f80a296d442f5c6aec43f404`. 426
tensors, 38 metadata keys, 216 `Q4_K` / 33 `Q6_K` / 177 `F32`.

**It said `nextn_predict_layers = 1` and did not contain `blk.32`.** The
conversion dropped it, which is what llama.cpp's converter did for this
architecture at the time, so these weights were trained with a head that the
file did not ship. That is why the first speculative decoding this runner got
drafts from the prompt and the text generated so far rather than from a head
(`docs/decisions.md`, "Speculative decoding"), and why prompt lookup stays:
it is the draft source that works on any file.

The loader still reads that file. `qwen35::layer_plan` reports the MTP block as
`Option<usize>` rather than requiring it, so a conversion without `blk.32` loads
with the same 32-block trunk and only MTP drafting is unavailable. A block past
the ones the metadata accounts for is still an error.

**The two files are different weights, not two conversions of one.** Swapping
meant re-deriving `fixtures/reference_generations.json` from scratch against the
new file, with its own permitted disagreements; the old ones say nothing about
these weights. `docs/decisions.md`, "The correctness gate", has what came out.
