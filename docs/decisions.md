# Decisions

Every crate and every data-layout choice, with its justification. Performance
first, then simplicity. A choice that is only simpler loses to one that is
measurably faster on this model.

## No C or C++, anywhere

No llama.cpp, no ggml, no FFI. Pure Rust crates are fine where they are the
right tool, which means `cargo tree --edges build` may not contain `cc`,
`cmake` or `bindgen`. CI checks that on every push, because a C toolchain
usually arrives quietly, as a transitive `-sys` dependency of something
harmless.

## Workspace layout: two crates

- `llm-core` — GGUF reader, ggml quantization block layouts, quantized matmul
  kernels and their dispatch.
- `rust-llm` — tokenizer, chat template, and later the model and the CLI.

Two crates rather than one, because the tokenizer pulls in a dependency tree
(`tokenizers`, `regex`, `serde`) that the loader and the kernels have no
business compiling against, and rather than four because the loader and the
kernels have to agree on the block layouts: the loader needs a block's size on
disk to index a tensor, and the kernels read those same bytes in place. Putting
them in separate crates would mean one of them owning the layouts and the other
depending on it for nothing else.

## Edition 2024, Rust stable 1.99

Stable only. `Cargo.lock` is committed because this is an application, not a
library: a reproducible build matters more than testing against the newest
versions of everything.

`[profile.release]` uses `lto = "thin"` and `codegen-units = 1`. The kernels
are built from small functions that must inline into their loops; with the
default 16 codegen units they do not reliably, and the microbenchmark measures
the wrong thing. `cargo bench` inherits this profile.

## Dependencies

| crate | why |
| --- | --- |
| `memmap2` | The only sane way to get 2.7 GB of weights addressable without reading them. Thin, pure Rust, no build script. |
| `half` | `f16` decoding for the ggml block scales. `f16` in `core` is still unstable, and hand-rolling the subnormal cases is a silent-wrongness risk for twenty saved lines. |
| `thiserror` | Error enums with a `Display` that names the file, the key or the tensor. Derive only, nothing at runtime. |
| `tokenizers` | See the tokenizer section. |
| `serde_json` | Reads the committed test fixtures. Already in the tree through `tokenizers`. |

Deliberately not taken:

- **`candle`**. It would hand us a forward pass, but not this architecture, and
  it owns the memory layout and the kernel path, which is exactly what we need
  to control. A linear-attention block would be written from scratch inside it
  anyway, against its tensor abstraction instead of against the mmap.
- **`rayon`** is not here yet. Threading lands with the forward pass, where
  there is something to parallelise over.
- **`bytemuck`**. Three `from_le_bytes` calls and one checked pointer cast.

## GGUF loader

**Weights stay in the map.** The header, metadata and tensor index are parsed
into owned structures; tensor bytes are never copied. `Gguf::tensor` returns a
slice borrowed from the map, and the kernels read the quantized bytes in place.
Opening the 2.71 GB file adds 50312 KiB of resident memory, 1.90% of the file,
essentially all of it the 248320-token vocabulary and the 247587 merges.
`cargo test -p llm-core --release --test real_model -- --ignored --nocapture`
prints it, and the number is the delta across `Gguf::open`, so the baseline the
test harness starts with does not flatter it. That test binary deliberately
holds one test: resident set size belongs to the process, so a second test
opening the model in another harness thread would be measured along with it.

**No self-referential struct.** `Gguf` owns the `Mmap` and stores byte offsets,
not slices; `tensor()` resolves an offset against `&self.mmap` on each call.
This is why there is no lifetime parameter on `Gguf`, no `Pin`, and no
owning-reference crate.

**Metadata arrays hold `Value`s, including nested arrays.** GGUF allows an
array of arrays, so a parallel element-type enum would have to recurse anyway.
The cost is the enum tag per element, about 16 MB on the vocabulary, which is
noise against a 2.7 GB model and is paid once at load.

**The block count comes out of the tensors, not out of `block_count`.** In this
conversion `qwen35.block_count` is 33 while `blk.32` does not exist, because
the MTP block was stripped. `qwen35::layer_plan` takes
`block_count - nextn_predict_layers` and then requires every block it implies
to be present, so a file that really is missing a block fails with the name of
the missing tensor instead of producing a 31-block model.

**Layer typing comes from `qwen35.attention.recurrent_layers`.** Deriving it
from `full_attention_interval` would be arithmetic that happens to agree with
this file.

**The loader validates, the kernels assume.** Bounds, alignment, block-aligned
row lengths, unknown type tags and a dimension list whose byte count does not
fit in a `u64` are all rejected at open time with an error that names the
tensor. The size arithmetic is checked rather than wrapping, because it runs on
numbers straight out of a file header: dimensions of `[1 << 32, 1 << 32]` wrap
to a length of zero, which would then pass the bounds check and open as a
short tensor instead of failing. After that the kernels index without
re-checking, which is what keeps the hot loop free of branches that cannot
fire.

## Quantized kernels

**Dequant-free is the only product path.** Weights stay `Q4_K`/`Q6_K`/`Q8_0` in
the map. Activations are quantized to int8 once per matmul, the dot products
accumulate in i32, and the f32 block scales are applied per block. Nothing
dequantizes a weight to f32 outside the tests. This is a bandwidth decision
before it is an arithmetic one: a 4B model on CPU is limited by how fast the
weight bytes can be streamed, and dequantizing to f32 would multiply the bytes
that cross the memory bus by eight.

**One activation format for all three weight types.** Activations go into
`Q8_K`-shaped blocks: 256 int8 quants, one f32 scale, and the sum of each group
of 16 quants. The group sums are what let the offset terms collapse into a dot
product instead of a second pass over the activations: `Q4_K` subtracts a
per-sub-block minimum, and `Q6_K` carries a constant -32 bias.

A single format means `Q8_0`, whose weight blocks are 32 wide, shares one
activation scale across eight of its blocks instead of having one each. That is
coarser than ggml's pairing, and it is the right trade here: one scratch buffer,
one quantization pass, one code path, against a quantization error that is
already well inside what int8 activations cost at all.

**No allocations in the hot loop.** `Activations::new` takes the largest `K` in
the model and allocates once; `quantize` refills it and panics rather than grow.
`matvec` writes into a caller-owned slice. There is no allocation anywhere below
the forward pass.

**A `K` that is not a whole number of blocks needs no special case.**
`quantize` zeroes the tail of its last super-block, so a zero activation kills
whatever the weight row holds past `K`. Every kernel therefore runs whole blocks
and never looks at `K` except to count them.

**Dispatch is a function table, resolved once.** `Kernels::detect` reads the CPU
features with `is_x86_feature_detected!` and fills an array indexed by the ggml
type discriminant. It is runtime detection, not `#[target_feature]` chosen at
compile time, because the machine that runs the binary need not be the machine
that built it. `Tier` is ordered weakest to strongest and an AVX-512 tier drops
in by adding a variant and one branch; nothing else changes. Looking a kernel up
is one array index, outside any loop.

The three tiers are scalar, AVX2+FMA and AVX2+VNNI. `Q5_K` returns an explicit
"unsupported quantization type" error: this file does not contain it.

**`VPDPBUSD` fits `Q4_K` and `Q6_K` but not `Q8_0`.** It needs an unsigned left
operand, which is exactly how the two k-quants store their quants (4-bit and
6-bit unsigned, bias applied afterwards). `Q8_0` quants are signed, and
correcting for that needs a per-block sum of the weights that is not in the
file, which costs more than the instruction saves; the VNNI tier keeps the AVX2
kernel for that type.

Applying `Q4_K`'s 6-bit scales with `VPMULLD` after `VPDPBUSD` made the VNNI
kernel *slower* than the AVX2 one. Every `VPDPBUSD` lane is at most
`4 * 15 * 127`, which fits in i16, so two sub-blocks are packed with `VPACKSSDW`
and scaled with one `VPMADDWD` instead. Measured on this box (single thread,
`2560x8192`): scalar 2.02 ms, AVX2 0.61 ms, VNNI 0.59 ms.

**Where the sign trick is exact and where it is not.** `Q8_0` meets two signed
operands on `VPMADDUBSW`, which wants one unsigned. Taking `|weight|` with
`VPSIGNB` and moving the weight's sign onto the activation is exact over the
whole i8 range, because `VPSIGNB` leaves -128 at -128 and -128 read as an
unsigned byte is 128, which is its absolute value. The mirror image, `|activation|`
with the sign moved onto the weight, is wrong for a weight byte of -128. The
kernel tests catch the difference.

**The scalar reference lives in the tests.** The product has a scalar *tier*,
because a machine without AVX2 still has to run, but the thing the fast kernels
are validated against is a separate reference in
`crates/llm-core/tests/kernels.rs` that dequantizes a row to f32 straight from
the ggml block layout and dots it in f64. It does not share the kernels'
restructured algebra, so it catches a misread block layout as well as a wrong
shuffle.

Tolerance is `1e-5` of the sum of the absolute terms, which bounds f32
accumulation drift and nothing else. That it is tight enough was checked by
introducing seven single-line errors (swapped scale lanes, swapped nibble
halves, flipped bias and minimum signs, a wrong scale index, an off-by-one
activation offset) one at a time; all seven failed the tests.

`cargo run --release -p llm-core --example bench_kernels` prints one line per
(type, shape, tier) with the weight bandwidth and the speedup over scalar. It is
a starting point for M2's tuning, not a tuned result: single threaded, one
matvec at a time.

## Tokenizer

**`tokenizers` on the model's own `tokenizer.json`, not a hand-rolled BPE.**
The pretokenizer regex contains a lookahead (`\s+(?!\S)`), the normalizer is
NFC, `add_bos` is false and `clean_spaces` is false. Byte-level BPE has enough
corners that a reimplementation is where token-for-token agreement quietly
dies, and tokenization is a one-time cost per prompt, not hot. This is the one
component where writing it ourselves buys nothing and risks the correctness
gate.

Taken as `default-features = false, features = ["fancy-regex"]`. The default
features pull `onig` and `esaxx_fast`, both of which need a C++ compiler;
`fancy-regex` is the pure-Rust engine that supports the lookahead. `esaxx-rs`
is still in the tree, but without its `cpp` feature, so it builds no C++ and
`cc` is absent from `Cargo.lock`. CI greps `Cargo.lock` for it, because
enabling `tokenizers`' default features is exactly how that would come back.

**`tokenizer.json` is committed, under `fixtures/`.** Checking exact token ids
with no network and no weights needs the tokenizer data, and there is no way
around that. It is 20 MB in the working tree and about 3.5 MB in the pack,
which is the price of a self-contained test; it is config, not weights. The
ignored `weights` test asserts the committed copy is byte-identical to the
model's, so it cannot drift away from the file the weights were converted with
while the id checks go on passing against the drift.

**The fixtures are generated by someone else's code.** `fixtures/generate.py`
produces `token_ids.json` from the Hugging Face `tokenizers` *package* and
`chat_prompts.json` by rendering the model's own `chat_template.jinja` with
Jinja2. Generating them from our own Rust would only assert that we agree with
ourselves. Cases cover ASCII, leading and trailing whitespace, tabs, newlines,
CRLF, CJK, emoji, an NFC-decomposed input, and text containing the literal
`<|im_start|>`, `<|endoftext|>` and `<think>`.

**A mismatched weight/tokenizer pair fails at load, not at generation.**
`Tokenizer::verify_against_weights` compares all 248077 tokens the tokenizer
can produce against the vocabulary embedded in the GGUF, checks
`tokenizer.ggml.model` and `.pre`, and requires the 243 slots past the end of
the tokenizer to be marked unused rather than being vocabulary we cannot
produce. The merge list is not reachable through the `tokenizers` crate's
public API, so the ignored test compares all 247587 merges against the raw
JSON instead.

**Control tokens in input text are always recognised.** `add_special_tokens`
turns out to be the post-processor's knob, not the added-vocabulary one, and
this model's post-processor adds nothing, so the flag makes no difference. A
user's literal `<|im_start|>` therefore becomes token 248045. That is what
makes a rendered prompt tokenize the way the reference runner tokenizes it,
which the correctness gate needs, and it means anything that pastes untrusted
text into a prompt has to screen it first. The tests pin both halves of this
down.

## Chat template

**Rendered natively, no Jinja engine.** The template this model ships is a few
hundred bytes of real behaviour wrapped in tool-call and vision handling that
is out of scope. A Jinja interpreter would be a dependency and a second place
for the prompt to go subtly wrong, and the behaviour that matters is twelve
lines of string building.

Implemented: system, user and assistant turns over multiple turns, the
generation prompt, and the thinking prefill (`<think>\n` with thinking on,
`<think>\n\n</think>\n\n` with it off). Also implemented, because the template
does it and it changes the prompt: content is trimmed, a system message is
emitted once from the top and skipped by the main loop, and an assistant turn
keeps its `<think>` block only when it comes after the last user message, so a
long conversation does not replay the model's own reasoning.

Not implemented: tool calls, tool responses and vision content. `Role` parses
from a string and refuses `tool` and `function` by name, which is where such a
turn arrives from outside; `Message::content` is a `String`, so there is no
place for an image part to be dropped silently.

The tests compare the renderer against all fifteen fixture prompts byte for
byte, and the fixture records the template's sha256, so a different template
fails rather than being rendered by the old rules.
