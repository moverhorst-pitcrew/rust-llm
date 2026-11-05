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
| `serde_json` | Parses and builds the HTTP surface's bodies, and reads the committed test fixtures. Already in the tree through `tokenizers`. |

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

**No allocations in the hot loop.** `Activations::with_batch` takes the largest
`K` in the model and the largest batch and allocates once; `quantize_batch`
refills it and panics rather than grow. `matvec` writes into a caller-owned
slice. There is no allocation anywhere below the forward pass.

**A batch is `t` rows of the same format, quantized exactly as they would be
alone.** The scale is per row and per super-block, so no row's quants depend on
another's, which is what makes a batched matmul produce the same bits as `t`
single ones. A kernel is handed one row at a time as an `ActRow`, and never
needs to know whether it came out of a batch.

**There are two entry points per type: one weight row against one activation
row, and one weight row against a group of them.** The second exists because
about half of what a `Q4_K` super-block costs does not depend on the
activations: splitting the nibbles, unpacking the six-bit scales out of
`get_scale_min_k4`'s bit shuffle, and building the scale vector. Per token a
batch paid that `t` times; per group it pays it once, into a 256-byte stack
buffer the group's dot products then read from L1. Measured on the prompt path,
which is where it shows: 31.6 to 44.0 tok/s at 8 threads, 39 per cent.

The group is 16 rows. Bigger divides the unpacking further, but past 16 the
gain is under a per cent and the outputs stop fitting a small stack array; a
batch longer than the group is run as several groups. `Q8_0` has no batched
kernel: this file holds no `Q8_0` tensor, so there is nothing here to measure
one on, and an unmeasured kernel is a second rounding behaviour to keep green
for nothing.

**The batched kernel is bit-identical to the per-row one**, not merely close.
The same products accumulate in the same order and only the unpacking moves, so
the kernel test asserts it with `==` rather than a tolerance, for both types on
every tier, over the model's shapes and the awkward ones, at batch sizes either
side of the group boundary. That is what lets a prompt be read in batches
without changing a single logit, which is the property the correctness gate
would otherwise have to absorb.

**Where the batch's win actually comes from is the caller's loop order**, not
the kernel: the pool hands a worker a block of weight rows and runs it against
every row of the batch before moving on, so the block's bytes cross the memory
bus once for the whole batch. The kernel's own win is the unpacking. Both are
needed: without the loop order the prompt stays bandwidth-bound, and with it the
unpacking becomes the next bottleneck.

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

**The VNNI tier stays, and this is the measurement that decided it.** M1's
number above is one shape, single threaded, 3 per cent apart, which is inside
this box's noise, and the plan was to delete the tier if it could not do better
than that. The reason it could not be seen is that batch-1 generation is
bandwidth-bound: no kernel change shows up in it at all. The batched prompt path
is compute-bound, and there the two tiers differ reproducibly. Prompt tok/s,
context 1024, 192-token prompt, batch 16, two passes each alternated from one
binary:

| | `avx2+vnni` | `avx2+fma` |
| --- | --- | --- |
| 8 threads | 46.19, 44.53 | 43.63, 43.27 |
| 1 thread | 10.48 | 9.96 |

4.4 per cent at 8 threads and 5.2 at 1, with generation a tie inside the spread
at both, as a bandwidth-bound path has to be. Both tiers have the batched kernel,
so it is like for like. 4 per cent is small, but it is the same sign in every
pass measured and it is free: the tier is 118 lines that the kernel tests hold
to the same bits as the AVX2 one.

That the comparison can be made at all is why `--tier` exists on the CLI. It
pins the tier by name for one run, the default stays the strongest the CPU
reports, and the gates are never run against a pinned tier.

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
(type, shape, tier) with the weight bandwidth and the speedup over scalar. It
measures one matvec at a time on one thread, which is a microbenchmark and not
the product's shape: the numbers that decide anything are in
`docs/benchmarks.md`, measured on the whole model at the real thread count.

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
## The forward pass

**`t` tokens at a time, and one token is `t = 1`.** M1 ran one token at a time
for the prompt as much as for generation, and reasoned that a batched path would
help only the eight full-attention blocks and the feed-forward halves and would
be a second code path through the twenty-four linear blocks. The first half of
that was wrong and the second half was avoidable, so M2 replaced it.

Wrong, because what a batch divides is the weight traffic, not the arithmetic:
batch-1 generation reads all 2.52 GiB of weights per token, and a block of
weight rows run against every token of a batch crosses the memory bus once for
the whole batch. That is the projections, the output matmuls and the output
head, which is 76 per cent of a token; it is not limited to the full-attention
blocks. Avoidable, because `forward_batch` *is* the forward pass and
`forward(token)` calls it with one token: every loop runs once, the pool sees
`t == 1` and uses the per-row kernel it always had, and the result is
bit-identical by construction rather than by a second test. Measured rather than
assumed: a batch of 1 reads a prompt at 13.9 tok/s against the 13.6 of the
commit before the batched path, inside the spread, and
`greedy_output_is_unchanged` holds.

Prompt tok/s went from 13.6 to 44.0 at 8 threads, 3.3 times generation.
`docs/benchmarks.md` has the per-commit pairs and the batch-size curve.

**What does not batch, and why that is the whole correctness story.** Two things
are sequential over tokens by construction:

- the depthwise convolution, whose tap `j` for token `i` is token
  `i - kernel + 1 + j`: inside a batch that is an earlier row of the same batch,
  and before the batch it is the history buffer;
- the delta-net recurrence, whose state for token `i` is the state token `i - 1`
  wrote.

Both run token by token inside the batch, over the same rolling state, in the
same order as the single-token path. So the state a batch leaves behind is the
state the single-token path would have left, and the hand-over to generation is
not a thing that has to be made to work, it is a thing that cannot be wrong
unless the loop order is. That is what the tests pin down: the convolution's
batch boundary against hand-computed taps, and the whole 248320-entry logit
vector after a batched prompt against the token-by-token one at five batch
sizes, including a prompt length that is not a multiple of the batch and a batch
larger than the prompt. The parallelism in both cases is over heads, each of
which owns its whole batch, so a batch is one pool job where it used to be `t`.

**Only the last token of a batch gets logits.** The output head is a single
2560 x 248320 matmul, 16 per cent of a single-token token, and a prompt throws
every position's logits away except the last. So `forward_batch` runs the final
norm and the head for the last row only, and a batch of 16 pays the head once.
The gate that feeds the reference's own tokens still sees every position,
because it feeds them one at a time.

**The batch is 16 tokens by default**, from the measured crossover in
`docs/benchmarks.md`: the curve rises steeply to 16 (13.75, 21.29, 30.68, 38.28,
44.26 tok/s at 1, 2, 4, 8, 16) and then by 8 per cent in total out to 64. It is
also exactly one group of the batched kernel, which is not a coincidence: past
one group the weight block's unpacking is already divided by 16 and only the
per-batch fixed costs are left to amortise. `--batch` moves it, and there is no
length below which batching is off, because the curve is monotone from 1: a
two-token prompt is already 55 per cent faster read as one batch.

**Nothing allocates per token in either path.** Every buffer is `max_batch` rows
of what one token needed, token-major, sized by `Runner::new`, and the
activation scratch holds `max_batch` rows of blocks. The allocation test covers
the batched prompt at three batch sizes next to the generation loop.

**Speculative decoding (M3) is what this path was shaped for**, and nothing for
it is built here. A drafted-token verification is exactly a batch: `n` drafted
tokens go through `forward_batch` in one pass over the weights, and what M3
needs that this does not have is the logits of *every* row rather than the last,
so that it can find the first position where the draft and the model disagree.
That is one more weight tensor's worth of work per batch (the head on `n` rows
instead of 1) and a second entry point next to `forward_batch`; the caches
already end up in the right state, because a batch leaves the state the tokens
would have left one at a time, and a rejected suffix is a `pos` the KV cache can
simply be rewound to for the full-attention blocks. The recurrent state cannot
be rewound, which is the one real design problem M3 has to solve: it will have
to keep a copy of the conv history and the delta-net state from before the
batch, 2.1 MiB per linear block, 50 MiB in total.

**Every buffer is allocated by `Runner::new`, from the config.** The pass
quantizes activations into scratch it owns, hands row ranges to the pool and
writes into buffers that already exist. Buffers are shared between the two
block kinds where their shapes allow it (`qkv` is sized for the larger of the
query-and-gate projection and the conv dimension, `heads` for the larger of the
attention output and the delta-net inner size), because there are only ever a
few live at once and sizing each one separately would cost tens of megabytes
for nothing.

**The arithmetic deliberately copies the reference's, step for step**, and not
in the way that is most accurate:

- RMSNorm forms the squares in f32, accumulates them in f64, rounds the mean
  back to f32 before adding `eps`, and takes an f32 root.
- The delta net's q/k normalisation is an L2 norm reached the way the reference
  reaches it: RMSNorm with `eps / n`, then a separate multiply by `1 / sqrt(n)`.
  Folding the two scalings into one is a different rounding.
- Softmax accumulates the normalising sum in f64 and rounds the reciprocal to
  f32 before applying it.

This looks like cargo-culting and is not. Quantizing activations to int8 is a
discontinuity: a difference of 1e-7 in one activation flips a quantization
level, and one flipped level moves every output of that matmul by about 1e-3.
Accumulating in a strictly more accurate way therefore costs the
token-for-token gate below, which is the thing of value here.

**Full-attention block.** GQA, 16 query heads over 4 KV heads, head dim 256.
`attn_q` is `[2560, 8192]` because the output gate is on: each head's slice
holds the query in its first 256 lanes and the gate in its second 256, so the
query is strided and the gate is read straight back out of the same buffer
after attention. Per-head RMSNorm on q and k comes before RoPE. RoPE covers
only the first 64 of the 256 head dimensions and pairs `i` with `i + 32`.

The file asks for mrope with sections `[11, 11, 10, 0]`. On a text-only path
all three position axes carry the same token position and the zero fourth
section leaves no dimension pair for the fourth axis, so plain partial RoPE is
exact. That is not something to take on trust, because it is silent when wrong:
a unit test replays the reference's interleaved sector-to-axis selection and
asserts no pair reaches the fourth axis.

**The causal mask is a loop bound, not a mask.** A batch takes the positions
`pos..pos + t`; its keys and values go into the cache before any score is
formed, and token `i` then attends over `pos + i + 1` positions. So there is
nothing to mask: the score loop stops where the token's own position is. A test
changes the last position's values and requires every earlier token's output not
to move.

**The walk over the cache is tiled over positions, with the queries inside the
tile.** One pool chunk is one head holding its whole batch, because the batch's
queries all read the same keys and values: per (token, head) chunks read a
head's span of the cache `t` times over, and at a context of 4096 that span is
8 MiB per KV head, which does not stay in the 36 MiB L3 next to everything else
a token touches. Tiled at 128 positions, a tile of keys is 128 KiB and every
query of the batch crosses it while it is in L2. Worth 9 per cent of a prompt
token at context 4096 and nothing measurable at 2048; for generation there is
one query per head, so no byte is read twice and the tiling has nothing to save
at any context. The numbers are in `docs/benchmarks.md`.

**The score rows stay materialised, which is the half of a flash-style
attention that is deliberately not here.** A running maximum and running sum
would let the rows go, and the rows are the only thing it would save: 16 KiB per
query at context 4096, read back by a loop that also reads 4 MiB of values from
memory, so under a per cent of an operator that is 20 per cent of a token at the
longest context in the budget. Against that, an incremental softmax rescales its
accumulator per tile, which is a different rounding of the operator the
correctness gate is most sensitive to. Keeping the rows makes the tiled walk
bit-identical to one query at a time, which the unit tests assert with `==` on
this model's GQA shapes, at a cache length and a causal bound either side of the
tile boundary, for a batch and for a single token. `docs/benchmarks.md` carries
the measurement and the projection for the contexts we cannot run.

**The output comes out head-major and the gate transposes it.** A head owns its
whole batch, so the natural layout of the per-head output is
`[head][token][head dim]`, and the output projection needs
`[token][head][head dim]`. The output gate's multiply is already a pass over
exactly those values, so it does the transpose on the way through and costs
nothing extra. The linear blocks' gated norm does the same with the delta net's
readout, and the two share the buffer.

**The embedding lookup is the only place a weight becomes f32.** The row it
wants is picked by a token id rather than dotted with anything, so it is the one
caller of `dequantize_row`; everything else meets the quantized bytes with int8
activations. That function knows five layouts and this model reaches one of
them, so all five are checked in `crates/llm-core/tests/kernels.rs` against the
independent scalar dequantizer the kernel tests already use, and the list it
accepts is pinned to the list the loader validates against: a type that passed
the loader's check and not the dequantizer's would be a panic in the middle of a
run instead of an error at load.

**Linear-attention block** (gated delta net). `attn_qkv` splits into q 16x128,
k 16x128 and v 32x128, two value heads per key head; a value head reads its key
head as `vh % n_k_heads`, which is the broadcast the reference does. The
depthwise causal convolution keeps `kernel - 1` samples per channel, oldest
first, and the state comes out holding the newest ones.

The recurrent state is stored transposed, as the reference stores it: row `j`
of a head's `[d_state, d_state]` state is `S[..][j]`. That is what lets one
pass over a row do the decay, the delta update and the readout for output
component `j`, with the row in cache once. `ssm_a` is already `-exp(A_log)` in
the file, so the per-head decay is `softplus(alpha + dt_bias) * a` and `exp` of
it lands in `(0, 1)` without a sign fix.

## Caches

**KV cache**, for the eight full-attention blocks only, laid out
`[block][k, v][kv head][position][head dim]`. Position-major within a head, so
a head's keys for the whole context are contiguous and the score loop walks
them forwards. At the default context of 4096 it is 256 MiB, which is a tenth
of the weights.

**Recurrent state**, for the twenty-four linear blocks: the convolution
history (`3 x 8192` f32) and the delta-net state (`32 x 128 x 128` f32), 2.1
MiB per block, 50 MiB in total and independent of the context length.

Both are allocated once by `Runner::new` and never reallocated. `reset` zeroes
them and sets the position back to 0; zeroing is the whole reset, because the
recurrence starts from a zero state, the convolution from zero history, and the
KV cache is only ever read up to the current position.

`resetting_the_caches_is_the_same_as_a_fresh_runner` tests that rather than
reasoning about it, and it compares **logits**, not tokens. Leftover state is
not a small effect in the logits, but the decay makes it a small effect in the
tokens: dropping the delta-net state from `reset` leaves the generated tokens
of a 24-token run completely unchanged, because the state is multiplied by a
factor below 1 at every step and a stale one has faded below what it takes to
move an argmax long before the first token comes out. Comparing tokens would
have passed that mutation; comparing the first token's logit vector catches it
on all 248320 entries.

The hand-over from prompt to generation is checked in the same test by feeding
the prompt plus all but the last generated token as one long prompt and
requiring it to predict that last token. M1 wrote that down as close to
tautological, both paths being the same `forward` over the same positions, and
as "the property a batched prompt path would break". It is a real check now:
`a_batched_prompt_is_the_same_as_one_token_at_a_time` reads the prompt in
batches of 2, 5, 7, exactly its length and more than its length, and requires
the whole 248320-entry logit vector to equal the token-by-token one and the
tokens generated afterwards to match as well. The logits alone would miss a
stale recurrent state that only the next token reads; the tokens alone would
miss a wrong state the decay fades out before an argmax moves, for the reason
above. The comparison is bit-exact rather than within a tolerance, because the
batched path is the same arithmetic in the same order and anything else is a
bug.

**The KV cache stays f32**, and that is now a measured decision rather than a
default. An f16 cache halves the bytes generation's attention reads, and that
operator is 20.1 per cent of a generation token at context 4096, 14.7 at 2048
and 1.5 at 1024, so the upper bound on the win is 10 per cent of a token at
4096 and nothing measurable at the default configurations. It is also a
rounding change in the attention scores, which is exactly what the correctness
gate measures, so taking it would move the fixture that records the four
permitted disagreements. That trade needs the board's acceptance rather than a
tuning commit, and `docs/benchmarks.md` carries the number for it. Note that the
reference's own default *is* an f16 cache, which is part of why its shipping
numbers are ahead of its matched-flags ones.

**Raising `--context` still costs nothing until the positions are used**, which
M1 reasoned about and this branch measured again on the batched path: a
16-token prompt peaks at 2.76 GiB resident at context 1024 and 2.75 GiB at
context 4096, the same number, because the cache's pages become resident as they
are first written. A 1024-token prompt at context 4096 is 180 MiB above a
192-token prompt at context 1024, which is exactly the KV positions the longer
prompt touches.

## Threading

**One persistent pool, built at load time**, with the submitting thread as one
of its workers. A job is published once, run by everyone, and `run` returns
only after every thread has left it; that is what lets a job borrow the
caller's stack and write into disjoint pieces of a caller-owned slice without
allocating. At 32 cores and a few hundred matmuls per token, spawning per
matmul would cost more than the work.

Two shapes of work cover the whole pass: matmul rows, and per-head chunks where
one attention head or one delta-net value head owns one output chunk and one
piece of scratch. Everything else is elementwise and cheaper than a job.

**Projections that read the same input are one job.** The feed-forward gate and
up, the full block's q, k and v, and the linear block's qkv together with its
output gate, alpha and beta each read the same normalised vector, so each group
is one job over one row range and one activation quantization instead of N jobs
over N quantizations of the same input. That took a token from 281 pool jobs to
161. It also balances better than separate jobs could: alpha and beta are 32
rows each next to tensors of thousands, so on their own they were a pair of
barriers around work one thread does in microseconds. The profile pays for it
by no longer being able to attribute time per tensor inside a group; the
per-tensor breakdown from before the change is in `docs/benchmarks.md`.

**A job is handed out through a sequence counter, not a barrier.** A
`std::sync::Barrier` is a mutex and a condvar, so the 161 jobs in a token cost
two of them each: a futex wake for every worker before the job and a futex wait
after it. Instead the submitter writes the job pointer, bumps a counter and
unparks the workers, and a worker waits for that counter to change by spinning
for about twenty microseconds before it parks. The completion side is a
countdown the workers release as they leave, which the submitter spins on and
then yields on. At 8 threads that took the pool's own share of a token from 67
per cent to 5.9 per cent (`docs/benchmarks.md`).

Twenty microseconds of spinning is deliberately shorter than a job and longer
than the serial stretch between two jobs (the activation quantization, the
norms, the elementwise work), which is what a worker is usually waiting
through. Past that it parks, because a spinning worker takes quota from the
thread it is waiting for. `unpark` on a thread that is still spinning is an
atomic store and no syscall, and it leaves a token that makes that thread's
next `park` return at once, which closes the window where a worker parks just
after being passed over.

**The work is claimed, not split up front.** Both shapes take their work from
an atomic counter: matvec groups in blocks of up to 64 rows, per-head jobs one
head at a time. A static split hands equal rows to unequal lanes here: the
cgroup quota means a thread can be descheduled in the middle of a job while the
others wait for it, and this CPU is 8 performance cores plus 16 efficiency
cores, so equal rows are not equal work even when nothing is descheduled. Of
the time the submitter used to spend at the two barriers, waiting for the
slowest worker was 23 ms per token against 6 ms for getting the workers
started, so the imbalance was the larger half. 64 rows of the largest tensor
here are 90 KiB of weights, which is tens of microseconds of streaming: long
enough that the `fetch_add` is noise and the weight reads stay sequential,
short enough that the last block does not decide the job's length.

**A panic in a job is carried out of the job, not out of `run`.** Both sides
run the job inside `catch_unwind`; the submitter resumes its panic only after
every worker has left, and a worker that panicked sets a flag the submitter
turns into a panic of its own. Letting the panic escape directly would leave
the other threads inside a job nobody waits for and `Drop` joining threads that
are still running, so a failed assert in the forward pass would be a hang
rather than a stack trace. The buffers are half written at that point, which is
why this resumes the panic instead of reporting it: the caller's results are not
usable either way.

**The split cannot change the result.** Each output row and each head is still
computed by exactly one worker from start to finish, so no reduction crosses a
thread and the output does not depend on the thread count: a block is handed out
once, and the rows of a group are one range for one claim. That it holds is now
a property of the claiming rather than of the arithmetic of a static split,
which is why it needs a test of its own: `every_chunk_is_claimed_exactly_once`
fails if a chunk is run twice or not at all at 1, 3, 4 and 8 threads, next to
`the_thread_count_does_not_change_the_output` on the whole forward pass at 1, 3
and all threads.

**Default thread count stays `available_parallelism`**, which on this box
reports 8 because the cgroup quota is 8 of the 32 cores. Before the three
changes above that default was 25 to 35 per cent off the peak, which was at 3
or 4 threads; it was left alone deliberately, pending a measurement. The
measurement, taken at the end of the threading work, says the curve now rises
to the quota instead of turning over inside it. Generation tok/s, context 1024,
19-token prompt, 16 generated tokens, median of 3, in the quietest window
measured (load average 5.0 to 6.2 on the 8-CPU quota):

| threads | 4 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- |
| tok/s | 11.15 | 14.02 | 14.73 | 15.23 |
| spread | 1.0% | 8.2% | 1.7% | 0.9% |

So 8 threads is the fastest setting, 37 per cent ahead of 4, where it used to be
the slowest. The measurement is repeated in `docs/benchmarks.md` with the other
windows, and they qualify it rather than contradict it: when this shared box is
oversubscribed, 6, 7 and 8 threads are indistinguishable (alternating passes of
8 and 6 give 10.04/9.99/8.89 against 10.73/9.70/9.93, spreads up to 54 per
cent), and in one window of four the peak was 6 threads. No window measured had
a setting below the quota that was faster than the quota by more than its own
spread.

That is why the default is not a fraction of the quota. The streaming
measurement shows memory saturating at 6 or 7 threads of the 8, so an argument
for a default of 6 or 7 exists on paper, but what makes 8 lose in some windows
is other work competing for the same bandwidth, and that is a property of this
machine at that moment, not of the hardware. `quota - 1` or `quota * 3/4` would
be fitting the product to our box's noise; `available_parallelism` reads the
quota it is given on any machine, and generation reaches 79 per cent of the
measured bandwidth ceiling there (against 29 per cent at the start of this
branch). `--threads` is there for the case where the box is busy.

AVX-512 was checked for and this machine does not have it: `/proc/cpuinfo` has
`avx2` and `avx_vnni` and no `avx512*` flag, so there is no AVX-512 tier in the
dispatch table. Adding one would be a kernel nobody here can run or test.

**Nothing in the generation loop allocates**, measured rather than asserted. A
counting global allocator wraps `System` in
`crates/rust-llm/tests/allocations.rs` and opens a window around the loop: 24
forward passes make 0 allocations, and `generate` around them makes exactly 2
whatever the token count, the output vector and the copy of the stop-token set,
both before the loop starts. Greedy and with every sampling stage on, which is
the same number: the sampler's buffers come out of `Sampler::new`, above the
window. It is one test and not two because the counter is global and a second
test in the harness's thread pool would be charged to the same window.

## Sampling

**One function with a settings argument.** `generate` takes a `&mut Sampler`;
greedy is `Sampling::GREEDY`, not a second code path. A greedy sampler is also
what the correctness gate passes, so the thing the gate measures is the thing
that ships.

**The pipeline, in this order**: repetition penalty, temperature, top-k, top-p,
min-p, the draw. The order is a choice, and it is the one that makes temperature
free: with it applied before the truncations, `exp((l - max) / t)` is the whole
of "divide by the temperature and softmax", so there is no pass over the
vocabulary to divide 248320 logits.

**`temperature == 0` is argmax**, the same `argmax` the gate ran against before
sampling existed, lowest token id on a tie. The penalty is the only stage that
could still touch the logits in that case, and `repeat_penalty == 1.0` skips it
entirely rather than multiplying by one, so a greedy run reads the forward
pass's own logits and is byte-identical to the committed fixture.

**No full sort.** The vocabulary is 248320 entries and the sampler runs once per
generated token, on a model that does about ten tokens a second. Measured on
this box, release build, per `sample` call over a 248320-entry vector:

| | per call |
| --- | --- |
| greedy (argmax only) | 0.26 ms |
| the defaults (top-k 20) | 0.65 ms |
| temperature, top-k 40, top-p, min-p and the penalty all on | 0.62 ms |
| `--top-k 0` with `--top-p 1`: no truncation, and no order needed | 1.58 ms |
| `--top-k 0` with top-p on: no truncation, and the order needed | 10.56 ms |
| for comparison: `sort_unstable_by` over the index vector on its own | 6.5 ms |

They wobble by a tenth of a millisecond between runs on this shared box, and the
shape rather than the digits is the point.

So at any `--top-k` of 1 or more the candidate set is grown by partial selection
and nothing sorts the logit vector: `select_nth_unstable_by` over a preallocated
index buffer puts the `k` largest logits in front in O(n) expected time, and only
those `k` are sorted, and only when top-p is on at all. The default is 20, which
is the 0.65 ms row, under one per cent of a 100 ms token.

**`--top-k 0` together with `--top-p` below 1 is the exception, and it does sort
the whole index vector.** `top_k = 0` means no truncation, so there is nothing to
select and the candidate set is all 248320 entries; top-p needs them in order, so
the sort runs over the lot. The two `--top-k 0` rows above are the same work
apart from that sort, so the sort is the 9 ms between them; the 1.58 ms the
cheaper of them costs is the exponential per entry that a draw over the whole
vocabulary needs whatever else happens. It is an accepted cost, not an oversight:

- growing the candidate set by repeated partial selection until the mass covers
  `top_p` has a worst case (a flat distribution, where 0.95 of the mass is
  thousands of candidates) of several O(n) passes plus a sort of what it found,
  and it needs a fallback threshold to stay bounded;
- that is more branches on the path every token takes, to make faster a setting
  nothing defaults to and which asks for a full-vocabulary draw in the first
  place.

`--top-k 0` with top-p off (`--top-p 1`) does not sort, which is what the gap
between the two rows is: the candidates are never ordered, because only top-p
uses the order.
`the_candidates_are_only_ordered_where_the_order_is_used` in
`crates/rust-llm/tests/sample.rs` holds that, since the selection leaves the
candidates unordered and a descending candidate set is therefore a sort that
ran.

**Why the selection is exact.** Both orders used are total: logit descending,
then token id ascending. Equal logits therefore do not leave the selected set
ambiguous, which they would with a bare comparison on an unstable algorithm, and
a naive full sort under the same order keeps exactly the same tokens.
`crates/rust-llm/tests/sample.rs` holds that against a reference that sorts the
whole vector and works in f64, per parameter and in combination. What it
requires exactly is the surviving set, token for token, because that is what
decides the draw; the probabilities are compared at 1e-5 relative, since the
product path takes its exponentials in f32 and accumulates in f64, as the
forward pass's own softmax does.

**top-p normalises over the top-k candidates**, not over the whole vocabulary.
The ticket fixes the order but not the denominator, and an exact
full-vocabulary one would put the exponential-per-entry pass of the 1.58 ms row
above on every token, whatever top-k left, and the sort with it: ranking the
whole vocabulary is the only way to find the prefix that covers the mass. This
is also what llama.cpp's default sampler chain does: its top-p
sampler softmaxes whatever its top-k sampler handed it. With `--top-k 0` the two
readings coincide, because the candidate set is then the whole vocabulary.

**min-p needs no normalisation at all.** The maximum is subtracted before the
exponential, so the largest unnormalised probability is exactly `exp(0)`, and
the threshold is `min_p` itself. One pass over the candidates, exact, and the
maximum always survives it. llama.cpp reaches the same filter from the other
side, as a logit threshold of `max + log(min_p)`.

**The repetition penalty iterates the window, never the vocabulary.** 64 recent
tokens by default. It follows llama.cpp: a positive logit is divided by the
penalty and a non-positive one multiplied, so a penalty above one only ever
lowers a logit, and it is applied once per distinct token in the window however
often that token occurs. Because it has to write, it works on a copy of the
logit vector, which is one 993 KB memcpy per token and only when the penalty is
on; everything downstream then has one mutable vector to read. Every token that
goes through the model enters the window exactly once, the prompt's included: a
penalty that only saw the generated tokens would not notice the prompt
repeating itself.

**Nothing in `sample` allocates.** Every buffer comes out of `Sampler::new`,
sized from the vocabulary: the candidate indices, the probability scratch and
the recent-token ring. The two that only the drawing path uses are sized to one
slot for a greedy sampler, and the penalty's copy is not allocated at all while
the penalty is off, which is the default.
`crates/rust-llm/tests/sample_allocations.rs` measures it with the same counting
global allocator as the generation-loop test, over a 248320-entry vector and for
every settings shape that reaches a different stage. It needs no weights, so it
runs in CI; it is its own test binary and holds one test, because the counter is
global and a second test in the harness's thread pool would be charged to the
same window.

**The RNG is xoshiro256++ seeded through splitmix64**, thirty lines in
`sample.rs` and no dependency. It is not cryptographic and does not need to be;
what it has to be is reproducible from a `u64` and well spread over the
candidate set. Without `--seed` the seed comes from the operating system's
entropy through `RandomState`, which is the standard library's own OS-seeded
hasher state, and the effective seed is printed to stderr: a run that cannot be
replayed is not reproducible, whatever the generator does.

**The defaults are Qwen3.5's own recommendation**, not llama.cpp's: temperature
0.6, top-k 20, top-p 0.95, min-p 0, repetition penalty 1 (off) over a window of
64. llama.cpp's defaults (0.8 / 40 / 0.95 / 0.05) are tuned for nothing in
particular; these are what the model's own card asks for in thinking mode, and
the model's guidance warns against a repetition penalty, so it is off and the
window size only matters once someone turns it on.

## The CLI

`rust-llm run --prompt "..."` renders the chat template, streams tokens as they
are produced and prints the timings and the peak RSS to stderr. `--model`,
`--tokenizer`, `--system`, `--context`, `--threads`, `--batch`, `--tier`,
`--max-tokens`, `--think/--no-think`, `--raw`, and the sampling flags `--temperature`,
`--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--repeat-last-n` and
`--seed`, with the defaults above shown in `USAGE`. `--batch` is the prompt's
batch size, defaulting to the measured crossover, and `--tier` pins the kernel
tier, which exists so that two tiers can be compared on the real model from one
binary. Arguments are parsed by hand: they are all `--name value`, there is no per-command grammar, and a
parser crate would be a dependency and a build-time cost for that shape.

**A value that cannot be a setting is an error that names the argument.** A
negative temperature, a `--top-p` above 1, a `--repeat-penalty` of 0: refused,
not clamped. A silently clamped `--top-p 1.5` looks like it worked, and whoever
typed it goes on believing their setting took effect.

`rust-llm chat` is the interactive surface: the conversation is a `Vec<Message>`,
`--system` sets the system turn, `/reset` starts over, `/exit` or the end of
stdin quits, and the sampling flags are the same. No line editing and no
history, so no readline dependency; `USAGE` says so, because the first thing
anyone tries is the up arrow.

**Every chat turn re-renders the whole conversation and resets the runner.** That
is not laziness. The chat template drops the reasoning of assistant turns before
the last user turn, so turn N+1's token prefix is not a prefix of turn N's: the
reasoning the previous turn generated is gone from the middle of the prompt and
everything after it has shifted. Reusing the KV cache across turns would not be
merely stale, it would be wrong. What that costs is re-reading the prompt,
which is now read in batches, so a turn pays about a third of what it used to
per token: batched prompt evaluation helps every surface at once rather than
this one.

A conversation that no longer fits the context is reported and the turn is
dropped, rather than generating into nothing: the runner would otherwise stop
with `ContextFull` after zero tokens and look like a model with nothing to say.

**The model path has no default.** `--model`, or `RUST_LLM_MODEL`, or an error
naming both: a path compiled in would be a path on whichever machine it was
written on. The tests that need the weights read the same variable and say so
when it is missing.

A streamed token is decoded on its own, so a character spread over two tokens
cannot be decoded from either and the decoder leaves a replacement character
where it was. The streaming callback records whether it saw one, and only then
is the whole output decoded again and printed after the stream. That is a
property of what was streamed, so it has to be recorded while streaming: the
clean text is decoded from all the ids at once and can never contain the
marker. Looking for the marker's first byte in the clean text instead both
missed every real case and fired on every character in U+F000..U+FFFF.

`rust-llm trace` runs the same pass with the activation tracer on and prints
one node per line, which is the input to the reference comparison below.

`rust-llm serve` is the fourth command and has a section of its own below.

## The HTTP surface

`rust-llm serve` answers `POST /v1/chat/completions` and `POST /v1/completions`,
streaming and not, in OpenAI's shapes, so anything that already speaks to an
OpenAI endpoint speaks to this.

**No HTTP crate and no async runtime.** There are two routes; a request is a
line, a few headers and a JSON body that `serde_json` already parses, and the
answer is a status line and a JSON body. What a framework would add on top of
that is a dependency tree, and the one rule this project has is that there is no
C or C++ anywhere in the build, which is a rule we have to keep checking rather
than hope for. An executor buys nothing either: the forward pass is blocking and
uses every core, so there is nothing to interleave it with. `crates/rust-llm/src/serve.rs`
is the whole server, and the request reading in it is about sixty lines.

**One request at a time, by having one accept loop.** Generation *is* the
`Runner` and its KV cache and there is one of each, so the loop accepts a
connection, answers it, and then accepts the next: a second client waits in the
listener's backlog rather than being refused, and no lock is needed anywhere to
say so. That is also all the hardware can do, since a single token already
saturates the cores. The cost is that a connection holds the slot while it is
being read, which is why a socket that goes quiet for 30 seconds is dropped, and
why the request line, the headers and the body are all capped: the other end of
a socket is not trusted to stop sending.

**The caches are reset between requests.** Requests are independent and the
previous one left its own conversation in the cache. A chat request also
re-renders its whole conversation, for the reason the chat section above gives:
the template drops the reasoning of earlier assistant turns, so turn N+1's
tokens are not a prefix of turn N's and the cache could not be reused even if
requests were related.

**The command line is the defaults.** Every option `run` takes is the default
for a request that does not send that field, and a request overrides the fields
it sends and keeps the rest. Honoured:

| field | |
| --- | --- |
| `messages` | the conversation, rendered through the model's chat template |
| `prompt` | `/v1/completions` only, and it goes to the model verbatim, as `--raw` does |
| `temperature`, `top_p`, `top_k`, `min_p` | the sampler, checked the way the flags are |
| `repetition_penalty` | `--repeat-penalty`; the name vLLM and Qwen clients send |
| `max_tokens` | `--max-tokens`, which is where the default of 512 comes from, capped at the context |
| `seed` | per request, so one answer can be replayed on its own |
| `stream` | `text/event-stream` instead of one object |
| `chat_template_kwargs.enable_thinking` | the thinking toggle, where Qwen clients put it |

`--seed` is the one default with a twist: given, every request without a seed of
its own replays that same seed; not given, each request draws a fresh one from
entropy and the server logs it, so any answer can still be reproduced.

**The context is the ceiling, and it is the one field that is capped rather than
refused.** `--context` bounds the prompt plus the answer together, so a
`max_tokens` above it cannot change what comes back: the run ends at
`ContextFull` at exactly the same token. Clients send whatever their own
configuration says without knowing this server's context, so refusing a request
this server can answer perfectly well would be an interop failure for nothing,
and the general rule about refusing rather than clamping exists for values that
*would* silently change the answer. It also keeps the number bounded: `generate`
reserves its output vector for the smaller of `max_tokens` and the context, and
a reservation sized by whatever a request asked for is a panic, or an abort
where the allocation is refused, and the accept loop is the main thread.

A prompt of `--context` tokens or more is the opposite case and is refused with
a 400 naming both numbers. Generating into it would answer 200 with an empty
completion, a `finish_reason` of `length` and no counts at all, which a client
cannot tell from a model that had nothing to say. `chat` already refuses the
same thing for the same reason, and `generate` now also reports the tokens that
did go through the model when it stops there, instead of zero.

**Everything else in the body is deliberately ignored**, which is the OpenAI
convention for unknown fields and the instruction this was built to. `n`,
`stop`, `logprobs`, `presence_penalty`, `frequency_penalty`, `tools`,
`stream_options` and the rest are read by nobody: a request asking for four
choices or an extra stop string gets one answer that ignores it, and the README
says so, because the alternative is a 400 on a field OpenAI clients send by
default. Two more that are not oversights: `repeat_last_n` has no OpenAI field
and stays at the command line's value, and the `model` a request names is
ignored because the answer reports the file that is loaded, which is what
actually answered.

**Streaming flushes per event.** Every token that decodes into something whole
is one `data:` event, flushed on its own, the last event carries the
`finish_reason` and no content, and the stream closes with `data: [DONE]`. A
token is held back until what is pending decodes without a replacement
character: a character spread over two tokens cannot be decoded from either of
them alone, and unlike the CLI, which reprints the whole answer afterwards, a
stream cannot take a chunk back.

**A client that drops mid-stream ends that request and nothing else.** The write
fails, the generation callback returns `false`, and the run stops with
`Stop::Cancelled`. That hook is why `generate`'s callback returns a `bool` at
all: otherwise a client hanging up after one token would cost every core up to
`--max-tokens` of work nobody will read, and the next request would wait in the
backlog for all of it.

**The error mapping**, every body in OpenAI's `{"error": {...}}` shape:

| | |
| --- | --- |
| 400 | a body that is not JSON or not an object, a sampler value that cannot be a setting, `max_tokens` of 0, a prompt of `--context` tokens or more, a `messages` array that is missing or malformed, a role this runner does not implement (`tool`, `function`), a system message that is not first, message content that is not a string, a missing `prompt` |
| 404 | any other path, with the two it does serve in the message |
| 405 | anything but `POST` on the two paths, with `Allow: POST` |
| 408 | the connection closed before a whole request arrived |
| 413 | `Content-Length` over a megabyte |
| 431 | request line and headers over 16 KB |
| 500 | only if what was generated cannot be decoded |

The 400s on the message shapes are not re-implemented here: `Role::from_str` and
`chat::render` already refuse a `tool` message and a misplaced system message,
because that is where a role from outside enters the program, and the server
turns their errors into the status instead of into a 500. Content that is not a
string is refused rather than flattened, for the reason `chat::Message` gives:
flattening a list of content parts would drop an image and change the prompt
without saying so.

**Loopback by default.** `--host 127.0.0.1 --port 8080`, and there is no
authentication: a server that answers whatever it is asked is not something to
bind to a network by default. `--port 0` takes whatever is free and the address
the listener actually got is printed, which is also how the test avoids racing
another run for a port number.

## The reference harness

llama.cpp on the **same GGUF file**, so quantization is out of the comparison.
It is a reference only: it is never linked into the product and does not appear
in the workspace's dependency tree. It is built in any directory outside this
repository, `$REF_DIR` below, and nothing in the repository holds that path:
the tests take it from `RUST_LLM_REFERENCE_BIN`. If the machine has no system
`cmake` and no root, as ours does not, a venv provides it:

```sh
python3 -m venv "$REF_DIR/venv"
"$REF_DIR/venv/bin/pip" install cmake
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$REF_DIR/llama.cpp"
cd "$REF_DIR/llama.cpp"
PATH="$REF_DIR/venv/bin:$PATH" cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON
PATH="$REF_DIR/venv/bin:$PATH" cmake --build build -j 32 \
  --target llama-completion llama-tokenize llama-bench llama-eval-callback
```

Binaries land in `build/bin` and need `LD_LIBRARY_PATH` pointing at that
directory. Two traps: at commit `3259ab6` the `llama-app` target fails to link
unless examples are enabled, so use `llama-completion`; and CMake caches
`LLAMA_BUILD_EXAMPLES`, so a second configure does not pick up a changed value
unless it is passed again.

**Per-layer comparison**, which is how a wrong block in a 32-block hybrid gets
found at all. Both implementations dump every node of one forward pass and
`scripts/compare-reference.py` walks them in graph order:

```sh
rust-llm trace --raw --prompt Hello > ours.txt
llama-eval-callback -m <model.gguf> -p Hello -n 1 -t 4 -c 512 \
  --no-repack -fa off -ctk f32 -ctv f32 > ref.log
scripts/compare-reference.py ref.log ours.txt
```

The reference flags are not optional. With its defaults it repacks the weights,
keeps an f16 KV cache and runs flash attention, none of which this runner does,
and the comparison then measures the difference between two optimisations.

Read the **first** block of the table, not the last. Through block 0 the two
agree in every digit the reference prints, with element sums within 1e-6. From
block 1 on they drift, reaching 1e-2 by block 3 and staying there, for the
reason given above: the first activation to land on the other side of an int8
rounding boundary moves a whole matmul by 1e-3, and it compounds. On this model
the first such flip is in `linear_attn_qkv_mixed-1`. The script is a bisection
tool, so look for the step out of a quiet neighbourhood, not for the absolute
value; whether the two implementations agree is settled by the gate, not by the
bottom of that table.

## The correctness gate

Ten prompts in `fixtures/gate_prompts.json`, 32 to 48 tokens each: thinking on
and off, a system persona, a multi-turn conversation, a non-ASCII prompt, Dutch
diacritics, two code prompts (one with tabs) and one raw completion with no
chat template. `fixtures/reference_generations.json` holds the reference's
tokens, ours, and the model's sha256, so CI needs neither llama.cpp nor the
weights.

**There are two gates rather than one, and the reason is not a concession.**
Free-running token-for-token agreement is not a property an implementation can
have unless its kernels are bit-identical to the reference's, which these
deliberately are not. On a logit scale whose next-token margins are typically
0.25 to 1.2, the two implementations' logits differ by about 1e-2, which leaves
a band of roughly +/-0.1 inside which neither one's choice is determined by the
model. llama.cpp disagrees with *itself* across its own two kernel paths on
these prompts for the same reason: with repacking on it answers "Here's a
thinking process that leads to the answer", with `--no-repack` "Here's my
thought process for solving this word problem", same file, same greedy sampler.

So:

- `every_next_token_matches_the_reference` is the correctness gate. It feeds
  the reference's own tokens and checks the prediction at every position
  independently, so a flipped near-tie cannot hide the predictions behind it. A
  disagreement is accepted only where the fixture records one and only when the
  top-two margin is inside the 0.25 band and the reference's token is at least
  our runner-up. Currently 380 of 384 positions are identical and the four that
  are not have margins of 0.0094, 0.0104, 0.0475 and 0.1262, against typical
  margins twenty times that.
- `greedy_output_is_unchanged` is the regression gate: free-running output must
  equal the committed sequence token for token. For 7 of the 10 prompts that
  sequence is the reference's own output for its whole length; the fixture
  records both and the test prints which.

Both run against the dispatched fast path (`avx2+vnni` on this box) and print
the tier. Neither is ever run against a pinned scalar build: a gate that passes
only on the slow path proves nothing about what ships.

`the_fixture_is_still_what_the_reference_produces` is the non-default test that
keeps the fixture from rotting. It re-derives a case against a local reference
build and compares, or writes it with `RUST_LLM_WRITE_FIXTURE=1`. One prompt
per command, selected by a substring of its name:

```sh
RUST_LLM_REFERENCE_BIN=$REF_DIR/llama.cpp/build/bin \
  RUST_LLM_GATE_ONLY=capital_no_think \
  cargo test -p rust-llm --release --test gate -- --ignored the_fixture_is_still
```

A case is merged into the fixture by name as soon as it is produced and the
file is rewritten in prompt-set order, so a command that is stopped part-way
keeps its work and the next one continues with what is missing. A case that is
in neither the fresh run nor the file is left out, and the two gates then fail
on the case count rather than quietly running a short set.

Omitting `RUST_LLM_GATE_ONLY` does the whole set in one command, which is about
four minutes. Per prompt it is 20 to 35 seconds, which is the size to keep a
check at.

It reads the reference's ids out of its verbose log, because `llama-completion`
prints a generated token's text and never its id: a token's bytes are exactly
the text between one log line and the `n_past` line after it. Inverting text
back to ids needs the whole vocabulary decoded once, and the inversion is
asserted to be a function rather than assumed. Tokens that are half a
multi-byte character cannot be inverted this way and would fail by name; none
of the committed prompts produces one. One extra token is always asked for and
dropped, so every token the gate uses is bounded by a log line on both sides.

## The tiny model, so the surfaces are covered without the weights

The gate above proves what the model produces, and needs 2.7 GB of weights to
do it. What it does not touch is the product around the model: argument parsing,
streaming, the stop conditions, the sampling flags, the conversation. That was
only ever exercised by hand.

`crates/rust-llm/tests/support/mod.rs` writes a minimal but valid Qwen3.5 GGUF,
178 KB, in milliseconds: two blocks, one linear and one full attention, `Q8_0`
weights so the real dequant-free kernels run, and the `tokenizer.json` that goes
with it. `crates/rust-llm/tests/tiny_model.rs` drives the real binary against
it through `CARGO_BIN_EXE_rust-llm`, and `crates/rust-llm/tests/serve.rs` starts
the real server against it the same way. Neither needs weights, so both run in
CI.

**It proves the surface, not the output.** The weights are noise from a fixed
seed, so the text is meaningless, but it is produced by the real loader, the
real kernels, the real chat template and the real sampler, and it is identical
on every machine and every run. That is what lets the tests compare streamed
text directly: a three-turn chat session's output has to equal the two-turn
session's followed by the one-turn session's, which is what says `/reset` really
cleared the conversation.

Three things about the fixture are not free choices.

- **`n_embd` is 256, not 32.** The embedding table is read by the embedding
  lookup, which needs `can_dequantize`, and dotted by the tied output head,
  which needs a `vec_dot` kernel. The only types with both are the k-quants, so
  the table is `Q6_K` (which is what the real `Q4_K_M` conversion uses for it)
  and its row length has to be a whole 256-value super-block. Every other
  quantized row length is a multiple of 32, which is `Q8_0`'s block. Nothing in
  `forward.rs` or the loader had to change for the small shapes; the only
  constraints that bit were these block widths.
- **The tokenizer covers printable ASCII, tab and newline, not all 256 bytes.**
  With the full byte alphabet most tokens the fixture generates are half a
  multi-byte character, the streaming path correctly reprints the whole answer,
  and every comparison in the test has to work around a doubled output. With 97
  byte tokens every token is one valid UTF-8 byte. The reprint path stays
  covered by the real model, where it is a real case.
- **The end-of-sequence token is a parameter of the writer.** What this model
  produces is noise, so a test that wants to see the end-of-sequence path asks
  the library which token a greedy run produces first and then writes a fixture
  that stops on exactly that token. Nothing is hard-coded, and the test cannot
  rot into asserting that generation stopped for some other reason.

Every run in that file is either greedy or carries an explicit `--seed`. The
fixture's end-of-turn tokens are inside the vocabulary it can produce, so a run
left on the default settings with a seed from entropy stops on one now and
then; the first version of these tests failed about one run in six that way.

One property the fixture cannot show is the streaming flush. It generates
microseconds per token, so every event is in the socket buffer before the
client's first `read` returns, and a server that built the whole answer and
wrote it once would produce the same bytes in the same number of reads. So the
serve test asserts the stream's shape instead: a whole event in hand before the
stream ends, several events rather than one, the `finish_reason` only on the
last, the closing `data: [DONE]`, and no `Content-Length`. The flush itself is
one call per event, and the real model, at ten tokens a second, is where it is
visible.

## Baseline numbers

These are M1's numbers, kept because the decisions above rest on them and
because they are what the throughput work measures itself against.
**`docs/benchmarks.md` holds the current ones**, the bandwidth ceiling, the
per-operator profile and the reference at matching thread counts, and it is the
only place new performance numbers go.

Qwen3.5-4B-heretic-Q4_K_M, `avx2+vnni`, this box (8 of 32 cores by cgroup
quota, and shared with other work, so these wobble by 10 to 20 per cent):

| | prompt tok/s | generation tok/s |
| --- | --- | --- |
| default context 4096, 384-token prompt, 8 threads | 6.68 | 3.90 |
| default context 4096, 384-token prompt, 4 threads | 7.37 | 8.12 |
| context 1024, 19-token prompt, 4 threads | 8.34 | 9.66 |

Peak RSS at the default context of 4096 is **2.78 GiB** for the 384-token run
above, and 2.75 GiB for a 13-token prompt. Of that, 2.52 GiB is the mapped
model file, read through the mmap and resident because the output head reads
the whole embedding table every token; about 190 MiB is the tokenizer's
in-memory vocabulary (248044 entries and 247587 merges); 50 MiB is the
recurrent state, which is written every token and therefore fully resident.

The KV cache is the part that is *not* in there. `vec![0.0; n]` of that size
comes from `mmap`, so its pages become resident as they are first written: the
same run at a context of 256 instead of 4096 peaks 1.4 MiB lower, and the whole
difference between the 13-token and the 384-token run is the 25 MiB of cache
those extra positions touch. Raising `--context` therefore costs nothing until
the positions are used.

Prompt processing was no faster per token than generation in those rows,
because the prompt went through the same single-token path. That is fixed: it is
3.3 times generation at 8 threads now, and the numbers are in
`docs/benchmarks.md`. The peak RSS numbers above still hold to within 10 MiB,
because the batched path's extra scratch is a few megabytes against 2.5
gibibytes of weights.

**The reference, measured properly, is in the same range**, and an earlier
figure saying otherwise was wrong. `llama-completion` on the same file and the
same box, same prompt, 8 generated tokens:

| threads | 4 | 8 | 32 |
| --- | --- | --- | --- |
| reference generation tok/s | 2.89 | 8.12 | 0.10 |
| this runner, same prompt | 9.99 | 7.08 | |

The 0.10 at 32 threads is where "llama.cpp does 0.20 tok/s on this
architecture" came from. It is not a property of llama.cpp or of this
architecture, it is four times oversubscription against an eight-core cgroup
quota, the same barrier effect that cost this runner half its speed at eight
threads. Best against best, this runner was about 23 per cent ahead; at the same
eight threads the reference was about 15 per cent ahead of it.

So the reference is not a floor to walk over. The bar is memory bandwidth, which
batch-1 generation hits by reading every weight once per token. That bar is
measured now, at about 48.6 GiB/s and so 19 generation tok/s on this box, and
`docs/benchmarks.md` has the comparison against the reference per thread count
from `llama-bench` rather than from one `llama-completion` run: generation is
within 11 per cent of it at every thread count from 1 to 8 with the flags
matched and ahead at 6, 7 and 8, and the prompt is a factor 2.1 behind, down
from the factor 4 to 7 it was before the prompt was batched. What is left on the
prompt side is a kernel shape, not a schedule: the reference turns a batch into
a true matrix-matrix multiply over a repacked weight layout.

## What the checks cost

Every check is sized to finish in one command, and none of them is a wait with
no signal in it. Measured on this box:

| command | time |
| --- | --- |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` | 1 s, 25 s cold |
| `cargo test --workspace --all-targets` (the default set, no weights needed) | 47 s cold, 25 s warm |
| of which `tests/sample.rs`, `tests/sample_allocations.rs` and `tests/tiny_model.rs` | 2 s, 2 s, 3 s |
| of which `tests/serve.rs`, which starts the server and drives both endpoints | 2 s |
| the whole `--ignored` set, which is both gates and everything else needing weights | 3:30 |
| gate: `every_next_token_matches_the_reference` | 2:01 |
| gate: `greedy_output_is_unchanged` | 2:10 |
| gate: `resetting_the_caches_is_the_same_as_a_fresh_runner` | 27 s |
| gate: `the_thread_count_does_not_change_the_output` | 21 s |
| gate: `a_batched_prompt_is_the_same_as_one_token_at_a_time` | 17 s |
| `the_generation_loop_allocates_nothing_per_token` | 12 s |
| `rust-llm bench` at one thread count, 192-token prompt, 3 repeats | 25 s to 1:30 |
| `rust-llm bench --threads 1,2,3,4,5,6,7,8`, the whole sweep | 5:07 |
| `rust-llm profile`, 192-token prompt | 30 s |
| `rust-llm profile`, 2048-token prompt (long-context attention) | 1:41 |
| `rust-llm profile`, 4096-token prompt | 3:52 |
| `llama-bench -p 192 -n 16 -t 1,2,3,4,5,6,7,8 -r 3`, one flag set | 2:44 |
| fixture re-derivation, one prompt | 20 to 35 s |
| fixture re-derivation, all ten prompts | about 4 min |
| per-layer trace comparison against the reference | 40 s |

The per-test times are from running them one at a time. In one command the
harness runs four of them at once, each asking for all 8 threads of the quota,
so they finish together in about the time of the slowest rather than in the sum;
the whole set took 3:30 on a box that was busy with other work at the time, 210
seconds with the batched-prompt test added to it on a quiet one, and the
threading changes on the throughput branch took it from 232 to 140 seconds. Three of its tok/s figures are therefore a quarter of what
`bench` reports, which is the oversubscription and not a regression: the gates
measure agreement, `docs/benchmarks.md` measures speed.

The measurement rows are there because the budget rule is per step: nothing in
that list is a wait with no signal in it, and the two longest ones are the price
of a prompt that fills the context, which is the only way attention shows up at
all. A configuration that does not fit in ten minutes is measured smaller and
the file says what the run was.

The default `cargo test` holds no weights and no reference, so it is the same
35 seconds in CI. The gate tests are `#[ignore]`d because they need the model
file, not because they are slow, so nothing had to be left out of the set to
keep a command short.
