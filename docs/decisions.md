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
## The forward pass

**One token at a time, for the prompt as much as for generation.** The
delta-net recurrence is sequential over tokens, so a batched prompt path would
only help the eight full-attention blocks and the feed-forward halves, and it
would be a second code path through the twenty-four linear blocks. One path
also makes the hand-over free: the state generation starts from is, by
construction, the state the prompt produced, because it is the same buffer and
nothing re-reads the prompt. Batched prompt evaluation belongs with the rest of
the throughput work.

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
requiring it to predict that last token. Today that is close to tautological,
since both paths are the same `forward` over the same positions, and it is
written down as such: it is the property a batched prompt path would break, and
it is free to keep nailed down until then. What checks the hand-over against
something outside this runner is the next-token gate below, which feeds the
reference's tokens.

## Threading

**One persistent pool, built at load time**, with the submitting thread as one
of its workers. A job is published once, run by everyone, and `run` returns
only after every thread has passed the second barrier; that is what lets a job
borrow the caller's stack and write into disjoint pieces of a caller-owned
slice without allocating. At 32 cores and a few hundred matmuls per token,
spawning per matmul would cost more than the work.

Two shapes of work cover the whole pass: matmul rows split over the threads,
and per-head chunks where one attention head or one delta-net value head owns
one output chunk and one piece of scratch. Everything else is elementwise and
cheaper than a barrier.

**The split cannot change the result.** Each output row and each head is
computed by exactly one worker from start to finish, so no reduction crosses a
thread and the output does not depend on the thread count. A test asserts that
for 1, 3 and all threads.

**Default thread count is `available_parallelism`**, which on this box reports
8 because the cgroup quota is 8 of the 32 cores. That default is not the
fastest setting here. Measured, short prompt, context 1024, generation tok/s:

| threads | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tok/s | 4.83 | 8.29 | 9.56 | 9.66 | 9.17 | 8.59 | 7.09 |

Batch-1 generation reads all 2.52 GiB of weights per token, so the ceiling is
memory bandwidth and it is reached at three or four threads; past that the
barriers cost more than the extra lanes earn, and on a shared box a thread that
is descheduled at a barrier holds up all of them. Tuning this, and finding what
the bandwidth ceiling actually is, is the throughput ticket's job; the default
stays the honest one.

**Nothing in the generation loop allocates**, measured rather than asserted. A
counting global allocator wraps `System` in
`crates/rust-llm/tests/allocations.rs` and opens a window around the loop: 24
forward passes make 0 allocations, and `generate` around them makes exactly 2
whatever the token count, the output vector and the copy of the stop-token set,
both before the loop starts. It is one test and not two because the counter is
global and a second test in the harness's thread pool would be charged to the
same window.

## Sampling and the CLI

Greedy argmax and nothing else; a temperature, top-k or top-p surface is a
separate change, and putting the hooks in now would be an interface with one
implementation. Ties go to the lowest token id, which matters only for
reproducibility.

`rust-llm run --prompt "..."` renders the chat template, streams tokens as they
are produced and prints the timings and the peak RSS to stderr. `--model`,
`--tokenizer`, `--system`, `--context`, `--threads`, `--max-tokens`,
`--think/--no-think` and `--raw`. Arguments are parsed by hand: there are ten,
they are all `--name value`, and a parser crate would be a dependency and a
build-time cost for that shape.

A streamed token is decoded on its own, so a multi-byte character split across
two tokens comes out as a replacement character. When that happens the whole
output is decoded again and printed after the stream, rather than buffering
every token and streaming nothing.

**The model path has no default.** `--model`, or `RUST_LLM_MODEL`, or an error
naming both: a path compiled in would be a path on whichever machine it was
written on. The tests that need the weights read the same variable and say so
when it is missing.

`rust-llm trace` runs the same pass with the activation tracer on and prints
one node per line, which is the input to the reference comparison below.

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

## Baseline numbers

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

Prompt processing is no faster per token than generation, because the prompt
goes through the same single-token path. That is the first thing to fix for
throughput, not a measurement error.

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
quota, the same barrier effect that costs this runner half its speed at eight
threads. Best against best, this runner is about 23 per cent ahead; at the same
eight threads the reference is about 15 per cent ahead of it.

So the reference is not a floor to walk over. The bar is memory bandwidth,
which batch-1 generation hits by reading every weight once per token, and
establishing where that bar actually is belongs to the throughput ticket.

## What the checks cost

Every check is sized to finish in one command, and none of them is a wait with
no signal in it. Measured on this box:

| command | time |
| --- | --- |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` | 1 s, 25 s cold |
| `cargo test --release` (the default set, no weights needed) | 35 s cold, 2 s warm |
| gate: `every_next_token_matches_the_reference` | 2:01 |
| gate: `greedy_output_is_unchanged` | 2:10 |
| gate: `resetting_the_caches_is_the_same_as_a_fresh_runner` | 27 s |
| gate: `the_thread_count_does_not_change_the_output` | 21 s |
| `the_generation_loop_allocates_nothing_per_token` | 14 s |
| fixture re-derivation, one prompt | 20 to 35 s |
| fixture re-derivation, all ten prompts | about 4 min |
| per-layer trace comparison against the reference | 40 s |

The default `cargo test` holds no weights and no reference, so it is the same
35 seconds in CI. The gate tests are `#[ignore]`d because they need the model
file, not because they are slow, so nothing had to be left out of the set to
keep a command short.
