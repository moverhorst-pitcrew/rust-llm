# Benchmarks

Every performance number about this runner lives here. `docs/decisions.md`
quotes the ones a decision rests on and points back at this file; it does not
keep a second copy.

Read the conditions on every row. This box is shared with other work, and the
same command half an hour apart differs by up to 30 per cent at the higher
thread counts. That is why `bench` prints the load average before and after
each configuration, why every table below says what the load was, and why a
before/after comparison in this repository is always a pair of runs alternated
in one window rather than a number from another session.

## The machine

| | |
| --- | --- |
| CPU | 13th-gen Core i9, 8 performance cores + 16 efficiency cores |
| as WSL2 reports it | 16 cores x 2 threads = 32 CPUs, one NUMA node |
| cache | 2 MiB L2 per core, 36 MiB shared L3 |
| ISA | AVX2 + `avx_vnni`; **no AVX-512 flag**, so the dispatched tier is `avx2+vnni` |
| cgroup quota | `/sys/fs/cgroup/cpu.max` = `800000/100000`, so 8 CPUs of the 32 |
| `available_parallelism` | 8, because of that quota |
| model | `Qwen3.5-4B-heretic-Q4_K_M.gguf`, 2.51 GiB on disk, 2.52 GiB of weights read per token |

A thread count only means something next to that quota: "8 threads" here is 8
of the 8 CPUs the cgroup allows, not 8 of 32. Which physical cores the quota
lands on is the host scheduler's choice, and a performance core and an
efficiency core are not the same lane, which is why the pool hands work out
dynamically rather than splitting it equally up front.

## The commands

```sh
export RUST_LLM_MODEL=<path>/Qwen3.5-4B-heretic-Q4_K_M.gguf

# throughput, median of repeats with the spread, per thread count
rust-llm bench --context 1024 --threads 1,2,3,4,5,6,7,8 --prompt-tokens 192 \
  --max-tokens 16 --repeats 3

# the batch-size curve, one invocation per batch size
rust-llm bench --context 1024 --threads 8 --prompt-tokens 192 \
  --max-tokens 4 --repeats 2 --batch <n>

# the long-context configuration
rust-llm bench --context 4096 --threads 8,4 --prompt-tokens 1024 \
  --max-tokens 16 --repeats 2

# where a token's time goes, one table per phase
rust-llm profile --context 1024 --threads 8 --prompt-tokens 192 --max-tokens 8

# the two kernel tiers against each other, from one binary
rust-llm bench --context 1024 --threads 8 --prompt-tokens 192 --max-tokens 8 \
  --repeats 2 --batch 16 --tier avx2+vnni     # and --tier avx2+fma

# what the machine can stream, which is the ceiling batch-1 generation hits
cargo run --release -p llm-core --example bench_bandwidth
```

**One model run at a time.** A second runner on this box streams the same
2.52 GiB of weights, so two at once are not two measurements but two wrong
ones: a `--ignored` set left to the test harness's own parallelism was measured
at 40 threads and 17 GiB resident inside a quota of 8 CPUs. Every number in this
file is taken with nothing else loading the model, and the `--ignored` tests are
run with `--test-threads=1`, which is also the faster way to run them: the six
tests of `tests/gate.rs` take 4:33 in sequence against 6:38 all at once. The load
average printed before and after each configuration is what says whether
anything else was on the box at all.

The first run of each configuration is a warm-up and is not counted. The prompt
is a fixed text in the binary truncated to exactly `--prompt-tokens` tokens,
repeated with a space between copies when more tokens are asked for than its
204, and generation runs exactly `--max-tokens` tokens through the forward pass
and ignores stop tokens, so two runs measure the same work whatever the model
decides to say.

`--prompt-tokens 192` rather than M1's 19: a prompt has to be several batches
long for the batched path's numbers to mean anything, and 192 is twelve batches
of the default 16.

## The ceiling: what this box can stream

Batch-1 generation reads all 2.52 GiB of weights for every token, so the ceiling
is memory bandwidth. `bench_bandwidth` streams a 1 GiB buffer (28 times the
36 MiB L3), each thread over its own contiguous region, in a 2-second window,
median of 3. The implied rate is the bandwidth divided by 2.52 GiB.

| threads | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GiB/s, load 0.7 -> 3.7 | 15.77 | 28.55 | 36.68 | 41.91 | 44.84 | 47.31 | 48.63 | 48.37 |
| GiB/s, load 3.7 -> 5.4 | 14.98 | 28.22 | 36.50 | 40.98 | 43.38 | 45.29 | 48.92 | 44.25 |
| implied generation tok/s | 6.26 | 11.33 | 14.56 | 16.63 | 17.79 | 18.77 | 19.30 | 19.19 |

The spread within a run was under 2 per cent at every thread count on the quiet
box. The quieter run is the ceiling used below; the busier one is there to show
what contention does to the same measurement, and that it hits the 8-thread
column hardest.

Memory saturates at about 48.6 GiB/s, which it reaches at 6 or 7 threads of the
8 the quota allows. So the memory system is full before the CPU quota is, and
above 6 threads a lane adds no bandwidth.

These numbers are a slight **under**estimate, by 1 to 3 per cent. A thread
checks the deadline between whole passes over its region, so it stops up to one
pass early and then idles while the others finish theirs, while the elapsed time
counted is the whole scope's. A pass is 68 ms of the 2-second window at 1 thread
and 23 ms at 8. A low ceiling makes the "fraction of the ceiling" rows below
read high, so they are flattering by about that much and not conservative.
Taking the elapsed time per thread instead of per scope would remove it; it was
left as it is, because the correction is smaller than the window-to-window
spread of everything it would be applied to.

**The ceiling is for generation only.** A prompt read in batches of 16 reads the
weights once per batch, so 157 MiB per token, and at the 47.16 prompt tok/s
below that is 7.4 GiB/s: 15 per cent of the same ceiling. The prompt is not
bounded by memory any more, which is the whole point of batching it, and a
"fraction of the ceiling" column for the prompt would say nothing.

## The runner against that ceiling

Context 1024, 192-token prompt, batch 16, 16 generated tokens, tier
`avx2+vnni`, quota 8 CPUs, 3 repeats, with no other run touching the model
(load 3.00 -> 6.17, most of it this bench's own threads):

| threads | prompt tok/s | spread | generation tok/s | spread | ceiling | fraction | prompt / generation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.55 | 1.6% | 3.66 | 1.4% | 6.26 | 58% | 2.9 |
| 2 | 18.85 | 0.3% | 6.66 | 1.4% | 11.33 | 59% | 2.8 |
| 3 | 25.36 | 0.8% | 8.80 | 0.9% | 14.56 | 60% | 2.9 |
| 4 | 31.43 | 0.7% | 10.57 | 1.4% | 16.63 | 64% | 3.0 |
| 5 | 36.12 | 0.5% | 12.14 | 3.5% | 17.79 | 68% | 3.0 |
| 6 | 40.36 | 3.9% | 13.22 | 3.3% | 18.77 | 70% | 3.1 |
| 7 | 43.88 | 1.7% | 13.46 | 4.7% | 19.30 | 70% | 3.3 |
| 8 | 47.16 | 7.1% | 14.87 | 3.0% | 19.19 | **77%** | **3.2** |

Peak RSS 2.82 GiB for the whole bench process, which loads the model once and
runs every configuration in it. Per configuration, from a single `rust-llm run`
with a 16-token prompt: 2.76 GiB at context 1024 and 2.75 GiB at context 4096,
the same number, because a 16-token prompt touches 32 KV positions whatever the
context was allocated for. See "Memory" below.

Honest notes on that table:

- **The prompt is 2.8 to 2.9 times generation at 1 to 3 threads and 3.0 to 3.3
  from 4 threads up**,
  and the reason it is not a flat factor is that the two phases are bounded by
  different things. Generation is bandwidth-bound only once there are enough
  threads to saturate memory: at 1 thread it reaches 57 per cent of the implied
  ceiling, so it is compute-bound as well, and batching has less waste to
  remove. The prompt keeps scaling with cores because it is compute-bound at
  every thread count.
- **What caps the ratio is the part of a token that does not batch.** From the
  profile below, of a 20.06 ms prompt token the delta-net recurrence is 10.0 per
  cent, the elementwise work 5.9, the convolution 4.1 and the norms 2.5: 22.5
  per cent that costs the same per token whether 1 token or 16 go through
  together. With that fixed and the matmuls divided by the batch, 3.2 at 8
  threads is close to what is available; the next factor would have to come out
  of the matmul kernel, not out of the batch.
- Generation reaches 77 per cent of the measured ceiling at 8 threads. The
  remaining fifth is the output head and everything per token that is not
  streaming weights; it is not barrier time, which is 3.8 per cent (see the
  profile).
- The spreads are under 5 per cent except the 8-thread prompt row at 7.1,
  which is the setting most exposed to whatever else runs on the box. Earlier
  windows gave 12 per cent there. Compare rows within a window, never across.

Where this started. On the first commit of this branch, same configuration
before any threading or batching change: 7.75 generation tok/s at 4 threads and
5.52 at 8, with the prompt equal to generation to within the spread because it
went through the same single-token path.

## What batched prompt evaluation moved

Three changes, each measured against the previous commit's binary, alternated in
one window. Prompt tok/s, context 1024, 192-token prompt, 8 threads, batch 32
for the first two rows. These pairs predate the one-run-at-a-time rule above, so
the absolute level in them is lower than the table at the top of this file by
the few per cent that another agent's work on the box costs; what each row says
is the ratio inside the pair, which the alternation is what makes sound.

| commit | change | before | after |
| --- | --- | --- | --- |
| `c0fd536` | the prompt goes through as a batch of tokens | 13.6 | 31.6 |
| `408dacf` | a weight block is unpacked once per batch, not once per token | 31.6 | 44.0 |
| `1828971` | the walk over the KV cache is tiled, once per batch | see "Attention" |

Generation is untouched by all three: 13.4 before, 13.3 after, and batch 1
measures 13.9 against the 13.6 of the commit before it, both inside the spread.
A batch of one goes down the same code path with every loop running once, so
that is what it should do.

### The batch-size curve, and why the default is 16

Context 1024, 192-token prompt, 8 threads, 2 repeats, one invocation per batch
size, load 3.60 -> 8.27 across the series, and from before the
one-run-at-a-time rule: the shape of the curve is what this is for, and the
knee does not move with the absolute level. Prompt tok/s:

| batch | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tok/s | 13.75 | 21.29 | 30.68 | 38.28 | 41.76 | 44.26 | 44.58 | 46.50 | 46.67 | 47.92 |
| gain over the previous | | +55% | +44% | +25% | +9% | +6% | +1% | +4% | +0% | +3% |

The knee is at 16, and the curve is monotone from 1, so there is no prompt
length below which batching should be turned off: a two-token prompt is already
55 per cent faster read as one batch. The 8 per cent still available between 16
and 64 is the per-batch fixed costs amortising further (the output head, which
runs once per batch, and the 161 pool jobs), and it costs four times the batch
scratch, of which the score buffer alone is batch times context times 16 floats.

16 is also exactly one group of the batched kernel, which is not a coincidence:
past one group the weight block's unpacking is already divided by 16 and only
the fixed costs are left to amortise.

## The long-context configuration

Context 4096, 1024-token prompt, batch 16, 16 generated tokens, 2 repeats,
load 6.37 -> 7.42:

| threads | prompt tok/s | spread | generation tok/s | spread |
| --- | --- | --- | --- | --- |
| 8 | 45.32 | 0.2% | 13.78 | 9.2% |
| 4 | 30.53 | 0.1% | 10.34 | 0.5% |

Peak RSS 3.00 GiB for the process, both configurations.

The prompt is the same 45.3 tok/s as at context 1024, to two decimal places,
because the work is the same: a 1024-token prompt touches 1024 positions
whatever the context was allocated for. Generation is the same too at these
lengths. What *does* move with the context is attention once the positions are
actually used, which is the next section.

## Attention, and what tiling the KV walk bought

Attention is invisible at short context and the largest single operator at long
context. Per-operator, 8 threads, batch 16, the prompt length equal to the
context so the positions are really used:

| context | prompt: attention | prompt: token | generation: attention | generation: token |
| --- | --- | --- | --- | --- |
| 1024 | 0.27 ms (1.4%) | 20.06 ms | 0.97 ms (1.5%) | 64.50 ms |
| 2048 | 2.92 ms (11.9%) | 24.49 ms | 12.50 ms (14.7%) | 85.11 ms |
| 4096 | 6.49 ms (22.8%) | 28.43 ms | 18.09 ms (20.1%) | 89.83 ms |

The 2048 and 4096 rows are from the commit before `1828971`, so they are the
"before" of the pair below. Generation's attention cost is the KV cache's bytes
and nothing else: at context 4096 one token reads 16 query heads times 4096
positions times 256 lanes times 4 bytes, for keys and for values, over the eight
full-attention blocks, which is 1.07 GB and 22 ms at the measured ceiling
against the 18 to 19 ms measured.

**Tiling the walk over KV positions landed** (`1828971`). A batch's queries read
the same keys and values, and the job used to be one chunk per (token, head), so
a head's span of the cache was read `t` times over. At context 4096 that span is
8 MiB per KV head, 32 MiB for the four of them, which does not stay in the
36 MiB L3 next to everything else a token touches. The chunk is now one head
holding its whole batch, tiled over 128 positions with the queries inside the
tile. Against the previous commit's binary, alternated in one window
(load 5.41 -> 8.61):

| | attention | prompt token | prompt tok/s |
| --- | --- | --- | --- |
| context 4096, 4096-token prompt, before | 6.49 ms | 28.43 ms | |
| context 4096, 4096-token prompt, after | 5.20 ms | 26.00 ms | +9% |
| context 2048, 2048-token prompt, before | 2.92 ms | 24.49 ms | |
| context 2048, 2048-token prompt, after | 2.80 ms | 24.79 ms | inside the noise |

So it pays from about context 4096 upwards and not below: up to 2048 a head's
span still fits the L3 and there is nothing to win. Generation has one query per
head, so no byte is read twice and tiling has nothing to save at any context;
its numbers across the pair move only with the box (89.83 against 84.35 ms per
token, in a window whose load went from 5.4 to 8.6).

### Flash-style attention, measured and not kept

The other half of a flash-style attention is the running maximum and running
sum that let the score row go unmaterialised. That is **not** here, and the
measurement is why.

The score row is the only thing it saves. At context 4096 a row is 4096 floats,
16 KiB per query, and the loop that reads it back also reads 4 MiB of values
from memory in the same pass: the second pass over the scores is a read from L2
against a read from DRAM three hundred times its size. The upper bound on what
removing it could win is therefore well under one per cent of an operator that
is 20 per cent of a token, so under 0.2 per cent of a token, at the context
where attention matters most of all the ones that fit the measurement budget.

Against that, an incremental softmax rescales its accumulator at every tile, so
it is a different rounding of the one operator the correctness gate is most
sensitive to: the gate's four disagreements with the reference sit at margins of
0.009 to 0.126 on a logit scale, and the softmax is where a score difference
turns into a logit difference. The tiling as it landed is bit-identical to one
query at a time instead, which the unit tests assert with `==` on this model's
GQA shapes, at a cache length and a causal bound either side of the tile
boundary, for a batch and for a single token.

Projection for the long context we cannot run: the ratio above only gets worse
for the running softmax as the context grows, because the score row grows
linearly while the values it is weighed against grow linearly too, and the row
stays in L2 until about context 32768 (128 KiB per query). There is no context
at which the saved pass over the scores becomes the larger term.

## KV-cache dtype: measured, not taken

An f16 KV cache halves the bytes generation's attention reads, and that operator
is measured above: 20.1 per cent of a generation token at context 4096, 14.7 at
2048, 1.5 at 1024. So the **upper bound** on the win is half of each, which is
10 per cent of a token at context 4096, 7 per cent at 2048 and nothing
measurable at 1024, before anything is paid for converting f16 to f32 in the
score loop.

It is also a rounding change in the attention scores, which is what the gate
measures, and it would move the fixture that records the four permitted
disagreements. That fixture moves with a decision that is written down, not as
a side effect of a change made for speed, so this is not taken here; the number
above is what such a decision needs. The
reference's own default is an f16 cache, which is part of why its `tg16` column
below is ahead of its matched-flags column.

## Memory

Peak RSS, from `VmHWM`, per configuration:

| configuration | peak RSS |
| --- | --- |
| `run`, 16-token prompt, context 1024 | 2.76 GiB |
| `run`, 16-token prompt, context 4096 | 2.75 GiB |
| `bench` sweep, context 1024, 192-token prompt, 8 thread counts | 2.82 GiB |
| `bench`, context 4096, 1024-token prompt, 2 thread counts | 3.00 GiB |

Of that, 2.52 GiB is the mapped model file, resident because the output head
reads the whole embedding table every token; about 190 MiB is the tokenizer's
in-memory vocabulary; 50 MiB is the recurrent state, written every token.

**Raising `--context` costs nothing until the positions are used**, which the
first two rows measure: the same 2.75 GiB at context 1024 and at 4096 for a
16-token prompt. `vec![0.0; n]` of the cache's size comes from `mmap`, so its
pages become resident as they are first written. The difference between the last
two rows, 180 MiB, is exactly the KV positions the 1024-token prompt at context
4096 touches that the 192-token prompt at context 1024 does not: 3072 extra
positions times 8 blocks times 2 times 4 KV heads times 256 lanes times 4 bytes
is 192 MiB.

The batched path adds scratch for a batch of 16: every intermediate is 16 rows
instead of 1, and the score buffer is batch times head count times context
floats, 4 MiB at context 4096. That is inside the noise of the table above.

## llama.cpp on the same file, at matching thread counts

Reference build: llama.cpp at commit `3259ab6`, built outside this repository
with the recipe in `docs/decisions.md` under "The reference harness". `$REF_BIN`
below is its `build/bin`, which holds `llama-bench` next to the shared libraries
it needs, so `LD_LIBRARY_PATH` has to point at that same directory. The model is
the same GGUF file this runner reads, so quantization is out of the comparison.

```sh
REF_BIN=$REF_DIR/llama.cpp/build/bin   # $REF_DIR as in "The reference harness"
export LD_LIBRARY_PATH=$REF_BIN
M=<path>/Qwen3.5-4B-heretic-Q4_K_M.gguf

# flags matched to what this runner does: no weight repacking, no flash attention
$REF_BIN/llama-bench -m $M -p 192 -n 16 -t 1,2,3,4,5,6,7,8 -r 3 --repack 0 -fa off

# the reference as it ships: repacking on, flash attention auto, f16 KV cache
$REF_BIN/llama-bench -m $M -p 192 -n 16 -t 1,2,3,4,5,6,7,8 -r 3
```

`-ctk f32 -ctv f32` is refused by `llama-bench` at this commit, so both
reference rows run an f16 KV cache where this runner keeps f32. At a 192-token
prompt and 16 generated tokens the KV cache is under a megabyte against 2.52 GiB
of weights per token, so it is not what these rows differ by; the per-layer
comparison, which does need f32, uses `llama-eval-callback` and gets it there.

The matched-flags column and our column are both measured with nothing else
loading the model, ours at load 3.00 -> 6.17 and the reference at 4.29 -> 5.47.
The defaults column is from an earlier pass over the same commands at load
6.38 -> 5.43 with the serialisation above not yet in place, so read it as the
reference against itself and not against our column: at a 192-token prompt the
two reference columns differ by its weight repacking, which is what that
comparison is for.

Generation, `tg16` against our generation tok/s:

| threads | reference, matched flags | reference, its defaults | this runner | against matched |
| --- | --- | --- | --- | --- |
| 1 | 4.05 | 4.83 | 3.66 | -10% |
| 2 | 7.42 | 8.35 | 6.66 | -10% |
| 3 | 9.91 | 10.40 | 8.80 | -11% |
| 4 | 11.74 | 12.11 | 10.57 | -10% |
| 5 | 12.86 | 13.06 | 12.14 | -6% |
| 6 | 13.71 | 13.61 | 13.22 | -4% |
| 7 | 14.29 | 14.07 | 13.46 | -6% |
| 8 | 14.25 | 13.85 | 14.87 | **+4%** |

Prompt, `pp192` against our prompt tok/s:

| threads | reference, matched flags | reference, its defaults | this runner | against matched |
| --- | --- | --- | --- | --- |
| 1 | 21.72 | 21.37 | 10.55 | -51% |
| 2 | 40.90 | 39.60 | 18.85 | -54% |
| 3 | 54.79 | 53.02 | 25.36 | -54% |
| 4 | 69.85 | 67.03 | 31.43 | -55% |
| 5 | 80.21 | 78.75 | 36.12 | -55% |
| 6 | 92.63 | 90.08 | 40.36 | -56% |
| 7 | 97.98 | 95.30 | 43.88 | -55% |
| 8 | 102.47 | 98.21 | 47.16 | -54% |

One note per place we are behind:

- **Generation is within 11 per cent of the reference at every thread count from
  1 to 8 with the flags matched**, and ahead of it only at 8 threads, where its
  own curve turns down and ours does not. An earlier unserialised pass had us
  ahead at 6 and 7 as well; under the serialisation both sides gained and the
  reference gained more there, which is the honest picture and the one in the
  table.
- **The gap is 4 to 11 per cent and it is kernel work, not scheduling.** It is
  widest at 1 to 4 threads (10 to 11 per cent), where there is no scheduling
  left to win at all: our Q4_K dot is about a tenth behind theirs on the same
  block layout. The reference's own two columns differ by its weight repacking,
  which reorders the quantization blocks into a layout its kernel prefers and is
  a different rounding of the same arithmetic; we deliberately read the blocks
  as the file stores them.
- **The prompt is a factor 2.1 to 2.3 behind, down from a factor 4 to 7 before
  this branch.** The remaining factor is a different algorithm, not a different
  schedule: the reference turns a batched prompt into a true matrix-matrix
  multiply over a repacked weight layout, blocked in both dimensions, where this
  runner keeps the matvec shape and walks one weight row against the batch. The
  profile says where that would have to be won: 76 per cent of a prompt token is
  matmul, and at batch 16 it is compute-bound (157 MiB of
  weights per token is 7.0 GiB/s at 45 tok/s, 14 per cent of the ceiling), so
  the next factor is register blocking over several weight rows at once. That is
  a kernel rewrite with a correctness story of its own, so it belongs to a
  separate change rather than to the batching work this table measures.
- The reference's defaults beat its matched flags by 4 to 20 per cent on
  generation and 1 to 5 per cent on the prompt, which is the price of the
  comparison being fair rather than flattering.

At matching thread counts the reference is a peer on generation, within a tenth
either way, and ahead on the prompt by a factor two. The bar for this runner is the bandwidth ceiling in the table above, not
llama.cpp.

## Where a token's time goes

`perf` is not installed in this container and `/proc/sys/kernel/perf_event_paranoid`
is 2, so there is no sampling profile and no flamegraph. These tables come from
per-operator timers inside the runner, behind `rust-llm profile`; off, each is one
branch. The pool rows are *inside* the matmul rows above them, not extra: a matmul
is a pool job, and what the submitting thread spends on handing that job out and on
waiting for the last worker is part of what that matmul costs.

`measured total` is the sum of the operator rows. It is below 1000/tok_s from
`bench` by what is not inside a timed operator, which at these sizes is the
argmax over 248320 logits.

### Now, at the end of M2

```
profile: prompt, 192 tokens, 8 threads, context 1024, batch 16, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul                2.0       5.35    26.7
ffn: down matmul                       2.0       3.78    18.8
linear: qkv, gate, alpha, beta matmul    1.5       2.88    14.4
linear: delta-net recurrence           1.5       2.00    10.0
linear: output matmul                  1.5       1.36     6.8
elementwise                           11.5       1.18     5.9
linear: convolution                    1.5       0.83     4.1
full: qg, k and v matmul               0.5       0.79     3.9
output head matmul                     0.1       0.68     3.4
norms and rope                         7.6       0.51     2.5
full: output matmul                    0.5       0.43     2.1
full: attention                        0.5       0.27     1.4
embed                                  1.0       0.00     0.0
measured total                                  20.06   100.0
of which pool hand-out                10.1       0.84     4.2
of which pool join (last worker)      10.1       0.80     4.0
```

```
profile: generation, 8 tokens, 8 threads, context 1024, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul               32.0      17.90    27.7
ffn: down matmul                      32.0      12.34    19.1
linear: qkv, gate, alpha, beta matmul   24.0      10.88    16.9
output head matmul                     1.0      10.58    16.4
linear: output matmul                 24.0       3.35     5.2
full: qg, k and v matmul               8.0       2.59     4.0
linear: delta-net recurrence          24.0       2.22     3.4
elementwise                          184.0       1.17     1.8
full: output matmul                    8.0       1.12     1.7
full: attention                        8.0       0.97     1.5
linear: convolution                   24.0       0.82     1.3
norms and rope                       121.0       0.56     0.9
embed                                  1.0       0.00     0.0
measured total                                  64.50   100.0
of which pool hand-out               161.0       0.78     1.2
of which pool join (last worker)     161.0       2.46     3.8
```

The `calls/t` column in the prompt table is per token, so a batch of 16 shows
the 32 feed-forward jobs of a token as 2 and the one output head as 0.1: ten
jobs per token instead of 161, which is where the pool's share went from 5.9 to
8.2 per cent of a much smaller token without growing in absolute terms.

What is left to win, in order:

- **The matmuls other than the output head are 73 per cent of a prompt token
  and 75 per cent of a generation token** (76 and 91 with the head). In
  generation they are streaming weights and so are the ceiling itself. In the prompt they are compute, and the next factor is in the
  kernel: register blocking over several weight rows, which is the shape the
  reference's repacked GEMM has and ours does not.
- **The output head is 16.4 per cent of a generation token** and 3.4 of a prompt
  token, because a batch computes it once. It is a single 2560 x 248320 matmul
  over the whole embedding table and it cannot be fused with anything.
- **The delta-net recurrence is 10 per cent of a prompt token**, up from 3.4 per
  cent of a generation token, because it is the one operator a batch does not
  make cheaper: it is sequential over tokens by construction. Its inner loops
  carry an f32 reduction, so they do not vectorize without reassociating the
  sum, which would change the result. That is why it is not on this branch; see
  "What was measured and not kept".
- Attention is 1.4 per cent at this context and 20 per cent at 4096; see
  "Attention" above.

### The tiers against each other

The VNNI tier was kept on this measurement rather than deleted. Prompt tok/s,
context 1024, 192-token prompt, batch 16, two passes each alternated from one
binary through `--tier`, from before the one-run-at-a-time rule, so again the
pair is what holds:

| | pass 1 | pass 2 | generation |
| --- | --- | --- | --- |
| 8 threads, `avx2+vnni` | 46.19 | 44.53 | 12.38, 13.46 |
| 8 threads, `avx2+fma` | 43.63 | 43.27 | 12.47, 14.08 |
| 1 thread, `avx2+vnni` | 10.48 | | 3.60 |
| 1 thread, `avx2+fma` | 9.96 | | 3.75 |

VNNI is 4.4 per cent ahead at 8 threads and 5.2 at 1 on the prompt, which is the
compute-bound path and so the one where a kernel can show at all, and a tie
inside the spread on generation, which is bandwidth-bound. Before the batched
kernel landed the same pair was 32.65/32.72 against 31.53/31.47, 3.8 per cent.
Both tiers have the batched kernel, so this is like for like.

### Before the threading work, for comparison

From commit `85dbbad`, the measurement surfaces before any change to the
scheduling. Generation, 4 threads, context 1024:

```
operator                           calls/t   ms/token       %
ffn: down matmul                      32.0      20.55    16.7
ffn: gate matmul                      32.0      17.25    14.0
ffn: up matmul                        32.0      17.22    14.0
output head matmul                     1.0      14.90    12.1
linear: qkv matmul                    24.0      13.81    11.2
linear: gate/alpha/beta matmul        72.0      12.94    10.5
linear: output matmul                 24.0       7.53     6.1
linear: delta-net recurrence          24.0       6.16     5.0
full: qg matmul                        8.0       3.93     3.2
full: output matmul                    8.0       2.41     2.0
full: v matmul                         8.0       1.43     1.2
full: k matmul                         8.0       1.36     1.1
elementwise                          184.0       1.04     0.8
full: attention                        8.0       0.98     0.8
linear: convolution                   24.0       0.81     0.7
norms and rope                       121.0       0.53     0.4
measured total                                 122.86   100.0
of which pool wake (barrier in)      281.0      12.36    10.1
of which pool join (barrier out)     281.0      28.51    23.2
```

And the same at 8 threads, which is where the barriers were paid for twice:

```
measured total                                 169.70   100.0
of which pool wake (barrier in)      281.0      44.40    26.2
of which pool join (barrier out)     281.0      69.40    40.9
```

That is 281 pool jobs per token meeting at two `std::sync::Barrier`s, costing 33
per cent of a token at 4 threads and 67 per cent at 8, which is the whole reason
8 threads used to be slower than 4. It is 5.0 per cent at 8 threads now, and the
per-tensor breakdown inside a group is gone because the group is one job.

## What the threading work moved

Each change landed with its own before/after pair, measured by alternating
passes against the previous commit's binary in one window. The commit bodies
carry the pairs and the gate margins.

| commit | change | 4 threads | 8 threads |
| --- | --- | --- | --- |
| `9f95131` | the projections that share an input are one job, 281 jobs/token -> 161 | 8.17 -> 9.10 (+11%) | not a measurement in that window, spreads to 74% |
| `cf4c9c3` | workers claim row blocks from an atomic counter instead of a static split | 9.12 -> 9.79 (+7%) | 7.63 -> 8.86 (+16%) |
| `dba9ecb` | a sequence counter and a bounded spin instead of the two barriers | 9.64 -> 11.14 (+16%) | 9.36 -> 15.01 (+60%) |

Generation tok/s, context 1024, 19-token prompt, 16 generated tokens. The
absolute level differs between rows because each row is its own window; only the
pair within a row is comparable.

None of the three changes the arithmetic, and neither does anything on the
batched path: the per-layer trace of a two-token prompt is byte-identical across
all of them, and both correctness gates keep the four disagreements the fixture
records at 0.00937, 0.01035, 0.04748 and 0.12625, to the last digit.

## What was measured and not kept

- **The running maximum and running sum of a flash-style attention**: see
  "Flash-style attention, measured and not kept" above. Under 0.2 per cent of a
  token at the context where attention matters most, against a rounding change
  in the softmax.
- **An f16 KV cache**: see "KV-cache dtype" above. Upper bound 10 per cent of a
  generation token at context 4096, nothing measurable at 1024, and a rounding
  change in the scores.
- **Vectorizing the delta-net recurrence**: not implemented. Its inner loops
  accumulate `sk` and the readout in f32, which is a serial dependency the
  compiler cannot vectorize without reassociating the sum, and reassociating it
  changes the result of the operator the reference's own arithmetic was copied
  for. It is 10 per cent of a prompt token, so there is something there, but it
  is a change to the arithmetic and belongs with a decision about the gate
  rather than inside a tuning commit.
- **A batch larger than 16 as the default**: measured, 8 per cent more prompt
  tok/s at batch 64 for four times the scratch. See the batch-size curve.
- **A batched Q8_0 kernel**: not written. This file holds no `Q8_0` tensor, so
  there is nothing here to measure it on, and a kernel nobody can measure is a
  second rounding behaviour to keep green for nothing.
- **Thread affinity**, pinning workers to CPUs: not implemented. The profile
  stopped pointing at the scheduler once the hand-out was a sequence counter and
  the split was dynamic, and the quota does not say which CPUs it will be given,
  so pinning here would be guessing at the host's placement on a box that is
  shared with other work.
- **A thread count below the quota as the default**: measured, does not hold.
  See `docs/decisions.md` under "Threading".
