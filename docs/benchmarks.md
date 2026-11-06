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

# speculation: what a verification row costs, and the drafter on against off
# with its acceptance histogram. --prompt-tokens above the fixed text's own
# length repeats that text, which is the deliberately repetitive case.
# --draft-source mtp adds what the head itself costs and its break-even.
rust-llm bench --context 1024 --threads 8 --prompt-tokens 400 \
  --max-tokens 32 --repeats 5 --speculate 4

# the acceptance histogram for a prompt of your own, from one generation
rust-llm run --speculate 4 --max-tokens 64 --prompt '<text>'
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
  profile below, of a 21.21 ms prompt token the delta-net recurrence is 4.3 per
  cent, the elementwise work 6.0, the convolution 5.0 and the norms 2.7: 18.2
  per cent that costs the same per token whether 1 token or 16 go through
  together. It was 22.5 per cent before the recurrence was blocked over its
  output components. With that fixed and the matmuls divided by the batch, 3.2
  at 8 threads is close to what is available; the next factor would have to come
  out of the matmul kernel, not out of the batch.
- Generation reaches 77 per cent of the measured ceiling at 8 threads. The
  remaining fifth is the output head and everything per token that is not
  streaming weights; it is not barrier time, which was 3.8 per cent in the
  window this sweep was taken in. (The re-run profile below shows 11.2 on a
  busier box: that row is where another tenant's work on the quota lands.)
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
| 1024 | 0.33 ms (1.6%) | 21.21 ms | 1.03 ms (1.4%) | 75.91 ms |
| 2048 | 2.92 ms (11.9%) | 24.49 ms | 12.50 ms (14.7%) | 85.11 ms |
| 4096 | 6.49 ms (22.8%) | 28.43 ms | 18.09 ms (20.1%) | 89.83 ms |

The 1024 row is the re-run profile at the end of M2 and the 2048 and 4096 rows
are from the commit before `1828971`, so those two are the "before" of the pair
below. Three windows, three levels: what each row is for is attention's share
of its own token, which is why the share is in the cell.

Generation's attention cost is the KV cache's bytes
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

## Blocking the delta-net recurrence

The recurrence was the largest non-matmul row of a prompt token and was left
alone for a long time, on the grounds that vectorizing it meant reassociating an
f32 sum and so changing the result of the one operator whose arithmetic was
copied from the reference step for step. That turned out to be a mix-up about
*which* sum. The two reductions over the contracted dimension are serial and
cannot be reassociated, and they are not the ones that had to be vectorized:
the `d_state` output components are independent of each other, each with its own
decay-and-contract, its own delta and its own readout. A block of components
therefore puts one whole scalar sequence in every lane, in the original order,
and the result is bit-identical by construction rather than within a tolerance.
What it cost was the state's layout, `[i][j]` per head instead of `[j][i]`,
which is internal: nothing outside the recurrence reads it, the reset only
zeroes it and no tracer node dumps it.

`50bb149`, against `4ebb916`'s binary alternated in one window, three passes,
context 1024, 192-token prompt, 8 threads, batch 16, tier `avx2+vnni`, load 1.8
to 7.5:

| | before | after |
| --- | --- | --- |
| the operator, prompt | 2.24 ms/token | 0.93 ms/token |
| the whole prompt token | 22.59 ms | 21.29 ms |
| prompt | 45.03 tok/s | 46.61 tok/s |
| the operator, generation | 3.07 ms/token | 2.35 ms/token |
| generation | 13.51 tok/s | 13.69 tok/s |

**Generation does not move, and could not have.** The recurrent state is 48 MiB
and the recurrence reads and writes all of it once per token, so 96 MiB against
the 48.6 GiB/s this box streams is a 2.0 ms floor under a row that measured
2.2 to 3.1: there was never more than a few tenths of a millisecond in it, and
the pair above is inside the spread on the token even though it is up in all
three passes. The prompt is where the win is for the mirror-image reason: a
batch of 16 reuses one head's 64 KiB of state out of cache across all 16 tokens,
so the traffic floor there is about 0.13 ms and what the loop was paying instead
was the latency of its serial chain.

The block is 64 components, picked on the whole token and not on the operator's
own row. Four binaries alternated over a 1024-token prompt, both orders so the
load drift inside a pass cancels, prompt phase in ms/token:

| block | this operator | the convolution | the whole token |
| --- | --- | --- | --- |
| scalar | 2.04 | 0.78 | 21.80 |
| 16 | 0.96 | 0.83 | 20.93 |
| 32 | 0.84 | 0.99 | 21.14 |
| 64 | 0.87 | 0.86 | 20.72 |

A block of 32 is the fastest row by 0.03 ms and the slowest of the three on the
token, because it reproducibly costs the convolution next to it 0.2 ms. The
convolution's loop is untouched by any of this and the cost does not grow with
the number of blocks a head is walked in, which is what a cache explanation
would predict, so it is read as the code layout of that one build rather than
as something to tune against. A block of 16 is too narrow to amortise the load
of the two inputs over.

Measuring this needed a longer prompt than the 192 tokens the rest of this file
uses. At 192 the prompt phase is under four seconds and `bench` cannot separate
the two builds at all: one window gave 43.21 against 47.51 tok/s and a quieter
one gave 47.71 against 47.45. The operator's own row is stable to 0.02 ms across
every window, which is why it is the number of record here.

## KV-cache dtype: measured, not taken

An f16 KV cache halves the bytes generation's attention reads, and that operator
is measured above: 20.1 per cent of a generation token at context 4096, 14.7 at
2048, 1.5 at 1024. So the **upper bound** on the win is half of each, which is
10 per cent of a token at context 4096, 7 per cent at 2048 and nothing
measurable at 1024, before anything is paid for converting f16 to f32 in the
score loop.

Those percentages are measured with the cache **filled** to the context. The
attention cost follows the positions actually in use and not the allocation, so
the default `--context 4096` with a short conversation in it sits on the 1024
row, where the win is nothing measurable. The configuration the upper bound
applies to is a context that is genuinely full.

The reason it is not taken is the gate, and not the fixture moving. The gate
pins the reference to `-ctk f32 -ctv f32` because, in its own words, f16 keys
are a different model and not a different rounding. Taking an f16 cache here
means moving the reference's flags with it, and then the gate is no longer
comparing two roundings of one model, which is the whole thing it exists to do.
A fixture that moves is a decision someone writes down; a gate that stops
measuring what it claims to measure is not a decision, it is the loss of the
instrument.

The memory is not the argument either way: an f16 cache saves 90 to 180 MiB of
a 2.75 to 3.00 GiB peak, of which 2.52 GiB is the mapped model file. See
"Memory" below.

What would reopen it: a requirement for generation throughput at a full long
context, arriving as a `--kv-dtype` option with f32 the gated default rather
than as a replacement, with its own fixture, both dtypes measured, and the
indeterminate band re-derived for the f16 path. The reference's own default is
an f16 cache, which is part of why its `tg16` column below is ahead of its
matched-flags column.

## Speculative decoding

`--speculate n` drafts `n` tokens per step from the text so far and verifies
them in one batch. It never changes what comes out: see `docs/decisions.md`.
What it changes is how many passes over the weights a run needs, and whether
that is a win depends on two numbers measured here.

### What a verification row costs

This is the number the whole feature turns on, and it was measured before the
feature was built. A verification batch runs the output head on **every** row,
which no other path does: a prompt computes it once per batch and generation
once per token. So a verification row is a prompt batch's marginal row plus
whatever the head's extra rows cost, and nothing had that number.

Context 1024, 8 threads, tier `avx2+vnni`, under the model lock, 3 to 5
repeats, median. The last column is break-even: a draft is worth running when
the passes it saves cost more than the rows it adds, which at `m` ms per
marginal row against `t` ms per sequentially generated token is `accepted drafts
per drafted token > m / t`.

| rows | ms per batch | ms per extra row | break-even acceptance |
| --- | --- | --- | --- |
| 1 | 64.3 to 71.4 | | |
| 2 | 87.8 to 101.2 | 18.7 to 29.7 | 0.27 to 0.42 |
| 3 | 97.9 to 119.4 | 15.3 to 24.0 | 0.22 to 0.33 |
| 4 | 118.3 to 136.0 | 16.2 to 21.5 | 0.24 to 0.31 |
| 5 | 138.2 to 149.7 | 17.1 to 20.3 | 0.27 to 0.30 |

The ranges are across windows, with 64 to 71 ms per sequential token in the same
windows; read the ratio in the last column, not the level. The `--speculate 8`
block carries the table out to nine rows, where the marginal row is still 19.2 to
23.6 ms (189.4 ms at six rows, 225.3 at nine), so it is flat in the row count:
nothing about a wider batch gets worse. **A marginal
verification row is a fifth to a third of a token**, where the ticket that
commissioned this would have stopped the milestone at half a token.

It is that cheap because the output head is one 2560 x 248320 matmul over the
whole embedding table, 0.5 GiB of weights, and at one row it is bandwidth-bound:
11 ms of streaming, which is 14.6 per cent of a generation token (see "Where a
token's time goes"). Extra rows reuse bytes that are already in flight and add
only arithmetic, so the head on five rows costs nearly what it costs on one, and
what is left in the 16 to 30 ms is the part of a row that does not batch.

**The ring costs nothing on a pass with no draft.** The same call at the same row
count, against a runner whose state is a single buffer updated in place, in the
same window: +0.1, +0.9, -7.0, -0.8, -1.0 and +5.2 per cent over six blocks. The
sign is not stable, so there is nothing to report beyond the noise, which is what
the design predicted: writing the next ring slot is the same memory traffic as
updating one in place. (An earlier reading of 7 per cent compared a verification
batch against the `generation` row above it, which is a different harness
measured earlier in the block; that was a window difference, not a cost.)

### What it is worth: the acceptance histograms

Both sides of the same `bench` invocation, on one runner, so the only difference
between the pair is whether drafts are offered; the ring is allocated and written
in both. 32 generated tokens, greedy, 3 to 5 repeats. The histogram is summed
over the repeats and reads `accepted:passes`, so `2:3 4:3` is three passes that
accepted two drafts and three that accepted four.

**Deliberately repetitive text**, a 400-token prompt that is the fixed bench text
twice over:

| `--speculate` | sequential | speculated | gain | acceptance | accepted lengths |
| --- | --- | --- | --- | --- | --- |
| 1 | 14.93 | 16.61 | +11% | 1.00 | `1:12` |
| 2 | 13.75 | 16.75 | +22% | 0.83 | `1:5 2:10` |
| 4 | 15.04 | 16.52 | +10% | 0.75 | `2:3 4:3` |
| 8 | 13.51 | 17.40 | +29% | 0.88 | `7:3` |

**Prose**, the same text at 192 tokens, which is one copy and so repeats nothing:

| `--speculate` | sequential | speculated | gain | acceptance | accepted lengths |
| --- | --- | --- | --- | --- | --- |
| 4 | 12.12 | 12.65 | +4% | 0.31 | `0:6 1:3 4:3` |
| 8 | not measured | | | 0.16 | `0:10 1:5 4:5` |

The prose pair is from the one window that was quiet enough for it (spreads 3.9
and 5.5 per cent). Every later attempt came back with 42 to 66 per cent spread on
the `sequential` row with other work on the box, so the eight-draft pair has no
timing to report. Its acceptance line is reproducible across windows, because
acceptance is a property of the model and the text and not of the clock, and it
is what the conclusion below rests on.

**The gate prompt set**, ten real chat prompts, from
`speculation_does_not_change_the_output`: 27 verification passes over 384 greedy
tokens at four drafts, 22 of 105 drafts accepted, so 0.21 per drafted token. Per
prompt it ranges from nothing at all (`system_persona_no_think` and
`non_ascii_no_think` never draft, because nothing in them repeats) to 1.00
(`capital_think`, one pass accepting all four).

What the two tables say:

- **On repetitive text it pays, by 10 to 29 per cent.** Acceptance is 0.75 to
  1.00 against a break-even of 0.22 to 0.42, so there is room to spare. The
  gains do not line up neatly with the draft length, and the draft-2 row is the
  reason to distrust the level rather than the ratio: the same configuration
  gave `15.10 -> 15.12` in one window and the `13.75 -> 16.75` above in the next,
  with an identical acceptance line both times. On a shared box the timing moves
  and the acceptance does not.
- **On prose it does not.** 0.31 at four drafts is just above break-even and the
  measured gain, +4 per cent, is inside the spread of the pair that produced it
  (3.9 and 5.5 per cent). On the gate prompt set 0.21 is at or below it.
- **A longer draft does not help prose.** At eight drafts the prose histogram is
  the same as at four, `0:10 1:5 4:5`: the drafter offers twice as many tokens
  and the model accepts no more of them, so acceptance halves from 0.31 to 0.16
  and every extra row is paid for nothing. That is the shape of prompt lookup
  rather than of the batch: the match it finds is short, so what follows it runs
  out.
- **The histogram is bimodal, not centred.** A pass either accepts nothing or
  accepts the whole draft, with little in between; prose is `0:10 1:5 4:5` and
  repetitive text is `2:3 4:3` or `7:3`. So the mean acceptance understates what
  a good pass does and overstates what a bad one does, which is why the
  distribution is reported and not only the rate. It is also the measurement the
  MTP head below is held against.

### Drafting from the MTP head

`--draft-source mtp` drafts from the model's own multi-token-prediction block
instead of from the text so far. That changes the arithmetic above in one way
that matters: **prompt lookup drafts for free and a head does not.** So the
verification table is no longer the whole cost and its last column is no longer
the whole break-even.

#### What the head costs

Context 1024, 8 threads, tier `avx2+vnni`, under the model lock, 3 to 5
repeats, median, from `bench --draft-source mtp`'s own `mtp` block.

| | measured |
| --- | --- |
| one draft step | 13.2 to 14.8 ms, over eight blocks |
| catch-up per committed token | +1.02 and +1.15 ms in the two quietest blocks |
| a sequential token in the same windows | 64 to 78 ms |

A draft step is one pass over the MTP block, about 75 MB of weights, plus
`eh_proj` at 7.4 MB, plus the output head on one row at 0.5 GiB. The head
dominates it, which is why 13.2 to 14.8 ms is almost exactly the 11 ms "Where a
token's time goes" prices the head at: **a draft step is the output head plus a
block, and a token is the output head plus thirty-two.** The reading is stable
across every window, quiet or not, because it is bandwidth and not scheduling.

The catch-up is a row through the same block per committed token, so the block's
cache holds every position the trunk holds. It is the one number here the pair
method could not pin down: it is the difference between a `forward` on a runner
with the head enabled and one without, and over eight blocks that difference
came out between -14.4 and +4.98 ms with the box under load. The two quietest
blocks give +1.02 and +1.15 ms, and they are the readings the arithmetic agrees
with: 75 MB against a token's 2.5 GiB is 2.8 per cent of 68 ms, which is 1.9 ms.
Treat it as one to two per cent of a token.

**A catch-up row runs more of the block than it has to.** Only the key and
value projections feed the block's cache, so it could stop after them and read
11 MB rather than 75. That is the upgrade path if the profile ever asks for it;
what it buys back is under 2 ms a token, and what running the whole block
through the trunk's own `full_attn` and `ffn` buys is that the head cannot
drift away from the block it drafts for.

#### The break-even that follows

Speculation at draft length `d` with `a` of those drafts accepted pays when

```text
verify(d + 1) + d * step + (a + 1) * catch_up  <  (a + 1) * token
```

so `a > (verify(d + 1) + d * step) / (token - catch_up) - 1`. As accepted drafts
per drafted token, which is what the acceptance column reports:

| `--speculate` | ms per batch | ms of draft steps | break-even acceptance |
| --- | --- | --- | --- |
| 1 | 79.2 | 13.7 | 0.38 |
| 2 | 101.0 | 27.3 | 0.45 |
| 3 | 104.8 | 40.9 | 0.39 |
| 4 | 120.9 | 54.6 | 0.40 |

**0.38 to 0.45, against 0.22 to 0.42 for prompt lookup.** Drafting from a head
has to be about twice as accurate as drafting from the text to be worth the
same, and that is entirely the output head: a draft step is 20 per cent of a
token and prompt lookup's is 0.3 ms.

#### What it is worth: the acceptance histograms

32 generated tokens, greedy, 3 repeats, histogram summed over them.
`--draft-source mtp --speculate d`, the same two texts the prompt-lookup tables
above use.

**Prose**, the fixed bench text at 192 tokens, which repeats nothing:

| `--speculate` | acceptance | accepted lengths | passes that drafted nothing |
| --- | --- | --- | --- |
| 1 | 0.83 | `0:6 1:30` | 27 of 63 |
| 2 | 0.87 | `0:3 1:9 2:15` | 27 of 54 |
| 4 | 0.78 | `0:3 1:6 2:6 3:0 4:6` | 30 of 51 |
| 8 | 0.80 | `0:3 1:6 2:6 3:0 4:3 6:3` | 24 of 45 |

**Deliberately repetitive text**, the same text twice over at 400 tokens:

| `--speculate` | acceptance | accepted lengths | passes that drafted nothing |
| --- | --- | --- | --- |
| 1 | 0.76 | `0:12 1:39` | 3 of 54 |
| 2 | 0.57 | `0:15 1:12 2:18` | 0 of 45 |
| 4 | 0.51 | `0:12 1:6 2:9 3:3 4:6` | 0 of 36 |
| 8 | 0.42 | `0:12 1:6 2:6 3:3 4:3 7:3` | 0 of 33 |

**The gate prompt set**, ten real chat prompts, from
`mtp_drafting_does_not_change_the_output`: 477 of 712 drafts accepted over 265
verification passes at draft 2 and draft 4 together, so **0.67 per drafted
token**, and 0.48 to 0.91 by prompt. Only 6 of 271 passes drafted nothing.

What those three tables say, and it is the opposite of prompt lookup's shape:

- **The head's best text is the text prompt lookup cannot do anything with.**
  On prose it accepts 0.78 to 0.87 where prompt lookup accepts 0.31 at four
  drafts and 0.16 at eight. On the gate prompt set it accepts 0.67 where prompt
  lookup accepts 0.21, and two of those ten prompts never draft at all from the
  text because nothing in them repeats. That is the whole reason to have built
  it.
- **A longer draft costs the head nothing in accuracy on prose.** 0.83, 0.87,
  0.78, 0.80 at one, two, four and eight drafts, where prompt lookup halves from
  0.31 to 0.16 between four and eight. The head predicts from the model, so it
  does not run out of match the way a lookup does.
- **On repetitive text the head is the *worse* drafter**, 0.42 to 0.76 against
  prompt lookup's 0.75 to 1.00, and it falls off with the draft length where
  prompt lookup does not. A literal repeat is exactly what a lookup is for, and
  a head trained to predict the next token has no special advantage there.
- **The confidence floor earns its place on prose and does nothing on
  repetitive text.** A quarter to a half of the prose passes draft nothing at
  all, which is the floor refusing a draft the trunk would probably have
  rejected; on repetitive text it almost never fires. The cost of a pass that
  drafts nothing is one draft step, 13.7 ms, because the block and the head run
  before the floor is consulted.
- **Both histograms are bimodal**, as prompt lookup's are: a pass either
  accepts nothing or accepts most of the draft.

#### The measured gain: +9.5 per cent at two drafts, +5.6 at four

Prose, 5 repeats, the two blocks taken in the quietest windows this box offered.
Both are a `sequential` against `speculated` pair on one runner, so the only
difference between the two rows is whether drafts are offered.

| `--speculate` | load | sequential | speculated | gain | acceptance |
| --- | --- | --- | --- | --- | --- |
| 2 | 2.35 -> 4.90 | 15.85 (2.5%) | 17.36 (2.9%) | **+9.5%** | 0.87 |
| 4 | 2.13 -> 4.20 | 16.01 (3.6%) | 16.91 (3.6%) | **+5.6%** | 0.78 |

The per-cent in brackets is that row's own spread over its five runs. At two
drafts the gain is three times the spread; at four it is one and a half times.

**Two drafts beats four, and the break-even table says why.** A round at four
drafts spends four draft steps, 52 ms, where one at two spends 27, and it
accepts a smaller fraction of them (0.78 against 0.87). The extra verification
rows are nearly free, as the prompt-lookup table above shows; the draft steps
are not. So depth buys less here than it does for a drafter that guesses for
nothing, and the shallow end is where this head pays best.

The levels reproduce less well than the acceptance does. The same two-draft
configuration in a less quiet window (load 1.99 to 4.35, spreads 5.7 and 6.4)
gave `15.31 -> 16.24`, +6.1 per cent, and three earlier four-draft blocks gave
+2.5, +4.9 and +9.7 at spreads of 6.8 to 9.9. Every one of those blocks had a
**byte-identical** acceptance line to its counterpart above, down to the
histogram. The acceptance reproduces exactly and the tok/s does not, which is
the same thing the prompt-lookup tables say about this box.

What the pieces predict is a little more than this: about +10 per cent at two
drafts and +15 at four, from the step cost, the batch cost and the number of
passes that draft nothing. Two drafts lands on it and four does not, and the
four-draft gap is the one thing in this section a better window could still
move.

Seven further blocks across the sweeps ran at load averages of 4.2 to 8.6, with
the `sequential` row swinging from 11.11 to 15.64 tok/s on one configuration,
which is further than the effect; those are not quoted at all. The reason is in
"The machine": one model run at a time is the rule on this box, and another
ticket was benchmarking through most of this work.

### The default is 0

Ordinary prose, which is what the gate prompt set is, sits at or below
break-even for prompt lookup, so speculation there pays verification rows for
nothing. Repetitive
text wins by 10 to 29 per cent, but a runner cannot tell in advance which it has
been given, and the cost of guessing wrong is a slower run. So it is a flag, not
a default: a caller who knows their text repeats itself, a long quoted passage,
a diff, boilerplate code, turns it on and gets the numbers above.

**The head is the first draft source here that is above its own break-even on
ordinary prose**, and it has a measured gain: 0.67 accepted on the gate prompt
set against a break-even of 0.38 to 0.45, and +9.5 per cent at two drafts and
+5.6 at four. That is a real case for turning it on, and it is still not this
section's call to make: two clean pairs on a box where another ticket was
benchmarking throughout is thinner evidence than the prompt-lookup tables rest
on, and the levels move between windows where the acceptance does not. So the
default stays 0 and the draft source stays `prompt-lookup`. Flipping either is
a decision taken against the numbers in code review, which is where the ticket
that commissioned this put it.

**The two draft sources are for opposite texts, which is worth knowing whichever
default wins.** Prompt lookup is for text that repeats literally and does
nothing at all otherwise; the head is for text that does not repeat and is the
worse drafter when it does. A caller who knows which they have should say so.

**`--speculate 0` measures as the build without speculation at all.** Measured
when prompt lookup landed, before the head existed: that branch's binary against
`origin/main`'s (`9bdd45c`), alternated three times in one window,
context 1024, 192-token prompt, 8 threads, 3 repeats each, medians over the
rounds:

| | `main` | this branch, `--speculate 0` |
| --- | --- | --- |
| prompt | 50.19 tok/s | 50.02 |
| generation | 14.83 tok/s | 14.63 |

-1.3 per cent on generation against per-block spreads of 1.3 to 33 per cent, so
inside the spread. It should be: at `--speculate 0` the ring is one buffer, the
read slot and the write slot are the same, and the recurrence updates it in place
through the function it always used. What the off path gains is one modulo and
one branch per row. The stronger check is not a timing one: `greedy_output_is_unchanged`
and `every_next_token_matches_the_reference` pass unchanged on this branch
against the committed fixture.

`--draft-source mtp` does not move that path either, and it cannot: the flag
only decides whether `enable_mtp` is called, and it is only called above
`--speculate 0`. The block's cache, its scratch and the extra `rms_norm` per
prompt row all come with that call.

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

**`--speculate n` adds `n` copies of the recurrent state**, which is the one
thing in this feature that is not free: the rollback is a ring of `n + 1`
convolution histories and delta-net states instead of one, 2.1 MiB per linear
block times 24 blocks times `n + 1`.

| `--speculate` | ring | peak RSS | above `--speculate 0` |
| --- | --- | --- | --- |
| 0 | 50.3 MiB | 2.76 GiB | |
| 4 | 252 MiB | 2.94 GiB | +192 MiB |
| 8 | 453 MiB | 3.15 GiB | +404 MiB |

One `rust-llm run`, 16-token prompt, 16 generated tokens, context 1024, 8
threads. The ring column is arithmetic and the peak column is `VmHWM`, and they
agree: 202 MiB of extra ring against 192 MiB of extra resident at four drafts,
403 against 404 at eight. The pages become resident as they are first written,
as the KV cache's do, so the ring is resident by the first batch of that width
and not before.

The ring is allocated by the constructor and only when speculation is on, so
`--speculate 0` is the 50 MiB it always was, and 2.76 GiB is the same figure as
the first row of the table above. The logit buffer grows the same way, one row
of 248320 floats per `n + 1`, which is 0.95 MiB a row and does not show.

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
  profile says where that would have to be won: 80 per cent of a prompt token is
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

Re-run on the final head of the branch, so the shares match what ships. Both
tables are one invocation, at a load of 4.1 on a box that had other work on it;
the same command earlier in the day gave a 20.06 ms prompt token and a 64.50 ms
generation token. **Read the shares, not the level.** The level here is this
window's, and generation is the half that shows it: it is bandwidth-bound, so
the workers get descheduled and the pool's join row absorbs the difference. The
comparable thing is a pair measured in one window, and the only operator this
branch's last change moves has its pair under "Blocking the delta-net
recurrence" above, where `4ebb916` gave 75.82 ms against the 75.91 here.

```
profile: prompt, 192 tokens, 8 threads, context 1024, batch 16, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul                2.0       5.96    28.1
ffn: down matmul                       2.0       4.17    19.7
linear: qkv, gate, alpha, beta matmul     1.5       3.29    15.5
linear: output matmul                  1.5       1.53     7.2
elementwise                           11.5       1.28     6.0
linear: convolution                    1.5       1.07     5.0
linear: delta-net recurrence           1.5       0.92     4.3
full: qg, k and v matmul               0.5       0.87     4.1
output head matmul                     0.1       0.71     3.3
norms and rope                         7.6       0.58     2.7
full: output matmul                    0.5       0.51     2.4
full: attention                        0.5       0.33     1.6
embed                                  1.0       0.00     0.0
measured total                                  21.21   100.0
of which pool hand-out                10.1       1.03     4.9
of which pool join (last worker)      10.1       1.20     5.7
```

```
profile: generation, 8 tokens, 8 threads, context 1024, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul               32.0      21.19    27.9
ffn: down matmul                      32.0      15.22    20.1
linear: qkv, gate, alpha, beta matmul    24.0      12.34    16.3
output head matmul                     1.0      11.05    14.6
linear: output matmul                 24.0       3.86     5.1
linear: delta-net recurrence          24.0       3.13     4.1
full: qg, k and v matmul               8.0       3.12     4.1
full: output matmul                    8.0       1.80     2.4
elementwise                          184.0       1.52     2.0
full: attention                        8.0       1.03     1.4
linear: convolution                   24.0       1.02     1.3
norms and rope                       121.0       0.61     0.8
embed                                  1.0       0.00     0.0
measured total                                  75.91   100.0
of which pool hand-out               161.0       3.02     4.0
of which pool join (last worker)     161.0       8.50    11.2
```

The `calls/t` column in the prompt table is per token, so a batch of 16 shows
the 32 feed-forward jobs of a token as 2 and the one output head as 0.1: ten
jobs per token instead of 161. The pool costs 2.23 ms of a prompt token against
11.52 ms of a generation token for that reason, and its *share* is the larger
one on generation here only because this window's load lands on the join row;
in a quiet window the two are 1.64 and 3.24 ms, which is the same story.

What is left to win, in order:

- **The matmuls other than the output head are 77 per cent of a prompt token
  and 76 per cent of a generation token** (80 and 90 with the head). In
  generation they are streaming weights and so are the ceiling itself. In the prompt they are compute, and the next factor is in the
  kernel: register blocking over several weight rows, which is the shape the
  reference's repacked GEMM has and ours does not.
- **The output head is 14.6 per cent of a generation token** and 3.3 of a prompt
  token, because a batch computes it once. It is a single 2560 x 248320 matmul
  over the whole embedding table and it cannot be fused with anything.
- **The delta-net recurrence is 4.3 per cent of a prompt token**, down from 10
  before it was blocked over its output components; see "Blocking the delta-net
  recurrence" above. It is still the one operator a batch does not make cheaper,
  because it is sequential over tokens by construction, and what is left of it
  on generation is the state's own memory traffic rather than the loop.
- Attention is 1.4 per cent of a generation token at this context (1.6 of a
  prompt token) and 20 per cent at 4096; see "Attention" above.

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
  generation token at a full context 4096, nothing measurable at 1024, against
  a gate that would stop comparing two roundings of one model.
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
