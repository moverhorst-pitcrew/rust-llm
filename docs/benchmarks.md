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
export RUST_LLM_MODEL=/paperclip/models/qwen3.5-4b-heretic/Qwen3.5-4B-heretic-Q4_K_M.gguf

# throughput, median of repeats with the spread, per thread count
rust-llm bench --context 1024 --threads 1,2,3,4,5,6,7,8 --prompt-tokens 19 \
  --max-tokens 16 --repeats 3

# the long-context configuration
rust-llm bench --context 4096 --threads 4,8 --prompt-tokens 204 \
  --max-tokens 16 --repeats 3

# where a token's time goes
rust-llm profile --context 1024 --threads 8 --prompt-tokens 19 --max-tokens 8

# what the machine can stream, which is the ceiling batch-1 generation hits
cargo run --release -p llm-core --example bench_bandwidth
```

The first run of each configuration is a warm-up and is not counted. The prompt
is a fixed text in the binary truncated to exactly `--prompt-tokens` tokens (204
tokens is the whole text), and generation runs exactly `--max-tokens` tokens
through the forward pass and ignores stop tokens, so two runs measure the same
work whatever the model decides to say.

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

## The runner against that ceiling

Context 1024, 19-token prompt, 16 generated tokens, tier `avx2+vnni`, quota 8
CPUs. Generation tok/s with its spread, and what fraction of the implied ceiling
above it reaches.

The quietest full sweep (1 to 8, 3 repeats, load 3.99 -> 4.55):

| threads | prompt tok/s | generation tok/s | spread | ceiling | fraction |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.55 | 3.47 | 2.2% | 6.26 | 55% |
| 2 | 6.37 | 6.40 | 1.7% | 11.33 | 56% |
| 3 | 8.79 | 8.52 | 2.5% | 14.56 | 59% |
| 4 | 10.80 | 10.60 | 1.9% | 16.63 | 64% |
| 5 | 12.12 | 11.80 | 4.0% | 17.79 | 66% |
| 6 | 12.95 | 12.79 | 0.8% | 18.77 | 68% |
| 7 | 12.78 | 12.69 | 1.1% | 19.30 | 66% |
| 8 | 11.75 | 11.57 | 12.7% | 19.19 | 60% |

The best window measured (8, 7, 6, 4 in that order, 3 repeats, load 4.96 -> 6.20):

| threads | prompt tok/s | generation tok/s | spread | ceiling | fraction |
| --- | --- | --- | --- | --- | --- |
| 4 | 11.30 | 11.15 | 1.0% | 16.63 | 67% |
| 6 | 14.07 | 14.02 | 8.2% | 18.77 | 75% |
| 7 | 14.65 | 14.73 | 1.7% | 19.30 | 76% |
| 8 | 15.03 | 15.23 | 0.9% | 19.19 | **79%** |

Peak RSS 2.82 GiB for the whole bench process, which loads the model once and
runs every configuration in it; a single `rust-llm run` at context 4096 with 8
threads peaks at 2.76 GiB, of which 2.52 GiB is the mapped model.

Honest notes on those two tables:

- The two disagree about the top of the curve: the first makes 6 threads the
  peak and the second makes 8 the peak and 30 per cent faster than itself at 6.
  Both ran under a load average between 4 and 6. The load average does not
  predict it on its own, because what matters is whether the other work on the
  box is competing for memory bandwidth or only for CPU.
- What every window agrees on: the curve rises steeply to 4 threads, 6, 7 and 8
  are within 10 per cent of each other, and 4 threads is 5 to 35 per cent behind
  the best setting. Nothing measured here is faster than the quota allows.
- 1 and 2 threads sit at 55 per cent of the ceiling, 8 threads at 79 per cent in
  the best window. The remaining fifth is not barrier time any more (see the
  profile below); it is the output head and everything per token that is not
  streaming weights.
- The 8-thread row of the first table has a 12.7 per cent spread, the widest in
  either table. Asking for the whole quota is the setting most exposed to
  whatever else runs on the box.

Where this started. On the first commit of this branch, same command and
configuration, before any threading change: 7.75 generation tok/s at 4 threads
(47 per cent of the ceiling) and 5.52 at 8 threads (29 per cent). Eight threads
was then the slowest useful setting and is now the fastest.

### The long-context configuration

Context 4096, 204-token prompt, 16 generated tokens, 3 repeats, load 4.63 -> 8.77:

| threads | prompt tok/s | generation tok/s | spread (gen) | peak RSS |
| --- | --- | --- | --- | --- |
| 4 | 11.18 | 11.17 | 3.6% | |
| 8 | 13.50 | 12.72 | 10.9% | 3.00 GiB for the process, both configurations |

Generation at context 4096 is within the spread of generation at context 1024,
which the profile predicts: attention is under 1 per cent of a token at these
lengths, so a four times larger KV cache costs almost nothing. The KV cache is
what the extra 240 MiB of RSS over the context-1024 run is.

M1's baseline for the same shape (a 384-token prompt, which does not fit the
bench text, so not the same row) was 6.68 prompt and 3.90 generation tok/s at 8
threads. The generation side of that is a factor 3.3 below what this branch
measures now.

## llama.cpp on the same file, at matching thread counts

Reference build: llama.cpp at commit `3259ab6`, binaries in
`/paperclip/models/ref/llama.cpp/build/bin` with `LD_LIBRARY_PATH` pointing at
that directory. Recipe in `docs/decisions.md` under "The reference harness". The
same GGUF file, so quantization is out of the comparison.

```sh
export LD_LIBRARY_PATH=/paperclip/models/ref/llama.cpp/build/bin
M=/paperclip/models/qwen3.5-4b-heretic/Qwen3.5-4B-heretic-Q4_K_M.gguf

# flags matched to what this runner does: no weight repacking, no flash attention
llama-bench -m $M -p 19 -n 16 -t 1,3,4,6,7,8 -r 3 --repack 0 -fa off

# the reference as it ships: repacking on, flash attention auto, f16 KV cache
llama-bench -m $M -p 19 -n 16 -t 1,3,4,6,7,8 -r 3
```

`-ctk f32 -ctv f32` is refused by `llama-bench` at this commit, so both
reference rows run an f16 KV cache where this runner keeps f32. At a 19-token
prompt and 16 generated tokens the KV cache is a few hundred KiB against
2.52 GiB of weights per token, so it is not what these rows differ by; the
per-layer comparison, which does need f32, uses `llama-eval-callback` and gets
it there.

Generation (`tg16`), three runs each, all at matching thread counts. Our column
is a `bench` sweep run in the same window as the reference's own defaults
(load 8.56 -> 9.68 against 7.41 -> 8.35), which is a busy window for all three.

| threads | reference, matched flags | reference, its defaults | this runner |
| --- | --- | --- | --- |
| 1 | 3.35 | 3.79 | 3.12 |
| 3 | 8.86 | 8.54 | 7.92 |
| 4 | 8.96 | 9.58 | 9.60 |
| 6 | 8.56 | 8.99 | 10.43 |
| 7 | 7.93 | 8.22 | 10.25 |
| 8 | 7.28 | 7.54 | 9.54 |

Prompt (`pp19` against our prompt tok/s), same runs:

| threads | reference, matched flags | reference, its defaults | this runner |
| --- | --- | --- | --- |
| 1 | 9.38 | 11.87 | 3.12 |
| 3 | 23.76 | 33.80 | 7.96 |
| 4 | 30.70 | 39.76 | 9.67 |
| 6 | 35.35 | 44.11 | 10.36 |
| 7 | 28.98 | 42.17 | 10.10 |
| 8 | 27.39 | 39.13 | 9.47 |

And at the long-context configuration (`-p 204 -n 16 -r 3`, reference defaults,
load 7.74 -> 7.80, against our context-4096 table above):

| threads | reference prompt | this runner prompt | reference generation | this runner generation |
| --- | --- | --- | --- | --- |
| 4 | 65.09 | 11.18 | 11.87 | 11.17 |
| 8 | 93.79 | 13.50 | 13.61 | 12.72 |

One note per place we are behind:

- **The prompt phase is a factor 4 to 7 behind, and that is the largest gap in
  this file.** The reference evaluates a prompt as a batch, so one pass over the
  weights produces many positions at once; this runner still pushes the prompt
  through the single-token path, so it pays the full 2.52 GiB per token. It is
  bounded by exactly the thing the generation path is bounded by, which is why
  our prompt and generation tok/s are equal to within the spread and the
  reference's are not. Batched prompt evaluation is the next child issue on this
  branch.
- **One thread is 10 to 20 per cent behind** (3.12 against 3.35 and 3.79). With
  one thread there is no scheduling left to win, so this is kernel work: the
  reference repacks weights into a layout its kernel likes even with `--repack 0`
  for some tensors, and its Q4_K kernel is better tuned than ours. Kernel tuning
  is the next child issue as well.
- **Generation past 4 threads we are ahead**, by 16 to 27 per cent at 6, 7 and 8,
  because the reference's own curve turns down there and ours does not. That is
  not a kernel win, it is the scheduling work on this branch. Best setting
  against best setting in one window, ours 10.43 at 6 threads against its 9.58 at
  4, we are 9 per cent ahead; in the quieter window where we reach 15.23 the gap
  is larger, but the reference was not re-measured in that window and this file
  does not pair numbers across windows.
- The reference's defaults beat its matched flags by 4 to 13 per cent on
  generation and 15 to 45 per cent on the prompt, which is the price of the
  comparison being fair rather than flattering.

At matching thread counts the reference is a peer and not a floor. The bar for
this runner is the bandwidth ceiling in the table above, not llama.cpp.

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

### Now, at the end of the threading work

```
profile: generation, 8 tokens, 8 threads, context 1024, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul               32.0      18.52    27.9
ffn: down matmul                      32.0      12.78    19.3
linear: qkv, gate, alpha, beta matmul    24.0      11.15    16.8
output head matmul                     1.0      10.95    16.5
linear: output matmul                 24.0       3.49     5.3
full: qg, k and v matmul               8.0       2.65     4.0
linear: delta-net recurrence          24.0       2.45     3.7
elementwise                          184.0       1.23     1.9
full: output matmul                    8.0       1.17     1.8
linear: convolution                   24.0       0.94     1.4
norms and rope                       121.0       0.58     0.9
full: attention                        8.0       0.48     0.7
embed                                  1.0       0.00     0.0
measured total                                  66.40   100.0
of which pool hand-out               161.0       1.21     1.8
of which pool join (last worker)     161.0       2.74     4.1
```

```
profile: generation, 8 tokens, 4 threads, context 1024, tier avx2+vnni
operator                           calls/t   ms/token       %
ffn: gate and up matmul               32.0      26.61    29.8
ffn: down matmul                      32.0      16.52    18.5
linear: qkv, gate, alpha, beta matmul    24.0      15.22    17.1
output head matmul                     1.0      14.18    15.9
linear: output matmul                 24.0       4.71     5.3
linear: delta-net recurrence          24.0       3.87     4.3
full: qg, k and v matmul               8.0       3.77     4.2
full: output matmul                    8.0       1.51     1.7
elementwise                          184.0       1.11     1.2
linear: convolution                   24.0       0.89     1.0
norms and rope                       121.0       0.54     0.6
full: attention                        8.0       0.23     0.3
embed                                  1.0       0.00     0.0
measured total                                  89.16   100.0
of which pool hand-out               161.0       0.16     0.2
of which pool join (last worker)     161.0       1.73     1.9
```

The prompt-phase table is the same table to within a per cent, because the
prompt runs through the same single-token path; `rust-llm profile` prints both.

What is left to win, in order: the four matmul groups are 84 per cent of a token
and they are streaming weights, so they are the ceiling itself. The output head
is 16 per cent on its own, a single 2560 x 248320 matmul over the whole embedding
table, the one matmul that cannot be fused with anything and the one place where
a batched prompt pass would pay most. Attention, the convolution, the delta-net
recurrence and the norms together are under 7 per cent at this context length,
so flash-style attention has to be judged at long contexts, not at 1024.

### Before the threading work, for comparison

From commit `7de450c`, the measurement surfaces before any change to the
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
embed                                  1.0       0.00     0.0
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
8 threads used to be slower than 4. It is 2.1 per cent at 4 threads and 5.9 per
cent at 8 now, and the per-tensor breakdown inside a group is gone because the
group is one job.

This is also why the table above has fewer rows: the projections that read the
same normalised input are one job, so the profile can no longer attribute time
to `ffn: gate` and `ffn: up` separately. The per-tensor split is this table.

## What the threading work moved

Each change landed with its own before/after pair, measured by alternating
passes against the previous commit's binary in one window. The commit bodies
carry the pairs and the gate margins.

| commit | change | 4 threads | 8 threads |
| --- | --- | --- | --- |
| `e1a2309` | the projections that share an input are one job, 281 jobs/token -> 161 | 8.17 -> 9.10 (+11%) | not a measurement in that window, spreads to 74% |
| `335f6c5` | workers claim row blocks from an atomic counter instead of a static split | 9.12 -> 9.79 (+7%) | 7.63 -> 8.86 (+16%) |
| `1cdf21b` | a sequence counter and a bounded spin instead of the two barriers | 9.64 -> 11.14 (+16%) | 9.36 -> 15.01 (+60%) |

Generation tok/s, context 1024, 19-token prompt, 16 generated tokens. The
absolute level differs between rows because each row is its own window; only the
pair within a row is comparable.

None of the three changes the arithmetic. The per-layer trace of a two-token
prompt is byte-identical across all three at 4 and at 7 threads, and both
correctness gates keep the four disagreements the fixture records at 0.0094,
0.0104, 0.0475 and 0.1262.

## What was measured and not kept

- **Thread affinity**, pinning workers to CPUs: not implemented. The profile
  stopped pointing at the scheduler once the hand-out was a sequence counter and
  the split was dynamic (the wait for the last worker is 1.7 ms of an 89 ms
  token at 4 threads), and the quota does not say which CPUs it will be given, so
  pinning here would be guessing at the host's placement on a box that is shared
  with other work.
- **A thread count below the quota as the default**: measured, does not hold.
  See `docs/decisions.md` under "Threading".
