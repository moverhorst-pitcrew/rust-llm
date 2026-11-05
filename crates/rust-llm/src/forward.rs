//! The hybrid forward pass, plus the caches and the scratch it runs on.
//!
//! Both caches and every buffer are allocated once, by [`Runner::new`], from
//! the config and a maximum batch size. The pass itself allocates nothing: it
//! quantizes activations into scratch, hands row ranges to the pool and writes
//! into buffers it already owns.
//!
//! [`Runner::forward_batch`] takes `t` tokens at once and is the only forward
//! pass there is; [`Runner::forward`] is it with `t = 1`. Reading a prompt as
//! batches is what makes it faster than generating: a weight row crosses the
//! memory bus once for the whole batch instead of once per token, and batch-1
//! generation is bounded by exactly that traffic. Every buffer is token-major,
//! `t` rows of the width the single-token path used, and everything elementwise
//! loops over the rows.
//!
//! What does *not* batch is the delta net's recurrence, which is sequential
//! over tokens by construction, and the depthwise convolution, which reads the
//! tokens before it. Both run token by token inside the batch, over the same
//! state in the same order, so the state a batch leaves behind is the state the
//! single-token path would have left: see `docs/decisions.md`, "Batched prompt
//! evaluation".

use std::time::Instant;

use llm_core::quant::Activations;
use llm_core::{Kernels, TensorView};

use crate::model::{Attn, Config, FullBlock, LinearBlock, Model};
use crate::pool::{Matvec, Pool};

/// Per-layer activations for the reference comparison, off unless asked for.
///
/// The names match the node names `llama-eval-callback` prints and so does the
/// shape of what is recorded: the element sum plus the first and last three
/// values, which is exactly what the reference dumps. The sum alone is a
/// misleading signal, because a 2560-wide residual cancels almost completely
/// and a per-element difference of 1e-7 shows up as 1e-3 of the sum; the
/// elements are what says whether two implementations agree. See
/// `scripts/compare-reference.py`.
#[derive(Debug, Default)]
pub struct Tracer {
    on: bool,
    nodes: Vec<Node>,
}

/// One recorded tensor.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub sum: f64,
    /// The first three and last three elements, in that order.
    pub edges: Vec<f32>,
}

impl Tracer {
    pub fn enable(&mut self) {
        self.on = true;
    }

    pub fn is_on(&self) -> bool {
        self.on
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Records one node. Returns immediately when tracing is off, which is why
    /// the allocations below cannot land on the generation path.
    #[inline]
    fn node(&mut self, name: &str, il: Option<usize>, v: &[f32]) {
        if !self.on {
            return;
        }
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let name = match il {
            Some(il) => format!("{name}-{il}"),
            None => name.to_string(),
        };
        let edge = 3.min(v.len());
        let mut edges = v[..edge].to_vec();
        edges.extend_from_slice(&v[v.len() - edge..]);
        self.nodes.push(Node { name, sum, edges });
    }
}

/// Where the time goes, per operator, off unless asked for.
///
/// `perf` is not installed in this container and `perf_event_paranoid` is 2,
/// so a sampling profiler was not an option here and these timers are what
/// produced the table in `docs/benchmarks.md`. Off they are one branch per
/// operator, which is the same thing [`Tracer`] costs; on they are two
/// `Instant::now()` calls per operator against the fifty-odd operators a token
/// goes through, which is under a tenth of a per cent of a token.
#[derive(Debug, Default)]
pub struct Profile {
    on: bool,
    /// Nanoseconds and calls, indexed by `Op as usize`.
    ops: [(u64, u64); Op::COUNT],
}

/// One line of the profile. Split per tensor where the tensors differ in
/// shape or in quantization type, because that is what a tuning change moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Op {
    Embed,
    /// Full block: the query-and-gate, key and value projections, one job.
    FullQkv,
    AttnOut,
    /// Scores, softmax and the weighted sum over the KV cache.
    Attention,
    /// Linear block: the q/k/v projection the convolution reads together with
    /// the output gate, `alpha` and `beta`, one job.
    LinearQkv,
    Conv,
    /// The delta-rule recurrence over the per-head state.
    DeltaNet,
    LinearOut,
    /// The feed-forward gate and up projections, one job.
    FfnGateUp,
    FfnDown,
    /// The output head over the tied embedding table.
    Head,
    /// Every RMS and L2 norm, including the per-head ones, and RoPE.
    Norms,
    /// Residual adds, SwiGLU, the sigmoid and SiLU gates, argmax-free.
    Elementwise,
}

impl Op {
    const COUNT: usize = 13;
    const ALL: [Op; Op::COUNT] = [
        Op::Embed,
        Op::FullQkv,
        Op::AttnOut,
        Op::Attention,
        Op::LinearQkv,
        Op::Conv,
        Op::DeltaNet,
        Op::LinearOut,
        Op::FfnGateUp,
        Op::FfnDown,
        Op::Head,
        Op::Norms,
        Op::Elementwise,
    ];

    fn label(self) -> &'static str {
        match self {
            Op::Embed => "embed",
            Op::FullQkv => "full: qg, k and v matmul",
            Op::AttnOut => "full: output matmul",
            Op::Attention => "full: attention",
            Op::LinearQkv => "linear: qkv, gate, alpha, beta matmul",
            Op::Conv => "linear: convolution",
            Op::DeltaNet => "linear: delta-net recurrence",
            Op::LinearOut => "linear: output matmul",
            Op::FfnGateUp => "ffn: gate and up matmul",
            Op::FfnDown => "ffn: down matmul",
            Op::Head => "output head matmul",
            Op::Norms => "norms and rope",
            Op::Elementwise => "elementwise",
        }
    }
}

impl Profile {
    pub fn enable(&mut self) {
        self.on = true;
    }

    pub fn is_on(&self) -> bool {
        self.on
    }

    pub fn clear(&mut self) {
        self.ops = [(0, 0); Op::COUNT];
    }

    /// The clock, or nothing at all when profiling is off.
    #[inline]
    fn start(&self) -> Option<Instant> {
        self.on.then(Instant::now)
    }

    #[inline]
    fn add(&mut self, op: Op, at: Option<Instant>) {
        if let Some(at) = at {
            let slot = &mut self.ops[op as usize];
            slot.0 += at.elapsed().as_nanos() as u64;
            slot.1 += 1;
        }
    }

    /// Everything recorded, longest first, as `(label, calls, nanoseconds)`.
    pub fn rows(&self) -> Vec<(&'static str, u64, u64)> {
        let mut rows: Vec<(&'static str, u64, u64)> = Op::ALL
            .iter()
            .map(|&op| {
                let (ns, calls) = self.ops[op as usize];
                (op.label(), calls, ns)
            })
            .filter(|r| r.1 > 0)
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        rows
    }

    pub fn total_ns(&self) -> u64 {
        self.ops.iter().map(|s| s.0).sum()
    }
}

/// The caches. Both are sized at load and never reallocated.
struct Caches {
    /// `[full block][k, v][kv head][position][head_dim]`.
    kv: Vec<f32>,
    /// `[linear block][channel][conv_kernel - 1]`, oldest sample first.
    conv: Vec<f32>,
    /// `[linear block][value head][d_state i][d_state j]`, so one `i` of a head
    /// is contiguous over `j` and [`delta_net`] can take a block of `j` at a
    /// time. The layout is internal: nothing outside the recurrence reads it,
    /// the reset only zeroes it and no tracer node dumps it.
    ssm: Vec<f32>,
    /// Tokens already in the caches, which is also the next token's position.
    pos: usize,
}

/// Every intermediate the pass writes to, each `max_batch` rows of the width
/// one token needs, token-major.
struct Buffers {
    x: Vec<f32>,
    xn: Vec<f32>,
    y: Vec<f32>,
    qkv: Vec<f32>,
    kb: Vec<f32>,
    vb: Vec<f32>,
    z: Vec<f32>,
    conv: Vec<f32>,
    heads: Vec<f32>,
    /// Per-head output, head-major (`[head][token][head width]`), because a
    /// head owns its whole batch: a value head owns one recurrent state and an
    /// attention head owns one walk over the KV cache, and neither can be split
    /// over the tokens. The gate that follows transposes it into `heads`.
    /// Both block kinds use it and they are never live at once; the two widths
    /// are the same, `d_inner` and `head_dim * n_head` both being 4096 here.
    head_major: Vec<f32>,
    scores: Vec<f32>,
    ffn_a: Vec<f32>,
    ffn_b: Vec<f32>,
    /// `alpha`/decay and `beta`, one scalar per value head each.
    decay: Vec<f32>,
    beta: Vec<f32>,
    logits: Vec<f32>,
}

/// Tokens per batch when the caller does not say.
///
/// Taken from the measured crossover rather than from a round number: see
/// `docs/benchmarks.md`, "What batched prompt evaluation moved".
pub const DEFAULT_BATCH: usize = 16;

/// A loaded model plus the threads, caches and scratch to run it.
pub struct Runner<'a> {
    model: &'a Model<'a>,
    pool: Pool,
    context: usize,
    max_batch: usize,
    caches: Caches,
    bufs: Buffers,
    acts: Activations,
    pub tracer: Tracer,
    pub profile: Profile,
}

impl<'a> Runner<'a> {
    /// Allocates for `context` positions, `threads` threads and batches of up
    /// to [`DEFAULT_BATCH`] tokens.
    pub fn new(model: &'a Model<'a>, context: usize, threads: usize) -> Self {
        Self::with_batch(model, context, threads, DEFAULT_BATCH)
    }

    /// The same with the maximum batch size spelled out, which is what
    /// `--batch` and the tests that compare batch sizes need.
    ///
    /// Every buffer is sized from `max_batch` here, so nothing in
    /// [`Runner::forward_batch`] allocates however long the batch is.
    pub fn with_batch(
        model: &'a Model<'a>,
        context: usize,
        threads: usize,
        max_batch: usize,
    ) -> Self {
        let c = &model.config;
        let context = context.max(1);
        // A batch longer than the context could never be run.
        let b = max_batch.clamp(1, context);
        let vocab = model.vocab();
        Self {
            pool: Pool::new(threads),
            context,
            max_batch: b,
            caches: Caches {
                kv: vec![0.0; c.n_full_blocks() * 2 * c.n_head_kv * context * c.head_dim],
                conv: vec![0.0; c.n_linear_blocks() * c.conv_dim() * (c.conv_kernel - 1)],
                ssm: vec![0.0; c.n_linear_blocks() * c.n_v_heads * c.d_state * c.d_state],
                pos: 0,
            },
            bufs: Buffers {
                x: vec![0.0; b * c.n_embd],
                xn: vec![0.0; b * c.n_embd],
                y: vec![0.0; b * c.n_embd],
                qkv: vec![0.0; b * c.qg_dim().max(c.conv_dim())],
                kb: vec![0.0; b * c.kv_dim()],
                vb: vec![0.0; b * c.kv_dim()],
                z: vec![0.0; b * c.d_inner],
                conv: vec![0.0; b * c.conv_dim()],
                heads: vec![0.0; b * c.attn_dim().max(c.d_inner)],
                head_major: vec![0.0; b * c.d_inner.max(c.attn_dim())],
                scores: vec![0.0; b * c.n_head * context],
                ffn_a: vec![0.0; b * c.n_ff],
                ffn_b: vec![0.0; b * c.n_ff],
                decay: vec![0.0; b * c.n_v_heads],
                beta: vec![0.0; b * c.n_v_heads],
                logits: vec![0.0; vocab],
            },
            acts: Activations::with_batch(c.n_embd.max(c.n_ff).max(c.d_inner).max(c.attn_dim()), b),
            model,
            tracer: Tracer::default(),
            profile: Profile::default(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.model.config
    }

    pub fn context(&self) -> usize {
        self.context
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn kernels(&self) -> &Kernels {
        &self.model.kernels
    }

    /// Position the next token will take, which is how many tokens the caches
    /// hold.
    pub fn pos(&self) -> usize {
        self.caches.pos
    }

    /// Clears both caches so the next token starts a fresh sequence.
    ///
    /// Zeroing is the whole reset: the delta-net recurrence starts from a zero
    /// state, the convolution from zero history, and the KV cache is only ever
    /// read up to `pos`.
    pub fn reset(&mut self) {
        self.caches.kv.fill(0.0);
        self.caches.conv.fill(0.0);
        self.caches.ssm.fill(0.0);
        self.caches.pos = 0;
        self.tracer.clear();
    }

    /// Tokens this runner can take in one [`Runner::forward_batch`].
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Runs one token and returns the logits for the next one.
    ///
    /// Panics if the context is full; the caller decides what to do about that
    /// long before a cache write could run off the end.
    pub fn forward(&mut self, token: u32) -> &[f32] {
        self.forward_batch(&[token])
    }

    /// Runs `tokens` as one batch and returns the logits for the token after
    /// the last of them.
    ///
    /// The batch takes the positions `pos..pos + tokens.len()`, so the caches
    /// and the recurrent state come out exactly as feeding the same tokens one
    /// at a time would have left them. Only the last token's logits are
    /// produced: the output head is a sixth of a token's work and a prompt
    /// throws every other position's logits away.
    ///
    /// Panics if the batch is longer than the scratch was sized for or does not
    /// fit the remaining context, both of which are the caller's to check.
    pub fn forward_batch(&mut self, tokens: &[u32]) -> &[f32] {
        let Self {
            model,
            pool,
            context,
            max_batch,
            caches,
            bufs,
            acts,
            tracer,
            profile,
        } = self;
        // Copied out of `self` so the borrow of the model outlives the mutable
        // borrows of the caches and the scratch rather than conflicting.
        let model: &'a Model<'a> = model;
        let c = &model.config;
        let context = *context;
        let t = tokens.len();
        let ne = c.n_embd;
        assert!(t > 0, "a batch of no tokens");
        assert!(
            t <= *max_batch,
            "batch of {t} tokens, but the scratch was sized for {max_batch}"
        );
        assert!(
            caches.pos + t <= context,
            "context of {context} tokens holds {} and cannot take {t} more",
            caches.pos
        );

        let mut ex = Exec {
            model,
            pool,
            context,
            t,
            acts,
            tracer,
            profile,
        };

        for (i, &token) in tokens.iter().enumerate() {
            ex.embed(token, &mut bufs.x[i * ne..][..ne]);
        }
        ex.tracer.node("model.input_embed", None, &bufs.x[..t * ne]);

        for il in 0..c.n_blocks {
            let at = ex.profile.start();
            for i in 0..t {
                rms_norm(
                    &bufs.x[i * ne..][..ne],
                    model.blocks[il].attn_norm,
                    c.rms_eps,
                    &mut bufs.xn[i * ne..][..ne],
                );
            }
            ex.profile.add(Op::Norms, at);
            ex.tracer.node("attn_norm", Some(il), &bufs.xn[..t * ne]);

            match &model.blocks[il].attn {
                Attn::Full(b) => {
                    let span = 2 * c.n_head_kv * context * c.head_dim;
                    let kv = &mut caches.kv[b.slot * span..(b.slot + 1) * span];
                    ex.full_attn(b, il, caches.pos, kv, bufs);
                }
                Attn::Linear(b) => {
                    let conv_span = c.conv_dim() * (c.conv_kernel - 1);
                    let ssm_span = c.n_v_heads * c.d_state * c.d_state;
                    let conv = &mut caches.conv[b.slot * conv_span..(b.slot + 1) * conv_span];
                    let ssm = &mut caches.ssm[b.slot * ssm_span..(b.slot + 1) * ssm_span];
                    ex.linear_attn(b, il, conv, ssm, bufs);
                }
            }

            let at = ex.profile.start();
            for (x, y) in bufs.x[..t * ne].iter_mut().zip(&bufs.y[..t * ne]) {
                *x += y;
            }
            ex.profile.add(Op::Elementwise, at);
            ex.tracer.node("attn_residual", Some(il), &bufs.x[..t * ne]);

            let at = ex.profile.start();
            for i in 0..t {
                rms_norm(
                    &bufs.x[i * ne..][..ne],
                    model.blocks[il].post_attn_norm,
                    c.rms_eps,
                    &mut bufs.xn[i * ne..][..ne],
                );
            }
            ex.profile.add(Op::Norms, at);
            ex.tracer
                .node("attn_post_norm", Some(il), &bufs.xn[..t * ne]);
            ex.ffn(&model.blocks[il], il, bufs);
            let at = ex.profile.start();
            for (x, y) in bufs.x[..t * ne].iter_mut().zip(&bufs.y[..t * ne]) {
                *x += y;
            }
            ex.profile.add(Op::Elementwise, at);
            ex.tracer.node("l_out", Some(il), &bufs.x[..t * ne]);
        }

        let at = ex.profile.start();
        rms_norm(
            &bufs.x[(t - 1) * ne..][..ne],
            model.output_norm,
            c.rms_eps,
            &mut bufs.xn[..ne],
        );
        ex.profile.add(Op::Norms, at);
        ex.tracer.node("result_norm", None, &bufs.xn[..ne]);
        // The batch is over here: what follows is one row, the last token's.
        ex.t = 1;
        // Tied embeddings: the table that read the token in is the output head.
        ex.matmul(
            Op::Head,
            &model.token_embd,
            &bufs.xn[..ne],
            &mut bufs.logits,
        );
        ex.tracer.node("result_output", None, &bufs.logits);

        caches.pos += t;
        &bufs.logits
    }

    /// The last logits, for a caller that wants them again without a step.
    pub fn logits(&self) -> &[f32] {
        &self.bufs.logits
    }

    /// Turns on the per-operator timers and the pool's barrier accounting.
    pub fn enable_profile(&mut self) {
        self.profile.enable();
        self.pool.measure(true);
    }

    /// Throws the recorded time away, which is how the prompt phase and the
    /// generation phase are kept apart.
    pub fn clear_profile(&mut self) {
        self.profile.clear();
        self.pool.stats().clear();
    }

    /// The profile table for the `tokens` tokens since the last clear.
    ///
    /// The two pool rows are *inside* the matmul rows above them, not extra:
    /// a matmul is a job, and the time the submitter spends on the other
    /// threads is part of what that matmul cost. They are listed separately
    /// because they are the part a scheduling change can remove: handing the
    /// job out and waking whoever parked, and then waiting for the slowest
    /// worker after the submitter has finished its own share.
    pub fn profile_report(&self, what: &str, tokens: usize) -> String {
        let mut out = String::new();
        let per = |ns: u64| ns as f64 / tokens as f64 / 1e6;
        let total = self.profile.total_ns();
        out.push_str(&format!(
            "profile: {what}, {tokens} tokens, {} threads, context {}, tier {}\n",
            self.threads(),
            self.context,
            self.kernels().tier().as_str(),
        ));
        out.push_str(&format!(
            "{:34}{:>8}{:>11}{:>8}\n",
            "operator", "calls/t", "ms/token", "%"
        ));
        for (label, calls, ns) in self.profile.rows() {
            out.push_str(&format!(
                "{label:34}{:>8.1}{:>11.2}{:>8.1}\n",
                calls as f64 / tokens as f64,
                per(ns),
                ns as f64 / total as f64 * 100.0,
            ));
        }
        out.push_str(&format!(
            "{:34}{:>8}{:>11.2}{:>8.1}\n",
            "measured total",
            "",
            per(total),
            100.0
        ));
        let (jobs, wake, join) = self.pool.stats().read();
        if jobs > 0 {
            for (label, ns) in [
                ("of which pool hand-out", wake),
                ("of which pool join (last worker)", join),
            ] {
                out.push_str(&format!(
                    "{label:34}{:>8.1}{:>11.2}{:>8.1}\n",
                    jobs as f64 / tokens as f64,
                    per(ns),
                    ns as f64 / total as f64 * 100.0,
                ));
            }
        }
        out
    }
}

/// The model, the threads and the two pieces of state every step needs.
///
/// Bundled so the block functions can take the buffers they write as separate
/// arguments: the activation scratch is read by the kernels while the output is
/// written, and those cannot be two fields of one borrowed struct.
struct Exec<'m, 'r> {
    model: &'m Model<'m>,
    pool: &'r Pool,
    context: usize,
    /// Tokens in the batch being run, which is how wide every buffer is.
    t: usize,
    acts: &'r mut Activations,
    tracer: &'r mut Tracer,
    profile: &'r mut Profile,
}

impl Exec<'_, '_> {
    /// One row of the embedding table, expanded to f32.
    fn embed(&mut self, token: u32, out: &mut [f32]) {
        let at_ = self.profile.start();
        let t = &self.model.token_embd;
        let k = t.dims[0] as usize;
        let stride = t.ty.bytes_for(k as u64).expect("row size fits in a u64") as usize;
        let at = token as usize;
        assert!(
            at < t.dims[1] as usize,
            "token {token} is outside the vocabulary"
        );
        llm_core::dequantize_row(t.ty, &t.data[at * stride..(at + 1) * stride], out);
        self.profile.add(Op::Embed, at_);
    }

    /// `out = W x`, with the rows of `W` split over the pool.
    fn matmul(&mut self, op: Op, w: &TensorView<'_>, x: &[f32], out: &mut [f32]) {
        self.matmul_group(op, x, [(w, out)]);
    }

    /// Several projections of the same input, as one job and one activation
    /// quantization.
    ///
    /// Every weight here has to read `x`, which is what makes the grouping
    /// free: the rows of the group are one range for the split to divide, so
    /// this is one pair of barriers where the separate matmuls were `N`, and
    /// the activations are quantized once. It changes no result, because a row
    /// is still one worker's work from start to finish.
    ///
    /// `x` is `self.t` rows of `K` and every output is `self.t` rows of that
    /// weight's row count, token-major.
    fn matmul_group<const N: usize>(
        &mut self,
        op: Op,
        x: &[f32],
        group: [(&TensorView<'_>, &mut [f32]); N],
    ) {
        let at = self.profile.start();
        let k = x.len() / self.t;
        self.acts.quantize_batch(x, k);
        let t = self.t;
        let group = group.map(|(w, out)| {
            debug_assert_eq!(w.dims[0] as usize, k);
            debug_assert_eq!(w.dims[1] as usize * t, out.len());
            Matvec {
                ty: w.ty,
                w: w.data,
                out,
            }
        });
        self.pool
            .matvec_group(&self.model.kernels, k, self.acts, group)
            .expect("every weight type was checked when the model loaded");
        self.profile.add(op, at);
    }

    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
    fn ffn(&mut self, block: &crate::model::Block<'_>, il: usize, bufs: &mut Buffers) {
        let c = &self.model.config;
        let (t, ne, n_ff) = (self.t, c.n_embd, c.n_ff);
        let Buffers {
            xn,
            y,
            ffn_a,
            ffn_b,
            ..
        } = bufs;
        let (ffn_a, ffn_b) = (&mut ffn_a[..t * n_ff], &mut ffn_b[..t * n_ff]);
        self.matmul_group(
            Op::FfnGateUp,
            &xn[..t * ne],
            [(&block.ffn_gate, ffn_a), (&block.ffn_up, ffn_b)],
        );
        self.tracer.node("ffn_gate", Some(il), ffn_a);
        self.tracer.node("ffn_up", Some(il), ffn_b);
        let at = self.profile.start();
        for (a, b) in ffn_a.iter_mut().zip(&*ffn_b) {
            *a = silu(*a) * b;
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("ffn_swiglu", Some(il), ffn_a);
        self.matmul(Op::FfnDown, &block.ffn_down, ffn_a, &mut y[..t * ne]);
        self.tracer.node("ffn_out", Some(il), &y[..t * ne]);
    }

    /// GQA attention with per-head q/k RMSNorm, partial RoPE and an output
    /// gate taken from the second half of the query projection.
    ///
    /// A batch takes the positions `pos..pos + t`. Its keys and values go into
    /// the cache before any score is formed, and token `i` then attends over
    /// `pos + i + 1` positions: everything that was already cached plus the
    /// tokens of this batch up to and including itself. That is the causal
    /// mask, and it needs no masking step because the score loop simply stops
    /// there.
    fn full_attn(
        &mut self,
        b: &FullBlock<'_>,
        il: usize,
        pos: usize,
        kv: &mut [f32],
        bufs: &mut Buffers,
    ) {
        let model = self.model;
        let c = &model.config;
        let (hd, n_head, n_kv_head) = (c.head_dim, c.n_head, c.n_head_kv);
        let group = n_head / n_kv_head;
        let context = self.context;
        let t = self.t;
        let (ne, qg_dim, kv_dim, attn_dim) = (c.n_embd, c.qg_dim(), c.kv_dim(), c.attn_dim());
        let Buffers {
            xn,
            y,
            qkv,
            kb,
            vb,
            heads,
            head_major,
            scores,
            ..
        } = bufs;
        let qg = &mut qkv[..t * qg_dim];
        let kb = &mut kb[..t * kv_dim];
        let vb = &mut vb[..t * kv_dim];

        self.matmul_group(
            Op::FullQkv,
            &xn[..t * ne],
            [(&b.qg, qg), (&b.v, vb), (&b.k, kb)],
        );
        self.tracer.node("Qcur_full", Some(il), qg);
        self.tracer.node("Vcur", Some(il), vb);
        self.tracer.node("Kcur", Some(il), kb);

        // The query occupies the first half of each head's slice of `attn_q`
        // and the gate the second half, so the query is strided.
        let at = self.profile.start();
        for i in 0..t {
            let p = pos + i;
            let qg = &mut qg[i * qg_dim..][..qg_dim];
            for h in 0..n_head {
                let q = &mut qg[h * 2 * hd..h * 2 * hd + hd];
                rms_norm_in_place(q, b.q_norm, c.rms_eps);
                rope(q, p, c.n_rot, c.rope_theta);
            }
            let kb = &mut kb[i * kv_dim..][..kv_dim];
            for h in 0..n_kv_head {
                let k = &mut kb[h * hd..(h + 1) * hd];
                rms_norm_in_place(k, b.k_norm, c.rms_eps);
                rope(k, p, c.n_rot, c.rope_theta);
            }
        }
        self.profile.add(Op::Norms, at);
        self.tracer.node("Kcur_roped", Some(il), kb);

        // Append the batch's keys and values, then read the cache shared.
        let at = self.profile.start();
        let (k_cache, v_cache) = kv.split_at_mut(n_kv_head * context * hd);
        for i in 0..t {
            for h in 0..n_kv_head {
                let to = (h * context + pos + i) * hd;
                let from = i * kv_dim + h * hd;
                k_cache[to..to + hd].copy_from_slice(&kb[from..from + hd]);
                v_cache[to..to + hd].copy_from_slice(&vb[from..from + hd]);
            }
        }
        let (k_cache, v_cache) = (&*k_cache, &*v_cache);
        self.profile.add(Op::Elementwise, at);

        let at = self.profile.start();
        let scale = 1.0 / (hd as f32).sqrt();
        let qg = &*qg;
        // One chunk per head, holding that head's whole batch: the batch walks
        // the same keys and values, so what the chunk owns is one walk over
        // them. The output comes out head-major and the gate below transposes
        // it.
        self.pool.chunks(
            &mut head_major[..n_head * t * hd],
            t * hd,
            &mut scores[..n_head * t * context],
            t * context,
            |h, out, scores| {
                let kh = h / group;
                attend_batch(
                    &qg[h * 2 * hd..],
                    qg_dim,
                    &k_cache[kh * context * hd..],
                    &v_cache[kh * context * hd..],
                    t,
                    pos,
                    hd,
                    scale,
                    context,
                    scores,
                    out,
                );
            },
        );
        self.profile.add(Op::Attention, at);
        self.tracer
            .node("attn_pregate", Some(il), &head_major[..t * attn_dim]);

        let at = self.profile.start();
        for i in 0..t {
            for h in 0..n_head {
                let gate = &qg[i * qg_dim + h * 2 * hd + hd..][..hd];
                let from = &head_major[(h * t + i) * hd..][..hd];
                let into = &mut heads[i * attn_dim + h * hd..][..hd];
                for ((o, &g), &a) in into.iter_mut().zip(gate).zip(from) {
                    *o = a * sigmoid(g);
                }
            }
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer
            .node("attn_gated", Some(il), &heads[..t * attn_dim]);

        self.matmul(Op::AttnOut, &b.o, &heads[..t * attn_dim], &mut y[..t * ne]);
        self.tracer.node("attn_output", Some(il), &y[..t * ne]);
    }

    /// The gated delta net: a depthwise causal convolution over the q/k/v
    /// projection, then the delta rule against a per-head recurrent state.
    ///
    /// The projections, the norms and the output batch; the convolution and the
    /// recurrence do not, and run token by token inside the batch. The
    /// convolution is token-sequential because tap `j` of token `i` is token
    /// `i - kernel + 1 + j`, which inside a batch is an earlier row of the same
    /// batch and before it is the history buffer; running it in order over the
    /// same rolling state is what makes the two the same thing. The recurrence
    /// is token-sequential because the state token `i` reads is the state token
    /// `i - 1` wrote. The parallelism there is over value heads, each of which
    /// owns one state and runs the whole batch over it, so a batch is one pool
    /// job where it used to be `t` of them.
    fn linear_attn(
        &mut self,
        b: &LinearBlock<'_>,
        il: usize,
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
        bufs: &mut Buffers,
    ) {
        let model = self.model;
        let c = &model.config;
        let (s, n_kh, n_vh) = (c.d_state, c.n_k_heads, c.n_v_heads);
        let (ne, key_dim, conv_dim, d_inner) = (c.n_embd, c.key_dim(), c.conv_dim(), c.d_inner);
        let t = self.t;
        let Buffers {
            xn,
            y,
            qkv,
            z,
            conv,
            heads,
            head_major,
            decay,
            beta,
            ..
        } = bufs;
        let qkv = &mut qkv[..t * conv_dim];
        let z = &mut z[..t * d_inner];
        let decay = &mut decay[..t * n_vh];
        let beta = &mut beta[..t * n_vh];
        let conv = &mut conv[..t * conv_dim];

        self.matmul_group(
            Op::LinearQkv,
            &xn[..t * ne],
            [
                (&b.qkv, qkv),
                (&b.gate, z),
                (&b.alpha, decay),
                (&b.beta, beta),
            ],
        );
        self.tracer.node("linear_attn_qkv_mixed", Some(il), qkv);
        self.tracer.node("z", Some(il), z);
        self.tracer.node("alpha", Some(il), decay);
        let at = self.profile.start();
        for row in decay.chunks_exact_mut(n_vh) {
            for (d, (&bias, &a)) in row.iter_mut().zip(b.dt_bias.iter().zip(b.a)) {
                // `ssm_a` is already `-exp(A_log)`, so this is the log of the
                // decay and `exp` of it lands in (0, 1).
                *d = softplus(*d + bias) * a;
            }
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("gate", Some(il), decay);

        let at = self.profile.start();
        for v in beta.iter_mut() {
            *v = sigmoid(*v);
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("beta_sigmoid", Some(il), beta);

        let at = self.profile.start();
        for i in 0..t {
            depthwise_conv(
                &qkv[i * conv_dim..][..conv_dim],
                b.conv1d,
                c.conv_kernel,
                conv_state,
                &mut conv[i * conv_dim..][..conv_dim],
            );
        }
        self.profile.add(Op::Conv, at);
        self.tracer.node("conv_output_raw", Some(il), conv);
        let at = self.profile.start();
        for v in conv.iter_mut() {
            *v = silu(*v);
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("conv_output_silu", Some(il), conv);

        // q and k are L2-normalised per head; v is used as it comes out.
        let at = self.profile.start();
        for row in conv.chunks_exact_mut(conv_dim) {
            for h in 0..2 * n_kh {
                l2_norm_in_place(&mut row[h * s..(h + 1) * s], c.rms_eps);
            }
        }
        let conv = &*conv;
        self.profile.add(Op::Norms, at);

        let at = self.profile.start();
        let scale = 1.0 / (s as f32).sqrt();
        let decay = &*decay;
        let beta = &*beta;
        // One chunk per value head, holding that head's whole batch: the state
        // is sequential over tokens, so the tokens cannot be split, and the
        // readout therefore comes out head-major and is transposed by the
        // gated norm below.
        self.pool.chunks(
            &mut head_major[..n_vh * t * s],
            t * s,
            ssm_state,
            s * s,
            |vh, out, state| {
                // Several value heads share one key head, taken modulo the key
                // head count as the reference broadcasts it.
                let kh = vh % n_kh;
                for i in 0..t {
                    let row = &conv[i * conv_dim..][..conv_dim];
                    let q = &row[kh * s..][..s];
                    let k = &row[key_dim + kh * s..][..s];
                    let v = &row[2 * key_dim + vh * s..][..s];
                    let g = decay[i * n_vh + vh].exp();
                    let beta = beta[i * n_vh + vh];
                    let out = &mut out[i * s..][..s];
                    delta_net(state, q, k, v, g, beta, scale, out);
                }
            },
        );
        self.profile.add(Op::DeltaNet, at);
        // Head-major and token-major are the same thing at `t = 1`, which is
        // what `trace` runs, so this node is still the vector the reference
        // prints.
        self.tracer
            .node("attn_output", Some(il), &head_major[..t * d_inner]);

        // The gated norm: RMSNorm per value head, then the sigmoid-linear
        // gate. This is also where the readout turns from head-major into the
        // token-major layout the output projection reads.
        let at = self.profile.start();
        for i in 0..t {
            for h in 0..n_vh {
                let head = &mut heads[i * d_inner + h * s..][..s];
                rms_norm(&head_major[(h * t + i) * s..][..s], b.norm, c.rms_eps, head);
                for (o, &g) in head.iter_mut().zip(&z[i * d_inner + h * s..][..s]) {
                    *o *= silu(g);
                }
            }
        }
        self.profile.add(Op::Norms, at);
        self.tracer
            .node("final_output", Some(il), &heads[..t * d_inner]);

        self.matmul(
            Op::LinearOut,
            &b.out,
            &heads[..t * d_inner],
            &mut y[..t * ne],
        );
        self.tracer.node("linear_attn_out", Some(il), &y[..t * ne]);
    }
}

/// `out = x / sqrt(mean(x^2) + eps) * w`.
fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let scale = rms_scale(x, eps);
    for ((o, &v), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * wi;
    }
}

fn rms_norm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let scale = rms_scale(x, eps);
    for (v, &wi) in x.iter_mut().zip(w) {
        *v = *v * scale * wi;
    }
}

/// The reference's arithmetic, step for step: the squares are formed in f32
/// and accumulated in f64, the mean is rounded back to f32 before `eps` is
/// added, and the root is an f32 root.
///
/// Every step here is a deliberate copy of the reference rather than the
/// obvious thing. Quantizing activations to int8 is a discontinuity: a
/// difference of 1e-7 in one activation flips a quantization level and moves
/// every output of that matmul by about 2e-3, which a few layers later picks a
/// different token out of a near-tie. Accumulating the squares in f64, which
/// is strictly more accurate, costs the token-for-token gate.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// `x / sqrt(sum(x^2) + eps)`, the delta net's q/k normalisation.
///
/// It is an L2 norm, not an RMS norm: the reference reaches it by calling RMS
/// norm with `eps / n` and then scaling by `1 / sqrt(n)`, which cancels the
/// mean and leaves `eps` inside the root. The two scalings are applied
/// separately here for the same reason as in [`rms_scale`]: folding them into
/// one multiply is a different rounding.
fn l2_norm_in_place(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let scale = rms_scale(x, eps / n);
    let shrink = 1.0 / n.sqrt();
    for v in x.iter_mut() {
        *v = (*v * scale) * shrink;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `log(1 + exp(x))`, switching to the identity where `exp` would overflow.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Positions a tile of the KV walk covers.
///
/// A tile of keys is `TILE * head_dim` floats, 128 KiB at this model's head
/// dimension of 256, so a tile of keys and the tile of values that follows it
/// both fit a core's 2 MiB L2 several times over while every query of the batch
/// goes over them.
const TILE: usize = 128;

/// One head's batch of `t` queries against the KV cache, tiled over the cached
/// positions.
///
/// `q` starts at this head's query inside the first token's projection and
/// `q_stride` steps to the next token's; `k` and `v` are this KV head's whole
/// span of the cache, position-major, and `row` is the stride of a score row.
/// Query `i` is the token at position `pos + i` and attends over `pos + i + 1`
/// positions: everything already cached plus the batch up to and including
/// itself. That is the causal mask, and it is a bound on the loops rather than
/// a masking step.
///
/// The tiling is the point. A head's span of the cache is 8 MiB at a context of
/// 4096, so it does not sit in the shared 36 MiB L3 next to the seven other
/// heads being computed at the same time, and a batch that walked it per token
/// read it from memory `t` times over. Tiled, the tile is read once for the
/// whole batch.
///
/// What is *not* here is the running maximum and running sum of a flash-style
/// attention. The score rows stay materialised, one per query, and that is a
/// deliberate choice measured in `docs/benchmarks.md`: a row is 16 KiB at a
/// context of 4096, it never leaves L2, and keeping it means the softmax still
/// sees the whole row at once. An incremental softmax would be a different
/// rounding of the one operator the correctness gate is most sensitive to, for
/// no measured gain.
#[allow(clippy::too_many_arguments)]
fn attend_batch(
    q: &[f32],
    q_stride: usize,
    k: &[f32],
    v: &[f32],
    t: usize,
    pos: usize,
    hd: usize,
    scale: f32,
    row: usize,
    scores: &mut [f32],
    out: &mut [f32],
) {
    // The last query of the batch has the longest row; every other one stops
    // earlier, which is the mask.
    let n_kv = pos + t;
    for base in (0..n_kv).step_by(TILE) {
        let tile = (base + TILE).min(n_kv);
        for i in 0..t {
            let upto = tile.min(pos + i + 1);
            let q = &q[i * q_stride..][..hd];
            for p in base..upto {
                scores[i * row + p] = dot(q, &k[p * hd..][..hd]) * scale;
            }
        }
    }
    for i in 0..t {
        softmax(&mut scores[i * row..][..pos + i + 1]);
    }
    out[..t * hd].fill(0.0);
    for base in (0..n_kv).step_by(TILE) {
        let tile = (base + TILE).min(n_kv);
        for i in 0..t {
            let upto = tile.min(pos + i + 1);
            let out = &mut out[i * hd..][..hd];
            for p in base..upto {
                let w = scores[i * row + p];
                for (o, &vd) in out.iter_mut().zip(&v[p * hd..][..hd]) {
                    *o += w * vd;
                }
            }
        }
    }
}

/// In-place softmax, with the normalising sum accumulated in f64 and the
/// reciprocal rounded to f32 before it is applied, which is what the reference
/// does.
fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Partial rotary embedding over the first `n_rot` of the head's dimensions,
/// pairing `i` with `i + n_rot / 2`; the rest of the head passes through.
///
/// The file asks for mrope with sections `[11, 11, 10, 0]`, which on a
/// text-only path is this: all three mrope axes carry the same position, and
/// with a zero fourth section no dimension pair is left for the fourth axis.
/// The tests pin that down rather than taking it on trust.
fn rope(x: &mut [f32], pos: usize, n_rot: usize, theta_base: f32) {
    let half = n_rot / 2;
    let theta_scale = theta_base.powf(-2.0 / n_rot as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (cos, sin) = (theta.cos(), theta.sin());
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
        theta *= theta_scale;
    }
}

/// Output components the delta-net recurrence takes at a time.
///
/// Measured, not assumed, on this model's `d_state` of 128: four binaries
/// alternated over a 1024-token prompt at 8 threads, tier `avx2+vnni`, both
/// orders, in `ms/token` of the prompt phase.
///
/// | block | this operator | whole token |
/// | --- | --- | --- |
/// | scalar | 2.04 | 21.80 |
/// | 16 | 0.96 | 20.93 |
/// | 32 | 0.84 | 21.14 |
/// | 64 | 0.87 | 20.72 |
///
/// 64 it is, on the whole token rather than on this operator's own row: 16 is
/// too narrow to amortise the load of `k[i]` and `q[i]` over, and 32 is the
/// fastest here by 0.03 ms but reproducibly costs the convolution next to it
/// 0.2 ms (0.78 scalar, 0.83 at 16, 0.99 at 32, 0.86 at 64). The convolution's
/// loop is untouched by any of this, and the cost does not grow with the number
/// of blocks a head is walked in, which is what a cache explanation would
/// predict, so it is the code layout of that one build and not something to
/// tune against. A block of 64 also walks a head in two passes rather than
/// eight, and its totals are the steadiest of the four (20.70 to 20.86 over
/// four runs).
const J_BLOCK: usize = 64;

/// One token through one value head's delta-net recurrence, blocked over the
/// output components.
///
/// `state` is this head's `[i][j]` matrix, `q`, `k` and `v` are its `d_state`
/// inputs for this token, `g` is the decay and `beta` the delta's gate. The
/// state is read, updated and read again in place, which is why a batch cannot
/// be split over its tokens.
///
/// **This is bit-identical to the scalar loop it replaces and has to stay that
/// way.** The two reductions a component needs, `sk` over `i` and the readout
/// `o` over `i`, are serial f32 sums that cannot be reassociated; they are not
/// what is vectorized here. The `d_state` components `j` are independent of
/// each other, each with its own `sk`, its own delta and its own readout, so
/// blocking over `j` puts one whole scalar sequence in every lane, in the
/// original order. `tests::the_blocked_recurrence_matches_one_component_at_a_time`
/// is what holds that down.
#[allow(clippy::too_many_arguments)]
fn delta_net(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    scale: f32,
    out: &mut [f32],
) {
    let s = out.len();
    let (mut sk, mut d, mut o) = ([0.0f32; J_BLOCK], [0.0f32; J_BLOCK], [0.0f32; J_BLOCK]);
    for j0 in (0..s).step_by(J_BLOCK) {
        let w = J_BLOCK.min(s - j0);
        let (sk, d, o) = (&mut sk[..w], &mut d[..w], &mut o[..w]);
        sk.fill(0.0);
        o.fill(0.0);
        // Pass one: decay the block's column of the state and contract it with
        // `k`, one `i` per step so every lane accumulates in ascending `i`.
        for (i, &ki) in k.iter().enumerate() {
            for (r, a) in state[i * s + j0..][..w].iter_mut().zip(sk.iter_mut()) {
                *r *= g;
                *a += *r * ki;
            }
        }
        for ((dj, &vj), &skj) in d.iter_mut().zip(&v[j0..][..w]).zip(sk.iter()) {
            *dj = (vj - skj) * beta;
        }
        // Pass two: add the delta's outer product and read out against `q`,
        // again one `i` per step and against the value just written.
        for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
            for ((r, &dj), oj) in state[i * s + j0..][..w]
                .iter_mut()
                .zip(d.iter())
                .zip(o.iter_mut())
            {
                *r += dj * ki;
                *oj += *r * qi;
            }
        }
        for (ot, &oj) in out[j0..][..w].iter_mut().zip(o.iter()) {
            *ot = oj * scale;
        }
    }
}

/// Depthwise causal convolution over `conv_kernel` taps, in place over the
/// per-channel history.
///
/// `w` is `[conv_kernel, channels]`, so one channel's taps are contiguous.
/// `state` holds `conv_kernel - 1` samples per channel, oldest first, and comes
/// out holding the newest ones.
fn depthwise_conv(x: &[f32], w: &[f32], kernel: usize, state: &mut [f32], out: &mut [f32]) {
    let hist = kernel - 1;
    for (c, (o, &xc)) in out.iter_mut().zip(x).enumerate() {
        let taps = &w[c * kernel..(c + 1) * kernel];
        let h = &mut state[c * hist..(c + 1) * hist];
        let mut sum = xc * taps[hist];
        for (j, &hv) in h.iter().enumerate() {
            sum += hv * taps[j];
        }
        *o = sum;
        h.rotate_left(1);
        h[hist - 1] = xc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Attention written the straightforward way, as the tiled version in the
    /// product is checked against.
    ///
    /// One query, one pass over the keys into a score row, one softmax, one
    /// pass over the values. This is what [`attend_batch`] has to agree with,
    /// and it lives here rather than in the pass for the same reason the
    /// scalar dequantizer lives in the kernel tests: a reference that shares
    /// the tiling and the loop order of the thing it checks would only assert
    /// that the code agrees with itself.
    fn attend(q: &[f32], k: &[f32], v: &[f32], n_kv: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = vec![0.0f32; n_kv];
        for (p, s) in scores.iter_mut().enumerate() {
            *s = dot(q, &k[p * hd..][..hd]) * scale;
        }
        softmax(&mut scores);
        out.fill(0.0);
        for (p, &w) in scores.iter().enumerate() {
            for (o, &vd) in out.iter_mut().zip(&v[p * hd..][..hd]) {
                *o += w * vd;
            }
        }
    }

    /// The recurrence written one output component at a time, as [`delta_net`]
    /// is checked against.
    ///
    /// This is the loop the blocked version replaced, down to the `[j][i]`
    /// state layout it read: one row of the state per component, decayed and
    /// contracted with `k`, then updated and read out against `q`. Keeping the
    /// old layout here rather than indexing the new one is the point, because a
    /// reference that shared the blocking and the walk order of the thing it
    /// checks would only assert that the code agrees with itself.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_scalar(
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: f32,
        beta: f32,
        scale: f32,
        out: &mut [f32],
    ) {
        let s = out.len();
        for (j, srow) in state.chunks_exact_mut(s).enumerate() {
            let mut sk = 0.0f32;
            for (r, &ki) in srow.iter_mut().zip(k) {
                *r *= g;
                sk += *r * ki;
            }
            let d = (v[j] - sk) * beta;
            let mut o = 0.0f32;
            for ((r, &ki), &qi) in srow.iter_mut().zip(k).zip(q) {
                *r += d * ki;
                o += *r * qi;
            }
            out[j] = o * scale;
        }
    }

    /// Blocking the recurrence over its output components has to give what one
    /// component at a time gives, **bit for bit**.
    ///
    /// A tolerance would be the wrong check, for the same reason as in
    /// [`the_tiled_attention_matches_one_query_at_a_time`]: this model's output
    /// is discontinuous in the arithmetic, so a difference too small to see in
    /// a readout is one flipped int8 level away from a different token. The
    /// whole claim of the rewrite is that each lane runs the original scalar
    /// sequence in the original order, and `==` is the only check that says so.
    ///
    /// Several tokens per head, because the state carries from one to the next:
    /// a blocking bug that touched an element twice or missed one would show in
    /// the *second* token's readout and not the first. The shapes are this
    /// model's own, `d_state` 128 over 32 value heads, plus `d_state` 32 for the
    /// tiny test model and a 40 that is not a multiple of [`J_BLOCK`], so the
    /// remainder block is covered too.
    #[test]
    fn the_blocked_recurrence_matches_one_component_at_a_time() {
        let mut rand = noise(0xde17_a4e7);
        for (s, heads, tokens) in [(128usize, 32usize, 4usize), (32, 1, 3), (40, 2, 3)] {
            for h in 0..heads {
                let mut blocked = vec![0.0f32; s * s];
                let mut scalar = vec![0.0f32; s * s];
                let scale = 1.0 / (s as f32).sqrt();
                for i in 0..tokens {
                    let q: Vec<f32> = (0..s).map(|_| rand()).collect();
                    let k: Vec<f32> = (0..s).map(|_| rand()).collect();
                    let v: Vec<f32> = (0..s).map(|_| rand()).collect();
                    // The ranges the real inputs land in: the decay is `exp` of
                    // a negative number and `beta` comes out of a sigmoid.
                    let g = (rand().abs() * -2.0).exp();
                    let beta = sigmoid(rand() * 4.0);
                    let mut got = vec![0.0f32; s];
                    let mut want = vec![0.0f32; s];
                    delta_net(&mut blocked, &q, &k, &v, g, beta, scale, &mut got);
                    // Transposed, because the reference keeps the old layout.
                    delta_net_scalar(&mut scalar, &q, &k, &v, g, beta, scale, &mut want);
                    assert_eq!(got, want, "d_state {s} head {h} token {i}: readout");
                    for j in 0..s {
                        for ii in 0..s {
                            assert_eq!(
                                blocked[ii * s + j],
                                scalar[j * s + ii],
                                "d_state {s} head {h} token {i}: state[{ii}][{j}]"
                            );
                        }
                    }
                }
            }
        }
    }

    /// xorshift64*, so a failure is reproducible from its seed.
    fn noise(seed: u64) -> impl FnMut() -> f32 {
        let mut x = seed | 1;
        move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            // 23 bits mapped onto [-1, 1), so a dot over 256 of them lands in
            // the range the real q and k reach after their per-head norm.
            ((x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 41) as f32 / 4_194_304.0) - 1.0
        }
    }

    /// The tiled walk over the KV cache has to give what one query at a time
    /// gives, **bit for bit**: the same dots, the same softmax over the whole
    /// row, and the value sum accumulated in ascending position.
    ///
    /// A tolerance would be the wrong check. This model's output is
    /// discontinuous in the arithmetic, so a difference too small to see in an
    /// attention output is one flipped int8 level away from a different token,
    /// which is why the tiling deliberately keeps the score rows instead of
    /// running the softmax incrementally.
    ///
    /// The shapes are the ones that can go wrong: the model's own GQA grouping
    /// (16 query heads over 4 KV heads, head dim 256), a short context and a
    /// long one, a batch and a single token, a cache that is not a multiple of
    /// the tile size, and positions that make the causal bound land inside a
    /// tile as well as on its edge.
    #[test]
    fn the_tiled_attention_matches_one_query_at_a_time() {
        let (hd, n_head, n_kv_head) = (256, 16, 4);
        let group = n_head / n_kv_head;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut rand = noise(0x5eed_a77e);
        // (positions already cached, tokens in the batch); the context is
        // whatever those need, and the cases straddle TILE = 128 both in the
        // cache length and in where the causal bound falls.
        let cases = [
            (0usize, 1usize),
            (0, 5),
            (1, 1),
            (7, 3),
            (TILE - 1, 1),
            (TILE, 1),
            (TILE - 2, 5),
            (TILE + 1, 4),
            (2 * TILE, 16),
            (3 * TILE - 3, 9),
            (1000, 16),
        ];
        for (pos, t) in cases {
            let context = pos + t;
            let q: Vec<f32> = (0..t * n_head * 2 * hd).map(|_| rand()).collect();
            let k: Vec<f32> = (0..n_kv_head * context * hd).map(|_| rand()).collect();
            let v: Vec<f32> = (0..n_kv_head * context * hd).map(|_| rand()).collect();
            let mut scores = vec![0.0f32; t * context];
            let mut tiled = vec![0.0f32; t * hd];
            let mut plain = vec![0.0f32; hd];
            for h in 0..n_head {
                let kh = h / group;
                attend_batch(
                    &q[h * 2 * hd..],
                    n_head * 2 * hd,
                    &k[kh * context * hd..],
                    &v[kh * context * hd..],
                    t,
                    pos,
                    hd,
                    scale,
                    context,
                    &mut scores,
                    &mut tiled,
                );
                for i in 0..t {
                    attend(
                        &q[i * n_head * 2 * hd + h * 2 * hd..][..hd],
                        &k[kh * context * hd..],
                        &v[kh * context * hd..],
                        pos + i + 1,
                        scale,
                        &mut plain,
                    );
                    assert_eq!(
                        &tiled[i * hd..(i + 1) * hd],
                        &plain[..],
                        "pos {pos} t {t} head {h} token {i}"
                    );
                }
            }
        }
    }

    /// The causal bound is the whole mask, so a query must not be able to see a
    /// position after its own: changing a later key or value cannot change an
    /// earlier token's output.
    #[test]
    fn a_token_cannot_see_the_tokens_after_it() {
        let (hd, t, pos) = (256, 8, TILE + 5);
        let context = pos + t;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut rand = noise(0x00ca_05a1);
        let q: Vec<f32> = (0..t * hd).map(|_| rand()).collect();
        let k: Vec<f32> = (0..context * hd).map(|_| rand()).collect();
        let mut v: Vec<f32> = (0..context * hd).map(|_| rand()).collect();
        let mut scores = vec![0.0f32; t * context];
        let mut before = vec![0.0f32; t * hd];
        attend_batch(
            &q,
            hd,
            &k,
            &v,
            t,
            pos,
            hd,
            scale,
            context,
            &mut scores,
            &mut before,
        );
        // The last token's value row: only the last token reads it.
        for x in v[(context - 1) * hd..].iter_mut() {
            *x += 10.0;
        }
        let mut after = vec![0.0f32; t * hd];
        attend_batch(
            &q,
            hd,
            &k,
            &v,
            t,
            pos,
            hd,
            scale,
            context,
            &mut scores,
            &mut after,
        );
        assert_eq!(
            before[..(t - 1) * hd],
            after[..(t - 1) * hd],
            "a token saw the value of a position after it"
        );
        assert_ne!(
            before[(t - 1) * hd..],
            after[(t - 1) * hd..],
            "the last token did not read the last position"
        );
    }

    /// The mrope section layout this file carries picks one of the first three
    /// position axes for every rotated dimension pair and never the fourth.
    ///
    /// That is what makes [`rope`] above, which knows only one position, exact
    /// for text: the batch fills the first three axes with the token position
    /// and the fourth with zero. This replays the reference's interleaved
    /// selection to prove it, because getting it wrong is silent.
    #[test]
    fn interleaved_mrope_never_reaches_the_fourth_axis() {
        let sections = [11, 11, 10, 0];
        let sect_dims: i32 = sections.iter().sum();
        let (n_rot, head_dim) = (64, 256);
        assert!(sect_dims <= head_dim);
        let mut used = [0usize; 4];
        for i in (0..n_rot).step_by(2) {
            let sector = (i / 2) % sect_dims;
            let axis = if sector % 3 == 1 && sector < 3 * sections[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                2
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                0
            } else {
                3
            };
            used[axis] += 1;
        }
        assert_eq!(
            used[3], 0,
            "a dimension pair fell through to the fourth axis"
        );
        assert_eq!(used[0] + used[1] + used[2], (n_rot / 2) as usize);
    }

    #[test]
    fn rope_is_the_identity_at_position_zero() {
        let mut x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let before = x.clone();
        rope(&mut x, 0, 64, 1e7);
        assert_eq!(x, before);
    }

    #[test]
    fn rope_leaves_the_tail_of_the_head_alone() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let before = x.clone();
        rope(&mut x, 7, 64, 1e7);
        assert_eq!(x[64..], before[64..], "dimensions past n_rot were rotated");
        assert_ne!(x[..64], before[..64]);
        // A rotation preserves the length of every pair it touches.
        for i in 0..32 {
            let after = x[i] * x[i] + x[i + 32] * x[i + 32];
            let before = before[i] * before[i] + before[i + 32] * before[i + 32];
            assert!((after - before).abs() < 1e-5, "pair {i} changed length");
        }
    }

    #[test]
    fn the_convolution_is_causal_over_its_history() {
        // One channel, kernel 4, taps [1, 10, 100, 1000] oldest to newest.
        let w = [1.0f32, 10.0, 100.0, 1000.0];
        let mut state = [0.0f32; 3];
        let mut out = [0.0f32];
        for (step, want) in [(1.0, 1000.0), (2.0, 2100.0), (3.0, 3210.0), (4.0, 4321.0)] {
            depthwise_conv(&[step], &w, 4, &mut state, &mut out);
            assert_eq!(out[0], want, "step {step}");
        }
        // The history is the last three samples, oldest first.
        assert_eq!(state, [2.0, 3.0, 4.0]);
    }

    /// The tap a batch crosses: inside a batch, token `i`'s older samples are
    /// the earlier rows of the same batch, and before those the history buffer.
    ///
    /// Getting that boundary wrong is silent, because the convolution's taps
    /// decay into the residual rather than producing anything obviously wrong,
    /// so the four outputs here are hand-computed from the taps rather than
    /// compared against another run of the same loop.
    #[test]
    fn the_convolution_crosses_a_batch_boundary() {
        // One channel, kernel 4, taps [1, 10, 100, 1000] oldest to newest.
        let w = [1.0f32, 10.0, 100.0, 1000.0];
        let mut state = [0.0f32; 3];
        let mut out = [0.0f32];
        // Three tokens before the batch, so the history is full and not zero.
        for x in [1.0f32, 2.0, 3.0] {
            depthwise_conv(&[x], &w, 4, &mut state, &mut out);
        }
        assert_eq!(state, [1.0, 2.0, 3.0]);

        // A batch of four, as `linear_attn` runs it: one call per row over the
        // same rolling state. Each output is the four newest samples against
        // the taps, which for the first rows means samples from before the
        // batch.
        let batch = [4.0f32, 5.0, 6.0, 7.0];
        let want = [
            1.0 * 1.0 + 10.0 * 2.0 + 100.0 * 3.0 + 1000.0 * 4.0,
            1.0 * 2.0 + 10.0 * 3.0 + 100.0 * 4.0 + 1000.0 * 5.0,
            1.0 * 3.0 + 10.0 * 4.0 + 100.0 * 5.0 + 1000.0 * 6.0,
            1.0 * 4.0 + 10.0 * 5.0 + 100.0 * 6.0 + 1000.0 * 7.0,
        ];
        let mut got = [0.0f32; 4];
        for (i, &x) in batch.iter().enumerate() {
            depthwise_conv(&[x], &w, 4, &mut state, &mut got[i..i + 1]);
        }
        assert_eq!(got, want);
        // And the history the next batch starts from is the newest three.
        assert_eq!(state, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn softplus_matches_its_definition_where_exp_still_fits() {
        for x in [-30.0f32, -1.0, 0.0, 1.0, 19.0] {
            let want = (1.0 + x.exp()).ln();
            assert!((softplus(x) - want).abs() < 1e-6, "softplus({x})");
        }
        assert_eq!(softplus(40.0), 40.0);
    }

    #[test]
    fn l2_norm_leaves_a_unit_vector() {
        let mut x = [3.0f32, 4.0, 0.0];
        l2_norm_in_place(&mut x, 0.0);
        assert!((x[0] - 0.6).abs() < 1e-6);
        assert!((x[1] - 0.8).abs() < 1e-6);
    }
}
