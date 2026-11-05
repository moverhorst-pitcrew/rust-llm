//! The hybrid forward pass, one token at a time, plus the caches and the
//! scratch it runs on.
//!
//! Both caches and every buffer are allocated once, by [`Runner::new`], from
//! the config. The pass itself allocates nothing: it quantizes activations into
//! scratch, hands row ranges to the pool and writes into buffers it already
//! owns.
//!
//! Tokens go through one at a time, for the prompt as much as for generation.
//! The delta-net recurrence is sequential over tokens anyway, and a single path
//! means the state generation starts from is, by construction, the state the
//! prompt produced. Batched prompt evaluation is a separate change.

use std::time::Instant;

use llm_core::quant::Activations;
use llm_core::{Kernels, TensorView};

use crate::model::{Attn, Config, FullBlock, LinearBlock, Model};
use crate::pool::Pool;

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
    /// Full block: the fused query-and-gate projection.
    Qg,
    K,
    V,
    AttnOut,
    /// Scores, softmax and the weighted sum over the KV cache.
    Attention,
    /// Linear block: the q/k/v projection the convolution reads.
    Qkv,
    /// Linear block: the output gate, `alpha` and `beta` projections.
    Zab,
    Conv,
    /// The delta-rule recurrence over the per-head state.
    DeltaNet,
    LinearOut,
    FfnGate,
    FfnUp,
    FfnDown,
    /// The output head over the tied embedding table.
    Head,
    /// Every RMS and L2 norm, including the per-head ones, and RoPE.
    Norms,
    /// Residual adds, SwiGLU, the sigmoid and SiLU gates, argmax-free.
    Elementwise,
}

impl Op {
    const COUNT: usize = 17;
    const ALL: [Op; Op::COUNT] = [
        Op::Embed,
        Op::Qg,
        Op::K,
        Op::V,
        Op::AttnOut,
        Op::Attention,
        Op::Qkv,
        Op::Zab,
        Op::Conv,
        Op::DeltaNet,
        Op::LinearOut,
        Op::FfnGate,
        Op::FfnUp,
        Op::FfnDown,
        Op::Head,
        Op::Norms,
        Op::Elementwise,
    ];

    fn label(self) -> &'static str {
        match self {
            Op::Embed => "embed",
            Op::Qg => "full: qg matmul",
            Op::K => "full: k matmul",
            Op::V => "full: v matmul",
            Op::AttnOut => "full: output matmul",
            Op::Attention => "full: attention",
            Op::Qkv => "linear: qkv matmul",
            Op::Zab => "linear: gate/alpha/beta matmul",
            Op::Conv => "linear: convolution",
            Op::DeltaNet => "linear: delta-net recurrence",
            Op::LinearOut => "linear: output matmul",
            Op::FfnGate => "ffn: gate matmul",
            Op::FfnUp => "ffn: up matmul",
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
    /// `[linear block][value head][d_state][d_state]`, transposed as the
    /// reference stores it: row `j` of a head is `S[..][j]`.
    ssm: Vec<f32>,
    /// Tokens already in the caches, which is also the next token's position.
    pos: usize,
}

/// Every intermediate the pass writes to.
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
    scores: Vec<f32>,
    ffn_a: Vec<f32>,
    ffn_b: Vec<f32>,
    /// `alpha`/decay and `beta`, one scalar per value head each.
    decay: Vec<f32>,
    beta: Vec<f32>,
    logits: Vec<f32>,
}

/// A loaded model plus the threads, caches and scratch to run it.
pub struct Runner<'a> {
    model: &'a Model<'a>,
    pool: Pool,
    context: usize,
    caches: Caches,
    bufs: Buffers,
    acts: Activations,
    pub tracer: Tracer,
    pub profile: Profile,
}

impl<'a> Runner<'a> {
    /// Allocates for `context` positions and `threads` threads.
    pub fn new(model: &'a Model<'a>, context: usize, threads: usize) -> Self {
        let c = &model.config;
        let context = context.max(1);
        let vocab = model.vocab();
        Self {
            pool: Pool::new(threads),
            context,
            caches: Caches {
                kv: vec![0.0; c.n_full_blocks() * 2 * c.n_head_kv * context * c.head_dim],
                conv: vec![0.0; c.n_linear_blocks() * c.conv_dim() * (c.conv_kernel - 1)],
                ssm: vec![0.0; c.n_linear_blocks() * c.n_v_heads * c.d_state * c.d_state],
                pos: 0,
            },
            bufs: Buffers {
                x: vec![0.0; c.n_embd],
                xn: vec![0.0; c.n_embd],
                y: vec![0.0; c.n_embd],
                qkv: vec![0.0; c.qg_dim().max(c.conv_dim())],
                kb: vec![0.0; c.kv_dim()],
                vb: vec![0.0; c.kv_dim()],
                z: vec![0.0; c.d_inner],
                conv: vec![0.0; c.conv_dim()],
                heads: vec![0.0; c.attn_dim().max(c.d_inner)],
                scores: vec![0.0; c.n_head * context],
                ffn_a: vec![0.0; c.n_ff],
                ffn_b: vec![0.0; c.n_ff],
                decay: vec![0.0; c.n_v_heads],
                beta: vec![0.0; c.n_v_heads],
                logits: vec![0.0; vocab],
            },
            acts: Activations::new(c.n_embd.max(c.n_ff).max(c.d_inner).max(c.attn_dim())),
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

    /// Runs one token and returns the logits for the next one.
    ///
    /// Panics if the context is full; the caller decides what to do about that
    /// long before a cache write could run off the end.
    pub fn forward(&mut self, token: u32) -> &[f32] {
        let Self {
            model,
            pool,
            context,
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
        assert!(caches.pos < context, "context of {context} tokens is full");

        let mut ex = Exec {
            model,
            pool,
            context,
            acts,
            tracer,
            profile,
        };

        ex.embed(token, &mut bufs.x);
        ex.tracer.node("model.input_embed", None, &bufs.x);

        for il in 0..c.n_blocks {
            let t = ex.profile.start();
            rms_norm(&bufs.x, model.blocks[il].attn_norm, c.rms_eps, &mut bufs.xn);
            ex.profile.add(Op::Norms, t);
            ex.tracer.node("attn_norm", Some(il), &bufs.xn);

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

            let t = ex.profile.start();
            for (x, y) in bufs.x.iter_mut().zip(&bufs.y) {
                *x += y;
            }
            ex.profile.add(Op::Elementwise, t);
            ex.tracer.node("attn_residual", Some(il), &bufs.x);

            let t = ex.profile.start();
            rms_norm(
                &bufs.x,
                model.blocks[il].post_attn_norm,
                c.rms_eps,
                &mut bufs.xn,
            );
            ex.profile.add(Op::Norms, t);
            ex.tracer.node("attn_post_norm", Some(il), &bufs.xn);
            ex.ffn(&model.blocks[il], il, bufs);
            let t = ex.profile.start();
            for (x, y) in bufs.x.iter_mut().zip(&bufs.y) {
                *x += y;
            }
            ex.profile.add(Op::Elementwise, t);
            ex.tracer.node("l_out", Some(il), &bufs.x);
        }

        let t = ex.profile.start();
        rms_norm(&bufs.x, model.output_norm, c.rms_eps, &mut bufs.xn);
        ex.profile.add(Op::Norms, t);
        ex.tracer.node("result_norm", None, &bufs.xn);
        // Tied embeddings: the table that read the token in is the output head.
        ex.matmul(Op::Head, &model.token_embd, &bufs.xn, &mut bufs.logits);
        ex.tracer.node("result_output", None, &bufs.logits);

        caches.pos += 1;
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
    /// a matmul is a job, and the time the submitter spends waiting for the
    /// other threads is part of what that matmul cost. They are listed
    /// separately because they are the part a scheduling change can remove.
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
                ("of which pool wake (barrier in)", wake),
                ("of which pool join (barrier out)", join),
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
        debug_assert_eq!(w.dims[0] as usize, x.len());
        debug_assert_eq!(w.dims[1] as usize, out.len());
        let at = self.profile.start();
        self.acts.quantize(x);
        self.pool
            .matvec(&self.model.kernels, w.ty, w.data, x.len(), self.acts, out)
            .expect("every weight type was checked when the model loaded");
        self.profile.add(op, at);
    }

    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
    fn ffn(&mut self, block: &crate::model::Block<'_>, il: usize, bufs: &mut Buffers) {
        let Buffers {
            xn,
            y,
            ffn_a,
            ffn_b,
            ..
        } = bufs;
        self.matmul(Op::FfnGate, &block.ffn_gate, xn, ffn_a);
        self.tracer.node("ffn_gate", Some(il), ffn_a);
        self.matmul(Op::FfnUp, &block.ffn_up, xn, ffn_b);
        self.tracer.node("ffn_up", Some(il), ffn_b);
        let at = self.profile.start();
        for (a, b) in ffn_a.iter_mut().zip(&*ffn_b) {
            *a = silu(*a) * b;
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("ffn_swiglu", Some(il), ffn_a);
        self.matmul(Op::FfnDown, &block.ffn_down, ffn_a, y);
        self.tracer.node("ffn_out", Some(il), y);
    }

    /// GQA attention with per-head q/k RMSNorm, partial RoPE and an output
    /// gate taken from the second half of the query projection.
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
        let Buffers {
            xn,
            y,
            qkv,
            kb,
            vb,
            heads,
            scores,
            ..
        } = bufs;
        let qg = &mut qkv[..c.qg_dim()];

        self.matmul(Op::Qg, &b.qg, xn, qg);
        self.tracer.node("Qcur_full", Some(il), qg);
        self.matmul(Op::V, &b.v, xn, vb);
        self.tracer.node("Vcur", Some(il), vb);
        self.matmul(Op::K, &b.k, xn, kb);
        self.tracer.node("Kcur", Some(il), kb);

        // The query occupies the first half of each head's slice of `attn_q`
        // and the gate the second half, so the query is strided.
        let at = self.profile.start();
        for h in 0..n_head {
            let q = &mut qg[h * 2 * hd..h * 2 * hd + hd];
            rms_norm_in_place(q, b.q_norm, c.rms_eps);
            rope(q, pos, c.n_rot, c.rope_theta);
        }
        for h in 0..n_kv_head {
            let k = &mut kb[h * hd..(h + 1) * hd];
            rms_norm_in_place(k, b.k_norm, c.rms_eps);
            rope(k, pos, c.n_rot, c.rope_theta);
        }
        self.profile.add(Op::Norms, at);
        self.tracer.node("Kcur_roped", Some(il), kb);

        // Append this token's keys and values, then read the cache shared.
        let at = self.profile.start();
        let (k_cache, v_cache) = kv.split_at_mut(n_kv_head * context * hd);
        for h in 0..n_kv_head {
            let at = (h * context + pos) * hd;
            k_cache[at..at + hd].copy_from_slice(&kb[h * hd..(h + 1) * hd]);
            v_cache[at..at + hd].copy_from_slice(&vb[h * hd..(h + 1) * hd]);
        }
        let (k_cache, v_cache) = (&*k_cache, &*v_cache);
        self.profile.add(Op::Elementwise, at);

        let at = self.profile.start();
        let scale = 1.0 / (hd as f32).sqrt();
        let n_kv = pos + 1;
        let qg = &*qg;
        self.pool.chunks(
            &mut heads[..c.attn_dim()],
            hd,
            scores,
            context,
            |h, out, scores| {
                let kh = h / group;
                let q = &qg[h * 2 * hd..h * 2 * hd + hd];
                let scores = &mut scores[..n_kv];
                for (t, s) in scores.iter_mut().enumerate() {
                    let k = &k_cache[(kh * context + t) * hd..][..hd];
                    *s = dot(q, k) * scale;
                }
                softmax(scores);
                out.fill(0.0);
                for (t, &p) in scores.iter().enumerate() {
                    let v = &v_cache[(kh * context + t) * hd..][..hd];
                    for (o, &vd) in out.iter_mut().zip(v) {
                        *o += p * vd;
                    }
                }
            },
        );
        self.profile.add(Op::Attention, at);
        self.tracer
            .node("attn_pregate", Some(il), &heads[..c.attn_dim()]);

        let at = self.profile.start();
        for h in 0..n_head {
            let gate = &qg[h * 2 * hd + hd..(h + 1) * 2 * hd];
            for (o, &g) in heads[h * hd..(h + 1) * hd].iter_mut().zip(gate) {
                *o *= sigmoid(g);
            }
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer
            .node("attn_gated", Some(il), &heads[..c.attn_dim()]);

        self.matmul(Op::AttnOut, &b.o, &heads[..c.attn_dim()], y);
        self.tracer.node("attn_output", Some(il), y);
    }

    /// The gated delta net: a depthwise causal convolution over the q/k/v
    /// projection, then the delta rule against a per-head recurrent state.
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
        let key_dim = c.key_dim();
        let Buffers {
            xn,
            y,
            qkv,
            z,
            conv,
            heads,
            decay,
            beta,
            ..
        } = bufs;
        let qkv = &mut qkv[..c.conv_dim()];

        self.matmul(Op::Qkv, &b.qkv, xn, qkv);
        self.tracer.node("linear_attn_qkv_mixed", Some(il), qkv);
        self.matmul(Op::Zab, &b.gate, xn, z);
        self.tracer.node("z", Some(il), z);

        self.matmul(Op::Zab, &b.alpha, xn, decay);
        self.tracer.node("alpha", Some(il), decay);
        let at = self.profile.start();
        for (d, (&bias, &a)) in decay.iter_mut().zip(b.dt_bias.iter().zip(b.a)) {
            // `ssm_a` is already `-exp(A_log)`, so this is the log of the
            // decay and `exp` of it lands in (0, 1).
            *d = softplus(*d + bias) * a;
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("gate", Some(il), decay);

        self.matmul(Op::Zab, &b.beta, xn, beta);
        let at = self.profile.start();
        for v in beta.iter_mut() {
            *v = sigmoid(*v);
        }
        self.profile.add(Op::Elementwise, at);
        self.tracer.node("beta_sigmoid", Some(il), beta);

        let at = self.profile.start();
        depthwise_conv(qkv, b.conv1d, c.conv_kernel, conv_state, conv);
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
        let (qk, _) = conv.split_at_mut(2 * key_dim);
        for h in 0..2 * n_kh {
            l2_norm_in_place(&mut qk[h * s..(h + 1) * s], c.rms_eps);
        }
        let conv = &*conv;
        self.profile.add(Op::Norms, at);

        let at = self.profile.start();
        let scale = 1.0 / (s as f32).sqrt();
        let decay = &*decay;
        let beta = &*beta;
        self.pool.chunks(
            &mut heads[..c.d_inner],
            s,
            ssm_state,
            s * s,
            |vh, out, state| {
                // Several value heads share one key head, taken modulo the key
                // head count as the reference broadcasts it.
                let kh = vh % n_kh;
                let q = &conv[kh * s..][..s];
                let k = &conv[key_dim + kh * s..][..s];
                let v = &conv[2 * key_dim + vh * s..][..s];
                let g = decay[vh].exp();
                let beta = beta[vh];
                // Row `j` of the state is `S[..][j]`, so one pass over it per
                // output component does the decay, the delta and the readout.
                for (j, row) in state.chunks_exact_mut(s).enumerate() {
                    let mut sk = 0.0f32;
                    for (r, &ki) in row.iter_mut().zip(k) {
                        *r *= g;
                        sk += *r * ki;
                    }
                    let d = (v[j] - sk) * beta;
                    let mut o = 0.0f32;
                    for ((r, &ki), &qi) in row.iter_mut().zip(k).zip(q) {
                        *r += d * ki;
                        o += *r * qi;
                    }
                    out[j] = o * scale;
                }
            },
        );
        self.profile.add(Op::DeltaNet, at);
        self.tracer
            .node("attn_output", Some(il), &heads[..c.d_inner]);

        // The gated norm: RMSNorm per value head, then the sigmoid-linear gate.
        let at = self.profile.start();
        for h in 0..n_vh {
            let head = &mut heads[h * s..(h + 1) * s];
            rms_norm_in_place(head, b.norm, c.rms_eps);
            for (o, &g) in head.iter_mut().zip(&z[h * s..(h + 1) * s]) {
                *o *= silu(g);
            }
        }
        self.profile.add(Op::Norms, at);
        self.tracer
            .node("final_output", Some(il), &heads[..c.d_inner]);

        self.matmul(Op::LinearOut, &b.out, &heads[..c.d_inner], y);
        self.tracer.node("linear_attn_out", Some(il), y);
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
