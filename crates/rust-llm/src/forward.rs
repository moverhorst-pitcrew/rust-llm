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
//!
//! [`Runner::verify`] is the third entry point, for speculative decoding: the
//! same pass, but the output head runs on every row rather than only the last,
//! and the batch is not committed until [`Runner::accept`] says how many of its
//! rows to keep. What makes a rollback free is that the two pieces of state a
//! position cannot be rewound out of, the recurrent state and the convolution
//! history, are a ring of `draft + 1` buffers instead of one: row `r` reads the
//! buffer row `r - 1` wrote and writes the next one, so every row leaves a
//! snapshot behind and rolling back is choosing which buffer is live. The ring
//! exists only when speculation is on; at `draft == 0` it is one buffer and the
//! recurrence updates it in place, exactly as it always did.

use std::time::Instant;

use llm_core::quant::Activations;
use llm_core::{Kernels, TensorView};

use crate::model::{Attn, Config, FullBlock, LinearBlock, Model, MtpBlock};
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
    /// The MTP block's `nextn.eh_proj`, which is the one matmul it has that
    /// the trunk's blocks do not.
    MtpProj,
    /// Every RMS and L2 norm, including the per-head ones, and RoPE.
    Norms,
    /// Residual adds, SwiGLU, the sigmoid and SiLU gates, argmax-free.
    Elementwise,
}

impl Op {
    const COUNT: usize = 14;
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
        Op::MtpProj,
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
            Op::MtpProj => "mtp: eh_proj matmul",
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

/// The caches. All three are sized at load and never reallocated.
struct Caches {
    /// `[full block][k, v][kv head][position][head_dim]`.
    kv: Vec<f32>,
    /// `[linear block][ring slot][channel][conv_kernel - 1]`, oldest sample
    /// first.
    conv: Vec<f32>,
    /// `[linear block][value head][ring slot][d_state i][d_state j]`, so one
    /// `i` of a head is contiguous over `j` and [`delta_net`] can take a block
    /// of `j` at a time, and one value head's whole ring is contiguous so the
    /// pool can hand it to that head's job as one chunk. The layout is
    /// internal: nothing outside the recurrence reads it, the reset only zeroes
    /// it and no tracer node dumps it.
    ssm: Vec<f32>,
    /// Tokens already in the caches, which is also the next token's position.
    pos: usize,
    /// Ring slot holding the state those `pos` tokens left behind. Always 0
    /// with speculation off, where the ring is one buffer.
    live: usize,
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
    /// One row of `vocab` per row the output head can be asked for, which is
    /// one with speculation off and `draft + 1` with it on.
    logits: Vec<f32>,
}

impl Buffers {
    /// Every buffer at once, `b` rows wide, with room for `logit_rows` rows of
    /// the output head.
    ///
    /// The MTP block allocates a second one of these rather than borrowing the
    /// trunk's. Two of them are live at the same moment: the trunk's `xn` holds
    /// the hidden states the block reads and its `logits` are still the
    /// caller's, so sharing would mean aliasing exactly the two buffers that
    /// carry a result across the call.
    fn new(c: &Config, context: usize, b: usize, vocab: usize, logit_rows: usize) -> Self {
        Self {
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
            logits: vec![0.0; vocab * logit_rows],
        }
    }
}

/// Tokens per batch when the caller does not say.
///
/// Taken from the measured crossover rather than from a round number: see
/// `docs/benchmarks.md`, "What batched prompt evaluation moved".
pub const DEFAULT_BATCH: usize = 16;

/// How sure the MTP head has to be of a token before it is worth drafting.
///
/// The reference calls this `p_min` and defaults it to zero, which drafts every
/// step however hopeless. It is not a tuning knob here, it is a measured
/// constant: the sweep is in `docs/benchmarks.md`, "Drafting from the MTP head".
///
/// What the floor saves is a **verification row**, not a draft step: the block
/// and the head run before it is consulted, so a step whose draft is dropped is
/// paid for anyway. The bound it should sit at is therefore where a drafted
/// token's chance of being accepted stops covering the row it costs, which on
/// this box is a 14.3 ms row against a 68.4 ms token, about 0.21. Measured, the
/// gain peaks between 0.15 and 0.35 and falls away on both sides: below it the
/// rows of hopeless drafts eat the gain, above it the head stays silent and its
/// steps are spent for nothing.
///
/// It is the probability of the top token after a softmax over the whole
/// vocabulary, which is one pass over logits the matmul has just written and
/// does not show next to it.
pub const MTP_P_MIN: f32 = 0.25;

/// The longest draft [`Runner::with_speculation`] takes.
///
/// Not a tuning knob: it is the bound that keeps the ring and the logit buffer
/// from being sized by an unchecked command-line number. At 16 the ring is
/// already 850 MiB and the acceptance histogram in `docs/benchmarks.md` is flat
/// long before it.
pub const MAX_DRAFT: usize = 16;

/// A loaded model plus the threads, caches and scratch to run it.
pub struct Runner<'a> {
    model: &'a Model<'a>,
    pool: Pool,
    context: usize,
    max_batch: usize,
    /// Ring slots for the recurrent state and the convolution history, which is
    /// `1` with speculation off and the longest verification batch with it on.
    slots: usize,
    /// Drafts [`crate::generate::generate`] offers per pass, at most
    /// `slots - 1`. Separate from `slots` because the ring is allocated once and
    /// a caller may ask for shorter drafts afterwards.
    draft: usize,
    caches: Caches,
    bufs: Buffers,
    acts: Activations,
    /// The MTP block's cache and scratch, allocated by
    /// [`Runner::enable_mtp`] and `None` until then.
    mtp: Option<Mtp>,
    /// Rows the last [`Runner::run`] produced, and how many of them
    /// [`Runner::accept`] has committed so far. Only the MTP catch-up reads
    /// them: it has to know which rows of the batch in flight are the ones
    /// being committed.
    batch: usize,
    committed: usize,
    /// The tokens of that batch, for the same reason.
    tokens: Vec<u32>,
    pub tracer: Tracer,
    pub profile: Profile,
}

/// Everything the multi-token-prediction block needs beside its weights.
struct Mtp {
    /// `[k, v][kv head][position][head_dim]`: one full-attention block's cache
    /// over the whole context, the layout and the walk the trunk's full blocks
    /// already use.
    ///
    /// No ring and no rollback. An entry is written at an absolute position and
    /// read only from that position onwards, so what a rejected draft left
    /// behind is overwritten by the catch-up after the next trunk pass and read
    /// by nothing in between. That is the reference's "own KV, growing in the
    /// draft region" mode.
    kv: Vec<f32>,
    /// Positions the block has been caught up on: the trunk's `pos` outside a
    /// draft, and above it inside one.
    caught_up: usize,
    /// The trunk's post-`output_norm` row at `caught_up - 1`, which is the
    /// hidden state the token at `caught_up` pairs with. Zero at the start of a
    /// sequence, which is what the first token gets in the reference too.
    h: Vec<f32>,
    /// `enorm(embed(token))` followed by `hnorm(h)`, one row of `2 * n_embd`
    /// per row of the call: the order `eh_proj` reads them in.
    eh: Vec<f32>,
    /// The hidden states handed in, one `n_embd` row per row of the call. A
    /// buffer of its own rather than a slice of the trunk's, because a catch-up
    /// reads one row of the trunk's `xn` per row and a draft step reads this
    /// block's own previous output.
    h_in: Vec<f32>,
    bufs: Buffers,
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
        Self::with_speculation(model, context, threads, max_batch, 0)
    }

    /// The same with room for speculative decoding: `draft` drafted tokens per
    /// pass, `0` for none.
    ///
    /// What `draft` costs is the ring: `draft + 1` copies of the recurrent state
    /// and the convolution history, about 50 MiB each on this model, allocated
    /// here and never again. At `0` there is one of each, which is what there
    /// always was, so nothing about the off path changes.
    ///
    /// A verification batch is `draft + 1` rows, so `max_batch` is raised to at
    /// least that: the scratch has to hold the widest batch either path can ask
    /// for, and a prompt read in batches of `draft + 1` rather than of 1 is
    /// faster anyway.
    pub fn with_speculation(
        model: &'a Model<'a>,
        context: usize,
        threads: usize,
        max_batch: usize,
        draft: usize,
    ) -> Self {
        let c = &model.config;
        let context = context.max(1);
        assert!(
            draft <= MAX_DRAFT,
            "a draft of {draft} tokens, but at most {MAX_DRAFT} are allocated for"
        );
        let slots = draft + 1;
        // A batch longer than the context could never be run.
        let b = max_batch.max(slots).clamp(1, context);
        let vocab = model.vocab();
        Self {
            pool: Pool::new(threads),
            context,
            max_batch: b,
            slots,
            draft,
            caches: Caches {
                kv: vec![0.0; c.n_full_blocks() * 2 * c.n_head_kv * context * c.head_dim],
                conv: vec![0.0; c.n_linear_blocks() * slots * c.conv_dim() * (c.conv_kernel - 1)],
                ssm: vec![0.0; c.n_linear_blocks() * c.n_v_heads * slots * c.d_state * c.d_state],
                pos: 0,
                live: 0,
            },
            bufs: Buffers::new(c, context, b, vocab, slots),
            // `2 * n_embd` is the MTP block's `eh_proj`, the widest activation
            // either path quantizes; it is wider than `n_embd` on every shape
            // and wider than the feed-forward on a small enough one.
            acts: Activations::with_batch(
                (2 * c.n_embd).max(c.n_ff).max(c.d_inner).max(c.attn_dim()),
                b,
            ),
            mtp: None,
            batch: 0,
            committed: 0,
            tokens: Vec::with_capacity(b),
            model,
            tracer: Tracer::default(),
            profile: Profile::default(),
        }
    }

    /// Allocates the MTP block's cache and scratch, so this runner can draft
    /// from the model's own head.
    ///
    /// Returns `false` and changes nothing when the file has no MTP block,
    /// which is what the conversion before this model's shipped
    /// (`docs/models.md`).
    ///
    /// It allocates, where everything else this runner owns comes out of the
    /// constructor. That is on purpose: a runner that does not draft from the
    /// head should not carry a second block's cache, and the three constructors
    /// and every caller of them stay as they were. It is still once per run and
    /// before the first token, which is what the assert holds it to.
    pub fn enable_mtp(&mut self) -> bool {
        if self.model.mtp.is_none() {
            return false;
        }
        assert_eq!(
            self.caches.pos, 0,
            "enable_mtp before the first token: the block is caught up as tokens \
             go through and cannot be caught up on positions already past"
        );
        let c = &self.model.config;
        self.mtp = Some(Mtp {
            kv: vec![0.0; 2 * c.n_head_kv * self.context * c.head_dim],
            caught_up: 0,
            h: vec![0.0; c.n_embd],
            eh: vec![0.0; self.max_batch * 2 * c.n_embd],
            h_in: vec![0.0; self.max_batch * c.n_embd],
            bufs: Buffers::new(c, self.context, self.max_batch, self.model.vocab(), 1),
        });
        true
    }

    /// Whether [`Runner::draft_mtp`] can produce anything.
    pub fn drafts_from_mtp(&self) -> bool {
        self.mtp.is_some()
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
        self.caches.live = 0;
        self.batch = 0;
        self.committed = 0;
        self.tokens.clear();
        if let Some(mtp) = &mut self.mtp {
            // The cache is only read up to `caught_up`, so zeroing it is not
            // what makes the reset a reset; the carried hidden state is. It is
            // zeroed anyway, for the same reason the trunk's is: a fresh runner
            // and a reset one have to be the same thing bit for bit.
            mtp.kv.fill(0.0);
            mtp.h.fill(0.0);
            mtp.caught_up = 0;
        }
        self.tracer.clear();
    }

    /// Tokens this runner can take in one [`Runner::forward_batch`].
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Drafted tokens per pass: `0` when speculation is off.
    pub fn draft(&self) -> usize {
        self.draft
    }

    /// Asks for shorter drafts than the ring was allocated for, which is what a
    /// per-request `speculate` on the HTTP surface sets.
    ///
    /// Returns `false` and changes nothing when `draft` is longer than the ring,
    /// because that would need buffers this runner never allocated.
    pub fn set_draft(&mut self, draft: usize) -> bool {
        if draft >= self.slots {
            return false;
        }
        self.draft = draft;
        true
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
        let t = self.run(tokens, false);
        self.accept(t);
        self.logits()
    }

    /// Runs `tokens` as one batch and produces the logits of **every** row,
    /// without committing any of them.
    ///
    /// This is the verification half of speculative decoding. Row `r` is the
    /// token at position `pos + r` and [`Runner::logit_row`] gives what follows
    /// it, which is what the caller compares against its draft. Nothing moves
    /// until [`Runner::accept`]: the batch has written the KV cache at
    /// `pos..pos + tokens.len()` and the next `tokens.len()` ring slots, and
    /// `accept` picks how far along that chain the runner actually is. Rows that
    /// are never accepted are overwritten by the next batch and read by nothing
    /// in between.
    ///
    /// Panics unless the batch fits the ring (`tokens.len() <= draft + 1`), the
    /// scratch and the remaining context.
    pub fn verify(&mut self, tokens: &[u32]) -> &[f32] {
        assert!(
            tokens.len() <= self.slots,
            "a verification batch of {} rows, but the ring holds {}",
            tokens.len(),
            self.slots
        );
        let t = self.run(tokens, true);
        &self.bufs.logits[..t * self.model.vocab()]
    }

    /// Commits the first `rows` rows of the batch that just ran, which must be
    /// at least one.
    ///
    /// For a committed row the KV cache entry is already written, so this is the
    /// `pos` that makes it readable and the ring slot that makes the state that
    /// row produced the live one. Both are a number, which is the whole reason
    /// the ring is a ring: a rollback is this call with a smaller `rows` and
    /// never a copy.
    ///
    /// **Not with `rows == 0`, though.** A full-width batch is `draft + 1` rows
    /// and the ring is `draft + 1` slots, so its last row writes
    /// `(live + draft + 1) % slots`, which is `live`: by the time [`Runner::verify`]
    /// returns, the state the batch started from has been overwritten. That is
    /// deliberate and it is what makes the ring exactly as long as the batch,
    /// because row 0 is a token the caller has already committed to and a
    /// rollback never needs to go behind the state row 0 produced. Committing
    /// none of the rows would therefore leave a rejected row's state live rather
    /// than undoing the batch. See `docs/decisions.md`, "Speculative decoding".
    /// Also where the MTP block is rolled back, which is the same thing as
    /// catching it up: its cache is written at absolute positions and read
    /// only from a position onwards, so committing fewer rows than the batch
    /// ran leaves the rejected positions holding entries nothing will read
    /// until the next catch-up overwrites them.
    pub fn accept(&mut self, rows: usize) {
        debug_assert!(
            rows > 0,
            "a batch commits at least its first row; accept(0) would leave the \
             last row of the batch live, not the state from before it"
        );
        debug_assert!(self.caches.pos + rows <= self.context);
        if self.mtp.is_some() {
            self.catch_up(rows);
        }
        self.caches.pos += rows;
        self.caches.live = (self.caches.live + rows) % self.slots;
        self.committed += rows;
    }

    /// Runs the MTP block over the `rows` rows of the batch in flight that
    /// [`Runner::accept`] is committing, so its cache holds every position the
    /// trunk holds.
    ///
    /// Row `r` of the batch is the token at position `pos + r`, and what the
    /// block takes with it is the hidden state of the position *before* it:
    /// the trunk's own post-`output_norm` row `r - 1`, or the state carried
    /// over from the previous call when `r` is the batch's first row. All of
    /// those are known before the call, so however many rows are being
    /// committed go through as one batch; nothing here is sequential, unlike a
    /// draft.
    ///
    /// The whole block runs, head excluded. Only the key and value projections
    /// feed its cache, so a catch-up could stop after them and read 11 MB of
    /// weights per row rather than 75; that is about 1.7 ms a row on this box,
    /// against the 15 ms a draft step costs, so it is the block as it is until
    /// the profile says otherwise. Running it through the same two functions the
    /// trunk's full blocks use is worth more than the 1.7 ms: the head cannot
    /// drift away from the block it drafts for.
    fn catch_up(&mut self, rows: usize) {
        let Self {
            model,
            pool,
            context,
            caches,
            bufs,
            acts,
            mtp,
            tracer,
            profile,
            tokens,
            batch,
            committed,
            ..
        } = self;
        let model: &'a Model<'a> = model;
        let block = model.mtp.as_ref().expect("enable_mtp checked the file");
        let Mtp {
            kv,
            caught_up,
            h,
            eh,
            h_in,
            bufs: mbufs,
        } = mtp.as_mut().expect("the caller checked");
        let ne = model.config.n_embd;
        // Which rows of the batch these are, and where they sit. The two are
        // not the same number and adding them would be wrong: `accept` has
        // already advanced `caches.pos` past every row committed before this
        // call, so `caches.pos` *is* the absolute position of row `from`.
        let from = *committed;
        let pos = caches.pos;
        debug_assert!(from + rows <= *batch);
        debug_assert!(
            *caught_up >= pos,
            "the block is at {caught_up} and is being caught up from {pos}, so it \
             would skip a position"
        );

        // Row `from` of the batch pairs with the carried hidden state; every
        // row after it with the trunk's own from the row in front of it.
        for r in from..from + rows {
            let row: &[f32] = match r {
                0 => h,
                _ => &bufs.xn[(r - 1) * ne..][..ne],
            };
            h_in[(r - from) * ne..][..ne].copy_from_slice(row);
        }
        Exec {
            model,
            pool,
            context: *context,
            t: rows,
            acts,
            tracer,
            profile,
        }
        .mtp_block(
            block,
            pos,
            &tokens[from..from + rows],
            h_in,
            eh,
            mbufs,
            kv,
            false,
        );
        h.copy_from_slice(&bufs.xn[(from + rows - 1) * ne..][..ne]);
        *caught_up = pos + rows;
    }

    fn run(&mut self, tokens: &[u32], all_rows: bool) -> usize {
        self.batch = tokens.len();
        self.committed = 0;
        self.tokens.clear();
        if self.mtp.is_some() {
            self.tokens.extend_from_slice(tokens);
        }
        let mtp = self.mtp.is_some();
        let Self {
            model,
            pool,
            context,
            max_batch,
            slots,
            draft: _,
            caches,
            bufs,
            acts,
            tracer,
            profile,
            ..
        } = self;
        // Copied out of `self` so the borrow of the model outlives the mutable
        // borrows of the caches and the scratch rather than conflicting.
        let model: &'a Model<'a> = model;
        let c = &model.config;
        let context = *context;
        let slots = *slots;
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
                    // Both spans hold this block's whole ring, which is what
                    // the rows of the batch walk through.
                    let conv_span = slots * c.conv_dim() * (c.conv_kernel - 1);
                    let ssm_span = c.n_v_heads * slots * c.d_state * c.d_state;
                    let conv = &mut caches.conv[b.slot * conv_span..(b.slot + 1) * conv_span];
                    let ssm = &mut caches.ssm[b.slot * ssm_span..(b.slot + 1) * ssm_span];
                    ex.linear_attn(b, il, slots, caches.live, conv, ssm, bufs);
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

        // How many rows get normed, and how many the output head then runs
        // on. The head wants the last token's row only, which is all a prompt
        // needs, or every row, which is what verifying a draft needs. The MTP
        // block is a third reader: it has to be caught up on every position the
        // trunk ran, including a whole prompt, and what it takes is exactly
        // this norm's output. So with the block live every row is normed and
        // the head still runs on one, which is one extra `rms_norm` per row and
        // not one extra weight read.
        //
        // With the block off and `all_rows` false this is one row through the
        // same code it always was, from the same input, so `--speculate 0` is
        // byte for byte the pass it was.
        let rows = if all_rows { t } else { 1 };
        let normed = if all_rows || mtp { t } else { 1 };
        let vocab = model.vocab();
        let at = ex.profile.start();
        for i in 0..normed {
            let from = if normed == t { i } else { t - 1 };
            rms_norm(
                &bufs.x[from * ne..][..ne],
                model.output_norm,
                c.rms_eps,
                &mut bufs.xn[i * ne..][..ne],
            );
        }
        ex.profile.add(Op::Norms, at);
        // What the head is fed: the last row that was normed when it wants one,
        // and all of them when it wants all of them. The reference calls the
        // whole thing `h_nextn` and what it crops out of it `result_norm`,
        // which is the same two tensors.
        let head_in = &bufs.xn[(normed - rows) * ne..][..rows * ne];
        ex.tracer.node("result_norm", None, head_in);
        // The batch the blocks ran is over here: what follows is the rows the
        // head was asked for.
        ex.t = rows;
        // Tied embeddings: the table that read the token in is the output head.
        ex.matmul(
            Op::Head,
            &model.token_embd,
            head_in,
            &mut bufs.logits[..rows * vocab],
        );
        ex.tracer
            .node("result_output", None, &bufs.logits[..rows * vocab]);

        t
    }

    /// Drafts up to `out.len()` tokens from the model's own MTP head, and
    /// returns how many it wrote. The same contract as
    /// [`crate::generate::prompt_lookup`], and `0` when the head is not
    /// available or has nothing it is confident about.
    ///
    /// Step 0 is `(the last token, the carried hidden state)` at the trunk's
    /// own next position, which is exactly the pair a catch-up would use there;
    /// step `i` is `(what step i - 1 drafted, the block's own hidden state from
    /// step i - 1)` at the position after. Every step writes the block's cache
    /// at its own position and attends over everything in front of it, which is
    /// the reference's `common_speculative_impl_draft_mtp` with `is_mem_shared`
    /// and `chain_heads` both false.
    ///
    /// The steps are sequential by construction: step `i` reads what step
    /// `i - 1` produced. That is why a draft of `n` tokens costs `n` passes
    /// over the block and the output head, and why [`MTP_P_MIN`] exists.
    ///
    /// It allocates nothing, so it can sit in the generation loop.
    pub fn draft_mtp(&mut self, history: &[u32], out: &mut [u32]) -> usize {
        if out.is_empty() || self.mtp.is_none() {
            return 0;
        }
        let Self {
            model,
            pool,
            context,
            caches,
            acts,
            mtp,
            tracer,
            profile,
            ..
        } = self;
        let model: &'a Model<'a> = model;
        let block = model.mtp.as_ref().expect("enable_mtp checked the file");
        let Mtp {
            kv,
            caught_up,
            h,
            eh,
            h_in,
            bufs,
        } = mtp.as_mut().expect("checked above");
        let ne = model.config.n_embd;
        let vocab = model.vocab();
        debug_assert_eq!(
            *caught_up, caches.pos,
            "the block drafts from the position the trunk is at"
        );

        let mut token = *history.last().expect("a draft needs a last token");
        h_in[..ne].copy_from_slice(h);
        let mut n = 0;
        while n < out.len() {
            let pos = caches.pos + n;
            // The cache has a row per context position and no more. The caller
            // caps the draft by the room it has left for the verification
            // batch, so this is the second half of the same bound.
            if pos >= *context {
                break;
            }
            Exec {
                model,
                pool,
                context: *context,
                t: 1,
                acts,
                tracer,
                profile,
            }
            .mtp_block(block, pos, &[token], h_in, eh, bufs, kv, true);
            let (pick, p) = top_probability(&bufs.logits[..vocab]);
            if p < MTP_P_MIN {
                break;
            }
            out[n] = pick;
            n += 1;
            token = pick;
            // What the block just produced is what the next step pairs with
            // its token: `bufs.xn` is this block's own post-`shared_head_norm`
            // row, the reference's `h_nextn` out of the MTP graph.
            h_in[..ne].copy_from_slice(&bufs.xn[..ne]);
        }
        n
    }

    /// The last logits, for a caller that wants them again without a step.
    pub fn logits(&self) -> &[f32] {
        self.logit_row(0)
    }

    /// Row `r` of the logits the last [`Runner::verify`] produced, which is the
    /// prediction that follows that row's token.
    pub fn logit_row(&self, r: usize) -> &[f32] {
        let vocab = self.model.vocab();
        &self.bufs.logits[r * vocab..][..vocab]
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

    /// `bufs.xn = rms_norm(bufs.x, w)`, row by row, traced under `name`.
    fn norm_rows(&mut self, w: &[f32], name: &str, il: usize, t: usize, bufs: &mut Buffers) {
        let ne = self.model.config.n_embd;
        let eps = self.model.config.rms_eps;
        let at = self.profile.start();
        for i in 0..t {
            rms_norm(
                &bufs.x[i * ne..][..ne],
                w,
                eps,
                &mut bufs.xn[i * ne..][..ne],
            );
        }
        self.profile.add(Op::Norms, at);
        self.tracer.node(name, Some(il), &bufs.xn[..t * ne]);
    }

    /// `bufs.x += bufs.y`, which is what both of a block's residuals are.
    fn residual(&mut self, t: usize, bufs: &mut Buffers) {
        let ne = self.model.config.n_embd;
        let at = self.profile.start();
        for (x, y) in bufs.x[..t * ne].iter_mut().zip(&bufs.y[..t * ne]) {
            *x += y;
        }
        self.profile.add(Op::Elementwise, at);
    }

    /// The multi-token-prediction block on `tokens.len()` rows.
    ///
    /// Row `i` is `tokens[i]` at position `pos + i`, paired with the hidden
    /// state in row `i` of `h`. The reference's `graph_mtp` step for step:
    /// `hnorm(h)` and `enorm(embed(token))`, concatenated along the embedding
    /// axis with the embedding first, `eh_proj`, and from there a
    /// full-attention Qwen3.5 block, which is why the rest of this goes through
    /// the same [`Exec::full_attn`] and [`Exec::ffn`] the trunk's full blocks
    /// use rather than through a copy of them.
    ///
    /// `head` asks for `shared_head_norm` and the tied output head. A draft
    /// step needs both: the logits to pick a token from and the normed row to
    /// hand the step after it. A catch-up needs neither, because nothing
    /// downstream of the second residual feeds this block's own cache.
    ///
    /// On return `bufs.xn` holds the post-`shared_head_norm` rows and
    /// `bufs.logits` the head's, when `head` was asked for.
    #[allow(clippy::too_many_arguments)]
    fn mtp_block(
        &mut self,
        m: &MtpBlock<'_>,
        pos: usize,
        tokens: &[u32],
        h: &[f32],
        eh: &mut [f32],
        bufs: &mut Buffers,
        kv: &mut [f32],
        head: bool,
    ) {
        let model = self.model;
        let c = &model.config;
        let ne = c.n_embd;
        let t = tokens.len();
        // The block index the reference gives these nodes, which is the trunk's
        // block count: `blk.32` on this model.
        let il = c.n_blocks;
        self.t = t;

        let at = self.profile.start();
        for (i, &token) in tokens.iter().enumerate() {
            let (e, hn) = eh[i * 2 * ne..][..2 * ne].split_at_mut(ne);
            rms_norm(&h[i * ne..][..ne], m.hnorm, c.rms_eps, hn);
            self.embed(token, e);
            rms_norm_in_place(e, m.enorm, c.rms_eps);
        }
        self.profile.add(Op::Norms, at);
        self.tracer.node("mtp_concat", Some(il), &eh[..t * 2 * ne]);

        // `eh_proj` is the residual stream the block-shaped half runs on, which
        // is `inpSA` in the reference.
        self.matmul(
            Op::MtpProj,
            &m.eh_proj,
            &eh[..t * 2 * ne],
            &mut bufs.x[..t * ne],
        );
        self.tracer.node("mtp_eh_proj", Some(il), &bufs.x[..t * ne]);

        let Attn::Full(full) = &m.block.attn else {
            unreachable!("the MTP block is loaded as a full-attention block")
        };
        self.norm_rows(m.block.attn_norm, "mtp_attn_norm", il, t, bufs);
        self.full_attn(full, il, pos, kv, bufs);
        self.residual(t, bufs);
        self.tracer
            .node("mtp_attn_residual", Some(il), &bufs.x[..t * ne]);

        self.norm_rows(m.block.post_attn_norm, "mtp_attn_post_norm", il, t, bufs);
        self.ffn(&m.block, il, bufs);
        self.residual(t, bufs);
        self.tracer
            .node("mtp_post_ffn", Some(il), &bufs.x[..t * ne]);

        if !head {
            return;
        }
        self.norm_rows(m.shared_head_norm, "mtp_h_nextn", il, t, bufs);
        let vocab = model.vocab();
        self.matmul(
            Op::Head,
            &model.token_embd,
            &bufs.xn[..t * ne],
            &mut bufs.logits[..t * vocab],
        );
        self.tracer
            .node("mtp_result_output", Some(il), &bufs.logits[..t * vocab]);
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
    ///
    /// Both of those states are a ring of `slots` buffers, `live` of which holds
    /// what the committed tokens left behind. Row `i` reads slot
    /// `(live + i) % slots` and writes the one after it, so the chain over the
    /// batch is the same chain the single buffer carried and every row leaves
    /// its own snapshot behind for a rollback to land on. With `slots == 1` the
    /// two are the same slot and the recurrence updates it in place, which is
    /// the only path there was before speculation and still the only one with
    /// `--speculate 0`.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn(
        &mut self,
        b: &LinearBlock<'_>,
        il: usize,
        slots: usize,
        live: usize,
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
        let conv_span = conv_dim * (c.conv_kernel - 1);
        for i in 0..t {
            let (from, to) = ((live + i) % slots, (live + i + 1) % slots);
            let x = &qkv[i * conv_dim..][..conv_dim];
            let out = &mut conv[i * conv_dim..][..conv_dim];
            if from == to {
                depthwise_conv(
                    x,
                    b.conv1d,
                    c.conv_kernel,
                    &mut conv_state[from * conv_span..][..conv_span],
                    out,
                );
            } else {
                let (prev, next) = ring_pair(conv_state, conv_span, from, to);
                depthwise_conv_into(x, b.conv1d, c.conv_kernel, prev, next, out);
            }
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
        // One chunk per value head, holding that head's whole batch and its
        // whole ring: the state is sequential over tokens, so the tokens cannot
        // be split, and the readout therefore comes out head-major and is
        // transposed by the gated norm below. A head owns its own slice of
        // every ring slot, so writing the next slot needs no synchronisation
        // beyond the one the chunk already is.
        self.pool.chunks(
            &mut head_major[..n_vh * t * s],
            t * s,
            ssm_state,
            slots * s * s,
            |vh, out, ring| {
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
                    let (from, to) = ((live + i) % slots, (live + i + 1) % slots);
                    if from == to {
                        delta_net(
                            &mut ring[from * s * s..][..s * s],
                            q,
                            k,
                            v,
                            g,
                            beta,
                            scale,
                            out,
                        );
                    } else {
                        let (prev, next) = ring_pair(ring, s * s, from, to);
                        delta_net_into(prev, next, q, k, v, g, beta, scale, out);
                    }
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

/// The argmax and its probability under a softmax over the whole vector.
///
/// One pass for the maximum and one for the sum, over logits the output head
/// has just written, so this is memory the matmul left in cache. The
/// normalising sum is accumulated in f64 as everywhere else in this file.
fn top_probability(logits: &[f32]) -> (u32, f32) {
    let mut best = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.0 {
            best = (v, i);
        }
    }
    let sum: f64 = logits.iter().map(|&v| (v - best.0).exp() as f64).sum();
    (best.1 as u32, (1.0 / sum) as f32)
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

/// [`delta_net`] reading the state from `prev` and writing it to `next`, which
/// is how a row of a verification batch leaves the state it found behind.
///
/// **The arithmetic is [`delta_net`]'s, unchanged.** The only difference is the
/// decay in pass one, which reads `prev` and writes `next` where the in-place
/// version multiplies one element: the same product, written somewhere else.
/// Pass two then runs over `next`, which is where the state now is.
/// `tests::writing_the_state_to_the_next_ring_slot_changes_nothing` is what
/// holds the two together, and it is why the in-place version is kept rather
/// than expressed through this one: with `prev` and `next` the same buffer they
/// would have to alias, and nothing here needs that.
#[allow(clippy::too_many_arguments)]
fn delta_net_into(
    prev: &[f32],
    next: &mut [f32],
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
        for (i, &ki) in k.iter().enumerate() {
            let from = &prev[i * s + j0..][..w];
            for ((r, &p), a) in next[i * s + j0..][..w]
                .iter_mut()
                .zip(from)
                .zip(sk.iter_mut())
            {
                *r = p * g;
                *a += *r * ki;
            }
        }
        for ((dj, &vj), &skj) in d.iter_mut().zip(&v[j0..][..w]).zip(sk.iter()) {
            *dj = (vj - skj) * beta;
        }
        for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
            for ((r, &dj), oj) in next[i * s + j0..][..w]
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

/// [`depthwise_conv`] reading the history from `prev` and writing it to `next`.
///
/// The output is the same sum in the same order; where the in-place version
/// rotates the history down and appends the new sample, this writes the shifted
/// history into `next` instead.
fn depthwise_conv_into(
    x: &[f32],
    w: &[f32],
    kernel: usize,
    prev: &[f32],
    next: &mut [f32],
    out: &mut [f32],
) {
    let hist = kernel - 1;
    for (c, (o, &xc)) in out.iter_mut().zip(x).enumerate() {
        let taps = &w[c * kernel..(c + 1) * kernel];
        let h = &prev[c * hist..(c + 1) * hist];
        let mut sum = xc * taps[hist];
        for (j, &hv) in h.iter().enumerate() {
            sum += hv * taps[j];
        }
        *o = sum;
        let to = &mut next[c * hist..(c + 1) * hist];
        to[..hist - 1].copy_from_slice(&h[1..]);
        to[hist - 1] = xc;
    }
}

/// Slot `from` to read and slot `to` to write, out of one owner's ring of
/// `span`-wide buffers.
///
/// The two slots are always different, so they are disjoint pieces of the same
/// buffer; `split_at_mut` is how that is said to the compiler, and the order of
/// the two in memory is whichever way round they happen to be.
fn ring_pair(ring: &mut [f32], span: usize, from: usize, to: usize) -> (&[f32], &mut [f32]) {
    debug_assert_ne!(from, to);
    if from < to {
        let (lo, hi) = ring.split_at_mut(to * span);
        (&lo[from * span..][..span], &mut hi[..span])
    } else {
        let (lo, hi) = ring.split_at_mut(from * span);
        (&hi[..span], &mut lo[to * span..][..span])
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

    /// Writing the new state into the next ring slot has to give what updating
    /// it in place gives, **bit for bit**, in the readout and in the state.
    ///
    /// This is the whole claim the ring rests on: a per-row snapshot is free
    /// because the recurrence already reads the old state and writes the new
    /// one, so the only thing that changed is the address it writes to. The
    /// comparison is `==` and not a tolerance for the same reason as everywhere
    /// else in this file: the model's output is discontinuous in the arithmetic.
    ///
    /// Several tokens, so the chain from one row's slot to the next is covered
    /// and not only one hand-over, and the shapes are this model's `d_state` of
    /// 128, the tiny model's 32 and a 40 that leaves [`J_BLOCK`] a remainder.
    #[test]
    fn writing_the_state_to_the_next_ring_slot_changes_nothing() {
        let mut rand = noise(0x1115_10c7);
        for (s, tokens) in [(128usize, 4usize), (32, 3), (40, 3)] {
            let mut in_place = vec![0.0f32; s * s];
            // Two slots, used alternately, as a ring of two would be.
            let mut ring = vec![0.0f32; 2 * s * s];
            let scale = 1.0 / (s as f32).sqrt();
            for i in 0..tokens {
                let q: Vec<f32> = (0..s).map(|_| rand()).collect();
                let k: Vec<f32> = (0..s).map(|_| rand()).collect();
                let v: Vec<f32> = (0..s).map(|_| rand()).collect();
                let g = (rand().abs() * -2.0).exp();
                let beta = sigmoid(rand() * 4.0);
                let mut got = vec![0.0f32; s];
                let mut want = vec![0.0f32; s];
                let (from, to) = (i % 2, (i + 1) % 2);
                let (prev, next) = ring_pair(&mut ring, s * s, from, to);
                delta_net_into(prev, next, &q, &k, &v, g, beta, scale, &mut got);
                delta_net(&mut in_place, &q, &k, &v, g, beta, scale, &mut want);
                assert_eq!(got, want, "d_state {s} token {i}: readout");
                assert_eq!(
                    &ring[to * s * s..][..s * s],
                    &in_place[..],
                    "d_state {s} token {i}: state"
                );
            }
        }
    }

    /// The same for the convolution's history: the output and the history left
    /// behind have to be the in-place version's, bit for bit.
    ///
    /// The taps are the powers of ten of
    /// [`the_convolution_is_causal_over_its_history`], so a history written one
    /// sample out of place shows in the output as a whole decimal digit rather
    /// than as rounding.
    #[test]
    fn writing_the_convolution_history_to_the_next_ring_slot_changes_nothing() {
        let (kernel, channels) = (4usize, 3usize);
        let hist = kernel - 1;
        let w: Vec<f32> = (0..channels)
            .flat_map(|c| [1.0, 10.0, 100.0, 1000.0].map(|t| t * (c + 1) as f32))
            .collect();
        let mut in_place = vec![0.0f32; channels * hist];
        let mut ring = vec![0.0f32; 2 * channels * hist];
        for i in 0..6usize {
            let x: Vec<f32> = (0..channels).map(|c| (i * channels + c) as f32).collect();
            let mut got = vec![0.0f32; channels];
            let mut want = vec![0.0f32; channels];
            let (from, to) = (i % 2, (i + 1) % 2);
            let (prev, next) = ring_pair(&mut ring, channels * hist, from, to);
            depthwise_conv_into(&x, &w, kernel, prev, next, &mut got);
            depthwise_conv(&x, &w, kernel, &mut in_place, &mut want);
            assert_eq!(got, want, "token {i}: output");
            assert_eq!(
                &ring[to * channels * hist..][..channels * hist],
                &in_place[..],
                "token {i}: history"
            );
        }
    }

    /// Which slot is read and which is written, whichever way round they sit in
    /// the buffer, because the ring wraps and `to` is then below `from`.
    #[test]
    fn a_ring_pair_is_the_two_slots_asked_for() {
        // Three slots of two, so a wrap from the last slot to the first is in
        // the cases below.
        let mut ring = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (from, to) in [(0usize, 1usize), (1, 2), (2, 0)] {
            let before = ring.clone();
            let (prev, next) = ring_pair(&mut ring, 2, from, to);
            assert_eq!(prev, &before[from * 2..][..2], "read slot {from}");
            next.fill(-1.0);
            assert_eq!(&ring[to * 2..][..2], &[-1.0, -1.0], "wrote slot {to}");
            ring.copy_from_slice(&before);
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
