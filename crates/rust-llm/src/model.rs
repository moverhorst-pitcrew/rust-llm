//! The model: what the file says about its shape, and where every weight is.
//!
//! Nothing here copies a weight. A block holds [`TensorView`]s borrowed from
//! the memory map and f32 slices cast in place, so loading is a few hundred
//! bounds checks and no reads. The forward pass in [`crate::forward`] then
//! indexes them without re-checking.

use llm_core::gguf::TensorInfo;
use llm_core::quant::{GgmlType, can_dequantize};
use llm_core::{Gguf, GgufError, KernelError, Kernels, TensorView, qwen35};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Config(#[from] qwen35::ConfigError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error("tensor {name:?} has dimensions {got:?}, expected {want:?}")]
    Dims {
        name: String,
        want: Vec<u64>,
        got: Vec<u64>,
    },
    #[error("{key} is {got}, but this runner needs {want}")]
    Unsupported {
        key: &'static str,
        got: String,
        want: &'static str,
    },
    #[error("the embedding table {name:?} is {ty:?}, which cannot be expanded to f32")]
    EmbeddingType { name: String, ty: GgmlType },
}

type Result<T> = std::result::Result<T, ModelError>;

/// Everything about the model's shape that the forward pass needs, read out of
/// the file rather than assumed.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_blocks: usize,
    /// `true` where the block is linear (gated delta net) attention.
    pub recurrent: Vec<bool>,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Attention head dimension (`key_length` == `value_length`).
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// How many of the `head_dim` dimensions RoPE covers (64 of 256 here).
    pub n_rot: usize,
    pub rope_sections: [i32; 4],
    pub train_context: usize,
    // Gated delta net.
    pub conv_kernel: usize,
    /// Delta-net head dimension, shared by keys and values.
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_inner: usize,
    /// Chat end-of-turn first, then the plain end-of-text token.
    pub eos: Vec<u32>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let plan = qwen35::layer_plan(g)?;
        let head_dim = g.meta_u32("qwen35.attention.key_length")? as usize;
        let value_length = g.meta_u32("qwen35.attention.value_length")? as usize;
        if value_length != head_dim {
            return Err(ModelError::Unsupported {
                key: "qwen35.attention.value_length",
                got: value_length.to_string(),
                want: "the same as attention.key_length",
            });
        }
        let n_head = g.meta_u32("qwen35.attention.head_count")? as usize;
        let n_head_kv = g.meta_u32("qwen35.attention.head_count_kv")? as usize;
        if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(ModelError::Unsupported {
                key: "qwen35.attention.head_count_kv",
                got: n_head_kv.to_string(),
                want: "a divisor of attention.head_count",
            });
        }
        let n_v_heads = g.meta_u32("qwen35.ssm.time_step_rank")? as usize;
        let n_k_heads = g.meta_u32("qwen35.ssm.group_count")? as usize;
        if n_k_heads == 0 || !n_v_heads.is_multiple_of(n_k_heads) {
            return Err(ModelError::Unsupported {
                key: "qwen35.ssm.group_count",
                got: n_k_heads.to_string(),
                want: "a divisor of ssm.time_step_rank",
            });
        }
        let d_state = g.meta_u32("qwen35.ssm.state_size")? as usize;
        let d_inner = g.meta_u32("qwen35.ssm.inner_size")? as usize;
        if d_inner != d_state * n_v_heads {
            return Err(ModelError::Unsupported {
                key: "qwen35.ssm.inner_size",
                got: d_inner.to_string(),
                want: "ssm.state_size times ssm.time_step_rank",
            });
        }
        let n_rot = g.meta_u32("qwen35.rope.dimension_count")? as usize;
        if !n_rot.is_multiple_of(2) || n_rot > head_dim {
            return Err(ModelError::Unsupported {
                key: "qwen35.rope.dimension_count",
                got: n_rot.to_string(),
                want: "an even number no larger than the head dimension",
            });
        }
        let sections = g.meta_array("qwen35.rope.dimension_sections")?;
        let mut rope_sections = [0i32; 4];
        for (slot, v) in rope_sections.iter_mut().zip(sections) {
            *slot = i32::try_from(v.as_u64().unwrap_or(0)).unwrap_or(0);
        }

        // The chat template ends a turn with `<|im_end|>`; `config.json`'s
        // `eos_token_id` is `<|endoftext|>`. Generation stops on either.
        let mut eos = vec![g.meta_u32("tokenizer.ggml.eos_token_id")?];
        if let Ok(pad) = g.meta_u32("tokenizer.ggml.padding_token_id")
            && !eos.contains(&pad)
        {
            eos.push(pad);
        }

        Ok(Self {
            n_blocks: plan.n_blocks,
            recurrent: plan.recurrent,
            n_embd: g.meta_u32("qwen35.embedding_length")? as usize,
            n_ff: g.meta_u32("qwen35.feed_forward_length")? as usize,
            n_head,
            n_head_kv,
            head_dim,
            rms_eps: g.meta_f32("qwen35.attention.layer_norm_rms_epsilon")?,
            rope_theta: g.meta_f32("qwen35.rope.freq_base")?,
            n_rot,
            rope_sections,
            train_context: g.meta_u32("qwen35.context_length")? as usize,
            conv_kernel: g.meta_u32("qwen35.ssm.conv_kernel")? as usize,
            d_state,
            n_k_heads,
            n_v_heads,
            d_inner,
            eos,
        })
    }

    /// Query plus output gate: `attn_q` is twice as wide as the query alone.
    pub fn qg_dim(&self) -> usize {
        self.head_dim * self.n_head * 2
    }

    /// One attention head's worth of keys or values, over all KV heads.
    pub fn kv_dim(&self) -> usize {
        self.head_dim * self.n_head_kv
    }

    /// `attn_output` reads this many values.
    pub fn attn_dim(&self) -> usize {
        self.head_dim * self.n_head
    }

    /// Delta-net keys (and queries) over all key heads.
    pub fn key_dim(&self) -> usize {
        self.d_state * self.n_k_heads
    }

    /// The width of `attn_qkv` and of the depthwise convolution: q, k and v.
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.d_inner
    }

    pub fn n_full_blocks(&self) -> usize {
        self.recurrent.iter().filter(|r| !**r).count()
    }

    pub fn n_linear_blocks(&self) -> usize {
        self.recurrent.iter().filter(|r| **r).count()
    }
}

/// A full-attention block: GQA with per-head q/k norms and an output gate.
pub struct FullBlock<'a> {
    /// Index into the KV cache, counting only full-attention blocks.
    pub slot: usize,
    /// `[n_embd, head_dim * n_head * 2]`: the query and the output gate.
    pub qg: TensorView<'a>,
    pub k: TensorView<'a>,
    pub v: TensorView<'a>,
    pub o: TensorView<'a>,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

/// A linear-attention block: depthwise conv plus the gated delta rule.
pub struct LinearBlock<'a> {
    /// Index into the recurrent state cache, counting only linear blocks.
    pub slot: usize,
    pub qkv: TensorView<'a>,
    /// The output gate, `z` in the reference graph.
    pub gate: TensorView<'a>,
    /// `[conv_kernel, conv_dim]`, depthwise and causal.
    pub conv1d: &'a [f32],
    /// Already `-exp(A_log)`, so the decay is `exp(a * softplus(..))`.
    pub a: &'a [f32],
    pub dt_bias: &'a [f32],
    pub alpha: TensorView<'a>,
    pub beta: TensorView<'a>,
    /// Per-value-head RMSNorm over the head dimension.
    pub norm: &'a [f32],
    pub out: TensorView<'a>,
}

pub enum Attn<'a> {
    Full(FullBlock<'a>),
    Linear(LinearBlock<'a>),
}

pub struct Block<'a> {
    pub attn_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub attn: Attn<'a>,
    pub ffn_gate: TensorView<'a>,
    pub ffn_up: TensorView<'a>,
    pub ffn_down: TensorView<'a>,
}

/// The multi-token-prediction block, `blk.{n_blocks}`.
///
/// It is a full-attention block of the trunk's own shape plus four tensors,
/// and that is why [`Self::block`] is a plain [`Block`]: the forward pass runs
/// it through the same attention and the same feed-forward as any full block,
/// so the head cannot drift away from the trunk it drafts for.
///
/// What it does not have is an embedding table or an output head of its own.
/// The file carries no `nextn.embed_tokens` and no `nextn.shared_head_head`, so
/// tokens come in through `token_embd` and logits go out through it too, the
/// same tied tensor the trunk uses.
pub struct MtpBlock<'a> {
    /// RMSNorm weights for the embedding of the token being predicted from.
    pub enorm: &'a [f32],
    /// RMSNorm weights for the hidden state handed in.
    pub hnorm: &'a [f32],
    /// `[2 * n_embd, n_embd]`, reading the normed embedding followed by the
    /// normed hidden state: `ggml_concat(e_norm, h_norm, dim = 0)` in the
    /// reference, so the embedding is the first half.
    pub eh_proj: TensorView<'a>,
    /// Stands in for `output_norm` in front of the tied head.
    pub shared_head_norm: &'a [f32],
    pub block: Block<'a>,
}

pub struct Model<'a> {
    pub config: Config,
    pub blocks: Vec<Block<'a>>,
    /// The multi-token-prediction block, when the conversion kept it. `None`
    /// leaves the runner exactly as it was: drafting from the head is then
    /// unavailable and nothing else differs.
    pub mtp: Option<MtpBlock<'a>>,
    /// Doubles as the output head: this model ties them.
    pub token_embd: TensorView<'a>,
    pub output_norm: &'a [f32],
    pub kernels: Kernels,
}

impl<'a> Model<'a> {
    /// Resolves every weight and checks every dimension against the config.
    ///
    /// Nothing is read from the map here beyond the tensor index, so this is
    /// fast even on the 2.5 GiB file.
    pub fn load(g: &'a Gguf) -> Result<Self> {
        Self::with_kernels(g, Kernels::detect())
    }

    pub fn with_kernels(g: &'a Gguf, kernels: Kernels) -> Result<Self> {
        let c = Config::from_gguf(g)?;
        let n_embd = c.n_embd as u64;

        let token_embd = g.tensor("token_embd.weight")?;
        check_dims(&token_embd, &[n_embd, token_embd.dims[1]])?;
        if !can_dequantize(token_embd.ty) {
            return Err(ModelError::EmbeddingType {
                name: token_embd.name.to_string(),
                ty: token_embd.ty,
            });
        }
        // Both uses of the table need a kernel: the embedding lookup expands
        // one row, and the tied output head dots every row.
        kernels.vec_dot(token_embd.ty)?;

        let mut blocks = Vec::with_capacity(c.n_blocks);
        let (mut full, mut linear) = (0, 0);
        for il in 0..c.n_blocks {
            let slot = match c.recurrent[il] {
                true => &mut linear,
                false => &mut full,
            };
            let block = load_block(g, &c, &kernels, il, c.recurrent[il], *slot)?;
            *slot += 1;
            blocks.push(block);
        }

        // The MTP block's own KV cache is one block's worth, so its slot is 0;
        // it is a full-attention block whatever the trunk's last block was.
        let mtp = match qwen35::layer_plan(g)?.mtp {
            None => None,
            Some(il) => {
                let f32s = |suffix: &str| -> Result<&'a [f32]> {
                    let name = format!("blk.{il}.nextn.{suffix}.weight");
                    check_info(&name, g.tensor_info(&name)?, &[n_embd])?;
                    Ok(g.tensor_f32(&name)?)
                };
                let eh_proj = g.tensor(&format!("blk.{il}.nextn.eh_proj.weight"))?;
                check_dims(&eh_proj, &[2 * n_embd, n_embd])?;
                kernels.vec_dot(eh_proj.ty)?;
                Some(MtpBlock {
                    enorm: f32s("enorm")?,
                    hnorm: f32s("hnorm")?,
                    eh_proj,
                    shared_head_norm: f32s("shared_head_norm")?,
                    block: load_block(g, &c, &kernels, il, false, 0)?,
                })
            }
        };

        let output_norm = {
            let info = g.tensor_info("output_norm.weight")?;
            check_info("output_norm.weight", info, &[n_embd])?;
            g.tensor_f32("output_norm.weight")?
        };

        Ok(Self {
            config: c,
            blocks,
            mtp,
            token_embd,
            output_norm,
            kernels,
        })
    }

    /// How many tokens the output head can produce. Taken from the embedding
    /// table, not from metadata: the table is what the logits come out of.
    pub fn vocab(&self) -> usize {
        self.token_embd.dims[1] as usize
    }
}

/// One block's weights, checked against the config.
///
/// The MTP block goes through this as well, with `recurrent` false and its own
/// cache slot: it is a full-attention block of the same shape, and loading it
/// anywhere else would be a second place for a tensor name to be wrong.
fn load_block<'a>(
    g: &'a Gguf,
    c: &Config,
    kernels: &Kernels,
    il: usize,
    recurrent: bool,
    slot: usize,
) -> Result<Block<'a>> {
    let (n_embd, n_ff) = (c.n_embd as u64, c.n_ff as u64);
    let t = |suffix: &str| -> Result<TensorView<'a>> { Ok(g.tensor(&format!("blk.{il}.{suffix}"))?) };
    let f32s = |suffix: &str, want: &[u64]| -> Result<&'a [f32]> {
        let name = format!("blk.{il}.{suffix}");
        let info = g.tensor_info(&name)?;
        check_info(&name, info, want)?;
        Ok(g.tensor_f32(&name)?)
    };

    let attn = if recurrent {
        let block = LinearBlock {
            slot,
            qkv: t("attn_qkv.weight")?,
            gate: t("attn_gate.weight")?,
            conv1d: f32s(
                "ssm_conv1d.weight",
                &[c.conv_kernel as u64, c.conv_dim() as u64],
            )?,
            a: f32s("ssm_a", &[c.n_v_heads as u64])?,
            dt_bias: f32s("ssm_dt.bias", &[c.n_v_heads as u64])?,
            alpha: t("ssm_alpha.weight")?,
            beta: t("ssm_beta.weight")?,
            norm: f32s("ssm_norm.weight", &[c.d_state as u64])?,
            out: t("ssm_out.weight")?,
        };
        check_dims(&block.qkv, &[n_embd, c.conv_dim() as u64])?;
        check_dims(&block.gate, &[n_embd, c.d_inner as u64])?;
        check_dims(&block.alpha, &[n_embd, c.n_v_heads as u64])?;
        check_dims(&block.beta, &[n_embd, c.n_v_heads as u64])?;
        check_dims(&block.out, &[c.d_inner as u64, n_embd])?;
        Attn::Linear(block)
    } else {
        let block = FullBlock {
            slot,
            qg: t("attn_q.weight")?,
            k: t("attn_k.weight")?,
            v: t("attn_v.weight")?,
            o: t("attn_output.weight")?,
            q_norm: f32s("attn_q_norm.weight", &[c.head_dim as u64])?,
            k_norm: f32s("attn_k_norm.weight", &[c.head_dim as u64])?,
        };
        check_dims(&block.qg, &[n_embd, c.qg_dim() as u64])?;
        check_dims(&block.k, &[n_embd, c.kv_dim() as u64])?;
        check_dims(&block.v, &[n_embd, c.kv_dim() as u64])?;
        check_dims(&block.o, &[c.attn_dim() as u64, n_embd])?;
        Attn::Full(block)
    };

    let block = Block {
        attn_norm: f32s("attn_norm.weight", &[n_embd])?,
        post_attn_norm: f32s("post_attention_norm.weight", &[n_embd])?,
        attn,
        ffn_gate: t("ffn_gate.weight")?,
        ffn_up: t("ffn_up.weight")?,
        ffn_down: t("ffn_down.weight")?,
    };
    check_dims(&block.ffn_gate, &[n_embd, n_ff])?;
    check_dims(&block.ffn_up, &[n_embd, n_ff])?;
    check_dims(&block.ffn_down, &[n_ff, n_embd])?;
    for w in [&block.ffn_gate, &block.ffn_up, &block.ffn_down] {
        kernels.vec_dot(w.ty)?;
    }
    Ok(block)
}

fn check_dims(t: &TensorView<'_>, want: &[u64]) -> Result<()> {
    if t.dims == want {
        return Ok(());
    }
    Err(ModelError::Dims {
        name: t.name.to_string(),
        want: want.to_vec(),
        got: t.dims.to_vec(),
    })
}

fn check_info(name: &str, info: &TensorInfo, want: &[u64]) -> Result<()> {
    if info.dims == want {
        return Ok(());
    }
    Err(ModelError::Dims {
        name: name.to_string(),
        want: want.to_vec(),
        got: info.dims.clone(),
    })
}
