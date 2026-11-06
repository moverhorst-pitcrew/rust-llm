//! The qwen35 facts that have to come out of the file rather than out of
//! arithmetic.
//!
//! Two of them bite immediately. `qwen35.block_count` counts the
//! multi-token-prediction block, so the trunk is `block_count` minus
//! `nextn_predict_layers`, and whether that last block is in the file at all
//! depends on the conversion: the MTP-preserved build ships it as `blk.32`,
//! the older one dropped it. And the hybrid layer split is published as a bool
//! array, which does not have to agree with `full_attention_interval`.

use crate::gguf::{Gguf, GgufError};

pub const ARCH: &str = "qwen35";
const RECURRENT_KEY: &str = "qwen35.attention.recurrent_layers";

/// Which blocks exist and what kind each one is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    /// Trunk blocks actually present in the file.
    pub n_blocks: usize,
    /// `true` where the block is linear (gated delta net) attention, `false`
    /// where it is full attention with a KV cache.
    pub recurrent: Vec<bool>,
    /// Index of the multi-token-prediction block, `blk.{n_blocks}`, when the
    /// conversion kept it. `None` where it was stripped: the trunk is the
    /// same either way and only drafting from the head is unavailable.
    pub mtp: Option<usize>,
}

impl LayerPlan {
    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.indices(false)
    }

    pub fn linear_attention_layers(&self) -> Vec<usize> {
        self.indices(true)
    }

    fn indices(&self, recurrent: bool) -> Vec<usize> {
        self.recurrent
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r == recurrent)
            .map(|(i, _)| i)
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("expected general.architecture {ARCH:?}, file says {0:?}")]
    Architecture(String),
    #[error(
        "qwen35.block_count {declared} minus nextn_predict_layers {skipped} implies {expected} blocks, but tensor {missing:?} is not in the file"
    )]
    MissingBlock {
        declared: u32,
        skipped: u32,
        expected: usize,
        missing: String,
    },
    #[error("the file has a block {found} beyond the {expected} the metadata implies")]
    UnexpectedBlock { found: usize, expected: usize },
    #[error("{RECURRENT_KEY} has {got} entries, need at least {expected}")]
    RecurrentLayersTooShort { got: usize, expected: usize },
}

/// Derives the real block count and the per-block layer kind.
pub fn layer_plan(g: &Gguf) -> Result<LayerPlan, ConfigError> {
    let arch = g.meta_str("general.architecture")?;
    if arch != ARCH {
        return Err(ConfigError::Architecture(arch.to_string()));
    }

    let declared = g.meta_u32("qwen35.block_count")?;
    let skipped = match g.meta("qwen35.nextn_predict_layers") {
        Some(_) => g.meta_u32("qwen35.nextn_predict_layers")?,
        None => 0,
    };
    let expected = declared.saturating_sub(skipped) as usize;

    for i in 0..expected {
        let probe = format!("blk.{i}.attn_norm.weight");
        if !g.has_tensor(&probe) {
            return Err(ConfigError::MissingBlock {
                declared,
                skipped,
                expected,
                missing: probe,
            });
        }
    }
    // The blocks the metadata skips are the MTP ones, `blk.{expected}` up to
    // `blk.{declared}`. Whether they are in the file is the one thing that
    // differs between the two conversions of these weights.
    let mtp = (skipped > 0 && g.has_tensor(&format!("blk.{expected}.attn_norm.weight")))
        .then_some(expected);
    let past = expected + usize::from(mtp.is_some());
    if g.has_tensor(&format!("blk.{past}.attn_norm.weight")) {
        return Err(ConfigError::UnexpectedBlock {
            found: past,
            expected,
        });
    }

    let recurrent = g.meta_bool_array(RECURRENT_KEY)?;
    if recurrent.len() < expected {
        return Err(ConfigError::RecurrentLayersTooShort {
            got: recurrent.len(),
            expected,
        });
    }

    Ok(LayerPlan {
        n_blocks: expected,
        recurrent: recurrent[..expected].to_vec(),
        mtp,
    })
}
