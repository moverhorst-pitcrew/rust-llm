//! The model's tokenizer, loaded from its `tokenizer.json`.
//!
//! This is the one component where writing it by hand buys nothing and risks
//! the correctness gate. The pretokenizer regex contains a lookahead
//! (`\s+(?!\S)`), the normalizer is NFC, and byte-level BPE has enough corners
//! that a reimplementation is where token-for-token agreement quietly dies.
//! Tokenization is also a one-time cost per prompt, not hot.
//!
//! `tokenizers` is taken with `default-features = false` and `fancy-regex`:
//! the default features pull `onig` and `esaxx_fast`, which need a C++
//! compiler, and `fancy-regex` is the pure-Rust engine that supports the
//! lookahead. CI fails if `cc` ever appears in `Cargo.lock`.

use llm_core::{Gguf, GgufError};

/// Token ids that end a turn. Re-exported from [`crate::chat`] because this is
/// where a caller tokenizing a prompt will look for them.
pub use crate::chat::{ENDOFTEXT_ID, IM_END_ID, STOP_TOKENS};

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("loading tokenizer from {path}: {message}")]
    Load { path: String, message: String },
    #[error("tokenizing: {0}")]
    Encode(String),
    #[error("detokenizing: {0}")]
    Decode(String),
}

/// What it means for the weights and the tokenizer to disagree.
///
/// Worth its own error type: a mismatched pair does not crash, it produces
/// token ids that index the wrong embeddings, and the output is wrong in a way
/// that looks like a bad model rather than a bad pairing.
#[derive(Debug, thiserror::Error)]
pub enum VocabMismatch {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("the weights say tokenizer.ggml.{key} = {got:?}, expected {want:?}")]
    Metadata {
        key: &'static str,
        want: &'static str,
        got: String,
    },
    #[error("the weights hold {got} tokens, fewer than the {want} in tokenizer.json")]
    TooFewTokens { got: usize, want: usize },
    #[error("token {id} is {got:?} in the weights and {want:?} in tokenizer.json")]
    Token { id: u32, got: String, want: String },
    #[error("token {id} has no entry in tokenizer.json, but the weights call it {got:?}")]
    MissingToken { id: u32, got: String },
    #[error(
        "token {id} is past the end of tokenizer.json, so the weights should mark it unused, but its type is {got}"
    )]
    NotUnused { id: u32, got: i64 },
}

/// ggml's `LLAMA_TOKEN_TYPE_UNUSED`: a slot that pads the embedding matrix out
/// to a round size and is not part of the vocabulary.
const TOKEN_TYPE_UNUSED: u64 = 5;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| TokenizerError::Load {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Ok(Self { inner })
    }

    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let inner = json
            .parse()
            .map_err(|e: tokenizers::Error| TokenizerError::Load {
                path: "<memory>".to_string(),
                message: e.to_string(),
            })?;
        Ok(Self { inner })
    }

    /// Token ids for `text`.
    ///
    /// Control tokens in the text are always recognised: a `<|im_start|>` in
    /// the input becomes token 248045, not its eleven characters. That is what
    /// makes a rendered chat prompt tokenize the way the reference runner
    /// tokenizes it, and it is also why text that came from a user must not be
    /// pasted into a prompt without being checked for control tokens first.
    ///
    /// `add_special_tokens` is the post-processor's knob, not that one. This
    /// model's post-processor is a bare ByteLevel which adds nothing, so the
    /// flag makes no difference here; the fixtures assert that, so a
    /// `tokenizer.json` that grows a post-processor shows up as a test
    /// failure rather than as a prompt that silently gained a token.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Number of ids the tokenizer can produce, added tokens included. Smaller
    /// than the embedding matrix, which is padded.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Checks that this tokenizer is the one these weights were trained with.
    ///
    /// Compares every token the tokenizer can produce against the vocabulary
    /// embedded in the GGUF, and requires the slots past the end of the
    /// tokenizer to be the padding the embedding matrix needs rather than
    /// vocabulary we are failing to produce. Call it at load time: it costs a
    /// few milliseconds and turns a mismatched pair into an error instead of
    /// subtly wrong ids.
    pub fn verify_against_weights(&self, weights: &Gguf) -> Result<(), VocabMismatch> {
        for (key, want) in [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "qwen35"),
        ] {
            let got = weights.meta_str(key)?;
            if got != want {
                return Err(VocabMismatch::Metadata {
                    key: key.rsplit('.').next().unwrap_or(key),
                    want,
                    got: got.to_string(),
                });
            }
        }

        let tokens = weights.meta_str_array("tokenizer.ggml.tokens")?;
        let types = weights.meta_array("tokenizer.ggml.token_type")?;
        let n = self.vocab_size();
        if tokens.len() < n {
            return Err(VocabMismatch::TooFewTokens {
                got: tokens.len(),
                want: n,
            });
        }

        for (id, &got) in tokens.iter().take(n).enumerate() {
            let id = id as u32;
            match self.id_to_token(id) {
                Some(want) if want == got => {}
                Some(want) => {
                    return Err(VocabMismatch::Token {
                        id,
                        got: got.to_string(),
                        want,
                    });
                }
                None => {
                    return Err(VocabMismatch::MissingToken {
                        id,
                        got: got.to_string(),
                    });
                }
            }
        }

        for (id, ty) in types.iter().enumerate().skip(n) {
            let ty = ty.as_u64();
            if ty != Some(TOKEN_TYPE_UNUSED) {
                return Err(VocabMismatch::NotUnused {
                    id: id as u32,
                    got: ty.map(|v| v as i64).unwrap_or(-1),
                });
            }
        }
        Ok(())
    }
}
