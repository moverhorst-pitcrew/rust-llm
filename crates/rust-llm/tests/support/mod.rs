//! A tiny but valid Qwen3.5 GGUF, and the `tokenizer.json` that belongs with
//! it, written byte by byte.
//!
//! The product is otherwise only exercised end to end against the 2.8 GB file,
//! which CI does not have. This is the same architecture at the smallest
//! dimensions its own constraints allow: two blocks, one linear and one full
//! attention, `Q8_0` weights so the real dequant-free kernel path runs, and a
//! few hundred kilobytes on disk. It proves the surface, not the output: the
//! weights are noise from a fixed seed, so what comes out is meaningless but
//! identical on every machine and every run.
//!
//! The byte layout is the one `crates/llm-core/tests/gguf.rs` already builds
//! against; the writer here adds the quantized block layouts and the metadata
//! a real conversion carries.
//!
//! Two shapes are not free choices:
//!
//! - **`n_embd` is 256, not 32.** The embedding table is both read by the
//!   embedding lookup, which needs `dequantize_row`, and dotted by the tied
//!   output head, which needs a kernel. The only types that have both are the
//!   k-quants, whose super-block is 256 values wide, so the table's row length
//!   has to be a multiple of 256. It is `Q6_K`, which is what the real
//!   `Q4_K_M` conversion uses for this tensor too.
//! - **Every other quantized row length is a multiple of 32**, which is
//!   `Q8_0`'s block.
//!
//! It carries the **multi-token-prediction block** as well, as `blk.2`, with
//! the four `nextn.*` tensors around a third full-attention block. Without it
//! the MTP draft path, its catch-up and its rollback would only be reachable
//! with the real weights, and the one way an MTP block can be wrong is the one
//! way nothing fails: verification is exact, so a wrong block costs acceptance
//! and not correctness.
//!
//! The tokenizer is byte-level BPE over the bytes a chat prompt and an ASCII
//! test prompt actually contain, rather than all 256. Every token is then one
//! printable byte, so whatever the fixture generates is valid UTF-8 and a test
//! can compare the streamed text directly. With the full byte alphabet most
//! generated tokens are half a multi-byte character, the streaming path
//! reprints the whole answer, and every comparison has to work around that
//! instead of checking what it came for.

use std::path::{Path, PathBuf};

use half::f16;
use serde_json::json;

// ------------------------------------------------------------- the shape

/// Trunk blocks, which is what `layer_plan` reports: `block_count` is one
/// more, because it counts the MTP block.
pub const N_BLOCKS: usize = 2;
pub const N_EMBD: usize = 256;
pub const N_FF: usize = 32;
pub const HEAD_DIM: usize = 32;
pub const N_HEAD: usize = 1;
pub const N_HEAD_KV: usize = 1;
pub const N_ROT: usize = 32;
/// Delta-net head dimension, which is also the inner size at one value head.
pub const D_STATE: usize = 32;
pub const N_V_HEADS: usize = 1;
pub const N_K_HEADS: usize = 1;
pub const D_INNER: usize = D_STATE * N_V_HEADS;
pub const CONV_KERNEL: usize = 4;
/// Keys plus queries plus values, the width of `attn_qkv` and of the
/// convolution.
pub const CONV_DIM: usize = D_STATE * N_K_HEADS * 2 + D_INNER;

/// Rows in the embedding table, a few more than the tokenizer can produce so
/// the padding the loader insists on is exercised.
pub const VOCAB: usize = 128;

/// `Q8_0` scale, picked so that a 256-wide matmul against a unit-RMS
/// activation lands within a few units of zero: the quants are uniform over
/// `-127..=127`, so the weights have a standard deviation of about 0.11 and
/// the products of 256 of them about 1.7. Far enough from zero that the model
/// does something, far enough from the exponentials in the gates that nothing
/// saturates.
const Q8_SCALE: f32 = 0.0015;
/// `Q6_K` scale, chosen the same way: the 6-bit quants are centred by the
/// constant -32 bias, so a scale of 0.006 gives about the same spread.
const Q6_SCALE: f32 = 0.006;

/// The special tokens the chat template needs, in the order they get their
/// ids after the byte-level ones.
const SPECIALS: [&str; 5] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
];

/// Tab, newline and the printable ASCII range: the bytes the chat template
/// and the tests use, and nothing else.
const BYTES: [u8; 97] = {
    let mut out = [0u8; 97];
    out[0] = b'\t';
    out[1] = b'\n';
    let mut i = 2;
    while i < 97 {
        out[i] = 0x20 + (i as u8 - 2);
        i += 1;
    }
    out
};

const BYTE_TOKENS: usize = BYTES.len();

/// A padding row past everything the tokenizer can produce, for a fixture that
/// should not stop by itself. The model can still land on it, which is why the
/// tests that use it check the stop reason they got rather than assuming one.
pub const UNREACHED_EOS: u32 = (VOCAB - 1) as u32;

/// Where the two files landed.
pub struct Tiny {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

/// Writes the pair into its own directory under Cargo's test temp directory
/// and returns their paths.
///
/// `eos` is the token id generation stops on. It is a parameter because what
/// this model produces is noise: a test that wants to see the end-of-sequence
/// path has to point `eos` at a token the model actually reaches, and it can
/// find that token by running the library and then asking for a fixture that
/// stops on it.
pub fn write(name: &str, eos: u32) -> Tiny {
    write_with(name, eos, true)
}

/// The same file without the MTP block, which is the conversion this runner
/// started on: the trunk is identical and only drafting from the head is
/// unavailable. One test needs it, to check that asking for a head the file
/// does not have is refused rather than quietly ignored.
// Not every test binary that includes this module needs it; the two that do
// are `tests/mtp.rs` and `tests/tiny_model.rs`.
#[allow(dead_code)]
pub fn write_without_mtp(name: &str, eos: u32) -> Tiny {
    write_with(name, eos, false)
}

fn write_with(name: &str, eos: u32, mtp: bool) -> Tiny {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let model = dir.join("model.gguf");
    let tokenizer = dir.join("tokenizer.json");
    std::fs::write(&model, gguf(eos, mtp)).expect("write the fixture model");
    std::fs::write(&tokenizer, tokenizer_json()).expect("write the fixture tokenizer");
    Tiny { model, tokenizer }
}

// ----------------------------------------------------------- the metadata

fn gguf(eos: u32, mtp: bool) -> Vec<u8> {
    let tokens = vocab_strings();
    let mut b = Builder::default();
    b = b
        .kv("general.architecture", kv(8, p_str("qwen35")))
        .kv("general.name", kv(8, p_str("tiny-qwen35")))
        // One more than the trunk, as the real conversion does: the last one
        // is the MTP block and `nextn_predict_layers` is what takes it out of
        // the trunk again.
        .kv("qwen35.block_count", u32v(N_BLOCKS as u32 + u32::from(mtp)))
        .kv("qwen35.nextn_predict_layers", u32v(u32::from(mtp)))
        .kv("qwen35.context_length", u32v(4096))
        .kv("qwen35.embedding_length", u32v(N_EMBD as u32))
        .kv("qwen35.feed_forward_length", u32v(N_FF as u32))
        .kv("qwen35.attention.head_count", u32v(N_HEAD as u32))
        .kv("qwen35.attention.head_count_kv", u32v(N_HEAD_KV as u32))
        .kv("qwen35.attention.key_length", u32v(HEAD_DIM as u32))
        .kv("qwen35.attention.value_length", u32v(HEAD_DIM as u32))
        .kv("qwen35.attention.layer_norm_rms_epsilon", f32v(1e-6))
        // Block 0 is the linear one and block 1 the full-attention one, the
        // same ratio the real file has at a much smaller scale. The third
        // entry is the MTP block, which is full attention as it is there.
        .kv("qwen35.attention.recurrent_layers", {
            let n = N_BLOCKS + usize::from(mtp);
            kv(9, p_array(7, n as u64, &[1u8, 0, 0][..n]))
        })
        .kv("qwen35.rope.freq_base", f32v(1e7))
        .kv("qwen35.rope.dimension_count", u32v(N_ROT as u32))
        // Sections for the three position axes a text-only path uses, with a
        // zero fourth, as in the real file: no rotated dimension pair is left
        // for the axis this runner has no position for.
        .kv(
            "qwen35.rope.dimension_sections",
            kv(
                9,
                p_array(
                    5,
                    4,
                    &[6i32, 6, 4, 0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            ),
        )
        .kv("qwen35.ssm.conv_kernel", u32v(CONV_KERNEL as u32))
        .kv("qwen35.ssm.state_size", u32v(D_STATE as u32))
        .kv("qwen35.ssm.inner_size", u32v(D_INNER as u32))
        .kv("qwen35.ssm.time_step_rank", u32v(N_V_HEADS as u32))
        .kv("qwen35.ssm.group_count", u32v(N_K_HEADS as u32))
        .kv("tokenizer.ggml.model", kv(8, p_str("gpt2")))
        .kv("tokenizer.ggml.pre", kv(8, p_str("qwen35")))
        .kv(
            "tokenizer.ggml.tokens",
            kv(
                9,
                p_array(
                    8,
                    tokens.len() as u64,
                    &tokens.iter().flat_map(|t| p_str(t)).collect::<Vec<u8>>(),
                ),
            ),
        )
        .kv(
            "tokenizer.ggml.token_type",
            kv(
                9,
                p_array(
                    4,
                    VOCAB as u64,
                    &(0..VOCAB)
                        .flat_map(|i| token_type(i).to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            ),
        )
        .kv("tokenizer.ggml.eos_token_id", u32v(eos))
        .kv("tokenizer.ggml.padding_token_id", u32v(BYTE_TOKENS as u32));

    // ------------------------------------------------------- the weights
    let mut n = Noise::new(0x5151_2026);
    b = b.tensor(
        "token_embd.weight",
        &[N_EMBD, VOCAB],
        Q6_K,
        q6_k(&mut n, N_EMBD, VOCAB),
    );
    b = b.tensor("output_norm.weight", &[N_EMBD], F32, ones(N_EMBD));

    // The trunk, then the MTP block, which is a full-attention block with four
    // tensors in front of it.
    for il in 0..N_BLOCKS + usize::from(mtp) {
        let t = |suffix: &str| format!("blk.{il}.{suffix}");
        b = b.tensor(&t("attn_norm.weight"), &[N_EMBD], F32, ones(N_EMBD));
        b = b.tensor(
            &t("post_attention_norm.weight"),
            &[N_EMBD],
            F32,
            ones(N_EMBD),
        );
        if il == N_BLOCKS {
            // `eh_proj` reads the normed embedding followed by the normed
            // hidden state, so it is twice as wide as the embedding.
            b = b.tensor(
                &t("nextn.eh_proj.weight"),
                &[2 * N_EMBD, N_EMBD],
                Q8_0,
                q8_0(&mut n, 2 * N_EMBD, N_EMBD),
            );
            for name in ["nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"] {
                b = b.tensor(&t(&format!("{name}.weight")), &[N_EMBD], F32, ones(N_EMBD));
            }
        }

        if il == 0 {
            // The linear (gated delta net) block.
            b = b.tensor(
                &t("attn_qkv.weight"),
                &[N_EMBD, CONV_DIM],
                Q8_0,
                q8_0(&mut n, N_EMBD, CONV_DIM),
            );
            b = b.tensor(
                &t("attn_gate.weight"),
                &[N_EMBD, D_INNER],
                Q8_0,
                q8_0(&mut n, N_EMBD, D_INNER),
            );
            b = b.tensor(
                &t("ssm_conv1d.weight"),
                &[CONV_KERNEL, CONV_DIM],
                F32,
                floats(&mut n, CONV_KERNEL * CONV_DIM, 0.5),
            );
            // `ssm_a` is already `-exp(A_log)` in a real file, so it is
            // negative and the decay `exp(softplus(..) * a)` lands in (0, 1).
            b = b.tensor(
                &t("ssm_a"),
                &[N_V_HEADS],
                F32,
                f32_bytes(&[-1.0; N_V_HEADS]),
            );
            b = b.tensor(
                &t("ssm_dt.bias"),
                &[N_V_HEADS],
                F32,
                f32_bytes(&[0.0; N_V_HEADS]),
            );
            b = b.tensor(
                &t("ssm_alpha.weight"),
                &[N_EMBD, N_V_HEADS],
                Q8_0,
                q8_0(&mut n, N_EMBD, N_V_HEADS),
            );
            b = b.tensor(
                &t("ssm_beta.weight"),
                &[N_EMBD, N_V_HEADS],
                Q8_0,
                q8_0(&mut n, N_EMBD, N_V_HEADS),
            );
            b = b.tensor(&t("ssm_norm.weight"), &[D_STATE], F32, ones(D_STATE));
            b = b.tensor(
                &t("ssm_out.weight"),
                &[D_INNER, N_EMBD],
                Q8_0,
                q8_0(&mut n, D_INNER, N_EMBD),
            );
        } else {
            // The full-attention block. `attn_q` is twice the query width:
            // the output gate sits in the second half of each head's slice.
            b = b.tensor(
                &t("attn_q.weight"),
                &[N_EMBD, HEAD_DIM * N_HEAD * 2],
                Q8_0,
                q8_0(&mut n, N_EMBD, HEAD_DIM * N_HEAD * 2),
            );
            b = b.tensor(
                &t("attn_k.weight"),
                &[N_EMBD, HEAD_DIM * N_HEAD_KV],
                Q8_0,
                q8_0(&mut n, N_EMBD, HEAD_DIM * N_HEAD_KV),
            );
            b = b.tensor(
                &t("attn_v.weight"),
                &[N_EMBD, HEAD_DIM * N_HEAD_KV],
                Q8_0,
                q8_0(&mut n, N_EMBD, HEAD_DIM * N_HEAD_KV),
            );
            b = b.tensor(
                &t("attn_output.weight"),
                &[HEAD_DIM * N_HEAD, N_EMBD],
                Q8_0,
                q8_0(&mut n, HEAD_DIM * N_HEAD, N_EMBD),
            );
            b = b.tensor(&t("attn_q_norm.weight"), &[HEAD_DIM], F32, ones(HEAD_DIM));
            b = b.tensor(&t("attn_k_norm.weight"), &[HEAD_DIM], F32, ones(HEAD_DIM));
        }

        b = b.tensor(
            &t("ffn_gate.weight"),
            &[N_EMBD, N_FF],
            Q8_0,
            q8_0(&mut n, N_EMBD, N_FF),
        );
        b = b.tensor(
            &t("ffn_up.weight"),
            &[N_EMBD, N_FF],
            Q8_0,
            q8_0(&mut n, N_EMBD, N_FF),
        );
        b = b.tensor(
            &t("ffn_down.weight"),
            &[N_FF, N_EMBD],
            Q8_0,
            q8_0(&mut n, N_FF, N_EMBD),
        );
    }

    b.build()
}

/// Normal for everything the tokenizer can produce, `UNUSED` for the padding
/// rows past it, which is what the loader requires.
fn token_type(id: usize) -> u32 {
    if id < BYTE_TOKENS + SPECIALS.len() {
        1
    } else {
        5
    }
}

// ---------------------------------------------------------- the tokenizer

/// GPT-2's byte-to-unicode table, for the bytes in [`BYTES`].
///
/// A printable byte stands for itself; the other 68 of the 256 are moved into
/// the private-use range, in byte order, which is what keeps a byte-level
/// vocabulary free of characters the JSON would have to escape. Tab and
/// newline are the 10th and 11th such byte, so they come out as U+0109 and
/// U+010A.
fn byte_level_alphabet() -> Vec<String> {
    let printable = |b: u8| {
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    BYTES
        .iter()
        .map(|&b| {
            let c = if printable(b) {
                char::from_u32(b as u32)
            } else {
                // Its index among the bytes that need moving.
                let moved = (0..b).filter(|&p| !printable(p)).count() as u32;
                char::from_u32(256 + moved)
            };
            c.expect("every code point here is valid").to_string()
        })
        .collect()
}

/// The vocabulary as the GGUF carries it: the byte-level alphabet, the special
/// tokens, then the padding the embedding table has rows for.
fn vocab_strings() -> Vec<String> {
    let mut out = byte_level_alphabet();
    out.extend(SPECIALS.iter().map(|s| s.to_string()));
    while out.len() < VOCAB {
        out.push(format!("[PAD{}]", out.len()));
    }
    out
}

/// A byte-level BPE with no merges, so a token is a byte and the only
/// multi-character tokens are the special ones. There is nothing to be gained
/// from merges here: what the tests check is the surface, and a prompt
/// tokenizing to more tokens only makes them slightly longer.
fn tokenizer_json() -> String {
    let alphabet = byte_level_alphabet();
    let mut vocab = serde_json::Map::new();
    for (i, token) in alphabet.iter().enumerate() {
        vocab.insert(token.clone(), json!(i));
    }
    let added: Vec<_> = SPECIALS
        .iter()
        .enumerate()
        .map(|(i, content)| {
            json!({
                "id": BYTE_TOKENS + i,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let byte_level = json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": true,
    });
    serde_json::to_string_pretty(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": byte_level,
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": [],
        },
    }))
    .expect("the fixture tokenizer serialises")
}

// ------------------------------------------------------------- the weights

const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q6_K: u32 = 14;

/// xorshift64, so the file is byte-identical on every machine and every run.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    }
}

fn ones(n: usize) -> Vec<u8> {
    f32_bytes(&vec![1.0f32; n])
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(n: &mut Noise, len: usize, scale: f32) -> Vec<u8> {
    (0..len)
        .flat_map(|_| (n.unit() * scale).to_le_bytes())
        .collect()
}

/// `rows` rows of `k` `Q8_0` values: an f16 scale and 32 signed quants per
/// block.
fn q8_0(n: &mut Noise, k: usize, rows: usize) -> Vec<u8> {
    assert!(
        k.is_multiple_of(32),
        "a Q8_0 row is a whole number of blocks"
    );
    let scale = f16::from_f32(Q8_SCALE).to_bits().to_le_bytes();
    let mut out = Vec::with_capacity(rows * (k / 32) * 34);
    for _ in 0..rows * (k / 32) {
        out.extend_from_slice(&scale);
        for _ in 0..32 {
            out.push((((n.next() % 255) as i32 - 127) as i8) as u8);
        }
    }
    out
}

/// `rows` rows of `k` `Q6_K` values.
///
/// One super-block is 256 values: 128 bytes of low nibbles, 64 bytes holding
/// two high bits each, 16 signed scales and an f16 scale. Value
/// `h * 128 + g * 32 + l` takes its low nibble from `ql[h * 64 + 32 * (g % 2)
/// + l]`, in the low half for `g < 2` and the high half above it, and its two
/// top bits from `qh[h * 32 + l] >> (2 * g)`. All sixteen scales are 1, so a
/// weight is `d * (q - 32)`.
fn q6_k(n: &mut Noise, k: usize, rows: usize) -> Vec<u8> {
    assert!(
        k.is_multiple_of(256),
        "a Q6_K row is a whole number of super-blocks"
    );
    let mut out = Vec::with_capacity(rows * (k / 256) * 210);
    for _ in 0..rows * (k / 256) {
        let q: Vec<u8> = (0..256).map(|_| (n.next() % 64) as u8).collect();
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for h in 0..2 {
            for g in 0..4 {
                for l in 0..32 {
                    let v = q[h * 128 + g * 32 + l];
                    let at = h * 64 + 32 * (g % 2) + l;
                    if g < 2 {
                        ql[at] |= v & 0xF;
                    } else {
                        ql[at] |= (v & 0xF) << 4;
                    }
                    qh[h * 32 + l] |= ((v >> 4) & 3) << (2 * g);
                }
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&[1u8; 16]);
        out.extend_from_slice(&f16::from_f32(Q6_SCALE).to_bits().to_le_bytes());
    }
    out
}

// --------------------------------------------------------- the byte layout

fn p_str(s: &str) -> Vec<u8> {
    let mut o = (s.len() as u64).to_le_bytes().to_vec();
    o.extend_from_slice(s.as_bytes());
    o
}

fn p_array(elem_ty: u32, n: u64, elems: &[u8]) -> Vec<u8> {
    let mut o = elem_ty.to_le_bytes().to_vec();
    o.extend_from_slice(&n.to_le_bytes());
    o.extend_from_slice(elems);
    o
}

fn kv(ty: u32, payload: Vec<u8>) -> Vec<u8> {
    let mut o = ty.to_le_bytes().to_vec();
    o.extend(payload);
    o
}

fn u32v(v: u32) -> Vec<u8> {
    kv(4, v.to_le_bytes().to_vec())
}

fn f32v(v: f32) -> Vec<u8> {
    kv(6, v.to_le_bytes().to_vec())
}

const ALIGNMENT: usize = 32;

#[derive(Default)]
struct Builder {
    kvs: Vec<(String, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, u64)>,
    data: Vec<u8>,
}

impl Builder {
    fn kv(mut self, key: &str, value: Vec<u8>) -> Self {
        self.kvs.push((key.to_string(), value));
        self
    }

    fn tensor(mut self, name: &str, dims: &[usize], ty: u32, bytes: Vec<u8>) -> Self {
        let offset = self.data.len().next_multiple_of(ALIGNMENT) as u64;
        self.data.resize(offset as usize, 0);
        self.data.extend_from_slice(&bytes);
        self.tensors.push((
            name.to_string(),
            dims.iter().map(|&d| d as u64).collect(),
            ty,
            offset,
        ));
        self
    }

    fn build(&self) -> Vec<u8> {
        let mut o = b"GGUF".to_vec();
        o.extend_from_slice(&3u32.to_le_bytes());
        o.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        o.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            o.extend(p_str(k));
            o.extend_from_slice(v);
        }
        for (name, dims, ty, offset) in &self.tensors {
            o.extend(p_str(name));
            o.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                o.extend_from_slice(&d.to_le_bytes());
            }
            o.extend_from_slice(&ty.to_le_bytes());
            o.extend_from_slice(&offset.to_le_bytes());
        }
        o.resize(o.len().next_multiple_of(ALIGNMENT), 0);
        o.extend_from_slice(&self.data);
        o
    }
}
