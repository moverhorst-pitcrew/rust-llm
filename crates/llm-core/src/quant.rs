//! ggml tensor types, their on-disk block layouts, and the int8 activation
//! blocks the quantized kernels consume.
//!
//! Weights are never dequantized to f32 on the product path. The kernels read
//! the quantized bytes straight out of the memory map and meet them with
//! activations that have been quantized to int8, so every dot product runs in
//! integer arithmetic and only the per-block scales are floating point.

use half::f16;

/// Super-block length of the k-quants (`Q4_K`, `Q6_K`, `Q8_K`).
pub const QK_K: usize = 256;
/// Block length of `Q8_0`.
pub const QK8_0: usize = 32;

/// `d: f16`, `dmin: f16`, `scales: [u8; 12]`, `qs: [u8; 128]`.
pub const Q4_K_BYTES: usize = 2 + 2 + 12 + 128;
/// `ql: [u8; 128]`, `qh: [u8; 64]`, `scales: [i8; 16]`, `d: f16`.
pub const Q6_K_BYTES: usize = 128 + 64 + 16 + 2;
/// `d: f16`, `qs: [i8; 32]`.
pub const Q8_0_BYTES: usize = 2 + 32;

/// A ggml tensor type, by its on-disk discriminant.
///
/// The variants carry the ggml names so that a type in the file and a type in
/// the code read the same.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

impl GgmlType {
    /// Returns `None` for a discriminant this build does not know, so a file
    /// written by a newer converter fails with a clear error instead of being
    /// read with a wrong block size.
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            30 => BF16,
            _ => return None,
        })
    }

    /// Number of values in one block of this type.
    pub fn block_len(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => QK_K,
        }
    }

    /// Number of bytes one block of this type occupies on disk.
    pub fn block_bytes(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 | BF16 => 2,
            Q4_0 => 2 + 16,
            Q4_1 => 2 + 2 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 2 + 2 + 4 + 16,
            Q8_0 => Q8_0_BYTES,
            Q8_1 => 2 + 2 + 32,
            Q2_K => 16 + 64 + 2 + 2,
            Q3_K => 32 + 64 + 12 + 2,
            Q4_K => Q4_K_BYTES,
            Q5_K => 2 + 2 + 12 + 32 + 128,
            Q6_K => Q6_K_BYTES,
            Q8_K => 4 + QK_K + 2 * (QK_K / 16),
        }
    }

    /// Bytes needed for `n` values, rounding up to whole blocks, or `None` if
    /// that does not fit in a `u64`.
    ///
    /// Checked rather than wrapping because the loader calls this with a
    /// dimension product straight out of a file header: on a corrupt one, a
    /// wrapped length would pass the bounds check and open as a short tensor
    /// instead of failing by name.
    pub fn bytes_for(self, n: u64) -> Option<u64> {
        n.div_ceil(self.block_len() as u64)
            .checked_mul(self.block_bytes() as u64)
    }
}

/// Reads a little-endian `f16` at `off`.
#[inline]
pub(crate) fn read_f16(b: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[off], b[off + 1]])).to_f32()
}

/// `true` where [`dequantize_row`] knows the layout of `ty`.
///
/// The model loader calls this once, so the forward pass can dequantize
/// without a type check it cannot act on.
pub fn can_dequantize(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(ty, F32 | F16 | BF16 | Q4_K | Q6_K)
}

/// Expands one weight row into `out`.
///
/// This exists for the embedding lookup, which is the single place the product
/// path needs a weight as f32: the row it wants is selected by a token id, not
/// dotted with anything. Everything else meets the quantized bytes with int8
/// activations and never materialises an f32 weight.
///
/// Panics on a type it does not know, which the loader has already rejected
/// through [`can_dequantize`].
pub fn dequantize_row(ty: GgmlType, w: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, b) in out.iter_mut().zip(w.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        GgmlType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = read_f16(w, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (o, b) in out.iter_mut().zip(w.as_chunks::<2>().0) {
                *o = f32::from_bits((u16::from_le_bytes(*b) as u32) << 16);
            }
        }
        GgmlType::Q4_K => {
            for (ib, chunk) in out.chunks_mut(QK_K).enumerate() {
                let b = &w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES];
                let d = read_f16(b, 0);
                let dmin = read_f16(b, 2);
                let (scales, qs) = (&b[4..16], &b[16..]);
                for (j, sub) in chunk.chunks_mut(32).enumerate() {
                    let (sc, m) = q4k_scale_min(j, scales);
                    // Sub-block 2t is the low nibbles of qs[32t..32t+32] and
                    // 2t+1 the high nibbles of the same bytes.
                    let qw = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                    let shift = if j % 2 == 0 { 0 } else { 4 };
                    for (o, &q) in sub.iter_mut().zip(qw) {
                        *o = d * sc as f32 * ((q >> shift) & 0xF) as f32 - dmin * m as f32;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            for (ib, chunk) in out.chunks_mut(QK_K).enumerate() {
                let b = &w[ib * Q6_K_BYTES..(ib + 1) * Q6_K_BYTES];
                let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                let d = read_f16(b, 208);
                // Two halves of 128 values, each with 64 low-nibble bytes, 32
                // high-bit bytes and 8 scales.
                for h in 0..2 {
                    let (ql_h, qh_h) = (&ql[h * 64..], &qh[h * 32..]);
                    for g in 0..4 {
                        for l in 0..32 {
                            let low = match g {
                                0 => ql_h[l] & 0xF,
                                1 => ql_h[32 + l] & 0xF,
                                2 => ql_h[l] >> 4,
                                _ => ql_h[32 + l] >> 4,
                            };
                            let q = (low | (((qh_h[l] >> (2 * g)) & 3) << 4)) as i32 - 32;
                            let s = sc[h * 8 + 2 * g + l / 16] as i8 as i32;
                            let at = h * 128 + g * 32 + l;
                            if at < chunk.len() {
                                chunk[at] = d * (s * q) as f32;
                            }
                        }
                    }
                }
            }
        }
        other => panic!("dequantize_row does not know {other:?}"),
    }
}

/// Unpacks the `j`-th 6-bit scale and min out of a `Q4_K` 12-byte scale block.
///
/// This is the `get_scale_min_k4` packing from ggml: the first four pairs are
/// stored plainly in the low 6 bits of bytes 0..8, the last four steal their
/// high two bits from the top of bytes 0..8.
#[inline]
pub(crate) fn q4k_scale_min(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

/// One super-block of activations quantized to int8.
///
/// `bsums` holds the sum of every group of 16 quants. The k-quant kernels need
/// them for the offset terms: `Q4_K` subtracts a per-sub-block minimum and
/// `Q6_K` carries a constant -32 bias, and both collapse into a dot product
/// with these sums instead of a second pass over the activations.
#[derive(Clone)]
pub struct ActBlock {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for ActBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// One row of int8 activations as a kernel sees it.
///
/// A kernel never needs to know whether its row came from a single token or
/// out of a batch, which is why this is what [`crate::kernels::VecDot`] takes
/// and not the scratch buffer below.
#[derive(Clone, Copy)]
pub struct ActRow<'a> {
    pub blocks: &'a [ActBlock],
    /// The `K` these blocks were quantized from. The tail of the last block is
    /// zeroed, so this is needed only to count the blocks of a weight type
    /// whose block length is shorter than [`QK_K`].
    pub k: usize,
}

/// Reusable int8 activation scratch, for one row or for a batch of them.
///
/// Allocated once from the largest `K` in the model and the largest batch, and
/// refilled per matmul, so the forward pass never allocates. The tail of the
/// last super-block of every row is zeroed rather than special-cased: a zero
/// activation kills whatever the weight row holds past `K`, which is what makes
/// a `K` that is not a multiple of the block length fall out for free in every
/// kernel.
///
/// A batch is `t` rows of the same `K`, each quantized exactly as it would be
/// on its own: the scale is per row and per super-block, so no row's quants
/// depend on another's. That is what makes a batched matmul produce the same
/// bits as `t` single ones.
pub struct Activations {
    blocks: Vec<ActBlock>,
    /// Blocks per row, from `max_k`; row `t` starts at `t * stride`.
    stride: usize,
    max_t: usize,
    k: usize,
    t: usize,
}

impl Activations {
    /// Allocates scratch for one row of up to `max_k` activations.
    pub fn new(max_k: usize) -> Self {
        Self::with_batch(max_k, 1)
    }

    /// Allocates scratch for up to `max_t` rows of up to `max_k` activations.
    pub fn with_batch(max_k: usize, max_t: usize) -> Self {
        let stride = max_k.div_ceil(QK_K);
        let max_t = max_t.max(1);
        Self {
            blocks: vec![ActBlock::default(); stride * max_t],
            stride,
            max_t,
            k: 0,
            t: 1,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Rows in the batch currently quantized here.
    pub fn t(&self) -> usize {
        self.t
    }

    pub fn max_t(&self) -> usize {
        self.max_t
    }

    /// The first row, which is the only one a single-token pass fills.
    pub fn blocks(&self) -> &[ActBlock] {
        self.row(0).blocks
    }

    pub fn row(&self, t: usize) -> ActRow<'_> {
        debug_assert!(t < self.t);
        ActRow {
            blocks: &self.blocks[t * self.stride..][..self.k.div_ceil(QK_K)],
            k: self.k,
        }
    }

    /// Quantizes one row. Panics if `x` is longer than the scratch was sized
    /// for, because growing it would allocate in the hot loop.
    pub fn quantize(&mut self, x: &[f32]) {
        self.quantize_batch(x, x.len());
    }

    /// Quantizes `x.len() / k` rows of `k` activations each.
    pub fn quantize_batch(&mut self, x: &[f32], k: usize) {
        let nb = k.div_ceil(QK_K);
        let t = x.len().checked_div(k).unwrap_or(1);
        assert_eq!(
            t * k,
            x.len(),
            "{} values are not whole rows of {k}",
            x.len()
        );
        assert!(
            nb <= self.stride,
            "activation scratch holds rows of {} values, got {k}",
            self.stride * QK_K,
        );
        assert!(
            t <= self.max_t,
            "activation scratch holds {} rows, got {t}",
            self.max_t
        );
        self.k = k;
        self.t = t;
        let stride = self.stride;
        for (row, src) in self.blocks.chunks_mut(stride).zip(x.chunks(k)) {
            for (b, chunk) in row[..nb].iter_mut().zip(src.chunks(QK_K)) {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                b.d = amax / 127.0;
                let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                b.qs.fill(0);
                for (q, v) in b.qs.iter_mut().zip(chunk) {
                    *q = (v * id).round().clamp(-127.0, 127.0) as i8;
                }
                for (i, s) in b.bsums.iter_mut().enumerate() {
                    *s = b.qs[i * 16..i * 16 + 16].iter().map(|&q| q as i16).sum();
                }
            }
        }
    }
}
