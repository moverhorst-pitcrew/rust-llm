//! ggml tensor types and their on-disk block layouts.

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

    /// Bytes needed for `n` values, rounding up to whole blocks.
    pub fn bytes_for(self, n: u64) -> u64 {
        let bl = self.block_len() as u64;
        n.div_ceil(bl) * self.block_bytes() as u64
    }
}
