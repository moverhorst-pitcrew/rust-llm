//! AVX2 + FMA kernels.
//!
//! The safe wrappers may only be called on a CPU with `avx2` and `fma`. That
//! holds because [`super::Kernels`] installs pointers to them in its table
//! solely after `is_x86_feature_detected!` has confirmed both, and nothing
//! else takes their address.

use std::arch::x86_64::*;

use crate::quant::{
    ActBlock, ActRow, Activations, Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K, QK8_0, q4k_scale_min,
    read_f16,
};

pub(super) fn q4k_dot(w: &[u8], a: ActRow<'_>) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks.iter().enumerate() {
        sum += unsafe { q4k_block(&w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES], ab) };
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_block(b: &[u8], ab: &ActBlock) -> f32 {
    let d = read_f16(b, 0);
    let dmin = read_f16(b, 2);
    let scales = &b[4..16];

    let mut imin = 0i32;
    let isum = unsafe {
        let qs = b[16..].as_ptr();
        let aqs = ab.qs.as_ptr();
        let low_nibble = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();
        for t in 0..4 {
            let qv = _mm256_loadu_si256(qs.add(32 * t).cast());
            // Sub-block 2t is the low nibbles of these 32 bytes, 2t+1 the high.
            let halves = [
                _mm256_and_si256(qv, low_nibble),
                _mm256_and_si256(_mm256_srli_epi16::<4>(qv), low_nibble),
            ];
            for (half, qw) in halves.into_iter().enumerate() {
                let (sc, m) = q4k_scale_min(2 * t + half, scales);
                let j = 2 * t + half;
                imin += m as i32 * (ab.bsums[j * 2] as i32 + ab.bsums[j * 2 + 1] as i32);
                let av = _mm256_loadu_si256(aqs.add(64 * t + 32 * half).cast());
                // Weights are genuinely unsigned 4-bit, so VPMADDUBSW needs no
                // sign juggling. Folding the 6-bit scale into the i16 products
                // keeps one i32 accumulator for the whole super-block.
                let p16 = _mm256_maddubs_epi16(qw, av);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(sc as i16)));
            }
        }
        hsum_i32(acc)
    };
    ab.d * (d * isum as f32 - dmin * imin as f32)
}

/// Splits a `Q4_K` super-block into 256 unsigned bytes in activation order,
/// plus its eight 6-bit scales and minima.
///
/// This is the part of a dot product that does not depend on the activations,
/// so a batch does it once for the whole group instead of once per token: the
/// nibble split, and `get_scale_min_k4`'s bit shuffling. Sub-block `2t` is the
/// low nibbles of `qs[32t..32t+32]` and `2t+1` the high nibbles of the same
/// bytes, so one load serves two.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4k_unpack(
    b: &[u8],
    qw: &mut [u8; QK_K],
    scs: &mut [u8; 8],
    mins: &mut [u8; 8],
) {
    unsafe {
        let qs = b[16..].as_ptr();
        let scales = &b[4..16];
        let low_nibble = _mm256_set1_epi8(0x0F);
        for t in 0..4 {
            let qv = _mm256_loadu_si256(qs.add(32 * t).cast());
            _mm256_storeu_si256(
                qw.as_mut_ptr().add(64 * t).cast(),
                _mm256_and_si256(qv, low_nibble),
            );
            _mm256_storeu_si256(
                qw.as_mut_ptr().add(64 * t + 32).cast(),
                _mm256_and_si256(_mm256_srli_epi16::<4>(qv), low_nibble),
            );
            for j in [2 * t, 2 * t + 1] {
                let (sc, m) = q4k_scale_min(j, scales);
                scs[j] = sc;
                mins[j] = m;
            }
        }
    }
}

/// The same for `Q6_K`: 256 unsigned 6-bit weights in activation order, the
/// constant -32 bias still to come off through the activation group sums. The
/// scales need no unpacking, they are plain `i8` in the block.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q6k_unpack(b: &[u8], qw: &mut [u8; QK_K]) {
    unsafe {
        let ql = b.as_ptr();
        let qh = b[128..].as_ptr();
        let low_nibble = _mm256_set1_epi8(0x0F);
        let two_bits = _mm256_set1_epi8(0x03);
        for h in 0..2 {
            let qhv = _mm256_loadu_si256(qh.add(32 * h).cast());
            for g in 0..4 {
                let qlv = _mm256_loadu_si256(ql.add(64 * h + 32 * (g % 2)).cast());
                let nib = if g < 2 {
                    _mm256_and_si256(qlv, low_nibble)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16::<4>(qlv), low_nibble)
                };
                let shift = _mm_cvtsi32_si128(2 * g as i32);
                let high = _mm256_and_si256(_mm256_srl_epi16(qhv, shift), two_bits);
                _mm256_storeu_si256(
                    qw.as_mut_ptr().add(128 * h + 32 * g).cast(),
                    _mm256_or_si256(nib, _mm256_slli_epi16::<4>(high)),
                );
            }
        }
    }
}

/// `out[j] += row . a.row(t0 + j)` for one `Q4_K` weight row against a group of
/// the activation batch.
///
/// Same arithmetic as [`q4k_dot`] in the same order, so the two agree bit for
/// bit; what the batch saves is [`q4k_unpack`] per token.
pub(super) fn q4k_dot_batch(w: &[u8], a: &Activations, t0: usize, out: &mut [f32]) {
    for ib in 0..a.k().div_ceil(QK_K) {
        unsafe { q4k_block_batch(&w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES], a, t0, ib, out) };
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_block_batch(b: &[u8], a: &Activations, t0: usize, ib: usize, out: &mut [f32]) {
    let d = read_f16(b, 0);
    let dmin = read_f16(b, 2);
    let mut qw = [0u8; QK_K];
    let mut scs = [0u8; 8];
    let mut mins = [0u8; 8];
    unsafe {
        q4k_unpack(b, &mut qw, &mut scs, &mut mins);
        for (j, o) in out.iter_mut().enumerate() {
            let ab = &a.row(t0 + j).blocks[ib];
            let aqs = ab.qs.as_ptr();
            let mut acc = _mm256_setzero_si256();
            let mut imin = 0i32;
            for sb in 0..8 {
                let p16 = _mm256_maddubs_epi16(
                    _mm256_loadu_si256(qw.as_ptr().add(32 * sb).cast()),
                    _mm256_loadu_si256(aqs.add(32 * sb).cast()),
                );
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_madd_epi16(p16, _mm256_set1_epi16(scs[sb] as i16)),
                );
                imin += mins[sb] as i32 * (ab.bsums[sb * 2] as i32 + ab.bsums[sb * 2 + 1] as i32);
            }
            *o += ab.d * (d * hsum_i32(acc) as f32 - dmin * imin as f32);
        }
    }
}

/// `out[j] += row . a.row(t0 + j)` for one `Q6_K` weight row, as above.
pub(super) fn q6k_dot_batch(w: &[u8], a: &Activations, t0: usize, out: &mut [f32]) {
    for ib in 0..a.k().div_ceil(QK_K) {
        unsafe { q6k_block_batch(&w[ib * Q6_K_BYTES..(ib + 1) * Q6_K_BYTES], a, t0, ib, out) };
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_block_batch(b: &[u8], a: &Activations, t0: usize, ib: usize, out: &mut [f32]) {
    let sc = &b[192..208];
    let d = read_f16(b, 208);
    let mut qw = [0u8; QK_K];
    unsafe {
        q6k_unpack(b, &mut qw);
        for (j, o) in out.iter_mut().enumerate() {
            let ab = &a.row(t0 + j).blocks[ib];
            let aqs = ab.qs.as_ptr();
            let mut acc = _mm256_setzero_si256();
            for group in 0..8 {
                let p16 = _mm256_maddubs_epi16(
                    _mm256_loadu_si256(qw.as_ptr().add(32 * group).cast()),
                    _mm256_loadu_si256(aqs.add(32 * group).cast()),
                );
                // i16 lanes 0..8 cover bytes 0..16 and carry the even scale,
                // lanes 8..16 cover bytes 16..32 and carry the odd one.
                let s0 = sc[2 * group] as i8 as i16;
                let s1 = sc[2 * group + 1] as i8 as i16;
                let scv = _mm256_set_epi16(
                    s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
                );
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scv));
            }
            let mut bias = 0i32;
            for (i, s) in sc.iter().enumerate() {
                bias += *s as i8 as i32 * ab.bsums[i] as i32;
            }
            *o += ab.d * d * (hsum_i32(acc) - 32 * bias) as f32;
        }
    }
}

pub(super) fn q6k_dot(w: &[u8], a: ActRow<'_>) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks.iter().enumerate() {
        sum += unsafe { q6k_block(&w[ib * Q6_K_BYTES..(ib + 1) * Q6_K_BYTES], ab) };
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_block(b: &[u8], ab: &ActBlock) -> f32 {
    let sc = &b[192..208];
    let d = read_f16(b, 208);

    let isum = unsafe {
        let ql = b.as_ptr();
        let qh = b[128..].as_ptr();
        let aqs = ab.qs.as_ptr();
        let low_nibble = _mm256_set1_epi8(0x0F);
        let two_bits = _mm256_set1_epi8(0x03);
        let mut acc = _mm256_setzero_si256();
        for h in 0..2 {
            let qhv = _mm256_loadu_si256(qh.add(32 * h).cast());
            for g in 0..4 {
                let qlv = _mm256_loadu_si256(ql.add(64 * h + 32 * (g % 2)).cast());
                let nib = if g < 2 {
                    _mm256_and_si256(qlv, low_nibble)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16::<4>(qlv), low_nibble)
                };
                let shift = _mm_cvtsi32_si128(2 * g as i32);
                let high = _mm256_and_si256(_mm256_srl_epi16(qhv, shift), two_bits);
                // 6-bit unsigned weights; the constant -32 bias comes off below
                // through the activation group sums.
                let qw = _mm256_or_si256(nib, _mm256_slli_epi16::<4>(high));
                let av = _mm256_loadu_si256(aqs.add(128 * h + 32 * g).cast());
                let p16 = _mm256_maddubs_epi16(qw, av);
                // i16 lanes 0..8 cover bytes 0..16 and carry scale 2g, lanes
                // 8..16 cover bytes 16..32 and carry scale 2g+1.
                let s0 = sc[h * 8 + 2 * g] as i8 as i16;
                let s1 = sc[h * 8 + 2 * g + 1] as i8 as i16;
                let scv = _mm256_set_epi16(
                    s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
                );
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scv));
            }
        }
        hsum_i32(acc)
    };

    let mut bias = 0i32;
    for (i, s) in sc.iter().enumerate() {
        bias += *s as i8 as i32 * ab.bsums[i] as i32;
    }
    ab.d * d * (isum - 32 * bias) as f32
}

pub(super) fn q80_dot(w: &[u8], a: ActRow<'_>) -> f32 {
    unsafe { q80_row(w, a) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q80_row(w: &[u8], a: ActRow<'_>) -> f32 {
    unsafe {
        let blocks = a.blocks;
        let per_super = QK_K / QK8_0;
        let mut acc = _mm256_setzero_ps();
        for ib in 0..a.k.div_ceil(QK8_0) {
            let b = &w[ib * Q8_0_BYTES..(ib + 1) * Q8_0_BYTES];
            let ab = &blocks[ib / per_super];
            let qw = _mm256_loadu_si256(b[2..].as_ptr().cast());
            let av = _mm256_loadu_si256(ab.qs.as_ptr().add((ib % per_super) * QK8_0).cast());
            // VPMADDUBSW needs one unsigned operand and both of these are
            // signed, so take |w| and move w's sign onto the activation.
            // VPSIGNB leaves -128 at -128, which as an unsigned byte is 128,
            // which is exactly |-128|, so the weight side is right for the
            // whole i8 range. The activation side is safe because
            // `Activations::quantize` clamps to [-127, 127], so negating it
            // cannot overflow. 128 * 127 twice over stays inside i16, so
            // VPMADDUBSW's saturation never bites either.
            let p = _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(av, qw)),
                _mm256_set1_epi16(1),
            );
            let scale = _mm256_set1_ps(read_f16(b, 0) * ab.d);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), scale, acc);
        }
        hsum_f32(acc)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum_f32(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
    _mm_cvtss_f32(s)
}
