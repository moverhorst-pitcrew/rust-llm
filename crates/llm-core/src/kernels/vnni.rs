//! AVX-VNNI kernels for the two k-quants.
//!
//! `VPDPBUSD` wants an unsigned left operand, which is exactly how `Q4_K` and
//! `Q6_K` store their quants (4-bit and 6-bit unsigned, with the bias applied
//! afterwards), so both map onto it without any sign correction.
//!
//! Same safety argument as the AVX2 module: only [`super::Kernels`] takes the
//! address of these wrappers, and only after `avxvnni` was detected.

use std::arch::x86_64::*;

use super::avx2::{
    hsum_i32, q4k_imin, q4k_mins_vec, q4k_unpack, q6k_bias, q6k_scales_vec, q6k_unpack,
};
use crate::quant::{
    ActBlock, ActRow, Activations, Q4_K_BYTES, Q6_K_BYTES, QK_K, q4k_scale_min, read_f16,
};

/// `out[j] += row . a.row(t0 + j)` for one `Q4_K` weight row against a group of
/// the activation batch.
///
/// Same arithmetic as [`q4k_dot`] in the same order, so the two agree bit for
/// bit. What the batch saves is the work that does not depend on the
/// activations: the nibble split, the 6-bit scale unpacking and building the
/// scale vector, once per super-block instead of once per token.
pub(super) fn q4k_dot_batch(w: &[u8], a: &Activations, t0: usize, out: &mut [f32]) {
    for ib in 0..a.k().div_ceil(QK_K) {
        unsafe { q4k_block_batch(&w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES], a, t0, ib, out) };
    }
}

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q4k_block_batch(b: &[u8], a: &Activations, t0: usize, ib: usize, out: &mut [f32]) {
    let d = read_f16(b, 0);
    let dmin = read_f16(b, 2);
    let mut qw = [0u8; QK_K];
    let mut scs = [0u8; 8];
    let mut mins = [0u8; 8];
    unsafe {
        q4k_unpack(b, &mut qw, &mut scs, &mut mins);
        let zero = _mm256_setzero_si256();
        // One vector per sub-block pair, in the lane order VPACKSSDW leaves
        // the two VPDPBUSD results in.
        let mut scv = [zero; 4];
        for (t, v) in scv.iter_mut().enumerate() {
            let (s0, s1) = (scs[2 * t] as i16, scs[2 * t + 1] as i16);
            *v = _mm256_set_epi16(
                s1, s1, s1, s1, s0, s0, s0, s0, s1, s1, s1, s1, s0, s0, s0, s0,
            );
        }
        let mins_vec = q4k_mins_vec(&mins);
        for (j, o) in out.iter_mut().enumerate() {
            let ab = &a.row(t0 + j).blocks[ib];
            let aqs = ab.qs.as_ptr();
            let mut acc = zero;
            for (t, &scv) in scv.iter().enumerate() {
                let dp0 = _mm256_dpbusd_avx_epi32(
                    zero,
                    _mm256_loadu_si256(qw.as_ptr().add(64 * t).cast()),
                    _mm256_loadu_si256(aqs.add(64 * t).cast()),
                );
                let dp1 = _mm256_dpbusd_avx_epi32(
                    zero,
                    _mm256_loadu_si256(qw.as_ptr().add(64 * t + 32).cast()),
                    _mm256_loadu_si256(aqs.add(64 * t + 32).cast()),
                );
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_packs_epi32(dp0, dp1), scv));
            }
            let imin = q4k_imin(mins_vec, &ab.bsums);
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

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q6k_block_batch(b: &[u8], a: &Activations, t0: usize, ib: usize, out: &mut [f32]) {
    let sc = &b[192..208];
    let d = read_f16(b, 208);
    let mut qw = [0u8; QK_K];
    unsafe {
        q6k_unpack(b, &mut qw);
        let zero = _mm256_setzero_si256();
        // i32 lanes 0..4 cover bytes 0..16 and carry the even scale, lanes
        // 4..8 cover bytes 16..32 and carry the odd one.
        let mut scv = [zero; 8];
        for (g, v) in scv.iter_mut().enumerate() {
            let s0 = sc[2 * g] as i8 as i32;
            let s1 = sc[2 * g + 1] as i8 as i32;
            *v = _mm256_set_epi32(s1, s1, s1, s1, s0, s0, s0, s0);
        }
        let scales_vec = q6k_scales_vec(sc);
        for (j, o) in out.iter_mut().enumerate() {
            let ab = &a.row(t0 + j).blocks[ib];
            let aqs = ab.qs.as_ptr();
            let mut acc = zero;
            for (g, &scv) in scv.iter().enumerate() {
                let dp = _mm256_dpbusd_avx_epi32(
                    zero,
                    _mm256_loadu_si256(qw.as_ptr().add(32 * g).cast()),
                    _mm256_loadu_si256(aqs.add(32 * g).cast()),
                );
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(dp, scv));
            }
            let bias = q6k_bias(scales_vec, &ab.bsums);
            *o += ab.d * d * (hsum_i32(acc) - 32 * bias) as f32;
        }
    }
}

pub(super) fn q4k_dot(w: &[u8], a: ActRow<'_>) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks.iter().enumerate() {
        sum += unsafe { q4k_block(&w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES], ab) };
    }
    sum
}

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q4k_block(b: &[u8], ab: &ActBlock) -> f32 {
    let d = read_f16(b, 0);
    let dmin = read_f16(b, 2);
    let scales = &b[4..16];

    let (mut scs, mut mins) = ([0u8; 8], [0u8; 8]);
    for j in 0..8 {
        (scs[j], mins[j]) = q4k_scale_min(j, scales);
    }

    let (isum, imin) = unsafe {
        let qs = b[16..].as_ptr();
        let aqs = ab.qs.as_ptr();
        let low_nibble = _mm256_set1_epi8(0x0F);
        let zero = _mm256_setzero_si256();
        let mut acc = zero;
        for t in 0..4 {
            let qv = _mm256_loadu_si256(qs.add(32 * t).cast());
            // Sub-blocks 2t and 2t+1 are the low and high nibbles of the same
            // 32 bytes, so they are done as a pair.
            let lo = _mm256_and_si256(qv, low_nibble);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), low_nibble);
            let dp0 = _mm256_dpbusd_avx_epi32(zero, lo, _mm256_loadu_si256(aqs.add(64 * t).cast()));
            let dp1 =
                _mm256_dpbusd_avx_epi32(zero, hi, _mm256_loadu_si256(aqs.add(64 * t + 32).cast()));
            // Each dp lane is at most 4 * 15 * 127, so both fit in i16 and the
            // pair can be packed into one register and scaled with VPMADDWD,
            // which is a cheaper way to apply the 6-bit scales than
            // VPMULLD per sub-block. VPACKSSDW interleaves per 128-bit half:
            // [dp0[0..4], dp1[0..4] | dp0[4..8], dp1[4..8]].
            let packed = _mm256_packs_epi32(dp0, dp1);
            let (s0, s1) = (scs[2 * t] as i16, scs[2 * t + 1] as i16);
            let scv = _mm256_set_epi16(
                s1, s1, s1, s1, s0, s0, s0, s0, s1, s1, s1, s1, s0, s0, s0, s0,
            );
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(packed, scv));
        }
        (hsum_i32(acc), q4k_imin(q4k_mins_vec(&mins), &ab.bsums))
    };
    ab.d * (d * isum as f32 - dmin * imin as f32)
}

pub(super) fn q6k_dot(w: &[u8], a: ActRow<'_>) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks.iter().enumerate() {
        sum += unsafe { q6k_block(&w[ib * Q6_K_BYTES..(ib + 1) * Q6_K_BYTES], ab) };
    }
    sum
}

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q6k_block(b: &[u8], ab: &ActBlock) -> f32 {
    let sc = &b[192..208];
    let d = read_f16(b, 208);

    let isum = unsafe {
        let ql = b.as_ptr();
        let qh = b[128..].as_ptr();
        let aqs = ab.qs.as_ptr();
        let low_nibble = _mm256_set1_epi8(0x0F);
        let two_bits = _mm256_set1_epi8(0x03);
        let zero = _mm256_setzero_si256();
        let mut acc = zero;
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
                let qw = _mm256_or_si256(nib, _mm256_slli_epi16::<4>(high));
                let av = _mm256_loadu_si256(aqs.add(128 * h + 32 * g).cast());
                // i32 lanes 0..4 cover bytes 0..16 and carry scale 2g, lanes
                // 4..8 cover bytes 16..32 and carry scale 2g+1.
                let dp = _mm256_dpbusd_avx_epi32(zero, qw, av);
                let s0 = sc[h * 8 + 2 * g] as i8 as i32;
                let s1 = sc[h * 8 + 2 * g + 1] as i8 as i32;
                let scv = _mm256_set_epi32(s1, s1, s1, s1, s0, s0, s0, s0);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(dp, scv));
            }
        }
        hsum_i32(acc)
    };

    let bias = unsafe { q6k_bias(q6k_scales_vec(sc), &ab.bsums) };
    ab.d * d * (isum - 32 * bias) as f32
}
