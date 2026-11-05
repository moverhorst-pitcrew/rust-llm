//! AVX-VNNI kernels for the two k-quants.
//!
//! `VPDPBUSD` wants an unsigned left operand, which is exactly how `Q4_K` and
//! `Q6_K` store their quants (4-bit and 6-bit unsigned, with the bias applied
//! afterwards), so both map onto it without any sign correction.
//!
//! Same safety argument as the AVX2 module: only [`super::Kernels`] takes the
//! address of these wrappers, and only after `avxvnni` was detected.

use std::arch::x86_64::*;

use super::avx2::hsum_i32;
use crate::quant::{ActBlock, ActRow, Q4_K_BYTES, Q6_K_BYTES, q4k_scale_min, read_f16};

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

    let mut imin = 0i32;
    let isum = unsafe {
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
            let (sc0, m0) = q4k_scale_min(2 * t, scales);
            let (sc1, m1) = q4k_scale_min(2 * t + 1, scales);
            for (j, m) in [(2 * t, m0), (2 * t + 1, m1)] {
                imin += m as i32 * (ab.bsums[j * 2] as i32 + ab.bsums[j * 2 + 1] as i32);
            }
            let dp0 = _mm256_dpbusd_avx_epi32(zero, lo, _mm256_loadu_si256(aqs.add(64 * t).cast()));
            let dp1 =
                _mm256_dpbusd_avx_epi32(zero, hi, _mm256_loadu_si256(aqs.add(64 * t + 32).cast()));
            // Each dp lane is at most 4 * 15 * 127, so both fit in i16 and the
            // pair can be packed into one register and scaled with VPMADDWD,
            // which is a cheaper way to apply the 6-bit scales than
            // VPMULLD per sub-block. VPACKSSDW interleaves per 128-bit half:
            // [dp0[0..4], dp1[0..4] | dp0[4..8], dp1[4..8]].
            let packed = _mm256_packs_epi32(dp0, dp1);
            let (s0, s1) = (sc0 as i16, sc1 as i16);
            let scv = _mm256_set_epi16(
                s1, s1, s1, s1, s0, s0, s0, s0, s1, s1, s1, s1, s0, s0, s0, s0,
            );
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(packed, scv));
        }
        hsum_i32(acc)
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

    let mut bias = 0i32;
    for (i, s) in sc.iter().enumerate() {
        bias += *s as i8 as i32 * ab.bsums[i] as i32;
    }
    ab.d * d * (isum - 32 * bias) as f32
}
