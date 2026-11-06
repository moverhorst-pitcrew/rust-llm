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

/// Tokens of a batch whose i32 accumulators stay live across the sub-block loop.
///
/// This is what turns the eight weight loads per (super-block, token) into eight
/// per (super-block, sub-group): with the sub-block loop outermost, its 32
/// weight bytes serve every token of the sub-group from one register. The bound
/// is the register file. An accumulator per token plus the weight vector, the
/// scale vector and one product in flight is `N + 3` of the sixteen `ymm`
/// registers on the AVX2 tier and `N + 6` on the VNNI one, where a `Q4_K`
/// sub-block pair has two weight vectors and two `VPDPBUSD` results live at
/// once. Measured on this box, see `docs/benchmarks.md`.
pub(super) const SUB_GROUP: usize = 8;

/// Runs a sub-group body over the tokens of `out`, [`SUB_GROUP`] at a time, with
/// a tail shorter than one sub-group done a token at a time.
///
/// A macro and not a function taking a closure because the body is generic over
/// a `const N`, which is the whole point: `N` accumulators can only live in
/// registers if `N` is a constant, so every sub-group size is its own
/// instantiation and a closure, having one type, cannot be both.
///
/// The super-block loop stays outside this, so each output still takes exactly
/// one f32 accumulation per super-block in super-block order however the tokens
/// are grouped.
macro_rules! in_sub_groups {
    ($a:expr, $t0:expr, $ib:expr, $out:expr, |$abs:ident, $o:ident| $body:expr) => {{
        let (a, t0, ib, out) = ($a, $t0, $ib, $out);
        let mut j = 0;
        while j + SUB_GROUP <= out.len() {
            let $abs = act_blocks::<SUB_GROUP>(a, t0 + j, ib);
            let $o = &mut out[j..j + SUB_GROUP];
            $body;
            j += SUB_GROUP;
        }
        while j < out.len() {
            let $abs = act_blocks::<1>(a, t0 + j, ib);
            let $o = &mut out[j..j + 1];
            $body;
            j += 1;
        }
    }};
}

/// Super-block `ib` of each of the `N` activation rows from `t0`.
#[inline]
pub(super) fn act_blocks<const N: usize>(a: &Activations, t0: usize, ib: usize) -> [&ActBlock; N] {
    std::array::from_fn(|j| &a.row(t0 + j).blocks[ib])
}

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

    let (mut scs, mut mins) = ([0u8; 8], [0u8; 8]);
    for j in 0..8 {
        (scs[j], mins[j]) = q4k_scale_min(j, scales);
    }

    let (isum, imin) = unsafe {
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
                let av = _mm256_loadu_si256(aqs.add(64 * t + 32 * half).cast());
                // Weights are genuinely unsigned 4-bit, so VPMADDUBSW needs no
                // sign juggling. Folding the 6-bit scale into the i16 products
                // keeps one i32 accumulator for the whole super-block.
                let p16 = _mm256_maddubs_epi16(qw, av);
                let sc = scs[2 * t + half] as i16;
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(sc)));
            }
        }
        (hsum_i32(acc), q4k_imin(q4k_mins_vec(&mins), &ab.bsums))
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
/// bit; what the batch saves is [`q4k_unpack`] per token and, through
/// [`sub_groups`], the weight loads of all but one token of a sub-group.
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
        let mins_vec = q4k_mins_vec(&mins);
        in_sub_groups!(a, t0, ib, out, |abs, o| q4k_sub_group(
            &qw, &scs, mins_vec, d, dmin, abs, o
        ));
    }
}

/// One sub-group of tokens against one unpacked `Q4_K` super-block.
///
/// The sub-block loop is the outer one here, so the sub-block's 32 weight bytes
/// are loaded once for the whole sub-group instead of once per token; what that
/// costs is `N` live i32 accumulators next to the weight and scale vectors,
/// which is what bounds `N`. Nothing about the arithmetic per output changes: a
/// token's accumulator still collects the same eight products in sub-block
/// order, and the one f32 line still runs once per super-block.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_sub_group<const N: usize>(
    qw: &[u8; QK_K],
    scs: &[u8; 8],
    mins_vec: __m256i,
    d: f32,
    dmin: f32,
    abs: [&ActBlock; N],
    out: &mut [f32],
) {
    unsafe {
        let mut acc = [_mm256_setzero_si256(); N];
        for (sb, &sc) in scs.iter().enumerate() {
            let wv = _mm256_loadu_si256(qw.as_ptr().add(32 * sb).cast());
            let scv = _mm256_set1_epi16(sc as i16);
            for (acc, ab) in acc.iter_mut().zip(&abs) {
                let av = _mm256_loadu_si256(ab.qs.as_ptr().add(32 * sb).cast());
                let p16 = _mm256_maddubs_epi16(wv, av);
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(p16, scv));
            }
        }
        q4k_finish(out, &abs, acc, mins_vec, d, dmin);
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
        let scales_vec = q6k_scales_vec(sc);
        // i16 lanes 0..8 cover bytes 0..16 and carry the even scale, lanes 8..16
        // cover bytes 16..32 and carry the odd one. One per group of 32 weights,
        // built once per super-block rather than once per (group, token).
        let scv: [__m256i; 8] = std::array::from_fn(|g| {
            let s0 = sc[2 * g] as i8 as i16;
            let s1 = sc[2 * g + 1] as i8 as i16;
            _mm256_set_epi16(
                s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
            )
        });
        in_sub_groups!(a, t0, ib, out, |abs, o| q6k_sub_group(
            &qw, &scv, scales_vec, d, abs, o
        ));
    }
}

/// One sub-group of tokens against one unpacked `Q6_K` super-block, as above.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_sub_group<const N: usize>(
    qw: &[u8; QK_K],
    scv: &[__m256i; 8],
    scales_vec: __m256i,
    d: f32,
    abs: [&ActBlock; N],
    out: &mut [f32],
) {
    unsafe {
        let mut acc = [_mm256_setzero_si256(); N];
        for (g, &scv) in scv.iter().enumerate() {
            let wv = _mm256_loadu_si256(qw.as_ptr().add(32 * g).cast());
            for (acc, ab) in acc.iter_mut().zip(&abs) {
                let av = _mm256_loadu_si256(ab.qs.as_ptr().add(32 * g).cast());
                let p16 = _mm256_maddubs_epi16(wv, av);
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(p16, scv));
            }
        }
        q6k_finish(out, &abs, acc, scales_vec, d);
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

    let bias = unsafe { q6k_bias(q6k_scales_vec(sc), &ab.bsums) };
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

/// The eight `Q4_K` sub-block minima as sixteen i16, each one twice.
///
/// Lane `2j` and `2j+1` both hold `mins[j]`, which is the layout
/// [`q4k_imin`] needs. It depends only on the weight block, so a batch builds
/// it once per super-block instead of once per token.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4k_mins_vec(mins: &[u8; 8]) -> __m256i {
    unsafe {
        let v = _mm_loadl_epi64(mins.as_ptr().cast());
        _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v, v))
    }
}

/// The `Q4_K` minimum term before its horizontal sum.
///
/// One `VPMADDWD` against the activation group sums: lane `j` of its result is
/// `mins[j] * bsums[2j] + mins[j] * bsums[2j+1]`, so the eight lanes sum to
/// what the scalar loop accumulated one sub-block at a time. The products are at
/// most `63 * 16 * 127` and there are two per lane, well inside i32, and integer
/// addition is associative, so this is the same integer rather than a close one.
///
/// Left unreduced so a sub-group can fold several tokens' terms down together.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4k_imin_vec(mins_vec: __m256i, bsums: &[i16; QK_K / 16]) -> __m256i {
    unsafe { _mm256_madd_epi16(_mm256_loadu_si256(bsums.as_ptr().cast()), mins_vec) }
}

/// `sum_j mins[j] * (bsums[2j] + bsums[2j+1])`, for the single-row kernels.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4k_imin(mins_vec: __m256i, bsums: &[i16; QK_K / 16]) -> i32 {
    unsafe { hsum_i32(q4k_imin_vec(mins_vec, bsums)) }
}

/// The sixteen `Q6_K` sub-block scales sign-extended to i16, for [`q6k_bias`].
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q6k_scales_vec(sc: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepi8_epi16(_mm_loadu_si128(sc.as_ptr().cast())) }
}

/// The term the constant -32 of `Q6_K` collapses into, before its horizontal
/// sum.
///
/// Same `VPMADDWD` as [`q4k_imin_vec`], with each scale meeting one group sum
/// instead of two. Exact for the same reason.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q6k_bias_vec(scales_vec: __m256i, bsums: &[i16; QK_K / 16]) -> __m256i {
    unsafe { _mm256_madd_epi16(_mm256_loadu_si256(bsums.as_ptr().cast()), scales_vec) }
}

/// `sum_i sc[i] * bsums[i]`, for the single-row kernels.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q6k_bias(scales_vec: __m256i, bsums: &[i16; QK_K / 16]) -> i32 {
    unsafe { hsum_i32(q6k_bias_vec(scales_vec, bsums)) }
}

/// The epilogue of a `Q4_K` sub-group: from the accumulators to the store into
/// `out`, four tokens at a time and without leaving the vector registers.
///
/// **Bit-identical, not equivalent.** Every lane performs exactly the
/// operations the scalar line performs, in the same order: the two integer
/// terms, convert, `d * isum`, `dmin * imin`, subtract, multiply by the
/// activation scale, add into the output. SSE single-precision multiply,
/// subtract and add are the same IEEE 754 operations on the same bits as Rust's
/// on `f32`, nothing is reassociated, and neither side contracts into an FMA
/// (Rust forms none without fast-math, and these are explicit instructions).
/// The integer reductions above them are integer, where reassociation is exact.
///
/// The outputs of a sub-group are adjacent in `out`, which is what lets four of
/// them be read, accumulated and written as one vector. A tail shorter than four
/// takes the scalar line, which is the `N = 1` instantiation on the product
/// path.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn q4k_finish<const N: usize>(
    out: &mut [f32],
    abs: &[&ActBlock; N],
    acc: [__m256i; N],
    mins_vec: __m256i,
    d: f32,
    dmin: f32,
) {
    debug_assert_eq!(out.len(), N);
    unsafe {
        let mut i = 0;
        while i + 4 <= N {
            let isum = hsum_i32_x4([acc[i], acc[i + 1], acc[i + 2], acc[i + 3]]);
            let imin = hsum_i32_x4(std::array::from_fn(|j| {
                q4k_imin_vec(mins_vec, &abs[i + j].bsums)
            }));
            let scaled = _mm_sub_ps(
                _mm_mul_ps(_mm_set1_ps(d), _mm_cvtepi32_ps(isum)),
                _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(imin)),
            );
            add_out_x4(&mut out[i..], _mm_mul_ps(act_d_x4(&abs[i..]), scaled));
            i += 4;
        }
        while i < N {
            let isum = hsum_i32(acc[i]);
            let imin = q4k_imin(mins_vec, &abs[i].bsums);
            out[i] += abs[i].d * (d * isum as f32 - dmin * imin as f32);
            i += 1;
        }
    }
}

/// The same for `Q6_K`, where the `- 32 * bias` is integer and stays integer and
/// the f32 line is `abs[i].d * d * (...)`, in that order.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn q6k_finish<const N: usize>(
    out: &mut [f32],
    abs: &[&ActBlock; N],
    acc: [__m256i; N],
    scales_vec: __m256i,
    d: f32,
) {
    debug_assert_eq!(out.len(), N);
    unsafe {
        let mut i = 0;
        while i + 4 <= N {
            let isum = hsum_i32_x4([acc[i], acc[i + 1], acc[i + 2], acc[i + 3]]);
            let bias = hsum_i32_x4(std::array::from_fn(|j| {
                q6k_bias_vec(scales_vec, &abs[i + j].bsums)
            }));
            let biased = _mm_sub_epi32(isum, _mm_mullo_epi32(bias, _mm_set1_epi32(32)));
            let scale = _mm_mul_ps(act_d_x4(&abs[i..]), _mm_set1_ps(d));
            add_out_x4(&mut out[i..], _mm_mul_ps(scale, _mm_cvtepi32_ps(biased)));
            i += 4;
        }
        while i < N {
            let isum = hsum_i32(acc[i]);
            let bias = q6k_bias(scales_vec, &abs[i].bsums);
            out[i] += abs[i].d * d * (isum - 32 * bias) as f32;
            i += 1;
        }
    }
}

/// The horizontal sums of four accumulators, as one `__m128i`.
///
/// `VPHADDD` folds within each 128-bit half, so after three of them lane `i` of
/// each half holds that half's share of accumulator `i` and one add of the two
/// halves finishes all four. Integer, so which pairs are summed first does not
/// change any of the four results.
///
/// It returns a register rather than `[i32; 4]` on purpose: the sub-group
/// epilogues carry the four sums straight into the f32 line, and an array here
/// puts a store and a reload between the two for no reason.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn hsum_i32_x4(v: [__m256i; 4]) -> __m128i {
    let lo = _mm256_hadd_epi32(v[0], v[1]);
    let hi = _mm256_hadd_epi32(v[2], v[3]);
    let q = _mm256_hadd_epi32(lo, hi);
    _mm_add_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q))
}

/// The four `d` scales of a sub-group's activation blocks, in lane order.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn act_d_x4(abs: &[&ActBlock]) -> __m128 {
    _mm_set_ps(abs[3].d, abs[2].d, abs[1].d, abs[0].d)
}

/// `out[0..4] += v`, the sub-group's four adjacent outputs as one vector.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn add_out_x4(out: &mut [f32], v: __m128) {
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), _mm_add_ps(_mm_loadu_ps(out.as_ptr()), v)) }
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
