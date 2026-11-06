//! Kernel tests against a scalar reference that lives only here.
//!
//! The reference dequantizes a weight row to f32 straight from the ggml block
//! layout and dots it with the dequantized activations in f64. The kernels
//! compute the algebraically identical sum with the scales pulled out of the
//! integer accumulation, so the two agree to f32 rounding: the tolerance is
//! tight enough that a wrong shuffle, a swapped nibble or a sign error on a
//! scale moves the result far outside it.
//!
//! Weight bytes are random. For a quantized block every bit pattern is a legal
//! block (only the f16 scales are kept to sane magnitudes), so random bytes
//! cover the whole range of nibbles, high bits and signed sub-block scales
//! without needing a quantizer in the test.

use half::f16;
use llm_core::quant::{Activations, QK_K};
use llm_core::{GgmlType, KernelError, Kernels, Tier};

const TIERS: [Tier; 3] = [Tier::Scalar, Tier::Avx2Fma, Tier::Avx2Vnni];
const TYPES: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0];

/// Shapes as `(k, rows)`.
const AWKWARD_SHAPES: &[(usize, usize)] = &[
    (256, 1),   // single row
    (1, 1),     // single column, and a block that is almost all padding
    (32, 3),    // exactly one Q8_0 block, a fraction of a k-quant super-block
    (37, 3),    // k not a multiple of any block length
    (255, 2),   // one short of a super-block
    (257, 2),   // one past a super-block
    (300, 4),   // partial trailing super-block
    (512, 5),   // whole super-blocks, odd row count
    (8192, 2),  // large k, to exercise accumulation
    (16384, 1), // larger still
];

/// The matmul shapes the model actually uses, as `(k, rows)`.
const MODEL_SHAPES: &[(usize, usize)] = &[
    (2560, 8192), // attn_qkv, attn_q
    (2560, 9216), // ffn_gate, ffn_up
    (9216, 2560), // ffn_down
    (4096, 2560), // attn_output, ssm_out
    (2560, 1024), // attn_k, attn_v
];

// ------------------------------------------------------------------- random

/// xorshift64*, so a failure is reproducible from the seed in the message.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// `n` blocks of random but well-formed quantized weights.
fn random_weights(ty: GgmlType, rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut o = Vec::with_capacity(n * ty.block_bytes());
    for _ in 0..n {
        match ty {
            GgmlType::Q4_K => {
                o.extend(f16::from_f32(rng.signed() * 0.05).to_le_bytes());
                o.extend(f16::from_f32(rng.signed() * 0.02).to_le_bytes());
                o.extend((0..140).map(|_| rng.byte()));
            }
            GgmlType::Q6_K => {
                o.extend((0..208).map(|_| rng.byte()));
                o.extend(f16::from_f32(rng.signed() * 0.01).to_le_bytes());
            }
            GgmlType::Q8_0 => {
                o.extend(f16::from_f32(rng.signed() * 0.05).to_le_bytes());
                o.extend((0..32).map(|_| rng.byte()));
            }
            other => panic!("no fixture for {other:?}"),
        }
    }
    assert_eq!(o.len(), n * ty.block_bytes());
    o
}

// ---------------------------------------------------------------- reference

/// `get_scale_min_k4` from ggml, written out again here so the reference does
/// not share the kernels' unpacking.
fn q4k_scale_min(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

/// Dequantizes a whole row to f32, straight from the block layout.
fn dequantize(ty: GgmlType, row: &[u8], nblocks: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nblocks * ty.block_len());
    for ib in 0..nblocks {
        let b = &row[ib * ty.block_bytes()..(ib + 1) * ty.block_bytes()];
        match ty {
            GgmlType::Q4_K => {
                let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
                let (scales, qs) = (&b[4..16], &b[16..]);
                for j in 0..8 {
                    let (sc, m) = q4k_scale_min(j, scales);
                    let (ds, ms) = (d * sc as f32, dmin * m as f32);
                    for l in 0..32 {
                        let byte = qs[(j / 2) * 32 + l];
                        let q = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        out.push(ds * q as f32 - ms);
                    }
                }
            }
            GgmlType::Q6_K => {
                let d = f16_at(b, 208);
                for n in 0..2 {
                    let (ql, qh, sc) = (&b[n * 64..], &b[128 + n * 32..], &b[192 + n * 8..]);
                    let mut half = [0.0f32; 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[32 + l] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[32 + l] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        half[l] = d * sc[is] as i8 as f32 * q1 as f32;
                        half[32 + l] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        half[64 + l] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        half[96 + l] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                    out.extend(half);
                }
            }
            GgmlType::Q8_0 => {
                let d = f16_at(b, 0);
                out.extend(b[2..].iter().map(|&q| d * q as i8 as f32));
            }
            other => panic!("no reference for {other:?}"),
        }
    }
    out
}

/// The activations as the kernels see them: quant times block scale.
fn dequantize_activations(a: &Activations) -> Vec<f32> {
    a.blocks()
        .iter()
        .flat_map(|b| b.qs.iter().map(move |&q| b.d * q as f32))
        .take(a.k())
        .collect()
}

/// Returns the reference dot product and the sum of the absolute terms, which
/// bounds how much f32 accumulation can legitimately drift.
fn reference_dot(ty: GgmlType, row: &[u8], a: &Activations) -> (f64, f64) {
    let w = dequantize(ty, row, a.k().div_ceil(ty.block_len()));
    let x = dequantize_activations(a);
    let mut dot = 0.0f64;
    let mut magnitude = 0.0f64;
    for (wi, xi) in w.iter().zip(&x) {
        let t = *wi as f64 * *xi as f64;
        dot += t;
        magnitude += t.abs();
    }
    (dot, magnitude)
}

// -------------------------------------------------------------------- checks

fn available_tiers() -> Vec<(Tier, Kernels)> {
    TIERS
        .iter()
        .filter_map(|&t| Kernels::for_tier(t).map(|k| (t, k)))
        .collect()
}

fn check_shape(ty: GgmlType, k: usize, rows: usize, seed: u64) {
    let nb = k.div_ceil(ty.block_len());
    let stride = ty.bytes_for(k as u64).expect("fits") as usize;
    let mut rng = Rng::new(seed);
    let w = random_weights(ty, &mut rng, nb * rows);
    let x: Vec<f32> = (0..k).map(|_| rng.signed() * 3.0).collect();
    let mut a = Activations::new(k);
    a.quantize(&x);

    let reference: Vec<(f64, f64)> = (0..rows)
        .map(|r| reference_dot(ty, &w[r * stride..(r + 1) * stride], &a))
        .collect();

    for (tier, kernels) in available_tiers() {
        let mut out = vec![f32::NAN; rows];
        kernels.matvec(ty, &w, k, &a, &mut out).expect("kernel");
        for (r, (&got, &(want, magnitude))) in out.iter().zip(&reference).enumerate() {
            let tol = (magnitude * 1e-5).max(1e-12);
            assert!(
                (got as f64 - want).abs() <= tol,
                "{ty:?} {tier:?} k={k} rows={rows} seed={seed} row {r}: \
                 got {got}, want {want}, tolerance {tol}"
            );
        }
    }
}

#[test]
fn awkward_shapes_match_the_reference() {
    for &ty in &TYPES {
        for (i, &(k, rows)) in AWKWARD_SHAPES.iter().enumerate() {
            check_shape(ty, k, rows, 0x51ed + i as u64);
        }
    }
}

#[test]
fn model_shapes_match_the_reference() {
    // Full row counts, every type, every tier. Thirteen seconds in a debug
    // build, one in release.
    for &ty in &TYPES {
        for (i, &(k, rows)) in MODEL_SHAPES.iter().enumerate() {
            check_shape(ty, k, rows, 0x5_1a9e + i as u64);
        }
    }
}

#[test]
fn zero_activations_and_zero_weights_give_zero() {
    for &ty in &TYPES {
        let k = 512;
        let nb = k / ty.block_len();
        let mut a = Activations::new(k);
        a.quantize(&vec![0.0; k]);
        assert_eq!(a.blocks().iter().map(|b| b.d).sum::<f32>(), 0.0);

        let mut rng = Rng::new(7);
        let w = random_weights(ty, &mut rng, nb);
        for (tier, kernels) in available_tiers() {
            let mut out = [f32::NAN];
            kernels.matvec(ty, &w, k, &a, &mut out).expect("kernel");
            assert_eq!(out[0], 0.0, "{ty:?} {tier:?} with zero activations");
        }

        let x: Vec<f32> = (0..k).map(|_| rng.signed()).collect();
        a.quantize(&x);
        let zero = vec![0u8; nb * ty.block_bytes()];
        for (tier, kernels) in available_tiers() {
            let mut out = [f32::NAN];
            kernels.matvec(ty, &zero, k, &a, &mut out).expect("kernel");
            assert_eq!(out[0], 0.0, "{ty:?} {tier:?} with zero weights");
        }
    }
}

#[test]
fn weight_lanes_past_k_do_not_contribute() {
    // The activation scratch zeroes the tail of its last super-block, which is
    // what lets a k that is not a whole number of blocks work without the
    // kernels special-casing it. Growing k by five, with five zero
    // activations, therefore pulls in a whole extra weight block that must
    // contribute nothing: the two results have to be bit-identical, because
    // the activation blocks the first k use are untouched.
    for &ty in &TYPES {
        let short = QK_K;
        let long = QK_K + 5;
        let mut rng = Rng::new(0xbeef);
        let w = random_weights(ty, &mut rng, long.div_ceil(ty.block_len()));
        let mut x: Vec<f32> = (0..short).map(|_| rng.signed()).collect();

        let mut a = Activations::new(long);
        a.quantize(&x);
        let mut short_out = [f32::NAN];
        x.extend([0.0; 5]);
        let mut long_a = Activations::new(long);
        long_a.quantize(&x);
        let mut long_out = [f32::NAN];

        for (tier, kernels) in available_tiers() {
            let short_bytes = ty.bytes_for(short as u64).expect("fits") as usize;
            kernels
                .matvec(ty, &w[..short_bytes], short, &a, &mut short_out)
                .expect("kernel");
            kernels
                .matvec(ty, &w, long, &long_a, &mut long_out)
                .expect("kernel");
            assert_eq!(
                short_out[0], long_out[0],
                "{ty:?} {tier:?}: the weight block past k changed the result"
            );
        }
    }
}

/// The batched kernel has to agree with the single one **bit for bit**, not
/// just within a tolerance.
///
/// It is a restructuring, not a different algorithm: the same products are
/// accumulated in the same order, and all that moves is where the weight block
/// is unpacked. Checking it against the scalar reference with a tolerance would
/// pass a batched kernel that reorders the accumulation, and that would change
/// the runner's output when a prompt is read in batches, which is exactly what
/// the correctness gate forbids.
///
/// Batch sizes deliberately straddle [`Kernels::BATCH_GROUP`]: a batch longer
/// than one group is run as several, and a group that is not full is the
/// remainder of every batch that is not a multiple of it.
///
/// They also straddle the kernels' internal token sub-group, which is where the
/// accumulators that stay live across the sub-blocks are bounded by the register
/// file. That size is not public, so rather than couple the test to it every
/// count from 1 to 9 is here along with the ones around `BATCH_GROUP`: whichever
/// sub-group the kernels pick out of the register file, this list has a batch
/// one short of it, exactly it, and one past it, and so exercises the tail that
/// is shorter than a sub-group.
#[test]
fn the_batched_kernel_is_bit_identical_to_the_single_one() {
    let group = Kernels::BATCH_GROUP;
    let batches = [
        1usize,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        12,
        group - 1,
        group,
        group + 1,
        2 * group + 3,
    ];
    for &ty in &TYPES {
        for &(k, rows) in AWKWARD_SHAPES.iter().chain(&[(2560, 64), (9216, 48)]) {
            let nb = k.div_ceil(ty.block_len());
            let stride = ty.bytes_for(k as u64).expect("fits") as usize;
            let mut rng = Rng::new(0xba7c4 + k as u64);
            let w = random_weights(ty, &mut rng, nb * rows);
            for &t in &batches {
                // One buffer of `t` rows, and the same rows one at a time.
                let x: Vec<f32> = (0..t * k).map(|_| rng.signed() * 3.0).collect();
                let mut batch = Activations::with_batch(k, t);
                batch.quantize_batch(&x, k);
                let mut single = Activations::new(k);
                for (tier, kernels) in available_tiers() {
                    let batched = kernels.vec_dot_batch(ty);
                    for r in 0..rows {
                        let row = &w[r * stride..(r + 1) * stride];
                        let mut got = vec![0.0f32; t];
                        let mut from = 0;
                        while from < t {
                            let n = (t - from).min(group);
                            match batched {
                                Some(dot) => dot(row, &batch, from, &mut got[from..from + n]),
                                // No batched kernel on this tier: then the
                                // batch runs through the single one, which is
                                // trivially identical, and this loop still
                                // checks that the batch's quantization of row
                                // `t` is the single path's.
                                None => {
                                    for (j, o) in got[from..from + n].iter_mut().enumerate() {
                                        *o = kernels.vec_dot(ty).expect("kernel")(
                                            row,
                                            batch.row(from + j),
                                        );
                                    }
                                }
                            }
                            from += n;
                        }
                        for (j, &got) in got.iter().enumerate() {
                            single.quantize(&x[j * k..(j + 1) * k]);
                            let mut want = [f32::NAN];
                            kernels
                                .matvec(ty, row, k, &single, &mut want)
                                .expect("kernel");
                            assert_eq!(
                                got, want[0],
                                "{ty:?} {tier:?} k={k} t={t} row={r} batch row {j}: \
                                 batched {got} against single {}",
                                want[0]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The saturation extremes of the lane layout, which random bytes never reach.
///
/// Every integer lane in these kernels has a ceiling, and the margin to it is a
/// property of how many products the lane collects before the next step. On the
/// VNNI tier a `VPDPBUSD` lane holds at most `4 * 15 * 127 = 7620` for `Q4_K`,
/// which is why the pair can go through `VPACKSSDW` into i16 at all; on the AVX2
/// tier a `VPMADDUBSW` lane holds `2 * 15 * 127 = 3810`, and `2 * 63 * 127 =
/// 16002` for `Q6_K`. Both are inside 32767 with room to spare, and a kernel
/// that puts more products in a lane before the pack eats that room silently: no
/// panic, no NaN, just a wrong number on the inputs that reach the bound.
///
/// The weights here are therefore not random. Every nibble is `0xF`, every
/// sub-block scale is the largest value its field can hold, and the activations
/// are `+/-127` throughout, which is what `quantize` produces from a row of
/// `+/-amax`. Checked both ways: each tier's batched kernel against its own
/// single-row one with `==`, which is what a saturating tile breaks, and each
/// tier's single-row kernel against the f64 reference, which is what a
/// saturating single-row kernel breaks.
#[test]
fn the_saturation_extremes_match_the_single_kernel() {
    let group = Kernels::BATCH_GROUP;
    for &ty in &[GgmlType::Q4_K, GgmlType::Q6_K] {
        // Enough k that the accumulator sees many super-blocks, and a k that is
        // not a whole number of them, both at a row count over the row tile.
        for &(k, rows) in &[(2560usize, 5usize), (300, 3), (256, 1)] {
            let nb = k.div_ceil(ty.block_len());
            let stride = ty.bytes_for(k as u64).expect("fits") as usize;
            for (wi, w) in saturated_weights(ty, nb * rows).iter().enumerate() {
                for (xi, x) in saturated_activations(k).iter().enumerate() {
                    let t = x.len() / k;
                    let mut batch = Activations::with_batch(k, t);
                    batch.quantize_batch(x, k);
                    // Past `k` the scratch is zeroed on purpose, so only the
                    // real values are at the extreme.
                    assert!(
                        batch
                            .blocks()
                            .iter()
                            .flat_map(|b| b.qs.iter())
                            .take(k)
                            .all(|&q| q.abs() == 127),
                        "the activation pattern did not quantize to +/-127"
                    );
                    let mut single = Activations::new(k);
                    for (tier, kernels) in available_tiers() {
                        for r in 0..rows {
                            let row = &w[r * stride..(r + 1) * stride];
                            let mut got = vec![0.0f32; t];
                            let mut from = 0;
                            while from < t {
                                let n = (t - from).min(group);
                                match kernels.vec_dot_batch(ty) {
                                    Some(dot) => dot(row, &batch, from, &mut got[from..from + n]),
                                    None => {
                                        for (j, o) in got[from..from + n].iter_mut().enumerate() {
                                            *o = kernels.vec_dot(ty).expect("kernel")(
                                                row,
                                                batch.row(from + j),
                                            );
                                        }
                                    }
                                }
                                from += n;
                            }
                            for (j, &got) in got.iter().enumerate() {
                                let xr = &x[j * k..(j + 1) * k];
                                single.quantize(xr);
                                let mut want = [f32::NAN];
                                kernels
                                    .matvec(ty, row, k, &single, &mut want)
                                    .expect("kernel");
                                assert_eq!(
                                    got, want[0],
                                    "{ty:?} {tier:?} k={k} weights #{wi} activations #{xi} \
                                     row={r} batch row {j}: batched {got} against single {}",
                                    want[0]
                                );
                                // And the single-row kernel itself against the
                                // reference, so a lane that saturates in both
                                // is caught by something other than their
                                // agreement.
                                let (reference, magnitude) = reference_dot(ty, row, &single);
                                let tol = (magnitude * 1e-5).max(1e-12);
                                assert!(
                                    (want[0] as f64 - reference).abs() <= tol,
                                    "{ty:?} {tier:?} k={k} weights #{wi} activations #{xi} \
                                     row={r}: single {} against reference {reference}, \
                                     tolerance {tol}",
                                    want[0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `n` blocks whose quants and sub-block scales are all at their maximum
/// magnitude, one buffer per sign the scales can take.
///
/// `Q4_K` scales and minima are 6-bit unsigned, and all-`0xFF` scale bytes give
/// 63 for both through `get_scale_min_k4`'s two branches, so that type has one
/// extreme. `Q6_K` scales are `i8`, so it has two: `127` and `-128`, and the
/// sign decides whether the `-32` bias adds to the accumulation or cancels it.
fn saturated_weights(ty: GgmlType, n: usize) -> Vec<Vec<u8>> {
    let scales: &[u8] = match ty {
        GgmlType::Q4_K => &[0xFF],
        GgmlType::Q6_K => &[0x7F, 0x80],
        other => panic!("no saturation fixture for {other:?}"),
    };
    scales
        .iter()
        .map(|&s| {
            let mut o = Vec::with_capacity(n * ty.block_bytes());
            for _ in 0..n {
                match ty {
                    GgmlType::Q4_K => {
                        // d and dmin at a magnitude the f32 sum stays exact at.
                        o.extend(f16::from_f32(0.0625).to_le_bytes());
                        o.extend(f16::from_f32(0.0156_25).to_le_bytes());
                        o.extend([s; 12]); // the packed 6-bit scales and minima
                        o.extend([0xFF; 128]); // every nibble 0xF
                    }
                    GgmlType::Q6_K => {
                        o.extend([0xFF; 128]); // ql: every nibble 0xF
                        o.extend([0xFF; 64]); // qh: every 2-bit pair 0x3, so q = 63
                        o.extend([s; 16]);
                        o.extend(f16::from_f32(0.0625).to_le_bytes());
                    }
                    other => panic!("no saturation fixture for {other:?}"),
                }
            }
            assert_eq!(o.len(), n * ty.block_bytes());
            o
        })
        .collect()
}

/// Activation batches of `+/-1.0`, which `quantize` turns into `+/-127` exactly
/// because the row's `amax` is 1.
///
/// The sign pattern decides the group sums the offset terms are a dot product
/// with: all one sign puts every `bsum` at its own extreme `+/-2032`, and
/// flipping per group of 16 makes neighbouring sums cancel, which is the other
/// end of the `Q4_K` minimum and `Q6_K` bias paths. Alternating per value zeroes
/// every group sum, which is the case where the offset term contributes nothing
/// and the raw accumulation is on its own.
fn saturated_activations(k: usize) -> Vec<Vec<f32>> {
    let patterns: [fn(usize) -> f32; 4] = [
        |_| 1.0,
        |i| if i % 2 == 0 { 1.0 } else { -1.0 },
        |i| if (i / 16) % 2 == 0 { 1.0 } else { -1.0 },
        |i| if (i / 128) % 3 == 0 { -1.0 } else { 1.0 },
    ];
    // Batch sizes either side of a group, so the last group of a batch is a
    // partial one here too.
    [1usize, Kernels::BATCH_GROUP + 3]
        .into_iter()
        .map(|t| {
            (0..t * k)
                .map(|i| patterns[(i / k) % patterns.len()](i % k))
                .collect()
        })
        .collect()
}

#[test]
fn unsupported_types_are_refused() {
    let kernels = Kernels::detect();
    for ty in [
        GgmlType::Q5_K,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::F32,
        GgmlType::F16,
    ] {
        let err = kernels.vec_dot(ty).unwrap_err();
        assert!(
            matches!(err, KernelError::UnsupportedType(t) if t == ty),
            "{err}"
        );
        assert!(err.to_string().contains("unsupported quantization type"));
    }
    for ty in TYPES {
        assert!(kernels.vec_dot(ty).is_ok(), "{ty:?}");
    }
}

#[test]
fn reports_the_dispatch_tier() {
    let kernels = Kernels::detect();
    println!("detected tier: {}", kernels.tier().as_str());
    println!(
        "available tiers: {:?}",
        available_tiers()
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
    );
    #[cfg(target_arch = "x86_64")]
    {
        println!("  avx2:    {}", std::arch::is_x86_feature_detected!("avx2"));
        println!("  fma:     {}", std::arch::is_x86_feature_detected!("fma"));
        println!(
            "  avxvnni: {}",
            std::arch::is_x86_feature_detected!("avxvnni")
        );
        println!(
            "  avx512f: {}",
            std::arch::is_x86_feature_detected!("avx512f")
        );
    }
    // Whatever the box is, the scalar tier must always be available.
    assert!(available_tiers().iter().any(|(t, _)| *t == Tier::Scalar));
}

// ------------------------------------------------- the embedding lookup

/// `dequantize_row` against the same scalar reference the kernels are checked
/// against.
///
/// This is the one place the product path turns a weight back into f32, for the
/// embedding lookup, and on this model only the `Q6_K` branch of it ever runs.
/// The other four are reached by a different file's embedding type, so they are
/// checked here rather than by the model.
///
/// The tolerance is relative and tiny: the two differ at most in the last bit,
/// because `dequantize_row` multiplies the integer scale and quant together
/// before the one f32 multiply while the reference above scales in f32 twice.
/// Any wrong nibble, lane, index or sign is orders of magnitude outside it.
#[test]
fn dequantize_row_matches_the_reference() {
    let mut rng = Rng::new(0x5EED_5EED);
    let ks: Vec<usize> = AWKWARD_SHAPES
        .iter()
        .chain(MODEL_SHAPES)
        .map(|&(k, _)| k)
        .collect();
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for &k in &ks {
            let nb = k.div_ceil(ty.block_len());
            let row = random_weights(ty, &mut rng, nb);
            let want = dequantize(ty, &row, nb);
            let mut got = vec![f32::NAN; k];
            llm_core::dequantize_row(ty, &row, &mut got);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let tol = w.abs() * 1e-6 + 1e-12;
                assert!(
                    (g - w).abs() <= tol,
                    "{ty:?} k={k} element {i}: {g} against {w}"
                );
            }
        }
    }
}

/// The three float layouts, which have no scalar reference above because no
/// kernel reads them.
///
/// Each is checked against what the format means rather than against the same
/// bit shuffle written twice: an f32 row comes back exactly, an f16 row comes
/// back as the `f16` it was built from, and a bf16 row comes back as the f32 it
/// was truncated from with its low mantissa bits cleared. Odd lengths are in
/// the list because a stride error is invisible at a round one.
#[test]
fn dequantize_row_reads_the_float_layouts() {
    let mut rng = Rng::new(0xF10A7);
    for k in [1usize, 2, 3, 7, 37, 256, 2560] {
        let values: Vec<f32> = (0..k).map(|_| rng.signed() * 100.0).collect();

        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend(v.to_le_bytes());
        }
        let mut got = vec![f32::NAN; k];
        llm_core::dequantize_row(GgmlType::F32, &bytes, &mut got);
        assert_eq!(got, values, "F32 k={k}");

        let mut bytes = Vec::new();
        let halves: Vec<f16> = values.iter().map(|&v| f16::from_f32(v)).collect();
        for h in &halves {
            bytes.extend(h.to_le_bytes());
        }
        let mut got = vec![f32::NAN; k];
        llm_core::dequantize_row(GgmlType::F16, &bytes, &mut got);
        let want: Vec<f32> = halves.iter().map(|h| h.to_f32()).collect();
        assert_eq!(got, want, "F16 k={k}");

        // bf16 is the top half of an f32, so truncating is the whole encoding.
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend(((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut got = vec![f32::NAN; k];
        llm_core::dequantize_row(GgmlType::BF16, &bytes, &mut got);
        let want: Vec<f32> = values
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
            .collect();
        assert_eq!(got, want, "BF16 k={k}");
    }
}

/// `can_dequantize` is what the loader checks before the forward pass starts,
/// and `dequantize_row` panics on anything else, so the two have to list the
/// same types. A type that passed the first and not the second would be a panic
/// in the middle of a run instead of an error at load.
#[test]
fn can_dequantize_lists_exactly_what_dequantize_row_handles() {
    for ty in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ] {
        assert!(llm_core::can_dequantize(ty), "{ty:?} should be accepted");
    }
    // Q8_0 has a matmul kernel but no dequantizer: nothing in the product path
    // needs one, and the loader refuses such a file by name.
    for ty in [GgmlType::Q8_0, GgmlType::Q5_K, GgmlType::Q4_0] {
        assert!(!llm_core::can_dequantize(ty), "{ty:?} should be refused");
    }
}
