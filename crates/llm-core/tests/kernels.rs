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
