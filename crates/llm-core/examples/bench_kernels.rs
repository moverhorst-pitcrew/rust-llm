//! Microbenchmark for the quantized matmul kernels, one line per
//! (type, shape, tier).
//!
//! A starting point for tuning, not a tuned result: single threaded, one
//! matvec at a time, cold-ish cache. The numbers that matter here are the
//! speedup over the scalar tier and the weight bandwidth, because a 4B model
//! on CPU is bound by how fast the quantized weights can be streamed.
//!
//! ```text
//! cargo run --release -p llm-core --example bench_kernels
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use llm_core::quant::Activations;
use llm_core::{GgmlType, Kernels, Tier};

/// `(label, k, rows)` for the shapes this model uses.
const SHAPES: &[(&str, usize, usize)] = &[
    ("attn_qkv/attn_q 2560x8192", 2560, 8192),
    ("ffn_gate/ffn_up 2560x9216", 2560, 9216),
    ("ffn_down       9216x2560", 9216, 2560),
    ("attn_out/ssm_out 4096x2560", 4096, 2560),
    ("attn_k/attn_v   2560x1024", 2560, 1024),
];

const TYPES: &[GgmlType] = &[GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0];
const TIERS: &[Tier] = &[Tier::Scalar, Tier::Avx2Fma, Tier::Avx2Vnni];

/// Deterministic filler, so two runs measure the same bytes.
fn fill(buf: &mut [u8], seed: u64) {
    let mut x = seed | 1;
    for b in buf.iter_mut() {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *b = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u8;
    }
}

/// Runs `f` for at least `budget` and returns the fastest observed call.
fn time(budget: Duration, mut f: impl FnMut()) -> Duration {
    f();
    let mut best = Duration::MAX;
    let start = Instant::now();
    while start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

fn main() {
    let detected = Kernels::detect();
    println!("detected tier: {}", detected.tier().as_str());
    println!(
        "{:<28} {:>9} {:>10} {:>9} {:>8} {:>8}",
        "shape", "type", "tier", "ms", "GB/s", "vs scalar"
    );

    for &ty in TYPES {
        for &(label, k, rows) in SHAPES {
            let stride = ty.bytes_for(k as u64).expect("fits") as usize;
            let mut w = vec![0u8; stride * rows];
            fill(&mut w, 0x9e37 + k as u64);
            // Scales are f16; random bits there would be infinities and NaNs,
            // which can change timing. Give every block the same sane scale.
            let scale_at = match ty {
                GgmlType::Q6_K => 208,
                _ => 0,
            };
            for block in w.chunks_mut(ty.block_bytes()) {
                block[scale_at..scale_at + 2].copy_from_slice(&[0x00, 0x2c]); // 0.0625
                if ty == GgmlType::Q4_K {
                    block[2..4].copy_from_slice(&[0x00, 0x24]); // 0.0156
                }
            }

            let x: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.37 - 3.0).collect();
            let mut a = Activations::new(k);
            a.quantize(&x);
            let mut out = vec![0.0f32; rows];

            let mut scalar_ms = 0.0f64;
            for &tier in TIERS {
                let Some(kernels) = Kernels::for_tier(tier) else {
                    continue;
                };
                let best = time(Duration::from_millis(400), || {
                    kernels
                        .matvec(ty, &w, k, &a, black_box(&mut out))
                        .expect("kernel");
                    black_box(&out);
                });
                let ms = best.as_secs_f64() * 1e3;
                let gbs = w.len() as f64 / best.as_secs_f64() / 1e9;
                if tier == Tier::Scalar {
                    scalar_ms = ms;
                }
                println!(
                    "{label:<28} {:>9} {:>10} {ms:>9.3} {gbs:>8.2} {:>8.2}x",
                    format!("{ty:?}"),
                    tier.as_str(),
                    scalar_ms / ms
                );
            }
        }
    }
}
