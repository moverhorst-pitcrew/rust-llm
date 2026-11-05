//! Dequant-free quantized matmul kernels and their runtime dispatch.
//!
//! Every kernel takes one row of quantized weight bytes straight out of the
//! memory map plus a block of int8 activations, accumulates the products in
//! i32 and applies the f32 block scales at the end. Weights are never
//! dequantized to f32.
//!
//! The CPU tier is resolved once, when [`Kernels::detect`] builds the function
//! table, not per call. Detection is at runtime (`is_x86_feature_detected!`)
//! rather than `#[target_feature]` at compile time, because the machine that
//! runs the binary need not be the machine that built it.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod vnni;

use crate::quant::{
    Activations, GgmlType, Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K, QK8_0, q4k_scale_min, read_f16,
};

/// One slot past the highest [`GgmlType`] discriminant, so a type indexes the
/// table directly instead of going through a match.
const GGML_TYPE_SLOTS: usize = 31;

/// Dot product of one quantized weight row with a block of int8 activations.
///
/// The row must hold `ceil(a.k() / block_len)` blocks of the kernel's type.
pub type VecDot = fn(w: &[u8], a: &Activations) -> f32;

/// The instruction set the kernels were resolved to.
///
/// Ordered weakest to strongest; [`Kernels::detect`] picks the strongest tier
/// the CPU reports. An AVX-512 tier slots in here and in the two tables in
/// [`Kernels::detect`] without touching anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Scalar,
    Avx2Fma,
    Avx2Vnni,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2Fma => "avx2+fma",
            Tier::Avx2Vnni => "avx2+vnni",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("unsupported quantization type {0:?} on the product path")]
    UnsupportedType(GgmlType),
}

/// The resolved kernel table.
pub struct Kernels {
    tier: Tier,
    /// Indexed by the ggml type discriminant. `None` means the product path has
    /// no dequant-free kernel for it.
    dots: [Option<VecDot>; GGML_TYPE_SLOTS],
}

impl Kernels {
    /// Resolves the kernel table from the CPU features reported at runtime.
    pub fn detect() -> Self {
        Self::build(Self::best_tier())
    }

    /// The strongest tier this CPU can run. Each step is a superset of the one
    /// before it, so an AVX-512 tier goes on the end.
    pub fn best_tier() -> Tier {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                if std::arch::is_x86_feature_detected!("avxvnni") {
                    return Tier::Avx2Vnni;
                }
                return Tier::Avx2Fma;
            }
        }
        Tier::Scalar
    }

    /// Builds the table pinned to `tier`, for the microbenchmark and the tests
    /// that compare tiers. `None` if this CPU cannot run it.
    pub fn for_tier(tier: Tier) -> Option<Self> {
        (tier <= Self::best_tier()).then(|| Self::build(tier))
    }

    /// Fills the table up to `tier`, strongest kernel last.
    fn build(tier: Tier) -> Self {
        let mut dots: [Option<VecDot>; GGML_TYPE_SLOTS] = [None; GGML_TYPE_SLOTS];
        dots[GgmlType::Q4_K as usize] = Some(q4k_dot_scalar as VecDot);
        dots[GgmlType::Q6_K as usize] = Some(q6k_dot_scalar as VecDot);
        dots[GgmlType::Q8_0 as usize] = Some(q80_dot_scalar as VecDot);

        #[cfg(target_arch = "x86_64")]
        {
            if tier >= Tier::Avx2Fma {
                dots[GgmlType::Q4_K as usize] = Some(avx2::q4k_dot as VecDot);
                dots[GgmlType::Q6_K as usize] = Some(avx2::q6k_dot as VecDot);
                dots[GgmlType::Q8_0 as usize] = Some(avx2::q80_dot as VecDot);
            }
            if tier >= Tier::Avx2Vnni {
                dots[GgmlType::Q4_K as usize] = Some(vnni::q4k_dot as VecDot);
                dots[GgmlType::Q6_K as usize] = Some(vnni::q6k_dot as VecDot);
                // Q8_0 weights are signed and VPDPBUSD needs an unsigned left
                // operand; correcting for that costs more than the instruction
                // saves, so this type keeps the AVX2 kernel.
            }
        }

        Self { tier, dots }
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Looks the kernel up once; call this outside the hot loop.
    pub fn vec_dot(&self, ty: GgmlType) -> Result<VecDot, KernelError> {
        self.dots
            .get(ty as usize)
            .copied()
            .flatten()
            .ok_or(KernelError::UnsupportedType(ty))
    }

    /// `out[r] = row_r . x` for a `[k, rows]` quantized weight tensor, with the
    /// activations already quantized into `a`.
    ///
    /// Allocates nothing: the caller owns both `a` and `out`.
    pub fn matvec(
        &self,
        ty: GgmlType,
        w: &[u8],
        k: usize,
        a: &Activations,
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        let dot = self.vec_dot(ty)?;
        let stride = ty.bytes_for(k as u64) as usize;
        assert_eq!(a.k(), k, "activations were quantized for a different k");
        assert!(
            w.len() >= stride * out.len(),
            "weight tensor holds {} bytes, need {} for {} rows",
            w.len(),
            stride * out.len(),
            out.len()
        );
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(&w[r * stride..(r + 1) * stride], a);
        }
        Ok(())
    }
}

fn q4k_dot_scalar(w: &[u8], a: &Activations) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks().iter().enumerate() {
        let b = &w[ib * Q4_K_BYTES..(ib + 1) * Q4_K_BYTES];
        let d = read_f16(b, 0);
        let dmin = read_f16(b, 2);
        let (scales, qs) = (&b[4..16], &b[16..]);
        let mut isum = 0i32;
        let mut imin = 0i32;
        for j in 0..8 {
            let (sc, m) = q4k_scale_min(j, scales);
            // Sub-block 2t comes from the low nibbles of qs[32t..32t+32] and
            // sub-block 2t+1 from the high nibbles of the same bytes.
            let qw = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
            let qa = &ab.qs[j * 32..j * 32 + 32];
            let shift = if j % 2 == 0 { 0 } else { 4 };
            let mut dot = 0i32;
            for (&q, &av) in qw.iter().zip(qa) {
                dot += ((q >> shift) & 0xF) as i32 * av as i32;
            }
            isum += sc as i32 * dot;
            imin += m as i32 * (ab.bsums[j * 2] as i32 + ab.bsums[j * 2 + 1] as i32);
        }
        sum += ab.d * (d * isum as f32 - dmin * imin as f32);
    }
    sum
}

fn q6k_dot_scalar(w: &[u8], a: &Activations) -> f32 {
    let mut sum = 0.0f32;
    for (ib, ab) in a.blocks().iter().enumerate() {
        let b = &w[ib * Q6_K_BYTES..(ib + 1) * Q6_K_BYTES];
        let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
        let d = read_f16(b, 208);
        let mut isum = 0i32;
        // Two halves of 128 values, each with 64 low-nibble bytes, 32 high-bit
        // bytes and 8 scales.
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
                    isum += s * q * ab.qs[h * 128 + g * 32 + l] as i32;
                }
            }
        }
        sum += ab.d * d * isum as f32;
    }
    sum
}

fn q80_dot_scalar(w: &[u8], a: &Activations) -> f32 {
    let blocks = a.blocks();
    let mut sum = 0.0f32;
    for ib in 0..a.k().div_ceil(QK8_0) {
        let b = &w[ib * Q8_0_BYTES..(ib + 1) * Q8_0_BYTES];
        let dw = read_f16(b, 0);
        let ab = &blocks[ib / (QK_K / QK8_0)];
        let qa = &ab.qs[(ib % (QK_K / QK8_0)) * QK8_0..];
        let mut dot = 0i32;
        for (&q, &av) in b[2..].iter().zip(&qa[..QK8_0]) {
            dot += q as i8 as i32 * av as i32;
        }
        sum += dw * ab.d * dot as f32;
    }
    sum
}
