//! GGUF loading and the quantized matmul kernels for the CPU runner.
//!
//! The two live together because they share the quantization block layouts:
//! the loader has to know a block's size on disk to index a tensor, and the
//! kernels read those same bytes in place.

pub mod gguf;
pub mod kernels;
pub mod quant;
pub mod qwen35;

pub use gguf::{Gguf, GgufError, TensorView, Value};
pub use kernels::{KernelError, Kernels, Tier};
pub use quant::{Activations, GgmlType, can_dequantize, dequantize_row};
