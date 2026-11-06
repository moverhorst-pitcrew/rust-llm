//! Tokenizer, chat template, model, HTTP surface and CLI for the CPU runner.

pub mod bench;
pub mod chat;
pub mod forward;
pub mod generate;
pub mod model;
pub mod pool;
pub mod sample;
pub mod serve;
pub mod tokenizer;

pub use chat::{ChatError, ChatOptions, Message, Role, STOP_TOKENS};
pub use forward::Runner;
pub use generate::{DraftSource, Stop, Timings};
pub use model::{Config, Model, ModelError};
pub use sample::{Sampler, Sampling};
pub use tokenizer::Tokenizer;

/// The thread count to use when the caller does not pick one.
///
/// `available_parallelism` respects the cgroup quota, which is why it reports
/// 8 on this box rather than the 32 CPUs the kernel has.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `VmHWM` from `/proc/self/status`, where it exists.
pub fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:")?.split_whitespace().next())
        .and_then(|v| v.parse().ok())
}
