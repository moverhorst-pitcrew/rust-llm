//! Checks that run against the real GGUF file.
//!
//! Ignored by default so CI stays green on a machine without the weights. Run
//! them with:
//!
//! ```text
//! cargo test -p llm-core --release --test real_model -- --ignored --nocapture
//! ```
//!
//! `RUST_LLM_MODEL` holds the path to the GGUF file; there is no default.
//!
//! **This binary holds exactly one test on purpose.** It measures resident set
//! size, which is a property of the process, not of the test, so a second test
//! that opened the model would be measured too: the test harness runs them in
//! parallel threads, and the number would silently become "two copies of the
//! metadata". Anything else that needs the model file goes in its own test
//! binary.

use std::collections::BTreeMap;

use llm_core::quant::GgmlType;
use llm_core::{Gguf, qwen35};

/// The model file, from `RUST_LLM_MODEL`.
///
/// No default: a path baked into the repository is a path on one machine. The
/// message is what someone who has just cloned this needs to read.
fn model_path() -> String {
    std::env::var("RUST_LLM_MODEL")
        .unwrap_or_else(|_| panic!("set RUST_LLM_MODEL to the GGUF file (see docs/models.md)"))
}

/// A `/proc/self/status` size field in KiB (`VmRSS`, `VmHWM`, ...).
fn status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    status
        .lines()
        .find_map(|l| l.strip_prefix(field)?.strip_prefix(':'))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("{field} in /proc/self/status"))
}

#[test]
#[ignore = "needs the model file"]
fn opens_the_real_model() {
    let path = model_path();
    let file_bytes = std::fs::metadata(&path).expect("model file").len();
    let before = status_kib("VmRSS");

    let g = Gguf::open(&path).expect("open gguf");
    let after_open = status_kib("VmRSS");
    let open_cost = after_open.saturating_sub(before);

    let plan = qwen35::layer_plan(&g).expect("layer plan");

    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for t in g.tensors().values() {
        *by_type.entry(format!("{:?}", t.ty)).or_default() += 1;
    }

    // Touch every tensor's first and last byte, so the report cannot be true
    // only because nothing was ever read through the map.
    let mut checksum = 0u64;
    for name in g.tensors().keys() {
        let t = g.tensor(name).expect("tensor");
        checksum += t.data[0] as u64 + t.data[t.data.len() - 1] as u64;
    }
    let after_touch = status_kib("VmRSS");

    println!("file:            {path}");
    println!(
        "file size:       {:.2} GiB ({file_bytes} bytes)",
        file_bytes as f64 / (1 << 30) as f64
    );
    println!("gguf version:    {}", g.version());
    println!("alignment:       {}", g.alignment());
    println!("tensors:         {}", g.tensor_count());
    println!("tensor types:    {by_type:?}");
    println!("metadata keys:   {}", g.metadata().len());
    println!(
        "block_count:     {}",
        g.meta_u32("qwen35.block_count").unwrap()
    );
    println!(
        "nextn_predict:   {}",
        g.meta_u32("qwen35.nextn_predict_layers").unwrap()
    );
    println!("real blocks:     {}", plan.n_blocks);
    println!("full attention:  {:?}", plan.full_attention_layers());
    println!("linear attention:{:?}", plan.linear_attention_layers());
    println!("tensor checksum: {checksum}");
    println!(
        "RSS before open: {before} KiB\n\
         RSS after open:  {after_open} KiB, so open cost {open_cost} KiB ({:.2}% of the file)\n\
         RSS after touch: {after_touch} KiB (first and last byte of all 426 tensors)\n\
         peak RSS:        {} KiB",
        (open_cost as f64 * 1024.0) / file_bytes as f64 * 100.0,
        status_kib("VmHWM")
    );

    assert_eq!(g.tensor_count(), 426, "tensor count");
    assert_eq!(plan.n_blocks, 32, "real block count");
    assert_eq!(
        plan.full_attention_layers(),
        vec![3, 7, 11, 15, 19, 23, 27, 31]
    );
    assert_eq!(plan.linear_attention_layers().len(), 24);
    assert_eq!(by_type.get("Q4_K").copied(), Some(216));
    assert_eq!(by_type.get("Q6_K").copied(), Some(33));
    assert_eq!(by_type.get("F32").copied(), Some(177));

    // The whole point of the memory map: a 2.7 GB file must not become 2.7 GB
    // of resident memory. What `open` does keep is the metadata, and nearly all
    // of that is the 248320-token vocabulary and the 247587 merges. The assert
    // is on what `open` itself added, not on the process total, so the baseline
    // the harness starts with does not count against it.
    //
    // Touching tensors afterwards costs more than the bytes touched, because
    // the kernel reads ahead around every fault. That is paging working as
    // intended, so it is reported but not asserted on.
    assert!(
        open_cost * 1024 < file_bytes / 50,
        "open cost {open_cost} KiB, not far below the {file_bytes} byte file"
    );

    // The typed accessors refuse a mismatch rather than reinterpreting bytes.
    assert!(g.tensor_typed("token_embd.weight", GgmlType::Q6_K).is_ok());
    assert!(g.tensor_typed("token_embd.weight", GgmlType::Q4_K).is_err());
    assert!(g.tensor_f32("token_embd.weight").is_err());
    assert!(g.tensor_f32("output_norm.weight").is_ok());

    // The metadata and a sample of the tensor map, for reading by eye. Printed
    // last so that formatting it cannot land inside the measurement.
    println!("\n--- metadata ---");
    for (k, v) in g.metadata() {
        let shown = match v {
            llm_core::Value::Array(a) if a.len() > 8 => {
                format!(
                    "array[{}] of {} = {:?}..",
                    a.len(),
                    a[0].type_name(),
                    &a[..4]
                )
            }
            other => format!("{other:?}"),
        };
        println!("{k} = {shown}");
    }
    println!("\n--- tensors: the globals, one linear block, one full block ---");
    for (name, t) in g.tensors() {
        if name.starts_with("blk.3.") || name.starts_with("blk.0.") || !name.starts_with("blk.") {
            println!("{name}: {:?} {:?} {} bytes", t.dims, t.ty, t.nbytes);
        }
    }
}
