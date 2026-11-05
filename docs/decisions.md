# Decisions

Every crate and every data-layout choice, with its justification. Performance
first, then simplicity. A choice that is only simpler loses to one that is
measurably faster on this model.

## No C or C++, anywhere

No llama.cpp, no ggml, no FFI. Pure Rust crates are fine where they are the
right tool, which means `cargo tree --edges build` may not contain `cc`,
`cmake` or `bindgen`. CI checks that on every push, because a C toolchain
usually arrives quietly, as a transitive `-sys` dependency of something
harmless.

## Workspace layout: two crates

- `llm-core` — GGUF reader, ggml quantization block layouts, quantized matmul
  kernels and their dispatch.
- `rust-llm` — tokenizer, chat template, and later the model and the CLI.

Two crates rather than one, because the tokenizer pulls in a dependency tree
(`tokenizers`, `regex`, `serde`) that the loader and the kernels have no
business compiling against, and rather than four because the loader and the
kernels have to agree on the block layouts: the loader needs a block's size on
disk to index a tensor, and the kernels read those same bytes in place. Putting
them in separate crates would mean one of them owning the layouts and the other
depending on it for nothing else.

## Edition 2024, Rust stable 1.99

Stable only. `Cargo.lock` is committed because this is an application, not a
library: a reproducible build matters more than testing against the newest
versions of everything.

`[profile.release]` uses `lto = "thin"` and `codegen-units = 1`. The kernels
are built from small functions that must inline into their loops; with the
default 16 codegen units they do not reliably, and the microbenchmark measures
the wrong thing. `cargo bench` inherits this profile.

## Dependencies

| crate | why |
| --- | --- |
| `memmap2` | The only sane way to get 2.7 GB of weights addressable without reading them. Thin, pure Rust, no build script. |
| `half` | `f16` decoding for the ggml block scales. `f16` in `core` is still unstable, and hand-rolling the subnormal cases is a silent-wrongness risk for twenty saved lines. |
| `thiserror` | Error enums with a `Display` that names the file, the key or the tensor. Derive only, nothing at runtime. |
| `tokenizers` | See the tokenizer section. |
| `flate2` | See the tokenizer section. `miniz_oxide` backend, pure Rust. |
| `serde_json` | Reads the committed test fixtures. Already in the tree through `tokenizers`. |

Deliberately not taken:

- **`candle`**. It would hand us a forward pass, but not this architecture, and
  it owns the memory layout and the kernel path, which is exactly what we need
  to control. A linear-attention block would be written from scratch inside it
  anyway, against its tensor abstraction instead of against the mmap.
- **`rayon`** is not here yet. Threading lands with the forward pass, where
  there is something to parallelise over.
- **`bytemuck`**. Three `from_le_bytes` calls and one checked pointer cast.

## GGUF loader

**Weights stay in the map.** The header, metadata and tensor index are parsed
into owned structures; tensor bytes are never copied. `Gguf::tensor` returns a
slice borrowed from the map, and the kernels read the quantized bytes in place.
Opening the 2.7 GB file costs about 52 MB of resident memory, essentially all
of it the 248320-token vocabulary and the 247587 merges, which is 2% of the
file.

**No self-referential struct.** `Gguf` owns the `Mmap` and stores byte offsets,
not slices; `tensor()` resolves an offset against `&self.mmap` on each call.
This is why there is no lifetime parameter on `Gguf`, no `Pin`, and no
owning-reference crate.

**Metadata arrays hold `Value`s, including nested arrays.** GGUF allows an
array of arrays, so a parallel element-type enum would have to recurse anyway.
The cost is the enum tag per element, about 16 MB on the vocabulary, which is
noise against a 2.7 GB model and is paid once at load.

**The block count comes out of the tensors, not out of `block_count`.** In this
conversion `qwen35.block_count` is 33 while `blk.32` does not exist, because
the MTP block was stripped. `qwen35::layer_plan` takes
`block_count - nextn_predict_layers` and then requires every block it implies
to be present, so a file that really is missing a block fails with the name of
the missing tensor instead of producing a 31-block model.

**Layer typing comes from `qwen35.attention.recurrent_layers`.** Deriving it
from `full_attention_interval` would be arithmetic that happens to agree with
this file.

**The loader validates, the kernels assume.** Bounds, alignment, block-aligned
row lengths and unknown type tags are all rejected at open time with an error
that names the tensor. After that the kernels index without re-checking, which
is what keeps the hot loop free of branches that cannot fire.
