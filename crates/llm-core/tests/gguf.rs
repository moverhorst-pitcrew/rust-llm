//! GGUF reader tests on fixtures built byte by byte in the test, so none of
//! them needs the model file.

use std::sync::atomic::{AtomicU32, Ordering};

use llm_core::gguf::{GgufError, Value};
use llm_core::{GgmlType, Gguf};

// ---------------------------------------------------------------- fixtures

fn p_str(s: &str) -> Vec<u8> {
    let mut o = (s.len() as u64).to_le_bytes().to_vec();
    o.extend_from_slice(s.as_bytes());
    o
}

fn p_array(elem_ty: u32, n: u64, elems: &[u8]) -> Vec<u8> {
    let mut o = elem_ty.to_le_bytes().to_vec();
    o.extend_from_slice(&n.to_le_bytes());
    o.extend_from_slice(elems);
    o
}

/// A metadata value: the type tag followed by its payload.
fn kv(ty: u32, payload: Vec<u8>) -> Vec<u8> {
    let mut o = ty.to_le_bytes().to_vec();
    o.extend(payload);
    o
}

struct Builder {
    magic: [u8; 4],
    version: u32,
    kvs: Vec<(String, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, u64)>,
    data: Vec<u8>,
    alignment: u64,
    /// Keep the file this many bytes shorter than it should be.
    truncate_by: usize,
    /// Claim this many tensors in the header regardless of how many follow.
    tensor_count_override: Option<u64>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            magic: *b"GGUF",
            version: 3,
            kvs: Vec::new(),
            tensors: Vec::new(),
            data: Vec::new(),
            alignment: 32,
            truncate_by: 0,
            tensor_count_override: None,
        }
    }
}

impl Builder {
    fn kv(mut self, key: &str, value: Vec<u8>) -> Self {
        self.kvs.push((key.to_string(), value));
        self
    }

    /// Adds a tensor whose data is appended at the next aligned offset.
    fn tensor(mut self, name: &str, dims: &[u64], ty: u32, bytes: &[u8]) -> Self {
        let offset = self.data.len().next_multiple_of(self.alignment as usize) as u64;
        self.data.resize(offset as usize, 0);
        self.data.extend_from_slice(bytes);
        self.tensors
            .push((name.to_string(), dims.to_vec(), ty, offset));
        self
    }

    /// Adds a tensor index entry pointing at an offset of our choosing.
    fn tensor_at(mut self, name: &str, dims: &[u64], ty: u32, offset: u64) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), ty, offset));
        self
    }

    fn build(&self) -> Vec<u8> {
        let mut o = self.magic.to_vec();
        o.extend_from_slice(&self.version.to_le_bytes());
        let n_tensors = self
            .tensor_count_override
            .unwrap_or(self.tensors.len() as u64);
        o.extend_from_slice(&n_tensors.to_le_bytes());
        o.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            o.extend(p_str(k));
            o.extend_from_slice(v);
        }
        for (name, dims, ty, offset) in &self.tensors {
            o.extend(p_str(name));
            o.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                o.extend_from_slice(&d.to_le_bytes());
            }
            o.extend_from_slice(&ty.to_le_bytes());
            o.extend_from_slice(&offset.to_le_bytes());
        }
        o.resize(o.len().next_multiple_of(self.alignment as usize), 0);
        o.extend_from_slice(&self.data);
        o.truncate(o.len() - self.truncate_by.min(o.len() - 1));
        o
    }

    fn open(&self) -> Result<Gguf, GgufError> {
        open_bytes(&self.build())
    }
}

static SEQ: AtomicU32 = AtomicU32::new(0);

/// Writes the bytes to a temporary file, opens it, and unlinks it right away.
/// The map survives the unlink, so the returned `Gguf` stays usable.
fn open_bytes(bytes: &[u8]) -> Result<Gguf, GgufError> {
    let path = std::env::temp_dir().join(format!(
        "rust-llm-gguf-{}-{}.gguf",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).expect("write fixture");
    let r = Gguf::open(&path);
    std::fs::remove_file(&path).expect("remove fixture");
    r
}

// ---------------------------------------------------------------- metadata

#[test]
fn reads_every_value_type() {
    let g = Builder::default()
        .kv("a.u8", kv(0, vec![1]))
        .kv("a.i8", kv(1, vec![(-2i8) as u8]))
        .kv("a.u16", kv(2, 3u16.to_le_bytes().to_vec()))
        .kv("a.i16", kv(3, (-4i16).to_le_bytes().to_vec()))
        .kv("a.u32", kv(4, 5u32.to_le_bytes().to_vec()))
        .kv("a.i32", kv(5, (-6i32).to_le_bytes().to_vec()))
        .kv("a.f32", kv(6, 7.5f32.to_le_bytes().to_vec()))
        .kv("a.bool_true", kv(7, vec![1]))
        .kv("a.bool_false", kv(7, vec![0]))
        .kv("a.string", kv(8, p_str("hello")))
        .kv("a.u64", kv(10, 8u64.to_le_bytes().to_vec()))
        .kv("a.i64", kv(11, (-9i64).to_le_bytes().to_vec()))
        .kv("a.f64", kv(12, 10.5f64.to_le_bytes().to_vec()))
        .kv(
            "a.array_u32",
            kv(
                9,
                p_array(
                    4,
                    3,
                    &[1u32, 2, 3]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
        )
        .kv(
            "a.array_string",
            kv(9, p_array(8, 2, &[p_str("x"), p_str("yz")].concat())),
        )
        .kv("a.array_empty", kv(9, p_array(6, 0, &[])))
        .kv(
            "a.array_nested",
            kv(
                9,
                p_array(
                    9,
                    2,
                    &[
                        p_array(5, 2, &[1i32.to_le_bytes(), 2i32.to_le_bytes()].concat()),
                        p_array(5, 1, &3i32.to_le_bytes()),
                    ]
                    .concat(),
                ),
            ),
        )
        .open()
        .expect("fixture opens");

    assert_eq!(g.meta("a.u8"), Some(&Value::U8(1)));
    assert_eq!(g.meta("a.i8"), Some(&Value::I8(-2)));
    assert_eq!(g.meta("a.u16"), Some(&Value::U16(3)));
    assert_eq!(g.meta("a.i16"), Some(&Value::I16(-4)));
    assert_eq!(g.meta("a.u32"), Some(&Value::U32(5)));
    assert_eq!(g.meta("a.i32"), Some(&Value::I32(-6)));
    assert_eq!(g.meta("a.f32"), Some(&Value::F32(7.5)));
    assert_eq!(g.meta("a.bool_true"), Some(&Value::Bool(true)));
    assert_eq!(g.meta("a.bool_false"), Some(&Value::Bool(false)));
    assert_eq!(g.meta_str("a.string").unwrap(), "hello");
    assert_eq!(g.meta("a.u64"), Some(&Value::U64(8)));
    assert_eq!(g.meta("a.i64"), Some(&Value::I64(-9)));
    assert_eq!(g.meta("a.f64"), Some(&Value::F64(10.5)));
    assert_eq!(
        g.meta_array("a.array_u32").unwrap(),
        &[Value::U32(1), Value::U32(2), Value::U32(3)]
    );
    assert_eq!(g.meta_str_array("a.array_string").unwrap(), ["x", "yz"]);
    assert_eq!(g.meta_array("a.array_empty").unwrap(), &[]);
    assert_eq!(
        g.meta_array("a.array_nested").unwrap(),
        &[
            Value::Array(vec![Value::I32(1), Value::I32(2)]),
            Value::Array(vec![Value::I32(3)]),
        ]
    );

    // Integers widen, everything else refuses.
    assert_eq!(g.meta_u64("a.u8").unwrap(), 1);
    assert_eq!(g.meta_u64("a.u64").unwrap(), 8);
    assert!(matches!(
        g.meta_u64("a.i8"),
        Err(GgufError::MetaTypeMismatch { .. })
    ));
    assert!(matches!(
        g.meta_str("a.u8"),
        Err(GgufError::MetaTypeMismatch { .. })
    ));
    assert!(matches!(
        g.meta_u64("missing.key"),
        Err(GgufError::MissingKey(_))
    ));
}

#[test]
fn rejects_unknown_value_type() {
    let err = Builder::default()
        .kv("a.weird", kv(42, vec![0; 8]))
        .open()
        .unwrap_err();
    assert!(
        matches!(err, GgufError::UnknownValueType { ty: 42, .. }),
        "{err}"
    );
}

#[test]
fn rejects_unknown_array_element_type() {
    let err = Builder::default()
        .kv("a.weird", kv(9, p_array(42, 1, &[0; 8])))
        .open()
        .unwrap_err();
    assert!(
        matches!(err, GgufError::UnknownValueType { ty: 42, .. }),
        "{err}"
    );
}

#[test]
fn rejects_an_array_longer_than_the_file() {
    let err = Builder::default()
        .kv("a.huge", kv(9, p_array(0, u64::MAX, &[])))
        .open()
        .unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }), "{err}");
}

// ---------------------------------------------------------------- tensors

#[test]
fn reads_tensor_offsets_and_respects_alignment() {
    let mut b = Builder {
        alignment: 64,
        ..Builder::default()
    };
    b = b.kv("general.alignment", kv(4, 64u32.to_le_bytes().to_vec()));
    // 5 f32s, so the next tensor cannot start on a 64-byte boundary by luck.
    let first: Vec<u8> = (0..5u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let second: Vec<u8> = (0..3u32).flat_map(|i| (100 + i).to_le_bytes()).collect();
    let g = b
        .tensor("first", &[5], 0, &first)
        .tensor("second", &[3], 0, &second)
        .open()
        .expect("fixture opens");

    assert_eq!(g.alignment(), 64);
    assert_eq!(g.tensor_count(), 2);
    assert_eq!(g.tensor_info("first").unwrap().offset, 0);
    assert_eq!(g.tensor_info("second").unwrap().offset, 64);
    assert_eq!(g.tensor("first").unwrap().data, &first[..]);
    assert_eq!(g.tensor("second").unwrap().data, &second[..]);
    assert_eq!(g.tensor_f32("first").unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(g.tensor("first").unwrap().dims, &[5]);
    assert_eq!(g.tensor("first").unwrap().ty, GgmlType::F32);
}

#[test]
fn computes_quantized_tensor_sizes() {
    // One Q4_K super-block per row, two rows.
    let data = vec![7u8; 2 * 144];
    let g = Builder::default()
        .tensor("w", &[256, 2], GgmlType::Q4_K as u32, &data)
        .open()
        .expect("fixture opens");
    let info = g.tensor_info("w").unwrap();
    assert_eq!(info.n_elements(), 512);
    assert_eq!(info.nbytes, 288);
    assert_eq!(g.tensor("w").unwrap().data.len(), 288);
}

#[test]
fn missing_tensor_is_an_error() {
    let g = Builder::default()
        .tensor("present", &[1], 0, &0.0f32.to_le_bytes())
        .open()
        .expect("fixture opens");
    assert!(g.has_tensor("present"));
    assert!(!g.has_tensor("absent"));
    assert!(matches!(
        g.tensor("absent"),
        Err(GgufError::MissingTensor(_))
    ));
}

#[test]
fn unknown_tensor_type_is_an_error() {
    let err = Builder::default()
        .tensor("w", &[32], 21, &[0u8; 64])
        .open()
        .unwrap_err();
    assert!(
        matches!(err, GgufError::UnknownTensorType { ty: 21, .. }),
        "{err}"
    );
}

#[test]
fn typed_access_refuses_a_mismatch() {
    let g = Builder::default()
        .tensor("f", &[1], GgmlType::F32 as u32, &1.0f32.to_le_bytes())
        .tensor("q", &[256], GgmlType::Q4_K as u32, &[0u8; 144])
        .open()
        .expect("fixture opens");

    assert!(g.tensor_typed("f", GgmlType::F32).is_ok());
    assert!(g.tensor_typed("q", GgmlType::Q4_K).is_ok());
    assert!(matches!(
        g.tensor_typed("q", GgmlType::Q6_K),
        Err(GgufError::TensorTypeMismatch {
            want: GgmlType::Q6_K,
            got: GgmlType::Q4_K,
            ..
        })
    ));
    assert!(matches!(
        g.tensor_f32("q"),
        Err(GgufError::TensorTypeMismatch { .. })
    ));
}

#[test]
fn rejects_a_row_that_is_not_a_whole_number_of_blocks() {
    let err = Builder::default()
        .tensor("w", &[100], GgmlType::Q4_K as u32, &[0u8; 144])
        .open()
        .unwrap_err();
    assert!(
        matches!(
            err,
            GgufError::RowNotBlockAligned {
                row: 100,
                block: 256,
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn rejects_a_tensor_past_the_end_of_the_data() {
    let err = Builder::default()
        .tensor_at("w", &[256], GgmlType::Q4_K as u32, 1 << 20)
        .open()
        .unwrap_err();
    assert!(matches!(err, GgufError::TensorOutOfBounds { .. }), "{err}");
}

// ---------------------------------------------------------------- headers

#[test]
fn rejects_bad_magic() {
    let err = Builder {
        magic: *b"GGML",
        ..Builder::default()
    }
    .kv("a", kv(0, vec![1]))
    .open()
    .unwrap_err();
    assert!(
        matches!(err, GgufError::BadMagic(m) if m == *b"GGML"),
        "{err}"
    );
}

#[test]
fn rejects_an_unsupported_version() {
    for version in [1u32, 4, 0xdead] {
        let err = Builder {
            version,
            ..Builder::default()
        }
        .kv("a", kv(0, vec![1]))
        .open()
        .unwrap_err();
        assert!(
            matches!(err, GgufError::UnsupportedVersion(v) if v == version),
            "{err}"
        );
    }
    assert!(
        Builder {
            version: 2,
            ..Builder::default()
        }
        .kv("a", kv(0, vec![1]))
        .open()
        .is_ok()
    );
}

#[test]
fn rejects_a_truncated_header() {
    // Shorter than magic + version + the two counts.
    let err = open_bytes(b"GGUF\x03\x00\x00\x00\x00").unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }), "{err}");
}

#[test]
fn rejects_a_file_cut_off_in_the_metadata() {
    let bytes = Builder::default()
        .kv("a.string", kv(8, p_str("hello")))
        .kv("b.u64", kv(10, 1u64.to_le_bytes().to_vec()))
        .tensor("w", &[1], 0, &0.0f32.to_le_bytes())
        .build();
    let err = open_bytes(&bytes[..40]).unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }), "{err}");
}

#[test]
fn rejects_a_file_cut_off_in_the_tensor_index() {
    let err = Builder {
        tensor_count_override: Some(4),
        ..Builder::default()
    }
    .tensor("w", &[1], 0, &0.0f32.to_le_bytes())
    .open()
    .unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }), "{err}");
}

#[test]
fn rejects_a_file_cut_off_in_the_tensor_data() {
    let err = Builder {
        truncate_by: 8,
        ..Builder::default()
    }
    .tensor("w", &[256], GgmlType::Q4_K as u32, &[0u8; 144])
    .open()
    .unwrap_err();
    assert!(matches!(err, GgufError::TensorOutOfBounds { .. }), "{err}");
}

#[test]
fn rejects_a_bad_alignment() {
    let err = Builder::default()
        .kv("general.alignment", kv(4, 48u32.to_le_bytes().to_vec()))
        .open()
        .unwrap_err();
    assert!(matches!(err, GgufError::BadAlignment(48)), "{err}");
}

// ---------------------------------------------------------------- qwen35

/// A fixture shaped like the real conversion: `block_count` 33 with one
/// `nextn_predict_layers`, so 32 blocks should be present.
fn qwen35_fixture(
    arch: &str,
    block_count: u32,
    nextn: u32,
    blocks: &[usize],
    recurrent_len: usize,
) -> Builder {
    let bools: Vec<u8> = (0..recurrent_len).map(|i| u8::from(i % 4 != 3)).collect();
    let mut b = Builder::default()
        .kv("general.architecture", kv(8, p_str(arch)))
        .kv(
            "qwen35.block_count",
            kv(4, block_count.to_le_bytes().to_vec()),
        )
        .kv(
            "qwen35.nextn_predict_layers",
            kv(4, nextn.to_le_bytes().to_vec()),
        )
        .kv(
            "qwen35.attention.recurrent_layers",
            kv(9, p_array(7, recurrent_len as u64, &bools)),
        );
    for &i in blocks {
        b = b.tensor(
            &format!("blk.{i}.attn_norm.weight"),
            &[1],
            0,
            &1.0f32.to_le_bytes(),
        );
    }
    b
}

#[test]
fn derives_the_real_block_count_from_the_tensors() {
    let g = qwen35_fixture("qwen35", 33, 1, &(0..32).collect::<Vec<_>>(), 33)
        .open()
        .expect("fixture opens");
    let plan = llm_core::qwen35::layer_plan(&g).expect("layer plan");
    assert_eq!(plan.n_blocks, 32);
    assert_eq!(
        plan.full_attention_layers(),
        vec![3, 7, 11, 15, 19, 23, 27, 31]
    );
    assert_eq!(plan.linear_attention_layers().len(), 24);
}

#[test]
fn a_missing_block_is_an_error() {
    let mut present: Vec<usize> = (0..32).collect();
    present.retain(|&i| i != 17);
    let g = qwen35_fixture("qwen35", 33, 1, &present, 33)
        .open()
        .expect("fixture opens");
    let err = llm_core::qwen35::layer_plan(&g).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("blk.17.attn_norm.weight"), "{msg}");
    assert!(msg.contains("implies 32 blocks"), "{msg}");
}

#[test]
fn a_block_beyond_the_metadata_is_an_error() {
    let g = qwen35_fixture("qwen35", 33, 1, &(0..33).collect::<Vec<_>>(), 33)
        .open()
        .expect("fixture opens");
    let err = llm_core::qwen35::layer_plan(&g).unwrap_err();
    assert!(
        matches!(
            err,
            llm_core::qwen35::ConfigError::UnexpectedBlock {
                found: 32,
                expected: 32
            }
        ),
        "{err}"
    );
}

#[test]
fn another_architecture_is_an_error() {
    let g = qwen35_fixture("llama", 33, 1, &(0..32).collect::<Vec<_>>(), 33)
        .open()
        .expect("fixture opens");
    let err = llm_core::qwen35::layer_plan(&g).unwrap_err();
    assert!(
        matches!(err, llm_core::qwen35::ConfigError::Architecture(a) if a == "llama"),
        "wrong error"
    );
}

#[test]
fn a_short_recurrent_layers_array_is_an_error() {
    let g = qwen35_fixture("qwen35", 33, 1, &(0..32).collect::<Vec<_>>(), 8)
        .open()
        .expect("fixture opens");
    let err = llm_core::qwen35::layer_plan(&g).unwrap_err();
    assert!(
        matches!(
            err,
            llm_core::qwen35::ConfigError::RecurrentLayersTooShort {
                got: 8,
                expected: 32
            }
        ),
        "{err}"
    );
}
