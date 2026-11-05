//! GGUF reader over a memory map.
//!
//! The header, the metadata key/values and the tensor index are parsed into
//! owned structures (they are a few megabytes at most); tensor data is never
//! copied. [`Gguf::tensor`] hands out a slice that borrows straight from the
//! map, so opening a 2.7 GB file costs a few megabytes of resident memory and
//! the kernels read the quantized bytes in place.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::quant::GgmlType;

/// `"GGUF"` little-endian.
const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("not a GGUF file: magic {0:02x?}, expected {MAGIC:02x?}")]
    BadMagic([u8; 4]),
    #[error("unsupported GGUF version {0}, this reader handles 2 and 3")]
    UnsupportedVersion(u32),
    #[error("file truncated: need {need} bytes at offset {at}, file has {have}")]
    Truncated { at: usize, need: usize, have: usize },
    #[error("metadata key {key:?} has unknown value type {ty}")]
    UnknownValueType { key: String, ty: u32 },
    #[error("tensor {name:?} has unknown or unsupported ggml type {ty}")]
    UnknownTensorType { name: String, ty: u32 },
    #[error("metadata string is not valid UTF-8")]
    InvalidUtf8,
    #[error("metadata has {0} dimensions, GGUF allows at most 4")]
    TooManyDims(u32),
    #[error("general.alignment is {0}, expected a power of two")]
    BadAlignment(u64),
    #[error("missing metadata key {0:?}")]
    MissingKey(String),
    #[error("metadata key {key:?} is {got}, expected {want}")]
    MetaTypeMismatch {
        key: String,
        want: &'static str,
        got: &'static str,
    },
    #[error("missing tensor {0:?}")]
    MissingTensor(String),
    #[error("tensor {name:?} is {got:?}, expected {want:?}")]
    TensorTypeMismatch {
        name: String,
        want: GgmlType,
        got: GgmlType,
    },
    #[error(
        "tensor {name:?} runs from {offset} to {end} but the tensor data section is {len} bytes"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        end: u64,
        len: u64,
    },
    #[error(
        "tensor {name:?} has row length {row} which is not a multiple of the {ty:?} block of {block}"
    )]
    RowNotBlockAligned {
        name: String,
        row: u64,
        ty: GgmlType,
        block: usize,
    },
    #[error("tensor {name:?} with dimensions {dims:?} has a size that does not fit in a u64")]
    TensorTooLarge { name: String, dims: Vec<u64> },
    #[error("tensor {name:?} is not aligned well enough to read as {want}")]
    Misaligned { name: String, want: &'static str },
}

type Result<T> = std::result::Result<T, GgufError>;

/// A GGUF metadata value.
///
/// Arrays nest, so they hold `Value`s rather than a parallel element enum; the
/// element type is checked once while parsing, so every element of one array
/// has the same variant.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::F32(_) => "f32",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F64(_) => "f64",
        }
    }

    /// Any unsigned or non-negative signed integer, widened.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Where a tensor lives and what it holds.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset from the start of the tensor data section.
    pub offset: u64,
    /// Length of the tensor on disk.
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A tensor's bytes, borrowed from the memory map.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub ty: GgmlType,
    pub data: &'a [u8],
}

pub struct Gguf {
    path: PathBuf,
    mmap: Mmap,
    data_offset: usize,
    version: u32,
    alignment: u64,
    metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, TensorInfo>,
}

impl std::fmt::Debug for Gguf {
    /// Deliberately does not print the metadata: one value of it is the
    /// 248320-token vocabulary.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("metadata_keys", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let io = |source| GgufError::Io {
            path: path.clone(),
            source,
        };
        let file = File::open(&path).map_err(io)?;
        // SAFETY: the map is read-only and the file is expected to be a stable
        // model file. A concurrent truncation would be observable as a SIGBUS,
        // which is the same exposure every mmap-based model loader has.
        let mmap = unsafe { Mmap::map(&file) }.map_err(io)?;
        Self::parse(path, mmap)
    }

    fn parse(path: PathBuf, mmap: Mmap) -> Result<Self> {
        let mut r = Reader::new(&mmap);
        let magic = *r.array::<4>()?;
        if magic != MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty, &key)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(v) => v.as_u64().ok_or_else(|| GgufError::MetaTypeMismatch {
                key: "general.alignment".into(),
                want: "integer",
                got: v.type_name(),
            })?,
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::BadAlignment(alignment));
        }

        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims > 4 {
                return Err(GgufError::TooManyDims(n_dims));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let raw_ty = r.u32()?;
            let ty = GgmlType::from_u32(raw_ty).ok_or_else(|| GgufError::UnknownTensorType {
                name: name.clone(),
                ty: raw_ty,
            })?;
            let offset = r.u64()?;
            let row = dims.first().copied().unwrap_or(0);
            let block = ty.block_len();
            if block > 1 && row % block as u64 != 0 {
                return Err(GgufError::RowNotBlockAligned {
                    name,
                    row,
                    ty,
                    block,
                });
            }
            // Both steps are checked: the dimensions come out of the file
            // header, and a wrapped length would sail through the bounds check
            // below and open as a short tensor instead of failing by name.
            let nbytes = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .and_then(|n| ty.bytes_for(n))
                .ok_or_else(|| GgufError::TensorTooLarge {
                    name: name.clone(),
                    dims: dims.clone(),
                })?;
            tensors.insert(
                name,
                TensorInfo {
                    dims,
                    ty,
                    offset,
                    nbytes,
                },
            );
        }

        let data_offset = r.pos.next_multiple_of(alignment as usize);
        if data_offset > mmap.len() {
            return Err(GgufError::Truncated {
                at: r.pos,
                need: data_offset - r.pos,
                have: mmap.len(),
            });
        }
        let data_len = (mmap.len() - data_offset) as u64;
        for (name, t) in &tensors {
            let end = t.offset.saturating_add(t.nbytes);
            if end > data_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: name.clone(),
                    offset: t.offset,
                    end,
                    len: data_len,
                });
            }
        }

        Ok(Self {
            path,
            mmap,
            data_offset,
            version,
            alignment,
            metadata,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    fn meta_required(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    fn mismatch(key: &str, want: &'static str, got: &Value) -> GgufError {
        GgufError::MetaTypeMismatch {
            key: key.to_string(),
            want,
            got: got.type_name(),
        }
    }

    pub fn meta_u64(&self, key: &str) -> Result<u64> {
        let v = self.meta_required(key)?;
        v.as_u64()
            .ok_or_else(|| Self::mismatch(key, "non-negative integer", v))
    }

    pub fn meta_u32(&self, key: &str) -> Result<u32> {
        let v = self.meta_u64(key)?;
        u32::try_from(v).map_err(|_| GgufError::MetaTypeMismatch {
            key: key.to_string(),
            want: "u32",
            got: "integer out of range",
        })
    }

    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        let v = self.meta_required(key)?;
        v.as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| Self::mismatch(key, "float", v))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        let v = self.meta_required(key)?;
        v.as_str().ok_or_else(|| Self::mismatch(key, "string", v))
    }

    pub fn meta_array(&self, key: &str) -> Result<&[Value]> {
        let v = self.meta_required(key)?;
        v.as_array().ok_or_else(|| Self::mismatch(key, "array", v))
    }

    pub fn meta_bool_array(&self, key: &str) -> Result<Vec<bool>> {
        self.meta_array(key)?
            .iter()
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| Self::mismatch(key, "bool array", v))
            })
            .collect()
    }

    pub fn meta_str_array(&self, key: &str) -> Result<Vec<&str>> {
        self.meta_array(key)?
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| Self::mismatch(key, "string array", v))
            })
            .collect()
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The tensor's bytes, borrowed from the map.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let (name, info) = self
            .tensors
            .get_key_value(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let start = self.data_offset + info.offset as usize;
        Ok(TensorView {
            name,
            dims: &info.dims,
            ty: info.ty,
            data: &self.mmap[start..start + info.nbytes as usize],
        })
    }

    /// Same as [`Gguf::tensor`], but refuses a tensor of another type.
    pub fn tensor_typed(&self, name: &str, want: GgmlType) -> Result<TensorView<'_>> {
        let t = self.tensor(name)?;
        if t.ty != want {
            return Err(GgufError::TensorTypeMismatch {
                name: name.to_string(),
                want,
                got: t.ty,
            });
        }
        Ok(t)
    }

    /// An `F32` tensor as `f32`s, without copying.
    pub fn tensor_f32(&self, name: &str) -> Result<&[f32]> {
        let t = self.tensor_typed(name, GgmlType::F32)?;
        if t.data.as_ptr().align_offset(align_of::<f32>()) != 0 {
            return Err(GgufError::Misaligned {
                name: name.to_string(),
                want: "f32",
            });
        }
        // SAFETY: the slice is `4 * n` bytes of a read-only map, checked above
        // to be 4-byte aligned, and every bit pattern is a valid f32.
        Ok(unsafe { std::slice::from_raw_parts(t.data.as_ptr().cast::<f32>(), t.data.len() / 4) })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated {
            at: self.pos,
            need: n,
            have: self.b.len(),
        })?;
        if end > self.b.len() {
            return Err(GgufError::Truncated {
                at: self.pos,
                need: n,
                have: self.b.len(),
            });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<&'a [u8; N]> {
        Ok(self.take(N)?.try_into().expect("take returned N bytes"))
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(*self.array::<2>()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(*self.array::<4>()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(*self.array::<8>()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }

    fn value(&mut self, ty: u32, key: &str) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()?;
                // Every element needs at least one byte, so a length beyond
                // what is left in the file is a corrupt header. Checking it
                // here keeps a bogus count from growing the vector first.
                let remaining = (self.b.len() - self.pos) as u64;
                if len > remaining {
                    return Err(GgufError::Truncated {
                        at: self.pos,
                        need: len as usize,
                        have: self.b.len(),
                    });
                }
                let mut out = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    out.push(self.value(elem_ty, key)?);
                }
                Value::Array(out)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => {
                return Err(GgufError::UnknownValueType {
                    key: key.into(),
                    ty,
                });
            }
        })
    }
}
