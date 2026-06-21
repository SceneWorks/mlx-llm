//! GGUF container parsing — header, metadata key/values, tensor infos, and the aligned data section.
//!
//! GGUF is llama.cpp's single-file model container: a little-endian header, a typed metadata
//! key/value table (hyperparameters, tokenizer, arbitrary strings/arrays), a tensor-info table
//! (name, dims, quant type, data offset), then an aligned blob of tensor data. This module parses
//! the structure faithfully (spec v2/v3 — the u64-length layout every modern `*.gguf` uses) and
//! hands [`super::dequant`] the raw per-tensor bytes. v1 (u32 lengths) is rejected — no model ships
//! it anymore.
//!
//! GGML stores tensor dimensions in `ne` order (fastest-varying axis first), the reverse of the
//! torch/row-major `[out, in]` a linear weight uses; [`TensorInfo::shape`] is already reversed back
//! to torch order so the rest of the engine sees its native layout.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

/// `"GGUF"` as a little-endian `u32` (`G=0x47, G, U=0x55, F=0x46`).
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Default tensor-data alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

// GGUF metadata value-type tags.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// A typed GGUF metadata value (the value side of a metadata key/value entry).
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    /// Unsigned 8-bit.
    U8(u8),
    /// Signed 8-bit.
    I8(i8),
    /// Unsigned 16-bit.
    U16(u16),
    /// Signed 16-bit.
    I16(i16),
    /// Unsigned 32-bit.
    U32(u32),
    /// Signed 32-bit.
    I32(i32),
    /// Unsigned 64-bit.
    U64(u64),
    /// Signed 64-bit.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array of values.
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Interpret any integer variant as `u64` (for counts / ids).
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            MetaValue::Bool(b) => *b as u64,
            _ => return None,
        })
    }

    /// Interpret any integer variant as `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            MetaValue::U8(v) => *v as i64,
            MetaValue::U16(v) => *v as i64,
            MetaValue::U32(v) => *v as i64,
            MetaValue::U64(v) => *v as i64,
            MetaValue::I8(v) => *v as i64,
            MetaValue::I16(v) => *v as i64,
            MetaValue::I32(v) => *v as i64,
            MetaValue::I64(v) => *v,
            _ => return None,
        })
    }

    /// Interpret any float variant as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F32(v) => Some(*v as f64),
            MetaValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Borrow the string, if this is a [`MetaValue::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// The boolean, if this is a [`MetaValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow the elements, if this is a [`MetaValue::Array`].
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One tensor's header entry: its name, torch-order shape, quant type, and data offset.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// GGML tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Logical shape in torch/row-major order (`[out, in]` for a linear weight) — the GGML `ne`
    /// dimensions reversed.
    pub shape: Vec<usize>,
    /// GGML quant type tag (see [`super::dequant::GgmlType`]).
    pub ggml_type: u32,
    /// Byte offset of this tensor's data relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count (`shape.product()`).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed GGUF file: its metadata, tensor table, and the raw bytes backing the data section.
#[derive(Debug)]
pub struct GgufFile {
    /// Metadata key/value table.
    pub metadata: HashMap<String, MetaValue>,
    /// Tensor header entries, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Data-section alignment (`general.alignment`, default 32).
    pub alignment: u64,
    /// The whole file, kept so tensor data can be sliced lazily.
    bytes: Vec<u8>,
    /// File offset where the (aligned) tensor data section begins.
    data_start: usize,
}

impl GgufFile {
    /// Read and parse a `.gguf` file from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Msg(format!("gguf: read {}: {e}", path.display())))?;
        Self::parse(bytes)
    }

    /// Parse an in-memory GGUF image.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let mut c = Cursor::new(&bytes);

        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::Msg(format!(
                "gguf: bad magic 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})"
            )));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Unsupported(format!(
                "gguf version {version} (only v2/v3 are supported)"
            )));
        }
        let tensor_count = c.u64()? as usize;
        let metadata_count = c.u64()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = c.string()?;
            let vtype = c.u32()?;
            let value = c.value(vtype)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                return Err(Error::Msg(format!(
                    "gguf: tensor {name:?} has {n_dims} dims (expected 1..=4)"
                )));
            }
            // GGML `ne` order is fastest-axis-first; reverse to torch `[out, in]` row-major order.
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(c.u64()? as usize);
            }
            ne.reverse();
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                shape: ne,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        // The data section begins at the first alignment boundary at or after the header end.
        let header_end = c.pos() as u64;
        let data_start = header_end.div_ceil(alignment) * alignment;
        // A metadata-only file (no tensors) legitimately ends before the aligned data boundary; only
        // require the data section to be in-range when tensors actually reference it (each tensor's
        // own slice is bounds-checked in `tensor_data`).
        if !tensors.is_empty() && (data_start as usize) > bytes.len() {
            return Err(Error::Msg(
                "gguf: data section starts past end of file".into(),
            ));
        }

        Ok(Self {
            metadata,
            tensors,
            alignment,
            bytes,
            data_start: data_start as usize,
        })
    }

    /// A metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// A metadata string by key.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(MetaValue::as_str)
    }

    /// A metadata unsigned integer by key.
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(MetaValue::as_u64)
    }

    /// A metadata float by key.
    pub fn meta_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(MetaValue::as_f64)
    }

    /// The raw data bytes backing `info`, sliced from the data section. The length is derived from
    /// the tensor's quant type and element count; an out-of-range offset/length errors.
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&[u8]> {
        let nbytes = super::dequant::tensor_byte_len(info.ggml_type, info.num_elements())?;
        let start = self
            .data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| Error::Msg("gguf: tensor offset overflow".into()))?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| Error::Msg("gguf: tensor length overflow".into()))?;
        self.bytes.get(start..end).ok_or_else(|| {
            Error::Msg(format!(
                "gguf: tensor {:?} data [{start}..{end}] out of range (file {} bytes)",
                info.name,
                self.bytes.len()
            ))
        })
    }
}

/// A forward-only little-endian cursor over the GGUF image with bounds-checked primitive reads.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Msg("gguf: read overflow".into()))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::Msg("gguf: unexpected end of file".into()))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|e| Error::Msg(format!("gguf: bad utf-8 string: {e}")))
    }

    /// Read a single typed value of the given GGUF value-type tag.
    fn value(&mut self, vtype: u32) -> Result<MetaValue> {
        Ok(match vtype {
            T_UINT8 => MetaValue::U8(self.u8()?),
            T_INT8 => MetaValue::I8(self.u8()? as i8),
            T_UINT16 => MetaValue::U16(self.u16()?),
            T_INT16 => MetaValue::I16(self.u16()? as i16),
            T_UINT32 => MetaValue::U32(self.u32()?),
            T_INT32 => MetaValue::I32(self.u32()? as i32),
            T_FLOAT32 => MetaValue::F32(f32::from_bits(self.u32()?)),
            T_BOOL => MetaValue::Bool(self.u8()? != 0),
            T_STRING => MetaValue::String(self.string()?),
            T_UINT64 => MetaValue::U64(self.u64()?),
            T_INT64 => MetaValue::I64(self.u64()? as i64),
            T_FLOAT64 => MetaValue::F64(f64::from_bits(self.u64()?)),
            T_ARRAY => {
                let elem_type = self.u32()?;
                if elem_type == T_ARRAY {
                    return Err(Error::Unsupported("gguf: nested metadata arrays".into()));
                }
                let len = self.u64()? as usize;
                let mut items = Vec::with_capacity(len.min(1 << 20));
                for _ in 0..len {
                    items.push(self.value(elem_type)?);
                }
                MetaValue::Array(items)
            }
            other => {
                return Err(Error::Msg(format!("gguf: unknown metadata value type {other}")))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF image in memory: two metadata entries (a string and a u32) and
    /// one f32 tensor `[2, 3]`, then assert the parser recovers all of it including the reversed
    /// (torch-order) shape and the correctly-aligned tensor data.
    #[test]
    fn parse_minimal_gguf_roundtrip() {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&2u64.to_le_bytes()); // metadata_count

        // metadata: general.architecture = "llama"
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&T_STRING.to_le_bytes());
        push_str(&mut b, "llama");
        // metadata: llama.block_count = 7u32
        push_str(&mut b, "llama.block_count");
        b.extend_from_slice(&T_UINT32.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());

        // tensor info: name "w", ne = [3, 2] (=> torch shape [2, 3]), type F32 (0), offset 0
        push_str(&mut b, "w");
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&3u64.to_le_bytes()); // ne[0] (fastest)
        b.extend_from_slice(&2u64.to_le_bytes()); // ne[1]
        b.extend_from_slice(&0u32.to_le_bytes()); // type F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // align data section to 32
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        // 6 f32 values
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }

        let g = GgufFile::parse(b).unwrap();
        assert_eq!(g.meta_str("general.architecture"), Some("llama"));
        assert_eq!(g.meta_u64("llama.block_count"), Some(7));
        assert_eq!(g.tensors.len(), 1);
        let t = &g.tensors[0];
        assert_eq!(t.name, "w");
        assert_eq!(t.shape, vec![2, 3]); // ne reversed
        assert_eq!(t.num_elements(), 6);
        let data = g.tensor_data(t).unwrap();
        let got: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vals);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = Vec::new();
        b.extend_from_slice(&0xdead_beefu32.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        assert!(GgufFile::parse(b).is_err());
    }

    #[test]
    fn rejects_v1() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // version 1
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(GgufFile::parse(b), Err(Error::Unsupported(_))));
    }

    #[test]
    fn parses_int_and_float_and_array_metadata() {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        b.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        push_str(&mut b, "a.f32");
        b.extend_from_slice(&T_FLOAT32.to_le_bytes());
        b.extend_from_slice(&1.5f32.to_le_bytes());
        push_str(&mut b, "a.i64");
        b.extend_from_slice(&T_INT64.to_le_bytes());
        b.extend_from_slice(&(-42i64).to_le_bytes());
        // array of two u32
        push_str(&mut b, "a.arr");
        b.extend_from_slice(&T_ARRAY.to_le_bytes());
        b.extend_from_slice(&T_UINT32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&10u32.to_le_bytes());
        b.extend_from_slice(&20u32.to_le_bytes());

        let g = GgufFile::parse(b).unwrap();
        assert_eq!(g.meta_f64("a.f32"), Some(1.5));
        assert_eq!(g.meta("a.i64").unwrap().as_i64(), Some(-42));
        let arr = g.meta("a.arr").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_u64(), Some(10));
        assert_eq!(arr[1].as_u64(), Some(20));
    }
}
