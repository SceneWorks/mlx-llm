//! Safetensors weight loading.
//!
//! [`Weights`] is a flat name → `Array` map loaded from a single file or a sharded HF snapshot
//! directory (`model-00001-of-0000N.safetensors`, …). Models look tensors up by their HF key via
//! [`Weights::require`] / [`Weights::get`]. MLX reads safetensors on the CPU stream by default; the
//! arrays are lifted to the GPU lazily on first use.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use crate::error::{Error, Result};

/// A loaded set of named weight tensors.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: HashMap<String, Array>,
}

impl Weights {
    /// Construct directly from an in-memory map (used by converters and tests).
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self { tensors }
    }

    /// Load every tensor from a single `.safetensors` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tensors = Array::load_safetensors(path)
            .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", path.display())))?;
        Ok(Self { tensors })
    }

    /// Load and merge every `*.safetensors` shard in a snapshot directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort(); // deterministic merge order
        let mut tensors = HashMap::new();
        for shard in shards {
            let part = Array::load_safetensors(&shard)
                .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", shard.display())))?;
            tensors.extend(part);
        }
        Ok(Self { tensors })
    }

    /// Fetch a tensor by key, erroring if absent.
    pub fn require(&self, key: &str) -> Result<&Array> {
        self.tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))
    }

    /// Fetch a tensor by key if present.
    pub fn get(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    /// Whether a key is present.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// All loaded tensor keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Consume into the underlying `name → Array` map (used by the snapshot writer, which drains the
    /// loaded tensor set into its safetensors output).
    pub fn into_map(self) -> HashMap<String, Array> {
        self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_and_get_on_in_memory_map() {
        let mut m = HashMap::new();
        m.insert("a.weight".to_string(), Array::from_slice(&[1.0f32, 2.0], &[2]));
        let w = Weights::from_map(m);
        assert_eq!(w.len(), 1);
        assert!(w.contains("a.weight"));
        assert!(w.require("a.weight").is_ok());
        assert!(w.get("missing").is_none());
        assert!(matches!(
            w.require("missing"),
            Err(Error::MissingTensor(_))
        ));
    }

    #[test]
    fn save_then_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mlx-llm-weights-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        Array::save_safetensors([("w", &a)], None, &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.require("w").unwrap().shape(), &[2, 2]);

        let w2 = Weights::from_dir(&dir).unwrap();
        assert!(w2.contains("w"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
