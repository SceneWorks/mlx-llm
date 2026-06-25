use std::path::{Path, PathBuf};

use core_llm::{
    detect_format, Error as CoreError, LoadSpec, ModelFormat, PrepareReport, PrepareSpec, Quantize,
    Result as CoreResult,
};

use crate::gguf::{convert_file, ConvertOptions};
use crate::primitives::projection::QuantSpec;
use crate::primitives::Weights;
use crate::snapshot::write_hf_snapshot;

fn backend() -> &'static str {
    "mlx"
}

fn to_quant_spec(q: Quantize) -> QuantSpec {
    match q {
        Quantize::Q4 => QuantSpec::q4(),
        Quantize::Q8 => QuantSpec::q8(),
    }
}

fn to_core(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Canceled => CoreError::Canceled,
        crate::Error::Unsupported(m) => CoreError::Unsupported(m),
        crate::Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => CoreError::Load(m),
        crate::Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

pub fn can_prepare(spec: &PrepareSpec) -> bool {
    matches!(
        detect_format(&spec.source),
        Ok(ModelFormat::Gguf | ModelFormat::Safetensors)
    )
}

pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    let format = detect_format(&spec.source)?;
    let quant = spec.quantize.map(to_quant_spec);

    match format {
        ModelFormat::Gguf => {
            let gguf_path = resolve_gguf_path(&spec.source)?;
            let report = convert_file(
                &gguf_path,
                &spec.out_dir,
                ConvertOptions { quantize: quant },
            )
            .map_err(to_core)?;
            Ok(PrepareReport {
                input_format: format,
                quantized: spec.quantize,
                out_dir: report.out_dir,
                num_tensors: report.num_tensors,
                passthrough: false,
            })
        }
        ModelFormat::Safetensors => match spec.quantize {
            None => {
                let num_tensors = validate_loadable_snapshot(&spec.source)?;
                Ok(PrepareReport {
                    input_format: format,
                    quantized: None,
                    out_dir: spec.source.clone(),
                    num_tensors,
                    passthrough: true,
                })
            }
            Some(_) => {
                validate_supported_hf_architecture(&spec.source)?;
                let report =
                    write_hf_snapshot(&spec.source, &spec.out_dir, quant).map_err(to_core)?;
                Ok(PrepareReport {
                    input_format: format,
                    quantized: spec.quantize,
                    out_dir: report.out_dir,
                    num_tensors: report.num_tensors,
                    passthrough: false,
                })
            }
        },
    }
}

fn resolve_gguf_path(source: &Path) -> CoreResult<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(source)
        .map_err(CoreError::Io)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    found.sort();
    found.into_iter().next().ok_or_else(|| {
        CoreError::Unsupported(format!(
            "cannot prepare '{}': directory contains no *.gguf file",
            source.display()
        ))
    })
}

fn validate_supported_hf_architecture(source: &Path) -> CoreResult<()> {
    let config_path = source.join("config.json");
    let text = std::fs::read_to_string(&config_path).map_err(|e| {
        CoreError::Unsupported(format!(
            "cannot prepare '{}': read config.json: {e}",
            source.display()
        ))
    })?;
    serde_json::from_str::<serde_json::Value>(&text).map_err(|e| {
        CoreError::Unsupported(format!(
            "cannot prepare '{}': config.json is not valid JSON: {e}",
            source.display()
        ))
    })?;
    let load_spec = LoadSpec::dense(source.to_string_lossy().to_string());
    if !crate::provider::can_load(&load_spec) {
        return Err(CoreError::Unsupported(format!(
            "cannot prepare '{}': architecture is not supported by mlx-llm",
            source.display()
        )));
    }
    Ok(())
}

fn validate_loadable_snapshot(source: &Path) -> CoreResult<usize> {
    validate_supported_hf_architecture(source)?;
    let weights = Weights::from_dir(source).map_err(to_core)?;
    if weights.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "cannot prepare '{}': snapshot has no weight tensors",
            source.display()
        )));
    }
    Ok(weights.len())
}

inventory::submit! {
    core_llm::SnapshotPreparerRegistration {
        backend,
        can_prepare,
        prepare,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;
    use serde_json::json;

    fn unique_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mlx-llm-prepare-{label}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_hf_dir(dir: &Path) {
        let config = json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 8,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        Array::save_safetensors(
            [("model.embed_tokens.weight", &a)],
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
    }

    #[test]
    fn can_prepare_accepts_hf_dir_and_rejects_unknown() {
        let hf = unique_dir("canprep-hf");
        write_hf_dir(&hf);
        assert!(can_prepare(&PrepareSpec::dense(&hf, hf.join("out"))));
        std::fs::remove_dir_all(&hf).ok();

        let empty = unique_dir("canprep-empty");
        assert!(!can_prepare(&PrepareSpec::dense(&empty, empty.join("out"))));
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn dense_hf_is_a_no_rewrite_passthrough() {
        let hf = unique_dir("passthrough-src");
        write_hf_dir(&hf);
        let out = unique_dir("passthrough-out");
        std::fs::remove_dir_all(&out).ok();

        let report = prepare(&PrepareSpec::dense(&hf, &out)).unwrap();
        assert!(report.passthrough, "dense HF must be a passthrough");
        assert_eq!(report.out_dir, hf, "passthrough returns the source dir");
        assert_eq!(report.quantized, None);
        assert!(report.num_tensors > 0);
        assert!(!out.exists(), "passthrough must not write the out_dir");

        std::fs::remove_dir_all(&hf).ok();
    }

    #[test]
    fn unknown_source_is_unsupported() {
        let empty = unique_dir("unknown");
        match prepare(&PrepareSpec::dense(&empty, empty.join("out"))) {
            Err(CoreError::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn quantized_hf_writes_a_loadable_quantized_snapshot() {
        let src = unique_dir("q-src");
        let out = unique_dir("q-out");
        std::fs::remove_dir_all(&out).ok();

        let (h, v, inter, qd, kvd) = (64i32, 4i32, 128i32, 64i32, 32i32);
        let mut tensors: Vec<(String, Array)> = Vec::new();
        let randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            Array::from_slice(&vec![0.05f32; n as usize], shape)
        };
        tensors.push(("model.embed_tokens.weight".into(), randn(&[v, h])));
        tensors.push((
            "model.norm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        tensors.push(("lm_head.weight".into(), randn(&[v, h])));
        tensors.push((
            "model.layers.0.input_layernorm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        tensors.push((
            "model.layers.0.post_attention_layernorm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        tensors.push((
            "model.layers.0.self_attn.q_proj.weight".into(),
            randn(&[qd, h]),
        ));
        tensors.push((
            "model.layers.0.self_attn.k_proj.weight".into(),
            randn(&[kvd, h]),
        ));
        tensors.push((
            "model.layers.0.self_attn.v_proj.weight".into(),
            randn(&[kvd, h]),
        ));
        tensors.push((
            "model.layers.0.self_attn.o_proj.weight".into(),
            randn(&[h, qd]),
        ));
        tensors.push((
            "model.layers.0.mlp.gate_proj.weight".into(),
            randn(&[inter, h]),
        ));
        tensors.push((
            "model.layers.0.mlp.up_proj.weight".into(),
            randn(&[inter, h]),
        ));
        tensors.push((
            "model.layers.0.mlp.down_proj.weight".into(),
            randn(&[h, inter]),
        ));
        let config = json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": v,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false
        });
        std::fs::write(
            src.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, src.join("model.safetensors")).unwrap();

        let report = prepare(&PrepareSpec::quantized(&src, &out, Quantize::Q4)).unwrap();
        assert!(
            !report.passthrough,
            "a quantization request must write a snapshot"
        );
        assert_eq!(report.quantized, Some(Quantize::Q4));
        assert_eq!(report.out_dir, out);

        let cfg = crate::config::ModelConfig::from_dir(&out).unwrap();
        assert_eq!(cfg.quantization, Some(QuantSpec::q4()));
        let model =
            crate::models::CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg)
                .unwrap();
        assert!(
            model.is_quantized(),
            "quantized snapshot must load as quantized"
        );

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }
}
