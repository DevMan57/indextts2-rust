//! Weight loading utilities for S2Mel model (DiT + Length Regulator)
//!
//! Handles loading safetensors checkpoints and mapping PyTorch keys to Rust model structure.

#![allow(missing_docs)]

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear};
use std::collections::HashMap;
use std::path::Path;

/// Load tensors from safetensors file
pub fn load_s2mel_safetensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();

    if !path.exists() {
        bail!("S2Mel safetensors file not found: {:?}", path);
    }

    let tensors = candle_core::safetensors::load(path, device)?;
    Ok(tensors)
}

/// Load a Linear layer from tensors
pub fn load_linear(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
    transpose: bool,
) -> Result<Linear> {
    let weight = tensors
        .get(weight_key)
        .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", weight_key))?;

    let weight = if transpose {
        weight.t()?.contiguous()?
    } else {
        weight.clone()
    };

    let bias = if let Some(bk) = bias_key {
        tensors.get(bk).cloned()
    } else {
        None
    };

    Ok(Linear::new(weight, bias))
}

/// Load a Conv1d-style weight (out_channels, in_channels, kernel_size)
pub fn load_conv1d_weight(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
) -> Result<Tensor> {
    let weight = tensors
        .get(weight_key)
        .ok_or_else(|| anyhow::anyhow!("Conv1d weight not found: {}", weight_key))?;
    Ok(weight.clone())
}

/// Load a LayerNorm from tensors
pub fn load_layer_norm(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
    eps: f64,
    dim: usize,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = match tensors.get(weight_key) {
        Some(w) => w.clone(),
        None => {
            tracing::warn!(
                "[S2Mel] Missing tensor '{}', using ones initialization",
                weight_key
            );
            Tensor::ones((dim,), DType::F32, device)?
        }
    };

    let bias = if let Some(bk) = bias_key {
        match tensors.get(bk) {
            Some(b) => b.clone(),
            None => {
                tracing::warn!(
                    "[S2Mel] Missing tensor '{}', using zeros initialization",
                    bk
                );
                Tensor::zeros((dim,), DType::F32, device)?
            }
        }
    } else {
        Tensor::zeros((dim,), DType::F32, device)?
    };

    Ok(LayerNorm::new(weight, bias, eps))
}

/// DiT (Diffusion Transformer) weights from cfm.estimator.*
pub struct DiTWeights {
    pub cond_embedder: Linear,
    pub cond_projection: Linear,
    pub cond_x_merge_linear: Linear,
    pub conv1_weight: Tensor,
    pub conv1_bias: Option<Tensor>,
    pub conv2_weight: Tensor,
    pub conv2_bias: Option<Tensor>,
    pub transformer_layers: Vec<DiTLayerWeights>,
}

/// AdaLayerNorm weights (norm + projection for scale/shift)
pub struct AdaLayerNormWeights {
    pub norm: LayerNorm,
    pub project_weight: Linear,
}

/// DiT transformer layer weights
pub struct DiTLayerWeights {
    pub wqkv: Linear,
    pub wo: Linear,
    pub w1: Linear,  // FFN first layer (fused w1 and w3 for SwiGLU)
    pub w2: Linear,  // FFN second layer
    pub w3: Option<Linear>,  // Optional if fused with w1
    pub attn_norm: Option<AdaLayerNormWeights>,
    pub ffn_norm: Option<AdaLayerNormWeights>,
    pub skip_in_linear: Option<Linear>,
}

impl DiTWeights {
    /// Load DiT weights from safetensors
    ///
    /// PyTorch keys:
    /// - cfm.estimator.cond_embedder.weight
    /// - cfm.estimator.cond_projection.weight/bias
    /// - cfm.estimator.cond_x_merge_linear.weight/bias
    /// - cfm.estimator.conv1.weight/bias
    /// - cfm.estimator.conv2.weight/bias
    /// - cfm.estimator.transformer.layers.{n}.attention.wqkv.weight
    /// - cfm.estimator.transformer.layers.{n}.attention.wo.weight
    /// - cfm.estimator.transformer.layers.{n}.feed_forward.w1.weight
    /// - cfm.estimator.transformer.layers.{n}.feed_forward.w2.weight
    /// - cfm.estimator.transformer.layers.{n}.feed_forward.w3.weight
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        num_layers: usize,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "cfm.estimator";

        // Conditioning embedder (no bias in checkpoint)
        let cond_embedder_weight = tensors
            .get(&format!("{}.cond_embedder.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("cond_embedder.weight not found"))?
            .t()?
            .contiguous()?;
        let cond_embedder = Linear::new(cond_embedder_weight, None);

        // Conditioning projection
        let cond_projection = load_linear(
            tensors,
            &format!("{}.cond_projection.weight", prefix),
            Some(&format!("{}.cond_projection.bias", prefix)),
            true,
        )?;

        // Merge linear
        let cond_x_merge_linear = load_linear(
            tensors,
            &format!("{}.cond_x_merge_linear.weight", prefix),
            Some(&format!("{}.cond_x_merge_linear.bias", prefix)),
            true,
        )?;

        // Conv layers
        let conv1_weight = load_conv1d_weight(tensors, &format!("{}.conv1.weight", prefix))?;
        let conv1_bias = tensors.get(&format!("{}.conv1.bias", prefix)).cloned();
        let conv2_weight = load_conv1d_weight(tensors, &format!("{}.conv2.weight", prefix))?;
        let conv2_bias = tensors.get(&format!("{}.conv2.bias", prefix)).cloned();

        // Transformer layers
        let mut transformer_layers = Vec::new();
        for i in 0..num_layers {
            let layer = DiTLayerWeights::load(tensors, i, hidden_dim, device)?;
            transformer_layers.push(layer);
        }

        Ok(Self {
            cond_embedder,
            cond_projection,
            cond_x_merge_linear,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            transformer_layers,
        })
    }
}

impl DiTLayerWeights {
    /// Load a single transformer layer
    ///
    /// AdaLayerNorm keys: {prefix}.attention_norm.norm.weight,
    ///   {prefix}.attention_norm.project_layer.weight/bias
    /// Skip connection: {prefix}.skip_in_linear.weight/bias
    fn load(
        tensors: &HashMap<String, Tensor>,
        layer_idx: usize,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("cfm.estimator.transformer.layers.{}", layer_idx);

        // Attention weights - wqkv is fused [3*hidden_dim, hidden_dim]
        let wqkv = load_linear(
            tensors,
            &format!("{}.attention.wqkv.weight", prefix),
            None,
            true,
        )?;

        let wo = load_linear(
            tensors,
            &format!("{}.attention.wo.weight", prefix),
            None,
            true,
        )?;

        // FFN weights
        let w1 = load_linear(
            tensors,
            &format!("{}.feed_forward.w1.weight", prefix),
            None,
            true,
        )?;

        let w2 = load_linear(
            tensors,
            &format!("{}.feed_forward.w2.weight", prefix),
            None,
            true,
        )?;

        let w3 = tensors
            .get(&format!("{}.feed_forward.w3.weight", prefix))
            .map(|w| Linear::new(w.t().unwrap().contiguous().unwrap(), None));

        // AdaLayerNorm for attention: norm.weight + project_layer.weight/bias
        let attn_norm = {
            let norm_key = format!("{}.attention_norm.norm.weight", prefix);
            let proj_key = format!("{}.attention_norm.project_layer.weight", prefix);
            let proj_bias_key = format!("{}.attention_norm.project_layer.bias", prefix);
            if tensors.contains_key(&norm_key) && tensors.contains_key(&proj_key) {
                let norm = load_layer_norm(tensors, &norm_key, None, 1e-5, hidden_dim, device)?;
                let project_weight = load_linear(tensors, &proj_key, Some(&proj_bias_key), false)?;
                Some(AdaLayerNormWeights { norm, project_weight })
            } else {
                None
            }
        };

        // AdaLayerNorm for FFN: norm.weight + project_layer.weight/bias
        let ffn_norm = {
            let norm_key = format!("{}.ffn_norm.norm.weight", prefix);
            let proj_key = format!("{}.ffn_norm.project_layer.weight", prefix);
            let proj_bias_key = format!("{}.ffn_norm.project_layer.bias", prefix);
            if tensors.contains_key(&norm_key) && tensors.contains_key(&proj_key) {
                let norm = load_layer_norm(tensors, &norm_key, None, 1e-5, hidden_dim, device)?;
                let project_weight = load_linear(tensors, &proj_key, Some(&proj_bias_key), false)?;
                Some(AdaLayerNormWeights { norm, project_weight })
            } else {
                None
            }
        };

        // UViT skip connection linear
        let skip_in_key = format!("{}.skip_in_linear.weight", prefix);
        let skip_in_bias_key = format!("{}.skip_in_linear.bias", prefix);
        let skip_in_linear = if tensors.contains_key(&skip_in_key) {
            Some(load_linear(tensors, &skip_in_key, Some(&skip_in_bias_key), false)?)
        } else {
            None
        };

        Ok(Self {
            wqkv,
            wo,
            w1,
            w2,
            w3,
            attn_norm,
            ffn_norm,
            skip_in_linear,
        })
    }
}

/// Length Regulator weights
pub struct LengthRegulatorWeights {
    pub mask_token: Tensor,
    pub embedding: Option<Tensor>,
    pub content_in_proj: Linear,
    pub conv_blocks: Vec<ConvBlockWeights>,
}

/// Conv block weights for length regulator
pub struct ConvBlockWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub norm_weight: Option<Tensor>,
    pub norm_bias: Option<Tensor>,
}

impl LengthRegulatorWeights {
    /// Load length regulator weights
    ///
    /// PyTorch keys:
    /// - length_regulator.mask_token
    /// - length_regulator.embedding.weight
    /// - length_regulator.content_in_proj.weight/bias
    /// - length_regulator.model.{n}.weight/bias (conv blocks)
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Self> {
        let prefix = "length_regulator";

        // Mask token
        let mask_token = tensors
            .get(&format!("{}.mask_token", prefix))
            .ok_or_else(|| anyhow::anyhow!("mask_token not found"))?
            .clone();

        // Embedding (optional discrete content embedding)
        let embedding = tensors
            .get(&format!("{}.embedding.weight", prefix))
            .cloned();

        // Content input projection
        let content_in_proj = load_linear(
            tensors,
            &format!("{}.content_in_proj.weight", prefix),
            Some(&format!("{}.content_in_proj.bias", prefix)),
            true,
        )?;

        // Conv blocks (model.0, model.2, model.4, model.6 - even indices are conv, odd are norm)
        let mut conv_blocks = Vec::new();
        for i in [0, 2, 4, 6] {
            let weight_key = format!("{}.model.{}.weight", prefix, i);
            if let Some(weight) = tensors.get(&weight_key) {
                let bias = tensors.get(&format!("{}.model.{}.bias", prefix, i)).cloned();
                // Norm is at i+1
                let norm_weight = tensors.get(&format!("{}.model.{}.weight", prefix, i + 1)).cloned();
                let norm_bias = tensors.get(&format!("{}.model.{}.bias", prefix, i + 1)).cloned();

                conv_blocks.push(ConvBlockWeights {
                    weight: weight.clone(),
                    bias,
                    norm_weight,
                    norm_bias,
                });
            }
        }

        Ok(Self {
            mask_token,
            embedding,
            content_in_proj,
            conv_blocks,
        })
    }
}

/// GPT layer projection weights (from GPT hidden dim to DiT hidden dim)
pub struct GptLayerWeights {
    pub layers: Vec<Linear>,
}

impl GptLayerWeights {
    /// Load gpt_layer weights (projection from 1280 -> 512)
    ///
    /// PyTorch keys:
    /// - gpt_layer.0.weight/bias (Linear 1280 -> 256)
    /// - gpt_layer.1.weight/bias (Linear 256 -> 128)
    /// - gpt_layer.2.weight/bias (Linear 128 -> 1024)
    pub fn load(tensors: &HashMap<String, Tensor>) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..3 {
            let weight_key = format!("gpt_layer.{}.weight", i);
            let bias_key = format!("gpt_layer.{}.bias", i);

            if let Some(weight) = tensors.get(&weight_key) {
                let bias = tensors.get(&bias_key).cloned();
                // Transpose for candle (PyTorch stores as [out_features, in_features])
                let weight = weight.t()?.contiguous()?;
                layers.push(Linear::new(weight, bias));
            }
        }

        Ok(Self { layers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_linear_with_transpose() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let bias = Tensor::zeros((64,), DType::F32, &device).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("test.weight".to_string(), weight);
        tensors.insert("test.bias".to_string(), bias);

        let linear = load_linear(&tensors, "test.weight", Some("test.bias"), true).unwrap();

        // After transpose, shape should be [128, 64] for candle Linear
        let shape = linear.weight().dims();
        assert_eq!(shape, &[128, 64]);
    }
}
