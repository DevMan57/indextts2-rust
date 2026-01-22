//! Weight loading utilities for GPT model
//!
//! Handles loading safetensors checkpoints and mapping PyTorch keys to Rust model structure.

#![allow(missing_docs)]

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear};
use std::collections::HashMap;
use std::path::Path;

/// Load tensors from safetensors file
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();

    if !path.exists() {
        bail!("Safetensors file not found: {:?}", path);
    }

    let tensors = candle_core::safetensors::load(path, device)?;
    Ok(tensors)
}

/// Load embedding tensor from safetensors
///
/// Handles the case where the embedding might have a different vocab size
/// than expected (e.g., 12001 vs 12000).
pub fn load_embedding(
    tensors: &HashMap<String, Tensor>,
    key: &str,
    expected_vocab_size: Option<usize>,
    dim: usize,
    _device: &Device,
) -> Result<Tensor> {
    let tensor = tensors
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Key not found: {}", key))?;

    let (vocab_size, embedding_dim) = tensor.dims2()?;

    if embedding_dim != dim {
        bail!(
            "Embedding dimension mismatch for {}: expected {}, got {}",
            key, dim, embedding_dim
        );
    }

    // If expected vocab size is provided and differs, truncate or warn
    if let Some(expected) = expected_vocab_size {
        if vocab_size != expected {
            // Just use the tensor as-is, the model will handle it
            eprintln!(
                "Warning: {} vocab size {} differs from expected {}",
                key, vocab_size, expected
            );
        }
    }

    Ok(tensor.clone())
}

/// Load a Linear layer from safetensors
pub fn load_linear(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
) -> Result<Linear> {
    let weight = tensors
        .get(weight_key)
        .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", weight_key))?
        .clone();

    let bias = if let Some(bk) = bias_key {
        tensors.get(bk).cloned()
    } else {
        None
    };

    Ok(Linear::new(weight, bias))
}

/// Load a LayerNorm from safetensors
pub fn load_layer_norm(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
    eps: f64,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = tensors
        .get(weight_key)
        .ok_or_else(|| anyhow::anyhow!("LayerNorm weight not found: {}", weight_key))?
        .clone();

    let dim = weight.dim(0)?;

    let bias = if let Some(bk) = bias_key {
        tensors.get(bk).cloned()
    } else {
        None
    };

    // If no bias, create zeros
    let bias = match bias {
        Some(b) => b,
        None => {
            tracing::warn!(
                "[GPT] Missing LayerNorm bias for weight '{}', using zeros initialization",
                weight_key
            );
            Tensor::zeros((dim,), DType::F32, device)?
        }
    };

    Ok(LayerNorm::new(weight, bias, eps))
}

/// Split GPT-2 combined c_attn weight into Q, K, V projections
///
/// GPT-2 uses a combined QKV projection stored as:
/// - c_attn.weight: [in_features, 3 * hidden_size]
/// - c_attn.bias: [3 * hidden_size]
///
/// This splits it into separate Q, K, V Linear layers.
pub fn split_qkv(
    tensors: &HashMap<String, Tensor>,
    c_attn_weight_key: &str,
    c_attn_bias_key: Option<&str>,
    hidden_size: usize,
) -> Result<(Linear, Linear, Linear)> {
    let c_attn_weight = tensors
        .get(c_attn_weight_key)
        .ok_or_else(|| anyhow::anyhow!("c_attn weight not found: {}", c_attn_weight_key))?;

    let (_in_features, out_features) = c_attn_weight.dims2()?;

    if out_features != 3 * hidden_size {
        bail!(
            "c_attn output size mismatch: expected {}, got {}",
            3 * hidden_size,
            out_features
        );
    }

    // Split weight: [in_features, 3*hidden_size] -> 3x [in_features, hidden_size]
    // Then transpose to [hidden_size, in_features] for candle Linear
    let q_weight = c_attn_weight.i((.., 0..hidden_size))?.t()?.contiguous()?;
    let k_weight = c_attn_weight.i((.., hidden_size..2*hidden_size))?.t()?.contiguous()?;
    let v_weight = c_attn_weight.i((.., 2*hidden_size..3*hidden_size))?.t()?.contiguous()?;

    let (q_bias, k_bias, v_bias) = if let Some(bias_key) = c_attn_bias_key {
        if let Some(c_attn_bias) = tensors.get(bias_key) {
            let q_bias = c_attn_bias.i(0..hidden_size)?.contiguous()?;
            let k_bias = c_attn_bias.i(hidden_size..2*hidden_size)?.contiguous()?;
            let v_bias = c_attn_bias.i(2*hidden_size..3*hidden_size)?.contiguous()?;
            (Some(q_bias), Some(k_bias), Some(v_bias))
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    let q_proj = Linear::new(q_weight, q_bias);
    let k_proj = Linear::new(k_weight, k_bias);
    let v_proj = Linear::new(v_weight, v_bias);

    Ok((q_proj, k_proj, v_proj))
}

/// Load GPT-2 decoder layer weights
pub struct Gpt2LayerWeights {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub attn_ln: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub ffn_ln: LayerNorm,
}

impl Gpt2LayerWeights {
    /// Load layer weights from safetensors
    ///
    /// PyTorch key format:
    /// - gpt.h.{layer_idx}.attn.c_attn.weight/bias (combined QKV)
    /// - gpt.h.{layer_idx}.attn.c_proj.weight/bias
    /// - gpt.h.{layer_idx}.ln_1.weight/bias (pre-attention LayerNorm)
    /// - gpt.h.{layer_idx}.mlp.c_fc.weight/bias
    /// - gpt.h.{layer_idx}.mlp.c_proj.weight/bias
    /// - gpt.h.{layer_idx}.ln_2.weight/bias (pre-FFN LayerNorm)
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        layer_idx: usize,
        hidden_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("gpt.h.{}", layer_idx);

        // Split combined c_attn into Q, K, V
        let (q_proj, k_proj, v_proj) = split_qkv(
            tensors,
            &format!("{}.attn.c_attn.weight", prefix),
            Some(&format!("{}.attn.c_attn.bias", prefix)),
            hidden_size,
        )?;

        // Output projection - need to transpose for candle
        let out_weight = tensors
            .get(&format!("{}.attn.c_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("c_proj weight not found"))?
            .t()?
            .contiguous()?;
        let out_bias = tensors.get(&format!("{}.attn.c_proj.bias", prefix)).cloned();
        let out_proj = Linear::new(out_weight, out_bias);

        // Pre-attention LayerNorm (ln_1)
        let attn_ln = load_layer_norm(
            tensors,
            &format!("{}.ln_1.weight", prefix),
            Some(&format!("{}.ln_1.bias", prefix)),
            1e-5,
            device,
        )?;

        // FFN layers - need to transpose for candle
        let fc1_weight = tensors
            .get(&format!("{}.mlp.c_fc.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("c_fc weight not found"))?
            .t()?
            .contiguous()?;
        let fc1_bias = tensors.get(&format!("{}.mlp.c_fc.bias", prefix)).cloned();
        let fc1 = Linear::new(fc1_weight, fc1_bias);

        let fc2_weight = tensors
            .get(&format!("{}.mlp.c_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("c_proj weight not found"))?
            .t()?
            .contiguous()?;
        let fc2_bias = tensors.get(&format!("{}.mlp.c_proj.bias", prefix)).cloned();
        let fc2 = Linear::new(fc2_weight, fc2_bias);

        // Pre-FFN LayerNorm (ln_2)
        let ffn_ln = load_layer_norm(
            tensors,
            &format!("{}.ln_2.weight", prefix),
            Some(&format!("{}.ln_2.bias", prefix)),
            1e-5,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            attn_ln,
            fc1,
            fc2,
            ffn_ln,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_qkv_shapes() {
        // Create mock tensors
        let device = Device::Cpu;
        let hidden_size = 64;

        // c_attn.weight shape in PyTorch: [in_features, 3*hidden_size]
        let c_attn_weight = Tensor::randn(0.0f32, 1.0, (hidden_size, 3 * hidden_size), &device).unwrap();
        let c_attn_bias = Tensor::zeros((3 * hidden_size,), DType::F32, &device).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("attn.c_attn.weight".to_string(), c_attn_weight);
        tensors.insert("attn.c_attn.bias".to_string(), c_attn_bias);

        let (q, k, v) = split_qkv(
            &tensors,
            "attn.c_attn.weight",
            Some("attn.c_attn.bias"),
            hidden_size,
        ).unwrap();

        // Verify output shapes - candle Linear expects [out_features, in_features]
        let q_shape = q.weight().dims();
        let k_shape = k.weight().dims();
        let v_shape = v.weight().dims();

        assert_eq!(q_shape, &[hidden_size, hidden_size]);
        assert_eq!(k_shape, &[hidden_size, hidden_size]);
        assert_eq!(v_shape, &[hidden_size, hidden_size]);
    }
}
