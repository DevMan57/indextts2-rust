//! VocosBackbone encoder for RepCodec/MaskGCT semantic codec
//!
//! A 1D convolutional encoder with 12 ConvNeXtV2 blocks, used as the encoder
//! component in the MaskGCT semantic codec pipeline. Takes W2V-BERT hidden
//! states (1024-dim) and produces encoded representations (1024-dim) that feed
//! into the vector quantizer.
//!
//! Architecture:
//! - Embed: Conv1d(1024, 384, kernel=7, padding=3)
//! - LayerNorm(384)
//! - 12x ConvNeXtBlock(384, intermediate=2048, kernel=7)
//! - Final LayerNorm(384)
//! - Output projection: Linear(384, 1024)
//!
//! Weight key prefix: `encoder.0.*` for the backbone, `encoder.1.*` for output projection.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration constants
// ---------------------------------------------------------------------------

/// ConvNeXt internal dimension
const VOCOS_DIM: usize = 384;
/// ConvNeXt MLP expansion dimension
const VOCOS_INTERMEDIATE_DIM: usize = 2048;
/// Number of ConvNeXt blocks
const VOCOS_NUM_LAYERS: usize = 12;
/// Input channels from W2V-BERT
const INPUT_CHANNELS: usize = 1024;
/// Output channels (via final linear projection)
const OUTPUT_CHANNELS: usize = 1024;
/// Convolution kernel size for embed and depthwise convolutions
const CONV_KERNEL_SIZE: usize = 7;
/// Padding for kernel=7 convolutions
const CONV_PADDING: usize = 3;
/// LayerNorm epsilon
const LN_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Helper: load a tensor from the map or bail
// ---------------------------------------------------------------------------

fn get_tensor(tensors: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    tensors
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("VocosBackbone: missing tensor key '{}'", key))
}

// ---------------------------------------------------------------------------
// Helper: load LayerNorm from tensor map
// ---------------------------------------------------------------------------

fn load_layer_norm(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: &str,
) -> Result<LayerNorm> {
    let weight = get_tensor(tensors, weight_key)?;
    let bias = get_tensor(tensors, bias_key)?;
    Ok(LayerNorm::new(weight, bias, LN_EPS))
}

// ---------------------------------------------------------------------------
// Conv1d wrapper (stores weight + optional bias, mirrors bigvgan pattern)
// ---------------------------------------------------------------------------

/// Minimal Conv1d that stores weight/bias tensors and delegates to
/// `Tensor::conv1d`. Supports grouped convolutions for depthwise conv.
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
    groups: usize,
}

impl Conv1d {
    /// Load a Conv1d from the tensor map.
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        padding: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = get_tensor(tensors, &format!("{}.weight", prefix))?;
        let bias = tensors.get(&format!("{}.bias", prefix)).cloned();
        Ok(Self {
            weight,
            bias,
            padding,
            stride,
            groups,
        })
    }

    /// Forward pass: conv1d with optional bias addition.
    /// Input shape: (B, C_in, T), output shape: (B, C_out, T').
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, self.stride, 1, self.groups)?;
        if let Some(ref bias) = self.bias {
            // bias: (C_out,) -> (1, C_out, 1) for broadcast
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            y.broadcast_add(&bias).map_err(Into::into)
        } else {
            Ok(y)
        }
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock
// ---------------------------------------------------------------------------

/// A single ConvNeXtV2 block.
///
/// ```text
/// residual = x                          // (B, 384, T)
/// x = depthwise_conv1d(x)              // (B, 384, T) groups=384
/// x = x.transpose -> layer_norm -> x.transpose
/// x = pointwise_conv1(x) -> GELU       // (B, T, 2048)
/// x = pointwise_conv2(x)               // (B, T, 384)
/// x = gamma * x                        // layer scale
/// x = x.transpose + residual           // (B, 384, T)
/// ```
struct ConvNeXtBlock {
    /// Depthwise Conv1d(384, 384, kernel=7, padding=3, groups=384)
    dwconv: Conv1d,
    /// LayerNorm(384) applied in channels-last format
    norm: LayerNorm,
    /// Pointwise expansion Linear(384, 2048)
    pwconv1: Linear,
    /// Pointwise projection Linear(2048, 384)
    pwconv2: Linear,
    /// Layer scale parameter, shape (384,)
    gamma: Tensor,
}

impl ConvNeXtBlock {
    /// Load a ConvNeXtBlock from the tensor map.
    ///
    /// Keys expected under `prefix`:
    /// - `{prefix}.dwconv.weight` [384, 1, 7]
    /// - `{prefix}.dwconv.bias` [384]
    /// - `{prefix}.norm.weight` [384]
    /// - `{prefix}.norm.bias` [384]
    /// - `{prefix}.pwconv1.weight` [2048, 384]
    /// - `{prefix}.pwconv1.bias` [2048]
    /// - `{prefix}.pwconv2.weight` [384, 2048]
    /// - `{prefix}.pwconv2.bias` [384]
    /// - `{prefix}.gamma` [384]
    fn from_tensors(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let dwconv = Conv1d::from_tensors(
            tensors,
            &format!("{}.dwconv", prefix),
            CONV_PADDING,
            1,
            VOCOS_DIM, // groups=384 for depthwise
        )?;

        let norm = load_layer_norm(
            tensors,
            &format!("{}.norm.weight", prefix),
            &format!("{}.norm.bias", prefix),
        )?;

        // Pointwise linear layers
        let pwconv1_w = get_tensor(tensors, &format!("{}.pwconv1.weight", prefix))?;
        let pwconv1_b = get_tensor(tensors, &format!("{}.pwconv1.bias", prefix))?;
        let pwconv1 = Linear::new(pwconv1_w, Some(pwconv1_b));

        let pwconv2_w = get_tensor(tensors, &format!("{}.pwconv2.weight", prefix))?;
        let pwconv2_b = get_tensor(tensors, &format!("{}.pwconv2.bias", prefix))?;
        let pwconv2 = Linear::new(pwconv2_w, Some(pwconv2_b));

        let gamma = get_tensor(tensors, &format!("{}.gamma", prefix))?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Forward pass with residual connection.
    ///
    /// Input/output: (B, C=384, T) in channels-first layout.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise convolution: (B, 384, T) -> (B, 384, T)
        let mut y = self.dwconv.forward(x)?;

        // Transpose to channels-last for LayerNorm: (B, 384, T) -> (B, T, 384)
        y = y.transpose(1, 2)?;
        y = self.norm.forward(&y)?;

        // Pointwise expansion + GELU: (B, T, 384) -> (B, T, 2048)
        y = self.pwconv1.forward(&y)?;
        y = y.gelu_erf()?;

        // Pointwise projection: (B, T, 2048) -> (B, T, 384)
        y = self.pwconv2.forward(&y)?;

        // Layer scale: element-wise multiply by gamma (384,)
        y = y.broadcast_mul(&self.gamma)?;

        // Transpose back to channels-first: (B, T, 384) -> (B, 384, T)
        y = y.transpose(1, 2)?;

        // Residual connection
        (residual + y).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// VocosBackbone
// ---------------------------------------------------------------------------

/// VocosBackbone: 1D convolutional encoder with ConvNeXtV2 blocks.
///
/// This is the encoder component of the RepCodec/MaskGCT semantic codec.
/// It transforms W2V-BERT hidden states into a compact representation
/// suitable for vector quantization.
///
/// # Forward pass
///
/// 1. Input `(B, T, 1024)` is transposed to channels-first `(B, 1024, T)`.
/// 2. Embedding Conv1d projects 1024 -> 384 channels.
/// 3. LayerNorm in channels-last format.
/// 4. 12 ConvNeXtV2 blocks with residual connections.
/// 5. Final LayerNorm.
/// 6. Linear output projection 384 -> 1024.
/// 7. Output: `(B, T, 1024)`.
///
/// # Weight keys
///
/// Backbone weights live under `encoder.0.*`, the output linear under `encoder.1.*`.
pub struct VocosBackbone {
    /// Embedding convolution: Conv1d(1024, 384, kernel=7, padding=3)
    embed: Conv1d,
    /// Post-embed LayerNorm(384)
    norm: LayerNorm,
    /// 12 ConvNeXtV2 blocks
    convnext: Vec<ConvNeXtBlock>,
    /// Final LayerNorm(384) before output projection
    final_layer_norm: LayerNorm,
    /// Output projection: Linear(384, 1024)
    output_proj: Linear,
}

impl VocosBackbone {
    /// Load VocosBackbone weights from a tensor map.
    ///
    /// The tensor map should contain keys starting with `encoder.0.` for the
    /// backbone layers and `encoder.1.` for the output projection. Typically
    /// obtained via `candle_core::safetensors::load(path, device)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any required tensor key is missing from the map.
    pub fn load(tensors: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let _ = device; // device is implicit in the tensors

        // --- Embedding Conv1d ---
        let embed = Conv1d::from_tensors(
            tensors,
            "encoder.0.embed",
            CONV_PADDING,
            1, // stride
            1, // groups (normal convolution)
        )?;

        // --- Post-embed LayerNorm ---
        let norm = load_layer_norm(
            tensors,
            "encoder.0.norm.weight",
            "encoder.0.norm.bias",
        )?;

        // --- ConvNeXt blocks ---
        let mut convnext = Vec::with_capacity(VOCOS_NUM_LAYERS);
        for i in 0..VOCOS_NUM_LAYERS {
            let prefix = format!("encoder.0.convnext.{}", i);
            convnext.push(ConvNeXtBlock::from_tensors(tensors, &prefix)?);
        }

        // --- Final LayerNorm ---
        let final_layer_norm = load_layer_norm(
            tensors,
            "encoder.0.final_layer_norm.weight",
            "encoder.0.final_layer_norm.bias",
        )?;

        // --- Output projection (encoder.1) ---
        let out_w = get_tensor(tensors, "encoder.1.weight")?;
        let out_b = get_tensor(tensors, "encoder.1.bias")?;
        let output_proj = Linear::new(out_w, Some(out_b));

        Ok(Self {
            embed,
            norm,
            convnext,
            final_layer_norm,
            output_proj,
        })
    }

    /// Forward pass through the VocosBackbone encoder.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `(B, T, 1024)` from W2V-BERT.
    ///
    /// # Returns
    ///
    /// Encoded tensor of shape `(B, T, 1024)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, 1024) -> transpose to channels-first for Conv1d
        let mut h = x.transpose(1, 2)?; // (B, 1024, T)

        // Embedding convolution: (B, 1024, T) -> (B, 384, T)
        h = self.embed.forward(&h)?;

        // LayerNorm in channels-last: transpose -> norm -> transpose
        h = h.transpose(1, 2)?; // (B, T, 384)
        h = self.norm.forward(&h)?;
        h = h.transpose(1, 2)?; // (B, 384, T)

        // 12 ConvNeXt blocks with residual connections
        for block in &self.convnext {
            h = block.forward(&h)?;
        }

        // Final LayerNorm in channels-last
        h = h.transpose(1, 2)?; // (B, T, 384)
        h = self.final_layer_norm.forward(&h)?;

        // Output projection: (B, T, 384) -> (B, T, 1024)
        let out = self.output_proj.forward(&h)?;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Build a minimal set of random tensors that match the expected checkpoint
    /// key layout so we can test `VocosBackbone::load` and `forward` on CPU.
    fn make_test_tensors(device: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let f = |shape: &[usize]| Tensor::randn(0.0f32, 0.02, shape, device).unwrap();

        // embed
        m.insert("encoder.0.embed.weight".into(), f(&[VOCOS_DIM, INPUT_CHANNELS, CONV_KERNEL_SIZE]));
        m.insert("encoder.0.embed.bias".into(), f(&[VOCOS_DIM]));

        // norm
        m.insert("encoder.0.norm.weight".into(), Tensor::ones((VOCOS_DIM,), DType::F32, device).unwrap());
        m.insert("encoder.0.norm.bias".into(), Tensor::zeros((VOCOS_DIM,), DType::F32, device).unwrap());

        // convnext blocks
        for i in 0..VOCOS_NUM_LAYERS {
            let p = format!("encoder.0.convnext.{}", i);
            // dwconv: depthwise, weight shape [384, 1, 7]
            m.insert(format!("{}.dwconv.weight", p), f(&[VOCOS_DIM, 1, CONV_KERNEL_SIZE]));
            m.insert(format!("{}.dwconv.bias", p), f(&[VOCOS_DIM]));
            // norm
            m.insert(format!("{}.norm.weight", p), Tensor::ones((VOCOS_DIM,), DType::F32, device).unwrap());
            m.insert(format!("{}.norm.bias", p), Tensor::zeros((VOCOS_DIM,), DType::F32, device).unwrap());
            // pwconv1: [2048, 384]
            m.insert(format!("{}.pwconv1.weight", p), f(&[VOCOS_INTERMEDIATE_DIM, VOCOS_DIM]));
            m.insert(format!("{}.pwconv1.bias", p), f(&[VOCOS_INTERMEDIATE_DIM]));
            // pwconv2: [384, 2048]
            m.insert(format!("{}.pwconv2.weight", p), f(&[VOCOS_DIM, VOCOS_INTERMEDIATE_DIM]));
            m.insert(format!("{}.pwconv2.bias", p), f(&[VOCOS_DIM]));
            // gamma
            m.insert(format!("{}.gamma", p), f(&[VOCOS_DIM]));
        }

        // final_layer_norm
        m.insert("encoder.0.final_layer_norm.weight".into(), Tensor::ones((VOCOS_DIM,), DType::F32, device).unwrap());
        m.insert("encoder.0.final_layer_norm.bias".into(), Tensor::zeros((VOCOS_DIM,), DType::F32, device).unwrap());

        // output projection (encoder.1)
        m.insert("encoder.1.weight".into(), f(&[OUTPUT_CHANNELS, VOCOS_DIM]));
        m.insert("encoder.1.bias".into(), f(&[OUTPUT_CHANNELS]));

        m
    }

    #[test]
    fn test_vocos_load_and_forward_shape() {
        let device = Device::Cpu;
        let tensors = make_test_tensors(&device);
        let backbone = VocosBackbone::load(&tensors, &device).unwrap();

        // (batch=1, time=50, channels=1024)
        let x = Tensor::randn(0.0f32, 1.0, (1, 50, INPUT_CHANNELS), &device).unwrap();
        let out = backbone.forward(&x).unwrap();

        let (b, t, c) = out.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(t, 50);
        assert_eq!(c, OUTPUT_CHANNELS);
    }

    #[test]
    fn test_vocos_batch() {
        let device = Device::Cpu;
        let tensors = make_test_tensors(&device);
        let backbone = VocosBackbone::load(&tensors, &device).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, 20, INPUT_CHANNELS), &device).unwrap();
        let out = backbone.forward(&x).unwrap();

        let (b, t, c) = out.dims3().unwrap();
        assert_eq!(b, 4);
        assert_eq!(t, 20);
        assert_eq!(c, OUTPUT_CHANNELS);
    }

    #[test]
    fn test_vocos_missing_key_errors() {
        let device = Device::Cpu;
        let tensors: HashMap<String, Tensor> = HashMap::new();
        let result = VocosBackbone::load(&tensors, &device);
        assert!(result.is_err());
    }
}
