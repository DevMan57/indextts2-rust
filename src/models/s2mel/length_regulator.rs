//! Length regulator for mel code expansion
//!
//! Processes continuous features from GPT and expands them to target length for synthesis.
//! The input is 1024-dim continuous features (after gpt_layer projection).
//!
//! Architecture (from checkpoint):
//! - content_in_proj: Linear [512, 1024] - projects input to 512-dim
//! - model: Conv1d blocks for processing
//! - Duration prediction for length expansion

#![allow(missing_docs)]

use anyhow::Result;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{Linear, Module, LayerNorm, VarBuilder};
use std::path::Path;

/// Conv1d block for duration prediction
struct ConvBlock {
    conv_weight: Tensor,
    conv_bias: Tensor,
    layer_norm: LayerNorm,
    kernel_size: usize,
}

impl ConvBlock {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, device: &Device) -> Result<Self> {
        let conv_weight = Tensor::randn(
            0.0f32,
            0.02,
            (out_channels, in_channels, kernel_size),
            device,
        )?;
        let conv_bias = Tensor::zeros((out_channels,), DType::F32, device)?;

        let ln_w = Tensor::ones((out_channels,), DType::F32, device)?;
        let ln_b = Tensor::zeros((out_channels,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_w, ln_b, 1e-5);

        Ok(Self {
            conv_weight,
            conv_bias,
            layer_norm,
            kernel_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, channels, seq)
        let padding = self.kernel_size / 2;
        let x = x.conv1d(&self.conv_weight, padding, 1, 1, 1)?;
        let bias = self.conv_bias.unsqueeze(0)?.unsqueeze(2)?;
        let x = x.broadcast_add(&bias)?;

        // Transpose for layer norm: (batch, seq, channels)
        let x = x.transpose(1, 2)?;
        let x = self.layer_norm.forward(&x)?;
        let x = x.relu()?;

        // Transpose back: (batch, channels, seq)
        x.transpose(1, 2).map_err(Into::into)
    }
}

/// Duration predictor network
struct DurationPredictor {
    conv1: ConvBlock,
    conv2: ConvBlock,
    output_layer: Linear,
}

impl DurationPredictor {
    fn new(channels: usize, device: &Device) -> Result<Self> {
        let conv1 = ConvBlock::new(channels, channels, 3, device)?;
        let conv2 = ConvBlock::new(channels, channels, 3, device)?;

        let w = Tensor::randn(0.0f32, 0.02, (1, channels), device)?;
        let b = Tensor::zeros((1,), DType::F32, device)?;
        let output_layer = Linear::new(w, Some(b));

        Ok(Self {
            conv1,
            conv2,
            output_layer,
        })
    }

    /// Predict duration for each input frame
    /// Input: (batch, channels, seq)
    /// Output: (batch, seq) - durations
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.conv2.forward(&x)?;

        // Transpose and project: (batch, seq, channels) -> (batch, seq, 1)
        let x = x.transpose(1, 2)?;
        let x = self.output_layer.forward(&x)?;

        // Squeeze and apply softplus for positive durations
        let x = x.squeeze(2)?;
        softplus(&x)
    }
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(x)?;
    let exp_x = x.exp()?;
    (one + exp_x)?.log().map_err(Into::into)
}

/// Mish activation: x * tanh(softplus(x))
/// Unlike ReLU, Mish preserves negative values and has smoother gradients
fn mish(x: &Tensor) -> Result<Tensor> {
    let sp = softplus(x)?;
    let tanh_sp = sp.tanh()?;
    (x * tanh_sp).map_err(Into::into)
}

/// Length regulator configuration
pub struct LengthRegulatorConfig {
    pub channels: usize,
    pub in_channels: usize,
}

impl Default for LengthRegulatorConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            in_channels: 1024, // Input from gpt_layer projection
        }
    }
}

/// GPT layer projection (from s2mel checkpoint)
/// Projects GPT mel embeddings (1280-dim) to length regulator input (1024-dim)
pub struct GptLayerProjection {
    layer0: Linear, // [256, 1280]
    layer1: Linear, // [128, 256]
    layer2: Linear, // [1024, 128]
}

impl GptLayerProjection {
    fn new(device: &Device) -> Result<Self> {
        // Random init for now
        let w0 = Tensor::randn(0.0f32, 0.02, (256, 1280), device)?;
        let b0 = Tensor::zeros((256,), DType::F32, device)?;
        let w1 = Tensor::randn(0.0f32, 0.02, (128, 256), device)?;
        let b1 = Tensor::zeros((128,), DType::F32, device)?;
        let w2 = Tensor::randn(0.0f32, 0.02, (1024, 128), device)?;
        let b2 = Tensor::zeros((1024,), DType::F32, device)?;

        Ok(Self {
            layer0: Linear::new(w0, Some(b0)),
            layer1: Linear::new(w1, Some(b1)),
            layer2: Linear::new(w2, Some(b2)),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, seq, 1280)
        // NOTE: No activation functions between layers - this is a simple MLP projection
        // The Python checkpoint shows only Linear layers (0, 1, 2) with no intermediate activations
        let x = self.layer0.forward(x)?; // -> (batch, seq, 256)
        let x = self.layer1.forward(&x)?; // -> (batch, seq, 128)
        let x = self.layer2.forward(&x)?; // -> (batch, seq, 1024)
        Ok(x)
    }
}

/// Conv1d + GroupNorm block (matching Python's architecture)
/// Python uses: nn.Conv1d -> nn.GroupNorm(groups=1, channels) -> nn.Mish()
/// GroupNorm with groups=1 normalizes across all channels for each spatial position
struct ConvGNBlock {
    conv_weight: Tensor,
    conv_bias: Tensor,
    gn_weight: Tensor,
    gn_bias: Tensor,
    kernel_size: usize,
    num_groups: usize,
}

impl ConvGNBlock {
    fn new(channels: usize, kernel_size: usize, num_groups: usize, device: &Device) -> Result<Self> {
        let conv_weight = Tensor::randn(
            0.0f32,
            0.02,
            (channels, channels, kernel_size),
            device,
        )?;
        let conv_bias = Tensor::zeros((channels,), DType::F32, device)?;
        let gn_weight = Tensor::ones((channels,), DType::F32, device)?;
        let gn_bias = Tensor::zeros((channels,), DType::F32, device)?;

        Ok(Self {
            conv_weight,
            conv_bias,
            gn_weight,
            gn_bias,
            kernel_size,
            num_groups,
        })
    }

    fn forward(&self, x: &Tensor, debug_idx: usize) -> Result<Tensor> {
        // x: (batch, channels, seq)
        let padding = self.kernel_size / 2;
        let x = x.conv1d(&self.conv_weight, padding, 1, 1, 1)?;
        let bias = self.conv_bias.unsqueeze(0)?.unsqueeze(2)?;
        let x = x.broadcast_add(&bias)?;

        if debug_idx == 0 {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG ConvGN: after conv+bias: mean={:.6}, var={:.6}", m, v);
        }

        // GroupNorm: normalize across channel groups for each spatial position
        // When num_groups=1, this normalizes across ALL channels (like LayerNorm on channels)
        // x: (batch, channels, seq)
        let (batch, channels, seq) = x.dims3()?;
        let channels_per_group = channels / self.num_groups;

        // Reshape to (batch, num_groups, channels_per_group, seq)
        let x = x.reshape((batch, self.num_groups, channels_per_group, seq))?;

        // Compute mean and variance over (channels_per_group, seq) for each group
        // Flatten last two dims for easier computation
        let x_flat = x.reshape((batch, self.num_groups, channels_per_group * seq))?;
        let mean = x_flat.mean_keepdim(2)?; // (batch, num_groups, 1)
        let var = x_flat.var_keepdim(2)?;   // (batch, num_groups, 1)

        // Normalize
        let eps = 1e-5;
        let mean = mean.unsqueeze(3)?; // (batch, num_groups, 1, 1)
        let var = var.unsqueeze(3)?;   // (batch, num_groups, 1, 1)
        let x_centered = x.broadcast_sub(&mean)?;
        let std = (var + eps)?.sqrt()?;
        let x_norm = x_centered.broadcast_div(&std)?;

        // Reshape back to (batch, channels, seq)
        let x = x_norm.reshape((batch, channels, seq))?;

        // Apply learnable affine transform (per-channel)
        let gn_weight = self.gn_weight.unsqueeze(0)?.unsqueeze(2)?; // (1, channels, 1)
        let gn_bias = self.gn_bias.unsqueeze(0)?.unsqueeze(2)?;     // (1, channels, 1)
        let x = x.broadcast_mul(&gn_weight)?;
        let x = x.broadcast_add(&gn_bias)?;

        if debug_idx == 0 {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG ConvGN: after groupnorm: mean={:.6}, var={:.6}", m, v);
        }

        // Mish activation: x * tanh(softplus(x))
        let x = mish(&x)?;

        if debug_idx == 0 {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG ConvGN: after mish: mean={:.6}, var={:.6}", m, v);
        }

        Ok(x)
    }
}

/// Length regulator for processing GPT features
///
/// Takes continuous 1024-dim features from gpt_layer and processes them
/// for mel spectrogram synthesis.
///
/// CRITICAL: Python's order of operations:
/// 1. content_in_proj: project to 512 dims
/// 2. F.interpolate with mode='nearest' to expand sequence length
/// 3. Apply conv+groupnorm+mish blocks on EXPANDED sequence
/// 4. Final conv layer
pub struct LengthRegulator {
    device: Device,
    config: LengthRegulatorConfig,
    /// GPT layer projection (1280 -> 1024)
    gpt_layer: Option<GptLayerProjection>,
    /// Content input projection [512, 1024]
    content_in_proj: Option<Linear>,
    /// Conv model blocks (Conv + GroupNorm + Mish) - applied AFTER interpolation
    conv_blocks: Vec<ConvGNBlock>,
    /// Final conv layer [512, 512, 1]
    final_conv_weight: Option<Tensor>,
    final_conv_bias: Option<Tensor>,
    /// Number of groups for GroupNorm (default 1)
    num_groups: usize,
    /// Whether initialized
    weights_loaded: bool,
}

impl LengthRegulator {
    /// Create with default config
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(LengthRegulatorConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: LengthRegulatorConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            gpt_layer: None,
            content_in_proj: None,
            conv_blocks: Vec::new(),
            final_conv_weight: None,
            final_conv_bias: None,
            num_groups: 1, // Python default: nn.GroupNorm(1, channels)
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut regulator = Self::new(device)?;
        regulator.load_weights(path)?;
        Ok(regulator)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        // GPT layer projection
        self.gpt_layer = Some(GptLayerProjection::new(&self.device)?);

        // Content input projection [512, 1024]
        let w = Tensor::randn(
            0.0f32,
            0.02,
            (self.config.channels, self.config.in_channels),
            &self.device,
        )?;
        let b = Tensor::zeros((self.config.channels,), DType::F32, &self.device)?;
        self.content_in_proj = Some(Linear::new(w, Some(b)));

        // Conv model: 4 conv+groupnorm+mish blocks with kernel sizes [3, 3, 3, 3]
        // Python: nn.Conv1d -> nn.GroupNorm(1, channels) -> nn.Mish()
        self.conv_blocks.clear();
        for _ in 0..4 {
            self.conv_blocks.push(ConvGNBlock::new(self.config.channels, 3, self.num_groups, &self.device)?);
        }

        // Final conv [512, 512, 1]
        self.final_conv_weight = Some(Tensor::randn(
            0.0f32,
            0.02,
            (self.config.channels, self.config.channels, 1),
            &self.device,
        )?);
        self.final_conv_bias = Some(Tensor::zeros((self.config.channels,), DType::F32, &self.device)?);

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            eprintln!("LengthRegulator: checkpoint not found at {:?}, using random weights", path);
            return self.initialize_random();
        }

        eprintln!("LengthRegulator: Loading weights from {:?}", path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)?
        };

        // Load gpt_layer (the projection from GPT embeddings)
        let gpt_vb = vb.pp("gpt_layer");
        let w0 = gpt_vb.get((256, 1280), "0.weight")?;
        let b0 = gpt_vb.get(256, "0.bias")?;
        let w1 = gpt_vb.get((128, 256), "1.weight")?;
        let b1 = gpt_vb.get(128, "1.bias")?;
        let w2 = gpt_vb.get((1024, 128), "2.weight")?;
        let b2 = gpt_vb.get(1024, "2.bias")?;
        self.gpt_layer = Some(GptLayerProjection {
            layer0: Linear::new(w0, Some(b0)),
            layer1: Linear::new(w1, Some(b1)),
            layer2: Linear::new(w2, Some(b2)),
        });

        // Load content_in_proj
        let lr_vb = vb.pp("length_regulator");
        let w = lr_vb.get((512, 1024), "content_in_proj.weight")?;
        let b = lr_vb.get(512, "content_in_proj.bias")?;
        self.content_in_proj = Some(Linear::new(w, Some(b)));

        // Load conv model blocks
        // Python structure: Conv1d, GroupNorm, Mish (indices 0,1,2 / 3,4,5 / 6,7,8 / 9,10,11)
        // model.0 (conv), model.1 (groupnorm), model.2 (mish - no params), etc.
        self.conv_blocks.clear();
        let conv_indices = [(0, 1), (3, 4), (6, 7), (9, 10)]; // (conv_idx, gn_idx)
        for (conv_idx, gn_idx) in conv_indices {
            let conv_weight = lr_vb.get((512, 512, 3), &format!("model.{}.weight", conv_idx))?;
            let conv_bias = lr_vb.get(512, &format!("model.{}.bias", conv_idx))?;
            let gn_weight = lr_vb.get(512, &format!("model.{}.weight", gn_idx))?;
            let gn_bias = lr_vb.get(512, &format!("model.{}.bias", gn_idx))?;

            self.conv_blocks.push(ConvGNBlock {
                conv_weight,
                conv_bias,
                gn_weight,
                gn_bias,
                kernel_size: 3,
                num_groups: self.num_groups,
            });
        }

        // Load final conv (model.12)
        self.final_conv_weight = Some(lr_vb.get((512, 512, 1), "model.12.weight")?);
        self.final_conv_bias = Some(lr_vb.get(512, "model.12.bias")?);

        self.weights_loaded = true;
        eprintln!("LengthRegulator: Weights loaded successfully");
        Ok(())
    }

    /// Project GPT mel embeddings through gpt_layer
    /// Input: (batch, seq, 1280) - mel embeddings from GPT
    /// Output: (batch, seq, 1024) - features for length regulator
    pub fn project_gpt_embeddings(&self, embeddings: &Tensor) -> Result<Tensor> {
        if let Some(ref gpt_layer) = self.gpt_layer {
            gpt_layer.forward(embeddings)
        } else {
            anyhow::bail!("GPT layer projection not initialized")
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `features` - Continuous features (batch, seq, 1024) from gpt_layer projection
    /// * `target_lengths` - Optional target lengths for each batch item
    ///
    /// # Returns
    /// * Tuple of (processed features, durations)
    ///
    /// CRITICAL: Order of operations matches Python:
    /// 1. content_in_proj: (batch, seq, 1024) -> (batch, seq, 512)
    /// 2. F.interpolate to target_length FIRST (nearest neighbor)
    /// 3. Apply conv+groupnorm+mish blocks on EXPANDED sequence
    /// 4. Final conv layer
    pub fn forward(
        &self,
        features: &Tensor,
        target_lengths: Option<&[usize]>,
    ) -> Result<(Tensor, Tensor)> {
        if !self.weights_loaded {
            let batch = features.dim(0)?;
            let seq_len = features.dim(1)?;
            let target_len = target_lengths.map(|l| l[0]).unwrap_or(seq_len);
            let output = Tensor::zeros(
                (batch, target_len, self.config.channels),
                DType::F32,
                &self.device,
            )?;
            let durations = Tensor::ones((batch, seq_len), DType::F32, &self.device)?;
            return Ok((output, durations));
        }

        // Step 1: Project input from 1024 to 512 dims
        let x = if let Some(ref proj) = self.content_in_proj {
            proj.forward(features)?
        } else {
            features.clone()
        };

        // Transpose for conv1d: (batch, seq, channels) -> (batch, channels, seq)
        let mut x = x.transpose(1, 2)?;

        {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG LR: after content_in_proj: mean={:.6}, var={:.6}", m, v);
        }

        // Step 2: INTERPOLATE FIRST to target length (before conv blocks!)
        // Python: x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        let (batch_size, _channels, seq_len) = x.dims3()?;
        let target_len = target_lengths.map(|l| l[0]).unwrap_or(seq_len);

        if target_len != seq_len {
            // Nearest neighbor interpolation
            x = self.interpolate_nearest(&x, target_len)?;
        }

        {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG LR: after interpolation: mean={:.6}, var={:.6}", m, v);
        }

        // Step 3: Apply conv blocks on EXPANDED sequence
        for (idx, block) in self.conv_blocks.iter().enumerate() {
            x = block.forward(&x, idx)?;
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG LR: after conv_block {}: mean={:.6}, var={:.6}", idx, m, v);
        }

        // Step 4: Final conv
        if let (Some(ref w), Some(ref b)) = (&self.final_conv_weight, &self.final_conv_bias) {
            x = x.conv1d(w, 0, 1, 1, 1)?; // kernel_size=1, no padding
            let bias = b.unsqueeze(0)?.unsqueeze(2)?;
            x = x.broadcast_add(&bias)?;
        }

        {
            let m: f32 = x.mean_all()?.to_scalar()?;
            let v: f32 = x.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG LR: after final_conv: mean={:.6}, var={:.6}", m, v);
        }

        // Transpose back: (batch, channels, seq) -> (batch, seq, channels)
        let x = x.transpose(1, 2)?;

        // Durations are implicit from the interpolation
        let (_, final_len, _) = x.dims3()?;
        let durations = Tensor::ones((batch_size, final_len), DType::F32, &self.device)?;

        Ok((x, durations))
    }

    /// Nearest neighbor interpolation (matching F.interpolate with mode='nearest')
    fn interpolate_nearest(&self, x: &Tensor, target_len: usize) -> Result<Tensor> {
        // x: (batch, channels, seq)
        let (batch, channels, seq) = x.dims3()?;

        if target_len == seq {
            return Ok(x.clone());
        }

        // For each target position, find the nearest source position
        let mut output_data = vec![0.0f32; batch * channels * target_len];
        let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;

        for b in 0..batch {
            for c in 0..channels {
                for t in 0..target_len {
                    // Nearest neighbor: find source index
                    // PyTorch F.interpolate(mode='nearest'): src_t = floor(t * src_len / dst_len)
                    let src_t = ((t as f32) * (seq as f32) / (target_len as f32)).floor() as usize;
                    let src_t = src_t.min(seq - 1);

                    let src_idx = b * channels * seq + c * seq + src_t;
                    let dst_idx = b * channels * target_len + c * target_len + t;
                    output_data[dst_idx] = x_vec[src_idx];
                }
            }
        }

        Tensor::from_slice(&output_data, (batch, channels, target_len), &self.device)
            .map_err(Into::into)
    }

    /// Expand features using durations
    fn length_regulate(&self, features: &Tensor, durations: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, channels) = features.dims3()?;
        let durations_vec: Vec<Vec<f32>> = durations.to_vec2()?;

        let mut expanded_batch = Vec::new();
        let mut max_len = 0;

        for b in 0..batch_size {
            let mut expanded_frames = Vec::new();
            for s in 0..seq_len {
                let dur = durations_vec[b][s].round().max(1.0) as usize;
                for _ in 0..dur {
                    expanded_frames.push((b, s));
                }
            }
            max_len = max_len.max(expanded_frames.len());
            expanded_batch.push(expanded_frames);
        }

        let mut output_data = vec![0.0f32; batch_size * max_len * channels];
        for (b, frames) in expanded_batch.iter().enumerate() {
            for (t, &(_, s)) in frames.iter().enumerate() {
                let dst_offset = b * max_len * channels + t * channels;
                let frame = features.i((b, s, ..))?;
                let frame_data: Vec<f32> = frame.to_vec1()?;
                for (c, &val) in frame_data.iter().enumerate() {
                    output_data[dst_offset + c] = val;
                }
            }
        }

        Tensor::from_slice(&output_data, (batch_size, max_len, channels), &self.device)
            .map_err(Into::into)
    }

    /// Adjust durations to match target lengths
    fn adjust_durations(&self, durations: &Tensor, targets: &[usize]) -> Result<Tensor> {
        let (batch_size, seq_len) = durations.dims2()?;
        let mut dur_vec: Vec<Vec<f32>> = durations.to_vec2()?;

        for (b, &target) in targets.iter().enumerate() {
            if b >= batch_size {
                break;
            }
            let current_sum: f32 = dur_vec[b].iter().sum();
            if current_sum > 0.0 {
                let scale = target as f32 / current_sum;
                for d in &mut dur_vec[b] {
                    *d *= scale;
                }
            }
        }

        let flat: Vec<f32> = dur_vec.into_iter().flatten().collect();
        Tensor::from_slice(&flat, (batch_size, seq_len), &self.device).map_err(Into::into)
    }

    /// Get output channels
    pub fn output_channels(&self) -> usize {
        self.config.channels
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_regulator_config_default() {
        let config = LengthRegulatorConfig::default();
        assert_eq!(config.channels, 512);
        assert_eq!(config.in_channels, 1024);
    }

    #[test]
    fn test_length_regulator_new() {
        let device = Device::Cpu;
        let regulator = LengthRegulator::new(&device).unwrap();
        assert_eq!(regulator.output_channels(), 512);
    }

    #[test]
    fn test_length_regulator_placeholder() {
        let device = Device::Cpu;
        let regulator = LengthRegulator::new(&device).unwrap();

        // Not initialized - should return zeros
        let features = Tensor::randn(0.0f32, 1.0, (2, 50, 1024), &device).unwrap();
        let (expanded, _durations) = regulator.forward(&features, Some(&[100, 100])).unwrap();

        assert_eq!(expanded.dims3().unwrap(), (2, 100, 512));
    }

    #[test]
    fn test_length_regulator_initialized() {
        let device = Device::Cpu;
        let mut regulator = LengthRegulator::new(&device).unwrap();
        regulator.initialize_random().unwrap();

        assert!(regulator.is_initialized());

        // Input is 1024-dim continuous features
        let features = Tensor::randn(0.0f32, 1.0, (1, 20, 1024), &device).unwrap();
        let (output, _durations) = regulator.forward(&features, None).unwrap();

        // Output should be 512-dim features
        let (batch, len, channels) = output.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(len, 20); // Same length (no expansion without target_lengths)
        assert_eq!(channels, 512);
    }

    #[test]
    fn test_gpt_layer_projection() {
        let device = Device::Cpu;
        let mut regulator = LengthRegulator::new(&device).unwrap();
        regulator.initialize_random().unwrap();

        // GPT mel embeddings are 1280-dim
        let embeddings = Tensor::randn(0.0f32, 1.0, (1, 10, 1280), &device).unwrap();
        let projected = regulator.project_gpt_embeddings(&embeddings).unwrap();

        // Output should be 1024-dim
        assert_eq!(projected.dims3().unwrap(), (1, 10, 1024));
    }
}
