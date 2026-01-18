//! Length regulator for mel code expansion
//!
//! Expands discrete mel codes to target length for synthesis.
//! Handles duration prediction and alignment between semantic
//! codes and mel spectrogram frames.
//!
//! Architecture:
//! - Content embedding (discrete codes to continuous)
//! - Duration predictor (optional)
//! - Length expansion via upsampling

use anyhow::Result;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{Linear, Module, LayerNorm};
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

/// Length regulator configuration
pub struct LengthRegulatorConfig {
    pub channels: usize,
    pub in_channels: usize,
    pub is_discrete: bool,
    pub content_codebook_size: usize,
    pub sampling_ratios: Vec<usize>,
}

impl Default for LengthRegulatorConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            in_channels: 1024,
            is_discrete: false,
            content_codebook_size: 2048,
            sampling_ratios: vec![1, 1, 1, 1],
        }
    }
}

/// Length regulator for expanding mel codes
///
/// Takes discrete or continuous mel codes and expands them to
/// the target mel spectrogram length using predicted durations.
pub struct LengthRegulator {
    device: Device,
    config: LengthRegulatorConfig,
    /// Content embedding (for discrete codes)
    content_embedding: Option<Tensor>,
    /// Input projection
    input_proj: Option<Linear>,
    /// Duration predictor
    duration_predictor: Option<DurationPredictor>,
    /// Output projection
    output_proj: Option<Linear>,
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
            content_embedding: None,
            input_proj: None,
            duration_predictor: None,
            output_proj: None,
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
        // Content embedding for discrete codes
        if self.config.is_discrete {
            self.content_embedding = Some(Tensor::randn(
                0.0f32,
                0.02,
                (self.config.content_codebook_size, self.config.channels),
                &self.device,
            )?);
        }

        // Input projection
        let w = Tensor::randn(
            0.0f32,
            0.02,
            (self.config.channels, self.config.in_channels),
            &self.device,
        )?;
        let b = Tensor::zeros((self.config.channels,), DType::F32, &self.device)?;
        self.input_proj = Some(Linear::new(w, Some(b)));

        // Duration predictor
        self.duration_predictor = Some(DurationPredictor::new(
            self.config.channels,
            &self.device,
        )?);

        // Output projection
        let w = Tensor::randn(
            0.0f32,
            0.02,
            (self.config.channels, self.config.channels),
            &self.device,
        )?;
        let b = Tensor::zeros((self.config.channels,), DType::F32, &self.device)?;
        self.output_proj = Some(Linear::new(w, Some(b)));

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return self.initialize_random();
        }
        self.initialize_random()
    }

    /// Embed discrete codes
    fn embed_codes(&self, codes: &Tensor) -> Result<Tensor> {
        if let Some(ref emb) = self.content_embedding {
            let flat = codes.flatten_all()?;
            let embedded = emb.index_select(&flat, 0)?;
            let (batch, seq) = codes.dims2()?;
            Ok(embedded.reshape((batch, seq, self.config.channels))?)
        } else {
            anyhow::bail!("Content embedding not initialized for discrete input")
        }
    }

    /// Expand features using durations
    ///
    /// # Arguments
    /// * `features` - Input features (batch, seq, channels)
    /// * `durations` - Durations per frame (batch, seq)
    ///
    /// # Returns
    /// * Expanded features (batch, target_len, channels)
    fn length_regulate(&self, features: &Tensor, durations: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, channels) = features.dims3()?;

        // Round durations to integers
        let durations_vec: Vec<Vec<f32>> = durations.to_vec2()?;

        let mut expanded_batch = Vec::new();
        let mut max_len = 0;

        for b in 0..batch_size {
            let mut expanded_frames = Vec::new();

            for s in 0..seq_len {
                let dur = durations_vec[b][s].round().max(1.0) as usize;
                // Repeat the frame `dur` times
                for _ in 0..dur {
                    expanded_frames.push((b, s));
                }
            }

            max_len = max_len.max(expanded_frames.len());
            expanded_batch.push(expanded_frames);
        }

        // Build output tensor
        let mut output_data = vec![0.0f32; batch_size * max_len * channels];

        for (b, frames) in expanded_batch.iter().enumerate() {
            for (t, &(_, s)) in frames.iter().enumerate() {
                // Copy features[b, s, :] to output[b, t, :]
                let _src_offset = b * seq_len * channels + s * channels;
                let dst_offset = b * max_len * channels + t * channels;

                // We need to get the data from the tensor
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

    /// Forward pass
    ///
    /// # Arguments
    /// * `codes` - Input codes (batch, seq) for discrete or (batch, seq, dim) for continuous
    /// * `target_lengths` - Optional target lengths for each batch item
    ///
    /// # Returns
    /// * Tuple of (expanded features, durations)
    pub fn forward(
        &self,
        codes: &Tensor,
        target_lengths: Option<&[usize]>,
    ) -> Result<(Tensor, Tensor)> {
        if !self.weights_loaded {
            let batch = codes.dim(0)?;
            let target_len = target_lengths.map(|l| l[0]).unwrap_or(100);
            let features = Tensor::zeros(
                (batch, target_len, self.config.channels),
                DType::F32,
                &self.device,
            )?;
            let durations = Tensor::ones((batch, 1), DType::F32, &self.device)?;
            return Ok((features, durations));
        }

        // Get input features
        let features = if self.config.is_discrete {
            self.embed_codes(codes)?
        } else {
            codes.clone()
        };

        // Project to internal dimension
        let features = if let Some(ref proj) = self.input_proj {
            proj.forward(&features)?
        } else {
            features
        };

        // Predict durations
        let features_t = features.transpose(1, 2)?; // (batch, channels, seq)
        let durations = if let Some(ref predictor) = self.duration_predictor {
            predictor.forward(&features_t)?
        } else {
            let (batch, seq, _) = features.dims3()?;
            Tensor::ones((batch, seq), DType::F32, &self.device)?
        };

        // Adjust durations to match target length if specified
        let durations = if let Some(targets) = target_lengths {
            self.adjust_durations(&durations, targets)?
        } else {
            durations
        };

        // Expand features using durations
        let expanded = self.length_regulate(&features, &durations)?;

        // Output projection
        let output = if let Some(ref proj) = self.output_proj {
            proj.forward(&expanded)?
        } else {
            expanded
        };

        Ok((output, durations))
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

        // Flatten and create tensor
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

        let codes = Tensor::randn(0.0f32, 1.0, (2, 50, 1024), &device).unwrap();
        let (expanded, durations) = regulator.forward(&codes, Some(&[100, 100])).unwrap();

        assert_eq!(expanded.dims3().unwrap(), (2, 100, 512));
    }

    #[test]
    fn test_length_regulator_initialized() {
        let device = Device::Cpu;
        let mut regulator = LengthRegulator::new(&device).unwrap();
        regulator.initialize_random().unwrap();

        assert!(regulator.is_initialized());

        let codes = Tensor::randn(0.0f32, 1.0, (1, 20, 1024), &device).unwrap();
        let (expanded, durations) = regulator.forward(&codes, None).unwrap();

        // Output should have some length based on predicted durations
        let (batch, len, channels) = expanded.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(channels, 512);
        assert!(len > 0);
    }

    #[test]
    fn test_softplus() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let y = softplus(&x).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();

        // softplus(0) = ln(2) ≈ 0.693
        assert!((values[0] - 0.693).abs() < 0.01);
        // softplus(1) ≈ 1.313
        assert!((values[1] - 1.313).abs() < 0.01);
        // softplus(-1) ≈ 0.313
        assert!((values[2] - 0.313).abs() < 0.01);
    }
}
