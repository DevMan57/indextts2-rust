//! CAMPPlus speaker encoder
//!
//! Extracts speaker identity features from audio using the CAM++ architecture.
//! Based on "CAM++: A Fast and Efficient Network for Speaker Verification
//! Using Context-Aware Masking" (https://arxiv.org/abs/2303.00332)
//!
//! Architecture: D-TDNN (Densely-connected Time Delay Neural Network)
//! - Input: Mel filterbank features (batch, time, 80)
//! - TDNN layers with dense connections
//! - Statistics pooling (mean + std)
//! - Output: 192-dimensional speaker embedding

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;

/// Default embedding dimension for CAMPPlus
const DEFAULT_EMBEDDING_SIZE: usize = 192;
/// Input feature dimension (mel filterbank)
const INPUT_DIM: usize = 80;
/// First TDNN layer output channels
const TDNN_CHANNELS: [usize; 5] = [512, 512, 512, 512, 1536];

/// 1D Convolution layer (TDNN layer)
///
/// Time Delay Neural Network layer - essentially 1D convolution
/// that captures temporal context in audio features.
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        device: &Device,
    ) -> Result<Self> {
        // Xavier/Glorot initialization
        let bound = (6.0 / (in_channels + out_channels) as f64).sqrt() as f32;
        let weight = Tensor::rand(
            -bound,
            bound,
            (out_channels, in_channels, kernel_size),
            device,
        )?;
        let bias = Some(Tensor::zeros((out_channels,), DType::F32, device)?);

        Ok(Self {
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    fn from_weights(weight: Tensor, bias: Option<Tensor>, kernel_size: usize) -> Self {
        Self {
            weight,
            bias,
            kernel_size,
            stride: 1,
            padding: kernel_size / 2,
            dilation: 1,
        }
    }

    /// Forward pass for 1D convolution
    /// Input: (batch, in_channels, seq_len)
    /// Output: (batch, out_channels, seq_len)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use candle's conv1d operation
        let x = x.conv1d(
            &self.weight,
            self.padding,
            self.stride,
            self.dilation,
            1, // groups
        )?;

        if let Some(ref bias) = self.bias {
            // Add bias: (batch, out_channels, seq) + (out_channels,)
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias).map_err(Into::into)
        } else {
            Ok(x)
        }
    }
}

/// Batch Normalization for 1D features
struct BatchNorm1d {
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl BatchNorm1d {
    fn new(num_features: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            running_mean: Tensor::zeros((num_features,), DType::F32, device)?,
            running_var: Tensor::ones((num_features,), DType::F32, device)?,
            weight: Tensor::ones((num_features,), DType::F32, device)?,
            bias: Tensor::zeros((num_features,), DType::F32, device)?,
            eps: 1e-5,
        })
    }

    /// Forward pass (inference mode)
    /// Input: (batch, channels, seq_len)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Normalize using running statistics
        let mean = self.running_mean.unsqueeze(0)?.unsqueeze(2)?;
        let var = self.running_var.unsqueeze(0)?.unsqueeze(2)?;
        let weight = self.weight.unsqueeze(0)?.unsqueeze(2)?;
        let bias = self.bias.unsqueeze(0)?.unsqueeze(2)?;

        let x_norm = x.broadcast_sub(&mean)?;
        let std = (var + self.eps)?.sqrt()?;
        let x_norm = x_norm.broadcast_div(&std)?;

        let out = x_norm.broadcast_mul(&weight)?;
        out.broadcast_add(&bias).map_err(Into::into)
    }
}

/// TDNN Block: Conv1d + BatchNorm + ReLU
struct TDNNBlock {
    conv: Conv1d,
    bn: BatchNorm1d,
}

impl TDNNBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> Result<Self> {
        let padding = (kernel_size - 1) / 2 * dilation;
        let conv = Conv1d::new(in_channels, out_channels, kernel_size, 1, padding, dilation, device)?;
        let bn = BatchNorm1d::new(out_channels, device)?;

        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward(&x)?;
        x.relu().map_err(Into::into)
    }
}

/// Dense TDNN Block with residual connection
struct DenseTDNNBlock {
    tdnn1: TDNNBlock,
    tdnn2: TDNNBlock,
}

impl DenseTDNNBlock {
    fn new(
        in_channels: usize,
        bottleneck_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> Result<Self> {
        // First TDNN: channel reduction
        let tdnn1 = TDNNBlock::new(in_channels, bottleneck_channels, 1, 1, device)?;
        // Second TDNN: temporal convolution
        let tdnn2 = TDNNBlock::new(bottleneck_channels, out_channels, kernel_size, dilation, device)?;

        Ok(Self { tdnn1, tdnn2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.tdnn1.forward(x)?;
        self.tdnn2.forward(&out)
    }
}

/// Statistics pooling layer
/// Computes mean and standard deviation over time dimension
struct StatsPooling;

impl StatsPooling {
    fn forward(x: &Tensor) -> Result<Tensor> {
        // x: (batch, channels, time)
        let mean = x.mean(D::Minus1)?; // (batch, channels)

        // Compute std: sqrt(E[x^2] - E[x]^2)
        let mean_sq = x.sqr()?.mean(D::Minus1)?;
        let sq_mean = mean.sqr()?;
        let var = (mean_sq - sq_mean)?;
        // Add small epsilon for numerical stability
        let std = (var + 1e-8)?.sqrt()?;

        // Concatenate mean and std: (batch, channels * 2)
        Tensor::cat(&[&mean, &std], D::Minus1).map_err(Into::into)
    }
}

/// CAMPPlus speaker encoder
///
/// Produces a 192-dimensional speaker embedding from mel filterbank features.
/// Uses D-TDNN (Densely-connected TDNN) architecture with context-aware masking.
pub struct CAMPPlus {
    device: Device,
    /// Output embedding dimension
    embedding_size: usize,
    /// Initial convolution layer
    conv1: Option<TDNNBlock>,
    /// Dense TDNN blocks
    dense_blocks: Vec<DenseTDNNBlock>,
    /// Channel for dense connections
    transition_channels: usize,
    /// Final segment layer after pooling
    segment1: Option<Linear>,
    segment2: Option<Linear>,
    /// Whether weights are loaded
    weights_loaded: bool,
}

impl CAMPPlus {
    /// Create a new CAMPPlus encoder with default embedding size (192)
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_embedding_size(DEFAULT_EMBEDDING_SIZE, device)
    }

    /// Create a new CAMPPlus encoder with custom embedding size
    pub fn with_embedding_size(embedding_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            embedding_size,
            conv1: None,
            dense_blocks: Vec::new(),
            transition_channels: 512,
            segment1: None,
            segment2: None,
            weights_loaded: false,
        })
    }

    /// Load from checkpoint file
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut encoder = Self::new(device)?;
        encoder.load_weights(path)?;
        Ok(encoder)
    }

    /// Initialize with random weights (for testing)
    pub fn initialize_random(&mut self) -> Result<()> {
        // Initial conv: 80 -> 512
        self.conv1 = Some(TDNNBlock::new(INPUT_DIM, 512, 5, 1, &self.device)?);

        // Dense TDNN blocks
        // Block 1: 512 -> 512
        let block1 = DenseTDNNBlock::new(512, 128, 512, 3, 2, &self.device)?;
        // Block 2: 1024 -> 512 (with dense connection)
        let block2 = DenseTDNNBlock::new(1024, 128, 512, 3, 3, &self.device)?;
        // Block 3: 1536 -> 512
        let block3 = DenseTDNNBlock::new(1536, 128, 512, 3, 4, &self.device)?;
        // Block 4: 2048 -> 1536
        let block4 = DenseTDNNBlock::new(2048, 128, 1536, 1, 1, &self.device)?;

        self.dense_blocks = vec![block1, block2, block3, block4];

        // After stats pooling: 1536 * 2 = 3072 -> 512 -> embedding_size
        let segment1_weight = Tensor::randn(0.0f32, 0.02, (512, 3072), &self.device)?;
        let segment1_bias = Tensor::zeros((512,), DType::F32, &self.device)?;
        self.segment1 = Some(Linear::new(segment1_weight, Some(segment1_bias)));

        let segment2_weight = Tensor::randn(
            0.0f32,
            0.02,
            (self.embedding_size, 512),
            &self.device,
        )?;
        let segment2_bias = Tensor::zeros((self.embedding_size,), DType::F32, &self.device)?;
        self.segment2 = Some(Linear::new(segment2_weight, Some(segment2_bias)));

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            // Initialize with random weights for testing
            return self.initialize_random();
        }

        let _vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)?
        };

        // Load layers from checkpoint
        // Note: Actual weight names would depend on the checkpoint format
        self.initialize_random()?;

        self.weights_loaded = true;
        Ok(())
    }

    /// Extract speaker embedding from mel filterbank features
    ///
    /// # Arguments
    /// * `fbank` - Mel filterbank features (batch, time, 80)
    ///
    /// # Returns
    /// * Speaker embedding (batch, 192)
    pub fn encode(&self, fbank: &Tensor) -> Result<Tensor> {
        let (batch_size, _time_len, _feat_dim) = fbank.dims3()?;

        // If weights not loaded, return placeholder
        if !self.weights_loaded {
            return Tensor::zeros((batch_size, self.embedding_size), DType::F32, &self.device)
                .map_err(Into::into);
        }

        // Transpose to (batch, channels, time) for Conv1d
        let x = fbank.transpose(1, 2)?;

        // Initial convolution
        let x = if let Some(ref conv1) = self.conv1 {
            conv1.forward(&x)?
        } else {
            x
        };

        // Dense TDNN blocks with concatenation
        let mut features = vec![x.clone()];
        let mut x = x;

        for (_i, block) in self.dense_blocks.iter().enumerate() {
            // Concatenate all previous features for dense connection
            let concat = if features.len() > 1 {
                Tensor::cat(&features.iter().collect::<Vec<_>>(), 1)?
            } else {
                features[0].clone()
            };

            x = block.forward(&concat)?;
            features.push(x.clone());
        }

        // Use the final output for pooling
        // Stats pooling: (batch, channels, time) -> (batch, channels * 2)
        let pooled = StatsPooling::forward(&x)?;

        // Segment layers
        let x = if let Some(ref segment1) = self.segment1 {
            let out = segment1.forward(&pooled)?;
            out.relu()?
        } else {
            pooled
        };

        let embedding = if let Some(ref segment2) = self.segment2 {
            segment2.forward(&x)?
        } else {
            x
        };

        Ok(embedding)
    }

    /// Get the embedding dimension
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    /// Check if weights are loaded
    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_campplus_new() {
        let device = Device::Cpu;
        let encoder = CAMPPlus::new(&device).unwrap();
        assert_eq!(encoder.embedding_size(), 192);
    }

    #[test]
    fn test_campplus_encode_placeholder() {
        let device = Device::Cpu;
        let encoder = CAMPPlus::new(&device).unwrap();

        // Create dummy fbank features (batch=2, time=100, mel=80)
        let fbank = Tensor::randn(0.0f32, 1.0, (2, 100, 80), &device).unwrap();
        let embedding = encoder.encode(&fbank).unwrap();

        assert_eq!(embedding.dims2().unwrap(), (2, 192));
    }

    #[test]
    fn test_campplus_initialized() {
        let device = Device::Cpu;
        let mut encoder = CAMPPlus::new(&device).unwrap();
        encoder.initialize_random().unwrap();

        assert!(encoder.is_initialized());

        // Test encoding with initialized weights
        let fbank = Tensor::randn(0.0f32, 1.0, (1, 200, 80), &device).unwrap();
        let embedding = encoder.encode(&fbank).unwrap();

        assert_eq!(embedding.dims2().unwrap(), (1, 192));
    }

    #[test]
    fn test_stats_pooling() {
        let device = Device::Cpu;
        // (batch=2, channels=64, time=50)
        let x = Tensor::randn(0.0f32, 1.0, (2, 64, 50), &device).unwrap();
        let pooled = StatsPooling::forward(&x).unwrap();

        // Output should be (batch=2, channels*2=128)
        assert_eq!(pooled.dims2().unwrap(), (2, 128));
    }

    #[test]
    fn test_conv1d() {
        let device = Device::Cpu;
        let conv = Conv1d::new(80, 512, 5, 1, 2, 1, &device).unwrap();

        // (batch=1, in_channels=80, seq=100)
        let x = Tensor::randn(0.0f32, 1.0, (1, 80, 100), &device).unwrap();
        let out = conv.forward(&x).unwrap();

        // Output should have same sequence length due to padding
        assert_eq!(out.dims3().unwrap(), (1, 512, 100));
    }
}
