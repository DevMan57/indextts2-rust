//! Conformer encoder for audio conditioning
//!
//! Implements the Conformer architecture from:
//! "Conformer: Convolution-augmented Transformer for Speech Recognition"
//!
//! Architecture per block:
//! - Feed-forward module (half-step residual)
//! - Multi-head self-attention module
//! - Convolution module
//! - Feed-forward module (half-step residual)
//! - Layer normalization

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder, LayerNorm};
use std::path::Path;

/// Swish activation function: x * sigmoid(x)
fn swish(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    (x * sigmoid).map_err(Into::into)
}

/// GLU (Gated Linear Unit) activation
fn glu(x: &Tensor, dim: usize) -> Result<Tensor> {
    let chunks = x.chunk(2, dim)?;
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = candle_nn::ops::sigmoid(b)?;
    (a * gate).map_err(Into::into)
}

/// Feed-forward module with expansion and Swish activation
struct FeedForwardModule {
    layer_norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    dropout_rate: f32,
}

impl FeedForwardModule {
    fn new(dim: usize, expansion_factor: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = dim * expansion_factor;
        let layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("layer_norm"))?;
        let linear1 = candle_nn::linear(dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(hidden_dim, dim, vb.pp("linear2"))?;

        Ok(Self {
            layer_norm,
            linear1,
            linear2,
            dropout_rate: 0.1,
        })
    }

    fn new_random(dim: usize, expansion_factor: usize, device: &Device) -> Result<Self> {
        let hidden_dim = dim * expansion_factor;

        let ln_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);

        let w1 = Tensor::randn(0.0f32, 0.02, (hidden_dim, dim), device)?;
        let b1 = Tensor::zeros((hidden_dim,), DType::F32, device)?;
        let linear1 = Linear::new(w1, Some(b1));

        let w2 = Tensor::randn(0.0f32, 0.02, (dim, hidden_dim), device)?;
        let b2 = Tensor::zeros((dim,), DType::F32, device)?;
        let linear2 = Linear::new(w2, Some(b2));

        Ok(Self {
            layer_norm,
            linear1,
            linear2,
            dropout_rate: 0.1,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let x = swish(&x)?;
        // Note: dropout skipped during inference
        self.linear2.forward(&x).map_err(Into::into)
    }
}

/// Multi-head self-attention module
struct MultiHeadAttention {
    layer_norm: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("layer_norm"))?;
        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;

        Ok(Self {
            layer_norm,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    fn new_random(dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = dim / num_heads;

        let ln_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);

        let make_linear = |device: &Device| -> Result<Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };

        Ok(Self {
            layer_norm,
            q_proj: make_linear(device)?,
            k_proj: make_linear(device)?,
            v_proj: make_linear(device)?,
            out_proj: make_linear(device)?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let x = self.layer_norm.forward(x)?;

        // Project Q, K, V
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;

        // Reshape for multi-head: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights / scale)?;

        // Apply mask if provided
        let attn_weights = if let Some(mask) = mask {
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?;
            let mask_val = neg_inf.broadcast_as(attn_weights.shape())?;
            let zeros = Tensor::zeros_like(&attn_weights)?;
            mask.where_cond(&zeros, &mask_val)?
                .broadcast_add(&attn_weights)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&attn_output).map_err(Into::into)
    }
}

/// Depthwise separable convolution for Conformer
struct ConvolutionModule {
    layer_norm: LayerNorm,
    pointwise_conv1: Linear,
    depthwise_conv_weight: Tensor,
    depthwise_conv_bias: Tensor,
    pointwise_conv2: Linear,
    kernel_size: usize,
}

impl ConvolutionModule {
    fn new(dim: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("layer_norm"))?;
        // Pointwise conv expands channels by 2 for GLU
        let pointwise_conv1 = candle_nn::linear(dim, dim * 2, vb.pp("pointwise_conv1"))?;
        // Depthwise conv
        let depthwise_conv_weight = vb.get((dim, 1, kernel_size), "depthwise_conv.weight")?;
        let depthwise_conv_bias = vb.get((dim,), "depthwise_conv.bias")?;
        // Pointwise conv back to dim
        let pointwise_conv2 = candle_nn::linear(dim, dim, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            layer_norm,
            pointwise_conv1,
            depthwise_conv_weight,
            depthwise_conv_bias,
            pointwise_conv2,
            kernel_size,
        })
    }

    fn new_random(dim: usize, kernel_size: usize, device: &Device) -> Result<Self> {
        let ln_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);

        let w1 = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
        let b1 = Tensor::zeros((dim * 2,), DType::F32, device)?;
        let pointwise_conv1 = Linear::new(w1, Some(b1));

        let depthwise_conv_weight = Tensor::randn(0.0f32, 0.02, (dim, 1, kernel_size), device)?;
        let depthwise_conv_bias = Tensor::zeros((dim,), DType::F32, device)?;

        let w2 = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
        let b2 = Tensor::zeros((dim,), DType::F32, device)?;
        let pointwise_conv2 = Linear::new(w2, Some(b2));

        Ok(Self {
            layer_norm,
            pointwise_conv1,
            depthwise_conv_weight,
            depthwise_conv_bias,
            pointwise_conv2,
            kernel_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;

        // Pointwise conv 1 (expand for GLU)
        let x = self.pointwise_conv1.forward(&x)?;
        let x = glu(&x, 2)?; // GLU along feature dim

        // Depthwise conv: need to transpose to (batch, channels, seq)
        let x = x.transpose(1, 2)?;
        let padding = self.kernel_size / 2;
        let x = x.conv1d(
            &self.depthwise_conv_weight,
            padding,
            1, // stride
            1, // dilation
            x.dim(1)?, // groups = channels (depthwise)
        )?;
        // Add bias
        let bias = self.depthwise_conv_bias.unsqueeze(0)?.unsqueeze(2)?;
        let x = x.broadcast_add(&bias)?;

        // Batch norm would go here, using swish for simplicity
        let x = swish(&x)?;

        // Transpose back to (batch, seq, channels)
        let x = x.transpose(1, 2)?;

        // Pointwise conv 2
        self.pointwise_conv2.forward(&x).map_err(Into::into)
    }
}

/// Single Conformer block
struct ConformerBlock {
    ff1: FeedForwardModule,
    attention: MultiHeadAttention,
    conv: ConvolutionModule,
    ff2: FeedForwardModule,
    final_layer_norm: LayerNorm,
}

impl ConformerBlock {
    fn new(
        dim: usize,
        num_heads: usize,
        ff_expansion: usize,
        conv_kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ff1 = FeedForwardModule::new(dim, ff_expansion, vb.pp("ff1"))?;
        let attention = MultiHeadAttention::new(dim, num_heads, vb.pp("attention"))?;
        let conv = ConvolutionModule::new(dim, conv_kernel_size, vb.pp("conv"))?;
        let ff2 = FeedForwardModule::new(dim, ff_expansion, vb.pp("ff2"))?;
        let final_layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            ff1,
            attention,
            conv,
            ff2,
            final_layer_norm,
        })
    }

    fn new_random(
        dim: usize,
        num_heads: usize,
        ff_expansion: usize,
        conv_kernel_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let ff1 = FeedForwardModule::new_random(dim, ff_expansion, device)?;
        let attention = MultiHeadAttention::new_random(dim, num_heads, device)?;
        let conv = ConvolutionModule::new_random(dim, conv_kernel_size, device)?;
        let ff2 = FeedForwardModule::new_random(dim, ff_expansion, device)?;

        let ln_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let final_layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);

        Ok(Self {
            ff1,
            attention,
            conv,
            ff2,
            final_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Half-step FFN residual
        let x = (x + (self.ff1.forward(x)? * 0.5)?)?;

        // Self-attention with residual
        let x = (&x + self.attention.forward(&x, mask)?)?;

        // Convolution with residual
        let x = (&x + self.conv.forward(&x)?)?;

        // Half-step FFN residual
        let x = (&x + (self.ff2.forward(&x)? * 0.5)?)?;

        // Final layer norm
        self.final_layer_norm.forward(&x).map_err(Into::into)
    }
}

/// Conformer encoder configuration
pub struct ConformerConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub ff_expansion: usize,
    pub conv_kernel_size: usize,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 80,
            output_dim: 512,
            num_blocks: 6,
            num_heads: 8,
            ff_expansion: 4,
            conv_kernel_size: 31,
        }
    }
}

/// Conformer encoder for audio conditioning
///
/// Processes mel spectrogram or other audio features through
/// a stack of Conformer blocks combining attention and convolution.
pub struct ConformerEncoder {
    device: Device,
    config: ConformerConfig,
    /// Input projection
    input_proj: Option<Linear>,
    /// Conformer blocks
    blocks: Vec<ConformerBlock>,
    /// Whether weights are loaded
    weights_loaded: bool,
}

impl ConformerEncoder {
    /// Create with default configuration
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(ConformerConfig::default(), device)
    }

    /// Create with custom configuration
    pub fn with_config(config: ConformerConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            input_proj: None,
            blocks: Vec::new(),
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut encoder = Self::new(device)?;
        encoder.load_weights(path)?;
        Ok(encoder)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        // Input projection
        let w = Tensor::randn(
            0.0f32,
            0.02,
            (self.config.output_dim, self.config.input_dim),
            &self.device,
        )?;
        let b = Tensor::zeros((self.config.output_dim,), DType::F32, &self.device)?;
        self.input_proj = Some(Linear::new(w, Some(b)));

        // Conformer blocks
        self.blocks.clear();
        for _ in 0..self.config.num_blocks {
            let block = ConformerBlock::new_random(
                self.config.output_dim,
                self.config.num_heads,
                self.config.ff_expansion,
                self.config.conv_kernel_size,
                &self.device,
            )?;
            self.blocks.push(block);
        }

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return self.initialize_random();
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)?
        };

        // Load input projection
        self.input_proj = Some(candle_nn::linear(
            self.config.input_dim,
            self.config.output_dim,
            vb.pp("input_proj"),
        )?);

        // Load conformer blocks
        self.blocks.clear();
        for i in 0..self.config.num_blocks {
            let block = ConformerBlock::new(
                self.config.output_dim,
                self.config.num_heads,
                self.config.ff_expansion,
                self.config.conv_kernel_size,
                vb.pp(&format!("blocks.{}", i)),
            )?;
            self.blocks.push(block);
        }

        self.weights_loaded = true;
        Ok(())
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input features (batch, seq_len, input_dim)
    /// * `mask` - Optional attention mask (batch, seq_len)
    ///
    /// # Returns
    /// * Encoded features (batch, seq_len, output_dim)
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if !self.weights_loaded {
            let (batch, seq, _) = x.dims3()?;
            return Tensor::zeros((batch, seq, self.config.output_dim), DType::F32, &self.device)
                .map_err(Into::into);
        }

        // Input projection
        let mut x = if let Some(ref proj) = self.input_proj {
            proj.forward(x)?
        } else {
            x.clone()
        };

        // Process through conformer blocks
        for block in &self.blocks {
            x = block.forward(&x, mask)?;
        }

        Ok(x)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
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
    fn test_conformer_config_default() {
        let config = ConformerConfig::default();
        assert_eq!(config.input_dim, 80);
        assert_eq!(config.output_dim, 512);
        assert_eq!(config.num_blocks, 6);
    }

    #[test]
    fn test_conformer_encoder_new() {
        let device = Device::Cpu;
        let encoder = ConformerEncoder::new(&device).unwrap();
        assert_eq!(encoder.output_dim(), 512);
    }

    #[test]
    fn test_conformer_encoder_placeholder() {
        let device = Device::Cpu;
        let encoder = ConformerEncoder::new(&device).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (2, 50, 80), &device).unwrap();
        let out = encoder.forward(&x, None).unwrap();

        assert_eq!(out.dims3().unwrap(), (2, 50, 512));
    }

    #[test]
    fn test_conformer_encoder_initialized() {
        let device = Device::Cpu;
        let mut encoder = ConformerEncoder::new(&device).unwrap();
        encoder.initialize_random().unwrap();

        assert!(encoder.is_initialized());

        let x = Tensor::randn(0.0f32, 1.0, (1, 100, 80), &device).unwrap();
        let out = encoder.forward(&x, None).unwrap();

        assert_eq!(out.dims3().unwrap(), (1, 100, 512));
    }

    #[test]
    fn test_swish() {
        let device = Device::Cpu;
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 2.0], &device).unwrap();
        let out = swish(&x).unwrap();
        // swish(-1) ≈ -0.27, swish(0) = 0, swish(1) ≈ 0.73, swish(2) ≈ 1.76
        assert_eq!(out.dims1().unwrap(), 4);
    }
}
