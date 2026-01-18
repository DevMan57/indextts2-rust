//! Perceiver resampler for cross-attention conditioning
//!
//! Implements the Perceiver resampler that uses learned latent queries
//! to resample variable-length audio features into fixed-length conditioning.
//!
//! Architecture:
//! - Learned latent queries (num_latents, dim)
//! - Cross-attention: latents attend to encoder outputs
//! - Self-attention: latents attend to each other
//! - Multiple layers of cross + self attention

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder, LayerNorm};
use std::path::Path;

/// Cross-attention layer: queries attend to keys/values from encoder
struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;

        Ok(Self {
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

        let make_linear = |device: &Device| -> Result<Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };

        Ok(Self {
            q_proj: make_linear(device)?,
            k_proj: make_linear(device)?,
            v_proj: make_linear(device)?,
            out_proj: make_linear(device)?,
            num_heads,
            head_dim,
        })
    }

    /// Cross-attention forward
    ///
    /// # Arguments
    /// * `queries` - Query tensor (batch, num_latents, dim)
    /// * `context` - Key/value context from encoder (batch, seq_len, dim)
    fn forward(&self, queries: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (batch_size, num_latents, _) = queries.dims3()?;
        let (_, ctx_len, _) = context.dims3()?;

        // Project queries, keys, values
        let q = self.q_proj.forward(queries)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, num_latents, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (batch, heads, num_latents, head_dim)
        let k = k
            .reshape((batch_size, ctx_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (batch, heads, ctx_len, head_dim)
        let v = v
            .reshape((batch_size, ctx_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, num_latents, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&attn_output).map_err(Into::into)
    }
}

/// Self-attention layer for latents
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;

        Ok(Self {
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

        let make_linear = |device: &Device| -> Result<Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };

        Ok(Self {
            q_proj: make_linear(device)?,
            k_proj: make_linear(device)?,
            v_proj: make_linear(device)?,
            out_proj: make_linear(device)?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&attn_output).map_err(Into::into)
    }
}

/// Feed-forward network
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn new(dim: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * mult;
        let linear1 = candle_nn::linear(dim, hidden, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(hidden, dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn new_random(dim: usize, mult: usize, device: &Device) -> Result<Self> {
        let hidden = dim * mult;
        let w1 = Tensor::randn(0.0f32, 0.02, (hidden, dim), device)?;
        let b1 = Tensor::zeros((hidden,), DType::F32, device)?;
        let w2 = Tensor::randn(0.0f32, 0.02, (dim, hidden), device)?;
        let b2 = Tensor::zeros((dim,), DType::F32, device)?;
        Ok(Self {
            linear1: Linear::new(w1, Some(b1)),
            linear2: Linear::new(w2, Some(b2)),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x).map_err(Into::into)
    }
}

/// Single Perceiver layer: cross-attention + self-attention + FFN
struct PerceiverLayer {
    cross_attn: CrossAttention,
    cross_norm: LayerNorm,
    self_attn: SelfAttention,
    self_norm: LayerNorm,
    ffn: FeedForward,
    ffn_norm: LayerNorm,
}

impl PerceiverLayer {
    fn new(dim: usize, num_heads: usize, ff_mult: usize, vb: VarBuilder) -> Result<Self> {
        let cross_attn = CrossAttention::new(dim, num_heads, vb.pp("cross_attn"))?;
        let cross_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("cross_norm"))?;
        let self_attn = SelfAttention::new(dim, num_heads, vb.pp("self_attn"))?;
        let self_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("self_norm"))?;
        let ffn = FeedForward::new(dim, ff_mult, vb.pp("ffn"))?;
        let ffn_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("ffn_norm"))?;

        Ok(Self {
            cross_attn,
            cross_norm,
            self_attn,
            self_norm,
            ffn,
            ffn_norm,
        })
    }

    fn new_random(dim: usize, num_heads: usize, ff_mult: usize, device: &Device) -> Result<Self> {
        let cross_attn = CrossAttention::new_random(dim, num_heads, device)?;
        let self_attn = SelfAttention::new_random(dim, num_heads, device)?;
        let ffn = FeedForward::new_random(dim, ff_mult, device)?;

        let make_layer_norm = |device: &Device| -> Result<LayerNorm> {
            let w = Tensor::ones((dim,), DType::F32, device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(LayerNorm::new(w, b, 1e-5))
        };

        Ok(Self {
            cross_attn,
            cross_norm: make_layer_norm(device)?,
            self_attn,
            self_norm: make_layer_norm(device)?,
            ffn,
            ffn_norm: make_layer_norm(device)?,
        })
    }

    fn forward(&self, latents: &Tensor, context: &Tensor) -> Result<Tensor> {
        // Cross-attention with pre-norm
        let normed = self.cross_norm.forward(latents)?;
        let latents = (latents + self.cross_attn.forward(&normed, context)?)?;

        // Self-attention with pre-norm
        let normed = self.self_norm.forward(&latents)?;
        let latents = (&latents + self.self_attn.forward(&normed)?)?;

        // FFN with pre-norm
        let normed = self.ffn_norm.forward(&latents)?;
        (&latents + self.ffn.forward(&normed)?).map_err(Into::into)
    }
}

/// Perceiver resampler configuration
pub struct PerceiverConfig {
    pub dim: usize,
    pub num_latents: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_mult: usize,
}

impl Default for PerceiverConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            num_latents: 32,
            num_heads: 8,
            num_layers: 2,
            ff_mult: 4,
        }
    }
}

/// Perceiver resampler for cross-attention conditioning
///
/// Uses learned latent queries to compress variable-length encoder
/// outputs into fixed-length conditioning for the GPT decoder.
pub struct PerceiverResampler {
    device: Device,
    config: PerceiverConfig,
    /// Learned latent queries
    latents: Option<Tensor>,
    /// Perceiver layers
    layers: Vec<PerceiverLayer>,
    /// Output projection (if dim mismatch)
    output_proj: Option<Linear>,
    /// Whether initialized
    weights_loaded: bool,
}

impl PerceiverResampler {
    /// Create with default config
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(PerceiverConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: PerceiverConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            latents: None,
            layers: Vec::new(),
            output_proj: None,
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut resampler = Self::new(device)?;
        resampler.load_weights(path)?;
        Ok(resampler)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        // Learned latent queries
        self.latents = Some(Tensor::randn(
            0.0f32,
            0.02,
            (1, self.config.num_latents, self.config.dim),
            &self.device,
        )?);

        // Perceiver layers
        self.layers.clear();
        for _ in 0..self.config.num_layers {
            let layer = PerceiverLayer::new_random(
                self.config.dim,
                self.config.num_heads,
                self.config.ff_mult,
                &self.device,
            )?;
            self.layers.push(layer);
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

        // Load latents
        self.latents = Some(vb.get(
            (1, self.config.num_latents, self.config.dim),
            "latents",
        )?);

        // Load layers
        self.layers.clear();
        for i in 0..self.config.num_layers {
            let layer = PerceiverLayer::new(
                self.config.dim,
                self.config.num_heads,
                self.config.ff_mult,
                vb.pp(&format!("layers.{}", i)),
            )?;
            self.layers.push(layer);
        }

        self.weights_loaded = true;
        Ok(())
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `context` - Encoder output (batch, seq_len, dim)
    ///
    /// # Returns
    /// * Resampled conditioning (batch, num_latents, dim)
    pub fn forward(&self, context: &Tensor) -> Result<Tensor> {
        let batch_size = context.dim(0)?;

        if !self.weights_loaded {
            return Tensor::zeros(
                (batch_size, self.config.num_latents, self.config.dim),
                DType::F32,
                &self.device,
            )
            .map_err(Into::into);
        }

        // Expand latents to batch size
        let latents = self
            .latents
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Latents not initialized"))?;
        let mut latents = latents.broadcast_as((batch_size, self.config.num_latents, self.config.dim))?
            .contiguous()?;

        // Process through perceiver layers
        for layer in &self.layers {
            latents = layer.forward(&latents, context)?;
        }

        // Optional output projection
        if let Some(ref proj) = self.output_proj {
            latents = proj.forward(&latents)?;
        }

        Ok(latents)
    }

    /// Get number of output latents
    pub fn num_latents(&self) -> usize {
        self.config.num_latents
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.dim
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
    fn test_perceiver_config_default() {
        let config = PerceiverConfig::default();
        assert_eq!(config.dim, 512);
        assert_eq!(config.num_latents, 32);
    }

    #[test]
    fn test_perceiver_new() {
        let device = Device::Cpu;
        let resampler = PerceiverResampler::new(&device).unwrap();
        assert_eq!(resampler.num_latents(), 32);
        assert_eq!(resampler.output_dim(), 512);
    }

    #[test]
    fn test_perceiver_placeholder() {
        let device = Device::Cpu;
        let resampler = PerceiverResampler::new(&device).unwrap();

        let context = Tensor::randn(0.0f32, 1.0, (2, 100, 512), &device).unwrap();
        let out = resampler.forward(&context).unwrap();

        // Output should be (batch, num_latents, dim)
        assert_eq!(out.dims3().unwrap(), (2, 32, 512));
    }

    #[test]
    fn test_perceiver_initialized() {
        let device = Device::Cpu;
        let mut resampler = PerceiverResampler::new(&device).unwrap();
        resampler.initialize_random().unwrap();

        assert!(resampler.is_initialized());

        let context = Tensor::randn(0.0f32, 1.0, (1, 50, 512), &device).unwrap();
        let out = resampler.forward(&context).unwrap();

        assert_eq!(out.dims3().unwrap(), (1, 32, 512));
    }

    #[test]
    fn test_cross_attention() {
        let device = Device::Cpu;
        let attn = CrossAttention::new_random(512, 8, &device).unwrap();

        let queries = Tensor::randn(0.0f32, 1.0, (2, 32, 512), &device).unwrap();
        let context = Tensor::randn(0.0f32, 1.0, (2, 100, 512), &device).unwrap();

        let out = attn.forward(&queries, &context).unwrap();
        assert_eq!(out.dims3().unwrap(), (2, 32, 512));
    }
}
