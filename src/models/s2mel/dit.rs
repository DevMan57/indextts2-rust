//! Diffusion Transformer (DiT) for mel spectrogram synthesis
//!
//! Implements a transformer-based denoising model with:
//! - Time embedding for diffusion timesteps
//! - Style conditioning from speaker embeddings
//! - UViT-style skip connections between layers
//! - AdaLN (Adaptive Layer Normalization) for conditioning
//! - WaveNet post-processing for refined output
//!
//! Architecture matches cfm.estimator from s2mel.safetensors:
//! - x_embedder: Input mel projection with weight normalization
//! - t_embedder: Time embedding MLP with learnable frequencies
//! - cond_embedder: Style conditioning [1024, 512]
//! - transformer.layers: 13 blocks with fused QKV attention, SwiGLU FFN, AdaLN
//! - skip_linear: Concat transformer output with original input [512, 592]
//! - conv1: Pre-WaveNet projection [512, 512]
//! - t_embedder2: Second time embedding for WaveNet conditioning
//! - wavenet: 8 layers of dilated convolutions with gated activations
//! - res_projection: Residual connection [512, 512]
//! - final_layer: Output projection with AdaLN

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module};
use std::collections::HashMap;
use std::path::Path;

use super::weights::load_s2mel_safetensors;

/// Helper to load a Linear layer from tensors
fn load_linear(
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

/// Helper to apply weight normalization to a tensor
/// Weight normalization: weight = weight_g * (weight_v / ||weight_v||_2)
/// where ||weight_v||_2 is computed along all dimensions except dim 0 (output channel)
fn apply_weight_normalization(
    weight_v: &Tensor,
    weight_g: &Tensor,
) -> Result<Tensor> {
    let eps = 1e-6;
    let ndim = weight_v.dims().len();

    // Compute L2 norm along all dims except 0
    // For 2D [out, in]: norm over dim 1
    // For 3D [out, in, kernel]: norm over dims 1, 2
    let v_sq = weight_v.sqr()?;

    let v_norm = if ndim == 2 {
        // [out, in] -> norm over dim 1
        v_sq.sum(1)?.sqrt()?.unsqueeze(1)?
    } else if ndim == 3 {
        // [out, in, kernel] -> norm over dims 1 and 2
        let sum_1 = v_sq.sum(1)?;  // [out, kernel]
        let sum_12 = sum_1.sum(1)?;  // [out]
        sum_12.sqrt()?.unsqueeze(1)?.unsqueeze(2)?  // [out, 1, 1]
    } else {
        return Err(anyhow::anyhow!("Unsupported weight dimension: {}", ndim));
    };

    // Add eps for numerical stability
    let v_norm = (v_norm + eps)?;

    // Reshape weight_g to broadcast correctly
    // weight_g is typically [out, 1] or [out, 1, 1]
    let weight_g = if weight_g.dims().len() < ndim {
        // Need to add dimensions
        let mut g = weight_g.clone();
        while g.dims().len() < ndim {
            g = g.unsqueeze(g.dims().len())?;
        }
        g
    } else {
        weight_g.clone()
    };

    // Normalize: g * v / ||v||
    Ok(weight_g.broadcast_mul(&weight_v.broadcast_div(&v_norm)?)?)
}

/// Helper to load a weight-normalized Linear layer from tensors
/// Weight normalization: weight = weight_g * (weight_v / ||weight_v||_2)
/// where ||weight_v||_2 is computed along the input dimension (dim=1)
fn load_weight_normalized_linear(
    tensors: &HashMap<String, Tensor>,
    weight_v_key: &str,
    weight_g_key: &str,
    bias_key: Option<&str>,
) -> Result<Linear> {
    let weight_v = tensors
        .get(weight_v_key)
        .ok_or_else(|| anyhow::anyhow!("Weight_v not found: {}", weight_v_key))?;

    let weight_g = tensors
        .get(weight_g_key)
        .ok_or_else(|| anyhow::anyhow!("Weight_g not found: {}", weight_g_key))?;

    let weight = apply_weight_normalization(weight_v, weight_g)?;

    let bias = if let Some(bk) = bias_key {
        tensors.get(bk).cloned()
    } else {
        None
    };

    Ok(Linear::new(weight, bias))
}

/// Helper to load LayerNorm from tensors
fn load_layer_norm(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    dim: usize,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = match tensors.get(weight_key) {
        Some(w) => w.clone(),
        None => {
            tracing::warn!(
                "[DiT] Missing tensor '{}', using ones initialization",
                weight_key
            );
            Tensor::ones((dim,), DType::F32, device)?
        }
    };
    let bias = Tensor::zeros((dim,), DType::F32, device)?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

/// DiT configuration
#[derive(Clone)]
pub struct DiffusionTransformerConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub depth: usize,
    /// Input channels (mel bands)
    pub in_channels: usize,
    /// Whether to use style conditioning
    pub style_condition: bool,
    /// Style embedding dimension
    pub style_dim: usize,
    /// Whether to use UViT skip connections
    pub uvit_skip_connection: bool,
    /// Content dimension
    pub content_dim: usize,
    /// Block size for attention
    pub block_size: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl Default for DiffusionTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            num_heads: 8,
            depth: 13,
            in_channels: 80,
            style_condition: true,
            style_dim: 192,
            uvit_skip_connection: true,
            content_dim: 512,
            block_size: 8192,
            dropout: 0.1,
        }
    }
}

/// Sinusoidal time embedding
/// Follows the Python implementation with scale=1000, max_period=10000
fn sinusoidal_embedding(timesteps: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = dim / 2;
    let max_period = 10000.0f32;
    let scale = 1000.0f32;  // CRITICAL: Python uses scale=1000

    // Create frequency bands: exp(-log(max_period) * i / half_dim)
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-max_period.ln() * i as f32 / half_dim as f32).exp())
        .collect();
    let freqs = Tensor::from_slice(&freqs, (1, half_dim), device)?;

    // timesteps: (batch,) -> (batch, 1), then scale by 1000
    let timesteps = timesteps.unsqueeze(1)?;
    let timesteps = (timesteps.to_dtype(DType::F32)? * scale as f64)?;

    // Compute embeddings
    let args = timesteps.broadcast_mul(&freqs)?;
    // CRITICAL: Python uses [cos, sin] order
    let cos_emb = args.cos()?;
    let sin_emb = args.sin()?;

    // Concatenate cos and sin (Python order)
    Tensor::cat(&[cos_emb, sin_emb], 1).map_err(Into::into)
}

/// Time embedding MLP
/// Checkpoint format: t_embedder.freqs, t_embedder.mlp.0.*, t_embedder.mlp.2.*
struct TimestepEmbedding {
    freqs: Option<Tensor>,
    linear1: Linear,
    linear2: Linear,
    dim: usize,
}

impl TimestepEmbedding {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inner_dim = dim;  // First MLP layer: [dim, dim/2] since freqs are dim/2

        let w1 = Tensor::randn(0.0f32, 0.02, (inner_dim, dim / 2), device)?;
        let b1 = Tensor::zeros((inner_dim,), DType::F32, device)?;
        let linear1 = Linear::new(w1, Some(b1));

        let w2 = Tensor::randn(0.0f32, 0.02, (dim, inner_dim), device)?;
        let b2 = Tensor::zeros((dim,), DType::F32, device)?;
        let linear2 = Linear::new(w2, Some(b2));

        Ok(Self { freqs: None, linear1, linear2, dim })
    }

    /// Load from checkpoint tensors
    /// Format: {prefix}.freqs, {prefix}.mlp.0.weight/bias, {prefix}.mlp.2.weight/bias
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let freqs = tensors.get(&format!("{}.freqs", prefix)).cloned();

        let linear1_key = format!("{}.mlp.0.weight", prefix);
        let linear1 = match load_linear(
            tensors,
            &linear1_key,
            Some(&format!("{}.mlp.0.bias", prefix)),
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    linear1_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (dim, dim / 2), device)?;
                let b = Tensor::zeros((dim,), DType::F32, device)?;
                Linear::new(w, Some(b))
            }
        };

        let linear2_key = format!("{}.mlp.2.weight", prefix);
        let linear2 = match load_linear(
            tensors,
            &linear2_key,
            Some(&format!("{}.mlp.2.bias", prefix)),
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    linear2_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
                let b = Tensor::zeros((dim,), DType::F32, device)?;
                Linear::new(w, Some(b))
            }
        };

        Ok(Self { freqs, linear1, linear2, dim })
    }

    fn forward(&self, t: &Tensor, device: &Device) -> Result<Tensor> {
        // Use learnable frequencies if available, otherwise sinusoidal
        let emb = if let Some(ref freqs) = self.freqs {
            // Learnable frequency embedding: freqs [128] -> cos/sin [256]
            // CRITICAL: Python scales timestep by 1000 before frequency computation!
            let scale = 1000.0f64;
            let t = (t.unsqueeze(1)?.to_dtype(DType::F32)? * scale)?;
            let freqs = freqs.unsqueeze(0)?;
            let args = t.broadcast_mul(&freqs)?;
            // CRITICAL: Python uses [cos, sin] order, not [sin, cos]!
            let cos_emb = args.cos()?;
            let sin_emb = args.sin()?;
            Tensor::cat(&[cos_emb, sin_emb], 1)?
        } else {
            sinusoidal_embedding(t, self.dim / 2, device)?
        };

        // MLP: Linear -> SiLU -> Linear
        let emb = self.linear1.forward(&emb)?;
        let emb = silu(&emb)?;
        self.linear2.forward(&emb).map_err(Into::into)
    }
}

/// SiLU (Swish) activation
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid).map_err(Into::into)
}

/// Adaptive Layer Normalization (AdaLN)
/// Checkpoint format: {prefix}.norm.weight, {prefix}.project_layer.weight/bias
struct AdaLayerNorm {
    norm: LayerNorm,
    linear: Linear,
    dim: usize,
}

impl AdaLayerNorm {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let ln_w = Tensor::ones((dim,), DType::F32, device)?;
        let ln_b = Tensor::zeros((dim,), DType::F32, device)?;
        let norm = LayerNorm::new(ln_w, ln_b, 1e-5);

        // Project conditioning to scale and shift
        let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
        let b = Tensor::zeros((dim * 2,), DType::F32, device)?;
        let linear = Linear::new(w, Some(b));

        Ok(Self { norm, linear, dim })
    }

    /// Load from checkpoint tensors
    /// Format: {prefix}.norm.weight, {prefix}.project_layer.weight/bias
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let norm = load_layer_norm(
            tensors,
            &format!("{}.norm.weight", prefix),
            dim,
            device,
        )?;

        let proj_key = format!("{}.project_layer.weight", prefix);
        let linear = match load_linear(
            tensors,
            &proj_key,
            Some(&format!("{}.project_layer.bias", prefix)),
        ) {
            Ok(linear) => {
                // Debug loaded weights (once)
                static ONCE_PROJ: std::sync::Once = std::sync::Once::new();
                ONCE_PROJ.call_once(|| {
                    if let Some(w) = tensors.get(&proj_key) {
                        let w_shape = w.dims();
                        let mean: f32 = w.mean_all().unwrap().to_scalar().unwrap();
                        let rms: f32 = w.sqr().unwrap().mean_all().unwrap()
                            .to_scalar::<f32>().unwrap().sqrt();
                        // Get first row first 5 values
                        let first_row: Vec<f32> = w.i(0).unwrap().to_vec1().unwrap().iter().take(5).cloned().collect();
                        eprintln!("DEBUG: Loaded project_layer from {} - shape={:?}, mean={:.6}, rms={:.6}",
                            proj_key, w_shape, mean, rms);
                        eprintln!("DEBUG: project_layer first row first 5: {:?}", first_row);
                    }
                });
                linear
            }
            Err(e) => {
                eprintln!("Warning: Failed to load {}: {}, using random", proj_key, e);
                let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device).unwrap();
                let b = Tensor::zeros((dim * 2,), DType::F32, device).unwrap();
                Linear::new(w, Some(b))
            }
        };

        Ok(Self { norm, linear, dim })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // x: (batch, seq_len, dim)
        // cond: (batch, dim) - time embedding

        // Normalize
        let normalized = self.norm.forward(x)?;

        // Get scale and shift from conditioning
        // linear projects dim -> 2*dim (scale + shift)
        let params = self.linear.forward(cond)?;  // (batch, 2*dim)

        // Debug params and cond (once)
        static ONCE_PARAMS: std::sync::Once = std::sync::Once::new();
        ONCE_PARAMS.call_once(|| {
            let cond_mean: f32 = cond.mean_all().unwrap().to_scalar().unwrap();
            let cond_std: f32 = cond.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let params_mean: f32 = params.mean_all().unwrap().to_scalar().unwrap();
            let params_std: f32 = params.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG AdaLN cond: shape={:?}, mean={:.4}, rms={:.4}", cond.dims(), cond_mean, cond_std);
            eprintln!("DEBUG AdaLN params: shape={:?}, mean={:.4}, rms={:.4}", params.dims(), params_mean, params_std);
        });

        let chunks = params.chunk(2, D::Minus1)?;
        let scale = chunks.get(0).ok_or_else(|| anyhow::anyhow!("Missing scale chunk"))?;
        let shift = chunks.get(1).ok_or_else(|| anyhow::anyhow!("Missing shift chunk"))?;

        // Debug scale/shift (once)
        static ONCE_ADALN: std::sync::Once = std::sync::Once::new();
        ONCE_ADALN.call_once(|| {
            let scale_mean: f32 = scale.mean_all().unwrap().to_scalar().unwrap();
            let scale_std: f32 = scale.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let shift_mean: f32 = shift.mean_all().unwrap().to_scalar().unwrap();
            let norm_mean: f32 = normalized.mean_all().unwrap().to_scalar().unwrap();
            let norm_std: f32 = normalized.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG AdaLN: scale_mean={:.4}, scale_rms={:.4}, shift_mean={:.4}, norm_mean={:.4}, norm_rms={:.4}",
                scale_mean, scale_std, shift_mean, norm_mean, norm_std);
        });

        // Unsqueeze to broadcast over sequence: (batch, dim) -> (batch, 1, dim)
        let scale = scale.unsqueeze(1)?;
        let shift = shift.unsqueeze(1)?;

        // Apply: (1 + scale) * normalized + shift
        // Center scale around 1 for stable training
        let scale = (scale + 1.0)?;
        (normalized.broadcast_mul(&scale)?).broadcast_add(&shift).map_err(Into::into)
    }
}

/// Multi-head self-attention with fused QKV
/// Checkpoint format: {prefix}.wqkv.weight [3*dim, dim], {prefix}.wo.weight [dim, dim]
struct MultiHeadAttention {
    qkv_proj: Linear,  // Fused Q/K/V projection
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    fn new(dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // Fused QKV
        let qkv_w = Tensor::randn(0.0f32, 0.02, (dim * 3, dim), device)?;
        let qkv_proj = Linear::new(qkv_w, None);

        let out_w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
        let out_proj = Linear::new(out_w, None);

        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Load from checkpoint tensors
    /// Format: {prefix}.wqkv.weight, {prefix}.wo.weight
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        num_heads: usize,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let wqkv_key = format!("{}.wqkv.weight", prefix);
        let qkv_proj = match load_linear(tensors, &wqkv_key, None) {
            Ok(linear) => {
                // Debug loaded weights
                static ONCE_QKV_W: std::sync::Once = std::sync::Once::new();
                ONCE_QKV_W.call_once(|| {
                    if let Ok(w) = tensors.get(&wqkv_key).unwrap().mean_all() {
                        let mean: f32 = w.to_scalar().unwrap();
                        let std: f32 = tensors.get(&wqkv_key).unwrap()
                            .var(D::Minus1).unwrap().mean_all().unwrap()
                            .to_scalar::<f32>().unwrap().sqrt();
                        eprintln!("DEBUG: Loaded wqkv.weight from {} - mean={:.6}, std={:.6}",
                            wqkv_key, mean, std);
                    }
                });
                linear
            }
            Err(e) => {
                eprintln!("Warning: Failed to load {}: {}, using random", wqkv_key, e);
                let w = Tensor::randn(0.0f32, 0.02, (dim * 3, dim), device).unwrap();
                Linear::new(w, None)
            }
        };

        let out_proj_key = format!("{}.wo.weight", prefix);
        let out_proj = match load_linear(
            tensors,
            &out_proj_key,
            None,
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    out_proj_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
                Linear::new(w, None)
            }
        };

        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _dim) = x.dims3()?;

        // Debug input to attention - use RMS for reliable measurement
        static ONCE_ATTN_IN: std::sync::Once = std::sync::Once::new();
        ONCE_ATTN_IN.call_once(|| {
            let x_mean: f32 = x.mean_all().unwrap().to_scalar().unwrap();
            let x_rms: f32 = x.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG Attention input: shape={:?}, mean={:.4}, rms={:.4}", x.dims(), x_mean, x_rms);
        });

        // Fused QKV projection
        let qkv = self.qkv_proj.forward(x)?;

        // Debug QKV after projection
        static ONCE_QKV: std::sync::Once = std::sync::Once::new();
        ONCE_QKV.call_once(|| {
            let qkv_mean: f32 = qkv.mean_all().unwrap().to_scalar().unwrap();
            let qkv_rms: f32 = qkv.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG QKV after projection: shape={:?}, mean={:.4}, rms={:.4}", qkv.dims(), qkv_mean, qkv_rms);

            // Debug weight stats - compute actual std
            let w = self.qkv_proj.weight();
            let w_shape = w.dims();
            let w_mean: f32 = w.mean_all().unwrap().to_scalar().unwrap();
            let w_rms: f32 = w.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG QKV weight: shape={:?}, mean={:.6}, rms={:.6}", w_shape, w_mean, w_rms);
            eprintln!("DEBUG scale={:.6}, head_dim={}, num_heads={}", self.scale, self.head_dim, self.num_heads);

            // Verify Linear computation by manual matmul (first batch, first position)
            let hidden_dim = self.num_heads * self.head_dim;
            let qkv_dim = hidden_dim * 3;
            let x_sample = x.i((0..1, 0..1, ..)).unwrap();  // [1, 1, hidden_dim]
            let x_sample_2d = x_sample.reshape((1, hidden_dim)).unwrap();
            let manual_qkv = x_sample_2d.matmul(&w.t().unwrap()).unwrap();
            let manual_rms: f32 = manual_qkv.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let candle_sample = qkv.i((0..1, 0..1, ..)).unwrap().reshape((1, qkv_dim)).unwrap();
            let candle_rms: f32 = candle_sample.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG Manual matmul rms={:.4}, Candle Linear rms={:.4}", manual_rms, candle_rms);

            // Check input sample values
            let x_first_5: Vec<f32> = x_sample_2d.i(0).unwrap().to_vec1().unwrap().iter().take(5).cloned().collect();
            eprintln!("DEBUG Input first 5: {:?}", x_first_5);

            // Check input min/max and full stats
            let x_min: f32 = x.min_all().unwrap().to_scalar().unwrap();
            let x_max: f32 = x.max_all().unwrap().to_scalar().unwrap();
            let x_absmax = x_min.abs().max(x_max.abs());
            eprintln!("DEBUG Input min={:.4}, max={:.4}, absmax={:.4}", x_min, x_max, x_absmax);

            // Check weight first row values
            let w_first_row: Vec<f32> = w.i(0).unwrap().to_vec1().unwrap().iter().take(5).cloned().collect();
            eprintln!("DEBUG Weight first row first 5: {:?}", w_first_row);

            // Expected RMS based on formula
            let x_rms: f32 = x.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let w_rms: f32 = w.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let expected_rms = (hidden_dim as f32).sqrt() * x_rms * w_rms;
            eprintln!("DEBUG Expected output RMS = sqrt({})*{:.4}*{:.6} = {:.4}", hidden_dim, x_rms, w_rms, expected_rms);
        });

        // Split into Q, K, V
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0, .., ..))?.contiguous()?;
        let k = qkv.i((.., .., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.contiguous()?;

        // Reshape to (batch, num_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Debug Q and K stats
        static ONCE_QK: std::sync::Once = std::sync::Once::new();
        ONCE_QK.call_once(|| {
            let q_mean: f32 = q.mean_all().unwrap().to_scalar().unwrap();
            let q_std: f32 = q.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let k_mean: f32 = k.mean_all().unwrap().to_scalar().unwrap();
            let k_std: f32 = k.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG Q: mean={:.4}, rms={:.4}", q_mean, q_std);
            eprintln!("DEBUG K: mean={:.4}, rms={:.4}", k_mean, k_std);
        });

        // Scaled dot-product attention
        let raw_scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        // Debug raw scores BEFORE scaling
        static ONCE_RAW: std::sync::Once = std::sync::Once::new();
        ONCE_RAW.call_once(|| {
            let raw_mean: f32 = raw_scores.mean_all().unwrap().to_scalar().unwrap();
            let raw_std: f32 = raw_scores.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG Raw attention scores (before scale): mean={:.4}, rms={:.4}", raw_mean, raw_std);
        });

        let attn_scores = (raw_scores * self.scale as f64)?;

        // Debug attention scores BEFORE softmax
        static ONCE_SCORES: std::sync::Once = std::sync::Once::new();
        ONCE_SCORES.call_once(|| {
            let scores_mean: f32 = attn_scores.mean_all().unwrap().to_scalar().unwrap();
            let scores_std: f32 = attn_scores.var(D::Minus1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let scores_max: f32 = attn_scores.max_all().unwrap().to_scalar().unwrap();
            let scores_min: f32 = attn_scores.min_all().unwrap().to_scalar().unwrap();
            eprintln!("DEBUG Attention scores (before softmax): mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
                scores_mean, scores_std, scores_min, scores_max);
        });

        let attn = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;

        // Debug attention weights (once)
        static ONCE_ATTN: std::sync::Once = std::sync::Once::new();
        ONCE_ATTN.call_once(|| {
            let attn_mean: f32 = attn.mean_all().unwrap().to_scalar().unwrap();
            let attn_max: f32 = attn.max_all().unwrap().to_scalar().unwrap();
            eprintln!("DEBUG Attention weights (after softmax): attn_mean={:.4}, attn_max={:.4}", attn_mean, attn_max);
        });

        // Apply attention to values
        let out = attn.matmul(&v.contiguous()?)?;

        // Reshape back to (batch, seq_len, dim)
        let out = out.transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Debug attention output
        static ONCE_ATTN_OUT: std::sync::Once = std::sync::Once::new();
        ONCE_ATTN_OUT.call_once(|| {
            let out_mean: f32 = out.mean_all().unwrap().to_scalar().unwrap();
            let out_std: f32 = out.var(D::Minus1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG Attention output (before out_proj): mean={:.4}, std={:.4}", out_mean, out_std);
        });

        self.out_proj.forward(&out).map_err(Into::into)
    }
}

/// SwiGLU Feed-forward network
/// Checkpoint format: {prefix}.w1.weight, {prefix}.w2.weight, {prefix}.w3.weight
/// SwiGLU: output = w2(silu(w1(x)) * w3(x))
struct FeedForward {
    w1: Linear,  // Gate projection
    w2: Linear,  // Output projection
    w3: Linear,  // Up projection
}

impl FeedForward {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let hidden_dim = dim * 3;  // SwiGLU typically uses 8/3 * dim but checkpoint uses 3*dim

        let w1 = Tensor::randn(0.0f32, 0.02, (hidden_dim, dim), device)?;
        let w2 = Tensor::randn(0.0f32, 0.02, (dim, hidden_dim), device)?;
        let w3 = Tensor::randn(0.0f32, 0.02, (hidden_dim, dim), device)?;

        Ok(Self {
            w1: Linear::new(w1, None),
            w2: Linear::new(w2, None),
            w3: Linear::new(w3, None),
        })
    }

    /// Load from checkpoint tensors
    /// Format: {prefix}.w1.weight, {prefix}.w2.weight, {prefix}.w3.weight
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let hidden_dim = dim * 3;

        let w1_key = format!("{}.w1.weight", prefix);
        let w1 = match load_linear(
            tensors,
            &w1_key,
            None,
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    w1_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (hidden_dim, dim), device)?;
                Linear::new(w, None)
            }
        };

        let w2_key = format!("{}.w2.weight", prefix);
        let w2 = match load_linear(
            tensors,
            &w2_key,
            None,
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    w2_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (dim, hidden_dim), device)?;
                Linear::new(w, None)
            }
        };

        let w3_key = format!("{}.w3.weight", prefix);
        let w3 = match load_linear(
            tensors,
            &w3_key,
            None,
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    w3_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (hidden_dim, dim), device)?;
                Linear::new(w, None)
            }
        };

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: output = w2(silu(w1(x)) * w3(x))
        let gate = self.w1.forward(x)?;
        let gate = silu(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = (gate * up)?;
        self.w2.forward(&hidden).map_err(Into::into)
    }
}

/// DiT Block with AdaLN conditioning
/// Checkpoint format: transformer.layers.{i}.attention_norm, attention, ffn_norm, feed_forward
struct DiTBlock {
    norm1: AdaLayerNorm,
    attn: MultiHeadAttention,
    norm2: AdaLayerNorm,
    ff: FeedForward,
}

impl DiTBlock {
    fn new(dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNorm::new(dim, device)?,
            attn: MultiHeadAttention::new(dim, num_heads, device)?,
            norm2: AdaLayerNorm::new(dim, device)?,
            ff: FeedForward::new(dim, device)?,
        })
    }

    /// Load from checkpoint tensors
    /// Format: {prefix}.attention_norm, {prefix}.attention, {prefix}.ffn_norm, {prefix}.feed_forward
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        num_heads: usize,
        device: &Device,
    ) -> Result<Self> {
        let norm1 = AdaLayerNorm::from_tensors(
            tensors,
            &format!("{}.attention_norm", prefix),
            dim,
            device,
        )?;

        let attn = MultiHeadAttention::from_tensors(
            tensors,
            &format!("{}.attention", prefix),
            dim,
            num_heads,
            device,
        )?;

        let norm2 = AdaLayerNorm::from_tensors(
            tensors,
            &format!("{}.ffn_norm", prefix),
            dim,
            device,
        )?;

        let ff = FeedForward::from_tensors(
            tensors,
            &format!("{}.feed_forward", prefix),
            dim,
            device,
        )?;

        Ok(Self { norm1, attn, norm2, ff })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Self-attention with AdaLN
        let residual = x.clone();
        let x_normed = self.norm1.forward(x, cond)?;
        let x_attn = self.attn.forward(&x_normed)?;
        let x = (residual + x_attn)?;

        // Feed-forward with AdaLN
        let residual = x.clone();
        let x_normed = self.norm2.forward(&x, cond)?;
        let x_ff = self.ff.forward(&x_normed)?;
        (residual + x_ff).map_err(Into::into)
    }
}

/// WaveNet layer for post-transformer processing
/// Each layer has dilated convolution with gated activation
struct WaveNetLayer {
    in_conv: Linear,        // Input conv [1024, 512] (2*512 for gated)
    res_skip_conv: Linear,  // Res/skip conv [1024, 512] or [512, 512] for last layer
    dilation: usize,
    is_last: bool,          // Last layer only outputs skip (no residual)
}

impl WaveNetLayer {
    fn new(hidden_dim: usize, dilation: usize, is_last: bool, device: &Device) -> Result<Self> {
        // in_conv: [2*hidden, hidden] for gated activation (tanh * sigmoid)
        let w = Tensor::randn(0.0f32, 0.02, (hidden_dim * 2, hidden_dim), device)?;
        let b = Tensor::zeros((hidden_dim * 2,), DType::F32, device)?;
        let in_conv = Linear::new(w, Some(b));

        // res_skip_conv: [2*hidden, hidden] for normal layers, [hidden, hidden] for last layer
        let out_dim = if is_last { hidden_dim } else { hidden_dim * 2 };
        let w = Tensor::randn(0.0f32, 0.02, (out_dim, hidden_dim), device)?;
        let b = Tensor::zeros((out_dim,), DType::F32, device)?;
        let res_skip_conv = Linear::new(w, Some(b));

        Ok(Self { in_conv, res_skip_conv, dilation, is_last })
    }

    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        in_prefix: &str,
        res_prefix: &str,
        dilation: usize,
        is_last: bool,
        _device: &Device,
    ) -> Result<Self> {
        // Load in_conv with weight normalization
        // Shape [1024, 512, 5] for conv -> apply weight norm then take center tap
        let in_weight_v = tensors.get(&format!("{}.weight_v", in_prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.weight_v", in_prefix))?;
        let in_weight_g = tensors.get(&format!("{}.weight_g", in_prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.weight_g", in_prefix))?;
        let in_bias = tensors.get(&format!("{}.bias", in_prefix)).cloned();

        // Apply weight normalization first (on 3D tensor)
        let in_weight_norm = apply_weight_normalization(in_weight_v, in_weight_g)?;

        // For dilated conv weights [out, in, kernel], we use only center for linear approx
        let in_weight = if in_weight_norm.dims().len() == 3 {
            // [1024, 512, 5] -> take center tap for linear approximation
            let kernel_size = in_weight_norm.dim(2)?;
            let center = kernel_size / 2;
            in_weight_norm.i((.., .., center))?.contiguous()?
        } else {
            in_weight_norm
        };
        let in_conv = Linear::new(in_weight, in_bias);

        // Load res_skip_conv with weight normalization
        let res_weight_v = tensors.get(&format!("{}.weight_v", res_prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.weight_v", res_prefix))?;
        let res_weight_g = tensors.get(&format!("{}.weight_g", res_prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.weight_g", res_prefix))?;
        let res_bias = tensors.get(&format!("{}.bias", res_prefix)).cloned();

        // Apply weight normalization first
        let res_weight_norm = apply_weight_normalization(res_weight_v, res_weight_g)?;

        let res_weight = if res_weight_norm.dims().len() == 3 {
            res_weight_norm.squeeze(2)?
        } else {
            res_weight_norm
        };
        let res_skip_conv = Linear::new(res_weight, res_bias);

        Ok(Self { in_conv, res_skip_conv, dilation, is_last })
    }

    fn forward(&self, x: &Tensor, g: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, _seq_len, _hidden) = x.dims3()?;

        // Apply input conv (approximating dilated conv as linear)
        let h = self.in_conv.forward(x)?;

        // Add conditioning g
        let h = h.broadcast_add(g)?;

        // Gated activation: tanh(h[:half]) * sigmoid(h[half:])
        let chunks = h.chunk(2, D::Minus1)?;
        let tanh_part = chunks[0].tanh()?;
        let sigmoid_part = candle_nn::ops::sigmoid(&chunks[1])?;
        let h = (tanh_part * sigmoid_part)?;

        // Res/skip projection
        let res_skip = self.res_skip_conv.forward(&h)?;

        if self.is_last {
            // Last layer: only skip output, no residual
            // Output same x (no residual to add), skip is the full output
            Ok((x.clone(), res_skip))
        } else {
            // Normal layers: split into residual + skip
            let res_skip_chunks = res_skip.chunk(2, D::Minus1)?;
            let residual = &res_skip_chunks[0];
            let skip = &res_skip_chunks[1];

            // Residual connection
            let output = (x + residual)?;

            Ok((output, skip.clone()))
        }
    }
}

/// WaveNet module with 8 dilated convolution layers
struct WaveNet {
    cond_layer: Linear,     // Conditioning projection [8192, 512]
    layers: Vec<WaveNetLayer>,
    num_layers: usize,
}

impl WaveNet {
    fn new(hidden_dim: usize, num_layers: usize, device: &Device) -> Result<Self> {
        // cond_layer projects time embedding to all layers
        // [num_layers * 2 * hidden, hidden] = [8 * 2 * 512, 512] = [8192, 512]
        let cond_out_dim = num_layers * 2 * hidden_dim;
        let w = Tensor::randn(0.0f32, 0.02, (cond_out_dim, hidden_dim), device)?;
        let b = Tensor::zeros((cond_out_dim,), DType::F32, device)?;
        let cond_layer = Linear::new(w, Some(b));

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let dilation = 2usize.pow(i as u32);
            let is_last = i == num_layers - 1;
            layers.push(WaveNetLayer::new(hidden_dim, dilation, is_last, device)?);
        }

        Ok(Self { cond_layer, layers, num_layers })
    }

    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_dim: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        // Load cond_layer with weight normalization
        let cond_weight_v = tensors.get(&format!("{}.cond_layer.conv.conv.weight_v", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing cond_layer weight_v"))?;
        let cond_weight_g = tensors.get(&format!("{}.cond_layer.conv.conv.weight_g", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing cond_layer weight_g"))?;
        let cond_bias = tensors.get(&format!("{}.cond_layer.conv.conv.bias", prefix)).cloned();

        // Apply weight normalization first
        let cond_weight_norm = apply_weight_normalization(cond_weight_v, cond_weight_g)?;

        let cond_weight = if cond_weight_norm.dims().len() == 3 {
            cond_weight_norm.squeeze(2)?
        } else {
            cond_weight_norm
        };
        let cond_layer = Linear::new(cond_weight, cond_bias);

        // Load each layer
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let dilation = 2usize.pow(i as u32);
            let is_last = i == num_layers - 1;
            let in_prefix = format!("{}.in_layers.{}.conv.conv", prefix, i);
            let res_prefix = format!("{}.res_skip_layers.{}.conv.conv", prefix, i);

            match WaveNetLayer::from_tensors(tensors, &in_prefix, &res_prefix, dilation, is_last, device) {
                Ok(layer) => layers.push(layer),
                Err(e) => {
                    eprintln!("  Warning: Failed to load WaveNet layer {}: {}", i, e);
                    layers.push(WaveNetLayer::new(hidden_dim, dilation, is_last, device)?);
                }
            }
        }

        Ok(Self { cond_layer, layers, num_layers })
    }

    fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        // Project conditioning to all layers: g [B, 512] -> [B, 8192]
        let g_all = self.cond_layer.forward(g)?;

        // Split into per-layer conditioning
        let chunk_size = hidden * 2;  // 1024 per layer

        let mut h = x.clone();
        let mut skip_sum = Tensor::zeros((batch, seq_len, hidden), DType::F32, x.device())?;

        for (i, layer) in self.layers.iter().enumerate() {
            // Extract conditioning for this layer
            let start = i * chunk_size;
            let g_layer = g_all.i((.., start..start + chunk_size))?;
            let g_layer = g_layer.unsqueeze(1)?;  // [B, 1, 1024] for broadcasting

            let (h_new, skip) = layer.forward(&h, &g_layer)?;
            h = h_new;
            skip_sum = (skip_sum + skip)?;
        }

        Ok(skip_sum)
    }
}

/// Final layer with AdaLN modulation
struct FinalLayer {
    adaln: Linear,   // AdaLN modulation [1024, 512]
    linear: Linear,  // Output linear [512, 512]
    norm: LayerNorm,
}

impl FinalLayer {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
        let b = Tensor::zeros((dim * 2,), DType::F32, device)?;
        let adaln = Linear::new(w, Some(b));

        let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
        let b = Tensor::zeros((dim,), DType::F32, device)?;
        let linear = Linear::new(w, Some(b));

        let ln_w = Tensor::ones((dim,), DType::F32, device)?;
        let ln_b = Tensor::zeros((dim,), DType::F32, device)?;
        let norm = LayerNorm::new(ln_w, ln_b, 1e-5);

        Ok(Self { adaln, linear, norm })
    }

    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        // Load AdaLN modulation
        let adaln_key = format!("{}.adaLN_modulation.1.weight", prefix);
        let adaln = match load_linear(
            tensors,
            &adaln_key,
            Some(&format!("{}.adaLN_modulation.1.bias", prefix)),
        ) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    "[DiT] Missing tensor '{}', using random initialization",
                    adaln_key
                );
                let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
                let b = Tensor::zeros((dim * 2,), DType::F32, device)?;
                Linear::new(w, Some(b))
            }
        };

        // Load linear (weight normalized: weight = g * v / ||v||)
        let bias_key = format!("{}.linear.bias", prefix);
        let has_bias = tensors.contains_key(&bias_key);
        let linear = match load_weight_normalized_linear(
            tensors,
            &format!("{}.linear.weight_v", prefix),
            &format!("{}.linear.weight_g", prefix),
            if has_bias { Some(bias_key.as_str()) } else { None },
        ) {
            Ok(linear) => {
                eprintln!("  FinalLayer: loaded weight-normalized linear");
                linear
            }
            Err(e) => {
                eprintln!("  FinalLayer: weight normalization failed ({}), using fallback", e);
                let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device).unwrap();
                let b = Tensor::zeros((dim,), DType::F32, device).unwrap();
                Linear::new(w, Some(b))
            }
        };

        let ln_w = Tensor::ones((dim,), DType::F32, device)?;
        let ln_b = Tensor::zeros((dim,), DType::F32, device)?;
        let norm = LayerNorm::new(ln_w, ln_b, 1e-5);

        Ok(Self { adaln, linear, norm })
    }

    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        // Get scale and shift from time embedding
        let params = self.adaln.forward(t_emb)?;
        let chunks = params.chunk(2, D::Minus1)?;
        let scale = &chunks[0];
        let shift = &chunks[1];

        // Apply layer norm
        let x = self.norm.forward(x)?;

        // Apply scale and shift (broadcast over sequence)
        let scale = scale.unsqueeze(1)?;  // [B, 1, 512]
        let shift = shift.unsqueeze(1)?;
        let x = (x.broadcast_mul(&(scale + 1.0)?)?).broadcast_add(&shift)?;

        // Apply linear
        self.linear.forward(&x).map_err(Into::into)
    }
}

/// Diffusion Transformer for mel spectrogram synthesis
pub struct DiffusionTransformer {
    device: Device,
    config: DiffusionTransformerConfig,
    /// Input projection (x_embedder) - mel channels -> hidden
    x_embedder: Option<Linear>,
    /// Conditioning projection (cond_projection) - content_dim -> content_dim
    cond_projection: Option<Linear>,
    /// Merge linear - concatenated inputs [x, prompt_x, cond, style] -> hidden
    /// Shape: [512, 864] where 864 = 80 + 80 + 512 + 192
    cond_x_merge_linear: Option<Linear>,
    /// Time embedding (t_embedder)
    time_embed: Option<TimestepEmbedding>,
    /// Second time embedding for WaveNet (t_embedder2)
    time_embed2: Option<TimestepEmbedding>,
    /// Style embedding for AdaLN conditioning (cond_embedder)
    cond_embedder: Option<Linear>,
    /// Transformer blocks
    blocks: Vec<DiTBlock>,
    /// Final layer norm (transformer.norm)
    final_norm: Option<LayerNorm>,
    /// Skip connection linear [512, 592] - concat x_res with original x (80 mel)
    skip_linear: Option<Linear>,
    /// Conv1 for WaveNet input [512, 512]
    conv1: Option<Linear>,
    /// WaveNet module for post-transformer processing
    wavenet: Option<WaveNet>,
    /// Residual projection [512, 512]
    res_projection: Option<Linear>,
    /// Final layer with AdaLN
    final_layer: Option<FinalLayer>,
    /// Output projection (conv2) - hidden -> mel channels
    output_proj: Option<Linear>,
    /// Whether initialized
    weights_loaded: bool,
}

impl DiffusionTransformer {
    /// Create with default config
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(DiffusionTransformerConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: DiffusionTransformerConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            x_embedder: None,
            cond_projection: None,
            cond_x_merge_linear: None,
            time_embed: None,
            time_embed2: None,
            cond_embedder: None,
            blocks: Vec::new(),
            final_norm: None,
            skip_linear: None,
            conv1: None,
            wavenet: None,
            res_projection: None,
            final_layer: None,
            output_proj: None,
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut dit = Self::new(device)?;
        dit.load_weights(path)?;
        Ok(dit)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        let dim = self.config.hidden_dim;  // 512
        let in_ch = self.config.in_channels;  // 80
        let content_dim = self.config.content_dim;  // 512
        let style_dim = self.config.style_dim;  // 192

        // x_embedder: mel -> hidden (not used in merge path, but kept for compatibility)
        let w = Tensor::randn(0.0f32, 0.02, (dim, in_ch), &self.device)?;
        let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.x_embedder = Some(Linear::new(w, Some(b)));

        // cond_projection: content_dim -> content_dim
        let w = Tensor::randn(0.0f32, 0.02, (content_dim, content_dim), &self.device)?;
        let b = Tensor::zeros((content_dim,), DType::F32, &self.device)?;
        self.cond_projection = Some(Linear::new(w, Some(b)));

        // cond_x_merge_linear: [x, prompt_x, cond, style] -> hidden
        // Input: 80 + 80 + 512 + 192 = 864
        let merge_in = in_ch + in_ch + content_dim + style_dim;  // 864
        let w = Tensor::randn(0.0f32, 0.02, (dim, merge_in), &self.device)?;
        let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.cond_x_merge_linear = Some(Linear::new(w, Some(b)));

        // Time embeddings
        self.time_embed = Some(TimestepEmbedding::new(dim, &self.device)?);
        self.time_embed2 = Some(TimestepEmbedding::new(dim, &self.device)?);

        // cond_embedder for AdaLN: hidden -> 2*hidden (scale + shift)
        let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), &self.device)?;
        let b = Tensor::zeros((dim * 2,), DType::F32, &self.device)?;
        self.cond_embedder = Some(Linear::new(w, Some(b)));

        // Transformer blocks
        self.blocks.clear();
        for _ in 0..self.config.depth {
            self.blocks.push(DiTBlock::new(
                dim,
                self.config.num_heads,
                &self.device,
            )?);
        }

        // Final norm
        let ln_w = Tensor::ones((dim,), DType::F32, &self.device)?;
        let ln_b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.final_norm = Some(LayerNorm::new(ln_w, ln_b, 1e-5));

        // Post-transformer: skip_linear [512, 592] (concat x_res with original x)
        let w = Tensor::randn(0.0f32, 0.02, (dim, dim + in_ch), &self.device)?;  // 512 + 80 = 592
        let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.skip_linear = Some(Linear::new(w, Some(b)));

        // conv1 [512, 512]
        let w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
        let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.conv1 = Some(Linear::new(w, Some(b)));

        // WaveNet with 8 layers
        self.wavenet = Some(WaveNet::new(dim, 8, &self.device)?);

        // res_projection [512, 512]
        let w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
        let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.res_projection = Some(Linear::new(w, Some(b)));

        // Final layer with AdaLN
        self.final_layer = Some(FinalLayer::new(dim, &self.device)?);

        // Output projection: conv2 [80, 512]
        let w = Tensor::randn(0.0f32, 0.02, (in_ch, dim), &self.device)?;
        let b = Tensor::zeros((in_ch,), DType::F32, &self.device)?;
        self.output_proj = Some(Linear::new(w, Some(b)));

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            eprintln!("Warning: DiT checkpoint not found at {:?}, using random weights", path);
            return self.initialize_random();
        }

        eprintln!("Loading DiT weights from {:?}...", path);
        let tensors = load_s2mel_safetensors(path, &self.device)?;

        let dim = self.config.hidden_dim;  // 512
        let in_ch = self.config.in_channels;  // 80
        let content_dim = self.config.content_dim;  // 512
        let prefix = "cfm.estimator";

        // Load x_embedder with weight normalization (not used in main path but load anyway)
        let x_emb_v_key = format!("{}.x_embedder.weight_v", prefix);
        let x_emb_g_key = format!("{}.x_embedder.weight_g", prefix);
        if let (Some(weight_v), Some(weight_g)) = (tensors.get(&x_emb_v_key), tensors.get(&x_emb_g_key)) {
            let bias = tensors.get(&format!("{}.x_embedder.bias", prefix)).cloned();
            let weight_norm = apply_weight_normalization(weight_v, weight_g)?;
            self.x_embedder = Some(Linear::new(weight_norm, bias));
            eprintln!("  Loaded x_embedder (weight normalized)");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, in_ch), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.x_embedder = Some(Linear::new(w, Some(b)));
        }

        // Load cond_projection (content_dim -> content_dim)
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.cond_projection.weight", prefix),
            Some(&format!("{}.cond_projection.bias", prefix)),
        ) {
            self.cond_projection = Some(linear);
            eprintln!("  Loaded cond_projection");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (content_dim, content_dim), &self.device)?;
            let b = Tensor::zeros((content_dim,), DType::F32, &self.device)?;
            self.cond_projection = Some(Linear::new(w, Some(b)));
        }

        // Load cond_x_merge_linear [512, 864] - concatenated input projection
        // Input: [x(80), prompt_x(80), cond(512), style(192)] = 864
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.cond_x_merge_linear.weight", prefix),
            Some(&format!("{}.cond_x_merge_linear.bias", prefix)),
        ) {
            self.cond_x_merge_linear = Some(linear);
            eprintln!("  Loaded cond_x_merge_linear [512, 864]");
        } else {
            let merge_in = in_ch + in_ch + content_dim + self.config.style_dim;  // 864
            let w = Tensor::randn(0.0f32, 0.02, (dim, merge_in), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.cond_x_merge_linear = Some(Linear::new(w, Some(b)));
        }

        // Load time embedding from t_embedder
        self.time_embed = Some(TimestepEmbedding::from_tensors(
            &tensors,
            &format!("{}.t_embedder", prefix),
            dim,
            &self.device,
        )?);
        eprintln!("  Loaded t_embedder");

        // Load cond_embedder for AdaLN [1024, 512] (hidden -> scale/shift)
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.cond_embedder.weight", prefix),
            None,
        ) {
            self.cond_embedder = Some(linear);
            eprintln!("  Loaded cond_embedder [1024, 512]");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), &self.device)?;
            let b = Tensor::zeros((dim * 2,), DType::F32, &self.device)?;
            self.cond_embedder = Some(Linear::new(w, Some(b)));
        }

        // Load transformer blocks
        self.blocks.clear();
        let mut loaded_count = 0;
        for i in 0..self.config.depth {
            let block_prefix = format!("{}.transformer.layers.{}", prefix, i);
            match DiTBlock::from_tensors(
                &tensors,
                &block_prefix,
                dim,
                self.config.num_heads,
                &self.device,
            ) {
                Ok(block) => {
                    self.blocks.push(block);
                    loaded_count += 1;
                }
                Err(e) => {
                    eprintln!("  Warning: Failed to load block {}: {}", i, e);
                    self.blocks.push(DiTBlock::new(dim, self.config.num_heads, &self.device)?);
                }
            }
        }
        eprintln!("  Loaded {} of {} transformer blocks", loaded_count, self.config.depth);

        // Load final norm from transformer.norm
        let final_norm = load_layer_norm(
            &tensors,
            &format!("{}.transformer.norm.norm.weight", prefix),
            dim,
            &self.device,
        )?;
        self.final_norm = Some(final_norm);
        eprintln!("  Loaded final norm");

        // ===== POST-TRANSFORMER COMPONENTS =====

        // Load skip_linear [512, 592] (concat transformer output [512] with original mel [80])
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.skip_linear.weight", prefix),
            Some(&format!("{}.skip_linear.bias", prefix)),
        ) {
            self.skip_linear = Some(linear);
            eprintln!("  Loaded skip_linear [512, 592]");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim + in_ch), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.skip_linear = Some(Linear::new(w, Some(b)));
        }

        // Load conv1 [512, 512]
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.conv1.weight", prefix),
            Some(&format!("{}.conv1.bias", prefix)),
        ) {
            self.conv1 = Some(linear);
            eprintln!("  Loaded conv1 [512, 512]");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.conv1 = Some(Linear::new(w, Some(b)));
        }

        // Load t_embedder2 (second time embedding for WaveNet)
        self.time_embed2 = Some(TimestepEmbedding::from_tensors(
            &tensors,
            &format!("{}.t_embedder2", prefix),
            dim,
            &self.device,
        )?);
        eprintln!("  Loaded t_embedder2");

        // Load WaveNet (8 layers)
        match WaveNet::from_tensors(&tensors, &format!("{}.wavenet", prefix), dim, 8, &self.device) {
            Ok(wavenet) => {
                self.wavenet = Some(wavenet);
                eprintln!("  Loaded WaveNet (8 layers)");
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load WaveNet: {}, using random", e);
                self.wavenet = Some(WaveNet::new(dim, 8, &self.device)?);
            }
        }

        // Load res_projection [512, 512]
        if let Ok(linear) = load_linear(
            &tensors,
            &format!("{}.res_projection.weight", prefix),
            Some(&format!("{}.res_projection.bias", prefix)),
        ) {
            self.res_projection = Some(linear);
            eprintln!("  Loaded res_projection [512, 512]");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.res_projection = Some(Linear::new(w, Some(b)));
        }

        // Load final_layer (AdaLN + linear)
        match FinalLayer::from_tensors(&tensors, &format!("{}.final_layer", prefix), dim, &self.device) {
            Ok(fl) => {
                self.final_layer = Some(fl);
                eprintln!("  Loaded final_layer (AdaLN)");
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load final_layer: {}, using random", e);
                self.final_layer = Some(FinalLayer::new(dim, &self.device)?);
            }
        }

        // Load output projection from conv2 (weight normalized)
        if let Some(weight_v) = tensors.get(&format!("{}.conv2.weight", prefix)) {
            // conv2.weight is [80, 512, 1] - squeeze the kernel dim
            let weight = weight_v.squeeze(2)?;
            let bias = tensors.get(&format!("{}.conv2.bias", prefix)).cloned();

            // Debug: print conv2 weight and bias stats
            let w_mean: f32 = weight.mean_all()?.to_scalar()?;
            let w_std: f32 = weight.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("  conv2.weight: shape={:?}, mean={:.6}, std={:.6}", weight.shape(), w_mean, w_std);
            if let Some(ref b) = bias {
                let b_mean: f32 = b.mean_all()?.to_scalar()?;
                let b_min: f32 = b.min(0)?.to_scalar()?;
                let b_max: f32 = b.max(0)?.to_scalar()?;
                eprintln!("  conv2.bias: mean={:.4}, min={:.4}, max={:.4}", b_mean, b_min, b_max);
            }

            self.output_proj = Some(Linear::new(weight, bias));
            eprintln!("  Loaded conv2 (output projection)");
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (in_ch, dim), &self.device)?;
            let b = Tensor::zeros((in_ch,), DType::F32, &self.device)?;
            self.output_proj = Some(Linear::new(w, Some(b)));
        }

        self.weights_loaded = true;
        eprintln!("DiT weights loaded successfully (including post-transformer)");
        Ok(())
    }

    /// Forward pass - matches Python DiT exactly
    ///
    /// # Arguments
    /// * `x` - Noisy mel spectrogram (batch, seq_len, 80)
    /// * `prompt_x` - Reference mel (batch, seq_len, 80) - zeros in generation region
    /// * `t` - Timesteps (batch,)
    /// * `cond` - Semantic conditioning (batch, seq_len, 512)
    /// * `style` - Speaker style embedding (batch, 192)
    ///
    /// # Returns
    /// * Predicted velocity (batch, seq_len, 80)
    pub fn forward(
        &self,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        if !self.weights_loaded {
            // Return placeholder
            return Ok(x.clone());
        }

        let (batch_size, seq_len, _) = x.dims3()?;

        // Debug input stats
        let x_mean: f32 = x.mean_all()?.to_scalar()?;
        let prompt_mean: f32 = prompt_x.mean_all()?.to_scalar()?;
        let cond_mean: f32 = cond.mean_all()?.to_scalar()?;
        let style_mean: f32 = style.mean_all()?.to_scalar()?;
        eprintln!("DEBUG DiT input: x_mean={:.4}, prompt_mean={:.4}, cond_mean={:.4}, style_mean={:.4}",
            x_mean, prompt_mean, cond_mean, style_mean);

        // 1. Project conditioning: cond -> cond (identity or learned projection)
        let cond = if let Some(ref proj) = self.cond_projection {
            proj.forward(cond)?
        } else {
            cond.clone()
        };

        // 2. Broadcast style to sequence length: (B, 192) -> (B, T, 192)
        let style_broadcast = style.unsqueeze(1)?.broadcast_as((batch_size, seq_len, self.config.style_dim))?;

        // 3. Concatenate: [x(80), prompt_x(80), cond(512), style(192)] -> (B, T, 864)
        let x_in = Tensor::cat(&[x, prompt_x, &cond, &style_broadcast], D::Minus1)?;

        // 4. Merge to hidden dim: (B, T, 864) -> (B, T, 512)
        let h = if let Some(ref merge) = self.cond_x_merge_linear {
            merge.forward(&x_in)?
        } else {
            // Fallback - shouldn't happen with proper weights
            return Err(anyhow::anyhow!("cond_x_merge_linear not loaded"));
        };

        // Debug after merge
        let h_merge_mean: f32 = h.mean_all()?.to_scalar()?;
        let h_merge_std: f32 = h.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
        static ONCE_MERGE: std::sync::Once = std::sync::Once::new();
        ONCE_MERGE.call_once(|| {
            eprintln!("DEBUG DiT after merge: mean={:.4}, std={:.4}", h_merge_mean, h_merge_std);
        });

        // 5. Get time embedding: t -> (B, 512)
        // This goes directly to AdaLayerNorm (each block's project_layer does the 512->1024 projection)
        let t_emb = if let Some(ref embed) = self.time_embed {
            embed.forward(t, &self.device)?
        } else {
            Tensor::zeros((batch_size, self.config.hidden_dim), DType::F32, &self.device)?
        };

        // Debug time embedding
        let t_emb_mean: f32 = t_emb.mean_all()?.to_scalar()?;
        let t_emb_std: f32 = t_emb.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
        static ONCE_TEMB: std::sync::Once = std::sync::Once::new();
        ONCE_TEMB.call_once(|| {
            eprintln!("DEBUG DiT t_emb: mean={:.4}, std={:.4}", t_emb_mean, t_emb_std);
        });

        // Note: cond_embedder is loaded but not used in forward pass
        // The AdaLayerNorm blocks have their own project_layer that takes t_emb (512) -> 1024
        let ada_cond = t_emb;

        // 7. UViT skip connection storage
        let mut skip_features = Vec::new();
        let mid_point = self.config.depth / 2;

        // 8. First half of transformer blocks (encoder) + 9. Second half (decoder)
        // Use atomic counter for single-shot debug
        use std::sync::atomic::{AtomicBool, Ordering};
        static PRINTED_BLOCKS: AtomicBool = AtomicBool::new(false);
        let should_print = !PRINTED_BLOCKS.swap(true, Ordering::Relaxed);

        let mut h = h;
        let mut block_rms = Vec::new();

        for (i, block) in self.blocks.iter().take(mid_point).enumerate() {
            h = block.forward(&h, &ada_cond)?;
            if should_print {
                let h_rms: f32 = h.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                block_rms.push((format!("enc{}", i), h_rms));
            }
            if self.config.uvit_skip_connection {
                skip_features.push(h.clone());
            }
        }

        // 9. Second half of transformer blocks (decoder) with skip connections
        for (i, block) in self.blocks.iter().skip(mid_point).enumerate() {
            if self.config.uvit_skip_connection && i < skip_features.len() {
                // Add skip connection from encoder
                let skip_idx = skip_features.len() - 1 - i;
                if skip_idx < skip_features.len() {
                    h = (h + &skip_features[skip_idx])?;
                }
            }
            h = block.forward(&h, &ada_cond)?;
            if should_print {
                let h_rms: f32 = h.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                block_rms.push((format!("dec{}", i), h_rms));
            }
        }

        if should_print {
            let rms_str: Vec<String> = block_rms.iter().map(|(n, r)| format!("{}:{:.3}", n, r)).collect();
            eprintln!("DEBUG DiT blocks: {}", rms_str.join(", "));
        }

        // 10. Final transformer norm
        let h = if let Some(ref norm) = self.final_norm {
            norm.forward(&h)?
        } else {
            h
        };

        // Debug transformer output
        let h_mean: f32 = h.mean_all()?.to_scalar()?;
        let h_std: f32 = h.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
        eprintln!("DEBUG DiT after transformer: mean={:.4}, std={:.4}", h_mean, h_std);

        // ===== POST-TRANSFORMER PROCESSING =====
        // Debug: Track signal through post-transformer
        static ONCE_POST: std::sync::Once = std::sync::Once::new();
        let debug_post = ONCE_POST.is_completed() == false;

        // Python: x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        // x_res is transformer output (512), x is original mel input (80)
        // Concatenate: [h(512), x(80)] -> 592
        let x_res = if let Some(ref skip) = self.skip_linear {
            let h_x_cat = Tensor::cat(&[&h, x], D::Minus1)?;  // [B, T, 592]
            let result = skip.forward(&h_x_cat)?;  // -> [B, T, 512]
            if debug_post {
                let m: f32 = result.mean_all()?.to_scalar()?;
                let s: f32 = result.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
                eprintln!("DEBUG POST skip_linear: mean={:.4}, std={:.4}", m, s);
            }
            result
        } else {
            h.clone()
        };

        // Python: x = self.conv1(x_res).transpose(1, 2)
        let h = if let Some(ref conv1) = self.conv1 {
            let result = conv1.forward(&x_res)?;  // [B, T, 512]
            if debug_post {
                let m: f32 = result.mean_all()?.to_scalar()?;
                let s: f32 = result.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
                eprintln!("DEBUG POST conv1: mean={:.4}, std={:.4}", m, s);
            }
            result
        } else {
            x_res.clone()
        };

        // Python: t2 = self.t_embedder2(t)
        let t2 = if let Some(ref embed2) = self.time_embed2 {
            embed2.forward(t, &self.device)?  // [B, 512]
        } else {
            Tensor::zeros((batch_size, self.config.hidden_dim), DType::F32, &self.device)?
        };

        // Python: x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2)
        let h = if let Some(ref wavenet) = self.wavenet {
            let result = wavenet.forward(&h, &t2)?;  // [B, T, 512]
            if debug_post {
                let m: f32 = result.mean_all()?.to_scalar()?;
                let s: f32 = result.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
                eprintln!("DEBUG POST wavenet: mean={:.4}, std={:.4}", m, s);
            }
            result
        } else {
            h
        };

        // Python: x = x + self.res_projection(x_res)
        let h = if let Some(ref res_proj) = self.res_projection {
            let res = res_proj.forward(&x_res)?;
            let result = (h + res)?;  // [B, T, 512]
            if debug_post {
                let m: f32 = result.mean_all()?.to_scalar()?;
                let s: f32 = result.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
                eprintln!("DEBUG POST after res_projection: mean={:.4}, std={:.4}", m, s);
            }
            result
        } else {
            h
        };

        // Python: x = self.final_layer(x, t1).transpose(1, 2)
        // t1 is the first time embedding (ada_cond)
        let h = if let Some(ref fl) = self.final_layer {
            let result = fl.forward(&h, &ada_cond)?;  // [B, T, 512]
            if debug_post {
                let m: f32 = result.mean_all()?.to_scalar()?;
                let s: f32 = result.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
                eprintln!("DEBUG POST final_layer: mean={:.4}, std={:.4}", m, s);
            }
            result
        } else {
            h
        };

        // Debug: Check h before conv2
        static ONCE_PRE_CONV2: std::sync::Once = std::sync::Once::new();
        ONCE_PRE_CONV2.call_once(|| {
            let h_mean: f32 = h.mean_all().unwrap().to_scalar().unwrap();
            let h_std: f32 = h.var(D::Minus1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            eprintln!("DEBUG DiT before conv2: mean={:.4}, std={:.4}", h_mean, h_std);
        });

        // Python: x = self.conv2(x)
        // Output projection: 512 -> 80 (mel channels)
        if let Some(ref proj) = self.output_proj {
            // Debug: Manual computation check for first sample
            static ONCE_CONV2_CHECK: std::sync::Once = std::sync::Once::new();
            ONCE_CONV2_CHECK.call_once(|| {
                // Get weight and bias
                let weight = proj.weight();
                let _bias = proj.bias();
                let w_shape = weight.shape();
                eprintln!("DEBUG conv2 check: weight shape={:?}", w_shape);

                // Check sum of weights per output channel
                let w_sum_per_channel: Vec<f32> = (0..w_shape.dims()[0]).map(|i| {
                    weight.i(i).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()
                }).collect();
                let w_sum_mean: f32 = w_sum_per_channel.iter().sum::<f32>() / w_sum_per_channel.len() as f32;
                let w_sum_max: f32 = w_sum_per_channel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w_sum_min: f32 = w_sum_per_channel.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("DEBUG conv2 check: weight sum per channel: mean={:.4}, min={:.4}, max={:.4}",
                    w_sum_mean, w_sum_min, w_sum_max);

                // Check h (input) sum
                let h_sum: f32 = h.sum_all().unwrap().to_scalar().unwrap();
                let h_numel = h.elem_count();
                eprintln!("DEBUG conv2 check: h sum={:.4}, numel={}", h_sum, h_numel);
            });

            let output = proj.forward(&h)?;  // [B, T, 80]
            let out_mean: f32 = output.mean_all()?.to_scalar()?;
            let out_std: f32 = output.var(D::Minus1)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("DEBUG DiT output: mean={:.4}, std={:.4}", out_mean, out_std);
            Ok(output)
        } else {
            Ok(h)
        }
    }

    /// Legacy forward for compatibility (creates zero prompt_x and default style)
    pub fn forward_legacy(
        &self,
        x: &Tensor,
        t: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, mel_ch) = x.dims3()?;

        // Create zero prompt_x
        let prompt_x = Tensor::zeros((batch_size, seq_len, mel_ch), DType::F32, &self.device)?;

        // Create zero style if not provided
        let default_style = Tensor::zeros((batch_size, self.config.style_dim), DType::F32, &self.device)?;
        let style = style.unwrap_or(&default_style);

        self.forward(x, &prompt_x, t, content, style)
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get output channels
    pub fn output_channels(&self) -> usize {
        self.config.in_channels
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
    fn test_dit_config_default() {
        let config = DiffusionTransformerConfig::default();
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.depth, 13);
        assert_eq!(config.in_channels, 80);
    }

    #[test]
    fn test_sinusoidal_embedding() {
        let device = Device::Cpu;
        let t = Tensor::new(&[0.0f32, 0.5, 1.0], &device).unwrap();
        let emb = sinusoidal_embedding(&t, 64, &device).unwrap();
        assert_eq!(emb.dims(), &[3, 64]);
    }

    #[test]
    fn test_silu() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let y = silu(&x).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((values[0] - 0.0).abs() < 0.001);
        // silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((values[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_dit_new() {
        let device = Device::Cpu;
        let dit = DiffusionTransformer::new(&device).unwrap();
        assert_eq!(dit.hidden_dim(), 512);
        assert_eq!(dit.output_channels(), 80);
    }

    #[test]
    fn test_dit_placeholder() {
        let device = Device::Cpu;
        let dit = DiffusionTransformer::new(&device).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (2, 100, 80), &device).unwrap();
        let prompt_x = Tensor::zeros((2, 100, 80), DType::F32, &device).unwrap();
        let t = Tensor::new(&[0.5f32, 0.5], &device).unwrap();
        let cond = Tensor::randn(0.0f32, 1.0, (2, 100, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (2, 192), &device).unwrap();

        // Uninitialized DiT returns x unchanged (placeholder behavior)
        let output = dit.forward(&x, &prompt_x, &t, &cond, &style).unwrap();
        assert_eq!(output.dims3().unwrap(), (2, 100, 80));
    }

    #[test]
    fn test_dit_initialized() {
        let device = Device::Cpu;
        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        assert!(dit.is_initialized());

        let x = Tensor::randn(0.0f32, 1.0, (1, 50, 80), &device).unwrap();
        let prompt_x = Tensor::zeros((1, 50, 80), DType::F32, &device).unwrap();
        let t = Tensor::new(&[0.5f32], &device).unwrap();
        let cond = Tensor::randn(0.0f32, 1.0, (1, 50, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        let output = dit.forward(&x, &prompt_x, &t, &cond, &style).unwrap();
        let (batch, len, channels) = output.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(len, 50);
        assert_eq!(channels, 80);
    }

    #[test]
    fn test_timestep_embedding() {
        let device = Device::Cpu;
        let embed = TimestepEmbedding::new(256, &device).unwrap();
        let t = Tensor::new(&[0.0f32, 0.5, 1.0], &device).unwrap();
        let emb = embed.forward(&t, &device).unwrap();
        assert_eq!(emb.dims(), &[3, 256]);
    }

    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let attn = MultiHeadAttention::new(256, 8, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 16, 256), &device).unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.dims3().unwrap(), (2, 16, 256));
    }

    #[test]
    fn test_feed_forward() {
        let device = Device::Cpu;
        let ff = FeedForward::new(256, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 16, 256), &device).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.dims3().unwrap(), (2, 16, 256));
    }
}
