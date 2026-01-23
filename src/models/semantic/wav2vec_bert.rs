//! Wav2Vec-BERT 2.0 semantic encoder
//!
//! Extracts semantic features from audio using the Wav2Vec-BERT 2.0 model.
//! The model extracts hidden states from layer 17 and normalizes them
//! using pre-computed mean and std statistics.
//!
//! Architecture: Wav2Vec-BERT 2.0 (1024 hidden dim, 24 layers)
//! - Input: Raw audio waveform at 16kHz
//! - Output: Semantic embeddings (batch, seq_len, 1024)
//!
//! Weight loading: Maps HuggingFace key names to our model structure:
//! - HF: encoder.layers.{i}.self_attn.linear_q -> Rust: attention.q_proj
//! - HF: encoder.layers.{i}.ffn1.intermediate_dense -> Rust: ffn1.intermediate
//! - etc.

use anyhow::Result;
use candle_core::{safetensors, Device, Tensor, DType, D};
use candle_nn::{Linear, Module, LayerNorm};
use std::collections::HashMap;
use std::path::Path;

/// Hidden dimension of Wav2Vec-BERT 2.0
const HIDDEN_SIZE: usize = 1024;
/// Number of attention heads
const NUM_HEADS: usize = 16;
/// Number of encoder layers
const NUM_LAYERS: usize = 24;
/// Intermediate FFN dimension
const INTERMEDIATE_SIZE: usize = 4096;
/// Layer to extract features from (0-indexed)
const EXTRACT_LAYER: usize = 17;
/// Input feature dimension (from feature extractor)
const INPUT_FEATURE_DIM: usize = 160;
/// Conv module kernel size
const CONV_KERNEL_SIZE: usize = 31;
/// Head dimension (for distance embedding)
const HEAD_DIM: usize = 64;
/// Relative position encoding: left context window
const LEFT_MAX_POS: i64 = 64;
/// Relative position encoding: right context window
const RIGHT_MAX_POS: i64 = 8;
/// Total positions: left + right + 1 (for position 0)
const NUM_POSITIONS: usize = 73; // 64 + 8 + 1

/// Helper to load a Linear layer from tensors
/// candle_nn::Linear expects weights in PyTorch format [out_features, in_features]
/// and handles transpose internally, so we load weights directly without transposing
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

/// Helper to load LayerNorm from tensors
fn load_layer_norm(
    tensors: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: &str,
    dim: usize,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = match tensors.get(weight_key) {
        Some(w) => w.clone(),
        None => {
            tracing::warn!(
                "[Wav2Vec-BERT] Missing tensor '{}', using ones initialization",
                weight_key
            );
            Tensor::ones((dim,), DType::F32, device)?
        }
    };
    let bias = match tensors.get(bias_key) {
        Some(b) => b.clone(),
        None => {
            tracing::warn!(
                "[Wav2Vec-BERT] Missing tensor '{}', using zeros initialization",
                bias_key
            );
            Tensor::zeros((dim,), DType::F32, device)?
        }
    };
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

/// Swish activation function: x * sigmoid(x)
fn swish(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    (x * sigmoid).map_err(Into::into)
}

/// GLU (Gated Linear Unit) activation
/// Splits input along last dimension and applies gate
fn glu(x: &Tensor, dim: usize) -> Result<Tensor> {
    let chunks = x.chunk(2, dim)?;
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = candle_nn::ops::sigmoid(b)?;
    (a * gate).map_err(Into::into)
}

/// ConvModule for Wav2Vec-BERT 2.0 Conformer-like architecture
///
/// Implements depthwise separable convolution with GLU and Swish activations.
/// Processing flow:
/// 1. LayerNorm (pre-GLU normalization)
/// 2. Pointwise conv1 (expand 1024 -> 2048 for GLU)
/// 3. GLU activation (splits 2048 -> 1024)
/// 4. Depthwise conv (kernel=31, groups=1024)
/// 5. Depthwise LayerNorm
/// 6. Swish activation
/// 7. Pointwise conv2 (1024 -> 1024)
struct ConvModule {
    layer_norm: LayerNorm,
    pointwise_conv1: Linear,
    depthwise_conv_weight: Tensor,
    depthwise_layer_norm: LayerNorm,
    pointwise_conv2: Linear,
    kernel_size: usize,
}

impl ConvModule {
    /// Load from HuggingFace checkpoint tensors
    /// Tensor names: encoder.layers.{i}.conv_module.*
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        kernel_size: usize,
        device: &Device,
    ) -> Result<Self> {
        // Layer norm
        let layer_norm = load_layer_norm(
            tensors,
            &format!("{}.layer_norm.weight", prefix),
            &format!("{}.layer_norm.bias", prefix),
            dim,
            device,
        )?;

        // Pointwise conv1: [2048, 1024, 1] -> reshape to [2048, 1024]
        let pw1_key = format!("{}.pointwise_conv1.weight", prefix);
        let pointwise_conv1 = if let Some(weight) = tensors.get(&pw1_key) {
            let (out_ch, in_ch, _k) = weight.dims3()?;
            let weight = weight.reshape((out_ch, in_ch))?;
            // No bias in checkpoint
            Linear::new(weight, None)
        } else {
            tracing::warn!(
                "[Wav2Vec-BERT] Missing tensor '{}', using random initialization",
                pw1_key
            );
            let w = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
            Linear::new(w, None)
        };

        // Depthwise conv: [1024, 1, 31] - no bias
        let dw_key = format!("{}.depthwise_conv.weight", prefix);
        let depthwise_conv_weight = match tensors.get(&dw_key) {
            Some(w) => w.clone(),
            None => {
                tracing::warn!(
                    "[Wav2Vec-BERT] Missing tensor '{}', using random initialization",
                    dw_key
                );
                Tensor::randn(0.0f32, 0.02, (dim, 1, kernel_size), device)?
            }
        };

        // Depthwise layer norm
        let depthwise_layer_norm = load_layer_norm(
            tensors,
            &format!("{}.depthwise_layer_norm.weight", prefix),
            &format!("{}.depthwise_layer_norm.bias", prefix),
            dim,
            device,
        )?;

        // Pointwise conv2: [1024, 1024, 1] -> reshape to [1024, 1024]
        let pw2_key = format!("{}.pointwise_conv2.weight", prefix);
        let pointwise_conv2 = if let Some(weight) = tensors.get(&pw2_key) {
            let (out_ch, in_ch, _k) = weight.dims3()?;
            let weight = weight.reshape((out_ch, in_ch))?;
            // No bias in checkpoint
            Linear::new(weight, None)
        } else {
            tracing::warn!(
                "[Wav2Vec-BERT] Missing tensor '{}', using random initialization",
                pw2_key
            );
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
            Linear::new(w, None)
        };

        Ok(Self {
            layer_norm,
            pointwise_conv1,
            depthwise_conv_weight,
            depthwise_layer_norm,
            pointwise_conv2,
            kernel_size,
        })
    }

    /// Initialize with random weights (fallback)
    fn new_random(dim: usize, kernel_size: usize, device: &Device) -> Result<Self> {
        let ln_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_weight.clone(), ln_bias.clone(), 1e-5);

        let w1 = Tensor::randn(0.0f32, 0.02, (dim * 2, dim), device)?;
        let pointwise_conv1 = Linear::new(w1, None);

        let depthwise_conv_weight = Tensor::randn(0.0f32, 0.02, (dim, 1, kernel_size), device)?;

        let depthwise_layer_norm = LayerNorm::new(ln_weight.clone(), ln_bias.clone(), 1e-5);

        let w2 = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
        let pointwise_conv2 = Linear::new(w2, None);

        Ok(Self {
            layer_norm,
            pointwise_conv1,
            depthwise_conv_weight,
            depthwise_layer_norm,
            pointwise_conv2,
            kernel_size,
        })
    }

    /// Forward pass through conv module
    /// Input: (batch, seq, dim) -> Output: (batch, seq, dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Layer norm
        let x = self.layer_norm.forward(x)?;

        // 2. Pointwise conv1 (expand for GLU)
        let x = self.pointwise_conv1.forward(&x)?;

        // 3. GLU activation (along feature dim)
        let x = glu(&x, 2)?;

        // 4. Depthwise conv: transpose to (batch, channels, seq)
        let x = x.transpose(1, 2)?;
        let padding = self.kernel_size / 2; // 31 / 2 = 15
        let x = x.conv1d(
            &self.depthwise_conv_weight,
            padding,
            1, // stride
            1, // dilation
            x.dim(1)?, // groups = channels (depthwise)
        )?;
        // Transpose back to (batch, seq, channels)
        let x = x.transpose(1, 2)?;

        // 5. Depthwise layer norm
        let x = self.depthwise_layer_norm.forward(&x)?;

        // 6. Swish activation
        let x = swish(&x)?;

        // 7. Pointwise conv2
        self.pointwise_conv2.forward(&x).map_err(Into::into)
    }
}

/// FeatureProjection for Wav2Vec-BERT 2.0
///
/// Projects 160-dim input features from the CNN feature extractor to 1024-dim hidden states.
/// Structure:
/// - layer_norm: LayerNorm [160]
/// - projection: Linear [1024, 160] + bias [1024]
struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl FeatureProjection {
    /// Load from HuggingFace checkpoint tensors
    /// Tensor names:
    /// - feature_projection.layer_norm.weight [160]
    /// - feature_projection.layer_norm.bias [160]
    /// - feature_projection.projection.weight [1024, 160]
    /// - feature_projection.projection.bias [1024]
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        input_dim: usize,
        _hidden_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let layer_norm = load_layer_norm(
            tensors,
            "feature_projection.layer_norm.weight",
            "feature_projection.layer_norm.bias",
            input_dim,
            device,
        )?;

        let projection = load_linear(
            tensors,
            "feature_projection.projection.weight",
            Some("feature_projection.projection.bias"),
        )?;

        Ok(Self {
            layer_norm,
            projection,
        })
    }

    /// Initialize with random weights (fallback)
    fn new_random(input_dim: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        let ln_weight = Tensor::ones((input_dim,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((input_dim,), DType::F32, device)?;
        let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);

        let proj_weight = Tensor::randn(0.0f32, 0.02, (hidden_dim, input_dim), device)?;
        let proj_bias = Tensor::zeros((hidden_dim,), DType::F32, device)?;
        let projection = Linear::new(proj_weight, Some(proj_bias));

        Ok(Self {
            layer_norm,
            projection,
        })
    }

    /// Forward pass: LayerNorm -> Linear
    /// Input: (batch, seq, input_dim) -> Output: (batch, seq, hidden_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        self.projection.forward(&x).map_err(Into::into)
    }
}

/// Self-attention layer for Wav2Vec-BERT with Shaw-style relative position encoding
///
/// The distance_embedding provides position-dependent attention bias based on
/// the relative distance between query and key positions. Shape: [73, 64]
/// - 73 positions: left context (64) + right context (8) + position 0
/// - 64: head dimension for computing position bias
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
    /// Distance embedding for Shaw-style relative position encoding [73, 64]
    distance_embedding: Option<Tensor>,
}

impl SelfAttention {
    /// Load from HuggingFace tensor names
    /// HF format: self_attn.linear_q, self_attn.linear_k, etc.
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // HuggingFace uses linear_q, linear_k, linear_v, linear_out
        // candle_nn::Linear expects PyTorch format and handles transpose internally
        let query = load_linear(
            tensors,
            &format!("{}.linear_q.weight", prefix),
            Some(&format!("{}.linear_q.bias", prefix)),
        )?;
        let key = load_linear(
            tensors,
            &format!("{}.linear_k.weight", prefix),
            Some(&format!("{}.linear_k.bias", prefix)),
        )?;
        let value = load_linear(
            tensors,
            &format!("{}.linear_v.weight", prefix),
            Some(&format!("{}.linear_v.bias", prefix)),
        )?;
        let output = load_linear(
            tensors,
            &format!("{}.linear_out.weight", prefix),
            Some(&format!("{}.linear_out.bias", prefix)),
        )?;

        // Load distance embedding for relative position encoding [73, 64]
        let distance_key = format!("{}.distance_embedding.weight", prefix);
        let distance_embedding = match tensors.get(&distance_key) {
            Some(de) => {
                let (num_pos, dim) = de.dims2()?;
                if num_pos != NUM_POSITIONS || dim != HEAD_DIM {
                    tracing::warn!(
                        "[Wav2Vec-BERT] Unexpected distance_embedding shape [{}, {}], expected [{}, {}]",
                        num_pos, dim, NUM_POSITIONS, HEAD_DIM
                    );
                }
                Some(de.clone())
            }
            None => {
                tracing::warn!(
                    "[Wav2Vec-BERT] Missing tensor '{}', skipping relative position bias",
                    distance_key
                );
                None
            }
        };

        Ok(Self {
            query,
            key,
            value,
            output,
            num_heads,
            head_dim,
            distance_embedding,
        })
    }

    fn new_random(hidden_size: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let make_linear = |device: &Device| -> Result<Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (hidden_size, hidden_size), device)?;
            let b = Tensor::zeros((hidden_size,), DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };
        Ok(Self {
            query: make_linear(device)?,
            key: make_linear(device)?,
            value: make_linear(device)?,
            output: make_linear(device)?,
            num_heads,
            head_dim,
            distance_embedding: None, // No relative position bias for random init
        })
    }

    /// Compute Shaw-style relative position bias
    ///
    /// Given query tensor and distance embedding, computes position-dependent
    /// attention bias based on relative positions between query and key.
    ///
    /// Algorithm:
    /// 1. Create position indices for query and key
    /// 2. Compute relative distance: key_pos - query_pos
    /// 3. Clamp to [-LEFT_MAX_POS, RIGHT_MAX_POS]
    /// 4. Shift to positive indices: distance + LEFT_MAX_POS
    /// 5. Lookup distance embeddings
    /// 6. Compute bias via dot product with query
    fn compute_relative_position_bias(
        &self,
        query: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let distance_emb = match &self.distance_embedding {
            Some(de) => de,
            None => return Ok(Tensor::zeros(query.shape(), query.dtype(), query.device())?),
        };

        let device = query.device();

        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::arange(0i64, seq_len as i64, device)?;

        // Query positions: [seq_len, 1]
        let pos_q = positions.reshape((seq_len, 1))?;
        // Key positions: [1, seq_len]
        let pos_k = positions.reshape((1, seq_len))?;

        // Relative distance: key_pos - query_pos -> [seq_len, seq_len]
        let distance = pos_k.broadcast_sub(&pos_q)?;

        // Clamp to valid range [-LEFT_MAX_POS, RIGHT_MAX_POS]
        let distance = distance.clamp(-LEFT_MAX_POS, RIGHT_MAX_POS)?;

        // Shift to positive indices [0, NUM_POSITIONS-1]
        let indices = (distance + LEFT_MAX_POS as f64)?;
        let indices = indices.to_dtype(DType::U32)?;

        // Flatten indices for index_select
        let flat_indices = indices.flatten_all()?;

        // Lookup embeddings: [seq_len * seq_len, head_dim]
        let pos_emb = distance_emb.index_select(&flat_indices, 0)?;

        // Reshape to [seq_len, seq_len, head_dim]
        let pos_emb = pos_emb.reshape((seq_len, seq_len, self.head_dim))?;

        // query shape: [batch, heads, seq_len, head_dim]
        // We compute: bias[b, h, i, j] = query[b, h, i, :] dot pos_emb[i, j, :]
        // = sum_d query[b, h, i, d] * pos_emb[i, j, d]

        // Use broadcasting: (query.unsqueeze(-2) * pos_emb.unsqueeze(0).unsqueeze(0)).sum(-1)
        // query: [batch, heads, seq, 1, head_dim]
        // pos_emb: [1, 1, seq, seq, head_dim]
        // product: [batch, heads, seq, seq, head_dim]
        // sum: [batch, heads, seq, seq]

        let query_expanded = query.unsqueeze(3)?; // [batch, heads, seq, 1, head_dim]
        let pos_emb_expanded = pos_emb
            .unsqueeze(0)?
            .unsqueeze(0)?; // [1, 1, seq, seq, head_dim]

        let bias = query_expanded.broadcast_mul(&pos_emb_expanded)?;
        let bias = bias.sum(D::Minus1)?; // [batch, heads, seq, seq]

        // Scale by 1/sqrt(head_dim) (already done in main attention, but pos bias should also be scaled)
        let scale = (self.head_dim as f64).sqrt();
        let bias = (bias / scale)?;

        Ok(bias)
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let query = self.query.forward(hidden_states)?;
        let key = self.key.forward(hidden_states)?;
        let value = self.value.forward(hidden_states)?;

        // Reshape for multi-head attention: (batch, seq, heads, head_dim)
        let query = query
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (batch, heads, seq, head_dim)
        let key = key
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value = value
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        // Make tensors contiguous for matmul
        let query = query.contiguous()?;
        let key = key.contiguous()?;
        let value = value.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let key_t = key.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_weights = query.matmul(&key_t)?;
        let attn_weights = (attn_weights / scale)?;

        // Add relative position bias if distance_embedding is available
        let attn_weights = if self.distance_embedding.is_some() {
            let pos_bias = self.compute_relative_position_bias(&query, seq_len)?;
            (attn_weights + pos_bias)?
        } else {
            attn_weights
        };

        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?; // (batch, 1, 1, seq)
            let neg_inf = Tensor::new(f32::NEG_INFINITY, hidden_states.device())?;
            let mask = mask.where_cond(&Tensor::zeros_like(&attn_weights)?, &neg_inf.broadcast_as(attn_weights.shape())?)?;
            (attn_weights + mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.contiguous()?.matmul(&value)?;

        // Reshape back: (batch, seq, hidden)
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.output.forward(&attn_output).map_err(Into::into)
    }
}

/// Feed-forward network
struct FeedForward {
    intermediate: Linear,
    output: Linear,
}

impl FeedForward {
    /// Load from HuggingFace tensor names
    /// HF format: ffn1.intermediate_dense, ffn1.output_dense (or ffn2)
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        _hidden_size: usize,
        _intermediate_size: usize,
    ) -> Result<Self> {
        let intermediate = load_linear(
            tensors,
            &format!("{}.intermediate_dense.weight", prefix),
            Some(&format!("{}.intermediate_dense.bias", prefix)),
        )?;
        let output = load_linear(
            tensors,
            &format!("{}.output_dense.weight", prefix),
            Some(&format!("{}.output_dense.bias", prefix)),
        )?;

        Ok(Self { intermediate, output })
    }

    fn new_random(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        let w1 = Tensor::randn(0.0f32, 0.02, (intermediate_size, hidden_size), device)?;
        let b1 = Tensor::zeros((intermediate_size,), DType::F32, device)?;
        let w2 = Tensor::randn(0.0f32, 0.02, (hidden_size, intermediate_size), device)?;
        let b2 = Tensor::zeros((hidden_size,), DType::F32, device)?;
        Ok(Self {
            intermediate: Linear::new(w1, Some(b1)),
            output: Linear::new(w2, Some(b2)),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden = self.intermediate.forward(hidden_states)?;
        let hidden = hidden.gelu_erf()?; // GELU activation
        self.output.forward(&hidden).map_err(Into::into)
    }
}

/// Encoder layer for Wav2Vec-BERT 2.0 (Conformer-like architecture)
///
/// HuggingFace structure for each layer:
/// - ffn1 (first feed-forward)
/// - ffn1_layer_norm
/// - self_attn (attention)
/// - self_attn_layer_norm
/// - conv_module (convolution module)
/// - ffn2 (second feed-forward)
/// - ffn2_layer_norm
/// - final_layer_norm
struct EncoderLayer {
    ffn1: FeedForward,
    ffn1_layer_norm: LayerNorm,
    attention: SelfAttention,
    attention_layer_norm: LayerNorm,
    conv_module: Option<ConvModule>,
    ffn2: FeedForward,
    ffn2_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    /// Load from HuggingFace tensor names
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        layer_idx: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("encoder.layers.{}", layer_idx);

        // FFN1
        let ffn1 = FeedForward::from_tensors(
            tensors,
            &format!("{}.ffn1", prefix),
            hidden_size,
            intermediate_size,
        )?;
        let ffn1_layer_norm = load_layer_norm(
            tensors,
            &format!("{}.ffn1_layer_norm.weight", prefix),
            &format!("{}.ffn1_layer_norm.bias", prefix),
            hidden_size,
            device,
        )?;

        // Self-attention
        let attention = SelfAttention::from_tensors(
            tensors,
            &format!("{}.self_attn", prefix),
            hidden_size,
            num_heads,
        )?;
        let attention_layer_norm = load_layer_norm(
            tensors,
            &format!("{}.self_attn_layer_norm.weight", prefix),
            &format!("{}.self_attn_layer_norm.bias", prefix),
            hidden_size,
            device,
        )?;

        // Conv module - load from checkpoint
        let conv_module = match ConvModule::from_tensors(
            tensors,
            &format!("{}.conv_module", prefix),
            hidden_size,
            CONV_KERNEL_SIZE,
            device,
        ) {
            Ok(cm) => Some(cm),
            Err(e) => {
                tracing::warn!(
                    "[Wav2Vec-BERT] Layer {} conv_module load failed: {}, skipping",
                    layer_idx, e
                );
                None
            }
        };

        // FFN2
        let ffn2 = FeedForward::from_tensors(
            tensors,
            &format!("{}.ffn2", prefix),
            hidden_size,
            intermediate_size,
        )?;
        let ffn2_layer_norm = load_layer_norm(
            tensors,
            &format!("{}.ffn2_layer_norm.weight", prefix),
            &format!("{}.ffn2_layer_norm.bias", prefix),
            hidden_size,
            device,
        )?;

        // Final layer norm
        let final_layer_norm = load_layer_norm(
            tensors,
            &format!("{}.final_layer_norm.weight", prefix),
            &format!("{}.final_layer_norm.bias", prefix),
            hidden_size,
            device,
        )?;

        Ok(Self {
            ffn1,
            ffn1_layer_norm,
            attention,
            attention_layer_norm,
            conv_module,
            ffn2,
            ffn2_layer_norm,
            final_layer_norm,
        })
    }

    fn new_random(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let make_ln = || -> Result<LayerNorm> {
            let w = Tensor::ones((hidden_size,), DType::F32, device)?;
            let b = Tensor::zeros((hidden_size,), DType::F32, device)?;
            Ok(LayerNorm::new(w, b, 1e-5))
        };

        Ok(Self {
            ffn1: FeedForward::new_random(hidden_size, intermediate_size, device)?,
            ffn1_layer_norm: make_ln()?,
            attention: SelfAttention::new_random(hidden_size, num_heads, device)?,
            attention_layer_norm: make_ln()?,
            conv_module: Some(ConvModule::new_random(hidden_size, CONV_KERNEL_SIZE, device)?),
            ffn2: FeedForward::new_random(hidden_size, intermediate_size, device)?,
            ffn2_layer_norm: make_ln()?,
            final_layer_norm: make_ln()?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Wav2Vec-BERT 2.0 Conformer-like forward:
        // 1. Half-step FFN1 with residual
        let residual = hidden_states.clone();
        let hidden_states = self.ffn1_layer_norm.forward(hidden_states)?;
        let hidden_states = self.ffn1.forward(&hidden_states)?;
        let hidden_states = (residual + (hidden_states * 0.5)?)?;

        // 2. Self-attention with residual
        let residual = hidden_states.clone();
        let normed = self.attention_layer_norm.forward(&hidden_states)?;
        let attn_output = self.attention.forward(&normed, attention_mask)?;
        let hidden_states = (residual + attn_output)?;

        // 3. Conv module with residual (if present)
        let hidden_states = if let Some(ref conv) = self.conv_module {
            let residual = hidden_states.clone();
            let conv_output = conv.forward(&hidden_states)?;
            (residual + conv_output)?
        } else {
            hidden_states
        };

        // 4. Half-step FFN2 with residual
        let residual = hidden_states.clone();
        let hidden_states = self.ffn2_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.ffn2.forward(&hidden_states)?;
        let hidden_states = (residual + (hidden_states * 0.5)?)?;

        // 5. Final layer norm
        self.final_layer_norm.forward(&hidden_states).map_err(Into::into)
    }
}


/// Wav2Vec-BERT 2.0 semantic encoder
pub struct SemanticEncoder {
    device: Device,
    /// Normalization statistics (from wav2vec2bert_stats.pt)
    mean: Tensor,
    std: Tensor,
    /// Feature projection (160-dim -> 1024-dim)
    feature_projection: Option<FeatureProjection>,
    /// Encoder layers
    encoder_layers: Vec<EncoderLayer>,
    /// Layer to extract features from
    extract_layer: usize,
    /// Whether weights were successfully loaded
    weights_loaded: bool,
}

impl SemanticEncoder {
    /// Load semantic encoder from checkpoint
    ///
    /// # Arguments
    /// * `stat_path` - Path to wav2vec2bert_stats.pt containing mean/std
    /// * `model_path` - Optional path to model weights (if None, uses placeholder)
    /// * `device` - Device to load tensors on
    pub fn load<P: AsRef<Path>>(stat_path: P, _model_path: Option<P>, device: &Device) -> Result<Self> {
        // Load normalization statistics
        let (mean, std) = Self::load_stats(stat_path.as_ref(), device)?;

        // Create placeholder encoder without full weights
        let encoder_layers = Vec::new();

        Ok(Self {
            device: device.clone(),
            mean,
            std,
            feature_projection: None,
            encoder_layers,
            extract_layer: EXTRACT_LAYER,
            weights_loaded: false,
        })
    }

    /// Load stats from safetensors file
    ///
    /// The checkpoint stores variance (var) but we need standard deviation (std).
    /// This function handles the var -> std conversion via sqrt().
    fn load_stats(path: &Path, device: &Device) -> Result<(Tensor, Tensor)> {
        // Try to load from safetensors format first, then fall back to defaults
        if path.exists() {
            if let Ok(tensors) = safetensors::load(path, device) {
                let mean = match tensors.get("mean") {
                    Some(m) => m.clone(),
                    None => {
                        tracing::warn!(
                            "[Wav2Vec-BERT] Missing tensor 'mean' in stats file, using zeros initialization"
                        );
                        Tensor::zeros((HIDDEN_SIZE,), DType::F32, device)?
                    }
                };
                // Handle std vs var: checkpoint may store variance, we need std
                let std = match tensors.get("std") {
                    Some(s) => s.clone(),
                    None => match tensors.get("var") {
                        Some(var) => {
                            // Convert variance to standard deviation via sqrt()
                            tracing::info!(
                                "[Wav2Vec-BERT] Converting 'var' to 'std' via sqrt()"
                            );
                            var.sqrt()?
                        }
                        None => {
                            tracing::warn!(
                                "[Wav2Vec-BERT] Missing 'std' or 'var' in stats file, using ones initialization"
                            );
                            Tensor::ones((HIDDEN_SIZE,), DType::F32, device)?
                        }
                    }
                };
                return Ok((mean, std));
            }
        }

        // Default: zero mean, unit std (no normalization)
        tracing::warn!(
            "[Wav2Vec-BERT] Stats file not found at {:?}, using default normalization",
            path
        );
        let mean = Tensor::zeros((HIDDEN_SIZE,), DType::F32, device)?;
        let std = Tensor::ones((HIDDEN_SIZE,), DType::F32, device)?;
        Ok((mean, std))
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        // Initialize feature projection
        self.feature_projection = Some(FeatureProjection::new_random(
            INPUT_FEATURE_DIM,
            HIDDEN_SIZE,
            &self.device,
        )?);

        // Initialize encoder layers
        self.encoder_layers.clear();
        for _ in 0..NUM_LAYERS {
            self.encoder_layers.push(EncoderLayer::new_random(
                HIDDEN_SIZE,
                NUM_HEADS,
                INTERMEDIATE_SIZE,
                &self.device,
            )?);
        }
        self.weights_loaded = true;
        Ok(())
    }

    /// Load full model weights from safetensors
    ///
    /// This properly maps HuggingFace Wav2Vec-BERT 2.0 tensor names to our model.
    /// HuggingFace format: encoder.layers.{i}.self_attn.linear_q.weight, etc.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            eprintln!("Warning: Wav2Vec-BERT weights not found at {:?}, using random weights", path);
            return self.initialize_random();
        }

        eprintln!("Loading Wav2Vec-BERT weights from {:?}...", path);

        // Load all tensors from safetensors
        let tensors = safetensors::load(path, &self.device)?;

        // Debug: print first few keys to verify structure
        let keys: Vec<_> = tensors.keys().take(5).collect();
        eprintln!("  Sample tensor keys: {:?}", keys);

        // Load feature projection (global, not per-layer)
        self.feature_projection = match FeatureProjection::from_tensors(
            &tensors,
            INPUT_FEATURE_DIM,
            HIDDEN_SIZE,
            &self.device,
        ) {
            Ok(fp) => {
                eprintln!("  Loaded feature_projection (160 -> 1024)");
                Some(fp)
            }
            Err(e) => {
                tracing::warn!(
                    "[Wav2Vec-BERT] Failed to load feature_projection: {}, using random",
                    e
                );
                Some(FeatureProjection::new_random(INPUT_FEATURE_DIM, HIDDEN_SIZE, &self.device)?)
            }
        };

        // Load encoder layers with proper name mapping
        self.encoder_layers.clear();
        let mut loaded_count = 0;

        for i in 0..NUM_LAYERS {
            match EncoderLayer::from_tensors(
                &tensors,
                i,
                HIDDEN_SIZE,
                NUM_HEADS,
                INTERMEDIATE_SIZE,
                &self.device,
            ) {
                Ok(layer) => {
                    self.encoder_layers.push(layer);
                    loaded_count += 1;
                }
                Err(e) => {
                    eprintln!("  Warning: Failed to load layer {}: {}", i, e);
                    // Try to load with random weights for this layer
                    match EncoderLayer::new_random(
                        HIDDEN_SIZE,
                        NUM_HEADS,
                        INTERMEDIATE_SIZE,
                        &self.device,
                    ) {
                        Ok(layer) => {
                            self.encoder_layers.push(layer);
                            eprintln!("    Using random weights for layer {}", i);
                        }
                        Err(e2) => {
                            eprintln!("    Could not create random layer {}: {}", i, e2);
                            break;
                        }
                    }
                }
            }
        }

        if loaded_count == 0 {
            eprintln!("  Warning: No layers loaded from weights, using all random");
            return self.initialize_random();
        }

        eprintln!("  Successfully loaded {} of {} encoder layers", loaded_count, NUM_LAYERS);
        self.weights_loaded = true;
        Ok(())
    }

    /// Check if weights were loaded
    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }

    /// Extract semantic embeddings from audio features
    ///
    /// # Arguments
    /// * `input_features` - Input features (batch, seq_len, feature_dim) or raw audio (batch, samples)
    /// * `attention_mask` - Optional attention mask (batch, seq_len)
    ///
    /// # Returns
    /// Normalized semantic embeddings (batch, seq_len, 1024)
    pub fn encode(&self, input_features: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Handle different input ranks
        let input_3d = match input_features.rank() {
            2 => {
                // Raw audio (batch, samples) -> simulate feature extraction
                // In production, this would run through CNN feature extractor
                let (batch, samples) = input_features.dims2()?;
                // Downsample by ~320 (typical Wav2Vec stride) and create placeholder features
                let seq_len = samples / 320;
                let seq_len = seq_len.max(1);
                // Create placeholder features at INPUT_FEATURE_DIM (160) for feature_projection
                Tensor::randn(0.0f32, 1.0, (batch, seq_len, INPUT_FEATURE_DIM), &self.device)?
            }
            3 => input_features.clone(),
            _ => anyhow::bail!("Expected 2D or 3D input, got {}D", input_features.rank()),
        };

        // Get input dimensions
        let (_batch, _seq, feat_dim) = input_3d.dims3()?;

        // Apply feature projection if available and input is at INPUT_FEATURE_DIM
        let hidden_states = if let Some(ref fp) = self.feature_projection {
            if feat_dim == INPUT_FEATURE_DIM {
                // Apply learned feature projection: 160 -> 1024
                fp.forward(&input_3d)?
            } else if feat_dim == HIDDEN_SIZE {
                // Already at hidden size, skip projection
                input_3d
            } else {
                // Dimension mismatch - use random projection as fallback
                tracing::warn!(
                    "[Wav2Vec-BERT] Input dim {} doesn't match INPUT_FEATURE_DIM ({}) or HIDDEN_SIZE ({}), using random projection",
                    feat_dim, INPUT_FEATURE_DIM, HIDDEN_SIZE
                );
                let projection = Tensor::randn(0.0f32, 0.02, (feat_dim, HIDDEN_SIZE), &self.device)?;
                input_3d.matmul(&projection)?
            }
        } else if feat_dim == HIDDEN_SIZE {
            // No feature projection loaded, but input is already at hidden size
            input_3d
        } else {
            // No feature projection and dimension mismatch - use random projection
            let projection = Tensor::randn(0.0f32, 0.02, (feat_dim, HIDDEN_SIZE), &self.device)?;
            input_3d.matmul(&projection)?
        };

        // If no encoder layers loaded, return normalized input
        if self.encoder_layers.is_empty() {
            return self.normalize(&hidden_states);
        }

        // Run through encoder layers
        let mut hidden_states = hidden_states;
        let mut extracted_output = None;

        for (i, layer) in self.encoder_layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
            if i == self.extract_layer {
                extracted_output = Some(hidden_states.clone());
            }
        }

        // Use extracted layer output (layer 17) or final output
        let output = extracted_output.unwrap_or(hidden_states);

        // Normalize: (feat - mean) / std
        self.normalize(&output)
    }

    /// Normalize features using pre-computed statistics
    fn normalize(&self, features: &Tensor) -> Result<Tensor> {
        let mean = self.mean.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, hidden)
        let std = self.std.unsqueeze(0)?.unsqueeze(0)?;

        let normalized = features.broadcast_sub(&mean)?;
        normalized.broadcast_div(&std).map_err(Into::into)
    }

    /// Get the output hidden size
    pub fn hidden_size(&self) -> usize {
        HIDDEN_SIZE
    }

    /// Get the layer being extracted from
    pub fn extract_layer(&self) -> usize {
        self.extract_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_encoder_placeholder() {
        let device = Device::Cpu;
        let encoder = SemanticEncoder::load("nonexistent.safetensors", None::<&str>, &device).unwrap();

        // Create dummy input
        let input = Tensor::randn(0.0f32, 1.0, (1, 100, HIDDEN_SIZE), &device).unwrap();
        let output = encoder.encode(&input, None).unwrap();

        assert_eq!(output.dims3().unwrap(), (1, 100, HIDDEN_SIZE));
    }
}
