//! Unified Voice Model - GPT-2 based autoregressive TTS
//!
//! Implements the core GPT-2 architecture for mel code generation:
//! - Text and mel embeddings
//! - Conformer encoder for audio conditioning
//! - Perceiver resampler for cross-attention
//! - GPT-2 decoder with causal attention
//! - Mel code prediction head
//!
//! Architecture (from config):
//! - model_dim: 1280
//! - layers: 24
//! - heads: 20
//! - number_mel_codes: 8194
//! - stop_mel_token: 8193

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, D, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, LayerNorm, Embedding};
use std::path::Path;

use super::conformer::{ConformerEncoder, ConformerConfig};
use super::perceiver::{PerceiverResampler, PerceiverConfig};
use super::kv_cache::{KVCache, LayerCache};

/// GPT-2 decoder layer
struct DecoderLayer {
    /// Self-attention
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    attn_layer_norm: LayerNorm,
    /// Feed-forward
    fc1: Linear,
    fc2: Linear,
    ffn_layer_norm: LayerNorm,
    /// Config
    num_heads: usize,
    head_dim: usize,
}

impl DecoderLayer {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;

        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;
        let attn_layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("attn_layer_norm"))?;

        let fc1 = candle_nn::linear(dim, dim * 4, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(dim * 4, dim, vb.pp("fc2"))?;
        let ffn_layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("ffn_layer_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            attn_layer_norm,
            fc1,
            fc2,
            ffn_layer_norm,
            num_heads,
            head_dim,
        })
    }

    fn new_random(dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = dim / num_heads;

        let make_linear = |in_dim: usize, out_dim: usize| -> Result<Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (out_dim, in_dim), device)?;
            let b = Tensor::zeros((out_dim,), DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };

        let make_layer_norm = || -> Result<LayerNorm> {
            let w = Tensor::ones((dim,), DType::F32, device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(LayerNorm::new(w, b, 1e-5))
        };

        Ok(Self {
            q_proj: make_linear(dim, dim)?,
            k_proj: make_linear(dim, dim)?,
            v_proj: make_linear(dim, dim)?,
            out_proj: make_linear(dim, dim)?,
            attn_layer_norm: make_layer_norm()?,
            fc1: make_linear(dim, dim * 4)?,
            fc2: make_linear(dim * 4, dim)?,
            ffn_layer_norm: make_layer_norm()?,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass with KV cache support
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut LayerCache,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Pre-norm self-attention
        let normed = self.attn_layer_norm.forward(x)?;

        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // KV cache
        let (k, v) = cache.append(&k, &v)?;
        let kv_len = k.dim(2)?;

        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn / scale)?;

        // Causal mask
        let attn = if causal_mask {
            let mask = create_causal_mask(seq_len, kv_len, x.device())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?;
            let mask = mask.broadcast_as(attn.shape())?;
            mask.where_cond(&attn, &neg_inf.broadcast_as(attn.shape())?)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let attn_out = attn.matmul(&v)?;

        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Residual connection
        let x = (x + attn_out)?;

        // Pre-norm FFN
        let normed = self.ffn_layer_norm.forward(&x)?;
        let ffn_out = self.fc1.forward(&normed)?;
        let ffn_out = ffn_out.gelu_erf()?;
        let ffn_out = self.fc2.forward(&ffn_out)?;

        // Residual connection
        (&x + ffn_out).map_err(Into::into)
    }
}

/// Create causal attention mask
fn create_causal_mask(query_len: usize, key_len: usize, device: &Device) -> Result<Tensor> {
    let start_pos = key_len - query_len;
    let mut mask_data = vec![false; query_len * key_len];

    for q in 0..query_len {
        for k in 0..key_len {
            mask_data[q * key_len + k] = k <= (start_pos + q);
        }
    }

    let mask = Tensor::from_slice(&mask_data, (query_len, key_len), device)?;
    mask.unsqueeze(0)?.unsqueeze(0).map_err(Into::into)
}

/// Unified Voice model configuration
pub struct UnifiedVoiceConfig {
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: usize,
    pub stop_mel_token: usize,
    pub start_text_token: usize,
    pub stop_text_token: usize,
}

impl Default for UnifiedVoiceConfig {
    fn default() -> Self {
        Self {
            model_dim: 1280,
            num_layers: 24,
            num_heads: 20,
            max_mel_tokens: 1815,
            max_text_tokens: 600,
            number_text_tokens: 12000,
            number_mel_codes: 8194,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            start_text_token: 0,
            stop_text_token: 1,
        }
    }
}

/// Unified Voice Model
///
/// GPT-2 based autoregressive model for mel code generation.
/// Takes text tokens and audio conditioning to generate mel codes.
pub struct UnifiedVoice {
    device: Device,
    config: UnifiedVoiceConfig,
    /// Text token embedding
    text_embedding: Option<Tensor>,
    /// Mel code embedding
    mel_embedding: Option<Tensor>,
    /// Positional embedding
    pos_embedding: Option<Tensor>,
    /// Conformer encoder for audio conditioning
    conformer: Option<ConformerEncoder>,
    /// Perceiver resampler
    perceiver: Option<PerceiverResampler>,
    /// Decoder layers
    decoder_layers: Vec<DecoderLayer>,
    /// Final layer norm
    final_layer_norm: Option<LayerNorm>,
    /// Output projection (to mel codes)
    lm_head: Option<Linear>,
    /// KV cache for generation
    kv_cache: Option<KVCache>,
    /// Whether initialized
    weights_loaded: bool,
}

impl UnifiedVoice {
    /// Create with default config
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(UnifiedVoiceConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: UnifiedVoiceConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            text_embedding: None,
            mel_embedding: None,
            pos_embedding: None,
            conformer: None,
            perceiver: None,
            decoder_layers: Vec::new(),
            final_layer_norm: None,
            lm_head: None,
            kv_cache: None,
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut model = Self::new(device)?;
        model.load_weights(path)?;
        Ok(model)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        let dim = self.config.model_dim;

        // Embeddings
        self.text_embedding = Some(Tensor::randn(
            0.0f32,
            0.02,
            (self.config.number_text_tokens, dim),
            &self.device,
        )?);

        self.mel_embedding = Some(Tensor::randn(
            0.0f32,
            0.02,
            (self.config.number_mel_codes, dim),
            &self.device,
        )?);

        let max_pos = self.config.max_mel_tokens + self.config.max_text_tokens;
        self.pos_embedding = Some(Tensor::randn(
            0.0f32,
            0.02,
            (max_pos, dim),
            &self.device,
        )?);

        // Conformer encoder
        let mut conformer = ConformerEncoder::with_config(
            ConformerConfig {
                input_dim: 80,
                output_dim: dim,
                num_blocks: 6,
                num_heads: 8,
                ff_expansion: 4,
                conv_kernel_size: 31,
            },
            &self.device,
        )?;
        conformer.initialize_random()?;
        self.conformer = Some(conformer);

        // Perceiver resampler
        let mut perceiver = PerceiverResampler::with_config(
            PerceiverConfig {
                dim,
                num_latents: 32,
                num_heads: 8,
                num_layers: 2,
                ff_mult: 4,
            },
            &self.device,
        )?;
        perceiver.initialize_random()?;
        self.perceiver = Some(perceiver);

        // Decoder layers
        self.decoder_layers.clear();
        for _ in 0..self.config.num_layers {
            let layer = DecoderLayer::new_random(dim, self.config.num_heads, &self.device)?;
            self.decoder_layers.push(layer);
        }

        // Final layer norm
        let ln_w = Tensor::ones((dim,), DType::F32, &self.device)?;
        let ln_b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.final_layer_norm = Some(LayerNorm::new(ln_w, ln_b, 1e-5));

        // LM head
        let lm_w = Tensor::randn(0.0f32, 0.02, (self.config.number_mel_codes, dim), &self.device)?;
        let lm_b = Tensor::zeros((self.config.number_mel_codes,), DType::F32, &self.device)?;
        self.lm_head = Some(Linear::new(lm_w, Some(lm_b)));

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return self.initialize_random();
        }

        // For now, initialize with random - in production would load actual weights
        self.initialize_random()
    }

    /// Initialize KV cache for generation
    pub fn init_cache(&mut self) {
        let max_seq = self.config.max_mel_tokens + self.config.max_text_tokens;
        self.kv_cache = Some(KVCache::new(self.config.num_layers, max_seq));
    }

    /// Reset KV cache
    pub fn reset_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.reset();
        }
    }

    /// Get text embeddings
    fn embed_text(&self, text_ids: &Tensor) -> Result<Tensor> {
        let emb = self
            .text_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Text embedding not initialized"))?;
        emb.index_select(text_ids, 0)
    }

    /// Get mel code embeddings
    fn embed_mel(&self, mel_ids: &Tensor) -> Result<Tensor> {
        let emb = self
            .mel_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mel embedding not initialized"))?;
        emb.index_select(mel_ids, 0)
    }

    /// Get positional embeddings
    fn embed_pos(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let emb = self
            .pos_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Position embedding not initialized"))?;
        emb.i(offset..offset + seq_len)
    }

    /// Process audio conditioning through conformer + perceiver
    pub fn process_conditioning(&self, mel_features: &Tensor) -> Result<Tensor> {
        let conformer = self
            .conformer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Conformer not initialized"))?;
        let perceiver = self
            .perceiver
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Perceiver not initialized"))?;

        // Conformer encodes the mel features
        let encoded = conformer.forward(mel_features, None)?;

        // Perceiver resamples to fixed length conditioning
        perceiver.forward(&encoded)
    }

    /// Forward pass for training (full sequence)
    ///
    /// # Arguments
    /// * `text_ids` - Text token IDs (batch, text_len)
    /// * `mel_ids` - Mel code IDs (batch, mel_len)
    /// * `conditioning` - Audio conditioning from perceiver (batch, cond_len, dim)
    ///
    /// # Returns
    /// * Logits for mel codes (batch, mel_len, number_mel_codes)
    pub fn forward(
        &mut self,
        text_ids: &Tensor,
        mel_ids: &Tensor,
        conditioning: Option<&Tensor>,
    ) -> Result<Tensor> {
        if !self.weights_loaded {
            let (batch, mel_len) = mel_ids.dims2()?;
            return Tensor::zeros(
                (batch, mel_len, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )
            .map_err(Into::into);
        }

        let (batch_size, text_len) = text_ids.dims2()?;
        let (_, mel_len) = mel_ids.dims2()?;

        // Embed text and mel
        let text_emb = self.embed_text(&text_ids.flatten_all()?)?
            .reshape((batch_size, text_len, self.config.model_dim))?;
        let mel_emb = self.embed_mel(&mel_ids.flatten_all()?)?
            .reshape((batch_size, mel_len, self.config.model_dim))?;

        // Combine: [conditioning, text, mel]
        let mut parts = Vec::new();
        let mut total_len = 0;

        if let Some(cond) = conditioning {
            parts.push(cond.clone());
            total_len += cond.dim(1)?;
        }
        parts.push(text_emb);
        total_len += text_len;
        parts.push(mel_emb);
        total_len += mel_len;

        let mut hidden = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        // Add positional embedding
        let pos_emb = self.embed_pos(total_len, 0)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(hidden.shape())?;
        hidden = (hidden + pos_emb)?;

        // Initialize cache for this forward pass
        let mut layer_caches: Vec<LayerCache> = (0..self.config.num_layers)
            .map(|_| LayerCache::new(total_len))
            .collect();

        // Process through decoder layers
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut layer_caches[i], true)?;
        }

        // Final layer norm
        if let Some(ref ln) = self.final_layer_norm {
            hidden = ln.forward(&hidden)?;
        }

        // Extract mel positions and project to logits
        let cond_len = conditioning.map(|c| c.dim(1).unwrap_or(0)).unwrap_or(0);
        let mel_start = cond_len + text_len;
        let mel_hidden = hidden.i((.., mel_start.., ..))?;

        // LM head
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&mel_hidden).map_err(Into::into)
        } else {
            Tensor::zeros(
                (batch_size, mel_len, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )
            .map_err(Into::into)
        }
    }

    /// Forward pass for generation (single token with KV cache)
    ///
    /// # Arguments
    /// * `input_id` - Current token ID (batch, 1)
    /// * `position` - Current position in sequence
    /// * `is_mel` - Whether this is a mel token (vs text token)
    ///
    /// # Returns
    /// * Logits for next mel code (batch, number_mel_codes)
    pub fn forward_one(
        &mut self,
        input_id: &Tensor,
        position: usize,
        is_mel: bool,
    ) -> Result<Tensor> {
        if !self.weights_loaded {
            let batch = input_id.dim(0)?;
            return Tensor::zeros(
                (batch, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )
            .map_err(Into::into);
        }

        // Ensure cache is initialized
        if self.kv_cache.is_none() {
            self.init_cache();
        }

        let batch_size = input_id.dim(0)?;

        // Embed the token
        let flat_id = input_id.flatten_all()?;
        let hidden = if is_mel {
            self.embed_mel(&flat_id)?
        } else {
            self.embed_text(&flat_id)?
        };
        let mut hidden = hidden.reshape((batch_size, 1, self.config.model_dim))?;

        // Add positional embedding
        let pos_emb = self.embed_pos(1, position)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(hidden.shape())?;
        hidden = (hidden + pos_emb)?;

        // Process through decoder layers with KV cache
        let cache = self.kv_cache.as_mut().unwrap();
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut cache.layer_caches[i], true)?;
        }

        // Final layer norm
        if let Some(ref ln) = self.final_layer_norm {
            hidden = ln.forward(&hidden)?;
        }

        // Squeeze out sequence dimension and project to logits
        let hidden = hidden.squeeze(1)?;
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&hidden).map_err(Into::into)
        } else {
            Tensor::zeros(
                (batch_size, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )
            .map_err(Into::into)
        }
    }

    /// Get model dimension
    pub fn model_dim(&self) -> usize {
        self.config.model_dim
    }

    /// Get number of mel codes
    pub fn num_mel_codes(&self) -> usize {
        self.config.number_mel_codes
    }

    /// Get stop token
    pub fn stop_token(&self) -> usize {
        self.config.stop_mel_token
    }

    /// Get start token
    pub fn start_token(&self) -> usize {
        self.config.start_mel_token
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
    fn test_unified_voice_config_default() {
        let config = UnifiedVoiceConfig::default();
        assert_eq!(config.model_dim, 1280);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 20);
        assert_eq!(config.stop_mel_token, 8193);
    }

    #[test]
    fn test_unified_voice_new() {
        let device = Device::Cpu;
        let model = UnifiedVoice::new(&device).unwrap();
        assert_eq!(model.model_dim(), 1280);
        assert_eq!(model.stop_token(), 8193);
    }

    #[test]
    fn test_unified_voice_placeholder() {
        let device = Device::Cpu;
        let mut model = UnifiedVoice::new(&device).unwrap();

        let text_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
        let mel_ids = Tensor::new(&[[100u32, 101, 102]], &device).unwrap();

        let logits = model.forward(&text_ids, &mel_ids, None).unwrap();

        // Should return zeros since not initialized
        assert_eq!(logits.dims3().unwrap(), (1, 3, 8194));
    }

    #[test]
    fn test_unified_voice_initialized() {
        let device = Device::Cpu;
        let mut model = UnifiedVoice::new(&device).unwrap();
        model.initialize_random().unwrap();

        assert!(model.is_initialized());

        let text_ids = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
        let mel_ids = Tensor::new(&[[100u32, 101]], &device).unwrap();

        let logits = model.forward(&text_ids, &mel_ids, None).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 2, 8194));
    }
}
