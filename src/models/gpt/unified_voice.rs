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

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, LayerNorm};
use crate::config::{GptConfig, ConditionModuleConfig};
use crate::utils::parity_dump;
use std::collections::HashMap;
use std::path::Path;

use super::conformer::{ConformerEncoder, ConformerConfig};
use super::perceiver::{PerceiverResampler, PerceiverConfig};
use super::kv_cache::{KVCache, LayerCache};
use super::weights::{load_safetensors, load_embedding, load_layer_norm, Gpt2LayerWeights};

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

    /// Create from loaded weights
    fn from_weights(weights: Gpt2LayerWeights, num_heads: usize) -> Result<Self> {
        // Infer head_dim from the q_proj weight shape
        let (out_dim, _in_dim) = weights.q_proj.weight().dims2()?;
        let head_dim = out_dim / num_heads;

        Ok(Self {
            q_proj: weights.q_proj,
            k_proj: weights.k_proj,
            v_proj: weights.v_proj,
            out_proj: weights.out_proj,
            attn_layer_norm: weights.attn_ln,
            fc1: weights.fc1,
            fc2: weights.fc2,
            ffn_layer_norm: weights.ffn_ln,
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
        trace_prefix: Option<&str>,
    ) -> Result<Tensor> {
        let dump = |suffix: &str, tensor: &Tensor| {
            if let Some(prefix) = trace_prefix {
                let name = format!("{prefix}_{suffix}");
                parity_dump::dump_tensor_f32(&name, tensor);
            }
        };
        dump("input", x);

        let (batch_size, seq_len, _) = x.dims3()?;

        // Pre-norm self-attention
        let normed = self.attn_layer_norm.forward(x)?;
        dump("ln1", &normed);

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

        // Make tensors contiguous before KV cache
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // KV cache
        let (k, v) = cache.append(&k, &v)?;
        dump("q", &q);
        dump("k", &k);
        dump("v", &v);
        let kv_len = k.dim(2)?;

        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = q.matmul(&k_t)?;
        let attn = (attn / scale)?;
        dump("attn_scores_pre_mask", &attn);

        // Causal mask
        let attn = if causal_mask {
            let mask = create_causal_mask(seq_len, kv_len, x.device())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?;
            let mask = mask.broadcast_as(attn.shape())?;
            mask.where_cond(&attn, &neg_inf.broadcast_as(attn.shape())?)?
        } else {
            attn
        };
        dump("attn_scores_post_mask", &attn);

        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let v = v.contiguous()?;
        let attn_out = attn.contiguous()?.matmul(&v)?;

        let attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        dump("attn_out_pre_out_proj", &attn_out);
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Residual connection
        let x = (x + attn_out)?;

        // Pre-norm FFN
        let normed = self.ffn_layer_norm.forward(&x)?;
        dump("ln2", &normed);
        let ffn_out = self.fc1.forward(&normed)?;
        let ffn_out = ffn_out.gelu_erf()?;
        let ffn_out = self.fc2.forward(&ffn_out)?;

        // Residual connection
        let out = (&x + ffn_out)?;
        dump("out", &out);
        Ok(out)
    }
}

/// Create causal attention mask
/// Returns u8 tensor: 1 = can attend, 0 = cannot attend
fn create_causal_mask(query_len: usize, key_len: usize, device: &Device) -> Result<Tensor> {
    let start_pos = key_len.saturating_sub(query_len);
    let mut mask_data = vec![0u8; query_len * key_len];

    for q in 0..query_len {
        for k in 0..key_len {
            if k <= (start_pos + q) {
                mask_data[q * key_len + k] = 1;
            }
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
fn default_condition_module() -> ConditionModuleConfig {
    ConditionModuleConfig {
        output_size: 512,
        linear_units: 2048,
        attention_heads: 8,
        num_blocks: 6,
        input_layer: "conv2d2".to_string(),
        perceiver_mult: 2,
    }
}

fn default_emo_condition_module() -> ConditionModuleConfig {
    ConditionModuleConfig {
        output_size: 512,
        linear_units: 1024,
        attention_heads: 4,
        num_blocks: 4,
        input_layer: "conv2d2".to_string(),
        perceiver_mult: 2,
    }
}

fn ff_expansion_from(cfg: &ConditionModuleConfig) -> usize {
    let denom = cfg.output_size.max(1);
    let exp = cfg.linear_units / denom;
    if exp == 0 { 4 } else { exp }
}
/// Unified Voice Model
///
/// GPT-2 based autoregressive model for mel code generation.
/// Takes text tokens and audio conditioning to generate mel codes.
pub struct UnifiedVoice {
    device: Device,
    config: UnifiedVoiceConfig,
    condition_type: String,
    cond_config: ConditionModuleConfig,
    emo_cond_config: ConditionModuleConfig,
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
    /// Emotion conditioning conformer
    emo_conformer: Option<ConformerEncoder>,
    /// Emotion perceiver resampler
    emo_perceiver: Option<PerceiverResampler>,
    /// Emotion vector projection layers
    emovec_layer: Option<Linear>,
    emo_layer: Option<Linear>,
    /// Speed embedding (2 entries)
    speed_emb: Option<Tensor>,
    /// Decoder layers
    decoder_layers: Vec<DecoderLayer>,
    /// Final layer norm
    final_layer_norm: Option<LayerNorm>,
    /// Extra norm before mel head (Python inference `final_norm`)
    lm_head_norm: Option<LayerNorm>,
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
    /// Create from model config
    pub fn from_gpt_config(cfg: &GptConfig, device: &Device) -> Result<Self> {
        let uv_cfg = UnifiedVoiceConfig {
            model_dim: cfg.model_dim,
            num_layers: cfg.layers,
            num_heads: cfg.heads,
            max_mel_tokens: cfg.max_mel_tokens,
            max_text_tokens: cfg.max_text_tokens,
            number_text_tokens: cfg.number_text_tokens,
            number_mel_codes: cfg.number_mel_codes,
            start_mel_token: cfg.start_mel_token,
            stop_mel_token: cfg.stop_mel_token,
            start_text_token: cfg.start_text_token,
            stop_text_token: cfg.stop_text_token,
        };
        let mut model = Self::with_config(uv_cfg, device)?;
        model.condition_type = cfg.condition_type.clone();
        model.cond_config = cfg.condition_module.clone();
        model.emo_cond_config = cfg.emo_condition_module.clone();
        Ok(model)
    }

    /// Create with custom config
    pub fn with_config(config: UnifiedVoiceConfig, device: &Device) -> Result<Self> {
        let condition_type = "conformer_perceiver".to_string();
        let cond_config = default_condition_module();
        let emo_cond_config = default_emo_condition_module();
        Ok(Self {
            device: device.clone(),
            config,
            condition_type,
            cond_config,
            emo_cond_config,
            text_embedding: None,
            mel_embedding: None,
            pos_embedding: None,
            conformer: None,
            perceiver: None,
            emo_conformer: None,
            emo_perceiver: None,
            emovec_layer: None,
            emo_layer: None,
            speed_emb: None,
            decoder_layers: Vec::new(),
            final_layer_norm: None,
            lm_head_norm: None,
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

        // Conformer encoder - outputs condition_module.output_size dim
        let cond_ff = ff_expansion_from(&self.cond_config);
        let mut conformer = ConformerEncoder::with_config(
            ConformerConfig {
                input_dim: 1024,
                output_dim: self.cond_config.output_size,
                num_blocks: self.cond_config.num_blocks,
                num_heads: self.cond_config.attention_heads,
                ff_expansion: cond_ff,
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
                context_dim: self.cond_config.output_size,
                num_latents: 32,
                num_heads: self.cond_config.attention_heads,
                num_layers: 2,
                ff_mult: self.cond_config.perceiver_mult,
                attn_dim: self.cond_config.output_size,
            },
            &self.device,
        )?;
        perceiver.initialize_random()?;
        self.perceiver = Some(perceiver);

        // Emotion conditioning encoder + perceiver
        let emo_ff = ff_expansion_from(&self.emo_cond_config);
        let mut emo_conformer = ConformerEncoder::with_config(
            ConformerConfig {
                input_dim: 1024,
                output_dim: self.emo_cond_config.output_size,
                num_blocks: self.emo_cond_config.num_blocks,
                num_heads: self.emo_cond_config.attention_heads,
                ff_expansion: emo_ff,
                conv_kernel_size: 31,
            },
            &self.device,
        )?;
        emo_conformer.initialize_random()?;
        self.emo_conformer = Some(emo_conformer);

        let mut emo_perceiver = PerceiverResampler::with_config(
            PerceiverConfig {
                dim: 1024,
                context_dim: self.emo_cond_config.output_size,
                num_latents: 1,
                num_heads: self.emo_cond_config.attention_heads,
                num_layers: 2,
                ff_mult: self.emo_cond_config.perceiver_mult,
                attn_dim: self.emo_cond_config.output_size,
            },
            &self.device,
        )?;
        emo_perceiver.initialize_random()?;
        self.emo_perceiver = Some(emo_perceiver);

        // Emotion projection layers
        let emovec_w = Tensor::randn(0.0f32, 0.02, (dim, 1024), &self.device)?;
        let emovec_b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.emovec_layer = Some(Linear::new(emovec_w, Some(emovec_b)));

        let emo_w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
        let emo_b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.emo_layer = Some(Linear::new(emo_w, Some(emo_b)));

        // Speed embedding (2 entries)
        self.speed_emb = Some(Tensor::zeros((2, dim), DType::F32, &self.device)?);

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

        // lm_head norm
        let lm_ln_w = Tensor::ones((dim,), DType::F32, &self.device)?;
        let lm_ln_b = Tensor::zeros((dim,), DType::F32, &self.device)?;
        self.lm_head_norm = Some(LayerNorm::new(lm_ln_w, lm_ln_b, 1e-5));

        // LM head
        let lm_w = Tensor::randn(0.0f32, 0.02, (self.config.number_mel_codes, dim), &self.device)?;
        let lm_b = Tensor::zeros((self.config.number_mel_codes,), DType::F32, &self.device)?;
        self.lm_head = Some(Linear::new(lm_w, Some(lm_b)));

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            eprintln!("Warning: Checkpoint not found at {:?}, using random weights", path);
            return self.initialize_random();
        }

        eprintln!("Loading GPT weights from {:?}...", path);
        let tensors = load_safetensors(path, &self.device)?;

        let dim = self.config.model_dim;

        // Load embeddings
        self.text_embedding = Some(load_embedding(
            &tensors,
            "text_embedding.weight",
            Some(self.config.number_text_tokens),
            dim,
            &self.device,
        )?);

        self.mel_embedding = Some(load_embedding(
            &tensors,
            "mel_embedding.weight",
            Some(self.config.number_mel_codes),
            dim,
            &self.device,
        )?);

        // Combine text and mel positional embeddings
        let text_pos = tensors
            .get("text_pos_embedding.emb.weight")
            .ok_or_else(|| anyhow::anyhow!("text_pos_embedding not found"))?;
        let mel_pos = tensors
            .get("mel_pos_embedding.emb.weight")
            .ok_or_else(|| anyhow::anyhow!("mel_pos_embedding not found"))?;

        // Concatenate: [text_pos (602), mel_pos (1818)] for combined position embedding
        self.pos_embedding = Some(Tensor::cat(&[text_pos, mel_pos], 0)?);

        // Load decoder layers
        self.decoder_layers.clear();
        for i in 0..self.config.num_layers {
            let layer_weights = Gpt2LayerWeights::load(
                &tensors,
                i,
                dim,
                &self.device,
            )?;
            let layer = DecoderLayer::from_weights(layer_weights, self.config.num_heads)?;
            self.decoder_layers.push(layer);
        }

        // Final layer norm (gpt.ln_f)
        self.final_layer_norm = Some(load_layer_norm(
            &tensors,
            "gpt.ln_f.weight",
            Some("gpt.ln_f.bias"),
            1e-5,
            &self.device,
        )?);

        self.lm_head_norm = Some(load_layer_norm(
            &tensors,
            "final_norm.weight",
            Some("final_norm.bias"),
            1e-5,
            &self.device,
        )?);

        // LM head (mel_head) - need to transpose for candle
        let mel_head_weight = tensors
            .get("mel_head.weight")
            .ok_or_else(|| anyhow::anyhow!("mel_head.weight not found"))?
            .clone();
        let mel_head_bias = tensors.get("mel_head.bias").cloned();
        self.lm_head = Some(Linear::new(mel_head_weight, mel_head_bias));

        // Note: Conformer and Perceiver loading requires separate implementation
        // For now, initialize them randomly (they'll need their own load methods)
        self.load_conformer_weights(&tensors)?;
        self.load_perceiver_weights(&tensors)?;
        self.load_emo_conditioning_weights(&tensors)?;
        self.load_emo_layers(&tensors)?;

        self.weights_loaded = true;
        eprintln!("GPT weights loaded successfully: {} layers", self.config.num_layers);
        Ok(())
    }

    /// Load conformer encoder weights
    #[allow(unused_variables)]
    fn load_conformer_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let cond_ff = ff_expansion_from(&self.cond_config);
        let mut conformer = ConformerEncoder::with_config(
            ConformerConfig {
                input_dim: 1024,
                output_dim: self.cond_config.output_size,
                num_blocks: self.cond_config.num_blocks,
                num_heads: self.cond_config.attention_heads,
                ff_expansion: cond_ff,
                conv_kernel_size: 31,
            },
            &self.device,
        )?;

        // Load actual conformer weights from tensors
        conformer.load_from_gpt_tensors_with_prefix(tensors, "conditioning_encoder")?;
        self.conformer = Some(conformer);
        Ok(())
    }

    /// Load perceiver resampler weights from GPT checkpoint tensors
    fn load_perceiver_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        // Check if perceiver_encoder weights exist in checkpoint
        let has_perceiver = tensors.keys().any(|k| k.starts_with("perceiver_encoder"));

        let mut perceiver = PerceiverResampler::with_config(
            PerceiverConfig {
                dim: self.config.model_dim,
                context_dim: self.cond_config.output_size,
                num_latents: 32,
                num_heads: self.cond_config.attention_heads,
                num_layers: 2,
                ff_mult: self.cond_config.perceiver_mult,
                attn_dim: self.cond_config.output_size,
            },
            &self.device,
        )?;

        if has_perceiver {
            // Load from GPT tensors using perceiver's own loader
            perceiver.load_from_gpt_tensors_with_prefix(tensors, "perceiver_encoder")?;
            eprintln!("  Perceiver weights loaded from checkpoint");
        } else {
            eprintln!("  Warning: No perceiver_encoder weights found, using random");
            perceiver.initialize_random()?;
        }

        self.perceiver = Some(perceiver);
        Ok(())
    }

    fn load_emo_conditioning_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let has_emo = tensors.keys().any(|k| k.starts_with("emo_conditioning_encoder"));
        let emo_ff = ff_expansion_from(&self.emo_cond_config);
        let mut emo_conformer = ConformerEncoder::with_config(
            ConformerConfig {
                input_dim: 1024,
                output_dim: self.emo_cond_config.output_size,
                num_blocks: self.emo_cond_config.num_blocks,
                num_heads: self.emo_cond_config.attention_heads,
                ff_expansion: emo_ff,
                conv_kernel_size: 31,
            },
            &self.device,
        )?;

        if has_emo {
            emo_conformer.load_from_gpt_tensors_with_prefix(tensors, "emo_conditioning_encoder")?;
        } else {
            emo_conformer.initialize_random()?;
        }
        self.emo_conformer = Some(emo_conformer);

        let mut emo_perceiver = PerceiverResampler::with_config(
            PerceiverConfig {
                dim: 1024,
                context_dim: self.emo_cond_config.output_size,
                num_latents: 1,
                num_heads: self.emo_cond_config.attention_heads,
                num_layers: 2,
                ff_mult: self.emo_cond_config.perceiver_mult,
                attn_dim: self.emo_cond_config.output_size,
            },
            &self.device,
        )?;

        if tensors.keys().any(|k| k.starts_with("emo_perceiver_encoder")) {
            emo_perceiver.load_from_gpt_tensors_with_prefix(tensors, "emo_perceiver_encoder")?;
        } else {
            emo_perceiver.initialize_random()?;
        }

        self.emo_perceiver = Some(emo_perceiver);
        Ok(())
    }

    fn load_emo_layers(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let dim = self.config.model_dim;

        // emovec_layer: 1024 -> model_dim
        let emovec_w = tensors.get("emovec_layer.weight").cloned();
        let emovec_b = tensors.get("emovec_layer.bias").cloned();
        if let Some(w) = emovec_w {
            self.emovec_layer = Some(Linear::new(w, emovec_b));
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, 1024), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.emovec_layer = Some(Linear::new(w, Some(b)));
        }

        // emo_layer: model_dim -> model_dim
        let emo_w = tensors.get("emo_layer.weight").cloned();
        let emo_b = tensors.get("emo_layer.bias").cloned();
        if let Some(w) = emo_w {
            self.emo_layer = Some(Linear::new(w, emo_b));
        } else {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), &self.device)?;
            let b = Tensor::zeros((dim,), DType::F32, &self.device)?;
            self.emo_layer = Some(Linear::new(w, Some(b)));
        }

        // speed_emb: [2, model_dim]
        if let Some(w) = tensors.get("speed_emb.weight") {
            self.speed_emb = Some(w.clone());
        } else {
            self.speed_emb = Some(Tensor::zeros((2, dim), DType::F32, &self.device)?);
        }

        Ok(())
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

    /// Current sequence length stored in the KV cache.
    pub fn kv_cache_current_len(&self) -> usize {
        self.kv_cache
            .as_ref()
            .map(|cache| cache.current_seq_len())
            .unwrap_or(0)
    }

    /// Get text embeddings
    fn embed_text(&self, text_ids: &Tensor) -> Result<Tensor> {
        let emb = self
            .text_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Text embedding not initialized"))?;
        Ok(emb.index_select(text_ids, 0)?)
    }

    /// Get mel code embeddings
    fn embed_mel(&self, mel_ids: &Tensor) -> Result<Tensor> {
        let emb = self
            .mel_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mel embedding not initialized"))?;
        Ok(emb.index_select(mel_ids, 0)?)
    }

    /// Get positional embeddings
    fn embed_pos(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let emb = self
            .pos_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Position embedding not initialized"))?;
        Ok(emb.i(offset..offset + seq_len)?)
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

        // Conformer encodes the mel features (outputs 512 dim)
        let encoded = conformer.forward(mel_features, None)?;

        // Perceiver resamples to fixed length conditioning (proj_context: 512->1280)
        perceiver.forward(&encoded)
    }
    /// Process emotion conditioning through emo conformer + perceiver
    pub fn process_emo_conditioning(&self, features: &Tensor) -> Result<Tensor> {
        let conformer = self
            .emo_conformer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Emotion conformer not initialized"))?;
        let perceiver = self
            .emo_perceiver
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Emotion perceiver not initialized"))?;

        let encoded = conformer.forward(features, None)?;
        perceiver.forward(&encoded)
    }

    fn get_emovec(&self, features: &Tensor) -> Result<Tensor> {
        let emovec_layer = self
            .emovec_layer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("emovec_layer not initialized"))?;
        let emo_layer = self
            .emo_layer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("emo_layer not initialized"))?;

        let cond = self.process_emo_conditioning(features)?; // expected (B, 1, 1024)
        let cond_shape = cond.shape().clone();
        let cond = cond.squeeze(1)?;

        let emovec_w_shape = emovec_layer.weight().shape().clone();
        let emo_w_shape = emo_layer.weight().shape().clone();
        eprintln!(
            "DEBUG get_emovec: cond={:?}, squeezed={:?}, emovec_w={:?}, emo_w={:?}",
            cond_shape,
            cond.shape(),
            emovec_w_shape,
            emo_w_shape
        );

        let emovec = emovec_layer
            .forward(&cond)
            .map_err(|e| anyhow::anyhow!("emovec_layer forward failed (cond={:?}, weight={:?}): {}", cond.shape(), emovec_w_shape, e))?;
        emo_layer
            .forward(&emovec)
            .map_err(|e| anyhow::anyhow!("emo_layer forward failed (emovec={:?}, weight={:?}): {}", emovec.shape(), emo_w_shape, e))
    }

    /// Merge emotion vector from base and emotion features
    pub fn merge_emovec(&self, base_features: &Tensor, emo_features: &Tensor, alpha: f32) -> Result<Tensor> {
        let base_vec = self.get_emovec(base_features)?;
        let emo_vec = self.get_emovec(emo_features)?;
        // base + alpha * (emo - base)
        let delta = (&emo_vec - &base_vec)?;
        (&base_vec + (delta * alpha as f64)?).map_err(Into::into)
    }

    /// Embed mel codes to continuous features
    ///
    /// # Arguments
    /// * `mel_codes` - Mel code IDs (batch, seq_len) as u32
    ///
    /// # Returns
    /// * Mel embeddings (batch, seq_len, 1280)
    pub fn embed_mel_codes(&self, mel_codes: &Tensor) -> Result<Tensor> {
        let emb = self
            .mel_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mel embedding not initialized"))?;

        // Debug: Check embedding table statistics
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let emb_mean: f32 = emb.mean_all().unwrap().to_scalar().unwrap();
            let emb_var: f32 = emb.var(candle_core::D::Minus1).unwrap().mean_all().unwrap().to_scalar().unwrap();
            let emb_shape = emb.shape();
            eprintln!("DEBUG: mel_embedding table shape={:?}, mean={:.6}, var={:.6}", emb_shape, emb_mean, emb_var);
        });

        let (batch_size, seq_len) = mel_codes.dims2()?;
        let flat = mel_codes.flatten_all()?;
        let embedded = emb.index_select(&flat, 0)?;

        embedded.reshape((batch_size, seq_len, self.config.model_dim))
            .map_err(Into::into)
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
            hidden = layer.forward(&hidden, &mut layer_caches[i], true, None)?;
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
        let hidden = hidden.reshape((batch_size, 1, self.config.model_dim))?;

        self.forward_one_embedding(&hidden, position, is_mel)
    }

    /// Forward pass for generation using pre-computed embeddings
    pub fn forward_one_embedding(
        &mut self,
        embedding: &Tensor,
        position: usize,
        is_mel: bool,
    ) -> Result<Tensor> {
        let (logits, _) = self.forward_one_embedding_with_hidden_opts(
            embedding,
            position,
            is_mel,
            true,
        )?;
        Ok(logits)
    }

    /// Forward pass for generation using pre-computed embeddings, returning logits and hidden states
    pub fn forward_one_embedding_with_hidden(
        &mut self,
        embedding: &Tensor,
        position: usize,
        is_mel: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_one_embedding_with_hidden_opts(embedding, position, is_mel, true)
    }

    /// Forward pass for generation using pre-computed embeddings with explicit positional embedding control.
    pub fn forward_one_embedding_with_hidden_opts(
        &mut self,
        embedding: &Tensor,
        position: usize,
        is_mel: bool,
        add_position_embedding: bool,
    ) -> Result<(Tensor, Tensor)> {
        if !self.weights_loaded {
            let batch = embedding.dim(0)?;
            let logits = Tensor::zeros(
                (batch, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )?;
            return Ok((logits, embedding.clone()));
        }

        // Ensure cache is initialized
        if self.kv_cache.is_none() {
            self.init_cache();
        }

        let (batch_size, seq_len, _) = embedding.dims3()?;
        if seq_len != 1 {
            anyhow::bail!("forward_one_embedding only supports seq_len=1, got {}", seq_len);
        }

        let mut hidden = embedding.clone();
        let trace_step = is_mel
            && position <= 1
            && std::env::var_os("INDEXTTS2_PARITY_DIR").is_some();
        let trace_step_prefix = trace_step.then(|| format!("rust_gpt_step{position}"));
        let parity_trace_blocks = std::env::var("INDEXTTS2_PARITY_TRACE_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2);

        // Add positional embedding
        if add_position_embedding {
            let text_pos_size = self.config.max_text_tokens + 2;
            let pos_index = if is_mel {
                // Match Python GPT2InferenceModel cached-decode indexing:
                // step0 (start token) uses mel pos 0, then step1 uses mel pos 2.
                // (Python computes this as attention_mask_len - mel_len.)
                let mel_pos = if position == 0 { 0 } else { position + 1 };
                text_pos_size + mel_pos
            } else {
                position
            };
            let pos_emb = self.embed_pos(1, pos_index)?;
            let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(hidden.shape())?;
            hidden = (hidden + pos_emb)?;
        }

        // Process through decoder layers with KV cache
        let cache = self.kv_cache.as_mut().unwrap();
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            let layer_cache = cache.get_layer_cache_mut(i)
                .ok_or_else(|| anyhow::anyhow!("Layer cache {} not found", i))?;
            let trace_prefix = if i < parity_trace_blocks {
                trace_step_prefix
                    .as_ref()
                    .map(|step_prefix| format!("{step_prefix}_block_{i:02}"))
            } else {
                None
            };
            hidden = layer.forward(&hidden, layer_cache, true, trace_prefix.as_deref())?;
        }

        // Final layer norm
        if let Some(step_prefix) = trace_step_prefix.as_deref() {
            let name = format!("{step_prefix}_pre_final_ln");
            parity_dump::dump_tensor_f32(&name, &hidden);
        }
        if let Some(ref ln) = self.final_layer_norm {
            hidden = ln.forward(&hidden)?;
        }
        if let Some(step_prefix) = trace_step_prefix.as_deref() {
            let name = format!("{step_prefix}_post_final_ln");
            parity_dump::dump_tensor_f32(&name, &hidden);
        }

        // Save hidden states after the transformer's final norm (used for latent features).
        let hidden_states = hidden.clone();

        // Python inference applies an additional `final_norm` before mel_head projection.
        // Mirror that behavior for logits parity while keeping `hidden_states` as post-ln_f.
        let mut hidden = hidden.squeeze(1)?;
        if let Some(ref ln) = self.lm_head_norm {
            hidden = ln.forward(&hidden)?;
        } else if let Some(ref ln) = self.final_layer_norm {
            hidden = ln.forward(&hidden)?;
        }
        if let Some(step_prefix) = trace_step_prefix.as_deref() {
            let name = format!("{step_prefix}_pre_lm_head");
            parity_dump::dump_tensor_f32(&name, &hidden);
        }
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&hidden)?
        } else {
            Tensor::zeros(
                (batch_size, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )?
        };
        if let Some(step_prefix) = trace_step_prefix.as_deref() {
            let name = format!("{step_prefix}_post_lm_head");
            parity_dump::dump_tensor_f32(&name, &logits);
        }

        Ok((logits, hidden_states))
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

    /// Get text start token
    pub fn start_text_token(&self) -> usize {
        self.config.start_text_token
    }

    /// Get text stop token
    pub fn stop_text_token(&self) -> usize {
        self.config.stop_text_token
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }

    /// Forward pass for generation (single token with KV cache) returning both logits and hidden states
    ///
    /// # Arguments
    /// * `input_id` - Current token ID (batch, 1)
    /// * `position` - Current position in sequence
    /// * `is_mel` - Whether this is a mel token (vs text token)
    ///
    /// # Returns
    /// * Tuple of (logits, hidden_states):
    ///   - logits: (batch, number_mel_codes)
    ///   - hidden_states: (batch, 1, model_dim) - the hidden state before lm_head projection
    pub fn forward_one_with_hidden(
        &mut self,
        input_id: &Tensor,
        position: usize,
        is_mel: bool,
    ) -> Result<(Tensor, Tensor)> {
        if !self.weights_loaded {
            let batch = input_id.dim(0)?;
            let logits = Tensor::zeros(
                (batch, self.config.number_mel_codes),
                DType::F32,
                &self.device,
            )?;
            let hidden = Tensor::zeros(
                (batch, 1, self.config.model_dim),
                DType::F32,
                &self.device,
            )?;
            return Ok((logits, hidden));
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
        let hidden = hidden.reshape((batch_size, 1, self.config.model_dim))?;

        self.forward_one_embedding_with_hidden(&hidden, position, is_mel)
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


















