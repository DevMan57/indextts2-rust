//! Autoregressive generation for mel code prediction
//!
//! Implements the generation loop with:
//! - Top-k / Top-p (nucleus) sampling
//! - Temperature control
//! - Stop token detection
//! - Repetition penalty

use anyhow::Result;
use candle_core::{IndexOp, Tensor, D};
use rand::Rng;

use super::unified_voice::UnifiedVoice;

/// Generation configuration
#[derive(Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Minimum number of tokens before allowing stop
    pub min_length: usize,
    /// Temperature for sampling (1.0 = no change)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
    /// Stop token ID
    pub stop_token: usize,
    /// Start token ID
    pub start_token: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 1815,
            min_length: 10,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            stop_token: 8193,
            start_token: 8192,
        }
    }
}

/// Sampling methods for token selection
pub struct Sampler {
    rng: rand::rngs::ThreadRng,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Apply temperature to logits
    pub fn apply_temperature(&self, logits: &Tensor, temperature: f32) -> Result<Tensor> {
        if temperature == 1.0 {
            return Ok(logits.clone());
        }
        (logits / temperature as f64).map_err(Into::into)
    }

    /// Apply top-k filtering
    pub fn apply_top_k(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        if k == 0 {
            return Ok(logits.clone());
        }

        let vocab_size = logits.dim(D::Minus1)?;
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Handle 1D and 2D tensors
        let is_1d = logits.rank() == 1;
        let logits = if is_1d {
            logits.unsqueeze(0)?
        } else {
            logits.clone()
        };

        // Get the k-th largest value as threshold
        let sorted = logits.sort_last_dim(false)?;
        let sorted = sorted.0; // Values
        let threshold = sorted.i((.., k - 1))?;
        let threshold = threshold.unsqueeze(D::Minus1)?;

        // Mask out values below threshold
        let neg_inf = Tensor::new(f32::NEG_INFINITY, logits.device())?
            .broadcast_as(logits.shape())?;
        let mask = logits.ge(&threshold.broadcast_as(logits.shape())?)?;
        let result = mask.where_cond(&logits, &neg_inf)?;

        // Squeeze back if input was 1D
        if is_1d {
            result.squeeze(0).map_err(Into::into)
        } else {
            Ok(result)
        }
    }

    /// Apply top-p (nucleus) filtering
    pub fn apply_top_p(&self, logits: &Tensor, p: f32) -> Result<Tensor> {
        if p >= 1.0 {
            return Ok(logits.clone());
        }

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;

        // Sort probabilities in descending order and get indices
        let (sorted_probs, sorted_indices) = probs.sort_last_dim(false)?;

        // Compute cumulative sum
        let cumsum = cumulative_sum(&sorted_probs)?;

        // Create mask for tokens to keep in sorted order (cumsum <= p)
        let threshold = Tensor::new(p, logits.device())?
            .broadcast_as(cumsum.shape())?;
        let sorted_mask = cumsum.le(&threshold)?;

        // Always keep at least the top token (index 0 in sorted order)
        let sorted_mask_vec: Vec<u8> = sorted_mask.to_vec1()?;
        let mut mask_vec = sorted_mask_vec.clone();
        if mask_vec.iter().all(|&x| x == 0) {
            if !mask_vec.is_empty() {
                mask_vec[0] = 1;
            }
        }

        // Map the mask back to original indices
        let sorted_indices_vec: Vec<u32> = sorted_indices.to_vec1()?;
        let vocab_size = sorted_indices_vec.len();
        let mut original_mask = vec![0u8; vocab_size];
        for (sorted_idx, &original_idx) in sorted_indices_vec.iter().enumerate() {
            if sorted_idx < mask_vec.len() && mask_vec[sorted_idx] == 1 {
                original_mask[original_idx as usize] = 1;
            }
        }

        // Apply mask to original logits
        let mask = Tensor::from_slice(&original_mask, (vocab_size,), logits.device())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, logits.device())?
            .broadcast_as(logits.shape())?;

        mask.where_cond(logits, &neg_inf).map_err(Into::into)
    }

    /// Apply repetition penalty to logits
    pub fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        generated_tokens: &[u32],
        penalty: f32,
    ) -> Result<Tensor> {
        if penalty == 1.0 || generated_tokens.is_empty() {
            return Ok(logits.clone());
        }

        let mut logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let vocab_size = logits_vec.len();

        for &token in generated_tokens {
            let idx = token as usize;
            if idx < vocab_size {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }

        Tensor::from_slice(&logits_vec, logits.shape(), logits.device()).map_err(Into::into)
    }

    /// Sample from logits using multinomial sampling
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        // Convert to probabilities
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        // Sample using inverse CDF
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;

        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if cumsum > r {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probs_vec.len() - 1) as u32)
    }

    /// Argmax sampling (greedy)
    pub fn argmax(&self, logits: &Tensor) -> Result<u32> {
        let idx = logits.argmax(D::Minus1)?;
        idx.to_scalar::<u32>().map_err(Into::into)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple cumulative sum along last dimension
fn cumulative_sum(tensor: &Tensor) -> Result<Tensor> {
    let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let mut cumsum = Vec::with_capacity(values.len());
    let mut sum = 0.0;

    for v in values {
        sum += v;
        cumsum.push(sum);
    }

    Tensor::from_slice(&cumsum, tensor.shape(), tensor.device()).map_err(Into::into)
}

/// Generate mel codes autoregressively
///
/// # Arguments
/// * `model` - UnifiedVoice model
/// * `text_ids` - Input text token IDs
/// * `conditioning` - Optional audio conditioning tensor
/// * `config` - Generation configuration
///
/// # Returns
/// * Generated mel code sequence
pub fn generate(
    model: &mut UnifiedVoice,
    text_ids: &Tensor,
    conditioning: Option<&Tensor>,
    config: &GenerationConfig,
) -> Result<Vec<u32>> {
    let device = text_ids.device();
    let batch_size = text_ids.dim(0)?;

    if batch_size != 1 {
        anyhow::bail!("Generation currently only supports batch_size=1");
    }

    // Initialize model cache
    model.reset_cache();
    model.init_cache();

    let mut sampler = Sampler::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    let text_len = text_ids.dim(1)?;
    let cond_len = conditioning.map(|c| c.dim(1).unwrap_or(0)).unwrap_or(0);

    eprintln!("DEBUG generation: text_len={}, cond_len={}, max_length={}, stop_token={}",
        text_len, cond_len, config.max_length, config.stop_token);

    // === PREFILL PHASE ===
    // Process conditioning through model first (if any)
    let mut position = 0;
    if let Some(_cond) = conditioning {
        // Process conditioning tokens (they use text embedding space)
        // The conditioning is already in embedding space from process_conditioning
        // For proper prefill, we'd need to handle this differently
        // For now, we skip conditioning prefill and just advance position
        position = cond_len;
        eprintln!("DEBUG generation: Skipping conditioning prefill, advancing position by {}", cond_len);
    }

    // Prefill: Process all text tokens to fill the KV cache
    // This gives the model context about what text to synthesize
    let text_ids_vec: Vec<u32> = text_ids.flatten_all()?.to_vec1()?;
    eprintln!("DEBUG generation: text_ids = {:?}", &text_ids_vec[..text_ids_vec.len().min(20)]);

    // Text tokens use positions 0, 1, 2, ... in their own positional space
    for (i, &token) in text_ids_vec.iter().enumerate() {
        let input_id = Tensor::new(&[[token]], device)?;
        // Process as text token (is_mel=false), position is in text space
        let _logits = model.forward_one(&input_id, position + i, false)?;
    }
    eprintln!("DEBUG generation: prefill complete, processed {} text tokens", text_ids_vec.len());

    // === GENERATION PHASE ===
    // Mel tokens use their own positional space starting at 0
    let mut mel_position = 0usize;

    // Start with start token
    let mut current_token = config.start_token as u32;

    // Generation loop
    for step in 0..config.max_length {
        // Create input tensor
        let input_id = Tensor::new(&[[current_token]], device)?;

        // Forward pass (as mel token) - mel_position is in mel's positional space
        let logits = model.forward_one(&input_id, mel_position, true)?;

        // Get logits for the single position (already squeezed in forward_one)
        let logits = logits.squeeze(0)?; // (vocab_size,)

        // Debug: Check logits statistics on first few steps
        if step < 5 || step % 50 == 0 {
            let logits_mean: f32 = logits.mean_all()?.to_scalar()?;
            let logits_max: f32 = logits.max(D::Minus1)?.to_scalar()?;
            let logits_argmax = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            eprintln!("DEBUG step {}: logits mean={:.4}, max={:.4}, argmax={}",
                step, logits_mean, logits_max, logits_argmax);
        }

        // Apply repetition penalty
        let logits = sampler.apply_repetition_penalty(
            &logits,
            &generated_tokens,
            config.repetition_penalty,
        )?;

        // Apply temperature
        let logits = sampler.apply_temperature(&logits, config.temperature)?;

        // Apply top-k
        let logits = sampler.apply_top_k(&logits, config.top_k)?;

        // Apply top-p
        let logits = sampler.apply_top_p(&logits, config.top_p)?;

        // Sample next token
        let next_token = sampler.sample(&logits)?;

        // Check for stop token (but not before min_length)
        if next_token as usize == config.stop_token && step >= config.min_length {
            eprintln!("DEBUG generation: Stop token {} detected at step {}", next_token, step);
            break;
        }

        generated_tokens.push(next_token);
        current_token = next_token;
        mel_position += 1;
    }

    eprintln!("DEBUG generation: Generated {} mel codes", generated_tokens.len());
    Ok(generated_tokens)
}

/// Generate mel codes and capture hidden states for each step
///
/// This is similar to `generate` but also returns the GPT hidden states
/// for each generated token. These hidden states are used by the length
/// regulator to compute content features: S_infer = vq2emb(codes) + gpt_layer(latent)
///
/// # Arguments
/// * `model` - UnifiedVoice model
/// * `text_ids` - Input text token IDs
/// * `conditioning` - Optional audio conditioning tensor
/// * `config` - Generation configuration
///
/// # Returns
/// * Tuple of (mel codes, hidden states tensor)
/// * hidden_states: (1, num_codes, model_dim=1280)
pub fn generate_with_hidden(
    model: &mut UnifiedVoice,
    text_ids: &Tensor,
    conditioning: Option<&Tensor>,
    config: &GenerationConfig,
) -> Result<(Vec<u32>, Tensor)> {
    let device = text_ids.device();
    let batch_size = text_ids.dim(0)?;

    if batch_size != 1 {
        anyhow::bail!("Generation currently only supports batch_size=1");
    }

    // Initialize model cache
    model.reset_cache();
    model.init_cache();

    let mut sampler = Sampler::new();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut hidden_states_list: Vec<Tensor> = Vec::new();

    let text_len = text_ids.dim(1)?;
    let cond_len = conditioning.map(|c| c.dim(1).unwrap_or(0)).unwrap_or(0);

    eprintln!("DEBUG generate_with_hidden: text_len={}, cond_len={}, max_length={}",
        text_len, cond_len, config.max_length);

    // === PREFILL PHASE ===
    let mut position = 0;
    if conditioning.is_some() {
        position = cond_len;
    }

    // Prefill text tokens
    let text_ids_vec: Vec<u32> = text_ids.flatten_all()?.to_vec1()?;
    for (i, &token) in text_ids_vec.iter().enumerate() {
        let input_id = Tensor::new(&[[token]], device)?;
        let _logits = model.forward_one(&input_id, position + i, false)?;
    }

    // === GENERATION PHASE ===
    let mut mel_position = 0usize;
    let mut current_token = config.start_token as u32;

    for step in 0..config.max_length {
        let input_id = Tensor::new(&[[current_token]], device)?;

        // Forward pass with hidden states
        let (logits, hidden) = model.forward_one_with_hidden(&input_id, mel_position, true)?;

        // Get logits for the single position
        let logits = logits.squeeze(0)?;

        // Debug on first step
        if step == 0 {
            let hidden_mean: f32 = hidden.mean_all()?.to_scalar()?;
            let hidden_var: f32 = hidden.var(D::Minus1)?.mean_all()?.to_scalar()?;
            eprintln!("DEBUG step 0: hidden_states shape={:?}, mean={:.4}, var={:.4}",
                hidden.shape(), hidden_mean, hidden_var);
        }

        // Apply sampling
        let logits = sampler.apply_repetition_penalty(&logits, &generated_tokens, config.repetition_penalty)?;
        let logits = sampler.apply_temperature(&logits, config.temperature)?;
        let logits = sampler.apply_top_k(&logits, config.top_k)?;
        let logits = sampler.apply_top_p(&logits, config.top_p)?;

        let next_token = sampler.sample(&logits)?;

        // Check for stop token
        if next_token as usize == config.stop_token && step >= config.min_length {
            eprintln!("DEBUG generate_with_hidden: Stop token at step {}", step);
            break;
        }

        generated_tokens.push(next_token);
        // Squeeze the hidden state from (1, 1, 1280) to (1, 1280) for concatenation later
        hidden_states_list.push(hidden.squeeze(1)?);
        current_token = next_token;
        mel_position += 1;
    }

    // Concatenate all hidden states along sequence dimension
    // Each hidden is (1, 1280), stack to (1, num_codes, 1280)
    let hidden_states = if hidden_states_list.is_empty() {
        Tensor::zeros((1, 0, 1280), candle_core::DType::F32, device)?
    } else {
        Tensor::stack(&hidden_states_list, 1)?
    };

    eprintln!("DEBUG generate_with_hidden: Generated {} codes, hidden_states shape={:?}",
        generated_tokens.len(), hidden_states.shape());

    Ok((generated_tokens, hidden_states))
}

/// Generate with greedy decoding (no sampling)
pub fn generate_greedy(
    model: &mut UnifiedVoice,
    text_ids: &Tensor,
    conditioning: Option<&Tensor>,
    max_length: usize,
    stop_token: usize,
) -> Result<Vec<u32>> {
    let device = text_ids.device();

    model.reset_cache();
    model.init_cache();

    let sampler = Sampler::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    let _text_len = text_ids.dim(1)?;
    let cond_len = conditioning.map(|c| c.dim(1).unwrap_or(0)).unwrap_or(0);

    // === PREFILL PHASE ===
    let mut position = 0;
    if conditioning.is_some() {
        position = cond_len;
    }

    // Prefill text tokens
    let text_ids_vec: Vec<u32> = text_ids.flatten_all()?.to_vec1()?;
    for (i, &token) in text_ids_vec.iter().enumerate() {
        let input_id = Tensor::new(&[[token]], device)?;
        let _logits = model.forward_one(&input_id, position + i, false)?;
    }

    // === GENERATION PHASE ===
    // Mel tokens use their own positional space starting at 0
    let mut mel_position = 0usize;
    let mut current_token = model.start_token() as u32;

    for _ in 0..max_length {
        let input_id = Tensor::new(&[[current_token]], device)?;
        let logits = model.forward_one(&input_id, mel_position, true)?;
        let logits = logits.squeeze(0)?;

        let next_token = sampler.argmax(&logits)?;

        if next_token as usize == stop_token {
            break;
        }

        generated_tokens.push(next_token);
        current_token = next_token;
        mel_position += 1;
    }

    Ok(generated_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 1815);
        assert_eq!(config.stop_token, 8193);
        assert_eq!(config.temperature, 0.8);
    }

    #[test]
    fn test_sampler_temperature() {
        let device = Device::Cpu;
        let sampler = Sampler::new();

        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();

        // No change at temperature 1.0
        let scaled = sampler.apply_temperature(&logits, 1.0).unwrap();
        let orig: Vec<f32> = logits.to_vec1().unwrap();
        let new: Vec<f32> = scaled.to_vec1().unwrap();
        assert_eq!(orig, new);

        // Lower temperature = sharper distribution
        let scaled = sampler.apply_temperature(&logits, 0.5).unwrap();
        let values: Vec<f32> = scaled.to_vec1().unwrap();
        assert!((values[0] - 2.0).abs() < 0.001);
        assert!((values[2] - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_sampler_argmax() {
        let device = Device::Cpu;
        let sampler = Sampler::new();

        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 3.0], &device).unwrap();
        let idx = sampler.argmax(&logits).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_sampler_repetition_penalty() {
        let device = Device::Cpu;
        let sampler = Sampler::new();

        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
        let generated = vec![1, 2]; // Penalize tokens 1 and 2

        let penalized = sampler
            .apply_repetition_penalty(&logits, &generated, 1.5)
            .unwrap();
        let values: Vec<f32> = penalized.to_vec1().unwrap();

        // Token 0 unchanged
        assert!((values[0] - 1.0).abs() < 0.001);
        // Tokens 1, 2 should be reduced (positive values divided by penalty)
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);
        // Token 3 unchanged
        assert!((values[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_cumulative_sum() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[0.2f32, 0.3, 0.1, 0.4], &device).unwrap();
        let cumsum = cumulative_sum(&tensor).unwrap();
        let values: Vec<f32> = cumsum.to_vec1().unwrap();

        assert!((values[0] - 0.2).abs() < 0.001);
        assert!((values[1] - 0.5).abs() < 0.001);
        assert!((values[2] - 0.6).abs() < 0.001);
        assert!((values[3] - 1.0).abs() < 0.001);
    }
}
