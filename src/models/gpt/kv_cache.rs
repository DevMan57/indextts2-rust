//! KV-Cache for efficient autoregressive generation
//!
//! Implements key-value caching for transformer attention layers.
//! During generation, only the new token needs to be processed while
//! previous key-value pairs are retrieved from cache.
//!
//! This dramatically speeds up autoregressive generation by avoiding
//! redundant computation of attention for past tokens.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

/// Single layer KV cache
///
/// Stores keys and values for one attention layer.
pub struct LayerCache {
    /// Cached keys (batch, num_heads, seq_len, head_dim)
    keys: Option<Tensor>,
    /// Cached values (batch, num_heads, seq_len, head_dim)
    values: Option<Tensor>,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length
    current_len: usize,
}

impl LayerCache {
    /// Create a new layer cache
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            keys: None,
            values: None,
            max_seq_len,
            current_len: 0,
        }
    }

    /// Append new keys and values to the cache
    ///
    /// # Arguments
    /// * `new_keys` - New keys (batch, num_heads, new_seq_len, head_dim)
    /// * `new_values` - New values (batch, num_heads, new_seq_len, head_dim)
    ///
    /// # Returns
    /// * Tuple of (all_keys, all_values) including cached and new
    pub fn append(&mut self, new_keys: &Tensor, new_values: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = new_keys.dim(2)?;

        let (keys, values) = if let (Some(ref cached_k), Some(ref cached_v)) =
            (&self.keys, &self.values)
        {
            // Concatenate with existing cache
            let keys = Tensor::cat(&[cached_k, new_keys], 2)?;
            let values = Tensor::cat(&[cached_v, new_values], 2)?;
            (keys, values)
        } else {
            // First append - just use the new values
            (new_keys.clone(), new_values.clone())
        };

        // Update cache
        self.keys = Some(keys.clone());
        self.values = Some(values.clone());
        self.current_len += new_seq_len;

        Ok((keys, values))
    }

    /// Get the current cached keys and values
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Get current sequence length in cache
    pub fn current_seq_len(&self) -> usize {
        self.current_len
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.current_len = 0;
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }
}

/// Full KV cache for all layers
///
/// Manages caches for each transformer layer.
pub struct KVCache {
    /// Per-layer caches
    layer_caches: Vec<LayerCache>,
    /// Number of layers
    num_layers: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl KVCache {
    /// Create a new KV cache for multiple layers
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        let layer_caches = (0..num_layers)
            .map(|_| LayerCache::new(max_seq_len))
            .collect();

        Self {
            layer_caches,
            num_layers,
            max_seq_len,
        }
    }

    /// Append to a specific layer's cache
    pub fn append(
        &mut self,
        layer_idx: usize,
        new_keys: &Tensor,
        new_values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.num_layers {
            anyhow::bail!(
                "Layer index {} out of range (num_layers={})",
                layer_idx,
                self.num_layers
            );
        }
        self.layer_caches[layer_idx].append(new_keys, new_values)
    }

    /// Get cached keys and values for a layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        self.layer_caches.get(layer_idx).and_then(|c| c.get())
    }

    /// Get current sequence length (from first layer)
    pub fn current_seq_len(&self) -> usize {
        self.layer_caches
            .first()
            .map(|c| c.current_seq_len())
            .unwrap_or(0)
    }

    /// Reset all layer caches
    pub fn reset(&mut self) {
        for cache in &mut self.layer_caches {
            cache.reset();
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.layer_caches.first().map(|c| c.is_empty()).unwrap_or(true)
    }

    /// Get mutable access to a specific layer cache
    pub fn get_layer_cache_mut(&mut self, layer_idx: usize) -> Option<&mut LayerCache> {
        self.layer_caches.get_mut(layer_idx)
    }

    /// Get immutable access to a specific layer cache
    pub fn get_layer_cache(&self, layer_idx: usize) -> Option<&LayerCache> {
        self.layer_caches.get(layer_idx)
    }
}

/// Attention with KV-cache support
///
/// Efficient attention computation that uses cached keys and values
/// for autoregressive generation.
pub struct CachedAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CachedAttention {
    /// Create new cached attention
    pub fn new(
        dim: usize,
        num_heads: usize,
        q_proj: candle_nn::Linear,
        k_proj: candle_nn::Linear,
        v_proj: candle_nn::Linear,
        out_proj: candle_nn::Linear,
    ) -> Self {
        let head_dim = dim / num_heads;
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        }
    }

    /// Create with random weights
    pub fn new_random(dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = dim / num_heads;

        let make_linear = |device: &Device| -> Result<candle_nn::Linear> {
            let w = Tensor::randn(0.0f32, 0.02, (dim, dim), device)?;
            let b = Tensor::zeros((dim,), DType::F32, device)?;
            Ok(candle_nn::Linear::new(w, Some(b)))
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

    /// Forward with KV-cache
    ///
    /// # Arguments
    /// * `x` - Input tensor (batch, seq_len, dim)
    /// * `cache` - Layer cache for storing/retrieving KV pairs
    /// * `causal_mask` - Whether to apply causal masking
    ///
    /// # Returns
    /// * Attention output (batch, seq_len, dim)
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut LayerCache,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = candle_nn::Module::forward(&self.q_proj, x)?;
        let k = candle_nn::Module::forward(&self.k_proj, x)?;
        let v = candle_nn::Module::forward(&self.v_proj, x)?;

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

        // Make tensors contiguous
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Append to cache and get full K, V
        let (k, v) = cache.append(&k, &v)?;
        let kv_seq_len = k.dim(2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = (attn_weights / scale)?;

        // Apply causal mask if needed
        let attn_weights = if causal_mask && seq_len > 1 {
            // Create causal mask
            let mask = self.create_causal_mask(seq_len, kv_seq_len, x.device())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?
                .broadcast_as(attn_weights.shape())?;
            let _zeros = Tensor::zeros_like(&attn_weights)?;
            // mask is (1, 1, seq_len, kv_seq_len), broadcast to attention weights
            let mask = mask.broadcast_as(attn_weights.shape())?;
            mask.where_cond(&attn_weights, &neg_inf)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let v = v.contiguous()?;
        let attn_output = attn_weights.contiguous()?.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        candle_nn::Module::forward(&self.out_proj, &attn_output).map_err(Into::into)
    }

    /// Create causal attention mask
    fn create_causal_mask(
        &self,
        query_len: usize,
        key_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // For autoregressive generation, position i can only attend to positions <= i
        // Shape: (1, 1, query_len, key_len)
        // Use u8: 1 = can attend, 0 = cannot attend
        let start_pos = key_len.saturating_sub(query_len);
        let mut mask_data = vec![0u8; query_len * key_len];

        for q in 0..query_len {
            for k in 0..key_len {
                // Can attend if key position <= query position (in absolute terms)
                if k <= (start_pos + q) {
                    mask_data[q * key_len + k] = 1;
                }
            }
        }

        let mask = Tensor::from_slice(&mask_data, (query_len, key_len), device)?;
        mask.unsqueeze(0)?.unsqueeze(0).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_new() {
        let cache = LayerCache::new(2048);
        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
    }

    #[test]
    fn test_layer_cache_append() {
        let device = Device::Cpu;
        let mut cache = LayerCache::new(2048);

        // First append
        let k1 = Tensor::randn(0.0f32, 1.0, (1, 8, 5, 64), &device).unwrap();
        let v1 = Tensor::randn(0.0f32, 1.0, (1, 8, 5, 64), &device).unwrap();
        let (keys, _values) = cache.append(&k1, &v1).unwrap();

        assert_eq!(keys.dims4().unwrap(), (1, 8, 5, 64));
        assert_eq!(cache.current_seq_len(), 5);

        // Second append
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 8, 1, 64), &device).unwrap();
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 8, 1, 64), &device).unwrap();
        let (keys, _values) = cache.append(&k2, &v2).unwrap();

        assert_eq!(keys.dims4().unwrap(), (1, 8, 6, 64));
        assert_eq!(cache.current_seq_len(), 6);
    }

    #[test]
    fn test_layer_cache_reset() {
        let device = Device::Cpu;
        let mut cache = LayerCache::new(2048);

        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 10, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 10, 64), &device).unwrap();
        cache.append(&k, &v).unwrap();

        assert!(!cache.is_empty());

        cache.reset();

        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(24, 2048);

        assert_eq!(cache.num_layers(), 24);
        assert!(cache.is_empty());

        // Append to different layers
        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 5, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 5, 64), &device).unwrap();

        cache.append(0, &k, &v).unwrap();
        cache.append(1, &k, &v).unwrap();

        assert_eq!(cache.current_seq_len(), 5);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cached_attention() {
        let device = Device::Cpu;
        let attn = CachedAttention::new_random(512, 8, &device).unwrap();
        let mut cache = LayerCache::new(2048);

        // First forward (prefill)
        let x = Tensor::randn(0.0f32, 1.0, (1, 10, 512), &device).unwrap();
        let out = attn.forward(&x, &mut cache, true).unwrap();

        assert_eq!(out.dims3().unwrap(), (1, 10, 512));
        assert_eq!(cache.current_seq_len(), 10);

        // Incremental forward (generation)
        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 512), &device).unwrap();
        let out = attn.forward(&x, &mut cache, true).unwrap();

        assert_eq!(out.dims3().unwrap(), (1, 1, 512));
        assert_eq!(cache.current_seq_len(), 11);
    }
}
