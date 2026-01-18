//! Utility functions and helpers for IndexTTS2
//!
//! This module provides common utilities used across the crate.

/// Tensor utilities
pub mod tensor_utils {
    use candle_core::{DType, Device, Result, Tensor};

    /// Create a causal mask as u8 tensor (1 = attend, 0 = mask)
    ///
    /// For autoregressive generation, position i can attend to positions <= i
    pub fn create_causal_mask_u8(
        query_len: usize,
        key_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
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
        mask.unsqueeze(0)?.unsqueeze(0)
    }

    /// Convert u8 mask to f32 mask for attention (1.0 = attend, -inf = mask)
    pub fn mask_to_attention_bias(mask: &Tensor) -> Result<Tensor> {
        // Where mask is 1, return 0.0; where mask is 0, return -inf
        let mask_f32 = mask.to_dtype(DType::F32)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, mask.device())?;
        let _zeros = Tensor::zeros_like(&mask_f32)?;
        
        // (1 - mask) * -inf: 0 where mask=1, -inf where mask=0
        let inv_mask = (Tensor::ones_like(&mask_f32)? - &mask_f32)?;
        let bias = inv_mask.broadcast_mul(&neg_inf.broadcast_as(inv_mask.shape())?)?;
        
        Ok(bias)
    }
}

/// String utilities
pub mod string_utils {
    /// Join a vector of strings with a separator
    pub fn join_strings(strings: &[String], separator: &str) -> String {
        strings.join(separator)
    }
}
