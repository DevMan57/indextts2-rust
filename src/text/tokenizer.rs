//! BPE Tokenization
//!
//! Wrapper around HuggingFace tokenizers for BPE-based text tokenization.
//! Supports loading from JSON tokenizer files and provides comprehensive
//! encode/decode functionality.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

use super::TextNormalizer;

/// BPE-based text tokenizer
pub struct TextTokenizer {
    /// Underlying HuggingFace tokenizer
    tokenizer: Tokenizer,
    /// Text normalizer for preprocessing
    normalizer: TextNormalizer,
    /// Unknown token ID
    pub unk_token_id: u32,
    /// Start text token ID (for GPT conditioning)
    pub start_text_token: u32,
    /// Stop text token ID
    pub stop_text_token: u32,
}

impl TextTokenizer {
    /// Load tokenizer from a BPE model file (JSON format)
    pub fn load<P: AsRef<Path>>(bpe_path: P, normalizer: TextNormalizer) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(bpe_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", bpe_path.as_ref(), e))?;

        // Get special token IDs
        let unk_token_id = tokenizer
            .token_to_id("[UNK]")
            .or_else(|| tokenizer.token_to_id("<unk>"))
            .unwrap_or(0);

        Ok(Self {
            tokenizer,
            normalizer,
            unk_token_id,
            start_text_token: 0,  // Will be set from config
            stop_text_token: 1,   // Will be set from config
        })
    }

    /// Set special token IDs from GPT config
    pub fn set_special_tokens(&mut self, start: u32, stop: u32) {
        self.start_text_token = start;
        self.stop_text_token = stop;
    }

    /// Tokenize text into token strings
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let normalized = self.normalizer.normalize(text);
        let encoding = self.tokenizer
            .encode(normalized.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        Ok(encoding.get_tokens().to_vec())
    }

    /// Encode text directly to token IDs (most common use case)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = self.normalizer.normalize(text);
        let encoding = self.tokenizer
            .encode(normalized.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens for GPT input
    pub fn encode_for_gpt(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.start_text_token];
        ids.extend(self.encode(text)?);
        ids.push(self.stop_text_token);
        Ok(ids)
    }

    /// Convert tokens to token IDs
    pub fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .map(|t| self.tokenizer.token_to_id(t).unwrap_or(self.unk_token_id))
            .collect()
    }

    /// Convert token IDs back to tokens
    pub fn convert_ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| self.tokenizer.id_to_token(id))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true) // skip_special_tokens = true
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Check if a token exists in vocabulary
    pub fn token_exists(&self, token: &str) -> bool {
        self.tokenizer.token_to_id(token).is_some()
    }

    /// Split tokens into segments respecting max_tokens limit
    pub fn split_segments(&self, tokens: &[String], max_tokens: usize) -> Vec<Vec<String>> {
        tokens
            .chunks(max_tokens)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_integration() {
        let normalizer = TextNormalizer::new(false);
        // Test would require actual tokenizer file
    }
}
