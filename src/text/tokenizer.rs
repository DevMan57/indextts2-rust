//! BPE Tokenization
//!
//! Wrapper around HuggingFace tokenizers for BPE-based text tokenization

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
}

impl TextTokenizer {
    /// Load tokenizer from a BPE model file
    pub fn load<P: AsRef<Path>>(bpe_path: P, normalizer: TextNormalizer) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(bpe_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        let unk_token_id = tokenizer
            .token_to_id("[UNK]")
            .unwrap_or(0);
        
        Ok(Self {
            tokenizer,
            normalizer,
            unk_token_id,
        })
    }
    
    /// Tokenize text into token strings
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let normalized = self.normalizer.normalize(text);
        let encoding = self.tokenizer
            .encode(normalized.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        Ok(encoding.get_tokens().to_vec())
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
    
    /// Split tokens into segments
    pub fn split_segments(&self, tokens: &[String], max_tokens: usize) -> Vec<Vec<String>> {
        tokens
            .chunks(max_tokens)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}
