//! Text Tokenization
//!
//! Wrapper around HuggingFace tokenizers for text tokenization.
//! Supports loading from tokenizer.json files (Unigram/BPE models).

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

use super::TextNormalizer;

/// Tokenize text by inserting spaces around CJK characters
/// This matches the Python tokenize_by_CJK_char function
fn tokenize_by_cjk_char(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);

    for c in text.chars() {
        // Check if character is CJK
        if is_cjk_character(c) {
            // Add space before and after CJK character
            if !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }
            result.push(c);
            result.push(' ');
        } else {
            result.push(c);
        }
    }

    // Clean up multiple spaces
    let mut cleaned = String::with_capacity(result.len());
    let mut prev_was_space = true;
    for c in result.chars() {
        if c == ' ' {
            if !prev_was_space {
                cleaned.push(' ');
                prev_was_space = true;
            }
        } else {
            cleaned.push(c);
            prev_was_space = false;
        }
    }

    cleaned.trim().to_string()
}

/// Check if a character is a CJK character
fn is_cjk_character(c: char) -> bool {
    let cp = c as u32;

    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp) ||
    // CJK Unified Ideographs Extension A
    (0x3400..=0x4DBF).contains(&cp) ||
    // CJK Unified Ideographs Extension B
    (0x20000..=0x2A6DF).contains(&cp) ||
    // CJK Unified Ideographs Extension C
    (0x2A700..=0x2B73F).contains(&cp) ||
    // CJK Unified Ideographs Extension D
    (0x2B740..=0x2B81F).contains(&cp) ||
    // CJK Unified Ideographs Extension E
    (0x2B820..=0x2CEAF).contains(&cp) ||
    // CJK Unified Ideographs Extension F
    (0x2CEB0..=0x2EBEF).contains(&cp) ||
    // CJK Compatibility Ideographs
    (0xF900..=0xFAFF).contains(&cp) ||
    // CJK Compatibility Ideographs Supplement
    (0x2F800..=0x2FA1F).contains(&cp)
}

/// Check if text contains any CJK characters
fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_character)
}

/// HuggingFace tokenizers-based text tokenizer
pub struct TextTokenizer {
    /// Underlying tokenizer
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
    /// Load tokenizer from a tokenizer.json file
    pub fn load<P: AsRef<Path>>(model_path: P, normalizer: TextNormalizer) -> Result<Self> {
        let path = model_path.as_ref();

        // Check if file exists
        if !path.exists() {
            anyhow::bail!("Tokenizer model not found: {:?}", path);
        }

        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", path, e))?;

        // Get UNK token ID (default to 2)
        let unk_token_id = tokenizer
            .token_to_id("<unk>")
            .unwrap_or(2);

        Ok(Self {
            tokenizer,
            normalizer,
            unk_token_id,
            start_text_token: 0,  // <s> token
            stop_text_token: 1,   // </s> token
        })
    }

    /// Set special token IDs from GPT config
    pub fn set_special_tokens(&mut self, start: u32, stop: u32) {
        self.start_text_token = start;
        self.stop_text_token = stop;
    }

    /// Preprocess text: normalize and uppercase English
    fn preprocess(&self, text: &str) -> String {
        // Normalize text
        let normalized = self.normalizer.normalize(text);

        // Check if text contains CJK characters
        if contains_cjk(&normalized) {
            // For Chinese text, apply CJK pre-tokenization
            tokenize_by_cjk_char(&normalized)
        } else {
            // For English text, convert to uppercase
            // The vocabulary uses uppercase English tokens
            normalized.to_uppercase()
        }
    }

    /// Tokenize text into token strings
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let preprocessed = self.preprocess(text);

        let encoding = self.tokenizer.encode(preprocessed, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        Ok(encoding.get_tokens().to_vec())
    }

    /// Encode text directly to token IDs (most common use case)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let preprocessed = self.preprocess(text);

        let encoding = self.tokenizer.encode(preprocessed, false)
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
        self.tokenizer.decode(ids, true)
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
    fn test_hello_tokenization() {
        let normalizer = TextNormalizer::new(false);
        // Try to load the real tokenizer if it exists, otherwise skip
        if std::path::Path::new("checkpoints/tokenizer_english.json").exists() {
            let tokenizer = TextTokenizer::load("checkpoints/tokenizer_english.json", normalizer).unwrap();
            let ids = tokenizer.encode("Hello").unwrap();
            let tokens = tokenizer.convert_ids_to_tokens(&ids);
            println!("HELLO IDS: {:?}", ids);
            println!("HELLO TOKENS: {:?}", tokens);
        }
    }

    #[test]
    fn test_cjk_tokenization() {
        let result = tokenize_by_cjk_char("你好world");
        // CJK characters are separated by spaces
        // Output: "你 好 world" (first char has no leading space)
        assert!(result.contains("你 "));
        assert!(result.contains("好 "));
        // Also test middle positions
        let result2 = tokenize_by_cjk_char("hello你好world");
        assert!(result2.contains(" 你 "));
        assert!(result2.contains(" 好 "));
    }

    #[test]
    fn test_is_cjk_character() {
        assert!(is_cjk_character('你'));
        assert!(is_cjk_character('好'));
        assert!(!is_cjk_character('a'));
        assert!(!is_cjk_character('!'));
    }

    #[test]
    fn test_contains_cjk() {
        assert!(contains_cjk("你好世界"));
        assert!(contains_cjk("Hello 你好"));
        assert!(!contains_cjk("Hello World"));
    }
}
