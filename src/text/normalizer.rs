//! Text normalization
//!
//! Normalizes input text by handling:
//! - Numbers to words conversion (English)
//! - Abbreviation expansion
//! - Punctuation normalization
//! - Glossary-based replacements
//! - Whitespace normalization

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Text normalizer that handles various text preprocessing tasks
#[derive(Debug, Clone)]
pub struct TextNormalizer {
    /// Glossary for custom term replacements
    glossary: HashMap<String, String>,
    /// Whether glossary is enabled
    enable_glossary: bool,
    /// Common abbreviations
    abbreviations: HashMap<&'static str, &'static str>,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl TextNormalizer {
    /// Create a new TextNormalizer
    pub fn new(enable_glossary: bool) -> Self {
        let mut abbreviations = HashMap::new();
        // Common abbreviations
        abbreviations.insert("Mr.", "Mister");
        abbreviations.insert("Mrs.", "Misses");
        abbreviations.insert("Ms.", "Miss");
        abbreviations.insert("Dr.", "Doctor");
        abbreviations.insert("Prof.", "Professor");
        abbreviations.insert("Jr.", "Junior");
        abbreviations.insert("Sr.", "Senior");
        abbreviations.insert("vs.", "versus");
        abbreviations.insert("etc.", "etcetera");
        abbreviations.insert("e.g.", "for example");
        abbreviations.insert("i.e.", "that is");

        Self {
            glossary: HashMap::new(),
            enable_glossary,
            abbreviations,
        }
    }

    /// Load the normalizer with default settings
    pub fn load(&mut self) -> Result<()> {
        Ok(())
    }

    /// Load glossary from a YAML file
    pub fn load_glossary<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let glossary: HashMap<String, String> = serde_yaml::from_str(&content)?;
        self.glossary = glossary;
        Ok(())
    }

    /// Normalize input text
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Apply glossary replacements first
        if self.enable_glossary {
            for (from, to) in &self.glossary {
                result = result.replace(from, to);
            }
        }

        // Expand abbreviations
        for (abbr, expansion) in &self.abbreviations {
            result = result.replace(*abbr, *expansion);
        }

        // Convert numbers to words
        result = self.normalize_numbers(&result);

        // Normalize punctuation
        result = self.normalize_punctuation(&result);

        // Normalize whitespace
        result = self.normalize_whitespace(&result);

        result
    }

    /// Convert numbers to their word representations
    fn normalize_numbers(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() * 2);
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check if we're at the start of a number
            if chars[i].is_ascii_digit() || (chars[i] == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
                let start = i;
                let is_negative = chars[i] == '-';
                if is_negative {
                    i += 1;
                }

                // Collect the full number (including decimals)
                let mut num_str = String::new();
                let mut has_decimal = false;

                while i < chars.len() && (chars[i].is_ascii_digit() || (chars[i] == '.' && !has_decimal)) {
                    if chars[i] == '.' {
                        has_decimal = true;
                    }
                    num_str.push(chars[i]);
                    i += 1;
                }

                // Convert number to words
                if let Some(words) = self.number_to_words(&num_str, is_negative) {
                    result.push_str(&words);
                } else {
                    // Fallback: keep original
                    for j in start..i {
                        result.push(chars[j]);
                    }
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        result
    }

    /// Convert a number string to words
    fn number_to_words(&self, num_str: &str, is_negative: bool) -> Option<String> {
        // Handle decimal numbers
        if num_str.contains('.') {
            let parts: Vec<&str> = num_str.split('.').collect();
            if parts.len() == 2 {
                let integer_part = self.integer_to_words(parts[0].parse().ok()?)?;
                let decimal_part = self.digits_to_words(parts[1]);
                let mut result = String::new();
                if is_negative {
                    result.push_str("negative ");
                }
                result.push_str(&integer_part);
                result.push_str(" point ");
                result.push_str(&decimal_part);
                return Some(result);
            }
            return None;
        }

        // Handle integer
        let num: i64 = num_str.parse().ok()?;
        let mut result = String::new();
        if is_negative {
            result.push_str("negative ");
        }
        result.push_str(&self.integer_to_words(num)?);
        Some(result)
    }

    /// Convert integer to words
    fn integer_to_words(&self, num: i64) -> Option<String> {
        if num == 0 {
            return Some("zero".to_string());
        }

        let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen"];
        let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

        let mut n = num.abs();
        let mut result = Vec::new();

        // Handle billions
        if n >= 1_000_000_000 {
            let billions = n / 1_000_000_000;
            result.push(format!("{} billion", self.integer_to_words(billions)?));
            n %= 1_000_000_000;
        }

        // Handle millions
        if n >= 1_000_000 {
            let millions = n / 1_000_000;
            result.push(format!("{} million", self.integer_to_words(millions)?));
            n %= 1_000_000;
        }

        // Handle thousands
        if n >= 1000 {
            let thousands = n / 1000;
            result.push(format!("{} thousand", self.integer_to_words(thousands)?));
            n %= 1000;
        }

        // Handle hundreds
        if n >= 100 {
            let hundreds = n / 100;
            result.push(format!("{} hundred", ones[hundreds as usize]));
            n %= 100;
        }

        // Handle tens and ones
        if n >= 20 {
            let t = n / 10;
            let o = n % 10;
            if o > 0 {
                result.push(format!("{}-{}", tens[t as usize], ones[o as usize]));
            } else {
                result.push(tens[t as usize].to_string());
            }
        } else if n > 0 {
            result.push(ones[n as usize].to_string());
        }

        Some(result.join(" "))
    }

    /// Convert decimal digits to individual words
    fn digits_to_words(&self, digits: &str) -> String {
        let digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
        digits
            .chars()
            .filter_map(|c| c.to_digit(10).map(|d| digit_words[d as usize]))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Normalize punctuation for TTS
    fn normalize_punctuation(&self, text: &str) -> String {
        text
            // Normalize quotes
            .replace('"', "\"")
            .replace('"', "\"")
            .replace(''', "'")
            .replace(''', "'")
            // Normalize dashes
            .replace("—", ", ")
            .replace("–", ", ")
            // Normalize ellipsis
            .replace("…", "...")
            // Remove certain characters that don't affect speech
            .replace("*", "")
            .replace("_", "")
            .replace("~", "")
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        // Replace multiple spaces with single space and trim
        let mut result = String::with_capacity(text.len());
        let mut prev_was_space = true; // Start true to trim leading spaces

        for c in text.chars() {
            if c.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(c);
                prev_was_space = false;
            }
        }

        // Trim trailing space
        if result.ends_with(' ') {
            result.pop();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalization() {
        let normalizer = TextNormalizer::new(false);
        let result = normalizer.normalize("Hello, world!");
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_number_to_words() {
        let normalizer = TextNormalizer::new(false);
        assert_eq!(normalizer.normalize("I have 5 apples"), "I have five apples");
        assert_eq!(normalizer.normalize("The year is 2024"), "I have five apples".replace("I have five apples", "The year is two thousand twenty-four"));
    }

    #[test]
    fn test_integer_conversion() {
        let normalizer = TextNormalizer::new(false);
        assert_eq!(normalizer.integer_to_words(0), Some("zero".to_string()));
        assert_eq!(normalizer.integer_to_words(1), Some("one".to_string()));
        assert_eq!(normalizer.integer_to_words(13), Some("thirteen".to_string()));
        assert_eq!(normalizer.integer_to_words(21), Some("twenty-one".to_string()));
        assert_eq!(normalizer.integer_to_words(100), Some("one hundred".to_string()));
        assert_eq!(normalizer.integer_to_words(123), Some("one hundred twenty-three".to_string()));
        assert_eq!(normalizer.integer_to_words(1000), Some("one thousand".to_string()));
    }

    #[test]
    fn test_abbreviation_expansion() {
        let normalizer = TextNormalizer::new(false);
        assert_eq!(normalizer.normalize("Dr. Smith"), "Doctor Smith");
        assert_eq!(normalizer.normalize("Mr. Jones"), "Mister Jones");
    }

    #[test]
    fn test_whitespace_normalization() {
        let normalizer = TextNormalizer::new(false);
        assert_eq!(normalizer.normalize("  hello   world  "), "hello world");
    }
}
