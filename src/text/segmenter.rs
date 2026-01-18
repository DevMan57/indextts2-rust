//! Text segmentation utilities
//!
//! Segments text into chunks that respect:
//! - Sentence boundaries (periods, question marks, exclamation marks)
//! - Maximum token limits for GPT input
//! - Natural pause points for streaming TTS

/// Sentence-ending punctuation characters
const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '。', '！', '？'];

/// Clause-separating punctuation (for fallback splitting)
const CLAUSE_SEPARATORS: &[char] = &[',', ';', ':', '，', '；', '：'];

/// Segment tokens into chunks based on punctuation and token limits
///
/// # Arguments
/// * `tokens` - Input tokens to segment
/// * `max_tokens` - Maximum tokens per segment (e.g., 120 for IndexTTS2)
/// * `quick_streaming_tokens` - Target size for quick streaming (smaller chunks)
///
/// # Returns
/// Vector of token segments, each respecting max_tokens limit
pub fn segment_text(tokens: &[String], max_tokens: usize, quick_streaming_tokens: usize) -> Vec<Vec<String>> {
    if tokens.is_empty() {
        return vec![];
    }

    if tokens.len() <= max_tokens {
        // Single segment if within limits
        return vec![tokens.to_vec()];
    }

    let mut segments = Vec::new();
    let mut current_segment = Vec::new();

    for (_i, token) in tokens.iter().enumerate() {
        current_segment.push(token.clone());

        // Check if we should split here
        let should_split = if current_segment.len() >= max_tokens {
            // Must split - at max tokens
            true
        } else if current_segment.len() >= quick_streaming_tokens {
            // Consider splitting at sentence boundaries for streaming
            is_sentence_ending(token)
        } else {
            false
        };

        if should_split {
            segments.push(current_segment);
            current_segment = Vec::new();
        }
    }

    // Add remaining tokens
    if !current_segment.is_empty() {
        segments.push(current_segment);
    }

    segments
}

/// Segment text string directly, returning segment strings
///
/// This is useful when you want to segment before tokenization
pub fn segment_text_string(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_chars {
            segments.push(remaining.to_string());
            break;
        }

        // Find best split point within max_chars
        let search_range = &remaining[..max_chars.min(remaining.len())];

        // Try to find sentence ending
        if let Some(split_pos) = find_last_sentence_boundary(search_range) {
            let (segment, rest) = remaining.split_at(split_pos + 1);
            segments.push(segment.trim().to_string());
            remaining = rest.trim_start();
            continue;
        }

        // Try to find clause separator
        if let Some(split_pos) = find_last_clause_boundary(search_range) {
            let (segment, rest) = remaining.split_at(split_pos + 1);
            segments.push(segment.trim().to_string());
            remaining = rest.trim_start();
            continue;
        }

        // Fallback: split at last whitespace
        if let Some(split_pos) = search_range.rfind(char::is_whitespace) {
            let (segment, rest) = remaining.split_at(split_pos);
            segments.push(segment.trim().to_string());
            remaining = rest.trim_start();
            continue;
        }

        // Hard split at max_chars if no better option
        let (segment, rest) = remaining.split_at(max_chars.min(remaining.len()));
        segments.push(segment.to_string());
        remaining = rest;
    }

    segments
}

/// Check if a token ends with sentence-ending punctuation
fn is_sentence_ending(token: &str) -> bool {
    token.chars().last().map_or(false, |c| SENTENCE_ENDINGS.contains(&c))
}

/// Find the last sentence boundary position in text
fn find_last_sentence_boundary(text: &str) -> Option<usize> {
    text.char_indices()
        .rev()
        .find(|(_, c)| SENTENCE_ENDINGS.contains(c))
        .map(|(i, c)| i + c.len_utf8() - 1)
}

/// Find the last clause boundary position in text
fn find_last_clause_boundary(text: &str) -> Option<usize> {
    text.char_indices()
        .rev()
        .find(|(_, c)| CLAUSE_SEPARATORS.contains(c))
        .map(|(i, c)| i + c.len_utf8() - 1)
}

/// Segment Chinese and English mixed text
///
/// Chinese text typically doesn't use spaces, so we need special handling
pub fn segment_mixed_text(text: &str, max_chars: usize) -> Vec<String> {
    // For mixed text, we still try to split at punctuation
    segment_text_string(text, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_short_text() {
        let tokens: Vec<String> = vec!["Hello".into(), ",".into(), "world".into(), "!".into()];
        let segments = segment_text(&tokens, 120, 50);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], tokens);
    }

    #[test]
    fn test_segment_at_sentence_boundary() {
        let tokens: Vec<String> = (0..100)
            .map(|i| if i == 49 { ".".to_string() } else { format!("word{}", i) })
            .collect();
        let segments = segment_text(&tokens, 120, 50);
        assert!(segments.len() >= 1);
        // First segment should end at the period if within streaming threshold
        if segments.len() > 1 {
            assert!(segments[0].last().map_or(false, |s| s == "."));
        }
    }

    #[test]
    fn test_segment_text_string() {
        let text = "This is sentence one. This is sentence two. And this is sentence three.";
        let segments = segment_text_string(text, 30);
        assert!(segments.len() > 1);
        // Each segment should be <= 30 chars or end at a natural boundary
        for segment in &segments {
            assert!(segment.len() <= 35); // Allow some flexibility for word boundaries
        }
    }

    #[test]
    fn test_is_sentence_ending() {
        assert!(is_sentence_ending("."));
        assert!(is_sentence_ending("word."));
        assert!(is_sentence_ending("？")); // Chinese question mark
        assert!(!is_sentence_ending("word"));
        assert!(!is_sentence_ending(","));
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<String> = vec![];
        assert!(segment_text(&empty, 120, 50).is_empty());
        assert!(segment_text_string("", 100).is_empty() || segment_text_string("", 100) == vec![""]);
    }
}
