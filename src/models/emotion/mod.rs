//! Emotion processing for TTS
//!
//! Provides emotion-conditioned speech synthesis through:
//! - Emotion matrix for 8 emotion categories
//! - Emotion blending with configurable alpha

mod matrix;

pub use matrix::EmotionMatrix;
