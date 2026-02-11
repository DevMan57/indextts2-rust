//! Emotion processing for TTS
//!
//! Provides emotion-conditioned speech synthesis through:
//! - Emotion matrix for 8 emotion categories
//! - Emotion blending with configurable alpha

mod matrix;
mod controls;

pub use matrix::EmotionMatrix;
pub use controls::EmotionControls;
