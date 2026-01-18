//! Inference module for text-to-speech synthesis
//!
//! This module provides the main entry point for TTS:
//! - IndexTTS2: Main inference pipeline
//! - StreamingSynthesizer: Real-time audio streaming
//! - InferenceConfig: Runtime configuration

mod pipeline;
mod streaming;

pub use pipeline::{IndexTTS2, InferenceConfig, InferenceResult};
pub use streaming::{StreamingSynthesizer, StreamingConfig, AudioChunk};
