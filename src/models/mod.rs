//! Neural network models for TTS
//!
//! This module contains all the neural network components:
//! - Semantic encoder (Wav2Vec-BERT)
//! - Speaker encoder (CAMPPlus)
//! - Emotion processing
//! - GPT model for autoregressive generation
//! - S2Mel (Semantic-to-Mel) diffusion model
//! - BigVGAN vocoder

pub mod semantic;
pub mod speaker;
pub mod emotion;
pub mod gpt;
pub mod s2mel;
pub mod vocoder;

// Re-exports
pub use gpt::UnifiedVoice;
pub use vocoder::BigVGAN;
