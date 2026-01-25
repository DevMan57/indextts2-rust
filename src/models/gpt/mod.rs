//! GPT-2 based autoregressive generation
//!
//! Implements the UnifiedVoice model for text-to-speech:
//! - Conformer encoder for audio conditioning
//! - Perceiver resampler for cross-attention
//! - GPT-2 decoder with KV-cache
//! - Autoregressive mel code generation

mod conformer;
mod perceiver;
mod kv_cache;
mod unified_voice;
mod generation;
mod weights;

pub use conformer::ConformerEncoder;
pub use perceiver::PerceiverResampler;
pub use kv_cache::KVCache;
pub use unified_voice::UnifiedVoice;
pub use generation::{GenerationConfig, generate, generate_with_hidden};
pub use weights::{load_safetensors, Gpt2LayerWeights};
