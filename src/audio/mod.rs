//! Audio processing modules
//!
//! - Audio file loading and decoding
//! - Sample rate conversion  
//! - Mel spectrogram computation
//! - Audio output/playback

mod loader;
mod resampler;
mod mel;
mod output;

pub use loader::AudioLoader;
pub use resampler::Resampler;
pub use mel::MelSpectrogram;
pub use output::AudioOutput;
