//! Audio processing modules
//!
//! - Audio file loading and decoding (WAV, MP3, FLAC, OGG)
//! - Sample rate conversion (16kHz, 22050Hz, 24kHz)
//! - Mel spectrogram computation (80 bands, librosa-compatible)
//! - Audio output/playback (file saving, real-time streaming)

mod loader;
mod resampler;
mod mel;
mod output;

pub use loader::AudioLoader;
pub use resampler::Resampler;
pub use mel::MelSpectrogram;
pub use output::{AudioOutput, StreamingPlayer};
