//! Vocoder module for mel-to-waveform conversion
//!
//! Implements BigVGAN v2 vocoder for high-quality audio synthesis.

mod bigvgan;

pub use bigvgan::{BigVGAN, BigVGANConfig};
