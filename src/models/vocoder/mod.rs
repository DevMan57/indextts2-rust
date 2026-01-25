//! Vocoder module for mel-to-waveform conversion
//!
//! Implements BigVGAN v2 vocoder for high-quality audio synthesis.

mod bigvgan;
mod weights;

pub use bigvgan::{BigVGAN, BigVGANConfig};
pub use weights::load_bigvgan_weights;
