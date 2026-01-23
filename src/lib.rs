//! # IndexTTS2 - Rust Implementation
//!
//! A high-performance Rust implementation of IndexTTS2, Bilibili's Industrial-Level
//! Controllable and Efficient Zero-Shot Text-To-Speech System.
//!
//! ## Features
//!
//! - Zero-shot voice cloning from a reference audio
//! - Emotion-controllable speech synthesis
//! - GPU-accelerated inference via Candle
//! - Streaming audio output support
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use indextts2::IndexTTS2;
//!
//! let tts = IndexTTS2::new("checkpoints/config.yaml")?;
//! let audio = tts.infer("Hello, world!", "voice.wav")?;
//! audio.save("output.wav")?;
//! ```

// Allow dead code for infrastructure that may be used in the future
#![allow(dead_code)]
// Require docs for public items, but not struct fields (too verbose)
#![warn(missing_docs)]
#![allow(rustdoc::missing_crate_level_docs)]

pub mod audio;
pub mod config;
pub mod debug;
pub mod inference;
pub mod models;
pub mod text;
pub mod utils;

// Re-exports for convenience
pub use config::ModelConfig;
pub use inference::IndexTTS2;
pub use debug::{WeightDiagnostics, ComponentReport};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate for output audio (22050 Hz)
pub const DEFAULT_SAMPLE_RATE: u32 = 22050;

/// Maximum text tokens per segment
pub const MAX_TEXT_TOKENS_PER_SEGMENT: usize = 120;
