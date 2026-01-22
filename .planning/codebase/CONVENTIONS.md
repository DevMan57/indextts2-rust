# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Conventions

**Files:**
- Use `snake_case.rs` for all Rust source files
- Module entry points: `mod.rs` in directories
- Binaries: `src/bin/{name}.rs` (e.g., `src/bin/debug_validate.rs`)
- Examples: `examples/{name}.rs` (e.g., `examples/basic_inference.rs`)
- Benchmarks: `benches/{name}.rs` (e.g., `benches/inference_bench.rs`)
- Tests: `tests/{name}.rs` for integration tests

**Modules:**
- Use `snake_case` for module names
- Mirror directory structure: `src/models/gpt/` -> `models::gpt`
- Group related functionality: `src/models/gpt/mod.rs` re-exports public types

**Types:**
- Use `PascalCase` for structs, enums, traits
- Configuration structs: `{Name}Config` (e.g., `InferenceConfig`, `BigVGANConfig`)
- Result types: `{Name}Result` (e.g., `InferenceResult`, `ValidationResult`)
- Model components: descriptive names (e.g., `UnifiedVoice`, `DiffusionTransformer`)

**Functions:**
- Use `snake_case` for all functions and methods
- Constructors: `new()` or `new_default()`, `with_config()` for parameterized
- Loaders: `load()`, `load_weights()`
- Initializers: `initialize_random()`
- Forward passes: `forward()`, `forward_one()`, `forward_one_with_hidden()`
- Getters: `sample_rate()`, `vocab_size()`, `device()`
- Predicates: `is_initialized()`, `all_passed()`

**Variables:**
- Use `snake_case` for all variables
- Tensor variables: descriptive names (`mel_spec`, `hidden_states`, `audio_tensor`)
- Loop indices: `i`, `j`, `step`, `idx`
- Temporary: `x`, `out`, `result`

**Constants:**
- Use `SCREAMING_SNAKE_CASE`
- Place at module level: `pub const VERSION: &str = ...`
- Domain-specific: `MEL_MEAN`, `LOG_FLOOR`, `DEFAULT_SAMPLE_RATE`

## Error Handling

**Pattern:** Use `anyhow::Result` with context chains

**Error types:**
- `anyhow` for application-level errors (main crate)
- `thiserror` available for custom error types (not heavily used)
- Error propagation via `?` operator

**Common patterns:**

```rust
// With context for file operations
use anyhow::{Context, Result};

pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;

    serde_yaml::from_str(&content)
        .with_context(|| "Failed to parse config YAML")
}
```
- **Used in:** `src/config/model_config.rs`, `src/inference/pipeline.rs`

```rust
// Bail for validation errors
if batch_size != 1 {
    anyhow::bail!("Generation currently only supports batch_size=1");
}
```
- **Used in:** `src/models/gpt/generation.rs`

```rust
// Missing tensor errors
let weight = weights
    .get(&format!("{}.weight", prefix))
    .cloned()
    .ok_or_else(|| anyhow::anyhow!("Missing weight: {}.weight", prefix))?;
```
- **Used in:** `src/models/vocoder/bigvgan.rs`

## Common Patterns

### Builder Pattern for Configuration

```rust
#[derive(Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    // ...
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            // ...
        }
    }
}

impl IndexTTS2 {
    pub fn new<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        Self::with_config(config_path, InferenceConfig::default())
    }

    pub fn with_config<P: AsRef<Path>>(
        config_path: P,
        inference_config: InferenceConfig,
    ) -> Result<Self> {
        // ...
    }
}
```
- **Used in:** `src/inference/pipeline.rs`, `src/debug/validator.rs`
- **Purpose:** Allow default construction with override capability

### Weight Loading Pattern

```rust
pub struct BigVGAN {
    device: Device,
    config: BigVGANConfig,
    conv_pre: Option<Conv1d>,  // Optional until loaded
    weights_loaded: bool,
}

impl BigVGAN {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config: BigVGANConfig::default(),
            conv_pre: None,
            weights_loaded: false,
        })
    }

    pub fn initialize_random(&mut self) -> Result<()> {
        self.conv_pre = Some(Conv1d::new_random(/*...*/)?);
        self.weights_loaded = true;
        Ok(())
    }

    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        if !path.as_ref().exists() {
            return self.initialize_random();
        }
        let weights = load_bigvgan_weights(path, &self.device)?;
        self.conv_pre = Some(Conv1d::from_weights(&weights, "conv_pre", /*...*/)?);
        self.weights_loaded = true;
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }
}
```
- **Used in:** `src/models/vocoder/bigvgan.rs`, `src/models/gpt/unified_voice.rs`
- **Purpose:** Support both random initialization and checkpoint loading

### Tensor Forward Pass Pattern

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    if !self.weights_loaded {
        // Return placeholder/zeros when not initialized
        let (batch, _channels, time) = x.dims3()?;
        return Tensor::zeros((batch, 1, time * 256), DType::F32, &self.device)
            .map_err(Into::into);
    }

    // Actual forward pass
    let mut out = self.conv_pre.as_ref().unwrap().forward(x)?;
    // ...
    out.tanh().map_err(Into::into)
}
```
- **Used in:** Most model files in `src/models/`
- **Purpose:** Graceful degradation when weights not loaded

### Debug Logging Pattern

```rust
// Extensive debug output during development
eprintln!("DEBUG: mel_codes count={}, min={}, max={}",
    mel_codes.len(),
    mel_codes.iter().min().unwrap_or(&0),
    mel_codes.iter().max().unwrap_or(&0));

// Tensor statistics for debugging
let hs_mean: f32 = hidden_states.mean_all()?.to_scalar()?;
let hs_var: f32 = hidden_states.var(D::Minus1)?.mean_all()?.to_scalar()?;
eprintln!("DEBUG: hidden_states shape={:?}, mean={:.4}, var={:.4}",
    hidden_states.shape(), hs_mean, hs_var);
```
- **Used in:** `src/inference/pipeline.rs`, `src/models/gpt/generation.rs`, `src/models/vocoder/bigvgan.rs`
- **Purpose:** Track tensor flow and values during inference debugging

## Documentation Style

**Module-level docs:**
```rust
//! Mel spectrogram computation
//!
//! Computes mel-frequency spectrograms from audio samples.
//! Compatible with librosa-style mel spectrograms used in TTS systems.
```

**Struct/function docs:**
```rust
/// Mel spectrogram computer
///
/// Configured to match IndexTTS2's mel spectrogram parameters:
/// - 80 mel bands
/// - 1024 FFT size
/// - 256 hop length
/// - 22050 Hz sample rate
pub struct MelSpectrogram { /* ... */ }

/// Create a new mel spectrogram computer
///
/// # Arguments
/// * `n_fft` - FFT size (typically 1024)
/// * `hop_length` - Hop between frames (typically 256)
///
/// # Returns
/// Mel spectrogram as `Vec<Vec<f32>>` where shape is `[n_frames, n_mels]`
pub fn compute(&self, audio: &[f32]) -> Result<Vec<Vec<f32>>> { /* ... */ }
```

**Lib-level docs (lib.rs):**
```rust
//! # IndexTTS2 - Rust Implementation
//!
//! A high-performance Rust implementation of IndexTTS2.
//!
//! ## Features
//!
//! - Zero-shot voice cloning
//! - Emotion-controllable synthesis
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use indextts2::IndexTTS2;
//! let tts = IndexTTS2::new("checkpoints/config.yaml")?;
//! ```
```

**Linting configuration:**
```rust
// In lib.rs
#![allow(dead_code)]           // Allow unused during development
#![warn(missing_docs)]         // Warn on missing docs for public items
#![allow(rustdoc::missing_crate_level_docs)]
```

## Import Organization

**Order:**
1. Standard library (`std::*`)
2. External crates (`anyhow`, `candle_core`, etc.)
3. Internal crate modules (`crate::*`, `super::*`)

**Example:**
```rust
use anyhow::{Context, Result};
use candle_core::{Device, DType, Tensor};
use std::path::Path;

use crate::config::ModelConfig;
use crate::models::gpt::UnifiedVoice;
use super::weights::load_bigvgan_weights;
```

**Path Aliases:**
- None used (standard `crate::` and `super::` paths)

## Function Design

**Size:**
- Most functions are 20-50 lines
- Complex forward passes can be 100+ lines with debug output
- Extract helper functions for repeated logic

**Parameters:**
- Use `&self` for read operations, `&mut self` for state changes
- Generic paths: `<P: AsRef<Path>>`
- Tensor references: `&Tensor`
- Optional conditioning: `Option<&Tensor>`

**Return Values:**
- Use `Result<T>` for fallible operations
- Return owned types (`Vec<f32>`, `Tensor`) rather than references
- Multi-value returns via tuple: `(Vec<u32>, Tensor)`

## Module Design

**Exports (mod.rs):**
```rust
// src/models/gpt/mod.rs
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
```

**Barrel Files:**
- Use `mod.rs` to re-export public types from submodules
- Keep internal implementation private, expose clean API

---

*Convention analysis: 2026-01-23*
