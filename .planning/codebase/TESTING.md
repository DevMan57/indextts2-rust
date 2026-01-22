# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Runner:**
- Rust's built-in test framework (`cargo test`)
- Criterion 0.5 for benchmarks
- No external test runner configuration

**Assertion Library:**
- Standard `assert!`, `assert_eq!`, `assert_ne!`
- `approx` 0.5 crate available for floating-point comparisons
- `insta` 1.41 for snapshot testing (available but not heavily used)
- `proptest` 1.5 for property-based testing (available but not heavily used)

**Run Commands:**
```bash
# Run all unit and integration tests
cargo test

# Run tests with output (see debug prints)
cargo test -- --nocapture

# Run specific test
cargo test test_mel_spectrogram

# Run tests in specific module
cargo test models::gpt::

# Run only integration tests
cargo test --test integration_test

# Run tests with verbose output
cargo test -- --show-output

# Run ignored tests (require model weights)
cargo test -- --ignored

# Run benchmarks
cargo bench
```

## Test File Organization

**Location:**
- **Unit tests:** Co-located with source in `#[cfg(test)] mod tests { ... }`
- **Integration tests:** `tests/integration_test.rs`
- **Benchmarks:** `benches/inference_bench.rs`
- **Examples:** `examples/*.rs` (runnable, not tests)

**Naming:**
- Test functions: `test_{what_is_tested}` (e.g., `test_mel_spectrogram_creation`)
- Test modules: `tests` (inline) or file name (integration)

**Structure:**
```
C:\AI\indextts2-rust\
├── src/
│   ├── models/
│   │   └── gpt/
│   │       └── generation.rs     # Contains #[cfg(test)] mod tests
│   └── audio/
│       └── mel.rs                # Contains #[cfg(test)] mod tests
├── tests/
│   └── integration_test.rs       # Full pipeline tests
└── benches/
    └── inference_bench.rs        # Performance benchmarks
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 1815);
        assert_eq!(config.stop_token, 8193);
    }

    #[test]
    fn test_sampler_temperature() {
        let device = Device::Cpu;
        let sampler = Sampler::new();

        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let scaled = sampler.apply_temperature(&logits, 0.5).unwrap();

        let values: Vec<f32> = scaled.to_vec1().unwrap();
        assert!((values[0] - 2.0).abs() < 0.001);
    }

    #[test]
    #[ignore = "Requires model weights in checkpoints/"]
    fn test_full_inference_with_weights() {
        // Heavy test skipped by default
    }
}
```

**Patterns:**
- **Setup:** Create Device, initialize structs directly in test
- **Teardown:** Rust's RAII handles cleanup automatically
- **Assertion:** Use `assert!`, `assert_eq!` with tolerance for floats

## Mocking

**Framework:** No dedicated mocking framework used

**Patterns:**
```rust
// Pattern 1: Create lightweight test instances
let device = Device::Cpu;
let vocoder = BigVGAN::new(&device).unwrap();
vocoder.initialize_random().unwrap();  // Random weights for testing

// Pattern 2: Use random tensors as input
let mel = Tensor::randn(0.0f32, 1.0, (1, 80, 100), &device).unwrap();
let audio = vocoder.forward(&mel).unwrap();

// Pattern 3: Create dummy data inline
let samples: Vec<f32> = (0..22050)
    .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
    .collect();
```

**What to Mock:**
- External I/O (files, network) - skip if files don't exist
- Heavy model weights - use `initialize_random()`

**What NOT to Mock:**
- Core tensor operations (use real Candle tensors)
- Math/signal processing (verify actual behavior)
- Configuration parsing (use literal YAML strings)

## Fixtures and Factories

**Test Data:**
```rust
// Inline fixture: YAML config string
let yaml = r#"
dataset:
    bpe_model: bpe.model
    sample_rate: 24000
gpt:
    model_dim: 1280
    layers: 24
"#;
let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();

// Generated test data: sine wave
let samples: Vec<f32> = (0..22050)
    .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
    .collect();

// Random tensor data
let x = Tensor::randn(0.0f32, 1.0, (1, 100, 80), &device).unwrap();
let prompt_x = Tensor::zeros((1, 100, 80), DType::F32, &device).unwrap();
```

**Location:**
- Test data created inline in tests
- No separate fixtures directory
- Full YAML configs embedded as string literals

## Coverage

**Requirements:** None enforced (no CI coverage gates)

**View Coverage:**
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --out Html

# With specific features
cargo tarpaulin --features "cuda" --out Html
```

**Current State:**
- 121 unit tests across 28 files with `#[cfg(test)]` modules
- 15 integration tests in `tests/integration_test.rs`
- 3 additional ignored tests requiring model weights
- Comprehensive coverage of:
  - Configuration parsing
  - Text normalization and tokenization
  - Audio processing (mel spectrogram, resampling)
  - Model initialization
  - Tensor operations and sampling
  - Validation utilities

**Key Gaps:**
- End-to-end inference (requires weights, marked `#[ignore]`)
- GPU-specific code paths
- Streaming functionality
- Error recovery paths

## Test Types

**Unit Tests:**
- Co-located in source files
- Test individual functions/methods
- Fast, no I/O dependencies
- Example: `src/text/normalizer.rs::tests`

**Integration Tests:**
- Located in `tests/integration_test.rs`
- Test component interactions
- May initialize multiple models
- Skip gracefully if resources unavailable

**E2E Tests:**
- Marked with `#[ignore]`
- Require model weights in `checkpoints/`
- Run with `cargo test -- --ignored`
- Test full pipeline from text to audio

## Common Patterns

**Async Testing:**
```rust
// Not used - all tests are synchronous
// The crate uses tokio but tests don't require async runtime
```

**Error Testing:**
```rust
#[test]
fn test_missing_file() {
    let result = TextTokenizer::load("nonexistent.json", normalizer);
    assert!(result.is_err());
}
```

**Floating Point Comparison:**
```rust
// Direct tolerance comparison
assert!((values[0] - 2.0).abs() < 0.001);
assert!((result.duration() - 1.0).abs() < 0.001);

// Relative tolerance for tensor validation
let tol = self.config.atol + self.config.rtol * expected.abs();
if diff > tol {
    // Record mismatch
}
```

**Shape Verification:**
```rust
let (batch, channels, samples) = audio.dims3().unwrap();
assert_eq!(batch, 1);
assert_eq!(channels, 1);
assert_eq!(samples, 100 * 256);  // 256x upsampling
```

**Output Range Validation:**
```rust
// Check audio values are in valid range
let max_val = result.audio.iter().cloned().fold(0.0f32, f32::max);
let min_val = result.audio.iter().cloned().fold(0.0f32, f32::min);
assert!(max_val <= 1.5, "Audio max value out of range: {}", max_val);
assert!(min_val >= -1.5, "Audio min value out of range: {}", min_val);
```

**Conditional Skip:**
```rust
#[test]
#[ignore = "Requires model weights in checkpoints/"]
fn test_full_inference_with_weights() {
    let config_path = Path::new("checkpoints/config.yaml");
    if !config_path.exists() {
        eprintln!("Skipping: checkpoints/config.yaml not found");
        return;
    }
    // ... actual test
}
```

## Benchmark Patterns

**Setup:**
```rust
// benches/inference_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_mel_spectrogram(c: &mut Criterion) {
    let mel = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0, None);

    let mut group = c.benchmark_group("mel_spectrogram");
    group.measurement_time(Duration::from_secs(10));

    for duration in [0.5, 1.0, 2.0, 5.0] {
        let samples: Vec<f32> = generate_sine_wave(duration);

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{:.1}s", duration)),
            &samples,
            |b, samples| {
                b.iter(|| mel.compute(black_box(samples)).unwrap())
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mel_spectrogram);
criterion_main!(benches);
```

**Slow Benchmarks:**
```rust
criterion_group!(
    name = slow_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_dit_forward, bench_vocoder_forward
);
```

## Validation Infrastructure

**Golden Data Comparison:**
```rust
// src/debug/validator.rs
pub struct Validator {
    golden_dir: PathBuf,
    config: ValidationConfig,
    results: Vec<ValidationResult>,
}

impl Validator {
    pub fn validate_tensor(&mut self, name: &str, actual: &Tensor, subdir: &str)
        -> Result<ValidationResult>
    {
        let golden_path = self.golden_dir.join(subdir).join(format!("{}.npy", name));
        let (golden_data, golden_shape) = load_npy_f32(&golden_path)?;

        // Compare shapes and values with tolerance
        // ...
    }
}
```

**Debug Binary:**
```bash
# Run validation against golden data
cargo run --release --bin debug_validate -- \
  --golden-dir debug/golden \
  --component all
```

---

*Testing analysis: 2026-01-23*
