# IndexTTS2 Rust - Phase 9: Testing & Polish

## Overview

Fix integration tests, add comprehensive unit tests, and polish the implementation
for production use. Ensure all components work end-to-end with actual weights.

---

## Phase 9 Tasks

### Fix Test Compilation
- [ ] **P9.1** Fix `tests/integration_test.rs` compilation errors
  - Fix `MelSpectrogram::new` signature (7 args, no Result)
  - Fix `SemanticEncoder::new` → use `load()` or add `new()` method
  - Fix `TextNormalizer::new` signature (needs bool arg)
  - Run `cargo test` to verify

- [ ] **P9.2** Fix remaining test issues
  - Add proper mock data for tests without weights
  - Handle `SemanticEncoder` path requirements
  - Update GPT forward() call signatures

### Unit Tests
- [ ] **P9.3** Add text processing tests
  - Tokenizer encode/decode round-trip
  - Normalizer number expansion (123 → "one hundred twenty three")
  - Segmenter sentence boundary detection

- [ ] **P9.4** Add audio processing tests
  - MelSpectrogram shape verification
  - Resampler quality check
  - WAV file save/load round-trip

- [ ] **P9.5** Add model tests (with mock weights)
  - GPT forward pass shape check
  - DiT diffusion step verification
  - Flow matching ODE solver accuracy

### End-to-End Test
- [ ] **P9.6** Create full inference test
  - Load actual weights from `checkpoints/`
  - Process sample text: "Hello world, this is a test."
  - Generate audio and save to file
  - Verify audio duration and sample rate

- [ ] **P9.7** Benchmark inference performance
  - Time each component: tokenizer, encoders, GPT, synthesis, vocoder
  - Compare CPU vs GPU performance
  - Log memory usage

### Documentation
- [ ] **P9.8** Update README.md
  - Installation instructions (Rust, CUDA)
  - Weight download instructions
  - Usage examples
  - API documentation links

- [ ] **P9.9** Add API documentation
  - Document public structs and functions
  - Add examples to doc comments
  - Run `cargo doc --open` to verify

### Polish
- [ ] **P9.10** Clean up warnings
  - Fix unused variable warnings
  - Fix dead code warnings
  - Add `#[allow(dead_code)]` where appropriate for future use

- [ ] **P9.11** Error handling improvements
  - Add context to all errors with `anyhow::Context`
  - Improve error messages for user-facing code
  - Handle missing files gracefully

- [ ] **P9.12** Performance optimizations
  - Profile hot paths with `cargo flamegraph`
  - Optimize tensor operations
  - Add CUDA stream parallelism where beneficial

---

## Test Categories

### Unit Tests (`src/**/tests`)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_new() { ... }

    #[test]
    fn test_forward_shape() { ... }
}
```

### Integration Tests (`tests/`)
```rust
// tests/integration_test.rs
#[test]
fn test_full_pipeline() {
    // Load weights
    // Process text
    // Generate audio
    // Verify output
}
```

### Benchmarks (`benches/`)
```rust
// benches/inference_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_inference(c: &mut Criterion) {
    c.bench_function("full_inference", |b| {
        b.iter(|| { /* inference code */ })
    });
}
```

---

## Ralph Loop Command

```bash
/ralph-loop "Implement Phase 9 from @fix_plan_phase9.md. Fix test compilation, add comprehensive tests, and polish for production. Run all tests with cargo test." --max-iterations 40 --completion-promise "PHASE9_COMPLETE"
```

---

## Completion Tracking

**Phase 9:** 0/12 complete
- [ ] P9.1 - Fix integration test compilation
- [ ] P9.2 - Fix remaining test issues
- [ ] P9.3 - Text processing tests
- [ ] P9.4 - Audio processing tests
- [ ] P9.5 - Model tests (mock weights)
- [ ] P9.6 - Full inference test
- [ ] P9.7 - Benchmark performance
- [ ] P9.8 - Update README.md
- [ ] P9.9 - API documentation
- [ ] P9.10 - Clean up warnings
- [ ] P9.11 - Error handling
- [ ] P9.12 - Performance optimizations

---

## Final Checklist

Before marking project complete:
- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` shows no errors
- [ ] `cargo doc` generates documentation
- [ ] End-to-end inference produces valid audio
- [ ] README has clear usage instructions
- [ ] Weights download and load correctly
