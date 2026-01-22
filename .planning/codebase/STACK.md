# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Rust 2021 Edition (rustc 1.92.0)

**Secondary:**
- None

## Runtime

**Environment:**
- Native binary (no runtime required)
- CUDA optional for GPU acceleration

**Package Manager:**
- Cargo
- Lockfile: `Cargo.lock` (present)

## Frameworks

**Core ML Framework:**
- Candle 0.8 (HuggingFace) - GPU-accelerated tensor operations
  - `candle-core` - Core tensor operations
  - `candle-nn` - Neural network layers
  - `candle-transformers` - Transformer architectures

**Testing:**
- Built-in Rust test framework
- Criterion 0.5 - Benchmarking
- Proptest 1.5 - Property-based testing
- Approx 0.5 - Float comparisons
- Insta 1.41 - Snapshot testing

**Build/Dev:**
- Cargo (build system)
- Release profile: LTO enabled, codegen-units=1, opt-level=3

## Key Dependencies

**Critical ML:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `candle-core` | 0.8 | Tensor operations (CPU/GPU) |
| `candle-nn` | 0.8 | Neural network layers (Linear, LayerNorm, etc.) |
| `candle-transformers` | 0.8 | Transformer architectures |
| `tokenizers` | 0.20 | HuggingFace tokenizers (BPE/Unigram) |

**Audio Processing:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `symphonia` | 0.5 | Multi-format audio decoding (MP3, FLAC, OGG) |
| `hound` | 3.5 | WAV file I/O |
| `rubato` | 0.16 | Sample rate conversion (resampling) |
| `rustfft` | 6.2 | FFT for mel spectrogram computation |
| `cpal` | 0.15 | Cross-platform audio I/O |
| `rodio` | 0.19 | Audio playback |

**Infrastructure:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `tokio` | 1.42 | Async runtime (full features) |
| `rayon` | 1.10 | Data parallelism |
| `crossbeam` | 0.8 | Concurrent data structures |

**Serialization:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `serde` | 1.0 | Serialization framework |
| `serde_yaml` | 0.9 | YAML config parsing |
| `serde_json` | 1.0 | JSON parsing |
| `toml` | 0.8 | TOML parsing |

**CLI/UX:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `clap` | 4.5 | Command-line argument parsing |
| `indicatif` | 0.17 | Progress bars |
| `console` | 0.15 | Terminal colors/formatting |
| `tracing` | 0.1 | Structured logging |
| `tracing-subscriber` | 0.3 | Log output formatting |

**Error Handling:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `anyhow` | 1.0 | Application error handling |
| `thiserror` | 2.0 | Library error types |

**Numerics:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `ndarray` | 0.16 | N-dimensional arrays (CPU operations) |
| `num-traits` | 0.2 | Numeric traits |
| `rand` | 0.8 | Random number generation |

**Network:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `reqwest` | 0.12 | HTTP client (model downloads) |
| `hf-hub` | 0.3 | HuggingFace Hub API |

**File System:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| `walkdir` | 2.5 | Directory traversal |
| `tempfile` | 3.14 | Temporary files |
| `memmap2` | 0.9 | Memory-mapped files (large model loading) |

## Configuration

**Environment:**
- No environment variables required for basic operation
- GPU selection via `--cpu` flag (defaults to CUDA if available)

**Build Features:**
```toml
[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
mkl = ["candle-core/mkl", "candle-nn/mkl"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]
```

**Build Commands:**
```bash
# CPU-only build
cargo build --release --no-default-features

# CUDA build (default)
cargo build --release

# Apple Silicon (Metal)
cargo build --release --features metal --no-default-features

# Intel MKL
cargo build --release --features mkl --no-default-features
```

**Config Files:**
- `checkpoints/config.yaml` - Model configuration (dimensions, layers, etc.)
- `Cargo.toml` - Build configuration

## Platform Requirements

**Development:**
- Rust 1.92+ (2021 edition)
- Cargo
- For CUDA: CUDA toolkit + cuDNN
- For Metal: macOS with Apple Silicon

**Production:**
- Native binary (no runtime dependencies)
- Model weights (~2GB total safetensors files)
- For CUDA: CUDA runtime libraries

**Supported Platforms:**
- Windows (x86_64)
- Linux (x86_64)
- macOS (x86_64, ARM64/Apple Silicon)

## Build Profiles

**Release Profile:**
```toml
[profile.release]
lto = true           # Link-time optimization
codegen-units = 1    # Single codegen unit for better optimization
opt-level = 3        # Maximum optimization
```

**Dev Profile:**
```toml
[profile.dev]
opt-level = 1        # Slight optimization for faster dev builds
```

**Bench Profile:**
```toml
[profile.bench]
inherits = "release" # Use release settings for benchmarks
```

---

*Stack analysis: 2026-01-23*
