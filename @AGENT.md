# Build and Run Instructions

## Prerequisites

- Rust 1.75+ (with cargo)
- CUDA Toolkit 12.0+ (for GPU support)
- Git LFS (for model weights)

## Build Commands

```bash
# Development build (faster compilation)
cargo check --features cuda

# Release build (optimized)
cargo build --release --features cuda

# Run tests
cargo test --features cuda

# Run clippy lints
cargo clippy --features cuda -- -W clippy::all

# Format code
cargo fmt
```

## Running

```bash
# Basic inference
cargo run --release --features cuda -- \
  --text "Hello, world!" \
  --speaker voice.wav \
  --output output.wav

# With emotion
cargo run --release --features cuda -- \
  --text "I'm so happy!" \
  --speaker voice.wav \
  --emotion happy:0.8 \
  --output output.wav
```

## Model Weights

Download from HuggingFace and convert to safetensors:

```bash
# Download
huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir checkpoints

# Convert (Python)
python scripts/convert_weights.py
```

## Project Structure

```
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library exports
├── config/          # Configuration parsing
├── text/            # Text processing
├── audio/           # Audio I/O
├── models/          # ML models
│   ├── semantic/    # Wav2Vec-BERT, codec
│   ├── speaker/     # CAMPPlus
│   ├── gpt/         # GPT-2, Conformer
│   ├── s2mel/       # DiT, Flow Matching
│   └── vocoder/     # BigVGAN
└── inference/       # Pipeline, streaming
```

## Feature Flags

- `cuda` - Enable CUDA GPU acceleration (default)
- `metal` - Enable Metal GPU acceleration (macOS)

## Common Issues

1. **CUDA not found**: Set `CUDA_HOME` environment variable
2. **Out of memory**: Reduce batch size or use FP16
3. **Slow compilation**: Use `cargo check` instead of `cargo build`
