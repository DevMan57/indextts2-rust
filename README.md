# IndexTTS2-Rust 🦀🔊

High-performance Rust implementation of [IndexTTS2](https://github.com/index-tts/index-tts) - Bilibili's Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System.

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Features

- 🚀 **Native Performance**: Compiled Rust with GPU acceleration via Candle/CUDA
- 🔒 **Memory Safe**: No memory leaks, no GC pauses
- 📦 **Single Binary**: Deploy without Python dependencies
- ⚡ **Low Latency**: Optimized for real-time voice synthesis
- 🎭 **Emotion Control**: 8 emotion categories with blending
- 🎤 **Zero-Shot Cloning**: Clone any voice with 3-10 seconds of audio

## Architecture

```
Text Input → BPE Tokenizer → GPT-2 (Conformer/Perceiver)
                                    ↓
Speaker Audio → Semantic Codec → Mel Codes
                                    ↓
                              S2Mel (DiT + CFM)
                                    ↓
                              BigVGAN Vocoder → Audio Output
```

## Requirements

- Rust 1.75+
- CUDA 12.0+ (for GPU acceleration)
- ~8GB VRAM for inference

## Installation

```bash
git clone https://github.com/DevMan57/indextts2-rust.git
cd indextts2-rust
cargo build --release --features cuda
```

## Usage

```bash
# Basic inference
indextts2 --text "Hello, world!" --speaker voice.wav --output output.wav

# With emotion
indextts2 --text "I'm so happy!" --speaker voice.wav --emotion happy:0.8 --output output.wav

# Streaming mode
indextts2 --text "Long text..." --speaker voice.wav --stream
```

## Model Weights

Download pretrained weights from HuggingFace:

```bash
huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir checkpoints
```

Convert PyTorch weights to safetensors:

```python
import torch
from safetensors.torch import save_file

for model in ["gpt", "dvae", "bigvgan_generator"]:
    state = torch.load(f"checkpoints/{model}.pth", map_location="cpu")
    save_file(state, f"checkpoints/{model}.safetensors")
```

## Development

This project uses [Claude Code](https://claude.ai/code) with:
- **Context7** for Rust crate documentation
- **Ralph Wiggum** for autonomous development loops
- **HuggingFace MCP** for ML research

See `CLAUDE.md` for development guidelines.

## Benchmarks

| Metric | Python (PyTorch) | Rust (Candle) |
|--------|------------------|---------------|
| First Token Latency | TBD | TBD |
| Tokens/Second | TBD | TBD |
| Memory Usage | TBD | TBD |

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [IndexTTS2](https://github.com/index-tts/index-tts) - Original Python implementation
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) - Neural vocoder

## Citation

```bibtex
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
