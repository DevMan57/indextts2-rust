# IndexTTS2 Rust Technical Specifications

## Overview

Rust implementation of IndexTTS2 - Bilibili's zero-shot TTS system.

## Target Metrics

| Metric | Python (PyTorch) | Rust (Candle) Target |
|--------|------------------|----------------------|
| First Token Latency | ~500ms | <200ms |
| Real-time Factor | ~0.8x | >2x |
| Memory (VRAM) | ~6GB | <4GB |
| Binary Size | N/A (Python) | <50MB |

## Model Architecture

### GPT-2 (UnifiedVoice)
- **Dimensions**: 1280 hidden, 24 layers, 20 heads
- **Vocabulary**: 12000 text tokens + 8194 mel codes
- **Stop Token**: 8193
- **Max Sequence**: 600 text + 1815 mel tokens

### S2Mel (Diffusion Transformer)
- **DiT**: 512 hidden, 13 layers, 8 heads
- **Flow Matching**: 25 steps, CFM sampler
- **CFG Rate**: 0.7

### BigVGAN Vocoder
- **Model**: nvidia/bigvgan_v2_22khz_80band_256x
- **Sample Rate**: 22050 Hz
- **Mel Bands**: 80
- **Hop Size**: 256

## Dependencies

### Core ML
- `candle-core` 0.8 - Tensor operations
- `candle-nn` 0.8 - Neural network layers
- `candle-transformers` 0.8 - Transformer models

### Audio
- `symphonia` 0.5 - Audio decoding
- `rubato` 0.16 - Resampling
- `rustfft` 6.2 - FFT for mel spectrograms
- `cpal` 0.15 - Audio playback

### Tokenization
- `tokenizers` 0.20 - HuggingFace BPE

## File Format Support

### Input
- WAV (16-bit PCM, float32)
- MP3
- FLAC
- OGG

### Output
- WAV (22050 Hz, 16-bit PCM)
- Real-time streaming via cpal

## API Design

```rust
pub struct IndexTTS {
    config: Config,
    tokenizer: TextTokenizer,
    semantic: SemanticEncoder,
    speaker: SpeakerEncoder,
    gpt: UnifiedVoice,
    s2mel: S2MelFlow,
    vocoder: BigVGAN,
}

impl IndexTTS {
    pub fn new(checkpoint_dir: &Path) -> Result<Self>;
    pub fn synthesize(&self, text: &str, speaker_audio: &[f32]) -> Result<Vec<f32>>;
    pub fn synthesize_stream(&self, text: &str, speaker_audio: &[f32]) -> impl Stream<Item=Vec<f32>>;
}
```

## Error Handling

```rust
#[derive(thiserror::Error, Debug)]
pub enum IndexTTSError {
    #[error("Config error: {0}")]
    Config(#[from] serde_yaml::Error),
    
    #[error("Model error: {0}")]
    Model(#[from] candle_core::Error),
    
    #[error("Audio error: {0}")]
    Audio(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}
```

## Performance Requirements

1. **GPU Memory**: Fit in 8GB VRAM
2. **Latency**: <200ms first token
3. **Throughput**: >2x real-time on RTX 3080+
4. **CPU Fallback**: Functional but slower
