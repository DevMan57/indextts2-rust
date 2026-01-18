# IndexTTS2 Rust Implementation - Fixes and Notes

This document tracks issues found during development and validation of the Rust implementation against the Python reference.

## Architecture Overview

The Rust implementation follows the Python IndexTTS2 architecture:

```
Text → Tokenizer → GPT (with Conformer/Perceiver conditioning) → Mel Codes
                                    ↑
Speaker Audio → CAMPPlus → Speaker Embedding
                                    ↓
Mel Codes → Length Regulator → DiT (Flow Matching) → Mel Spectrogram → BigVGAN → Audio
```

## Implementation Notes

### Phase 1: Foundation

#### P1.1 Config Parsing
- Used serde_yaml for YAML config parsing
- All fields marked with proper Option types for optional fields
- Default values provided via serde defaults

#### P1.2 Tokenizer
- Uses HuggingFace tokenizers crate
- BPE tokenization with proper special tokens
- Note: Token IDs must match Python exactly for reproducibility

#### P1.7 Mel Spectrogram
- librosa-compatible implementation
- Uses rustfft for FFT computation
- Hann window, pre-emphasis optional
- Parameters: n_fft=1024, hop_length=256, n_mels=80

### Phase 2: Core Encoders

#### P2.1 Wav2Vec-BERT 2.0
- Extracts layer 17 features (1024-dim)
- Uses mean and variance normalization
- Note: Statistics loaded from w2v_stat checkpoint

#### P2.2 Semantic Codec (VQ)
- Codebook size: 8192
- Codebook dimension: 8
- Hidden size: 1024
- Uses L2 distance for nearest neighbor search

#### P2.3 CAMPPlus Speaker Encoder
- D-TDNN (Densely-connected Time Delay Neural Network)
- Output: 192-dimensional speaker embedding
- Statistics pooling (mean + std concatenation)

### Phase 3: GPT Generation

#### P3.1 Conformer Encoder
- Macaron-style: FFN → Attention → Conv → FFN
- Swish activation, GLU gating
- Relative positional encoding

#### P3.2 Perceiver Resampler
- 32 learned latent queries (default)
- Cross-attention to encoder outputs
- Creates fixed-length conditioning

#### P3.3 KV Cache
- Per-layer key-value caching
- Causal attention masking
- Efficient incremental append for autoregressive generation

#### P3.4 Unified Voice (GPT-2)
- Model dim: 1280
- Layers: 24
- Attention heads: 20
- Max mel tokens: 1815
- Stop token: 8193

### Phase 4: Synthesis

#### P4.1 Length Regulator
- Duration predictor with Conv1d blocks
- Expands mel codes to target spectrogram length
- Softplus activation for positive durations

#### P4.2 DiT (Diffusion Transformer)
- Hidden dim: 512
- Layers: 13
- Attention heads: 8
- AdaLN conditioning
- UViT skip connections

#### P4.3 Flow Matching
- CFM (Conditional Flow Matching)
- 25 inference steps
- CFG rate: 0.7
- Euler and Heun ODE solvers

#### P4.4 BigVGAN Vocoder
- 256x upsampling (4×4×2×2×2×2)
- Snake activation (anti-aliased)
- MRF (Multi-Resolution Fusion) blocks
- Output: 22050 Hz audio

## Known Issues and Workarounds

### 1. Candle Conv1d Transpose Limitation
**Issue**: Candle doesn't have native `conv_transpose1d` operation.
**Workaround**: Implemented via upsampling + regular conv1d.
**Impact**: May affect numerical precision in vocoder.

### 2. Weight Loading
**Issue**: PyTorch .pth files not directly loadable.
**Workaround**: Initialize with random weights for structure testing.
**TODO**: Implement SafeTensors or custom loader.

### 3. Memory Usage
**Issue**: Large model sizes may exceed memory on CPU.
**Workaround**: Components can be initialized lazily.
**Recommendation**: Use GPU for production inference.

## Validation Results

### Text Processing
- [ ] Tokenizer matches Python output exactly
- [ ] Special tokens (BOS/EOS) handled correctly
- [ ] Padding/truncation matches

### Audio Processing
- [ ] Mel spectrogram within 1e-4 tolerance
- [ ] FFT window matches librosa
- [ ] Filterbank coefficients match

### Encoders
- [ ] CAMPPlus output within 1e-3
- [ ] Wav2Vec-BERT features within 1e-3
- [ ] VQ codes match exactly

### GPT
- [ ] Layer outputs within tolerance
- [ ] KV-cache accumulation correct
- [ ] Mel codes match Python

### Synthesis
- [ ] DiT outputs within tolerance
- [ ] Flow matching trajectory matches
- [ ] Vocoder audio quality acceptable

### End-to-End
- [ ] Waveform correlation > 0.95
- [ ] Mel MSE < 0.01
- [ ] Subjective audio quality acceptable

## Performance Metrics

| Component | Python (CPU) | Rust (CPU) | Rust (GPU) |
|-----------|--------------|------------|------------|
| Tokenizer | - | - | - |
| Mel Spec  | - | - | N/A |
| GPT       | - | - | - |
| DiT       | - | - | - |
| Vocoder   | - | - | - |
| **Total** | - | - | - |

*Note: Metrics to be filled after benchmarking*

## Changelog

### v0.1.0 (Initial Implementation)
- Complete pipeline implementation
- All neural network components
- CLI interface
- Integration tests
- Debug validation tools

## References

- [IndexTTS2 Paper](https://arxiv.org/abs/...)
- [Candle Documentation](https://huggingface.github.io/candle/)
- [BigVGAN Paper](https://arxiv.org/abs/2206.04658)
- [Conformer Paper](https://arxiv.org/abs/2005.08100)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
