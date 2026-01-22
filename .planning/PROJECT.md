# IndexTTS2 Rust

## What This Is

A Rust implementation of IndexTTS2, a text-to-speech system that generates natural-sounding speech from text using a multi-stage neural pipeline. Originally implemented in Python, this port uses the Candle ML framework to bring TTS inference to Rust with the goal of running efficiently on GPU.

## Core Value

**Generate intelligible, natural-sounding speech from text input using a speaker reference audio for voice cloning.**

The system must produce audio that sounds like speech, not noise. This is the fundamental capability that makes everything else meaningful.

## Who It's For

- Developers wanting a Rust-native TTS solution
- Users who need GPU-accelerated voice synthesis
- Projects requiring voice cloning capabilities

## Current State

### What's Built (Validated)

The codebase is ~81% complete with a full pipeline that compiles and runs:

- ✓ **Text processing** — Tokenization, normalization, sentence segmentation
- ✓ **Audio I/O** — Loading, resampling, WAV output at 22050 Hz
- ✓ **Model architecture** — All neural network components implemented:
  - Wav2Vec-BERT 2.0 encoder (semantic extraction)
  - CAMPPlus speaker encoder (voice characteristics)
  - GPT-2 UnifiedVoice (autoregressive mel code generation)
  - Conformer encoder (audio encoding)
  - Perceiver resampler (cross-modal attention)
  - DiT flow matching (mel spectrogram refinement)
  - BigVGAN vocoder (mel → waveform)
- ✓ **CLI interface** — Inference command with text/speaker/output args
- ✓ **Test infrastructure** — 121 unit tests, 15 integration tests passing
- ✓ **BigVGAN weight loading** — Vocoder works correctly

### What's Broken

- ✗ **Weight loading for 4 components** — Wav2Vec-BERT, DiT, Conformer, Perceiver all use random weights due to tensor name mismatches between HuggingFace models and Rust implementation
- ✗ **Audio output quality** — Sounds like wind/noise because of random weights
- ✗ **CUDA support** — Features defined but untested, currently CPU-only

## Problem Statement

The TTS pipeline runs end-to-end but produces garbage audio because pre-trained weights aren't being loaded correctly. The model architectures are implemented, but the weight loading code expects different tensor names than what the downloaded HuggingFace models provide.

**Example mismatch:**
```
HuggingFace:  encoder.layers.0.self_attn.q_proj.weight
Rust expects: layers.0.attention.q_proj.weight
```

This causes silent fallback to random weight initialization, which produces noise instead of speech.

## Success Criteria

1. **Audio output is intelligible speech** — When you play the output WAV, you hear the input text spoken clearly
2. **Runs on GPU** — Inference uses RTX 3090 CUDA cores, not CPU
3. **All weights load without fallback** — No "using random weights" warnings

## Constraints

- **Must use existing architecture** — Not rewriting models, just fixing weight loading
- **Must use Candle framework** — Core ML library is already integrated
- **Target hardware** — RTX 3090 (CUDA)
- **Existing checkpoint format** — SafeTensors files already downloaded

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Weight name mapping approach | Simpler than restructuring Rust model code | — Pending |
| CUDA feature flags | Already defined in Cargo.toml | — Ready to enable |

## Out of Scope

- Python bindings (PyO3) — future work
- HTTP/WebSocket server — placeholder exists, not priority
- Model downloading from HuggingFace — manual download acceptable
- Performance optimization beyond GPU — future work
- Multi-speaker batch inference — single speaker focus

## Requirements

### Validated

- ✓ Text input tokenization and normalization — existing
- ✓ Audio file loading and resampling — existing
- ✓ WAV output at 22050 Hz — existing
- ✓ CLI interface for inference — existing
- ✓ BigVGAN vocoder integration — existing

### Active

- [ ] **WEIGHT-01**: Wav2Vec-BERT loads pre-trained weights correctly
- [ ] **WEIGHT-02**: DiT loads pre-trained weights correctly
- [ ] **WEIGHT-03**: Conformer loads pre-trained weights correctly
- [ ] **WEIGHT-04**: Perceiver loads pre-trained weights correctly
- [ ] **AUDIO-01**: Output audio is intelligible speech (not noise)
- [ ] **GPU-01**: Inference runs on CUDA device (RTX 3090)
- [ ] **GPU-02**: GPU inference produces same quality as CPU would with correct weights

### Out of Scope

- HTTP server API — not needed for core TTS functionality
- Model auto-download — manual checkpoint management acceptable
- Python bindings — future enhancement
- Batch processing multiple texts — single inference focus

## Links

- Codebase: `C:\AI\indextts2-rust`
- Weight fix plan: `@fix_weight_architecture.md`
- Status tracking: `CURRENT_STATUS.md`

---
*Last updated: 2026-01-23 after initialization*
