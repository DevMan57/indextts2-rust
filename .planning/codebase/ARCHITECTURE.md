# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Pipeline-based Neural TTS Architecture

**Key Characteristics:**
- Multi-stage inference pipeline with discrete model components
- Autoregressive GPT-2 decoder with KV-cache for mel code generation
- Flow matching diffusion for continuous mel spectrogram synthesis
- Candle-based tensor operations (HuggingFace ML framework for Rust)

## High-Level Pipeline

```
┌─────────────┐     ┌─────────────────┐     ┌────────────────┐
│    Text     │────▶│  TextNormalizer │────▶│  TextTokenizer │
│   Input     │     │                 │     │   (BPE)        │
└─────────────┘     └─────────────────┘     └───────┬────────┘
                                                    │
┌─────────────┐     ┌─────────────────┐             │
│   Speaker   │────▶│ MelSpectrogram  │────┐        │
│   Audio     │     │   Extractor     │    │        │
└─────────────┘     └─────────────────┘    │        │
                           │               │        │
                           ▼               ▼        ▼
                    ┌──────────────────────────────────────┐
                    │       IndexTTS2 Pipeline             │
                    │  ┌─────────────────────────────────┐ │
                    │  │ CAMPPlus (Speaker Encoder)      │ │
                    │  │ → 192-dim speaker embedding     │ │
                    │  └─────────────────────────────────┘ │
                    │  ┌─────────────────────────────────┐ │
                    │  │ SemanticEncoder (Wav2Vec-BERT)  │ │
                    │  │ → Semantic features             │ │
                    │  └─────────────────────────────────┘ │
                    │  ┌─────────────────────────────────┐ │
                    │  │ UnifiedVoice GPT                │ │
                    │  │ Conformer→Perceiver→GPT-2       │ │
                    │  │ → Mel codes + hidden states     │ │
                    │  └─────────────────────────────────┘ │
                    │  ┌─────────────────────────────────┐ │
                    │  │ LengthRegulator                 │ │
                    │  │ → Duration-aligned features     │ │
                    │  └─────────────────────────────────┘ │
                    │  ┌─────────────────────────────────┐ │
                    │  │ FlowMatching + DiT              │ │
                    │  │ → Mel spectrogram (80-band)     │ │
                    │  └─────────────────────────────────┘ │
                    │  ┌─────────────────────────────────┐ │
                    │  │ BigVGAN Vocoder                 │ │
                    │  │ → Audio waveform (22050 Hz)     │ │
                    │  └─────────────────────────────────┘ │
                    └──────────────────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  WAV Output    │
                              │  (22050 Hz)    │
                              └────────────────┘
```

## Layers

**Text Processing Layer:**
- Purpose: Normalize and tokenize input text
- Location: `src/text/`
- Contains: `TextNormalizer`, `TextTokenizer`, sentence segmentation
- Depends on: Standard Rust libraries, tokenizers crate
- Used by: `IndexTTS2` pipeline

**Audio Processing Layer:**
- Purpose: Load audio, extract mel spectrograms, output audio
- Location: `src/audio/`
- Contains: `AudioLoader`, `Resampler`, `MelSpectrogram`, `AudioOutput`
- Depends on: hound (WAV), rubato (resampling), mel computation
- Used by: `IndexTTS2` pipeline, speaker/emotion processing

**Encoder Layer:**
- Purpose: Extract speaker identity and semantic features from reference audio
- Location: `src/models/semantic/`, `src/models/speaker/`
- Contains: `SemanticEncoder` (Wav2Vec-BERT), `SemanticCodec`, `CAMPPlus`
- Depends on: Candle tensors, audio layer
- Used by: `IndexTTS2` conditioning

**GPT Generation Layer:**
- Purpose: Autoregressively generate mel codes from text + conditioning
- Location: `src/models/gpt/`
- Contains: `UnifiedVoice`, `ConformerEncoder`, `PerceiverResampler`, `KVCache`
- Depends on: Candle NN layers, encoder outputs
- Used by: `IndexTTS2` for mel code generation

**Synthesis Layer:**
- Purpose: Convert mel codes to continuous mel spectrograms via diffusion
- Location: `src/models/s2mel/`
- Contains: `LengthRegulator`, `DiffusionTransformer`, `FlowMatching`
- Depends on: GPT hidden states, speaker embeddings
- Used by: `IndexTTS2` mel spectrogram generation

**Vocoder Layer:**
- Purpose: Convert mel spectrograms to audio waveforms
- Location: `src/models/vocoder/`
- Contains: `BigVGAN` with anti-aliased upsampling
- Depends on: Mel spectrograms from synthesis layer
- Used by: `IndexTTS2` final audio output

## Data Flow

**Primary Inference Flow:**

1. Text Input → `TextNormalizer::normalize()` → normalized text
2. Normalized text → `TextTokenizer::encode()` → token IDs tensor
3. Speaker audio → `AudioLoader::load()` → audio samples
4. Audio samples → `MelSpectrogram::compute()` → mel features
5. Mel features → `CAMPPlus::encode()` → speaker embedding (192-dim)
6. Audio samples → `SemanticEncoder::encode()` → semantic features
7. Mel features → `ConformerEncoder::forward()` → encoded conditioning
8. Encoded → `PerceiverResampler::forward()` → resampled conditioning
9. Text tokens + conditioning → `UnifiedVoice` GPT → mel codes + hidden states
10. Hidden states → `LengthRegulator::project_gpt_embeddings()` → latent projection
11. Mel codes → `UnifiedVoice::embed_mel_codes()` → code embeddings
12. S_infer = code_embeddings + latent_projection (critical computation)
13. S_infer → `LengthRegulator::forward()` → content features (512-dim)
14. Noise + content + speaker → `FlowMatching::sample()` → mel spectrogram
15. Mel spectrogram → `BigVGAN::forward()` → audio waveform

**State Management:**
- KV-cache in `UnifiedVoice` for efficient autoregressive generation
- Layer caches stored per decoder layer in `KVCache`
- No global state; all state passed through function parameters

## Key Abstractions

**IndexTTS2:**
- Purpose: Main inference orchestrator
- Location: `src/inference/pipeline.rs`
- Holds all model components and coordinates inference flow
- Public API: `infer()`, `infer_with_emotion()`, `load_weights()`

**UnifiedVoice:**
- Purpose: GPT-2 based autoregressive mel code generator
- Location: `src/models/gpt/unified_voice.rs`
- Combines text embeddings, mel embeddings, conformer, perceiver, GPT-2 decoder
- Public API: `forward()`, `forward_one()`, `process_conditioning()`, `embed_mel_codes()`

**FlowMatching:**
- Purpose: Conditional flow matching sampler
- Location: `src/models/s2mel/flow_matching.rs`
- Implements Euler ODE solver with classifier-free guidance
- Public API: `sample()`, `sample_noise()`

**DiffusionTransformer:**
- Purpose: DiT model for velocity prediction in flow matching
- Location: `src/models/s2mel/dit.rs`
- 13-layer transformer with 512 hidden dim, 8 heads
- Public API: `forward()`

**BigVGAN:**
- Purpose: Neural vocoder for mel-to-waveform conversion
- Location: `src/models/vocoder/bigvgan.rs`
- Anti-aliased upsampling with Snake-Beta activations
- Public API: `forward()`, `sample_rate()`

## Entry Points

**CLI Application:**
- Location: `src/main.rs`
- Triggers: `cargo run --release --bin indextts2`
- Responsibilities: Parse CLI args, load model, run inference, save output
- Commands: `infer`, `serve`, `download`, `info`

**Library API:**
- Location: `src/lib.rs`
- Triggers: `use indextts2::IndexTTS2`
- Responsibilities: Export public API for library consumers
- Re-exports: `IndexTTS2`, `ModelConfig`, `InferenceConfig`

**Debug Validator:**
- Location: `src/bin/debug_validate.rs`
- Triggers: `cargo run --bin debug_validate`
- Responsibilities: Validate Rust implementation against Python golden data

## Error Handling

**Strategy:** Result-based error propagation with anyhow

**Patterns:**
- All fallible operations return `Result<T>` (anyhow for ergonomics)
- Context added via `.context()` and `.with_context()`
- Early returns on missing weights with informative errors
- Graceful degradation: random weight initialization when checkpoints missing

## Cross-Cutting Concerns

**Logging:** tracing crate with configurable verbosity
**Validation:** Debug utilities for Python parity checking (`src/debug/`)
**Configuration:** YAML-based model config via serde (`src/config/`)
**Device Handling:** Candle Device abstraction (CPU/CUDA)

---

*Architecture analysis: 2026-01-23*
