# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
indextts2-rust/
├── src/
│   ├── main.rs                    # CLI entry point
│   ├── lib.rs                     # Library exports
│   ├── audio/                     # Audio I/O and processing
│   │   ├── mod.rs                 # Module exports
│   │   ├── loader.rs              # Audio file loading (WAV, MP3, FLAC, OGG)
│   │   ├── resampler.rs           # Sample rate conversion
│   │   ├── mel.rs                 # Mel spectrogram computation
│   │   └── output.rs              # Audio output/playback
│   ├── config/                    # Configuration management
│   │   ├── mod.rs                 # Module exports
│   │   └── model_config.rs        # YAML config parsing
│   ├── debug/                     # Debug and validation tools
│   │   ├── mod.rs                 # Module exports
│   │   ├── validator.rs           # Tensor comparison utilities
│   │   └── npy_loader.rs          # NumPy file loading
│   ├── inference/                 # Inference pipeline
│   │   ├── mod.rs                 # Module exports
│   │   ├── pipeline.rs            # Main IndexTTS2 orchestrator
│   │   └── streaming.rs           # Real-time streaming synthesis
│   ├── models/                    # Neural network models
│   │   ├── mod.rs                 # Module exports with re-exports
│   │   ├── emotion/               # Emotion processing
│   │   │   ├── mod.rs             # EmotionMatrix exports
│   │   │   └── matrix.rs          # Emotion embedding matrix
│   │   ├── gpt/                   # GPT-2 autoregressive model
│   │   │   ├── mod.rs             # Module exports
│   │   │   ├── unified_voice.rs   # Main GPT model (1280 dim, 24 layers)
│   │   │   ├── conformer.rs       # Conformer encoder (audio conditioning)
│   │   │   ├── perceiver.rs       # Perceiver resampler (cross-attention)
│   │   │   ├── kv_cache.rs        # KV cache for generation
│   │   │   ├── generation.rs      # Autoregressive generation loop
│   │   │   └── weights.rs         # Safetensors weight loading
│   │   ├── s2mel/                 # Semantic-to-Mel synthesis
│   │   │   ├── mod.rs             # Module exports
│   │   │   ├── length_regulator.rs # Duration/alignment
│   │   │   ├── dit.rs             # Diffusion Transformer (13 layers)
│   │   │   ├── flow_matching.rs   # CFM sampler (25 steps)
│   │   │   └── weights.rs         # S2Mel weight loading
│   │   ├── semantic/              # Semantic encoding
│   │   │   ├── mod.rs             # Module exports
│   │   │   ├── wav2vec_bert.rs    # Wav2Vec-BERT 2.0 encoder
│   │   │   └── codec.rs           # Vector quantization codec
│   │   ├── speaker/               # Speaker encoding
│   │   │   ├── mod.rs             # Module exports
│   │   │   └── campplus.rs        # CAMPPlus encoder (192-dim)
│   │   └── vocoder/               # Audio synthesis
│   │       ├── mod.rs             # Module exports
│   │       ├── bigvgan.rs         # BigVGAN v2 vocoder
│   │       └── weights.rs         # BigVGAN weight loading
│   ├── text/                      # Text processing
│   │   ├── mod.rs                 # Module exports
│   │   ├── normalizer.rs          # Text normalization
│   │   ├── tokenizer.rs           # BPE tokenization
│   │   └── segmenter.rs           # Sentence segmentation
│   ├── utils/                     # Shared utilities
│   │   └── mod.rs                 # Tensor utilities, string helpers
│   └── bin/                       # Additional binaries
│       └── debug_validate.rs      # Validation CLI tool
├── benches/                       # Performance benchmarks
│   └── inference_bench.rs         # Inference benchmarks
├── tests/                         # Integration tests
│   └── integration_test.rs        # Full pipeline tests
├── checkpoints/                   # Model weights directory
│   ├── config.yaml                # Model configuration
│   ├── gpt.pth                    # GPT model weights (safetensors)
│   ├── s2mel.pth                  # S2Mel weights (safetensors)
│   └── bigvgan-v2/                # BigVGAN weights
├── examples/                      # Usage examples
├── scripts/                       # Build/deploy scripts
├── Cargo.toml                     # Rust package manifest
├── CLAUDE.md                      # Project documentation
└── README.md                      # Project readme
```

## Module Responsibilities

### `src/audio/`
- **Purpose:** Audio file I/O and signal processing
- **Key files:**
  - `loader.rs`: Load WAV/MP3/FLAC/OGG, resample to target rate
  - `mel.rs`: 80-band mel spectrogram extraction
  - `output.rs`: WAV file writing, streaming playback
- **Exports:** `AudioLoader`, `Resampler`, `MelSpectrogram`, `AudioOutput`, `StreamingPlayer`

### `src/config/`
- **Purpose:** YAML configuration parsing and model config types
- **Key files:**
  - `model_config.rs`: `ModelConfig`, `GptConfig`, `S2MelConfig`, `VocoderConfig`
- **Exports:** `ModelConfig` and all config structs

### `src/debug/`
- **Purpose:** Development tools for validating against Python reference
- **Key files:**
  - `validator.rs`: Tensor comparison with tolerances
  - `npy_loader.rs`: Load NumPy .npy files
- **Exports:** `Validator`, `ValidationResult`, `load_npy`

### `src/inference/`
- **Purpose:** Main inference pipeline and streaming support
- **Key files:**
  - `pipeline.rs`: `IndexTTS2` main orchestrator, `InferenceConfig`, `InferenceResult`
  - `streaming.rs`: `StreamingSynthesizer` for real-time output
- **Exports:** `IndexTTS2`, `InferenceConfig`, `InferenceResult`, `StreamingSynthesizer`

### `src/models/gpt/`
- **Purpose:** GPT-2 based autoregressive mel code generation
- **Key files:**
  - `unified_voice.rs`: Main model (1280 dim, 24 layers, 20 heads)
  - `conformer.rs`: Audio conditioning encoder (6 blocks)
  - `perceiver.rs`: Cross-attention resampler (32 latents, 2 layers)
  - `kv_cache.rs`: Efficient KV cache for generation
  - `generation.rs`: `generate()`, `generate_with_hidden()` functions
- **Exports:** `UnifiedVoice`, `ConformerEncoder`, `PerceiverResampler`, `KVCache`, `GenerationConfig`

### `src/models/s2mel/`
- **Purpose:** Semantic-to-Mel conversion via flow matching
- **Key files:**
  - `length_regulator.rs`: Duration prediction and feature alignment
  - `dit.rs`: Diffusion Transformer (512 hidden, 13 layers, 8 heads)
  - `flow_matching.rs`: CFM sampler (25 steps, cfg_rate=0.7)
- **Exports:** `LengthRegulator`, `DiffusionTransformer`, `FlowMatching`

### `src/models/semantic/`
- **Purpose:** Semantic feature extraction from audio
- **Key files:**
  - `wav2vec_bert.rs`: Wav2Vec-BERT 2.0 encoder (24 layers)
  - `codec.rs`: Vector quantization codec
- **Exports:** `SemanticEncoder`, `SemanticCodec`

### `src/models/speaker/`
- **Purpose:** Speaker identity extraction
- **Key files:**
  - `campplus.rs`: CAMPPlus D-TDNN encoder (192-dim output)
- **Exports:** `CAMPPlus`

### `src/models/vocoder/`
- **Purpose:** Mel-to-waveform neural vocoder
- **Key files:**
  - `bigvgan.rs`: BigVGAN v2 (22kHz, 80-band, 256x upsampling)
  - `weights.rs`: Weight loading utilities
- **Exports:** `BigVGAN`, `BigVGANConfig`

### `src/text/`
- **Purpose:** Text normalization and tokenization
- **Key files:**
  - `normalizer.rs`: Number/abbreviation expansion
  - `tokenizer.rs`: BPE tokenization
  - `segmenter.rs`: Sentence boundary detection
- **Exports:** `TextNormalizer`, `TextTokenizer`, `segment_text`

## Key File Locations

**Entry Points:**
- `src/main.rs`: CLI application entry point
- `src/lib.rs`: Library crate root with public exports

**Configuration:**
- `checkpoints/config.yaml`: Model hyperparameters
- `Cargo.toml`: Rust dependencies and features

**Core Logic:**
- `src/inference/pipeline.rs`: Main inference orchestration (638 lines)
- `src/models/gpt/unified_voice.rs`: GPT model implementation (945 lines)
- `src/models/s2mel/flow_matching.rs`: Flow matching sampler
- `src/models/vocoder/bigvgan.rs`: Vocoder implementation

**Testing:**
- `tests/integration_test.rs`: End-to-end tests
- `benches/inference_bench.rs`: Performance benchmarks
- `src/bin/debug_validate.rs`: Validation against Python

## Naming Conventions

**Files:**
- Lowercase with underscores: `unified_voice.rs`, `flow_matching.rs`
- Module files: `mod.rs` for directory modules

**Directories:**
- Lowercase, descriptive: `models/`, `audio/`, `inference/`
- Model subdirs by component: `gpt/`, `s2mel/`, `vocoder/`

**Types:**
- PascalCase: `UnifiedVoice`, `FlowMatching`, `BigVGAN`
- Config structs suffix: `*Config` (e.g., `InferenceConfig`, `BigVGANConfig`)

**Functions:**
- snake_case: `load_weights()`, `process_conditioning()`, `embed_mel_codes()`

## Where to Add New Code

**New Model Component:**
- Create directory under `src/models/` (e.g., `src/models/new_component/`)
- Add `mod.rs` with exports
- Add component file(s) with implementation
- Export in `src/models/mod.rs`
- Integrate in `src/inference/pipeline.rs`

**New Utility Function:**
- General utilities: `src/utils/mod.rs`
- Audio utilities: `src/audio/`
- Tensor utilities: `src/utils/mod.rs::tensor_utils`

**New CLI Command:**
- Add variant to `Commands` enum in `src/main.rs`
- Implement handler in `main()` match block

**New Test:**
- Unit tests: `#[cfg(test)] mod tests` in same file
- Integration tests: `tests/integration_test.rs`
- Benchmark: `benches/inference_bench.rs`

**New Configuration:**
- Add field to relevant struct in `src/config/model_config.rs`
- Update `checkpoints/config.yaml` with defaults

## Special Directories

**`checkpoints/`:**
- Purpose: Model weights and configuration
- Generated: Downloaded from HuggingFace
- Committed: Only `config.yaml`, not weights

**`benches/`:**
- Purpose: Criterion benchmarks
- Generated: No
- Committed: Yes

**`examples/`:**
- Purpose: Usage examples
- Generated: No
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning documents
- Generated: By GSD commands
- Committed: Optional (project-specific)

---

*Structure analysis: 2026-01-23*
