# IndexTTS2 Rust Rewrite - Task Plan

## Phase 1: Foundation [COMPLETE]

### Config & Setup
- [x] **P1.1** Implement `src/config/model_config.rs` - YAML config parsing with serde
  - Parse `checkpoints/config.yaml` structure
  - Create `GptConfig`, `S2MelConfig`, `VocoderConfig` structs
  - Context7: `Context7:resolve-library-id "serde yaml"`

### Text Processing
- [x] **P1.2** Implement `src/text/tokenizer.rs` - BPE tokenization
  - Use HuggingFace tokenizers crate
  - Load tokenizer from JSON file, encode/decode text
  - Added: encode_for_gpt(), vocab_size(), set_special_tokens()

- [x] **P1.3** Implement `src/text/normalizer.rs` - Text normalization
  - Number to words conversion (integers, decimals, negatives)
  - Abbreviation expansion (Dr., Mr., etc.)
  - Punctuation and whitespace normalization

- [x] **P1.4** Implement `src/text/segmenter.rs` - Sentence segmentation
  - Split at sentence boundaries (. ! ? and Chinese equivalents)
  - Respect max_tokens limit with fallback to clause/word boundaries
  - segment_text(), segment_text_string(), segment_mixed_text()

### Audio I/O
- [x] **P1.5** Implement `src/audio/loader.rs` - Audio file loading
  - Support WAV, MP3, FLAC, OGG via symphonia
  - Automatic resampling to target rate
  - Mono conversion from stereo

- [x] **P1.6** Implement `src/audio/resampler.rs` - Sample rate conversion
  - High-quality sinc interpolation via rubato
  - Chunked processing for long audio
  - Convenience methods: resample_to_16k(), resample_to_22k()

- [x] **P1.7** Implement `src/audio/mel.rs` - Mel spectrogram
  - FFT via rustfft with cached planner
  - 80 mel bands, librosa-compatible
  - new_default() for IndexTTS2 parameters, compute_transposed()

- [x] **P1.8** Implement `src/audio/output.rs` - Audio playback
  - WAV file saving (16-bit PCM, 32-bit float)
  - Real-time playback via cpal
  - StreamingPlayer for real-time TTS output

---

## Phase 2: Core Encoders [COMPLETE]

### Semantic Encoder
- [x] **P2.1** Implement `src/models/semantic/wav2vec_bert.rs`
  - Wav2Vec-BERT 2.0 feature extraction
  - Self-attention, feed-forward, encoder layers
  - Layer 17 extraction with normalization

- [x] **P2.2** Implement `src/models/semantic/codec.rs`
  - Vector quantization with 8192 codebook
  - L2 distance nearest neighbor search
  - Projection layers for hidden_size/codebook_dim conversion

### Speaker Encoder
- [x] **P2.3** Implement `src/models/speaker/campplus.rs`
  - CAMPPlus speaker embedding (192-dim output)
  - D-TDNN architecture with dense connections
  - Statistics pooling (mean + std)

### Emotion (Optional)
- [x] **P2.4** Implement `src/models/emotion/matrix.rs`
  - 8 emotion categories matrix lookup
  - Emotion blending with configurable alpha
  - Category-based and global index access

---

## Phase 3: GPT Generation [PENDING]

### Conformer Encoder
- [ ] **P3.1** Implement `src/models/gpt/conformer.rs`
  - Conformer encoder blocks
  - Convolution + self-attention
  - Reference: `indextts/gpt/conformer_encoder.py`
  - HuggingFace: `Hugging Face:paper_search query="conformer encoder"`

### Perceiver Resampler
- [ ] **P3.2** Implement `src/models/gpt/perceiver.rs`
  - Cross-attention resampler
  - Reference: `indextts/gpt/perceiver.py`

### KV Cache
- [ ] **P3.3** Implement `src/models/gpt/kv_cache.rs`
  - Efficient key-value caching
  - Use candle_nn::kv_cache::Cache
  - Context7: `Context7:get-library-docs "/huggingface/candle" topic="kv cache"`

### Unified Voice Model
- [ ] **P3.4** Implement `src/models/gpt/unified_voice.rs`
  - GPT-2 architecture: 1280 dim, 24 layers, 20 heads
  - Text + audio conditioning
  - Reference: `indextts/gpt/model_v2.py`

### Generation Loop
- [ ] **P3.5** Implement `src/models/gpt/generation.rs`
  - Autoregressive generation
  - Top-k/top-p sampling
  - Stop token detection (8193)

---

## Phase 4: Synthesis [PENDING]

### Length Regulator
- [ ] **P4.1** Implement `src/models/s2mel/length_regulator.rs`
  - Mel code to target length expansion
  - Reference: `indextts/s2mel/modules/length_regulator.py`

### Diffusion Transformer
- [ ] **P4.2** Implement `src/models/s2mel/dit.rs`
  - DiT: 13 layers, 512 hidden, 8 heads
  - Reference: `indextts/s2mel/modules/diffusion_transformer.py`

### Flow Matching
- [ ] **P4.3** Implement `src/models/s2mel/flow_matching.rs`
  - CFM with 25 steps
  - Euler ODE solver
  - cfg_rate=0.7
  - Reference: `indextts/s2mel/modules/flow_matching.py`

### BigVGAN Vocoder
- [ ] **P4.4** Implement `src/models/vocoder/bigvgan.rs`
  - BigVGAN v2 22kHz 80-band
  - Anti-alias activation
  - Reference: `indextts/BigVGAN/bigvgan.py`
  - HuggingFace: `Hugging Face:model_search query="BigVGAN"`

---

## Phase 5: Integration [PENDING]

### Pipeline
- [ ] **P5.1** Implement `src/inference/pipeline.rs`
  - Wire all components together
  - Reference: `indextts/infer_v2.py`

### Streaming
- [ ] **P5.2** Implement `src/inference/streaming.rs`
  - Real-time audio output
  - Chunk-based synthesis

### CLI
- [ ] **P5.3** Complete `src/main.rs`
  - clap argument parsing
  - --text, --speaker, --emotion, --output flags

### Tests
- [ ] **P5.4** Integration tests in `tests/`
  - End-to-end inference test
  - Audio quality validation

---

## Completion Tracking

**Phase 1:** 8/8 complete ✅
**Phase 2:** 4/4 complete ✅
**Phase 3:** 0/5 complete
**Phase 4:** 0/4 complete
**Phase 5:** 0/4 complete

**Total:** 12/25 tasks complete
