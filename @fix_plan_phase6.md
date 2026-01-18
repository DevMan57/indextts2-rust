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

## Phase 6: Debug & Validate [PENDING]

### Reference Data Generation
- [ ] **P6.1** Create `debug/dump_python.py` script
  - Hook into Python IndexTTS model
  - Save intermediate tensors at each layer
  - Use numpy .npy format for cross-language compatibility

- [ ] **P6.2** Generate golden reference data
  - Run Python with test input: "Hello world" + reference speaker
  - Save: tokens, mel, speaker_emb, semantic, gpt_layers, mel_codes, audio

### Rust Validation Harness
- [ ] **P6.3** Implement `src/debug/mod.rs` and `src/debug/validator.rs`
  - NPY loading via `ndarray-npy` crate
  - Shape comparison
  - Value comparison with configurable tolerance
  - Context7: `Context7:resolve-library-id "ndarray-npy"`

- [ ] **P6.4** Create debug CLI: `src/bin/debug_validate.rs`
  - `--component` flag for targeted validation
  - `--verbose` for detailed diff output

### Layer-by-Layer Validation
- [ ] **P6.5** Validate text processing
  - Tokenizer output matches
  - Check special tokens, padding

- [ ] **P6.6** Validate audio processing
  - Mel spectrogram matches within 1e-4
  - Check FFT, filterbank, normalization

- [ ] **P6.7** Validate encoders
  - Speaker embedding (CAMPPlus)
  - Semantic features (Wav2Vec-BERT)

- [ ] **P6.8** Validate GPT forward pass
  - Layer 0, 12, 23 outputs
  - KV-cache values
  - Final mel codes

- [ ] **P6.9** Validate synthesis
  - DiT output
  - Flow matching trajectory
  - BigVGAN mel-to-audio

### Fix & Iterate
- [ ] **P6.10** Document all fixes in `FIXES.md`
  - What was wrong
  - Root cause
  - Solution applied

- [ ] **P6.11** End-to-end audio comparison
  - Waveform correlation > 0.95
  - Mel spectrogram MSE < 0.01
  - Perceptual quality check

---

## Completion Tracking

**Phase 1:** 8/8 complete ✅
**Phase 2:** 4/4 complete ✅
**Phase 3:** 0/5 complete
**Phase 4:** 0/4 complete
**Phase 5:** 0/4 complete
**Phase 6:** 0/11 complete

**Total:** 12/36 tasks complete
