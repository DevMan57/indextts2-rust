# Requirements

**Project:** IndexTTS2 Rust
**Version:** v1 (Weight Loading Fix + CUDA)
**Last Updated:** 2026-01-23

## v1 Requirements

### Weight Loading

- [ ] **WEIGHT-01**: Wav2Vec-BERT encoder loads pre-trained weights from `wav2vec2_bert_2.safetensors` without fallback to random initialization
- [ ] **WEIGHT-02**: DiT flow matching model loads pre-trained weights from `s2mel.safetensors` without fallback to random initialization
- [ ] **WEIGHT-03**: Conformer encoder loads pre-trained weights from `gpt.safetensors` without fallback to random initialization
- [ ] **WEIGHT-04**: Perceiver resampler loads pre-trained weights from `gpt.safetensors` without fallback to random initialization

### GPU Support

- [ ] **GPU-01**: User can run inference on CUDA device (RTX 3090) instead of CPU
- [ ] **GPU-02**: cuDNN acceleration is enabled for faster convolution operations
- [ ] **GPU-03**: GPU inference produces equivalent quality output as CPU inference would with correct weights

### Verification

- [ ] **VERIFY-01**: Output audio is intelligible speech when given text input and speaker reference
- [ ] **VERIFY-02**: Integration test runs end-to-end with real checkpoint weights and produces valid audio

## v2 Requirements (Deferred)

### Developer Experience
- [ ] Diagnostic logging to show actual vs expected tensor names during weight loading
- [ ] Weight statistics logging (mean/std) to verify loaded vs random weights
- [ ] Strict validation mode that errors on missing weights instead of silent fallback
- [ ] Device selection CLI flag (`--gpu` / `--cpu`)

### Features
- [ ] HTTP/WebSocket server for API access
- [ ] Model auto-download from HuggingFace
- [ ] Python bindings (PyO3)
- [ ] Batch processing multiple texts

## Out of Scope

- **Python bindings** — Future enhancement, not needed for core TTS
- **HTTP server** — Placeholder exists, not priority for v1
- **Model auto-download** — Manual checkpoint management acceptable
- **Performance optimization** — Beyond GPU acceleration
- **Multi-speaker batch inference** — Single speaker focus

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WEIGHT-01 | Phase 2: Wav2Vec-BERT Weights | Pending |
| WEIGHT-02 | Phase 4: DiT Weights | Pending |
| WEIGHT-03 | Phase 3: GPT Components | Pending |
| WEIGHT-04 | Phase 3: GPT Components | Pending |
| GPU-01 | Phase 5: CUDA Foundation | Pending |
| GPU-02 | Phase 6: CUDA Optimization | Pending |
| GPU-03 | Phase 7: Weight Validation | Pending |
| VERIFY-01 | Phase 8: Integration Testing | Pending |
| VERIFY-02 | Phase 8: Integration Testing | Pending |

---
*Requirements defined: 2026-01-23*
*Traceability updated: 2026-01-23*
