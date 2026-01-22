# Roadmap: IndexTTS2 Weight Loading + CUDA

## Overview

This roadmap fixes weight loading for four neural network components (Wav2Vec-BERT, Conformer, Perceiver, DiT) and enables CUDA acceleration. The BigVGAN vocoder already loads correctly, proving the architecture works. The fix requires identifying tensor name mismatches between HuggingFace checkpoints and Rust loaders, then patching each component. Once weights load correctly, the TTS pipeline will produce intelligible speech instead of noise.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Diagnostic Infrastructure** - Add strict validation to expose weight loading failures
- [ ] **Phase 2: Wav2Vec-BERT Weights** - Fix semantic encoder weight loading
- [ ] **Phase 3: GPT Components** - Fix Conformer and Perceiver weight loading
- [ ] **Phase 4: DiT Weights** - Fix flow matching model weight loading
- [ ] **Phase 5: CUDA Foundation** - Enable GPU device selection and basic CUDA inference
- [ ] **Phase 6: CUDA Optimization** - Enable cuDNN acceleration for convolutions
- [ ] **Phase 7: Weight Validation** - Verify all components load correctly without fallback
- [ ] **Phase 8: Integration Testing** - End-to-end verification with real weights

## Phase Details

### Phase 1: Diagnostic Infrastructure
**Goal**: Expose silent weight loading failures so subsequent fixes can be validated
**Depends on**: Nothing (first phase)
**Requirements**: None (enables other requirements)
**Plans:** 1 plan
**Success Criteria** (what must be TRUE):
  1. Running inference prints actual tensor names from each safetensors file
  2. Running inference prints which tensors were found vs missing for each component
  3. Attempting to load with missing tensors produces a visible warning (not silent fallback)

Plans:
- [ ] 01-01-PLAN.md — Create WeightDiagnostics module and add tracing::warn! to silent fallbacks

### Phase 2: Wav2Vec-BERT Weights
**Goal**: Semantic encoder loads pre-trained weights and produces meaningful embeddings
**Depends on**: Phase 1 (need diagnostics to verify fix)
**Requirements**: WEIGHT-01
**Success Criteria** (what must be TRUE):
  1. Wav2Vec-BERT loads all 24 encoder layers from wav2vec2_bert_2.safetensors
  2. No "using random weights" warnings appear for Wav2Vec-BERT
  3. Encoder output statistics (mean/std) differ from random initialization
**Plans**: TBD

Plans:
- [ ] 02-01: Map HuggingFace tensor names to Rust loader expectations

### Phase 3: GPT Components
**Goal**: Conformer encoder and Perceiver resampler load pre-trained weights from gpt.safetensors
**Depends on**: Phase 1 (need diagnostics to verify fix)
**Requirements**: WEIGHT-03, WEIGHT-04
**Success Criteria** (what must be TRUE):
  1. Conformer loads all 24 layers from gpt.safetensors without fallback
  2. Perceiver loads 32 latents and 2 attention layers from gpt.safetensors without fallback
  3. No "using random weights" warnings appear for either component
  4. GPT generation produces non-random mel codes
**Plans**: TBD

Plans:
- [ ] 03-01: Fix Conformer tensor name mapping
- [ ] 03-02: Fix Perceiver tensor name mapping and fused KV splitting

### Phase 4: DiT Weights
**Goal**: Flow matching model loads pre-trained weights and produces quality mel spectrograms
**Depends on**: Phase 1 (need diagnostics to verify fix)
**Requirements**: WEIGHT-02
**Success Criteria** (what must be TRUE):
  1. DiT loads all 13 transformer blocks from s2mel.safetensors
  2. Weight normalization applied correctly to x_embedder
  3. Fused QKV tensors split correctly for transformer attention
  4. No "using random weights" warnings appear for DiT
**Plans**: TBD

Plans:
- [ ] 04-01: Fix DiT tensor name mapping and weight transformations

### Phase 5: CUDA Foundation
**Goal**: User can run inference on CUDA device instead of CPU
**Depends on**: Phases 2-4 (weights must load before GPU testing)
**Requirements**: GPU-01
**Success Criteria** (what must be TRUE):
  1. Running with --gpu flag uses CUDA device for inference
  2. CLI displays "Using device: CUDA" when GPU mode enabled
  3. Inference completes without CUDA errors on RTX 3090
  4. Output audio file is generated successfully
**Plans**: TBD

Plans:
- [ ] 05-01: Enable CUDA feature flags and device selection

### Phase 6: CUDA Optimization
**Goal**: cuDNN acceleration enabled for faster convolution operations
**Depends on**: Phase 5 (CUDA must work before optimization)
**Requirements**: GPU-02
**Success Criteria** (what must be TRUE):
  1. cuDNN acceleration active during BigVGAN vocoder inference
  2. Convolution operations use cuDNN kernels (not fallback implementations)
  3. GPU inference is noticeably faster than CPU inference
**Plans**: TBD

Plans:
- [ ] 06-01: Configure cuDNN integration with Candle

### Phase 7: Weight Validation
**Goal**: Confirm all components load weights correctly and produce equivalent quality
**Depends on**: Phases 2-4 (all weight fixes), Phase 5 (CUDA for comparison)
**Requirements**: GPU-03
**Success Criteria** (what must be TRUE):
  1. All 4 components (Wav2Vec-BERT, Conformer, Perceiver, DiT) load without warnings
  2. Weight statistics for each component match expected ranges (not random)
  3. GPU inference produces equivalent output to CPU inference
  4. No silent fallback to random weights in any component
**Plans**: TBD

Plans:
- [ ] 07-01: Create weight validation test suite

### Phase 8: Integration Testing
**Goal**: End-to-end TTS pipeline produces intelligible speech
**Depends on**: Phase 7 (all weights validated)
**Requirements**: VERIFY-01, VERIFY-02
**Success Criteria** (what must be TRUE):
  1. Integration test runs end-to-end with real checkpoint weights
  2. Output audio plays as intelligible speech (text is recognizable)
  3. Audio is not noise/static/wind sounds
  4. Voice characteristics match speaker reference audio
  5. Test passes consistently (not flaky)
**Plans**: TBD

Plans:
- [ ] 08-01: Create integration test with real weights
- [ ] 08-02: Manual audio quality verification

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Diagnostic Infrastructure | 0/1 | Planned | - |
| 2. Wav2Vec-BERT Weights | 0/1 | Not started | - |
| 3. GPT Components | 0/2 | Not started | - |
| 4. DiT Weights | 0/1 | Not started | - |
| 5. CUDA Foundation | 0/1 | Not started | - |
| 6. CUDA Optimization | 0/1 | Not started | - |
| 7. Weight Validation | 0/1 | Not started | - |
| 8. Integration Testing | 0/2 | Not started | - |
