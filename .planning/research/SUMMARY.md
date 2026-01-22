# Research Summary: Weight Name Mapping Architecture

**Domain:** TTS Model Weight Loading
**Researched:** 2026-01-23
**Overall Confidence:** HIGH

## Executive Summary

The IndexTTS2 Rust codebase already has the correct architecture for weight loading - each model type (Wav2Vec-BERT, DiT, Conformer, Perceiver) has dedicated loading functions that transform checkpoint tensors into model structures. The BigVGAN vocoder demonstrates this pattern working correctly.

The issue is not architectural but operational: the loading functions may have incomplete tensor name mappings or incorrect transformations (weight normalization, fused tensor splitting). The fix requires validating actual checkpoint tensor names against expected names in each loader, then patching the mismatches.

## Key Findings

**Stack:** Candle ML framework with safetensors for model loading. VarBuilder available but not needed - direct HashMap loading with custom transformation is the established pattern.

**Architecture:** Load-time transformation pattern where each model's `from_tensors()` or `from_gpt_tensors()` method handles name mapping and structural transformations inline.

**Critical Pitfall:** Current code falls back to random weights on missing keys, masking loading failures. Need strict mode validation during debugging.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Diagnostic (1-2 hours)
- Add debug logging to print actual tensor keys from each safetensors file
- Compare against expected keys in loader code
- Document specific mismatches

**Rationale:** Cannot fix what you cannot measure. Must identify actual vs expected differences before changing code.

### Phase 2: Wav2Vec-BERT Fix (2-4 hours)
- Verify HuggingFace checkpoint format matches loader expectations
- Current code already uses correct HuggingFace naming convention
- May just need checkpoint path fix or minor key adjustments

**Rationale:** Standalone model with clear HuggingFace standard. Easiest to validate.

### Phase 3: GPT Component Fixes (4-6 hours)
- Conformer: Verify `conditioning_encoder.*` keys match
- Perceiver: Verify `perceiver_encoder.*` keys, check fused KV splitting

**Rationale:** Both load from same gpt.safetensors, share similar patterns.

### Phase 4: DiT Fix (4-6 hours)
- Verify `cfm.estimator.*` keys
- Ensure weight normalization applied to x_embedder
- Check fused QKV splitting for transformer layers

**Rationale:** Most complex transformations. Save for last when other models working.

### Phase 5: Integration Validation (2-4 hours)
- Run end-to-end inference
- Verify encoder outputs are non-random (check statistics)
- Test audio quality

**Phase ordering rationale:**
- Diagnostic first prevents wasted effort
- Wav2Vec-BERT is standalone - validate loading pattern works
- GPT components share checkpoint - fix together
- DiT has most complexity - fix last
- Integration confirms all pieces work together

**Research flags for phases:**
- Phase 2: Standard HuggingFace pattern, unlikely to need more research
- Phase 4: May need checkpoint inspection if loading still fails

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Architecture pattern | HIGH | BigVGAN proves the pattern works |
| Wav2Vec-BERT naming | HIGH | Verified from HuggingFace source |
| DiT naming | MEDIUM | Based on code analysis, needs checkpoint verification |
| Conformer naming | MEDIUM | Code exists, needs validation |
| Perceiver naming | MEDIUM | Code exists, needs validation |

## Gaps to Address

- **Checkpoint inspection:** Need to run Python script to list actual tensor keys from each safetensors file
- **Candle transpose behavior:** Some `load_linear()` calls may incorrectly transpose. Review each call site.
- **Weight normalization edge cases:** DiT x_embedder uses weight norm but handling may be incomplete

## Files Created

| File | Purpose |
|------|---------|
| `.planning/research/SUMMARY.md` | This summary with roadmap implications |
| `.planning/research/ARCHITECTURE.md` | Detailed tensor naming and mapping patterns |

## Ready for Roadmap

Research complete. The fix is not architectural redesign but operational debugging:
1. Log actual vs expected tensor keys
2. Patch mismatches in each model's loader
3. Validate with inference test

Total estimated effort: 2-3 days for full fix and validation.
