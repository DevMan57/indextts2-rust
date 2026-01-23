---
phase: 04-dit-weights
plan: 01
subsystem: s2mel
tags: [dit, weight-loading, skip-connection, uvit]
depends_on:
  requires: [01, 02, 03]
  provides: [skip_in_linear loading, DiT weight verification]
  affects: [05-synthesis, 06-integration]
tech-stack:
  added: []
  patterns: [Optional<Linear> for conditional loading, checkpoint tensor mapping]
key-files:
  created: []
  modified:
    - src/models/s2mel/dit.rs
decisions:
  - id: "04-01-skip-deferred"
    choice: "Load skip_in_linear weights but defer forward pass usage"
    reason: "UViT skip connection wiring requires careful integration; weight loading is prerequisite"
metrics:
  duration: "9 minutes"
  completed: "2026-01-23"
---

# Phase 4 Plan 01: Add skip_in_linear to DiT Summary

**One-liner:** Added skip_in_linear field to DiTBlock for UViT skip connections, all 13 transformer blocks load from checkpoint

## What Was Built

### Core Change: skip_in_linear in DiTBlock

Added `skip_in_linear: Option<Linear>` field to DiTBlock struct to support UViT-style skip connections between encoder and decoder transformer blocks.

**Key code additions:**

1. **Struct field:**
   ```rust
   struct DiTBlock {
       // ... existing fields ...
       skip_in_linear: Option<Linear>,  // [512, 1024] for UViT skip connections
   }
   ```

2. **Weight loading in from_tensors():**
   ```rust
   let skip_key = format!("{}.skip_in_linear.weight", prefix);
   let skip_in_linear = match load_linear(tensors, &skip_key, Some(&format!("{}.skip_in_linear.bias", prefix))) {
       Ok(linear) => Some(linear),
       Err(_) => {
           tracing::warn!("[DiT] Missing tensor '{}', skip connection disabled", skip_key);
           None
       }
   };
   ```

3. **Forward signature updated:**
   ```rust
   fn forward(&self, x: &Tensor, cond: &Tensor, _skip: Option<&Tensor>) -> Result<Tensor>
   ```
   Note: Skip parameter accepted but not wired (deferred to future phase)

### Verification Results

**Weight Loading Output:**
```
  Loaded skip_in_linear [512, 1024]
  Loaded 13 of 13 transformer blocks
DiT weights loaded successfully (including post-transformer)
```

**Test Results:**
- 11 DiT-related tests pass
- 22 s2mel tests pass
- Zero "[DiT]" random weight warnings
- Zero "using random weights" fallbacks for DiT components

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 04-01-skip-deferred | Load skip_in_linear but defer forward usage | Weight loading is prerequisite; forward pass wiring needs careful integration |

## Deviations from Plan

### Observation: Inference Failure in GPT Component

**Context:** Task 3 attempted full inference but encountered shape mismatch error:
```
shape mismatch in matmul, lhs: [1, 126, 1280], rhs: [1, 512, 1280]
```

**Analysis:** This error originates in the GPT perceiver component (proj_context), not DiT. The DiT weight loading completed successfully as evidenced by:
- "Loaded 13 of 13 transformer blocks"
- "DiT weights loaded successfully"
- Zero DiT-related warnings

**Impact:** The plan objective (DiT weight loading) was achieved. The inference failure is a pre-existing issue in a different component (GPT perceiver) and is outside Phase 4 scope.

## Key Files Modified

| File | Changes |
|------|---------|
| `src/models/s2mel/dit.rs` | Added skip_in_linear field, loading logic, updated forward signature |

## Next Phase Readiness

### Prerequisites Met
- [x] DiT loads all 13 transformer blocks from s2mel.safetensors
- [x] Weight normalization applied correctly to x_embedder
- [x] Fused QKV tensors split correctly
- [x] No "using random weights" warnings for DiT
- [x] skip_in_linear loaded for each transformer block
- [x] All tests pass

### Blockers/Concerns
1. **GPT perceiver shape mismatch** - Pre-existing issue affecting full inference
   - Error: `shape mismatch in matmul, lhs: [1, 126, 1280], rhs: [1, 512, 1280]`
   - Location: GPT perceiver proj_context
   - Impact: Blocks end-to-end inference but not DiT weight loading

### Future Work
1. Wire skip_in_linear in forward pass for actual UViT skip connections
2. Fix GPT perceiver dimension mismatch (separate issue)
3. Validate mel spectrogram output quality after skip connections enabled

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 4f03e50 | feat | Add skip_in_linear to DiTBlock for UViT skip connections |
