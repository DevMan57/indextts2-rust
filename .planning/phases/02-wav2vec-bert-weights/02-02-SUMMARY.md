---
phase: 02-wav2vec-bert-weights
plan: 02
subsystem: semantic-encoder
tags: [wav2vec-bert, relative-position, normalization, attention]
requires: ["02-01"]
provides: ["distance_embedding", "var_to_std_conversion"]
affects: ["inference-accuracy"]
tech-stack:
  added: []
  patterns: ["Shaw-style relative position encoding", "variance to std conversion"]
key-files:
  created: []
  modified:
    - src/models/semantic/wav2vec_bert.rs
decisions:
  - id: "02-02-01"
    choice: "var->std via sqrt() with fallback chain"
    why: "Stats file stores variance, not std. Chain: try std key -> try var with sqrt() -> warn and use ones"
  - id: "02-02-02"
    choice: "Shaw-style relative position via broadcast multiply"
    why: "Efficient computation of position bias using broadcasting rather than einsum"
metrics:
  duration: "10 min"
  completed: "2026-01-23"
---

# Phase 02 Plan 02: Distance Embedding and Stats Conversion Summary

**One-liner:** Implemented Shaw-style relative position encoding and fixed stats var->std conversion for correct Wav2Vec-BERT normalization.

## Objective

Implement distance_embedding (relative position encoding) and fix stats var->std conversion to ensure correct attention computation and feature normalization in Wav2Vec-BERT.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix stats file var to std conversion | 9d2cb32 | wav2vec_bert.rs |
| 2 | Implement distance_embedding (relative position encoding) | c2e1cc6 | wav2vec_bert.rs |
| 3 | Add unit tests for stats conversion and relative position | 87e3619 | wav2vec_bert.rs |

## Changes Made

### 1. Stats File Var->Std Conversion

Updated `load_stats()` to handle the checkpoint storing variance instead of standard deviation:

```rust
let std = match tensors.get("std") {
    Some(s) => s.clone(),
    None => match tensors.get("var") {
        Some(var) => {
            tracing::info!("[Wav2Vec-BERT] Converting 'var' to 'std' via sqrt()");
            var.sqrt()?
        }
        None => {
            tracing::warn!("[Wav2Vec-BERT] Missing 'std' or 'var', using ones");
            Tensor::ones((HIDDEN_SIZE,), DType::F32, device)?
        }
    }
};
```

### 2. Distance Embedding (Relative Position Encoding)

Added Shaw-style relative position encoding to SelfAttention:

**New constants:**
- `HEAD_DIM = 64` - head dimension for distance embedding
- `LEFT_MAX_POS = 64` - left context window
- `RIGHT_MAX_POS = 8` - right context window
- `NUM_POSITIONS = 73` - total positions (64 + 8 + 1)

**SelfAttention changes:**
- Added `distance_embedding: Option<Tensor>` field [73, 64]
- Load from checkpoint key `{prefix}.distance_embedding.weight`
- `compute_relative_position_bias()` computes position-dependent attention bias:
  1. Create position indices for query and key
  2. Compute relative distance: key_pos - query_pos
  3. Clamp to [-64, 8] range
  4. Shift to positive indices [0, 72]
  5. Lookup embeddings via index_select
  6. Compute bias via broadcast multiply with query
  7. Scale by 1/sqrt(head_dim)

**Forward pass update:**
```rust
let attn_weights = if self.distance_embedding.is_some() {
    let pos_bias = self.compute_relative_position_bias(&query, seq_len)?;
    (attn_weights + pos_bias)?
} else {
    attn_weights
};
```

### 3. Unit Tests Added

- `test_var_to_std_conversion`: Verifies sqrt() conversion (4->2, 9->3, etc.)
- `test_relative_position_indices`: Verifies distance matrix computation
- `test_relative_position_clamping`: Verifies clamp to [-64, 8] range

## Verification Results

```
Build: SUCCESS (no errors)
Tests: 128 passed, 0 failed
Patterns verified:
  - sqrt() in load_stats (line 873)
  - distance_embedding: Option<Tensor> (line 361)
  - broadcast_sub for distance computation (line 482)
  - broadcast_mul for position bias (line 515)
```

## Deviations from Plan

None - plan executed exactly as written.

## Key Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| 02-02-01 | var->std fallback chain | Checkpoint stores variance; graceful degradation if missing |
| 02-02-02 | Broadcast multiply for position bias | More efficient than einsum; clearer code |

## Technical Details

**Distance embedding shape:** [73, 64]
- 73 positions cover relative distances from -64 to +8
- 64 matches head dimension for dot product with query

**Position bias computation:**
```
bias[b, h, i, j] = query[b, h, i, :] dot pos_emb[i, j, :]
                = sum_d query[b, h, i, d] * pos_emb[i, j, d]
```

This adds position-dependent attention bias that helps the model attend based on relative positions, important for speech/audio where temporal relationships matter.

## Next Phase Readiness

Phase 02 is now complete with all Wav2Vec-BERT weight loading components:
- [x] 02-01: ConvModule and FeatureProjection
- [x] 02-02: Distance embedding and stats conversion

Ready for Phase 03 or further integration testing.
