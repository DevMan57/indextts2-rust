# Architecture Research: Weight Name Mapping for TTS Pipeline

**Domain:** TTS Model Weight Loading
**Researched:** 2026-01-23
**Overall Confidence:** HIGH for patterns, MEDIUM for specific implementations

## Executive Summary

This research examines how weight name mapping should be structured in a multi-model TTS pipeline. The codebase already has a working pattern (BigVGAN vocoder) that successfully loads weights from safetensors. The other models (Wav2Vec-BERT, DiT, Conformer, Perceiver) partially implement similar patterns but have gaps.

**Key Finding:** The existing codebase already implements the correct architecture pattern - each model has a dedicated `weights.rs` module or inline `from_tensors()` methods. The issue is incomplete mapping, not architectural.

---

## Weight Name Patterns by Model

### 1. Wav2Vec-BERT 2.0 (HuggingFace format)

**Source:** [HuggingFace Transformers modeling_wav2vec2_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py)

**Confidence:** HIGH (verified from official HuggingFace source)

| HuggingFace Tensor Name | Expected Rust Path | Description |
|------------------------|-------------------|-------------|
| `encoder.layers.{i}.self_attn.linear_q.weight` | `layers[i].attention.query` | Query projection |
| `encoder.layers.{i}.self_attn.linear_k.weight` | `layers[i].attention.key` | Key projection |
| `encoder.layers.{i}.self_attn.linear_v.weight` | `layers[i].attention.value` | Value projection |
| `encoder.layers.{i}.self_attn.linear_out.weight` | `layers[i].attention.output` | Output projection |
| `encoder.layers.{i}.self_attn_layer_norm.weight` | `layers[i].attention_layer_norm` | Pre-attention norm |
| `encoder.layers.{i}.ffn1.intermediate_dense.weight` | `layers[i].ffn1.intermediate` | FFN1 up-proj |
| `encoder.layers.{i}.ffn1.output_dense.weight` | `layers[i].ffn1.output` | FFN1 down-proj |
| `encoder.layers.{i}.ffn1_layer_norm.weight` | `layers[i].ffn1_layer_norm` | Pre-FFN1 norm |
| `encoder.layers.{i}.ffn2.intermediate_dense.weight` | `layers[i].ffn2.intermediate` | FFN2 up-proj |
| `encoder.layers.{i}.ffn2.output_dense.weight` | `layers[i].ffn2.output` | FFN2 down-proj |
| `encoder.layers.{i}.ffn2_layer_norm.weight` | `layers[i].ffn2_layer_norm` | Pre-FFN2 norm |
| `encoder.layers.{i}.final_layer_norm.weight` | `layers[i].final_layer_norm` | Final norm |

**Current Status:** The Rust code in `wav2vec_bert.rs` already expects these exact names. Loading works if checkpoint uses HuggingFace naming.

### 2. DiT (s2mel.safetensors format)

**Source:** Existing `src/models/s2mel/weights.rs` + checkpoint inspection

**Confidence:** MEDIUM (based on existing code analysis)

| Checkpoint Tensor Name | Description |
|----------------------|-------------|
| `cfm.estimator.transformer.layers.{i}.attention.wqkv.weight` | Fused QKV [3*dim, dim] |
| `cfm.estimator.transformer.layers.{i}.attention.wo.weight` | Output projection |
| `cfm.estimator.transformer.layers.{i}.feed_forward.w1.weight` | FFN first (SwiGLU) |
| `cfm.estimator.transformer.layers.{i}.feed_forward.w2.weight` | FFN second |
| `cfm.estimator.transformer.layers.{i}.feed_forward.w3.weight` | FFN gate (SwiGLU) |
| `cfm.estimator.cond_embedder.weight` | Conditioning embedder |
| `cfm.estimator.cond_projection.weight/bias` | Conditioning projection |
| `cfm.estimator.t_embedder.freqs` | Learnable time frequencies |
| `cfm.estimator.t_embedder.mlp.0.weight/bias` | Time MLP layer 1 |
| `cfm.estimator.t_embedder.mlp.2.weight/bias` | Time MLP layer 2 |
| `cfm.estimator.x_embedder.weight_v/weight_g` | Weight-normalized input embed |

**Current Status:** `weights.rs` has `DiTWeights` struct but loading may have issues with fused QKV split and weight normalization handling.

### 3. Conformer (gpt.safetensors format)

**Source:** Existing `conformer.rs` code analysis

**Confidence:** MEDIUM (based on code inspection)

| Checkpoint Tensor Name | Description |
|----------------------|-------------|
| `conditioning_encoder.encoders.{i}.self_attn.linear_q.weight` | Query projection |
| `conditioning_encoder.encoders.{i}.self_attn.linear_k.weight` | Key projection |
| `conditioning_encoder.encoders.{i}.self_attn.linear_v.weight` | Value projection |
| `conditioning_encoder.encoders.{i}.self_attn.linear_out.weight` | Output projection |
| `conditioning_encoder.encoders.{i}.feed_forward.w_1.weight/bias` | FFN layer 1 |
| `conditioning_encoder.encoders.{i}.feed_forward.w_2.weight/bias` | FFN layer 2 |
| `conditioning_encoder.encoders.{i}.conv_module.pointwise_conv1.weight` | Conv pointwise 1 |
| `conditioning_encoder.encoders.{i}.conv_module.depthwise_conv.weight` | Depthwise conv |
| `conditioning_encoder.encoders.{i}.conv_module.pointwise_conv2.weight` | Conv pointwise 2 |
| `conditioning_encoder.encoders.{i}.norm_mha.weight/bias` | MHA layer norm |
| `conditioning_encoder.encoders.{i}.norm_ff.weight/bias` | FFN layer norm |
| `conditioning_encoder.encoders.{i}.norm_conv.weight/bias` | Conv layer norm |
| `conditioning_encoder.encoders.{i}.norm_final.weight/bias` | Final layer norm |

**Current Status:** `from_gpt_tensors()` method exists in `conformer.rs` and attempts to load these tensors.

### 4. Perceiver Resampler (gpt.safetensors format)

**Source:** Existing `perceiver.rs` code analysis

**Confidence:** MEDIUM (based on code inspection)

| Checkpoint Tensor Name | Description |
|----------------------|-------------|
| `perceiver_encoder.latents` | Learned latent queries [32, 1280] |
| `perceiver_encoder.layers.{i}.0.to_q.weight` | Cross-attn query projection |
| `perceiver_encoder.layers.{i}.0.to_kv.weight` | Fused K+V projection [2*dim, ctx_dim] |
| `perceiver_encoder.layers.{i}.0.to_out.weight` | Cross-attn output projection |
| `perceiver_encoder.layers.{i}.1.0.weight/bias` | FFN layer 1 |
| `perceiver_encoder.layers.{i}.1.2.weight/bias` | FFN layer 2 |
| `perceiver_encoder.norm.gamma` | Output normalization |
| `perceiver_encoder.proj_context` | Context projection |

**Current Status:** `from_gpt_tensors()` and `load_from_gpt_tensors()` methods exist and handle the fused KV split.

### 5. BigVGAN (WORKING - Reference Pattern)

**Source:** `src/models/vocoder/weights.rs`

**Confidence:** HIGH (verified working)

Key patterns:
- Uses weight normalization: `weight_v` + `weight_g` -> computed `weight`
- Converts at load time via `apply_weight_norm()`
- Direct tensor name matching with string prefix patterns

---

## Mapping Strategy

### Recommended Pattern: Load-Time Transformation

The existing codebase uses the correct pattern:

```
[Safetensors File]
       |
       v
[HashMap<String, Tensor>]  <-- Raw tensor dictionary
       |
       v
[Transformation Logic]     <-- Weight normalization, fused tensor splitting
       |
       v
[Model-Specific Loader]    <-- from_tensors(), from_gpt_tensors()
       |
       v
[Constructed Model]
```

### Where Mapping Logic Should Live

| Model | Location | Pattern |
|-------|----------|---------|
| Wav2Vec-BERT | `wav2vec_bert.rs` inline | `EncoderLayer::from_tensors()` |
| DiT | `weights.rs` + `dit.rs` inline | `DiTWeights::load()` + `from_tensors()` |
| Conformer | `conformer.rs` inline | `ConformerBlock::from_gpt_tensors()` |
| Perceiver | `perceiver.rs` inline | `PerceiverLayer::from_gpt_tensors()` |
| BigVGAN | `weights.rs` separate | `load_bigvgan_weights()` (converts weight norm) |

**Recommendation:** Keep the current pattern. Each model type has its own loading logic close to the model definition. This is cleaner than a central mapping file because:

1. Model-specific transformations (weight norm, fused QKV) are co-located with model code
2. Easier to maintain - changes to model structure stay with model
3. Matches the working BigVGAN pattern

### Candle VarBuilder Renaming (Alternative)

Candle's VarBuilder supports a `rename_f` method for name remapping:

```rust
let vb = vb.rename_f(|name: &str| {
    name.replace("encoder.layers", "layers")
        .replace("self_attn", "attention")
});
```

**When to use:** Only if tensor names differ by simple string patterns and no structural transformation (like fused weight splitting) is needed.

**Current models don't need this** because they load tensors directly via `safetensors::load()` and do manual matching.

---

## Implementation Order

### Build Order (Dependencies)

```
1. Wav2Vec-BERT  <-- Standalone, no dependencies on other models
2. Conformer     <-- Depends on being loaded alongside GPT
3. Perceiver     <-- Depends on Conformer context
4. DiT           <-- Depends on GPT layer projections
```

### Recommended Fix Order (Risk-Based)

**Phase 1: Validate Current Loading (Day 1)**
1. Add debug logging to print actual tensor keys from each safetensors file
2. Compare against expected keys in each `from_tensors()` method
3. Identify specific mismatches

**Phase 2: Fix Simple Mismatches (Day 1-2)**
1. **Wav2Vec-BERT:** Already has correct HuggingFace naming - verify checkpoint format matches
2. **Conformer:** Minor fixes to `from_gpt_tensors()` if keys don't match

**Phase 3: Fix Complex Transformations (Day 2-3)**
1. **DiT:** Weight normalization for x_embedder, fused QKV handling
2. **Perceiver:** Fused KV splitting, context projection

**Phase 4: Integration Test (Day 3)**
1. Run end-to-end inference
2. Verify non-random encoder outputs
3. Check audio quality

---

## Data Flow: How Weights Flow Through Loading

### Wav2Vec-BERT Loading Flow

```
checkpoints/w2v-bert-2.0/model.safetensors
        |
        v
safetensors::load() --> HashMap<String, Tensor>
        |
        v
SemanticEncoder::load_weights()
        |
        v
for layer_idx in 0..24:
    EncoderLayer::from_tensors(tensors, layer_idx, ...)
        |
        v
    SelfAttention::from_tensors()  --> query, key, value, output Linear
    FeedForward::from_tensors()    --> intermediate, output Linear
    LayerNorm::from tensors        --> weight, bias
        |
        v
self.encoder_layers.push(layer)
```

### GPT (Conformer + Perceiver) Loading Flow

```
checkpoints/gpt.safetensors
        |
        v
safetensors::load() --> HashMap<String, Tensor>
        |
        +---> ConformerEncoder::load_weights()
        |         |
        |         v
        |     detect "conditioning_encoder" prefix
        |         |
        |         v
        |     load_from_gpt_tensors()
        |         |
        |         v
        |     for block_idx in 0..num_blocks:
        |         ConformerBlock::from_gpt_tensors()
        |
        +---> PerceiverResampler::load_weights()
                  |
                  v
              detect "perceiver_encoder" prefix
                  |
                  v
              load_from_gpt_tensors()
                  |
                  v
              Load latents tensor directly
              for layer_idx in 0..num_layers:
                  PerceiverLayer::from_gpt_tensors()
```

### DiT Loading Flow

```
checkpoints/s2mel.safetensors
        |
        v
load_s2mel_safetensors() --> HashMap<String, Tensor>
        |
        v
DiffusionTransformer::load_weights()
        |
        +---> TimestepEmbedding::from_tensors()  (t_embedder.*)
        +---> AdaLayerNorm::from_tensors()       (*.norm.*, *.project_layer.*)
        +---> load_weight_normalized_linear()    (x_embedder.weight_v/g)
        +---> DiTBlock::from_tensors()           (transformer.layers.*)
                  |
                  v
              Attention: load fused wqkv, split into Q, K, V
              FFN: load w1, w2, w3 for SwiGLU
              AdaLN: load norm weights + projection
```

---

## Critical Implementation Notes

### 1. PyTorch vs Candle Weight Format

**PyTorch Linear:** Stores weights as `[out_features, in_features]`
**Candle Linear:** Expects same format, handles transpose internally

**Action:** Do NOT transpose when loading unless specifically needed for reshape operations.

Current code sometimes transposes unnecessarily. Check each `load_linear()` call.

### 2. Weight Normalization Conversion

BigVGAN handles this correctly:

```rust
// weight_v: [out, in, kernel] or [out, in]
// weight_g: [out, 1, 1] or [out, 1]
// result: weight_g * (weight_v / ||weight_v||_2)
fn apply_weight_norm(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor>
```

DiT's x_embedder uses weight normalization - ensure it's applied.

### 3. Fused QKV Splitting

For DiT: `wqkv.weight` is `[3*hidden_dim, hidden_dim]`
For Perceiver: `to_kv.weight` is `[2*dim, context_dim]`

Split along dim 0:
```rust
let q = wqkv.i((0..dim, ..))?;
let k = wqkv.i((dim..2*dim, ..))?;
let v = wqkv.i((2*dim..3*dim, ..))?;
```

### 4. Missing Keys Fallback

Current code falls back to random weights when keys missing. This masks loading failures.

**Recommendation:** Add a "strict mode" that fails on missing keys during development.

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Wav2Vec-BERT naming | HIGH | Verified from HuggingFace transformers source |
| DiT naming | MEDIUM | Based on existing weights.rs, not checkpoint verified |
| Conformer naming | MEDIUM | Based on code analysis, has from_gpt_tensors() |
| Perceiver naming | MEDIUM | Based on code analysis, has load_from_gpt_tensors() |
| BigVGAN (reference) | HIGH | Currently working in production |
| VarBuilder rename | HIGH | Documented in candle-nn docs |

---

## Sources

- [HuggingFace Wav2Vec2-BERT modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py)
- [Candle VarBuilder documentation](https://docs.rs/candle-nn/latest/candle_nn/var_builder/index.html)
- [Facebook DiT Repository](https://github.com/facebookresearch/DiT)
- [HuggingFace Safetensors documentation](https://huggingface.co/docs/safetensors/main/en/index)
- Local codebase analysis: `src/models/*/weights.rs`, `src/models/*/*.rs`
