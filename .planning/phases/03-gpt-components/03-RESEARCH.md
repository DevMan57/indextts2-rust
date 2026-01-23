# Phase 3: GPT Components - Research

**Researched:** 2026-01-23
**Domain:** Conformer encoder and Perceiver resampler weight loading from gpt.safetensors
**Confidence:** HIGH

## Summary

This phase fixes weight loading for the Conformer encoder (6 layers) and Perceiver resampler (2 layers) from gpt.safetensors. The existing Rust implementation has the right structure but is using random initialization fallback for several components. Research confirms the checkpoint format matches the code's expected structure with minor adjustments needed.

Key findings:
1. **Conformer loads correctly** - The existing `load_from_gpt_tensors()` in conformer.rs matches checkpoint keys
2. **Perceiver has architecture mismatch** - FFN uses SwiGLU (3412->1706->1280), not standard GELU (5120->1280)
3. **Relative position attention** - Conformer uses pos_bias_u/v [8, 64] and linear_pos [512, 512]
4. **Missing proj_context** - Perceiver needs context projection before cross-attention
5. **Emotion encoders** - Additional 4-layer emo_conditioning_encoder and 2-layer emo_perceiver_encoder

**Primary recommendation:** Fix Perceiver FFN to use SwiGLU activation with correct dimensions. Add proj_context loading. The Conformer is mostly working but needs to load the relative position components.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| candle-core | 0.x | Tensor operations | Already in use, handles shapes well |
| candle-nn | 0.x | Linear, LayerNorm | Already in use throughout codebase |
| safetensors | 0.x | Weight loading | Already loading gpt.safetensors |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tracing | 0.1 | Warning logs | Phase 1 decision for fallback warnings |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual SwiGLU | candle-nn GELU | Must use SwiGLU per checkpoint |

**Installation:**
Already available - no new dependencies needed.

## Architecture Patterns

### Recommended Weight Loading Order

```
1. Conformer (6 layers)
   conditioning_encoder.encoders.{0-5}.*
   - Already mostly working, needs pos_bias_u/v loading

2. Perceiver (2 layers)
   perceiver_encoder.latents [32, 1280]
   perceiver_encoder.proj_context.weight [1280, 512]
   perceiver_encoder.layers.{0-1}.*
   perceiver_encoder.norm.gamma [1280]
   - Needs FFN architecture fix (SwiGLU)
   - Needs proj_context loading

3. Emotion Conformer (4 layers) - Optional
   emo_conditioning_encoder.encoders.{0-3}.*

4. Emotion Perceiver (2 layers) - Optional
   emo_perceiver_encoder.*
```

### Pattern 1: Conformer Layer Key Mapping

**What:** Complete key mapping for conditioning_encoder layer weights
**When to use:** Loading Conformer encoder blocks

```
Checkpoint Key                                          | Rust Field
--------------------------------------------------------|------------------
conditioning_encoder.encoders.{i}.self_attn.linear_q.weight    | attention.q_proj
conditioning_encoder.encoders.{i}.self_attn.linear_k.weight    | attention.k_proj
conditioning_encoder.encoders.{i}.self_attn.linear_v.weight    | attention.v_proj
conditioning_encoder.encoders.{i}.self_attn.linear_out.weight  | attention.out_proj
conditioning_encoder.encoders.{i}.self_attn.linear_pos.weight  | attention.linear_pos (NEW)
conditioning_encoder.encoders.{i}.self_attn.pos_bias_u         | attention.pos_bias_u (NEW)
conditioning_encoder.encoders.{i}.self_attn.pos_bias_v         | attention.pos_bias_v (NEW)
conditioning_encoder.encoders.{i}.norm_mha.weight              | attention.layer_norm
conditioning_encoder.encoders.{i}.feed_forward.w_1.weight      | ff1.linear1
conditioning_encoder.encoders.{i}.feed_forward.w_2.weight      | ff1.linear2
conditioning_encoder.encoders.{i}.norm_ff.weight               | ff1.layer_norm
conditioning_encoder.encoders.{i}.conv_module.pointwise_conv1.weight | conv.pointwise_conv1
conditioning_encoder.encoders.{i}.conv_module.depthwise_conv.weight  | conv.depthwise_conv_weight
conditioning_encoder.encoders.{i}.conv_module.pointwise_conv2.weight | conv.pointwise_conv2
conditioning_encoder.encoders.{i}.conv_module.norm.weight      | conv.layer_norm
conditioning_encoder.encoders.{i}.norm_conv.weight             | conv.layer_norm (duplicate)
conditioning_encoder.encoders.{i}.norm_final.weight            | final_layer_norm
```

### Pattern 2: Perceiver Architecture (CRITICAL FIX NEEDED)

**What:** Perceiver uses asymmetric attention and SwiGLU FFN
**When to use:** Loading perceiver_encoder weights

**Current Rust (WRONG):**
```rust
// Current code assumes:
// - dim = 1280 (latent dim) for all projections
// - FFN expansion 4x (5120)
// - Standard GELU activation
```

**Checkpoint Reality:**
```
perceiver_encoder.latents: [32, 1280]              // Latent queries
perceiver_encoder.proj_context.weight: [1280, 512] // Project context 512 -> 1280
perceiver_encoder.layers.{i}.0.to_q.weight: [512, 1280]   // Q from latents: 1280 -> 512
perceiver_encoder.layers.{i}.0.to_kv.weight: [1024, 1280] // K+V from context: 1280 -> 1024 (fused)
perceiver_encoder.layers.{i}.0.to_out.weight: [1280, 512] // Output: 512 -> 1280
perceiver_encoder.layers.{i}.1.0.weight: [3412, 1280]     // FFN1: 1280 -> 3412 (SwiGLU gate)
perceiver_encoder.layers.{i}.1.2.weight: [1280, 1706]     // FFN2: 1706 -> 1280
perceiver_encoder.norm.gamma: [1280]                // Final norm (gamma only, no beta!)
```

**SwiGLU FFN Architecture:**
```rust
// SwiGLU: split first linear output, apply SiLU to half, multiply
fn swiglu_forward(&self, x: &Tensor) -> Result<Tensor> {
    let hidden = self.linear1.forward(x)?;  // [batch, seq, 3412]
    let (gate, up) = hidden.chunk(2, 2)?;   // Each [batch, seq, 1706]
    let gate = candle_nn::ops::silu(&gate)?;
    let hidden = (gate * up)?;               // [batch, seq, 1706]
    self.linear2.forward(&hidden)            // [batch, seq, 1280]
}
```

### Pattern 3: Perceiver Cross-Attention Flow

**What:** Query from latents, K/V from projected context
**When to use:** Implementing Perceiver attention forward pass

```rust
fn cross_attention(&self, latents: &Tensor, context: &Tensor) -> Result<Tensor> {
    // latents: [batch, 32, 1280]
    // context: [batch, seq, 512]

    // 1. Project context to latent dimension
    let context = self.proj_context.forward(context)?;  // [batch, seq, 1280]

    // 2. Q from latents (1280 -> 512)
    let q = self.to_q.forward(latents)?;  // [batch, 32, 512]

    // 3. K+V from context (1280 -> 1024, split to 512 each)
    let kv = self.to_kv.forward(&context)?;  // [batch, seq, 1024]
    let (k, v) = kv.chunk(2, 2)?;            // Each [batch, seq, 512]

    // 4. Attention: Q @ K.T -> [batch, heads, 32, seq]
    // Note: head_dim = 512 / num_heads (likely 8 heads, 64 dim each)

    // 5. Output projection (512 -> 1280)
    let out = self.to_out.forward(&attn_output)?;  // [batch, 32, 1280]

    latents + out  // Residual connection
}
```

### Pattern 4: Conformer Relative Position Attention

**What:** Shaw-style relative position bias for self-attention
**When to use:** Conformer attention layers

**Components:**
- `linear_pos.weight: [512, 512]` - Position encoding projection
- `pos_bias_u: [8, 64]` - Per-head content bias (num_heads=8, head_dim=64)
- `pos_bias_v: [8, 64]` - Per-head position bias

```rust
fn relative_attention(&self, x: &Tensor, pos_enc: &Tensor) -> Result<Tensor> {
    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    // Add content and position biases
    let q_with_bias_u = q + pos_bias_u;  // Content-dependent
    let q_with_bias_v = q + pos_bias_v;  // Position-dependent

    // Position encoding
    let pos = self.linear_pos.forward(pos_enc)?;

    // Compute attention: content-content + content-position
    let ac = q_with_bias_u.matmul(&k.t())?;      // Content-content
    let bd = q_with_bias_v.matmul(&pos.t())?;    // Content-position

    // Combine and scale
    let scores = (ac + bd) / sqrt(head_dim);
    softmax(scores).matmul(v)
}
```

### Anti-Patterns to Avoid

- **Assuming symmetric attention dims:** Perceiver uses asymmetric dims (Q: 512, K/V: 512, but latents: 1280)
- **Ignoring SwiGLU:** The FFN is NOT standard expansion, it uses SwiGLU with 2x gate dimension
- **Missing proj_context:** Context MUST be projected to 1280 before attention
- **Forgetting gamma-only norm:** Perceiver final norm has no beta (LayerNorm::new_no_bias)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LayerNorm (gamma-only) | Custom norm | `LayerNorm::new(gamma, zeros, eps)` | Set bias to zeros |
| SwiGLU | Manual split+silu | Copy pattern from DiT code | Already implemented in s2mel |
| Fused K/V split | Manual indexing | `tensor.chunk(2, dim)` | Candle handles this |
| Conv1d kernel=1 | Conv1d layer | Linear after reshape | Simpler for kernel=1 |

**Key insight:** The s2mel module already implements SwiGLU for the DiT model. Reuse that pattern for Perceiver FFN.

## Common Pitfalls

### Pitfall 1: Perceiver FFN Dimension Mismatch
**What goes wrong:** Assuming 4x expansion (1280 -> 5120 -> 1280) when it's actually SwiGLU (1280 -> 3412 -> 1706 -> 1280)
**Why it happens:** Standard transformer FFN uses 4x expansion
**How to avoid:** Load actual weight shapes, use 3412/2 = 1706 for SwiGLU gate
**Warning signs:** Shape mismatch errors on FFN forward pass

### Pitfall 2: Missing Context Projection
**What goes wrong:** Passing 512-dim context directly to attention expecting 1280-dim
**Why it happens:** Conformer output is 512-dim, Perceiver latents are 1280-dim
**How to avoid:** Load and apply proj_context before cross-attention
**Warning signs:** Broadcasting errors, wrong attention scores

### Pitfall 3: Gamma-Only LayerNorm
**What goes wrong:** Looking for `norm.bias` which doesn't exist
**Why it happens:** Perceiver final norm has only `norm.gamma`, no beta
**How to avoid:** Check key exists, use zeros for missing beta
**Warning signs:** "Key not found: perceiver_encoder.norm.beta"

### Pitfall 4: Relative Position Not Implemented
**What goes wrong:** Conformer attention ignores pos_bias_u/v, produces wrong output
**Why it happens:** Current code loads Q/K/V but not relative position components
**How to avoid:** Add fields for linear_pos, pos_bias_u, pos_bias_v; apply in forward
**Warning signs:** Conformer output statistics don't match Python implementation

### Pitfall 5: Wrong num_heads for Emotion Encoder
**What goes wrong:** Using 8 heads when emotion encoder has 4 heads
**Why it happens:** Main conformer has 8 heads (512/8=64), emotion has 4 heads (512/4=128)
**How to avoid:** Check pos_bias_u shape to infer num_heads
**Warning signs:** Shape mismatch in attention computation

## Code Examples

Verified patterns from checkpoint analysis:

### Loading Perceiver Weights with Correct Dimensions
```rust
// Source: Checkpoint analysis of perceiver_encoder keys
fn load_perceiver_layer(
    tensors: &HashMap<String, Tensor>,
    layer_idx: usize,
    device: &Device,
) -> Result<PerceiverLayer> {
    let prefix = format!("perceiver_encoder.layers.{}", layer_idx);

    // Cross-attention (asymmetric dimensions)
    let to_q = load_linear(tensors, &format!("{}.0.to_q.weight", prefix), None)?;
    // to_q: [512, 1280] -> 1280-dim latents to 512-dim queries

    let to_kv = tensors.get(&format!("{}.0.to_kv.weight", prefix))
        .ok_or_else(|| anyhow!("to_kv not found"))?;
    let (kv_out, kv_in) = to_kv.dims2()?;  // [1024, 1280]
    let k_weight = to_kv.i((0..kv_out/2, ..))?.contiguous()?;  // [512, 1280]
    let v_weight = to_kv.i((kv_out/2..kv_out, ..))?.contiguous()?;  // [512, 1280]
    let to_k = Linear::new(k_weight, None);
    let to_v = Linear::new(v_weight, None);

    let to_out = load_linear(tensors, &format!("{}.0.to_out.weight", prefix), None)?;
    // to_out: [1280, 512] -> 512-dim attention output to 1280-dim latents

    // SwiGLU FFN
    let ffn1 = load_linear(
        tensors,
        &format!("{}.1.0.weight", prefix),
        Some(&format!("{}.1.0.bias", prefix)),
    )?;  // [3412, 1280]

    let ffn2 = load_linear(
        tensors,
        &format!("{}.1.2.weight", prefix),
        Some(&format!("{}.1.2.bias", prefix)),
    )?;  // [1280, 1706]

    Ok(PerceiverLayer { to_q, to_k, to_v, to_out, ffn1, ffn2 })
}
```

### SwiGLU Forward Pass
```rust
// Source: Based on DiT implementation pattern
fn swiglu(&self, x: &Tensor) -> Result<Tensor> {
    let hidden = self.ffn1.forward(x)?;  // [batch, seq, 3412]
    let chunks = hidden.chunk(2, 2)?;    // Split dim 2
    let gate = &chunks[0];               // [batch, seq, 1706]
    let up = &chunks[1];                 // [batch, seq, 1706]
    let gate = candle_nn::ops::silu(gate)?;
    let hidden = (gate * up)?;
    self.ffn2.forward(&hidden)           // [batch, seq, 1280]
}
```

### Loading proj_context
```rust
// Source: Checkpoint key analysis
fn load_perceiver(tensors: &HashMap<String, Tensor>, device: &Device) -> Result<Perceiver> {
    // Load latents
    let latents = tensors.get("perceiver_encoder.latents")
        .ok_or_else(|| anyhow!("latents not found"))?
        .unsqueeze(0)?;  // [32, 1280] -> [1, 32, 1280]

    // Load proj_context: 512 -> 1280
    let proj_context = load_linear(
        tensors,
        "perceiver_encoder.proj_context.weight",
        Some("perceiver_encoder.proj_context.bias"),
    )?;

    // Load final norm (gamma only!)
    let gamma = tensors.get("perceiver_encoder.norm.gamma")
        .ok_or_else(|| anyhow!("norm.gamma not found"))?;
    let beta = Tensor::zeros_like(gamma)?;
    let final_norm = LayerNorm::new(gamma.clone(), beta, 1e-5);

    // Load layers...
}
```

### Conformer Relative Position Loading
```rust
// Source: Checkpoint key analysis
fn load_attention_with_relative_pos(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    dim: usize,
    num_heads: usize,
    device: &Device,
) -> Result<MultiHeadAttention> {
    // Standard Q/K/V/Out projections
    let q_proj = load_linear(tensors, &format!("{}.linear_q.weight", prefix),
                             Some(&format!("{}.linear_q.bias", prefix)))?;
    // ... k, v, out ...

    // Relative position components (NEW)
    let linear_pos = load_linear(tensors, &format!("{}.linear_pos.weight", prefix), None)?;

    let pos_bias_u = tensors.get(&format!("{}.pos_bias_u", prefix))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("[Conformer] Missing pos_bias_u, using zeros");
            Tensor::zeros((num_heads, dim / num_heads), DType::F32, device).unwrap()
        });

    let pos_bias_v = tensors.get(&format!("{}.pos_bias_v", prefix))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("[Conformer] Missing pos_bias_v, using zeros");
            Tensor::zeros((num_heads, dim / num_heads), DType::F32, device).unwrap()
        });

    Ok(MultiHeadAttention {
        q_proj, k_proj, v_proj, out_proj,
        linear_pos,
        pos_bias_u,
        pos_bias_v,
        num_heads,
        head_dim: dim / num_heads,
    })
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate K/V projections | Fused K/V weight | Perceiver IO 2023 | Single matmul, faster |
| GELU FFN | SwiGLU FFN | LLaMA (2023) | Better quality, adopted by many |
| Full LayerNorm | RMSNorm / gamma-only | Various 2023 | Simpler, same quality |
| Absolute position | Relative position bias | Conformer (2020) | Better for variable length |

**Deprecated/outdated:**
- Assuming 4x FFN expansion: Many modern models use SwiGLU with 2.7x expansion
- Symmetric attention dimensions: Perceiver-style uses asymmetric Q vs K/V dims

## Open Questions

Things that couldn't be fully resolved:

1. **Emotion encoder integration**
   - What we know: emo_conditioning_encoder (4 layers) and emo_perceiver_encoder exist
   - What's unclear: How they're combined with main path in inference
   - Recommendation: Load weights, defer integration to later phase if not blocking

2. **Exact num_heads for Perceiver attention**
   - What we know: Q is 512-dim, latents are 1280-dim
   - What's unclear: Whether it's 8 heads (64 dim) or different
   - Recommendation: Try 8 heads first (matches Conformer), validate with output

3. **Conformer input embedding**
   - What we know: Uses 2D conv + large linear (261632 -> 512)
   - What's unclear: How input mel features map to this
   - Recommendation: Keep using random input_proj for now (current approach)

## Sources

### Primary (HIGH confidence)
- Local checkpoint inspection: `checkpoints/gpt.safetensors` - All tensor shapes verified
- Existing codebase: `src/models/gpt/conformer.rs` - Working depthwise conv pattern
- Existing codebase: `src/models/gpt/perceiver.rs` - Current implementation (needs fixes)

### Secondary (MEDIUM confidence)
- Existing codebase: `src/models/s2mel/` - SwiGLU implementation for DiT
- Phase 2 research: Relative position encoding patterns

### Tertiary (LOW confidence)
- General Perceiver IO paper patterns - May differ from this specific implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing codebase patterns
- Conformer architecture: HIGH - Checkpoint keys match existing code
- Perceiver architecture: HIGH - Verified all tensor shapes, identified issues
- Pitfalls: HIGH - Based on checkpoint analysis, not assumptions

**Research date:** 2026-01-23
**Valid until:** 60 days (architecture is stable, checkpoint format won't change)

---

## Appendix: Complete Checkpoint Key Reference

### conditioning_encoder (6 layers)
```
conditioning_encoder.after_norm.weight/bias: [512]
conditioning_encoder.embed.conv.0.weight: [512, 1, 3, 3]
conditioning_encoder.embed.out.0.weight: [512, 261632]
conditioning_encoder.embed.pos_enc.pe: [1, 5000, 512]
conditioning_encoder.encoders.{0-5}.self_attn.linear_q/k/v.weight: [512, 512]
conditioning_encoder.encoders.{0-5}.self_attn.linear_out.weight: [512, 512]
conditioning_encoder.encoders.{0-5}.self_attn.linear_pos.weight: [512, 512]
conditioning_encoder.encoders.{0-5}.self_attn.pos_bias_u/v: [8, 64]
conditioning_encoder.encoders.{0-5}.feed_forward.w_1.weight: [2048, 512]
conditioning_encoder.encoders.{0-5}.feed_forward.w_2.weight: [512, 2048]
conditioning_encoder.encoders.{0-5}.conv_module.pointwise_conv1.weight: [1024, 512, 1]
conditioning_encoder.encoders.{0-5}.conv_module.depthwise_conv.weight: [512, 1, 15]
conditioning_encoder.encoders.{0-5}.conv_module.pointwise_conv2.weight: [512, 512, 1]
conditioning_encoder.encoders.{0-5}.norm_mha/ff/conv/final.weight/bias: [512]
```

### perceiver_encoder (2 layers)
```
perceiver_encoder.latents: [32, 1280]
perceiver_encoder.proj_context.weight: [1280, 512]
perceiver_encoder.proj_context.bias: [1280]
perceiver_encoder.layers.{0-1}.0.to_q.weight: [512, 1280]
perceiver_encoder.layers.{0-1}.0.to_kv.weight: [1024, 1280]
perceiver_encoder.layers.{0-1}.0.to_out.weight: [1280, 512]
perceiver_encoder.layers.{0-1}.1.0.weight: [3412, 1280]
perceiver_encoder.layers.{0-1}.1.0.bias: [3412]
perceiver_encoder.layers.{0-1}.1.2.weight: [1280, 1706]
perceiver_encoder.layers.{0-1}.1.2.bias: [1280]
perceiver_encoder.norm.gamma: [1280]
```

### emo_conditioning_encoder (4 layers)
```
emo_conditioning_encoder.encoders.{0-3}.*  (same pattern as main, 4 heads instead of 8)
pos_bias_u/v shape: [4, 128] (4 heads, 128 dim each)
feed_forward.w_1: [1024, 512] (2x expansion, not 4x)
```

### emo_perceiver_encoder (2 layers)
```
emo_perceiver_encoder.latents: [1, 1024]
emo_perceiver_encoder.proj_context.weight: [1024, 512]
emo_perceiver_encoder.layers.{0-1}.0.to_q.weight: [256, 1024]
emo_perceiver_encoder.layers.{0-1}.0.to_kv.weight: [512, 1024]
emo_perceiver_encoder.layers.{0-1}.0.to_out.weight: [1024, 256]
emo_perceiver_encoder.layers.{0-1}.1.0.weight: [2730, 1024]
emo_perceiver_encoder.layers.{0-1}.1.2.weight: [1024, 1365]
emo_perceiver_encoder.norm.gamma: [1024]
```
