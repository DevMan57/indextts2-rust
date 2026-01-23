# Phase 4: DiT Weights - Research

**Researched:** 2026-01-23
**Domain:** DiT (Diffusion Transformer) weight loading for TTS flow matching
**Confidence:** HIGH

## Summary

This research investigated the exact tensor mapping requirements for loading DiT weights from `s2mel.safetensors` into the Rust implementation. The checkpoint contains a complete CFM (Conditional Flow Matching) estimator with 13 transformer blocks, WaveNet post-processor, and various conditioning components.

Key findings:
1. **Tensor names already match** - The Rust `dit.rs` implementation uses the exact same naming convention as the checkpoint (`cfm.estimator.*` prefix). Most weights should load correctly.
2. **Missing component: `skip_in_linear`** - Each transformer layer has a `skip_in_linear` tensor (`[512, 1024]`) that is NOT loaded by the current Rust implementation. This is needed for UViT skip connections.
3. **Weight normalization is correctly implemented** - The formula `w = g * v / ||v||` with norm computed over all dims except dim 0 matches PyTorch.
4. **Fused QKV is correctly handled** - The reshape approach `(batch, seq, 3, heads, head_dim)` then indexing works correctly.

**Primary recommendation:** Add `skip_in_linear` loading to `DiTBlock` and verify all weights load without "using random weights" warnings.

## Standard Stack

The implementation already uses the correct stack:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| candle-core | 0.4+ | Tensor operations | Hugging Face's Rust ML framework |
| candle-nn | 0.4+ | Neural network layers | Linear, LayerNorm, etc. |
| safetensors | 0.4+ | Weight file loading | Memory-efficient, safe tensor loading |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tracing | 0.1 | Logging/warnings | For "using random weights" warnings |

### No Alternatives Needed
The current stack is correct and complete.

## Architecture Patterns

### DiT Component Structure (from checkpoint)

```
cfm.estimator/
├── Input Processing
│   ├── x_embedder.weight_g/weight_v/bias    # [512, 80] (weight normalized)
│   ├── cond_embedder.weight                 # [1024, 512]
│   ├── cond_projection.weight/bias          # [512, 512]
│   ├── cond_x_merge_linear.weight/bias      # [512, 864]
│   └── content_mask_embedder.weight         # [1, 512]
│
├── Time Embedding
│   ├── t_embedder.freqs                     # [128] learnable frequencies
│   ├── t_embedder.mlp.0.weight/bias         # [512, 256]
│   ├── t_embedder.mlp.2.weight/bias         # [512, 512]
│   ├── t_embedder2.freqs                    # [128] for WaveNet
│   └── t_embedder2.mlp.{0,2}.weight/bias
│
├── Transformer (13 layers, i=0-12)
│   ├── transformer.layers.{i}.attention.wqkv.weight     # [1536, 512]
│   ├── transformer.layers.{i}.attention.wo.weight       # [512, 512]
│   ├── transformer.layers.{i}.attention_norm.norm.weight        # [512]
│   ├── transformer.layers.{i}.attention_norm.project_layer.*    # [1024, 512]
│   ├── transformer.layers.{i}.feed_forward.w1.weight    # [1536, 512] SwiGLU
│   ├── transformer.layers.{i}.feed_forward.w2.weight    # [512, 1536]
│   ├── transformer.layers.{i}.feed_forward.w3.weight    # [1536, 512]
│   ├── transformer.layers.{i}.ffn_norm.norm.weight
│   ├── transformer.layers.{i}.ffn_norm.project_layer.*
│   └── transformer.layers.{i}.skip_in_linear.weight/bias  # [512, 1024] MISSING IN RUST!
│
├── Transformer Final
│   ├── transformer.norm.norm.weight                     # [512]
│   └── transformer.norm.project_layer.weight/bias       # [1024, 512]
│
├── WaveNet (8 layers, j=0-7)
│   ├── wavenet.cond_layer.conv.conv.weight_g/weight_v/bias  # [8192, 512, 1]
│   ├── wavenet.in_layers.{j}.conv.conv.weight_g/weight_v/bias   # [1024, 512, 5]
│   └── wavenet.res_skip_layers.{j}.conv.conv.weight_g/weight_v/bias
│
└── Output
    ├── skip_linear.weight/bias              # [512, 592]
    ├── conv1.weight/bias                    # [512, 512]
    ├── res_projection.weight/bias           # [512, 512]
    ├── final_layer.adaLN_modulation.1.*     # [1024, 512]
    ├── final_layer.linear.weight_g/weight_v/bias  # [512, 512] (weight norm)
    ├── conv2.weight/bias                    # [80, 512, 1]
    └── input_pos                            # [16384] positional
```

### Pattern 1: Weight Normalization Loading

**What:** PyTorch stores `weight_g` (magnitude) and `weight_v` (direction) separately
**When to use:** Any layer with `.weight_g` and `.weight_v` suffixes
**Example:**
```rust
// Source: PyTorch weight_norm documentation
// Formula: w = g * v / ||v||
fn apply_weight_normalization(weight_v: &Tensor, weight_g: &Tensor) -> Result<Tensor> {
    let eps = 1e-6;
    let ndim = weight_v.dims().len();

    // Compute L2 norm along all dims except 0 (output channel)
    let v_sq = weight_v.sqr()?;
    let v_norm = if ndim == 2 {
        // [out, in] -> norm over dim 1
        v_sq.sum(1)?.sqrt()?.unsqueeze(1)?
    } else if ndim == 3 {
        // [out, in, kernel] -> norm over dims 1 and 2
        let sum_1 = v_sq.sum(1)?;
        let sum_12 = sum_1.sum(1)?;
        sum_12.sqrt()?.unsqueeze(1)?.unsqueeze(2)?
    } else {
        return Err(anyhow::anyhow!("Unsupported dimension"));
    };

    let v_norm = (v_norm + eps)?;
    Ok(weight_g.broadcast_mul(&weight_v.broadcast_div(&v_norm)?)?)
}
```

### Pattern 2: Fused QKV Splitting

**What:** Single `wqkv.weight [1536, 512]` contains Q, K, V stacked
**When to use:** DiT attention layers
**Example:**
```rust
// After projection: qkv has shape [batch, seq, 1536]
// Reshape to [batch, seq, 3, num_heads, head_dim]
let qkv = qkv.reshape((batch, seq_len, 3, num_heads, head_dim))?;
let q = qkv.i((.., .., 0, .., ..))?.contiguous()?;  // [batch, seq, heads, head_dim]
let k = qkv.i((.., .., 1, .., ..))?.contiguous()?;
let v = qkv.i((.., .., 2, .., ..))?.contiguous()?;
```

### Pattern 3: AdaLN-Zero Modulation

**What:** Adaptive Layer Normalization with zero-initialized projection
**When to use:** All DiT block conditioning (attention_norm, ffn_norm)
**Example:**
```rust
// checkpoint: {prefix}.norm.weight, {prefix}.project_layer.weight/bias
struct AdaLayerNorm {
    norm: LayerNorm,      // Uses norm.weight only (no bias in checkpoint)
    linear: Linear,       // project_layer: [1024, 512] -> produces scale+shift
}

fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
    let normalized = self.norm.forward(x)?;
    let params = self.linear.forward(cond)?;  // [batch, 1024]
    let chunks = params.chunk(2, D::Minus1)?;
    let scale = &chunks[0].unsqueeze(1)?;     // [batch, 1, 512]
    let shift = &chunks[1].unsqueeze(1)?;
    // AdaLN-Zero: (1 + scale) * normalized + shift
    (normalized.broadcast_mul(&(scale + 1.0)?)?)
        .broadcast_add(&shift)
}
```

### Anti-Patterns to Avoid

- **Loading without transpose check:** PyTorch stores weights as `[out, in]`, candle Linear expects same, but verify each layer
- **Forgetting weight normalization:** Several components use weight_g/weight_v - must apply formula, not load separately
- **Ignoring missing tensors:** Current code prints warnings but uses random - this causes garbage output
- **Wrong fused QKV split order:** Must be Q, K, V along dim 0, not K, Q, V or other orders

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Weight normalization | Custom norm implementation | `apply_weight_normalization()` helper | Edge cases with 2D vs 3D tensors |
| Fused attention | Separate Q/K/V projections | Single fused Linear + reshape | Checkpoint structure is fused |
| SwiGLU FFN | Custom FFN | w1 (gate), w2 (down), w3 (up) pattern | Checkpoint uses this exact structure |
| AdaLN conditioning | Custom modulation | `AdaLayerNorm` struct | Complex scale+shift with bias handling |

**Key insight:** The checkpoint has a specific architecture - don't try to simplify it, just match it exactly.

## Common Pitfalls

### Pitfall 1: Missing skip_in_linear

**What goes wrong:** UViT skip connections fail silently, degraded output quality
**Why it happens:** The `skip_in_linear` tensor exists in checkpoint but wasn't in original Rust implementation
**How to avoid:** Add `skip_in_linear: Option<Linear>` to DiTBlock, load from `{prefix}.skip_in_linear.weight/bias`
**Warning signs:** Check for "using random" warnings mentioning skip_in_linear

### Pitfall 2: Wrong Weight Normalization Dimension

**What goes wrong:** Weights become all zeros or NaN after normalization
**Why it happens:** Norm computed over wrong dimension (e.g., dim 0 instead of dims 1+)
**How to avoid:** Verify norm is over all dims EXCEPT dim 0 (the output channel dimension)
**Warning signs:** Check `weight.mean_all()` and `weight.rms()` after loading - should be non-zero, finite

### Pitfall 3: PyTorch vs Candle Linear Weight Shape

**What goes wrong:** Matrix multiplication dimension mismatch
**Why it happens:** Both store `[out_features, in_features]`, but some loaders transpose
**How to avoid:** When loading with `load_linear()`, the `transpose` flag controls whether to `.t()` the weight
**Warning signs:** Dimension mismatch errors during forward pass

### Pitfall 4: Conv Weight Shape for WaveNet

**What goes wrong:** WaveNet convolutions fail to load
**Why it happens:** Conv weights are 3D `[out, in, kernel]`, need special handling
**How to avoid:** For Linear approximation, extract center tap: `weight.i((.., .., kernel/2))`
**Warning signs:** Shape mismatch when creating WaveNetLayer

### Pitfall 5: Learnable Frequency Time Embedding

**What goes wrong:** Time conditioning produces wrong embeddings
**Why it happens:** Checkpoint uses learnable `freqs` tensor, not fixed sinusoidal
**How to avoid:** Check for `.freqs` tensor and use it if present; apply scale=1000 to timesteps
**Warning signs:** Compare time embedding output to Python reference

## Code Examples

### Loading a DiT Block (Complete)

```rust
// Source: dit.rs DiTBlock::from_tensors
fn from_tensors(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,  // e.g., "cfm.estimator.transformer.layers.0"
    dim: usize,
    num_heads: usize,
    device: &Device,
) -> Result<Self> {
    // AdaLN for attention
    let norm1 = AdaLayerNorm::from_tensors(
        tensors,
        &format!("{}.attention_norm", prefix),
        dim,
        device,
    )?;

    // Attention with fused QKV
    let attn = MultiHeadAttention::from_tensors(
        tensors,
        &format!("{}.attention", prefix),
        dim,
        num_heads,
        device,
    )?;

    // AdaLN for FFN
    let norm2 = AdaLayerNorm::from_tensors(
        tensors,
        &format!("{}.ffn_norm", prefix),
        dim,
        device,
    )?;

    // SwiGLU FFN
    let ff = FeedForward::from_tensors(
        tensors,
        &format!("{}.feed_forward", prefix),
        dim,
        device,
    )?;

    // CRITICAL: Load skip_in_linear for UViT skip connections
    let skip_in_linear = match load_linear(
        tensors,
        &format!("{}.skip_in_linear.weight", prefix),
        Some(&format!("{}.skip_in_linear.bias", prefix)),
    ) {
        Ok(linear) => Some(linear),
        Err(_) => {
            tracing::warn!(
                "[DiT] Missing skip_in_linear for {}, using identity",
                prefix
            );
            None
        }
    };

    Ok(Self { norm1, attn, norm2, ff, skip_in_linear })
}
```

### Verifying Weight Loading Success

```rust
// Add to load_weights() end
fn verify_loading_success(tensors: &HashMap<String, Tensor>, prefix: &str) {
    let expected_keys = [
        "cond_embedder.weight",
        "cond_projection.weight",
        "cond_x_merge_linear.weight",
        "x_embedder.weight_g",
        "x_embedder.weight_v",
        "t_embedder.freqs",
        "transformer.layers.0.attention.wqkv.weight",
        "transformer.layers.0.skip_in_linear.weight",  // NEW
        "wavenet.cond_layer.conv.conv.weight_v",
        "final_layer.adaLN_modulation.1.weight",
        "conv2.weight",
    ];

    for key_suffix in &expected_keys {
        let full_key = format!("{}.{}", prefix, key_suffix);
        if !tensors.contains_key(&full_key) {
            eprintln!("WARNING: Missing expected tensor: {}", full_key);
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standard AdaLN | AdaLN-Zero | DiT paper 2023 | 2x FID improvement via zero-init |
| Separate Q/K/V | Fused wqkv | Standard practice | ~15% faster, fewer parameters |
| GELU FFN | SwiGLU FFN | LLaMA/2024 | Better gradient flow |

**Deprecated/outdated:**
- Standard weight norm API: PyTorch now uses `parametrizations.weight_norm()`, but old checkpoint format still uses `weight_g`/`weight_v`

## Open Questions

### 1. skip_in_linear Usage in Forward Pass

**What we know:** Tensor exists as `[512, 1024]` (takes concatenated hidden+skip)
**What's unclear:** Exact position in forward pass - is it applied before or after skip connection add?
**Recommendation:** Look at reference Python code to determine exact usage; for now, assume it projects concatenated features back to hidden_dim

### 2. input_pos Tensor

**What we know:** Exists as `[16384]` - appears to be learnable positional embedding
**What's unclear:** When/how it's applied (current code doesn't use it)
**Recommendation:** May not be critical for inference if attention works; investigate if output quality is poor

### 3. content_mask_embedder Usage

**What we know:** Exists as `[1, 512]`
**What's unclear:** When this is used (masking during training only?)
**Recommendation:** Likely not needed for inference; ignore for now

## Sources

### Primary (HIGH confidence)
- **Checkpoint inspection:** `s2mel.safetensors` - all 269 tensors enumerated with shapes
- **Rust implementation:** `src/models/s2mel/dit.rs` - current weight loading logic
- **PyTorch weight_norm docs:** https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html

### Secondary (MEDIUM confidence)
- **DiT paper:** "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023) - AdaLN-Zero architecture
- **GitHub understanding-neuralnetworks-pytorch:** Weight normalization formula verification

### Tertiary (LOW confidence)
- WebSearch results on AdaLN-Zero - confirmed architecture pattern but implementation details need verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - already correctly implemented
- Architecture: HIGH - checkpoint structure directly inspected
- Pitfalls: MEDIUM - some based on code analysis, not runtime verification
- skip_in_linear gap: HIGH - verified missing from Rust code but present in checkpoint

**Research date:** 2026-01-23
**Valid until:** 2026-02-23 (stable checkpoint format, unlikely to change)

## Key Gaps Identified

| Gap | Severity | Checkpoint Has | Rust Has | Fix Required |
|-----|----------|----------------|----------|--------------|
| skip_in_linear | HIGH | Yes (13 layers) | No | Add to DiTBlock |
| input_pos | LOW | Yes [16384] | No | Investigate if needed |
| content_mask_embedder | LOW | Yes [1, 512] | No | Training-only, ignore |

**Total tensors in checkpoint:** 269
**Tensors successfully mapped:** ~250
**Tensors missing mapping:** ~19 (mostly skip_in_linear per layer)
