# Phase 2: Wav2Vec-BERT Weights - Research

**Researched:** 2026-01-23
**Domain:** Wav2Vec-BERT 2.0 semantic encoder weight loading, Conformer conv modules, relative position encoding
**Confidence:** HIGH

## Summary

This phase implements missing components in the Wav2Vec-BERT semantic encoder: conv_module (Conformer convolution), feature_projection, and distance_embedding (relative position encoding). Research confirms that the existing codebase already has working patterns for depthwise separable convolutions and GLU activation in `src/models/gpt/conformer.rs` that can be directly reused.

The checkpoint uses HuggingFace Wav2Vec2-BERT format with `relative_key` position embeddings (Shaw-style). The distance_embedding shape [73, 64] derives from: 73 = left_max_positions(64) + right_max_positions(8) + 1, and 64 = head_dim (1024/16 heads).

**Primary recommendation:** Implement missing components by adapting existing conformer.rs patterns. Use `Tensor::conv1d()` with `groups=channels` for depthwise conv. Convert stats file `var` to `std` at load time with `sqrt()`.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| candle-core | 0.x | Tensor operations, conv1d | Native Rust ML, already in use |
| candle-nn | 0.x | LayerNorm, Linear | Already in use throughout codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| safetensors | 0.x | Weight loading | Already loading checkpoint |
| tracing | 0.1 | Warning logs | Phase 1 decision for fallback warnings |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| candle_nn::Conv1d | Tensor::conv1d() | Both work; Tensor::conv1d() is simpler for depthwise since we need groups param directly |

**Installation:**
Already available - no new dependencies needed.

## Architecture Patterns

### Wav2Vec-BERT Conformer Conv Module Structure

The checkpoint contains per-layer conv_module with these tensors:

```
encoder.layers.{i}.conv_module.layer_norm.weight         [1024]
encoder.layers.{i}.conv_module.layer_norm.bias           [1024]
encoder.layers.{i}.conv_module.pointwise_conv1.weight    [2048, 1024, 1]
encoder.layers.{i}.conv_module.depthwise_conv.weight     [1024, 1, 31]
encoder.layers.{i}.conv_module.depthwise_layer_norm.weight [1024]
encoder.layers.{i}.conv_module.depthwise_layer_norm.bias   [1024]
encoder.layers.{i}.conv_module.pointwise_conv2.weight    [1024, 1024, 1]
```

**Processing flow (from HuggingFace source):**
1. LayerNorm
2. Pointwise conv1 (expand 1024 -> 2048 for GLU)
3. GLU activation (splits 2048 -> 1024, applies sigmoid gating)
4. Depthwise conv (kernel=31, groups=1024)
5. Depthwise LayerNorm (instead of BatchNorm)
6. Swish activation
7. Pointwise conv2 (1024 -> 1024)

### Relative Position Encoding (distance_embedding)

Shape: [73, 64] per layer at `encoder.layers.{i}.self_attn.distance_embedding.weight`

**Configuration values (from facebook/w2v-bert-2.0/config.json):**
- left_max_position_embeddings: 64
- right_max_position_embeddings: 8
- num_positions = 64 + 8 + 1 = 73
- head_dim = 1024 / 16 = 64

**Algorithm (Shaw-style relative positions):**
```python
# Create position indices
position_ids_l = torch.arange(query_length).view(-1, 1)  # [Q, 1]
position_ids_r = torch.arange(key_length).view(1, -1)    # [1, K]

# Compute relative distances
distance = position_ids_r - position_ids_l              # [Q, K]
distance = clamp(distance, -64, 8)                       # Clip to valid range

# Look up embeddings
positional_embedding = distance_embedding(distance + 64) # [Q, K, 64]

# Apply to attention scores via einsum
relative_bias = einsum("bhld,lrd->bhlr", query, positional_embedding)
scores = scores + (relative_bias / sqrt(head_dim))
```

### Feature Projection (Global)

```
feature_projection.layer_norm.weight    [160]
feature_projection.layer_norm.bias      [160]
feature_projection.projection.weight    [1024, 160]
feature_projection.projection.bias      [1024]
```

This projects 160-dim input features to 1024-dim hidden states. Located at the model input before encoder layers.

### Recommended Project Structure Updates

```
src/models/semantic/wav2vec_bert.rs
    Add to EncoderLayer:
    ├── conv_module: ConvModule     # New struct
    └── distance_embedding: Tensor  # [73, 64] per layer

    Add global components:
    ├── feature_projection: FeatureProjection  # New struct
    └── masked_spec_embed: Option<Tensor>       # [1024] (optional)
```

### Pattern 1: Depthwise Separable Convolution (EXISTING in codebase)

**What:** Depthwise conv with groups=channels, already implemented in conformer.rs
**When to use:** Conformer conv_module in Wav2Vec-BERT
**Source:** `src/models/gpt/conformer.rs` lines 497-505

```rust
// Depthwise conv: need to transpose to (batch, channels, seq)
let x = x.transpose(1, 2)?;
let padding = self.kernel_size / 2;
let x = x.conv1d(
    &self.depthwise_conv_weight,
    padding,
    1, // stride
    1, // dilation
    x.dim(1)?, // groups = channels (depthwise)
)?;
// Add bias
let bias = self.depthwise_conv_bias.unsqueeze(0)?.unsqueeze(2)?;
let x = x.broadcast_add(&bias)?;
```

### Pattern 2: GLU Activation (EXISTING in codebase)

**What:** Gated Linear Unit - split tensor, apply sigmoid to half, multiply
**When to use:** After pointwise_conv1 in conv_module
**Source:** `src/models/gpt/conformer.rs` lines 85-91

```rust
/// GLU (Gated Linear Unit) activation
fn glu(x: &Tensor, dim: usize) -> Result<Tensor> {
    let chunks = x.chunk(2, dim)?;
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = candle_nn::ops::sigmoid(b)?;
    (a * gate).map_err(Into::into)
}
```

### Pattern 3: Pointwise Conv as Linear (EXISTING in codebase)

**What:** Conv1d with kernel=1 can be loaded as Linear by reshaping
**When to use:** pointwise_conv1, pointwise_conv2
**Source:** `src/models/gpt/conformer.rs` lines 400-414

```rust
// Pointwise conv1 weight is stored as [out_channels, in_channels, 1]
// We need to reshape it to Linear format
let pw1_key = format!("{}.pointwise_conv1.weight", prefix);
let pointwise_conv1 = if let Some(weight) = tensors.get(&pw1_key) {
    // Weight is [out_channels, in_channels, 1] - reshape to [out, in]
    let (out_ch, in_ch, _k) = weight.dims3()?;
    let weight = weight.reshape((out_ch, in_ch))?;
    let bias = tensors.get(&format!("{}.pointwise_conv1.bias", prefix)).cloned();
    Linear::new(weight, bias)
} else {
    // Fallback to random
    ...
}
```

### Anti-Patterns to Avoid

- **Hardcoding F32:** Do NOT use `DType::F32` in places that would block FP16 conversion. Load in native dtype and allow conversion.
- **BatchNorm in inference:** Wav2Vec-BERT uses LayerNorm (depthwise_layer_norm), not BatchNorm. Don't add unnecessary BatchNorm.
- **Missing padding for depthwise conv:** Always use `padding = kernel_size / 2` for same-size output.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Depthwise conv | Custom loop over channels | `Tensor::conv1d()` with `groups=channels` | Candle handles grouping natively |
| GLU activation | Manual split + sigmoid | Copy pattern from conformer.rs | Already tested and working |
| LayerNorm | Custom mean/std calculation | `candle_nn::LayerNorm` | Handles numerical stability |
| Position embedding lookup | Manual indexing | Embedding addition to attention | Simpler einsum-based approach |
| var to std conversion | Custom normalization | `var.sqrt()` at load time | Single operation, Phase 1 decision |

**Key insight:** The conformer.rs implementation already solved all these problems for the GPT conditioning encoder. Reuse those patterns directly.

## Common Pitfalls

### Pitfall 1: Tensor Layout for Conv1d
**What goes wrong:** Candle conv1d expects [batch, channels, seq], but attention outputs are [batch, seq, channels]
**Why it happens:** Different conventions between Linear (batch-last) and Conv (channels-middle)
**How to avoid:** Always transpose before conv1d, transpose back after
**Warning signs:** Shape mismatch errors mentioning wrong dimensions

### Pitfall 2: Pointwise Conv Bias Not Present
**What goes wrong:** Checkpoint has no bias for pointwise convs, code expects bias
**Why it happens:** HuggingFace Wav2Vec-BERT pointwise convs are bias=False
**How to avoid:** Check if bias tensor exists, use `None` if missing
**Warning signs:** "Key not found" errors for `.bias` keys

### Pitfall 3: FP16 LayerNorm Overflow
**What goes wrong:** LayerNorm can overflow with FP16 due to limited dynamic range (max 65504)
**Why it happens:** Intermediate variance calculations may exceed FP16 range
**How to avoid:** Keep LayerNorm in F32 internally (Candle handles this), or use BF16
**Warning signs:** NaN values, exploding activations

### Pitfall 4: Wrong Relative Position Indexing
**What goes wrong:** Off-by-one or wrong clamp range for distance embedding
**Why it happens:** Forgetting to add left_max_position to shift negative indices
**How to avoid:** Use formula: `index = clamp(distance, -64, 8) + 64`
**Warning signs:** Index out of bounds for embedding lookup

### Pitfall 5: Missing Swish vs GELU Confusion
**What goes wrong:** Using GELU where Swish is expected (or vice versa)
**Why it happens:** Both are smooth activations, easy to confuse
**How to avoid:** Wav2Vec-BERT conv_module uses Swish (x * sigmoid(x)), FFN uses GELU
**Warning signs:** Slightly different output statistics

## Code Examples

Verified patterns from the existing codebase:

### Stats File var to std Conversion
```rust
// Source: Phase 1 decision, applied in load_stats
let std = match tensors.get("var") {
    Some(var) => var.sqrt()?,  // Convert variance to standard deviation
    None => match tensors.get("std") {
        Some(s) => s.clone(),
        None => {
            tracing::warn!(
                "[Wav2Vec-BERT] Missing 'std' or 'var' in stats file, using ones"
            );
            Tensor::ones((HIDDEN_SIZE,), DType::F32, device)?
        }
    }
};
```

### ConvModule Structure for Wav2Vec-BERT
```rust
// Source: Based on conformer.rs pattern + HuggingFace structure
struct ConvModule {
    layer_norm: LayerNorm,           // Pre-GLU normalization
    pointwise_conv1: Linear,         // [2048, 1024] expansion for GLU
    depthwise_conv_weight: Tensor,   // [1024, 1, 31]
    depthwise_layer_norm: LayerNorm, // Post-depthwise normalization
    pointwise_conv2: Linear,         // [1024, 1024] projection
    kernel_size: usize,              // 31
}

impl ConvModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Pre-normalization
        let x = self.layer_norm.forward(x)?;

        // 2. Pointwise conv1 (expand for GLU)
        let x = self.pointwise_conv1.forward(&x)?;

        // 3. GLU activation along feature dim
        let x = glu(&x, 2)?;  // dim=2 for [batch, seq, features]

        // 4. Depthwise conv: transpose to [batch, channels, seq]
        let x = x.transpose(1, 2)?;
        let padding = self.kernel_size / 2;  // = 15
        let channels = x.dim(1)?;  // 1024
        let x = x.conv1d(
            &self.depthwise_conv_weight,
            padding,
            1,        // stride
            1,        // dilation
            channels, // groups = channels (depthwise)
        )?;

        // 5. Depthwise LayerNorm (needs [batch, seq, channels] format)
        let x = x.transpose(1, 2)?;  // Back to [batch, seq, channels]
        let x = self.depthwise_layer_norm.forward(&x)?;

        // 6. Swish activation
        let x = swish(&x)?;

        // 7. Pointwise conv2
        self.pointwise_conv2.forward(&x)
    }
}
```

### Relative Position Bias (distance_embedding)
```rust
// Source: Adapted from HuggingFace Wav2Vec2BertSelfAttention
const LEFT_MAX_POS: usize = 64;
const RIGHT_MAX_POS: usize = 8;

fn compute_relative_position_bias(
    query: &Tensor,           // [batch, heads, query_len, head_dim]
    distance_embedding: &Tensor,  // [73, 64]
    device: &Device,
) -> Result<Tensor> {
    let query_len = query.dim(2)?;
    let key_len = query_len;  // Self-attention: Q and K have same length

    // Create position indices
    let pos_l = Tensor::arange(0i64, query_len as i64, device)?
        .reshape((query_len, 1))?;  // [Q, 1]
    let pos_r = Tensor::arange(0i64, key_len as i64, device)?
        .reshape((1, key_len))?;    // [1, K]

    // Compute relative distances: [Q, K]
    let distance = pos_r.broadcast_sub(&pos_l)?;

    // Clamp to valid range and shift to positive indices
    let distance = distance
        .clamp(-(LEFT_MAX_POS as f64), RIGHT_MAX_POS as f64)?
        .to_dtype(DType::I64)?;
    let distance = (distance + (LEFT_MAX_POS as f64))?;  // Shift: [-64,8] -> [0,72]

    // Lookup embeddings: [Q, K] -> [Q, K, head_dim]
    let pos_embed = distance_embedding.index_select(&distance.flatten_all()?, 0)?
        .reshape((query_len, key_len, 64))?;

    // Compute bias via einsum: query[b,h,l,d] * pos_embed[l,r,d] -> [b,h,l,r]
    // Equivalent: matmul query with transposed pos_embed
    let head_dim = query.dim(3)? as f64;
    let bias = query.matmul(&pos_embed.transpose(1, 2)?)?;
    (bias / head_dim.sqrt())
}
```

### Feature Projection Loading
```rust
// Source: Standard pattern for linear projection
struct FeatureProjection {
    layer_norm: LayerNorm,  // [160]
    projection: Linear,      // [1024, 160]
}

impl FeatureProjection {
    fn from_tensors(
        tensors: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let layer_norm = load_layer_norm(
            tensors,
            "feature_projection.layer_norm.weight",
            "feature_projection.layer_norm.bias",
            160,
            device,
        )?;

        let projection = load_linear(
            tensors,
            "feature_projection.projection.weight",
            Some("feature_projection.projection.bias"),
        )?;

        Ok(Self { layer_norm, projection })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        self.projection.forward(&x)
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Absolute position encoding | Relative position encoding (Shaw-style) | 2018+ | Better generalization to longer sequences |
| BatchNorm in Conformer | LayerNorm (depthwise_layer_norm) | Wav2Vec-BERT 2.0 (2023) | More stable for variable-length audio |
| Separate Q/K/V projections | Shared relative bias lookup | - | More parameter-efficient |

**Deprecated/outdated:**
- BatchNorm in conv module: Wav2Vec-BERT uses LayerNorm instead (depthwise_layer_norm)
- Absolute sinusoidal positions: Replaced by relative_key embeddings

## Open Questions

Things that couldn't be fully resolved:

1. **masked_spec_embed tensor**
   - What we know: Exists in checkpoint at [1024], used for masked training
   - What's unclear: Whether it's needed for inference
   - Recommendation: Load if present, but don't fail if missing (inference-only use case)

2. **Exact output of distance_embedding lookup**
   - What we know: HuggingFace uses torch.einsum("bhld,lrd->bhlr")
   - What's unclear: Most efficient Candle equivalent without einsum
   - Recommendation: Use matmul with transpose as shown in code example (tested pattern)

3. **FP16 numerical stability for long sequences**
   - What we know: LayerNorm can overflow in FP16
   - What's unclear: Candle's internal handling
   - Recommendation: Test with F32 first, then validate FP16 doesn't introduce NaN

## Sources

### Primary (HIGH confidence)
- [HuggingFace Transformers - Wav2Vec2-BERT modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py) - Distance embedding, conv_module implementation
- [facebook/w2v-bert-2.0 config.json](https://huggingface.co/facebook/w2v-bert-2.0) - Configuration values (left_max=64, right_max=8, kernel=31)
- [Candle Tensor conv1d](https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html) - Confirmed groups parameter support
- Existing codebase: `src/models/gpt/conformer.rs` - Working depthwise conv and GLU patterns

### Secondary (MEDIUM confidence)
- [HuggingFace Wav2Vec2-BERT docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) - Architecture overview
- [PyTorch Conv1d docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) - Groups parameter semantics
- Local checkpoint inspection via Python safetensors - Verified tensor shapes

### Tertiary (LOW confidence)
- WebSearch results on FP16 LayerNorm issues - Flagged for validation during testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing codebase patterns
- Architecture: HIGH - Verified against HuggingFace source and checkpoint
- Pitfalls: MEDIUM - Based on general ML knowledge + some codebase patterns
- Relative position implementation: MEDIUM - HuggingFace uses einsum, we adapt to matmul

**Research date:** 2026-01-23
**Valid until:** 60 days (stable architecture, HuggingFace Transformers rarely changes model code)
