# Feature Landscape: TTS Model Weight Loading

**Domain:** TTS Model Weight Loading for Rust/Candle
**Researched:** 2026-01-23
**Context:** Brownfield project - fixing weight loading for existing TTS pipeline (Wav2Vec-BERT, DiT, Conformer, Perceiver; BigVGAN already working)

---

## Table Stakes (Must Have)

Features users/developers expect. Missing = weight loading is broken or dangerous.

| Feature | Why Expected | Complexity | Current Status |
|---------|--------------|------------|----------------|
| **Name mapping from source format** | HuggingFace/PyTorch checkpoints use different naming conventions than Rust model structs | Medium | Partially implemented - BigVGAN works, others broken |
| **Shape validation before loading** | Mismatched shapes cause runtime crashes or silent corruption | Low | Missing - shapes not validated |
| **Complete weight coverage** | All model parameters must be loaded - missing weights = broken model | Medium | Broken - falls back to random silently |
| **Error on critical missing weights** | Core model weights (attention, FFN) must load or fail loudly | Low | Missing - silent fallback to random |
| **Missing/unexpected key reporting** | Return lists of what loaded vs. what didn't (like PyTorch `strict=False`) | Low | Missing - no visibility into partial failures |
| **Safetensors format support** | Industry standard for safe, fast weight loading | Already done | Implemented via candle |
| **Weight normalization handling** | BigVGAN uses weight_g/weight_v normalization that must be converted | Medium | Done for BigVGAN, may need for others |
| **Dtype consistency** | Weights must be loaded in correct dtype (F32 for inference) | Low | Implemented |
| **Device placement** | Weights load to correct device (CPU or CUDA) | Low | Implemented |
| **Transpose handling** | PyTorch stores Linear weights as [out, in], Candle may need transpose | Medium | Implemented in GPT/S2Mel weights |

### Critical Table Stakes Detail

**1. Name Mapping (Highest Priority)**

The core blocker. Pre-trained checkpoints use naming conventions that differ from the Rust struct field names:

```
Wav2Vec-BERT 2.0:
  Checkpoint:  encoder.layers.0.self_attn.linear_q.weight
  Rust struct: layers.0.attention.q_proj.weight

DiT:
  Checkpoint:  cfm.estimator.transformer.layers.0.attn.wqkv.weight
  Rust struct: blocks.0.attention.qkv_proj.weight

Conformer (in GPT checkpoint):
  Checkpoint:  conditioning_encoder.encoders.0.self_attn.linear_q.weight
  Rust struct: blocks.0.attention.q_proj.weight
```

Without name mapping, `HashMap::get()` returns `None` and code falls back to random weights.

**2. Fail on Critical Missing Weights**

Current behavior is dangerous:
```rust
// Current pattern (BAD - silent failure)
let weight = tensors.get(weight_key)
    .cloned()
    .unwrap_or_else(|| Tensor::randn(...));  // Silent random fallback!
```

Required behavior (like PyTorch `strict=False` which returns missing/unexpected keys):
```rust
// Required pattern (GOOD - loud failure for critical weights)
let weight = tensors.get(weight_key)
    .ok_or_else(|| anyhow::anyhow!("CRITICAL: Weight not found: {}", weight_key))?;

// Or for partial loading awareness:
let (missing_keys, unexpected_keys) = model.load_state_dict(tensors, strict=false);
if !missing_keys.is_empty() {
    tracing::error!("Missing keys: {:?}", missing_keys);
    return Err(...);
}
```

**3. Shape Validation**

Before loading a weight into a Linear layer:
```rust
let expected_shape = [out_features, in_features];
let actual_shape = tensor.dims();
if expected_shape != actual_shape {
    bail!("Shape mismatch for {}: expected {:?}, got {:?}",
          key, expected_shape, actual_shape);
}
```

This is critical because shape mismatches can cause:
- Silent tensor broadcast corruption
- Runtime dimension errors in forward pass
- Incorrect output dimensions

---

## Differentiators (Nice to Have)

Features that improve DX but aren't strictly required for functionality.

| Feature | Value Proposition | Complexity | Priority |
|---------|-------------------|------------|----------|
| **Loading progress reporting** | Shows which components loaded, which failed | Low | P1 |
| **Weight statistics logging** | Log mean/std/min/max of loaded weights for debugging | Low | P1 (debugging aid) |
| **Checkpoint introspection tool** | List all keys in checkpoint without loading | Low | P1 (debugging aid) |
| **Partial loading report** | Summary: "Loaded 45/50 weights, 5 missing" | Low | P2 |
| **Shape mismatch details** | "Expected [1024, 1024], got [768, 768]" in error | Low | P1 |
| **Key name suggestions** | "Did you mean 'encoder.layers.0' instead of 'layers.0'?" | Medium | P3 |
| **Weight comparison tool** | Compare Rust output vs Python output for same input | Medium | P2 |
| **Lazy loading** | Load weights on first use, not at init | High | P4 |
| **Strict mode toggle** | Option to fail on ANY missing weight vs. graceful degradation | Low | P2 |
| **Automatic weight downloading** | Download from HuggingFace if missing locally | Medium | P4 |
| **Weight caching** | Cache loaded weights to avoid re-parsing | Medium | P4 |
| **Memory-mapped loading** | Use mmap for large models to reduce memory | Already done | Implemented in Candle |
| **Golden tensor comparison** | Compare loaded weights against known-good reference values | Medium | P2 (debugging) |

### Priority P1: Essential for Current Debugging

**1. Weight Statistics Logging (High Priority)**

Currently the only debugging output is:
```
Loading Wav2Vec-BERT weights from ...
  Sample tensor keys: [...]
  Successfully loaded 24 of 24 encoder layers
```

Should add:
```
  Layer 0 attention.q_proj: shape=[1024, 1024], mean=-0.0003, std=0.0412
  Layer 0 attention.k_proj: shape=[1024, 1024], mean=0.0001, std=0.0398
  ...
  Total weights: 450 tensors, 892MB
  Weight distribution: mean=-0.0002, std=0.0405 (expected ~0.02 for init)
```

This immediately reveals if weights loaded vs. random:
- Random init has std ~0.02 (Xavier/He initialization)
- Trained weights have different, model-specific distributions
- If all weights have identical std=0.02, something is wrong

**2. Checkpoint Introspection (High Priority)**

Before attempting to load, enumerate the checkpoint:
```rust
pub fn inspect_checkpoint<P: AsRef<Path>>(path: P) -> Result<CheckpointInfo> {
    let tensors = safetensors::load(path, &Device::Cpu)?;
    let mut info = CheckpointInfo::new();
    for (name, tensor) in &tensors {
        info.add(name, tensor.dims(), tensor.dtype());
    }
    info
}
```

This enables:
- Discovering the actual key names in a checkpoint
- Creating the mapping table
- Documenting expected vs. actual structure

**3. Shape Mismatch Details**

When shapes don't match, the error should be immediately diagnostic:
```rust
Err(anyhow!(
    "Shape mismatch for '{}': expected {:?}, got {:?}. \
     This usually means the checkpoint is for a different model configuration.",
    key, expected_shape, actual_shape
))
```

### Priority P2: Useful for Validation

**4. Strict Mode Toggle (Medium Priority)**

```rust
pub enum LoadMode {
    /// Fail on any missing weight (production)
    Strict,
    /// Warn but continue with random for missing (development)
    Permissive,
    /// Log only, no warnings (testing)
    Silent,
}

// CLI flag: --allow-partial-weights
```

This allows:
- Strict mode for production (fail fast)
- Permissive mode for development (see how far you get)
- Silent mode for testing infrastructure

**5. Golden Tensor Comparison**

For debugging, compare loaded weights against known-good reference:
```rust
// Load reference from Python export
let golden = load_golden("debug/golden/wav2vec_bert_layer0_q.safetensors")?;
let loaded = model.layers[0].attention.q_proj.weight();
let diff = (golden - loaded)?.abs()?.max_all()?.to_scalar::<f32>()?;
assert!(diff < 1e-5, "Weight mismatch: max diff = {}", diff);
```

---

## Anti-Features (Don't Build)

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Silent fallback to random** | Core anti-pattern causing current issues; produces garbage output that looks like "working" | Fail loudly, or log WARNING with explicit "USING RANDOM WEIGHTS" |
| **Automatic name guessing/fuzzy matching** | Fuzzy matching ("linear_q" ~= "q_proj") leads to subtle bugs where wrong weights load | Require explicit mapping tables |
| **Auto-transpose weights** | Different frameworks have different conventions; auto-transpose masks bugs | Explicitly document and handle transpose requirements |
| **Dynamic architecture inference** | Inferring model structure from weights is fragile | Define model architecture explicitly, load weights into it |
| **Weight conversion on load** | Converting between formats (GGUF, GGML, safetensors, .bin) at runtime is slow and error-prone | Provide offline conversion scripts, load only safetensors |
| **Checkpoint migration** | Automatically upgrading old checkpoints to new format | Version checkpoints explicitly, reject incompatible versions |
| **Partial model loading for inference** | Loading only some layers "to save memory" | Load full model or don't - partial models produce garbage |
| **Weight quantization on load** | Quantizing F32 to F16/INT8 during loading | Quantize ahead of time or use pre-quantized checkpoints |
| **Pickle file support** | Security vulnerability (arbitrary code execution) | Safetensors only |
| **Automatic weight downloading** | Security concerns, network dependencies in inference | Require pre-downloaded weights |

### Why These Are Anti-Features

**Silent Fallback to Random (Most Critical)**

The codebase has this pattern throughout:
```rust
// In wav2vec_bert.rs, conformer.rs, perceiver.rs, dit.rs:
.unwrap_or_else(|| Tensor::randn(0.0f32, 0.02, shape, device)?)
```

This is the root cause of the "noisy output" bug:
1. Model runs without errors
2. Produces output (not silence)
3. Output is garbage because attention/FFN weights are random
4. Error is invisible - no warning, no logging
5. Developer spends hours debugging audio quality instead of weight loading

**Automatic Name Guessing**

It's tempting to do:
```rust
fn fuzzy_match(target: &str, available: &[String]) -> Option<String> {
    // Find "closest" key by edit distance
}
```

But this leads to loading the wrong weight into the wrong layer:
- `layer.0.attention.q_proj` matched to `layer.10.attention.q_proj`
- Silently corrupts model
- Produces "almost working" output that's actually broken

Instead: Require explicit mapping or fail.

**Dynamic Architecture Inference**

Don't do:
```rust
fn load_model_from_checkpoint(path: &str) -> Result<Box<dyn Model>> {
    let tensors = load(path)?;
    let num_layers = count_layers(&tensors);  // What if optimizer state is included?
    let hidden_dim = infer_hidden_dim(&tensors);  // Ambiguous from weight shapes
    // Construct model from inferred config
}
```

This breaks when:
- Checkpoint has extra keys (optimizer state, gradient buffers)
- Model has optional components
- Weight shapes are ambiguous (1024 hidden or 1024 vocab?)

Instead: Define model config explicitly, load weights into pre-defined structure.

---

## Feature Dependencies

```
                    ┌─────────────────────────────────┐
                    │  Name Mapping Implementation    │
                    │  (MUST HAVE FIRST)              │
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ Shape Validation  │   │ Error on Missing  │   │ Checkpoint        │
│ (Pre-load check)  │   │ Critical Weights  │   │ Introspection     │
└───────────────────┘   └───────────────────┘   └───────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Missing/Unexpected Key Report  │
                    │  (Like PyTorch strict=False)    │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Weight Statistics Logging      │
                    │  (Verification & Debugging)     │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Strict Mode Toggle             │
                    │  (Production vs. Development)   │
                    └─────────────────────────────────┘
```

---

## MVP Recommendation

For immediate fix of the weight loading issue:

### Phase 1: Critical Fixes (Required for Working TTS)

1. **Remove Silent Fallback** - Convert all `unwrap_or_else(|| random)` to `ok_or_else(|| error)?` for attention and FFN weights

2. **Name Mapping Functions** - Create mapping for each model:
   - Wav2Vec-BERT: `map_hf_wav2vec_bert_to_rust()`
   - DiT: `map_s2mel_dit_to_rust()`
   - Conformer: `map_gpt_conformer_to_rust()`
   - Perceiver: `map_gpt_perceiver_to_rust()`

3. **Shape Validation** - Add pre-load shape checks

4. **Missing/Unexpected Key Return** - Return lists from `load_weights()` like PyTorch's `strict=False`

### Phase 2: Debugging Aids (Required for Validation)

5. **Checkpoint Introspection** - Tool to list all keys in a checkpoint

6. **Weight Statistics Logging** - Log mean/std of loaded weights

7. **Loading Summary** - Report "loaded X/Y weights, Z missing"

### Phase 3: Production Hardening (Nice to Have)

8. **Strict Mode Toggle** - Toggle between fail-fast and permissive

9. **Progress Reporting** - Show loading progress for large models

### Defer to Post-Fix:
- Memory-mapped loading (already done)
- Lazy loading (optimization)
- Quantization support (future feature)
- Auto-download (security concern)

---

## Implementation Guidance

### Pattern: Name Mapping Function

```rust
/// Maps HuggingFace Wav2Vec-BERT 2.0 key names to our Rust struct names
fn map_wav2vec_bert_key(hf_key: &str) -> Option<String> {
    // Strip prefix
    let key = hf_key.strip_prefix("encoder.")?;

    // Map component names
    let mapped = key
        .replace("self_attn.linear_q", "attention.q_proj")
        .replace("self_attn.linear_k", "attention.k_proj")
        .replace("self_attn.linear_v", "attention.v_proj")
        .replace("self_attn.linear_out", "attention.out_proj")
        .replace("ffn1.intermediate_dense", "ffn1.intermediate")
        .replace("ffn1.output_dense", "ffn1.output")
        .replace("ffn2.intermediate_dense", "ffn2.intermediate")
        .replace("ffn2.output_dense", "ffn2.output");

    Some(mapped)
}
```

### Pattern: Loud Failure for Critical Weights

```rust
/// Load required weight - fails if not found
fn load_required_weight(
    tensors: &HashMap<String, Tensor>,
    key: &str,
) -> Result<Tensor> {
    tensors.get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!(
            "CRITICAL: Required weight '{}' not found. Available keys (first 10): {:?}",
            key,
            tensors.keys().take(10).collect::<Vec<_>>()
        ))
}

/// Load optional weight - returns None if not found, with warning
fn load_optional_weight(
    tensors: &HashMap<String, Tensor>,
    key: &str,
) -> Option<Tensor> {
    match tensors.get(key) {
        Some(t) => Some(t.clone()),
        None => {
            tracing::warn!("Optional weight '{}' not found, using default", key);
            None
        }
    }
}
```

### Pattern: Missing/Unexpected Key Reporting

```rust
pub struct LoadResult {
    pub loaded: usize,
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
}

pub fn load_weights_with_report(
    tensors: &HashMap<String, Tensor>,
    expected_keys: &[&str],
) -> Result<LoadResult> {
    let mut missing = Vec::new();
    let mut unexpected = Vec::new();
    let mut loaded = 0;

    for key in expected_keys {
        if tensors.contains_key(*key) {
            loaded += 1;
        } else {
            missing.push(key.to_string());
        }
    }

    for key in tensors.keys() {
        if !expected_keys.contains(&key.as_str()) {
            unexpected.push(key.clone());
        }
    }

    Ok(LoadResult { loaded, missing_keys: missing, unexpected_keys: unexpected })
}
```

### Pattern: Shape Validation

```rust
fn validate_and_load_linear(
    tensors: &HashMap<String, Tensor>,
    key: &str,
    expected_out: usize,
    expected_in: usize,
) -> Result<Linear> {
    let weight = load_required_weight(tensors, key)?;
    let (out, inp) = weight.dims2()?;

    if out != expected_out || inp != expected_in {
        bail!(
            "Shape mismatch for '{}': expected [{}, {}], got [{}, {}]. \
             This usually means the checkpoint is for a different model configuration \
             (e.g., hidden_dim=768 vs hidden_dim=1024).",
            key, expected_out, expected_in, out, inp
        );
    }

    Ok(Linear::new(weight, None))
}
```

### Pattern: Weight Statistics for Debugging

```rust
fn log_weight_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let mean: f32 = tensor.mean_all()?.to_scalar()?;
    let flat = tensor.flatten_all()?;
    let min: f32 = flat.min(0)?.to_scalar()?;
    let max: f32 = flat.max(0)?.to_scalar()?;

    // Compute std
    let centered = tensor.broadcast_sub(&tensor.mean_all()?)?;
    let variance = centered.sqr()?.mean_all()?;
    let std: f32 = variance.sqrt()?.to_scalar()?;

    tracing::debug!(
        "{}: shape={:?}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
        name, tensor.dims(), mean, std, min, max
    );

    // Flag suspicious values (random init signature)
    if (std - 0.02).abs() < 0.005 && mean.abs() < 0.001 {
        tracing::warn!(
            "{}: Statistics suggest random initialization (std~0.02, mean~0). \
             Weight may not have loaded correctly.",
            name
        );
    }

    Ok(())
}
```

---

## Sources

### Official Documentation
- [Candle GitHub - HuggingFace ML Framework for Rust](https://github.com/huggingface/candle)
- [Safetensors Rust Docs](https://docs.rs/safetensors/)
- [HuggingFace safetensors Loading Guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors)

### PyTorch Weight Loading Patterns
- [PyTorch load_state_dict strict=False Discussion](https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379)
- [Understanding Missing Keys in state_dict](https://www.codegenes.net/blog/missing-key-s-in-state_dict-pytorch/)
- [Handling Unexpected Keys in PyTorch Lightning](https://www.restack.io/p/pytorch-lightning-answer-unexpected-keys-state-dict-cat-ai)
- [PyTorch Saving and Loading Models](https://brsoff.github.io/tutorials/beginner/saving_loading_models.html)

### TTS Model Loading
- [PyTorch TTS Tacotron2 Tutorial](https://docs.pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html)
- [NVIDIA NeMo TTS Configuration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/configs.html)
- [HuggingFace Safetensors Convert Weights](https://huggingface.co/docs/safetensors/en/convert-weights)

### Weight Validation & Debugging
- [TensorFlow Layer Weight Shape Mismatch](https://www.omi.me/blogs/tensorflow-errors/layer-weight-shape-mismatch-in-tensorflow-causes-and-how-to-fix)
- [Weight Initialization in Neural Networks](https://www.pinecone.io/learn/weight-initialization/)
- [The Effects of Weight Initialization - Weights & Biases](https://wandb.ai/wandb_fc/articles/reports/The-effects-of-weight-initialization-on-neural-nets--Vmlldzo1NDc1NjU3)

### Candle/Rust ML References
- [Using HuggingFace with Rust - Shuttle](https://www.shuttle.dev/blog/2024/05/01/using-huggingface-rust)
- [Getting Started with Candle](https://medium.com/@cursor0p/getting-started-with-candle-%EF%B8%8F-535d7a85e30a)
- [Candle Documentation - Introduction](https://huggingface.github.io/candle/)
