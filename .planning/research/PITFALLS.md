# Domain Pitfalls: Candle/Rust ML Weight Loading and Windows CUDA

**Domain:** Rust ML inference with Candle framework, TTS pipeline
**Researched:** 2026-01-23
**Confidence:** HIGH (verified with codebase analysis, Candle GitHub issues, and official sources)

---

## Critical Pitfalls

Mistakes that cause "model runs but produces garbage" issues. These are the highest priority to prevent.

### Pitfall 1: Silent Fallback to Random Weights

**What goes wrong:** When weight tensors are not found during loading, code silently initializes with random values instead of failing. The model runs but produces noise/garbage output.

**Why it happens:** Defensive coding pattern where missing weights trigger `initialize_random()` or `Tensor::randn()` instead of returning an error. This is common in Candle projects that want to support both inference (with weights) and training (random init).

**Consequences:**
- Model appears to work (no errors, correct output shapes)
- Output is random noise instead of meaningful predictions
- Very hard to debug because everything "seems fine"
- Tests pass but audio sounds like wind/noise

**Evidence from codebase:**
```rust
// src/models/gpt/perceiver.rs:556-558
if !path.exists() {
    eprintln!("Warning: Perceiver weights not found at {:?}, using random weights", path);
    return self.initialize_random();
}

// src/models/gpt/perceiver.rs:612-620 - SILENT fallback in tensor loading
} else {
    // Fallback to random latents
    self.latents = Some(Tensor::randn(
        0.0f32, 0.02,
        (1, self.config.num_latents, self.config.dim),
        &self.device,
    )?);
}

// src/models/semantic/wav2vec_bert.rs:504-515 - Falls back per layer
Err(e) => {
    eprintln!("  Warning: Failed to load layer {}: {}", i, e);
    // Try to load with random weights for this layer
    match EncoderLayer::new_random(...) {
        Ok(layer) => {
            self.encoder_layers.push(layer);
            eprintln!("    Using random weights for layer {}", i);
        }
        ...
    }
}
```

**Warning signs:**
- `eprintln!` warnings about "using random weights" in logs
- Output audio sounds like wind/noise (not speech)
- Model weights file exists but specific tensors are missing
- `weights_loaded = true` even when actual weights weren't loaded
- Tensor statistics show mean near 0, variance near 0.02-0.04

**Prevention:**
1. **Fail loudly on missing weights** - Change random fallback to `bail!()` or `Err()`
2. **Add weight validation** - Check tensor statistics after loading
3. **Log tensor counts** - Compare expected vs actual tensors loaded
4. **Integration test with known input** - Compare output against golden reference

**Detection code:**
```rust
// Add after weight loading
fn validate_weights_loaded(tensor: &Tensor, name: &str) -> Result<()> {
    let mean = tensor.mean_all()?.to_scalar::<f32>()?;
    let var = tensor.var(0)?.mean_all()?.to_scalar::<f32>()?;

    // Random init typically has mean ~0, var ~0.02-0.04
    // Trained weights have different distributions
    if mean.abs() < 0.001 && (var - 0.02).abs() < 0.01 {
        eprintln!("WARNING: {} appears to have random weights (mean={:.4}, var={:.4})",
                  name, mean, var);
    }
    Ok(())
}
```

**Phase to address:** Phase 1 (Weight Loading Fix) - This is THE critical issue causing noisy output

---

### Pitfall 2: Weight Name Mismatch Between PyTorch and Candle

**What goes wrong:** PyTorch models use different naming conventions than Candle expects. Weights exist in the safetensors file but are not found because the keys don't match.

**Why it happens:**
- HuggingFace models use `encoder.layers.0.self_attn.q_proj.weight`
- Custom models use `layers.0.attention.q_proj.weight`
- GPT-2 uses combined `c_attn` weights that need splitting
- Some models prefix everything with `model.` or `bert.`
- LayerNorm uses `gamma`/`beta` vs `weight`/`bias`

**Evidence from Candle issues:**
> "The model spec in candle-transformers/src/models/bert.rs results in: Error: TensorNotFound('embeddings.word_embeddings.weight'). The Safetensors version prepends all variables with 'bert'" - [GitHub Issue #1887](https://github.com/huggingface/candle/issues/1887)

> "The problem is in layer_norm which doesn't expect gamma and beta but weight and bias" - [GitHub Issue #1887](https://github.com/huggingface/candle/issues/1887)

**Consequences:**
- `TensorNotFound` errors (if code is strict)
- Silent fallback to random (if code is defensive)
- Partial model loading (some layers work, others don't)

**Warning signs:**
- Errors mentioning `TensorNotFound` or "Key not found"
- Only some layers loading successfully
- Mismatched layer counts between model definition and loaded weights
- Log shows "Warning: Failed to load layer X"

**Prevention:**
1. **Inspect safetensors keys first:**
   ```python
   from safetensors import safe_open
   with safe_open("model.safetensors", framework="pt") as f:
       for key in sorted(f.keys())[:20]:
           print(f"{key}: {f.get_tensor(key).shape}")
   ```

2. **Create explicit name mappings:**
   ```rust
   fn map_hf_to_rust(hf_name: &str) -> Option<String> {
       hf_name
           .strip_prefix("encoder.")?
           .replace("self_attn", "attention")
           .replace("gamma", "weight")  // LayerNorm
           .replace("beta", "bias")     // LayerNorm
           .into()
   }
   ```

3. **Use VarBuilder with prefix navigation:**
   ```rust
   let vb = VarBuilder::from_mmaped_safetensors(&[path], dtype, device)?;
   let layer_vb = vb.pp("encoder").pp("layers").pp(&i.to_string());
   ```

4. **Document expected vs actual key names in a mapping table**

**Phase to address:** Phase 1 (Weight Loading Fix)

---

### Pitfall 3: Weight Transposition (PyTorch vs Candle Convention)

**What goes wrong:** PyTorch stores Linear weights as `[out_features, in_features]`. Some code transposes when it shouldn't, or doesn't transpose when it should.

**Why it happens:**
- Candle's `Linear::new()` expects PyTorch format `[out, in]` - no transpose needed
- Manual `matmul` operations need weights transposed
- Confusion about when transpose is required

**Evidence from codebase:**
```rust
// src/models/gpt/weights.rs:144-146 - GPT-2 c_attn needs transpose after split
let q_weight = c_attn_weight.i((.., 0..hidden_size))?.t()?.contiguous()?;
let k_weight = c_attn_weight.i((.., hidden_size..2*hidden_size))?.t()?.contiguous()?;
let v_weight = c_attn_weight.i((.., 2*hidden_size..3*hidden_size))?.t()?.contiguous()?;
```

**Consequences:**
- Runtime errors about shape mismatches: `mat1 and mat2 shapes cannot be multiplied`
- Silent shape broadcasting that produces wrong results
- Model runs but output is scrambled

**Warning signs:**
- Dimension mismatch errors during forward pass
- Output has unexpected dimensions
- Weights load but inference produces nonsense

**Prevention:**
1. **For `Linear::new()`: NO transpose needed** - Candle expects PyTorch format
2. **For manual matmul: transpose is needed** - `x.matmul(&weight.t()?)?`
3. **Print shapes during development:**
   ```rust
   eprintln!("Weight shape: {:?}, expected: [{}, {}]",
             weight.dims(), out_features, in_features);
   ```
4. **Verify with small test:**
   ```rust
   let test_input = Tensor::ones((1, in_features), dtype, device)?;
   let output = linear.forward(&test_input)?;
   assert_eq!(output.dims(), &[1, out_features]);
   ```

**Phase to address:** Phase 1 (Weight Loading Fix)

---

### Pitfall 4: Weight Normalization Not Applied

**What goes wrong:** Checkpoint stores `weight_v` and `weight_g` separately (weight normalization), but loader uses `weight_v` directly without computing `weight = g * v / ||v||`.

**Why it happens:** Not recognizing weight normalization pattern in checkpoint keys. BigVGAN and other vocoders commonly use this.

**Evidence from codebase (correct implementation):**
```rust
// src/models/vocoder/weights.rs:33-56 - BigVGAN handles this correctly
if name.ends_with(".weight_v") {
    let base_name = name.strip_suffix(".weight_v").unwrap();
    let g_name = format!("{}.weight_g", base_name);

    if let Some(weight_g) = tensors.get(&g_name) {
        // Apply weight normalization: weight = g * v / ||v||
        let weight = apply_weight_norm(weight_g, tensor)?;
        converted.insert(format!("{}.weight", base_name), weight);
    }
}
```

**Consequences:**
- Weights have wrong scale (orders of magnitude off)
- Model produces very wrong outputs but doesn't crash
- Particularly affects vocoder quality

**Warning signs:**
- Checkpoint has keys ending in `_v` and `_g`
- Audio output is distorted or has wrong amplitude
- Works in PyTorch but not in Rust

**Prevention:**
1. **Check for weight_v/weight_g pattern:**
   ```bash
   python -c "from safetensors import safe_open; f = safe_open('model.safetensors', 'pt'); print([k for k in f.keys() if '_v' in k or '_g' in k])"
   ```
2. **Implement weight norm conversion** (see BigVGAN weights.rs)
3. **Verify output amplitude is reasonable** after loading

**Phase to address:** Phase 1 (Weight Loading Fix)

---

## CUDA Pitfalls (Windows + RTX 3090)

### Pitfall 5: CUDA Toolkit Version Mismatch with Candle

**What goes wrong:** Candle's `--features cuda` fails to compile or produces runtime errors due to CUDA version incompatibility.

**Why it happens:**
- RTX 3090 (Ampere architecture) requires CUDA 11.1+ and cuDNN 8.0+
- Candle-kernels needs specific CUDA versions for compilation
- Windows CUDA support in Candle "lags behind Linux"

**Evidence:**
> "Windows CUDA support lags behind Linux; contributors welcome!" - [Candle Installation Docs](https://huggingface.github.io/candle/guide/installation.html)

> "RTX 3090 needs CUDA version >= 11.1" - [NVIDIA Forums](https://forums.developer.nvidia.com/t/lot-of-ai-repos-cannot-run-on-rtx-30-series-gpu/180646)

> "Build failures can occur in candle-kernels with the custom build command" - [GitHub Issue #3166](https://github.com/huggingface/candle/issues/3166)

**Warning signs:**
- `LINK : fatal error LNK1181: cannot open input file`
- `nvcc fatal: A single input file is required`
- Compilation errors in `candle-kernels`
- `CUDA_ERROR_OUT_OF_MEMORY` on models that should fit

**Prevention:**
1. **Use recommended CUDA version:** CUDA 12.3+ with cuDNN 9.x (recommended 2025)
2. **Verify GPU and drivers:**
   ```bash
   nvidia-smi
   nvcc --version
   ```
3. **Test basic CUDA before building Candle:**
   ```rust
   let device = Device::new_cuda(0)?;
   let x = Tensor::ones((2, 2), DType::F32, &device)?;
   println!("CUDA works: {:?}", x.dims());
   ```
4. **Start with CPU, add CUDA after weights work**

**Phase to address:** Phase 2 (CUDA Integration) - After weights work on CPU

---

### Pitfall 6: Windows Path Issues with CUDA Compilation

**What goes wrong:** CUDA compilation fails due to Windows path handling, especially with spaces or special characters.

**Evidence:**
> "LINK : fatal error LNK1181: Unable to open the input file 'Files/NVIDIA.obj'" - [GitHub Issue opencv#15321](https://github.com/opencv/opencv/issues/15321)

> "$(CUDA_LIB_PATH) on CUDA 3.2 seems to be a bit different from earlier versions" - [GitHub Issue cudpp#72](https://github.com/cudpp/cudpp/issues/72)

**Consequences:**
- Build fails with cryptic linker errors
- Works on some machines but not others
- CI/CD builds fail inconsistently

**Warning signs:**
- Error messages contain `Files/NVIDIA` or similar split paths
- Build works on Linux but fails on Windows
- `LNK1181` errors referencing partial paths

**Prevention:**
1. **Install CUDA to path without spaces:**
   ```
   C:\CUDA             (good)
   C:\Program Files\NVIDIA GPU Computing Toolkit  (bad)
   ```
2. **Set environment variables explicitly:**
   ```batch
   set CUDA_PATH=C:\CUDA
   set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
   set PATH=%CUDA_PATH%\bin;%PATH%
   ```
3. **Use `cargo build -vv` to see actual compiler commands**
4. **Quote all paths in build scripts**

**Phase to address:** Phase 2 (CUDA Integration)

---

### Pitfall 7: Memory Allocation on RTX 3090

**What goes wrong:** Model loads but inference fails with `CUDA_ERROR_OUT_OF_MEMORY` even though GPU has 24GB VRAM.

**Evidence:**
> "Users have reported getting CUDA_ERROR_OUT_OF_MEMORY with Stable Diffusion 3.5 Large on RTX 3090" - [GitHub Issue #2597](https://github.com/huggingface/candle/issues/2597)

**Why it happens:**
- Windows reserves ~1-2GB VRAM for display
- CUDA memory fragmentation from repeated allocations
- Multiple tensors allocated at peak (especially in flow matching)
- Batch size too large

**Warning signs:**
- OOM errors despite `nvidia-smi` showing free memory
- Works for short inputs, fails for longer ones
- Memory usage spikes during specific pipeline stages

**Prevention:**
1. **Start with batch_size=1, increase gradually**
2. **Monitor with `nvidia-smi -l 1` during inference**
3. **Close other GPU applications (browsers, games)**
4. **Consider memory-efficient attention if available**
5. **Profile peak memory usage:**
   ```rust
   fn log_cuda_memory() {
       if let Ok(device) = Device::new_cuda(0) {
           // Candle doesn't expose memory stats directly
           // Use nvidia-smi or add custom CUDA bindings
       }
   }
   ```

**Phase to address:** Phase 2 (CUDA Integration)

---

## TTS-Specific Silent Failures

### Pitfall 8: Mel Spectrogram Parameter Mismatch

**What goes wrong:** TTS produces mel spectrograms that vocoder can't synthesize correctly, resulting in noise or robotic audio.

**Why it happens:** TTS model and vocoder were trained with different audio parameters.

**Evidence:**
> "Feeding mel spectrogram output from one TTS model into a pretrained vocoder from another project can result in garbage output unless you scale the spectrograms" - [GitHub Issue mozilla/TTS#607](https://github.com/mozilla/TTS/issues/607)

> "Different TTS libraries use different normalization methods for spectrogram normalization" - [GitHub Issue mozilla/TTS#377](https://github.com/mozilla/TTS/issues/377)

**Critical parameters that must match:**
| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| sample_rate | Audio sample rate | 22050, 24000 |
| n_mels | Mel frequency bins | 80, 100 |
| n_fft | FFT window size | 1024, 2048 |
| hop_length | Samples between frames | 256, 512 |
| win_length | Window length | 1024 |
| mel_fmin | Minimum mel frequency | 0.0, 50.0 |
| mel_fmax | Maximum mel frequency | 8000.0, 11025.0 |

**Warning signs:**
- Griffin-Lim produces okay audio but vocoder produces noise
- Audio has correct rhythm but wrong pitch
- Output has metallic/robotic quality
- Spectrogram looks reasonable but audio is garbage

**Prevention:**
1. **Document all audio parameters in config:**
   ```yaml
   audio:
     sample_rate: 22050
     n_mels: 80
     mel_fmin: 0.0
     mel_fmax: 8000.0
     hop_length: 256
     win_length: 1024
     n_fft: 1024
   ```
2. **Verify parameters match between all pipeline stages**
3. **Test vocoder with ground-truth mels first** (before testing full pipeline)
4. **Check mel spectrogram value ranges** (different normalizations produce different ranges)

**Phase to address:** After weights load correctly (validation phase)

---

### Pitfall 9: VarBuilder Backend Selection for Training vs Inference

**What goes wrong:** Using wrong VarBuilder backend causes either missing gradients (for training) or slow inference (for deployment), or silent random initialization.

**Evidence:**
> "When you want to finetune a model you have to load model weights using varmap instead of other varbuilder methods, but when doing inference you want to load weights using other methods than varmap to turn off the gradients" - [Candle Tutorial](https://medium.com/@cursor0p/lets-learn-candle-%EF%B8%8F-ml-framework-for-rust-9c3011ca3cd9)

**VarBuilder backends:**
| Backend | Behavior | Use Case |
|---------|----------|----------|
| `VarMap` | Creates tensors with gradients, random init for missing | Training |
| `from_mmaped_safetensors` | Memory-maps weights, errors on missing | Inference |
| `Zeros` | All zeros | Debugging |

**The trap:** VarMap silently creates random weights for any tensor not found in the checkpoint.

**Prevention:**
- **For inference:** Always use `VarBuilder::from_mmaped_safetensors`
- **Ensure all tensors exist** before loading
- **Don't mix backends** in same model
- **Add explicit checks** for expected tensor count

**Phase to address:** Phase 1 (Weight Loading Fix)

---

## Moderate Pitfalls

### Pitfall 10: Fused Tensor Splitting Errors

**What goes wrong:** Checkpoints often store fused QKV projections as single tensor. Splitting along wrong dimension or at wrong offsets produces broken attention.

**Why it happens:** Different models use different fusion patterns (Q|K|V vs K|V vs interleaved).

**Evidence from codebase:**
```rust
// src/models/gpt/weights.rs - GPT-2 QKV splitting
// c_attn.weight shape in PyTorch: [in_features, 3*hidden_size]
let q_weight = c_attn_weight.i((.., 0..hidden_size))?;
let k_weight = c_attn_weight.i((.., hidden_size..2*hidden_size))?;
let v_weight = c_attn_weight.i((.., 2*hidden_size..3*hidden_size))?;
```

**Prevention:**
- Document fusion pattern from original model code
- Verify split dimensions match model's hidden_size
- Test attention output shapes before full inference
- Print intermediate shapes during development

**Phase to address:** Phase 1 (Weight Loading Fix)

---

### Pitfall 11: DType Mismatch (F32 vs F16 vs BF16)

**What goes wrong:** Loading F16 weights into F32 model or vice versa causes silent precision loss or explicit errors.

**Warning signs:**
- NaN or Inf values in output
- Numerical overflow during forward pass
- Results differ significantly from PyTorch

**Prevention:**
1. Check safetensors dtype before loading:
   ```python
   from safetensors import safe_open
   with safe_open("model.safetensors", "pt") as f:
       for k in list(f.keys())[:5]:
           print(f"{k}: {f.get_tensor(k).dtype}")
   ```
2. Convert explicitly if needed: `tensor.to_dtype(DType::F32)?`
3. Use BF16 on Ampere GPUs for better performance with less precision loss

**Phase to address:** Phase 2 (CUDA Integration) - BF16 is mainly beneficial on GPU

---

### Pitfall 12: Contiguous Tensor Requirements

**What goes wrong:** Operations fail or produce wrong results because tensors aren't contiguous after transpose/slice operations.

**Why it happens:** Candle operations may require contiguous memory layout, and transpose/slice create views without copying.

**Warning signs:**
- Cryptic errors about tensor layout
- Operations work on some tensors but not others
- Different results than PyTorch for same operations

**Prevention:**
- Call `.contiguous()?` after transpose, slice, or view operations:
  ```rust
  let weight = tensor.t()?.contiguous()?;  // Always after transpose
  let slice = tensor.i(0..10)?.contiguous()?;  // After slicing
  ```
- Check with `tensor.is_contiguous()` during debugging

**Phase to address:** Phase 1 (Weight Loading Fix)

---

## Minor Pitfalls

### Pitfall 13: Missing Bias Tensors

**What goes wrong:** Some checkpoints omit bias tensors (implicitly zero), loader expects them to exist.

**Prevention:**
- Make bias loading optional
- Create zero bias tensor if not present
- Current code already handles this in most places

---

### Pitfall 14: Device Mismatch Between Tensors

**What goes wrong:** Operations fail when one tensor is on CPU and another on GPU.

**Prevention:**
- Create all tensors with same device
- Use `tensor.to_device(device)?` when needed
- Validate device consistency in tests

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Detection | Mitigation |
|-------|---------------|-----------|------------|
| Weight Loading | Silent random fallback | Check logs for "using random" warnings; validate tensor statistics | Fail loudly on missing weights |
| Weight Loading | Name mismatch | TensorNotFound errors; mismatched layer counts | Inspect safetensors keys, create mappings |
| Weight Loading | Transpose issues | Shape mismatch errors or wrong output | Print shapes, verify dimensions |
| Weight Loading | Weight norm not applied | Check for `_v`/`_g` keys in checkpoint | Implement conversion like BigVGAN |
| CUDA Setup | Version mismatch | Compilation errors | Use CUDA 12.3+, avoid path spaces |
| CUDA Setup | Path spaces | LNK1181 errors | Install to `C:\CUDA` |
| CUDA Setup | OOM errors | Runtime CUDA errors | Start small, monitor VRAM |
| TTS Pipeline | Mel parameter mismatch | Noise with correct rhythm | Verify all audio params match |
| Integration | DType mismatch | Precision loss or NaN | Explicit dtype conversion |

---

## Quick Detection Checklist

Before running inference, verify:

- [ ] No "using random weights" warnings in logs
- [ ] Tensor count matches expected (e.g., BigVGAN: 667 tensors, Wav2Vec-BERT: 24 layers)
- [ ] Weight statistics are not all near zero mean / 0.02 variance
- [ ] All layer counts match config (24 encoder layers, 13 DiT blocks, etc.)
- [ ] No `_v`/`_g` keys left unconverted (weight normalization)
- [ ] Audio parameters match between pipeline stages

For CUDA (Phase 2):
- [ ] CUDA path has no spaces
- [ ] `nvidia-smi` shows driver is loaded
- [ ] `nvcc --version` matches expected version
- [ ] Basic CUDA tensor operations work before running model

---

## Quick Validation Script

Add this to detect silent loading failures:

```rust
fn validate_loaded_weights(tensors: &HashMap<String, Tensor>, expected_count: usize) -> Result<()> {
    let actual_count = tensors.len();
    if actual_count < expected_count * 9 / 10 {
        bail!("Only {}/{} tensors loaded - check weight names",
              actual_count, expected_count);
    }

    // Check for suspiciously random-looking weights
    let mut suspicious = 0;
    for (name, tensor) in tensors.iter().take(10) {
        let flat = tensor.flatten_all()?;
        let mean: f32 = flat.mean_all()?.to_scalar()?;
        let variance: f32 = flat.var(0)?.to_scalar()?;

        // Random init typically has mean ~0, variance ~0.02-0.04
        if mean.abs() < 0.001 && variance > 0.015 && variance < 0.05 {
            eprintln!("WARNING: {} looks randomly initialized (mean={:.4}, var={:.4})",
                      name, mean, variance);
            suspicious += 1;
        }
    }

    if suspicious > 3 {
        bail!("Multiple tensors appear randomly initialized - weights may not have loaded correctly");
    }

    Ok(())
}
```

---

## Sources

### Candle/Rust ML
- [Candle GitHub Repository](https://github.com/huggingface/candle)
- [Candle Installation Guide](https://huggingface.github.io/candle/guide/installation.html)
- [BERT Safetensors Variable Mismatch - Issue #1887](https://github.com/huggingface/candle/issues/1887)
- [VarBuilder Usage - Issue #883](https://github.com/huggingface/candle/issues/883)
- [Candle CUDA Issues - Issue #3166](https://github.com/huggingface/candle/issues/3166)
- [Cannot Find Tensor lm_head.weight - Issue #2516](https://github.com/huggingface/candle/issues/2516)
- [Candle Tutorial - Medium](https://medium.com/@cursor0p/lets-learn-candle-%EF%B8%8F-ml-framework-for-rust-9c3011ca3cd9)
- [Porting PyTorch to Candle Tutorial](https://github.com/ToluClassics/candle-tutorial)

### CUDA/Windows
- [CUDA Path Issues - OpenCV Issue #15321](https://github.com/opencv/opencv/issues/15321)
- [CUDA LIB_PATH Issues - cudpp Issue #72](https://github.com/cudpp/cudpp/issues/72)
- [Candle CUDA OOM - Issue #2597](https://github.com/huggingface/candle/issues/2597)
- [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)
- [RTX 3090 Deep Learning Setup - Medium](https://medium.com/@deeplch/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc)
- [RTX 30 Series CUDA Version Requirements - NVIDIA Forums](https://forums.developer.nvidia.com/t/lot-of-ai-repos-cannot-run-on-rtx-30-series-gpu/180646)

### TTS/Vocoder
- [Vocoder Compatibility - mozilla/TTS Issue #607](https://github.com/mozilla/TTS/issues/607)
- [Mel Spectrogram Normalization - mozilla/TTS Issue #377](https://github.com/mozilla/TTS/issues/377)
- [TTS Foundational Knowledge - Towards Data Science](https://towardsdatascience.com/text-to-speech-foundational-knowledge-part-2-4db2a3657335/)

### Silent Failures in ML
- [The Silent Problem - ML Model Failure - arXiv](https://arxiv.org/abs/2204.10227)
- [Detecting Silent Failures in ML - Conf42](https://www.conf42.com/Machine_Learning_2022_Wojtek_Kuberski_detect_silent_failures_ml_models)
- [Why Production ML Fails - Monte Carlo Data](https://www.montecarlodata.com/blog-why-production-machine-learning-fails-and-how-to-fix-it/)
- [When ML Fails Silently - Medium](https://medium.com/@shja22csds/when-machine-learning-fails-silently-the-hidden-cost-of-models-that-seem-to-work-bbc889472880)
