# Technology Stack Research

**Project:** IndexTTS2 Rust - Weight Loading and CUDA Support
**Researched:** 2026-01-23
**Overall Confidence:** MEDIUM-HIGH (Context7 unavailable for Candle, verified via official docs and GitHub)

## Executive Summary

This research covers two critical areas for the IndexTTS2 Rust project:
1. **Weight Loading with Name Remapping** - How to load HuggingFace SafeTensors weights when tensor names differ from Rust model structure
2. **CUDA Setup on Windows with RTX 3090** - Step-by-step configuration for GPU acceleration

The recommended approach uses Candle's `VarBuilder::rename_f()` method for systematic name remapping, combined with direct HashMap-based loading for complex transformations (which the project already uses effectively). For CUDA, the setup requires CUDA Toolkit 12.x with careful DLL management on Windows.

---

## Candle Weight Loading

### Overview

Candle provides multiple approaches for loading pre-trained weights from SafeTensors files. The project currently uses **Candle v0.8** as specified in `Cargo.toml`.

### Method 1: VarBuilder with rename_f (Recommended for Simple Remapping)

The `rename_f` method creates a VarBuilder that transforms tensor names before retrieval. This is ideal when the mapping is systematic (e.g., prefix changes).

**API Signature:**
```rust
pub fn rename_f<F: Fn(&str) -> String + Sync + Send + 'static>(self, f: F) -> Self
```

**Example - Systematic Prefix Remapping:**
```rust
use candle_nn::VarBuilder;
use candle_core::{DType, Device};

// Load from SafeTensors
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)?
};

// Remap "encoder.layers.X.self_attn.linear_q" -> "layers.X.attention.q_proj"
let vb = vb.rename_f(|name: &str| {
    name
        .replace("encoder.layers", "layers")
        .replace("self_attn.linear_q", "attention.q_proj")
        .replace("self_attn.linear_k", "attention.k_proj")
        .replace("self_attn.linear_v", "attention.v_proj")
        .replace("self_attn.linear_out", "attention.out_proj")
        .replace("ffn1.intermediate_dense", "ffn.fc1")
        .replace("ffn2.output_dense", "ffn.fc2")
});

// Now use vb with Rust model names
let q_weight = vb.pp("layers.0.attention").get((hidden_dim, hidden_dim), "q_proj.weight")?;
```

**Source:** [VarBuilder rename_f documentation](https://docs.rs/candle-nn/latest/candle_nn/var_builder/type.VarBuilder.html)

### Method 2: Direct HashMap Loading (Current Project Approach)

For complex transformations like tensor splitting or transposition, direct HashMap access is more flexible. The project already uses this pattern effectively in `src/models/gpt/weights.rs`.

**Pattern:**
```rust
use candle_core::safetensors;
use std::collections::HashMap;

// Load all tensors into HashMap
let tensors: HashMap<String, Tensor> = safetensors::load(path, &device)?;

// Manual remapping with transformations
fn load_attention_weights(
    tensors: &HashMap<String, Tensor>,
    hf_prefix: &str,  // "encoder.layers.0.self_attn"
) -> Result<(Linear, Linear, Linear, Linear)> {
    // Map HuggingFace names to tensors
    let q_weight = tensors
        .get(&format!("{}.linear_q.weight", hf_prefix))
        .ok_or_else(|| anyhow::anyhow!("q weight not found"))?
        .clone();

    // Optional: transpose if needed for Candle Linear
    // Candle expects [out_features, in_features], same as PyTorch
    let q_proj = Linear::new(q_weight, q_bias);

    // ... similar for k, v, out
    Ok((q_proj, k_proj, v_proj, out_proj))
}
```

### Method 3: VarBuilder with push_prefix (pp)

Use `pp` (alias for `push_prefix`) to navigate the weight hierarchy - like `cd` in a filesystem.

```rust
// Navigate to nested weights
let encoder_vb = vb.pp("encoder");
let layer_0_vb = encoder_vb.pp("layers").pp("0");
let attn_vb = layer_0_vb.pp("self_attn");

// Get specific weight
let q_weight = attn_vb.get((hidden_dim, hidden_dim), "linear_q.weight")?;
```

**Source:** [VarBuilder push_prefix](https://github.com/huggingface/candle/discussions/2076)

### Method 4: Custom Renamer Trait

For complex mappings, implement the `Renamer` trait:

```rust
use candle_nn::var_builder::Renamer;

struct HuggingFaceToRustMapper;

impl Renamer for HuggingFaceToRustMapper {
    fn rename(&self, name: &str) -> String {
        // Complex mapping logic
        let mut result = name.to_string();

        // Handle Wav2Vec-BERT specific mappings
        if result.contains("encoder.layers") {
            result = result
                .replace("self_attn.linear_q", "attention.query")
                .replace("self_attn.linear_k", "attention.key")
                .replace("self_attn.linear_v", "attention.value")
                .replace("self_attn.linear_out", "attention.output");
        }

        result
    }
}
```

### Specific Mappings for IndexTTS2 Components

Based on the project's model files, here are the required mappings:

#### Wav2Vec-BERT 2.0 (from `src/models/semantic/wav2vec_bert.rs`)

| HuggingFace Name | Rust Expected Name |
|------------------|-------------------|
| `encoder.layers.{i}.self_attn.linear_q.weight` | `layers.{i}.attention.query.weight` |
| `encoder.layers.{i}.self_attn.linear_k.weight` | `layers.{i}.attention.key.weight` |
| `encoder.layers.{i}.self_attn.linear_v.weight` | `layers.{i}.attention.value.weight` |
| `encoder.layers.{i}.self_attn.linear_out.weight` | `layers.{i}.attention.output.weight` |
| `encoder.layers.{i}.ffn1.intermediate_dense.weight` | `layers.{i}.ffn.fc1.weight` |
| `encoder.layers.{i}.ffn2.output_dense.weight` | `layers.{i}.ffn.fc2.weight` |

#### GPT UnifiedVoice (from `src/models/gpt/weights.rs`)

The current `Gpt2LayerWeights::load()` already handles the GPT-2 format correctly:
- `gpt.h.{i}.attn.c_attn.weight` - Combined QKV (requires splitting)
- `gpt.h.{i}.attn.c_proj.weight` - Output projection
- `gpt.h.{i}.ln_1.weight` - Pre-attention LayerNorm
- `gpt.h.{i}.mlp.c_fc.weight` - FFN first layer
- `gpt.h.{i}.mlp.c_proj.weight` - FFN second layer
- `gpt.h.{i}.ln_2.weight` - Pre-FFN LayerNorm

#### DiT (from `src/models/s2mel/weights.rs`)

Already correctly implemented - weights match expected format:
- `cfm.estimator.transformer.layers.{i}.attention.wqkv.weight`
- `cfm.estimator.transformer.layers.{i}.feed_forward.w1.weight`

### Weight Transposition Rules

**Critical:** Candle's `Linear` layer expects weights in `[out_features, in_features]` format, same as PyTorch. However:

1. **Direct loading usually works** - PyTorch saves in `[out_features, in_features]`
2. **Transpose when splitting combined weights** - See `split_qkv` in `weights.rs`
3. **Check tensor shapes** - If forward pass produces wrong dimensions, weights likely need transpose

```rust
// Safe pattern for ambiguous cases
let weight = tensors.get("layer.weight")?;
let (dim_0, dim_1) = weight.dims2()?;

// Verify expected dimensions
let weight = if dim_0 == expected_out && dim_1 == expected_in {
    weight.clone()
} else if dim_0 == expected_in && dim_1 == expected_out {
    weight.t()?.contiguous()?
} else {
    bail!("Unexpected weight shape: {:?}", weight.dims());
};
```

---

## CUDA Setup (Windows + RTX 3090)

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| NVIDIA Driver | 450+ | RTX 30-series requires Ampere-compatible drivers |
| CUDA Toolkit | 12.0 - 12.9 | CUDA 13.0 has issues with current cudarc |
| cuDNN | 8.0+ or 9.x | Match major version to CUDA |
| RTX 3090 | - | Compute capability 8.6 |

**Source:** [Candle Installation Guide](https://huggingface.github.io/candle/guide/installation.html)

### Step 1: Verify GPU and Driver

```powershell
# Check NVIDIA driver and CUDA availability
nvidia-smi

# Verify compute capability (should show 8.6 for RTX 3090)
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Step 2: Install CUDA Toolkit

1. Download CUDA Toolkit 12.4 or 12.6 from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Run installer with default options
3. Verify installation:

```powershell
# Verify nvcc is available
nvcc --version
# Should show: Cuda compilation tools, release 12.x
```

### Step 3: Install cuDNN

1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. Extract and copy files:

```powershell
# From extracted cuDNN folder:
# Copy bin\cudnn*.dll to CUDA bin
copy "cudnn-*\bin\cudnn*.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\"

# Copy include\cudnn*.h to CUDA include
copy "cudnn-*\include\cudnn*.h" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\"

# Copy lib\x64\cudnn*.lib to CUDA lib
copy "cudnn-*\lib\x64\cudnn*.lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\"
```

**Source:** [NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)

### Step 4: Set Environment Variables

```powershell
# Set via System Properties > Environment Variables, or PowerShell:
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "Machine")
[Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "Machine")

# Add to PATH
$path = [Environment]::GetEnvironmentVariable("PATH", "Machine")
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
$cudaLibnvvp = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp"
[Environment]::SetEnvironmentVariable("PATH", "$path;$cudaBin;$cudaLibnvvp", "Machine")
```

### Step 5: Windows-Specific DLL Fix

If you encounter `LoadLibraryExW { source: Os { code: 126 } }` errors:

```powershell
# These DLLs need to be in PATH or renamed
# Option A: Add CUDA bin to PATH (preferred)
# Option B: Copy and rename DLLs (workaround)

# From Administrator PowerShell:
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
copy cublas64_12.dll cublas.dll
copy curand64_10.dll curand.dll

# nvcuda.dll is typically in System32
cd C:\Windows\System32
copy nvcuda.dll cuda.dll
```

**Source:** [GitHub Issue #2410](https://github.com/huggingface/candle/issues/2410)

### Step 6: Build Candle with CUDA

```powershell
# Set compute capability for RTX 3090
$env:CUDA_COMPUTE_CAP = "86"

# Build with CUDA feature
cargo build --release --features cuda

# Or if using cuDNN for additional speedups
cargo build --release --features cuda,cudnn
```

### Step 7: Verify CUDA Works

```rust
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    // Attempt to create CUDA device
    let device = Device::cuda_if_available(0)?;

    match &device {
        Device::Cuda(_) => println!("CUDA device available!"),
        Device::Cpu => println!("Falling back to CPU"),
        _ => println!("Other device"),
    }

    Ok(())
}
```

---

## Key APIs

### Weight Loading APIs

| API | Purpose | Use Case |
|-----|---------|----------|
| `safetensors::load(path, device)` | Load all tensors to HashMap | Complex transformations needed |
| `VarBuilder::from_mmaped_safetensors(paths, dtype, device)` | Memory-mapped loading | Large models, structured access |
| `vb.rename_f(closure)` | Transform tensor names | Systematic name differences |
| `vb.pp(prefix)` / `vb.push_prefix(prefix)` | Navigate hierarchy | Nested model structure |
| `vb.get((dims), "name")` | Retrieve tensor | Direct weight access |
| `vb.contains_tensor("name")` | Check existence | Optional weights |

### Device APIs

| API | Purpose |
|-----|---------|
| `Device::Cpu` | CPU computation |
| `Device::cuda_if_available(ordinal)` | Use CUDA if available, else CPU |
| `Device::new_cuda(ordinal)` | Require CUDA (fails if unavailable) |

### Tensor Operations for Weight Processing

| Operation | Purpose | Example |
|-----------|---------|---------|
| `tensor.t()` | Transpose 2D | Weight shape conversion |
| `tensor.i(range)` | Index/slice | Split combined weights |
| `tensor.contiguous()` | Ensure contiguous memory | After transpose/slice |
| `tensor.dims2()` | Get 2D shape | Verify weight dimensions |

---

## What NOT To Do

### Weight Loading Mistakes

1. **Do NOT assume weight shapes match**
   - Always verify dimensions before use
   - PyTorch and Candle use same convention, but custom models may differ

2. **Do NOT forget `.contiguous()` after transpose/slice**
   ```rust
   // WRONG - may cause issues
   let weight = tensor.t()?;

   // CORRECT
   let weight = tensor.t()?.contiguous()?;
   ```

3. **Do NOT use `VarBuilder::from_varmap` for inference**
   - `from_varmap` is for training (enables gradients)
   - Use `from_mmaped_safetensors` for inference

4. **Do NOT ignore missing keys**
   ```rust
   // WRONG - silent failure
   let weight = tensors.get("key").unwrap_or_default();

   // CORRECT - explicit error handling
   let weight = tensors
       .get("key")
       .ok_or_else(|| anyhow!("Required weight 'key' not found"))?;
   ```

5. **Do NOT mix model versions without checking**
   - Different model versions have different weight names
   - Always verify checkpoint matches expected architecture

### CUDA Setup Mistakes

1. **Do NOT use CUDA 13.0 with current cudarc**
   - cudarc v0.16.x has issues with CUDA 13
   - Stick with CUDA 12.x for stability
   - **Source:** [GitHub Issue #3249](https://github.com/huggingface/candle/issues/3249)

2. **Do NOT forget CUDA_COMPUTE_CAP**
   ```powershell
   # WRONG - may build for wrong architecture
   cargo build --features cuda

   # CORRECT - specify compute capability
   $env:CUDA_COMPUTE_CAP = "86"
   cargo build --features cuda
   ```

3. **Do NOT skip DLL verification on Windows**
   - cudarc's `bindgen_cuda` has issues finding CUDA on Windows
   - Verify DLLs are in PATH before building
   - **Source:** [GitHub Issue #3166](https://github.com/huggingface/candle/issues/3166)

4. **Do NOT use CUDA 10.x or 11.0-11.3 with RTX 3090**
   - RTX 30-series (Ampere) requires CUDA 11.0+
   - Best compatibility with CUDA 12.x

5. **Do NOT ignore memory limits**
   - RTX 3090 has 24GB VRAM
   - Large models may still OOM
   - Use memory-mapped loading for large checkpoints

### General Candle Mistakes

1. **Do NOT use interactive git commands**
   - `git rebase -i` and similar are not supported in this environment

2. **Do NOT assume training data is current**
   - Always verify API details against official docs
   - Candle is actively developed; APIs may change

---

## Recommended Stack for IndexTTS2

### Current Stack (Keep)

| Component | Version | Confidence |
|-----------|---------|------------|
| candle-core | 0.8 | HIGH |
| candle-nn | 0.8 | HIGH |
| candle-transformers | 0.8 | HIGH |
| tokenizers | 0.20 | HIGH |
| hf-hub | 0.3 | HIGH |

### CUDA Requirements

| Component | Recommended Version | Notes |
|-----------|---------------------|-------|
| CUDA Toolkit | 12.4 or 12.6 | Avoid 13.0 |
| cuDNN | 9.x | Match CUDA major version |
| NVIDIA Driver | 535+ | For CUDA 12.x |

### Build Configuration

```toml
# Cargo.toml features (already correct)
[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
```

---

## Sources

### Official Documentation
- [Candle Installation Guide](https://huggingface.github.io/candle/guide/installation.html)
- [VarBuilder API Documentation](https://docs.rs/candle-nn/latest/candle_nn/var_builder/type.VarBuilder.html)
- [VarBuilderArgs Documentation](https://docs.rs/candle-nn/latest/candle_nn/var_builder/struct.VarBuilderArgs.html)
- [NVIDIA cuDNN Installation](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)

### GitHub References
- [Candle Repository](https://github.com/huggingface/candle)
- [VarBuilder Source Code](https://github.com/huggingface/candle/blob/main/candle-nn/src/var_builder.rs)
- [Cudarc Repository](https://github.com/coreylowman/cudarc)
- [Candle Tutorial - Porting Models](https://github.com/ToluClassics/candle-tutorial)

### Issue Discussions
- [CUDA 12.6 Support Issue](https://github.com/huggingface/candle/issues/2410)
- [Windows CUDA Issues](https://github.com/huggingface/candle/issues/3166)
- [CUDA 13.0 Issues](https://github.com/huggingface/candle/issues/3249)
- [VarBuilder Discussion](https://github.com/huggingface/candle/discussions/2076)

### Tutorials and Guides
- [Using HuggingFace with Rust](https://www.shuttle.dev/blog/2024/05/01/using-huggingface-rust)
- [Building GPT from Scratch in Candle](https://www.perceptivebits.com/building-gpt-from-scratch-in-rust-and-candle/)
- [Medium: Learn Candle ML](https://medium.com/@cursor0p/lets-learn-candle-%EF%B8%8F-ml-framework-for-rust-9c3011ca3cd9)

---

## Confidence Assessment

| Topic | Level | Reason |
|-------|-------|--------|
| VarBuilder APIs | MEDIUM-HIGH | Verified via official docs.rs and GitHub source |
| rename_f pattern | HIGH | Confirmed in official documentation with examples |
| Weight transposition | HIGH | Verified against project code and tutorials |
| CUDA 12.x setup | MEDIUM | Multiple sources agree, some Windows-specific gaps |
| Windows DLL issues | MEDIUM | Documented in GitHub issues, workarounds verified |
| CUDA 13.0 warning | HIGH | Recent issue confirms incompatibility |

---

## Open Questions

1. **cudarc Windows improvements** - Is the bindgen_cuda PR merged? May simplify Windows setup.
2. **Candle 0.9 changes** - Are there breaking changes to VarBuilder API?
3. **cuDNN integration** - Does `--features cudnn` work reliably on Windows?
