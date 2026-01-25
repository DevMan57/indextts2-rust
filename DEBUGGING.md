# IndexTTS2-Rust Debugging Log

This document tracks all debugging efforts to resolve the audio quality issue where the generated output sounds like noise/rumbling water instead of speech.

## Problem Summary

The Rust implementation produces audio that:
- Has correct duration (proportional to text length)
- Has mel spectrogram values in a plausible range (-10 to -6)
- Passes through BigVGAN vocoder without errors
- But sounds like rumbling water/noise, not speech

## Current Pipeline Statistics

From a typical inference run ("Hello world"):
```
Speaker mel:    mean=-6.48, range=[-11.5, 4.4]
Generated mel:  mean=-10.09, range=[-14.9, -3.1]
Difference:     3.6 dB (generated is darker/lower energy)

DiT velocity:   starts at -6.5, converges to -10 over 25 steps
Audio output:   22050 Hz, 16-bit PCM, duration matches expected
```

## Fixes Applied (Still Not Working)

### 1. FinalLayer AdaLN Fix (Jan 25, 2026)
**Location:** `src/models/s2mel/dit.rs:1155-1177`

**Problem:** DiT was producing velocity with mean +0.4 instead of -6.5

**Root cause (3 bugs):**
1. Missing SiLU activation before adaln linear
2. Chunk order reversed: was `scale, shift` but Python uses `shift, scale`
3. Wrong modulate formula: was `x * scale + shift`, should be `x * (1 + scale) + shift`

**Fix:**
```rust
fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
    let t_emb_silu = silu(t_emb)?;  // Added SiLU
    let params = self.adaln.forward(&t_emb_silu)?;
    let chunks = params.chunk(2, D::Minus1)?;
    let shift = &chunks[0];  // Fixed order
    let scale = &chunks[1];

    let x = self.norm.forward(x)?;
    let scale_plus_one = (scale + 1.0)?;  // Fixed formula
    let x = (x.broadcast_mul(&scale_plus_one)?).broadcast_add(&shift)?;

    self.linear.forward(&x)
}
```

**Result:** DiT velocity now has correct sign (-6.5 instead of +0.4), but audio still noise.

### 2. LengthRegulator GroupNorm Fix (Jan 25, 2026)
**Location:** `src/models/s2mel/length_regulator.rs`

**Problem:** Python uses GroupNorm, Rust was using LayerNorm

**Fix:** Implemented GroupNorm with groups=1, changed order to interpolate→conv blocks

**Result:** Content features statistics improved, but audio still noise.

### 3. prompt_x Format Fix (Jan 25, 2026)
**Location:** `src/inference/pipeline.rs:593-616`

**Problem:** Was passing full speaker mel as prompt_x, but Python creates:
```python
prompt_x = torch.zeros_like(x)
prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
```

**Fix:** Create prompt_x as mostly zeros with only first `prompt_len` frames from reference mel.

**Result:** prompt_x mean changed from -6.5 to -0.92, but audio still noise.

## Components Verified Working

### Wav2Vec-BERT Encoder
- 24/24 layers loaded
- Feature projection: 160 → 1024
- Stats file loaded for normalization

### GPT UnifiedVoice
- 24 transformer layers loaded
- Conformer encoder: 24 layers
- Perceiver resampler: 32 latents, 2 layers
- Generates mel codes, stops at token 8193

### DiT (Diffusion Transformer)
- 13/13 transformer blocks loaded
- UViT skip connections loaded
- WaveNet post-processor: 8 layers
- FinalLayer with weight-normalized linear

### BigVGAN Vocoder
- 667 tensors loaded
- Snake activation working
- MRF (Multi-Receptive Field) blocks working
- Produces audio in valid range

## Remaining Hypotheses

### 1. Flow Matching Integration Direction
The Euler ODE integrates from t=0 (noise) to t=1 (data). Verify:
- Is the velocity sign correct for this direction?
- Python: `x = x + dt * dphi_dt` - are we matching this exactly?

### 2. Classifier-Free Guidance
CFG formula: `v = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond`
- Currently cfg_rate = 0.7
- Unconditional uses zero style, zero cond, zero prompt

### 3. Mel Spectrogram Range Mismatch
- Generated mel mean: -10.09
- Speaker mel mean: -6.48
- 3.6 dB difference is significant

### 4. BigVGAN Input Format
- Expects [B, C, T] or [B, T, C]?
- Currently transposing before vocoder

### 5. DiT Transformer Block Implementation
The transformer uses AdaptiveLayerNorm with formula `weight * norm(x) + bias` (no +1).
This is DIFFERENT from FinalLayer which uses `(1 + scale) * x + shift`.
- Verified this matches Python's `AdaptiveLayerNorm.forward()`

## Debug Output Explanation

```
DEBUG DiT input: x_mean=0.0031, prompt_mean=-6.7583, cond_mean=-0.0007, style_mean=0.0027
```
- `x_mean`: Current sample being denoised (starts as noise ~0)
- `prompt_mean`: Reference mel region (should be ~-6.5)
- `cond_mean`: Content/semantic features from LengthRegulator
- `style_mean`: Speaker embedding from CAMPPlus

```
DEBUG DiT output: mean=-6.4836, std=1.6755
```
- Velocity prediction, should push sample toward target distribution
- After FinalLayer fix, this now has correct negative sign

```
DEBUG: Generated mel mean: -10.0921, Speaker mel mean: -6.4840, diff: 3.6081
```
- The 3.6 dB gap suggests the flow matching isn't converging to the right target

## Files to Investigate

1. `src/models/s2mel/flow_matching.rs` - ODE integration, CFG
2. `src/models/s2mel/dit.rs` - Transformer blocks, attention
3. `src/inference/pipeline.rs` - Component orchestration
4. `src/models/vocoder/bigvgan.rs` - Mel to audio conversion

## Python Reference Implementation

The reference implementation is in `C:\AI\index-tts\indextts\`:
- `s2mel/modules/flow_matching.py` - CFM sample loop
- `s2mel/modules/diffusion_transformer.py` - DiT, FinalLayer
- `s2mel/modules/gpt_fast/model.py` - AdaptiveLayerNorm

## How to Help

1. Compare intermediate tensors between Python and Rust at each pipeline stage
2. Check if there are additional normalization/scaling steps we're missing
3. Verify the mel spectrogram format (log scale, normalization) matches exactly
4. Test individual components in isolation with known inputs

## Test Commands

```bash
# Build
cargo build --release

# Run inference
cargo run --release --bin indextts2 -- --cpu infer \
  --text "Hello world" \
  --speaker checkpoints/speaker_16k.wav \
  --output output.wav

# Run tests
cargo test --release
```
