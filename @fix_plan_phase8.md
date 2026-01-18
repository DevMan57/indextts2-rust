# IndexTTS2 Rust - Phase 8: BigVGAN Vocoder Integration

## Overview

Download and integrate the BigVGAN v2 vocoder for mel-to-audio conversion.
The model is hosted on HuggingFace: `nvidia/bigvgan_v2_22khz_80band_256x`

## Model Specifications

```yaml
vocoder:
  type: "bigvgan"
  name: "nvidia/bigvgan_v2_22khz_80band_256x"
  sample_rate: 22050
  mel_bands: 80
  upsample_factor: 256  # hop_length
```

---

## Phase 8 Tasks

### Download & Convert
- [ ] **P8.1** Download BigVGAN from HuggingFace
  - Repository: `nvidia/bigvgan_v2_22khz_80band_256x`
  - Files needed: `bigvgan_generator.pt` or `.safetensors`
  - Config: `config.json`
  - Place in `checkpoints/bigvgan/`
  - HuggingFace MCP: `Hugging Face:model_search query="bigvgan v2 22khz"`

- [ ] **P8.2** Convert BigVGAN weights to safetensors (if needed)
  - Map PyTorch state dict keys
  - Handle generator vs discriminator (only need generator)
  - Output: `checkpoints/bigvgan/generator.safetensors`

### Architecture Refinement
- [ ] **P8.3** Verify `src/models/vocoder/bigvgan.rs` architecture
  - Snake activation: `x + (1/alpha) * sin^2(alpha * x)`
  - Anti-aliased activations (AMPBlock)
  - Multi-receptive field fusion (MRF)
  - Upsample layers: [8, 8, 2, 2] rates
  - Reference: `nvidia/BigVGAN` GitHub

- [ ] **P8.4** Implement missing BigVGAN components
  - `SnakeBeta` activation (learnable alpha)
  - `AMPBlock` with anti-aliasing
  - `ResBlock` variants (type 1 & 2)
  - Proper weight normalization

### Weight Loading
- [ ] **P8.5** Implement `src/models/vocoder/bigvgan.rs::load_safetensors()`
  - Load conv_pre weights
  - Load upsample layers
  - Load MRF blocks
  - Load conv_post weights
  - Verify output shape: (batch, 1, time * 256)

### Testing
- [ ] **P8.6** Test BigVGAN forward pass
  - Input: (1, 80, 100) mel spectrogram
  - Output: (1, 1, 25600) waveform
  - Verify audio quality with known mel input

- [ ] **P8.7** Integrate with pipeline
  - Update `src/inference/pipeline.rs` vocoder loading
  - Test end-to-end: text → mel codes → mel spec → audio
  - Verify sample rate and duration

---

## BigVGAN Architecture

```
Input: (batch, 80, time) mel spectrogram

conv_pre: Conv1d(80, 512, 7, padding=3)

For each upsample rate [8, 8, 2, 2]:
    ConvTranspose1d(ch, ch//2, kernel, stride=rate)
    MRF Block (Multi-Receptive Field):
        ResBlock1 with kernels [3, 7, 11]
        ResBlock2 with kernels [3, 5, 7]
    Snake activation (learnable alpha)

conv_post: Conv1d(32, 1, 7, padding=3)
tanh activation

Output: (batch, 1, time * 256) waveform
```

## Weight Key Mappings

```python
# PyTorch keys → Rust field names
generator.conv_pre.weight/bias
generator.ups.{n}.weight/bias           # ConvTranspose1d
generator.resblocks.{n}.convs1.{m}.*    # ResBlock convs
generator.resblocks.{n}.convs2.{m}.*
generator.resblocks.{n}.activations.{m}.alpha  # Snake alpha
generator.conv_post.weight/bias
```

---

## Ralph Loop Command

```bash
/ralph-loop "Implement Phase 8 from @fix_plan_phase8.md. Download BigVGAN, convert weights, and integrate vocoder. Test mel-to-audio conversion." --max-iterations 35 --completion-promise "PHASE8_COMPLETE"
```

---

## Completion Tracking

**Phase 8:** 0/7 complete
- [ ] P8.1 - Download BigVGAN
- [ ] P8.2 - Convert weights
- [ ] P8.3 - Verify architecture
- [ ] P8.4 - Implement missing components
- [ ] P8.5 - Load weights in Rust
- [ ] P8.6 - Test forward pass
- [ ] P8.7 - Pipeline integration
