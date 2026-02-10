# IndexTTS2 Rust - Current Status

**Last Updated:** January 19, 2026
**Status:** Pipeline runs end-to-end, quality improvements needed

---

## Progress Summary

```
Phase 1: Foundation        8/8   ✅ Complete
Phase 2: Core Encoders     4/4   ✅ Complete
Phase 3: GPT Generation    5/5   ✅ Complete
Phase 4: Synthesis         4/4   ✅ Complete
Phase 5: Integration       4/4   ✅ Complete
Phase 6: Debug             11/11 ✅ Complete
Phase 7: Weight Loading    5/8   🔶 Partial (BigVGAN loaded)
Phase 8: BigVGAN Vocoder   7/7   ✅ Complete
Phase 9: Testing & Polish  0/12  ⏳ Pending

Total: ~51/63 tasks (81%)
```

---

## What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| Compilation | ✅ | `cargo build --release` succeeds |
| CLI | ✅ | Full inference pipeline runs |
| Generation Loop | ✅ | **Fixed!** Produces 708 mel codes (was 10) |
| BigVGAN Vocoder | ✅ | Weights loaded, generates audio |
| Wav2Vec-BERT | ✅ Downloaded / ❌ Loaded | Architecture mismatch |
| Audio I/O | ✅ | Loading, resampling, saving works |
| Tokenizer | ✅ | SentencePiece working |

---

## Current Blocker: Weight Architecture Mismatch

### The Problem

Downloaded pre-trained models have different layer naming conventions than our Rust implementation:

```
Wav2Vec-BERT 2.0:
  HuggingFace:  encoder.layers.0.self_attn.q_proj.weight
  Our Rust:     layers.0.attention.q_proj.weight

DiT:
  Python:       dit.blocks.0.attn.qkv.weight
  Our Rust:     blocks.0.attention.qkv.weight

Conformer:
  Python:       conformer.layers.0.self_attn.q_proj.weight
  Our Rust:     encoder.layers.0.attention.q_proj.weight
```

### Components Affected

| Component | Weights | Status |
|-----------|---------|--------|
| Wav2Vec-BERT encoder | wav2vec2_bert_2.safetensors | ❌ Random weights |
| DiT flow matching | s2mel.safetensors | ❌ Random weights |
| Conformer encoder | gpt.safetensors | ❌ Random weights |
| Perceiver resampler | gpt.safetensors | ❌ Random weights |
| BigVGAN vocoder | bigvgan_generator.safetensors | ✅ Loaded correctly |

### Solution

Create weight name mapping functions in each model's `load()` implementation:

```rust
fn map_weight_name(original: &str) -> String {
    original
        .replace("encoder.layers", "layers")
        .replace("self_attn", "attention")
        // ... other mappings
}
```

See: `@fix_weight_architecture.md` for detailed fix plan.

---

## Recent Fixes Applied

### Generation Loop Fix (Latest)
- **Issue:** Only generating 10 mel codes instead of hundreds
- **Root Cause:** Stop token was being generated too early
- **Fix:** Adjusted generation loop logic, proper KV-cache handling
- **Result:** Now produces 708 mel codes for typical sentences

### BigVGAN Integration (Complete)
- Downloaded from HuggingFace: `nvidia/bigvgan_v2_22khz_80band_256x`
- Converted weights to safetensors
- Loaded into Rust implementation
- Verified audio output (waveform generation works)

---

## Quick Commands

```bash
# Build
cargo build --release --bin indextts2

# Run inference (audio will be noisy until weights loaded)
cargo run --release --bin indextts2 -- --cpu infer \
  --text "Hello world, this is a test." \
  --speaker "speaker_16k.wav" \
  --output "output.wav"

# Run validation
cargo run --release --bin debug_validate -- \
  --golden-dir debug/golden \
  --component all

# Run tests
cargo test
```

---

## Next Steps

1. **IMMEDIATE: Fix weight architecture mismatch**
   - Map Wav2Vec-BERT layer names
   - Map DiT layer names
   - Map Conformer layer names
   - Map Perceiver layer names
   - Test each component with loaded weights

2. **Phase 9: Testing & Polish**
   - Fix integration test compilation
   - Add unit tests
   - Benchmark performance
   - Clean up warnings
   - Improve error handling
   - Update documentation

---

## Ralph Loop Command

To continue work, run:

```bash
/ralph-loop "Fix weight loading for Wav2Vec-BERT, DiT, Conformer, and Perceiver. The downloaded models have different layer names than our Rust implementation. Create weight name mappings to correctly load the safetensors weights. See @fix_weight_architecture.md. Test by running inference and checking that encoders produce non-random output." --max-iterations 25 --completion-promise "WEIGHTS_LOADED"
```
