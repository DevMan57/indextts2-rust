# IndexTTS2 Rust - Phase 7: Weight Loading

## Overview

Load actual model weights from PyTorch checkpoints into the Rust implementation.
The models use PyTorch `.pth` format which needs conversion to safetensors or direct loading.

## Files Available

```
checkpoints/
├── gpt.pth              (3.5 GB - GPT-2 model)
├── s2mel.pth            (1.2 GB - DiT + Length Regulator)
├── wav2vec2bert_stats.pt (9.3 KB - normalization stats)
├── feat1.pt             (57 KB - speaker matrix)
├── feat2.pt             (375 KB - emotion matrix)
└── qwen0.6bemo4-merge/  (Qwen emotion model)
```

---

## Phase 7 Tasks

### Weight Conversion
- [ ] **P7.1** Create `scripts/convert_weights.py` - PyTorch to safetensors converter
  - Load PyTorch state dict with `torch.load()`
  - Extract tensor data and names
  - Save as safetensors format
  - Handle nested state dicts (model.state_dict())
  - Context7: `Context7:resolve-library-id "safetensors"`

- [ ] **P7.2** Convert GPT weights
  - Map Python keys to Rust struct field names
  - `gpt.text_embedding` → `text_embedding.weight`
  - `gpt.mel_embedding` → `mel_embedding.weight`
  - `gpt.layers.{n}.*` → decoder layers
  - Output: `checkpoints/gpt.safetensors`

- [ ] **P7.3** Convert S2Mel weights
  - Length regulator: `length_regulator.*`
  - DiT blocks: `dit.blocks.{n}.*`
  - Flow matching: already in DiT
  - Output: `checkpoints/s2mel.safetensors`

### Rust Loading
- [ ] **P7.4** Implement `src/models/gpt/unified_voice.rs::load_safetensors()`
  - Use `candle_core::safetensors::load()`
  - Map tensor names to struct fields
  - Verify shapes match config
  - Error on missing/extra keys

- [ ] **P7.5** Implement `src/models/s2mel/dit.rs::load_safetensors()`
  - Load DiT blocks
  - Load AdaLN parameters
  - Load time embeddings

- [ ] **P7.6** Implement `src/models/s2mel/length_regulator.rs::load_safetensors()`
  - Load duration predictor convolutions
  - Load content embedding (if discrete)
  - Load input/output projections

- [ ] **P7.7** Load speaker/emotion matrices
  - `feat1.pt` → `src/models/speaker/campplus.rs`
  - `feat2.pt` → `src/models/emotion/matrix.rs`
  - These are simple tensor files

### Validation
- [ ] **P7.8** Test weight loading
  - Verify tensor shapes match
  - Compare layer outputs to Python reference
  - Test with small input tensors

---

## Key Mappings

### GPT Weight Keys (Python → Rust)
```python
# Python: indextts/gpt/model_v2.py
model.text_embedding.weight          # (12000, 1280)
model.mel_embedding.weight           # (8194, 1280)
model.text_position_embedding.weight # (602, 1280)
model.mel_position_embedding.weight  # (1817, 1280)
model.gpt.ln_f.weight/bias           # Final layer norm
model.gpt.h.{n}.attn.*               # Attention layers
model.gpt.h.{n}.mlp.*                # MLP layers
model.gpt.h.{n}.ln_1/ln_2.*          # Layer norms
model.final_norm.weight/bias
model.mel_head.weight/bias
```

### S2Mel Weight Keys
```python
# Python: indextts/s2mel/modules/
dit.time_embed.mlp.*                 # Time embeddings
dit.content_embed.*                  # Content embeddings
dit.blocks.{n}.norm1/norm2.*         # AdaLN
dit.blocks.{n}.attn.*                # Attention
dit.blocks.{n}.ff.*                  # Feed-forward
dit.final_layer.*                    # Output projection
length_regulator.duration_predictor.* # Duration
length_regulator.input_proj.*
length_regulator.output_proj.*
```

---

## Ralph Loop Command

```bash
/ralph-loop "Implement Phase 7 from @fix_plan_phase7.md. Convert PyTorch weights to safetensors and implement loading in Rust. Use Context7 for safetensors docs." --max-iterations 40 --completion-promise "PHASE7_COMPLETE"
```

---

## Completion Tracking

**Phase 7:** 0/8 complete
- [ ] P7.1 - Weight conversion script
- [ ] P7.2 - Convert GPT weights
- [ ] P7.3 - Convert S2Mel weights
- [ ] P7.4 - Load GPT in Rust
- [ ] P7.5 - Load DiT in Rust
- [ ] P7.6 - Load Length Regulator in Rust
- [ ] P7.7 - Load speaker/emotion matrices
- [ ] P7.8 - Validation tests
