# IndexTTS2 Rust Rewrite Project

## Quick Start with Ralph Wiggum

The Ralph Wiggum plugin is already installed in Claude Code. Start autonomous development:

```bash
# In Claude Code, run:
/ralph-loop "Implement Phase 1 from @fix_plan.md. Use Context7 for crate docs." --max-iterations 30 --completion-promise "PHASE1_COMPLETE"
```

**Cancel anytime:** `/cancel-ralph`

---

## 🔧 MCP Tools - USE THESE FIRST!

### Context7 (MANDATORY for Rust Development)

**ALWAYS use Context7 before writing ANY Rust code:**

```bash
# Step 1: Find the library ID
Context7:resolve-library-id "candle machine learning"

# Step 2: Get focused documentation
Context7:get-library-docs "/huggingface/candle" topic="transformer attention kv-cache"
```

**Key Libraries:**
| Purpose | Query |
|---------|-------|
| ML Framework | `Context7:resolve-library-id "candle"` |
| Audio I/O | `Context7:resolve-library-id "cpal"` |
| Resampling | `Context7:resolve-library-id "rubato"` |
| FFT | `Context7:resolve-library-id "rustfft"` |
| Tokenization | `Context7:resolve-library-id "tokenizers"` |

### HuggingFace MCP

```bash
Hugging Face:model_search query="BigVGAN vocoder"
Hugging Face:paper_search query="conformer encoder speech"
```

---

## 📐 Architecture Overview

```
IndexTTS2 Pipeline:
┌─────────────────────────────────────────────────────────────────────┐
│  Input: Text + Speaker Reference Audio + (Optional) Emotion Audio  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. TEXT PROCESSING                                                 │
│     - TextNormalizer: Normalize input text                          │
│     - TextTokenizer: BPE tokenization → token IDs                   │
│     Python: indextts/utils/front.py                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. SPEAKER CONDITIONING                                            │
│     - Wav2Vec-BERT 2.0 → semantic embeddings                        │
│     - CAMPPlus → speaker style vector (192-dim)                     │
│     Python: indextts/s2mel/wav2vecbert_extract.py                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. GPT-2 AUTOREGRESSIVE GENERATION (UnifiedVoice)                  │
│     - model_dim: 1280, layers: 24, heads: 20                        │
│     - Conformer encoder + Perceiver resampler                       │
│     - Generate mel codes (stop_token=8193)                          │
│     Python: indextts/gpt/model_v2.py                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. S2MEL Flow Matching                                             │
│     - DiT: 13 layers, 512 hidden                                    │
│     - CFM: 25 steps, cfg_rate=0.7                                   │
│     - Output: 80-band mel spectrogram                               │
│     Python: indextts/s2mel/modules/flow_matching.py                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. VOCODER (BigVGAN v2)                                            │
│     - Mel → 22050 Hz waveform                                       │
│     Python: indextts/BigVGAN/bigvgan.py                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🦀 Rust Module Structure

```
src/
├── main.rs                  # CLI entry point
├── lib.rs                   # Library exports
├── config/                  # YAML config parsing
├── text/                    # Tokenizer, normalizer
├── audio/                   # Loader, resampler, mel, output
├── models/
│   ├── semantic/            # Wav2Vec-BERT, codec
│   ├── speaker/             # CAMPPlus
│   ├── gpt/                 # UnifiedVoice, Conformer, Perceiver
│   ├── s2mel/               # DiT, Flow Matching
│   └── vocoder/             # BigVGAN
└── inference/               # Pipeline, streaming
```

---

## 📦 Key Configuration

```yaml
# From checkpoints/config.yaml
gpt:
  model_dim: 1280
  layers: 24
  heads: 20
  max_mel_tokens: 1815
  max_text_tokens: 600
  number_mel_codes: 8194
  stop_mel_token: 8193

s2mel:
  sr: 22050
  DiT:
    hidden_dim: 512
    depth: 13
```

---

## 🚀 Ralph Loop Commands by Phase

### Completed Phases (1-6)
```bash
# Phases 1-6 are complete - see @fix_plan.md
```

### Remaining Phases (7-9)
```bash
# Phase 7: Weight Loading
/ralph-loop "Implement Phase 7 from @fix_plan_phase7.md. Convert PyTorch weights to safetensors and implement loading in Rust. Use Context7 for safetensors docs." --max-iterations 40 --completion-promise "PHASE7_COMPLETE"

# Phase 8: BigVGAN Vocoder
/ralph-loop "Implement Phase 8 from @fix_plan_phase8.md. Download BigVGAN, convert weights, and integrate vocoder. Test mel-to-audio conversion." --max-iterations 35 --completion-promise "PHASE8_COMPLETE"

# Phase 9: Testing & Polish
/ralph-loop "Implement Phase 9 from @fix_plan_phase9.md. Fix test compilation, add comprehensive tests, and polish for production. Run all tests with cargo test." --max-iterations 40 --completion-promise "PHASE9_COMPLETE"
```

---

## 📚 Reference Files

| Rust Module | Python Reference |
|-------------|------------------|
| `config/` | `checkpoints/config.yaml` |
| `text/tokenizer.rs` | `indextts/utils/front.py` |
| `models/gpt/unified_voice.rs` | `indextts/gpt/model_v2.py` |
| `models/gpt/conformer.rs` | `indextts/gpt/conformer_encoder.py` |
| `models/s2mel/dit.rs` | `indextts/s2mel/modules/diffusion_transformer.py` |
| `models/vocoder/bigvgan.rs` | `indextts/BigVGAN/bigvgan.py` |

---

## ✅ Progress Tracker

Check `@fix_plan.md` for detailed task list.

### Completed
- [x] Phase 1: Foundation (8 tasks)
- [x] Phase 2: Core Models (4 tasks)
- [x] Phase 3: Generation (5 tasks)
- [x] Phase 4: Synthesis (4 tasks)
- [x] Phase 5: Integration (4 tasks)
- [x] Phase 6: Debug & Validate (11 tasks)

### Remaining
- [ ] Phase 7: Weight Loading (8 tasks) - see `@fix_plan_phase7.md`
- [ ] Phase 8: BigVGAN Vocoder (7 tasks) - see `@fix_plan_phase8.md`
- [ ] Phase 9: Testing & Polish (12 tasks) - see `@fix_plan_phase9.md`

**Total:** 36/63 tasks complete (57%)
