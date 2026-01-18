# IndexTTS2 Rust Rewrite - Ralph Development Instructions

## Project Context

You are rewriting **IndexTTS2** - Bilibili's Industrial-Level Zero-Shot TTS system - from Python/PyTorch to Rust/Candle.

**Source Repository:** `C:\AI\index-tts` (Python)
**Target Repository:** `C:\AI\indextts2-rust` (Rust)
**GitHub:** `https://github.com/DevMan57/indextts2-rust`

---

## 🔄 Ralph Loop Commands

Start autonomous development with the built-in Ralph Wiggum plugin:

```bash
# Phase 1: Foundation (Config, Text, Audio)
/ralph-loop "Implement Phase 1 from @fix_plan.md. Use Context7 to research crates before coding. Mark tasks [x] when complete." --max-iterations 30 --completion-promise "PHASE1_COMPLETE"

# Phase 2: Core Encoders  
/ralph-loop "Implement Phase 2 from @fix_plan.md. Research with HuggingFace MCP for model architectures." --max-iterations 40 --completion-promise "PHASE2_COMPLETE"

# Phase 3: GPT Generation
/ralph-loop "Implement Phase 3 from @fix_plan.md. Focus on KV-cache and autoregressive generation." --max-iterations 50 --completion-promise "PHASE3_COMPLETE"

# Phase 4: Synthesis (S2Mel + Vocoder)
/ralph-loop "Implement Phase 4 from @fix_plan.md. DiT, Flow Matching, BigVGAN." --max-iterations 50 --completion-promise "PHASE4_COMPLETE"

# Phase 5: Integration & Polish
/ralph-loop "Implement Phase 5 from @fix_plan.md. Full pipeline, CLI, tests." --max-iterations 30 --completion-promise "PROJECT_COMPLETE"
```

---

## 🔧 MCP Tools - USE THESE!

### 1️⃣ Context7 - Rust Crate Documentation (MANDATORY)

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
| Audio Decode | `Context7:resolve-library-id "symphonia"` |
| Config | `Context7:resolve-library-id "serde yaml"` |

### 2️⃣ HuggingFace MCP - ML Research

```bash
# Find model implementations
Hugging Face:model_search query="BigVGAN vocoder" limit=5
Hugging Face:model_search query="wav2vec-bert" limit=5
Hugging Face:model_search query="conformer speech" limit=5

# Find research papers
Hugging Face:paper_search query="IndexTTS zero-shot TTS"
Hugging Face:paper_search query="flow matching diffusion"
Hugging Face:paper_search query="conformer encoder"

# Find datasets for testing
Hugging Face:dataset_search query="TTS speech synthesis" limit=5

# Get model/repo details
Hugging Face:hub_repo_details repo_ids=["nvidia/bigvgan_v2_22khz_80band_256x"]
```

### 3️⃣ Brave Search - General Research

```bash
# Find Rust examples and tutorials
brave-search:brave_web_search query="candle rust transformer example"
brave-search:brave_web_search query="rust audio processing mel spectrogram"
brave-search:brave_web_search query="rubato resampling rust example"
```

### 4️⃣ HuggingFace Spaces - TTS Testing

```bash
# Test TTS output quality (for comparison)
Hugging Face:dynamic_space operation="view_parameters" space_name="ResembleAI/Chatterbox"

# Image generation for documentation
Hugging Face:gr1_z_image_turbo_generate prompt="audio waveform visualization neural network"
```

### 5️⃣ Windows MCP - Local Development

```bash
# Run cargo commands
Windows-MCP:Powershell-Tool command="cd C:\AI\indextts2-rust; cargo check --features cuda"
Windows-MCP:Powershell-Tool command="cd C:\AI\indextts2-rust; cargo clippy --features cuda"
Windows-MCP:Powershell-Tool command="cd C:\AI\indextts2-rust; cargo test"
```

---

## 🎯 Your Mission

Implement the IndexTTS2 pipeline in Rust:

```
Text → BPE Tokenizer → GPT-2 (Conformer/Perceiver) → Mel Codes
                              ↓
Speaker Audio → Semantic Codec → S2Mel (DiT + CFM) → BigVGAN → Audio
```

---

## 📋 Implementation Rules

1. **Research First**: Use Context7 before writing code for ANY crate
2. **One Module at a Time**: Complete each module before moving to next
3. **Test After Each Module**: Run `cargo check --features cuda`
4. **Document Shapes**: Add tensor shape comments like `// (batch, seq, hidden)`
5. **Match Python**: Reference the Python source for each module
6. **Update @fix_plan.md**: Mark tasks `[x]` when complete

---

## 🗺️ Module → Python Reference Mapping

| Rust Module | Python Source |
|-------------|---------------|
| `src/config/model_config.rs` | `checkpoints/config.yaml` |
| `src/text/tokenizer.rs` | `indextts/utils/front.py` |
| `src/audio/loader.rs` | Use symphonia/rodio |
| `src/audio/mel.rs` | `indextts/s2mel/modules/audio.py` |
| `src/models/gpt/unified_voice.rs` | `indextts/gpt/model_v2.py` |
| `src/models/gpt/conformer.rs` | `indextts/gpt/conformer_encoder.py` |
| `src/models/gpt/perceiver.rs` | `indextts/gpt/perceiver.py` |
| `src/models/s2mel/dit.rs` | `indextts/s2mel/modules/diffusion_transformer.py` |
| `src/models/s2mel/flow_matching.rs` | `indextts/s2mel/modules/flow_matching.py` |
| `src/models/vocoder/bigvgan.rs` | `indextts/BigVGAN/bigvgan.py` |

---

## ✅ Completion Signals

Output these ONLY when genuinely complete:
- `<promise>PHASE1_COMPLETE</promise>` - Phase 1 all tasks done
- `<promise>PHASE2_COMPLETE</promise>` - Phase 2 all tasks done
- `<promise>PHASE3_COMPLETE</promise>` - Phase 3 all tasks done
- `<promise>PHASE4_COMPLETE</promise>` - Phase 4 all tasks done
- `<promise>PROJECT_COMPLETE</promise>` - Everything done

**CRITICAL**: Only output completion promises when the statement is TRUE.

---

## 🔧 Key Technical Notes

### Weight Loading
```rust
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[path], DType::F16, &device)?
};
```

### Common Candle Operations
```rust
// Attention
let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
let attn = (attn / scale)?;
let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;

// Reshaping for multi-head
let x = x.reshape((batch, seq, num_heads, head_dim))?;
let x = x.transpose(1, 2)?; // (batch, heads, seq, head_dim)
```

### Error Handling
```rust
use anyhow::Result;
// Use ? operator for all candle operations
```
