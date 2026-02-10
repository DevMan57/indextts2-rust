# IndexTTS2-Rust: Fix Audio Quality + Cleanup — Fresh Session Prompt

Paste everything below into a fresh Claude session (with the same project files attached: `builder.md`, `validator.md`, `plan_w_team.md`).

---

## CONTEXT

I have an IndexTTS2 text-to-speech system rewritten in Rust at `C:\AI\indextts2-rust`. The code **compiles and runs end-to-end** (131 tests pass, CLI works, WAV files are generated), but **the audio output is noise/rumbling water instead of speech**.

The root cause is known: **4 of 5 model components are running with random weights** because the pre-trained weight files use different tensor naming conventions than the Rust structs. Only BigVGAN vocoder weights load correctly. The other four — Wav2Vec-BERT, DiT, Conformer, and Perceiver — silently fall back to random initialization when the weight key names don't match.

The original Python reference implementation is at `C:\AI\index-tts` (the `indextts` subdirectory contains the source). The Rust project's existing Python checkpoint files (`.pth` and `.safetensors`) are at `C:\AI\indextts2-rust\checkpoints\`.

## TWO GOALS

### Goal 1: Fix the weight loading to produce real speech audio
### Goal 2: Clean up the project directory (it's full of junk files from months of debugging)

---

## GOAL 1: FIX WEIGHT ARCHITECTURE MISMATCH

### The Problem

The Rust model structs use Candle's `VarBuilder` to load safetensors, but the tensor key names in the checkpoint files don't match the Rust struct field paths. Example:

```
Wav2Vec-BERT 2.0 (file: checkpoints/wav2vec_bert.safetensors):
  Checkpoint key:  encoder.layers.0.self_attn.q_proj.weight
  Rust VarBuilder:  layers.0.attention.q_proj.weight  ← MISSES

DiT (file: checkpoints/s2mel.safetensors):
  Checkpoint key:  dit.blocks.0.attn.qkv.weight
  Rust VarBuilder:  blocks.0.attention.qkv.weight  ← MISSES

Conformer (file: checkpoints/gpt.safetensors):
  Checkpoint key:  conformer.layers.0.self_attn.q_proj.weight
  Rust VarBuilder:  encoder.layers.0.attention.q_proj.weight  ← MISSES

Perceiver (file: checkpoints/gpt.safetensors):
  Checkpoint key:  perceiver.layers.0.cross_attn.q_proj.weight
  Rust VarBuilder:  (unknown — needs mapping)  ← MISSES
```

BigVGAN (`checkpoints/bigvgan.safetensors`) loads correctly because its weight names already match.

### What Needs to Happen

For each of the 4 broken components:

1. **Dump the actual tensor names** from the checkpoint `.safetensors` files using a Python script or Candle's safetensors API.
2. **Dump the Rust VarBuilder expected names** by adding debug logging to each model's `load()` function.
3. **Create a mapping function** that translates checkpoint names → Rust expected names.
4. **Apply the mapping** during weight loading (using `VarBuilder::rename` or a custom prefix/rename wrapper).
5. **Verify** that the model now loads all weights (no tensor should remain at random init).

### Reference Files

The Python reference implementation shows the exact model architecture and weight key paths:

- Python GPT model (Conformer + Perceiver): `C:\AI\index-tts\indextts\gpt\model.py`
- Python DiT model: `C:\AI\index-tts\indextts\s2mel\model.py` and `C:\AI\index-tts\indextts\s2mel\dit.py`
- Python Wav2Vec-BERT usage: `C:\AI\index-tts\indextts\infer.py` (see how `Wav2VecFeatureExtractor` is loaded)
- Python BigVGAN: `C:\AI\index-tts\indextts\BigVGAN\` directory
- Python VQVAE/codec: `C:\AI\index-tts\indextts\vqvae\`

The Rust implementations that need fixing:

- Wav2Vec-BERT: `C:\AI\indextts2-rust\src\models\semantic\wav2vec_bert.rs`
- DiT: `C:\AI\indextts2-rust\src\models\s2mel\dit.rs`
- Conformer: `C:\AI\indextts2-rust\src\models\gpt\conformer.rs`
- Perceiver: `C:\AI\indextts2-rust\src\models\gpt\perceiver.rs`
- GPT (UnifiedVoice): `C:\AI\indextts2-rust\src\models\gpt\unified_voice.rs`
- BigVGAN (working reference): `C:\AI\indextts2-rust\src\models\vocoder\bigvgan.rs`
- Pipeline: `C:\AI\indextts2-rust\src\inference\pipeline.rs`

Checkpoint files:

- `C:\AI\indextts2-rust\checkpoints\wav2vec_bert.safetensors` — Wav2Vec-BERT 2.0 weights
- `C:\AI\indextts2-rust\checkpoints\gpt.safetensors` — GPT + Conformer + Perceiver weights
- `C:\AI\indextts2-rust\checkpoints\s2mel.safetensors` — DiT + S2Mel weights
- `C:\AI\indextts2-rust\checkpoints\bigvgan.safetensors` — BigVGAN vocoder (WORKING)
- `C:\AI\indextts2-rust\checkpoints\config.yaml` — Model configuration (dimensions, layer counts)

### Existing Debugging Docs (Read These)

- `C:\AI\indextts2-rust\CLAUDE.md` — Full project overview, architecture, status
- `C:\AI\indextts2-rust\CURRENT_STATUS.md` — Detailed component status
- `C:\AI\indextts2-rust\DEBUGGING.md` — All fixes already attempted (FinalLayer AdaLN, GroupNorm, prompt_x)
- `C:\AI\indextts2-rust\@fix_weight_architecture.md` — Detailed weight mapping plan (partially complete)
- `C:\AI\indextts2-rust\FIXES.md` — Log of all fixes applied

### Skills to Use

- Read skill at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md` — covers Candle patterns, weight loading, PyTorch→Candle operation mapping, and testing strategies.
- Use `Context7` MCP tool to fetch up-to-date Candle docs: `Context7:resolve-library-id "candle machine learning"` → `Context7:get-library-docs` with topic `"VarBuilder safetensors rename"`.

### Validation

After fixing weight loading, run:
```bash
cd C:\AI\indextts2-rust
cargo run --release --bin indextts2 -- --cpu infer --text "Hello world, this is a test of the emergency broadcast system." --speaker "speaker_16k.wav" --output "output_fixed.wav"
```

The output WAV should contain recognizable speech (not noise/rumbling). Compare mel spectrogram statistics: generated mel mean should be close to speaker mel mean (~-6.5, not -10).

---

## GOAL 2: CLEAN UP THE PROJECT DIRECTORY

The project accumulated months of debugging artifacts. Clean it up:

### Files to DELETE (junk/temp outputs)
- All `test_output*.wav`, `test_debug.wav`, `test_final.wav`, `output_*.wav` in project root (8+ files)
- `dit_test_output.txt`
- `nul` (Windows redirect artifact)
- `default.wav`

### Files to DELETE (stale planning docs — info is captured in CLAUDE.md)
- `@AGENT.md`
- `@fix_compilation_errors.md`
- `@fix_plan.md` (phases 1-8 are done)
- `@fix_plan_phase6.md`, `@fix_plan_phase7.md`, `@fix_plan_phase8.md`, `@fix_plan_phase9.md`
- `@fix_weight_architecture.md` (superseded by this plan)
- `AGENTS.md`
- `GEMINI.md`
- `FIXES.md` (merge key info into DEBUGGING.md)
- `FIX_TOKENIZER.md`
- `DEBUG_STRATEGY.md`
- `DUAL_WAVEFORM_PLAN.md` (wrong project)
- `plan-with-team.md` (wrong project)
- `claude-code-execution-guide.md` (wrong project)
- `PROMPT.md`
- `cleanup_github.bat`, `setup_github.bat`, `push_to_github.sh` (git scripts, not needed)
- `VOICES_AND_SKILLS.md` (wrong project)
- `.gemini-clipboard/` directory
- `.planning/` directory

### Files to KEEP
- `CLAUDE.md` (update with final status after fix)
- `CURRENT_STATUS.md` (update after fix)
- `DEBUGGING.md` (update with final resolution)
- `README.md`
- `Cargo.toml`, `Cargo.lock`, `.gitignore`
- `speaker.wav`, `speaker_16k.wav` (test reference audio)
- `src/` (all source code)
- `checkpoints/` (model weights)
- `tests/`, `benches/`, `examples/` (test infrastructure)
- `scripts/` (if contains useful utilities)
- `specs/` (keep for future plans)
- `.claude/` (Claude Code config)
- `logs/` (keep but clean old logs)
- `debug/` (keep — contains golden reference data)

### After Cleanup

Update `CLAUDE.md` to reflect:
- Remove references to deleted files
- Update status to reflect weight loading fix
- Remove stale Ralph Loop commands
- Simplify the "Next Steps" section

---

## EXECUTION INSTRUCTIONS

Generate a `specs/plan.md` using the **plan_w_team.md** methodology (attached). The plan should have:

1. **Phase 1: Cleanup** — Delete junk files, reorganize. (1 builder task + 1 validator task)
2. **Phase 2: Weight Diagnosis** — For each of the 4 broken components, dump checkpoint tensor names and Rust expected names side-by-side. (1 builder task — produces a diagnostic report)
3. **Phase 3: Weight Mapping Implementation** — Implement name mapping functions for Wav2Vec-BERT, DiT, Conformer, and Perceiver. (4 parallel builder tasks, one per component)
4. **Phase 4: Integration Validation** — Run full inference, compare audio output to Python reference, verify mel statistics. (1 builder task + 1 validator task)
5. **Phase 5: Documentation Update** — Update CLAUDE.md, CURRENT_STATUS.md, DEBUGGING.md. (1 builder task)

Use the **builder** and **validator** agent definitions from the attached project files.

**CRITICAL CONTEXT FOR BUILDERS:**
- The project uses **Candle** (HuggingFace's Rust ML framework), NOT PyTorch or ONNX.
- Model weights are in **safetensors** format loaded via `VarBuilder::from_mmaped_safetensors()`.
- The CLI command to test is: `cargo run --release --bin indextts2 -- --cpu infer --text "Hello world" --speaker "speaker_16k.wav" --output "output.wav"`
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST in every task — it has the full architecture and module map.
- Read the skill at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md` for Candle patterns.
- The Python reference at `C:\AI\index-tts\indextts\` is the ground truth for weight names and model architecture.

**DO NOT:**
- Rewrite any model architecture (the Rust implementations are correct, only the weight *loading* is broken)
- Touch BigVGAN (it already works)
- Change the inference pipeline logic (the three AdaLN/GroupNorm/prompt_x fixes already applied are correct)
- Add new dependencies
