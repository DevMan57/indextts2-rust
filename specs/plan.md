# Plan: Fix Weight Architecture Mismatch + Project Cleanup

## Progress (Updated 2026-02-07 Session 2)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 1 | Clean Up Project Directory | DONE | ~25 stale files removed |
| 2 | Validate Cleanup | DONE | Only expected files remain |
| 3 | Diagnose Weight Loading | DONE | Diagnostic binary created and run |
| 4 | Fix Wav2Vec-BERT | SKIPPED | Already loading 628/628 tensors correctly |
| 5 | Fix DiT | DONE | Was already loading correctly (diagnostic tool had wrong expected keys, now fixed to 252/252) |
| 6 | Fix Conformer | DONE | Changed `initialize_random()` → `load_from_gpt_tensors()` in unified_voice.rs:488; added LayerNorm to ConvModule |
| 7 | Fix Perceiver | SKIPPED | Already loading 18/18 tensors correctly |
| 8 | Integration Test | DONE | Build passes, 132 tests pass, WAV generated but still noise |
| 9 | Fix vq2emb Architecture Bug | IN PROGRESS | Builder agents deployed |
| 10 | Validate Audio Quality | PENDING | Blocked by Step 9 |

### Critical Architecture Bug Found (Session 2)
After Steps 1-8, audio was STILL noise ("rumbling water"). Investigation found `cond_mean=0.0002` — near-zero conditioning reaching DiT.

**Root cause**: `pipeline.rs:517-532` uses `gpt.embed_mel_codes()` + `gpt_layer` projection for mel code embeddings. Python reference uses `semantic_codec.quantizer.vq2emb()` (MaskGCT codebook) — a fundamentally different embedding:

| | Python (correct) | Rust (broken) |
|---|---|---|
| Mel code embedding | `semantic_codec.quantizer.vq2emb(codes)` | `gpt.embed_mel_codes()` → `gpt_layer()` |
| Codebook | MaskGCT semantic codec [8192, 8] + proj_out [1024, 8] | GPT mel_embedding [8194, 1280] |
| Output dim | 1024 (direct from codebook + proj) | 1024 (via wrong 1280→256→128→1024 path) |
| Result | Correct acoustic embeddings | Wrong representation → near-zero S_infer → noise |

**Fix**: Downloaded MaskGCT checkpoint from `amphion/MaskGCT`. Two builder agents deployed:
1. Update `SemanticCodec` to load quantizer codebook weights (agent a3faddd)
2. Fix `pipeline.rs` to use `semantic_codec.vq2emb()` (agent a26b84b)

### Previous Key Findings
- **Conformer**: was the only component with broken weight loading
- **DiT**: was misdiagnosed (diagnostic tool had wrong keys)
- **Wav2Vec-BERT and Perceiver**: already loading correctly
- **gpt_layer has no activations**: confirmed correct per checkpoint structure (consecutive indices 0,1,2)

## Task Description

The IndexTTS2 Rust TTS system compiles and runs end-to-end (131 tests pass, CLI works, WAV files are generated), but audio output is noise/rumbling water instead of speech. The root cause is confirmed: 4 of 5 model components (Wav2Vec-BERT, DiT, Conformer, Perceiver) are running with random weights because the pre-trained checkpoint files use different tensor naming conventions than the Rust structs. Only BigVGAN vocoder weights load correctly. Additionally, the project directory is cluttered with months of debugging artifacts that need removal.

## Objective

1. Fix weight loading for all 4 broken model components so that pre-trained weights are correctly mapped from safetensors checkpoint files to the Rust model structs, producing recognizable speech audio instead of noise.
2. Clean up the project directory by removing ~25 stale files/directories and updating surviving documentation.

## Problem Statement

The Rust model structs load safetensors via `HashMap<String, Tensor>` with manual key lookups. The tensor key names in the checkpoint files do not match the key names the Rust code expects. When a key is missing, each component silently falls back to random initialization. This means the forward pass runs without errors but produces garbage output because the learned weights are never applied. Example mismatches:

- **Wav2Vec-BERT** (`wav2vec_bert.safetensors`): Checkpoint uses `encoder.layers.0.self_attn.linear_q.weight` — Rust looks up `encoder.layers.0.self_attn.linear_q.weight` (this component may actually be loading correctly based on code inspection; diagnosis step will confirm).
- **DiT** (`s2mel.safetensors`): Checkpoint uses `cfm.estimator.transformer.layers.0.attention.wqkv.weight` — Rust already uses `cfm.estimator` prefix (may be loading; diagnosis will confirm).
- **Conformer** (`gpt.safetensors`): Checkpoint uses `conditioning_encoder.encoders.{i}.*` — Rust expects same prefix (needs verification).
- **Perceiver** (`gpt.safetensors`): Checkpoint uses `perceiver_encoder.*` — Rust expects same prefix (needs verification).

The real issue may be more subtle than simple name mismatches — it could be missing tensors, wrong shapes, incorrect prefix stripping, or silent fallback to random init when any single tensor in a layer fails to load. The diagnosis phase will identify exactly which tensors are missing vs loaded.

## Solution Approach

1. **Diagnose first**: Write a diagnostic binary that loads each safetensors file, enumerates every tensor key, and compares against the exact keys the Rust code attempts to look up. Produce a per-component report showing loaded vs missing vs random-init tensors.
2. **Fix per-component**: For each broken component, implement the minimal mapping (rename, prefix adjustment, or key translation) needed to load all tensors from the checkpoint.
3. **Validate end-to-end**: Run full inference and verify the output WAV contains recognizable speech with mel statistics close to the speaker reference.
4. **Clean up**: Remove junk files, update documentation.

## Relevant Files

Use these files to complete the task:

**Rust model implementations (weight loading code to fix):**
- `C:\AI\indextts2-rust\src\models\semantic\wav2vec_bert.rs` — Wav2Vec-BERT encoder, loads from `wav2vec_bert.safetensors`
- `C:\AI\indextts2-rust\src\models\s2mel\dit.rs` — DiT transformer, loads from `s2mel.safetensors` via `load_s2mel_safetensors()`
- `C:\AI\indextts2-rust\src\models\gpt\conformer.rs` — Conformer encoder, loads from `gpt.safetensors`
- `C:\AI\indextts2-rust\src\models\gpt\perceiver.rs` — Perceiver resampler, loads from `gpt.safetensors`
- `C:\AI\indextts2-rust\src\models\gpt\unified_voice.rs` — GPT/UnifiedVoice, orchestrates Conformer + Perceiver loading
- `C:\AI\indextts2-rust\src\models\gpt\weights.rs` — GPT weight loading helpers
- `C:\AI\indextts2-rust\src\models\s2mel\weights.rs` — S2Mel weight loading helpers (contains `load_s2mel_safetensors`)
- `C:\AI\indextts2-rust\src\models\vocoder\bigvgan.rs` — BigVGAN vocoder (WORKING — use as reference for correct weight loading pattern)
- `C:\AI\indextts2-rust\src\inference\pipeline.rs` — Inference pipeline (orchestrates all models)

**Checkpoint files (safetensors format):**
- `C:\AI\indextts2-rust\checkpoints\wav2vec_bert.safetensors` — Wav2Vec-BERT 2.0 weights
- `C:\AI\indextts2-rust\checkpoints\gpt.safetensors` — GPT + Conformer + Perceiver weights
- `C:\AI\indextts2-rust\checkpoints\s2mel.safetensors` — DiT + S2Mel weights
- `C:\AI\indextts2-rust\checkpoints\bigvgan.safetensors` — BigVGAN vocoder (WORKING reference)
- `C:\AI\indextts2-rust\checkpoints\config.yaml` — Model configuration

**Python reference implementation (ground truth for weight names and architecture):**
- `C:\AI\index-tts\indextts\gpt\model.py` — Python GPT model (Conformer + Perceiver architecture)
- `C:\AI\index-tts\indextts\s2mel\model.py` — Python S2Mel/DiT model
- `C:\AI\index-tts\indextts\s2mel\dit.py` — Python DiT implementation
- `C:\AI\index-tts\indextts\infer.py` — Python inference pipeline (shows how Wav2Vec-BERT is loaded/used)
- `C:\AI\index-tts\indextts\BigVGAN\` — Python BigVGAN (working reference)
- `C:\AI\index-tts\indextts\vqvae\` — Python VQVAE/codec

**Documentation:**
- `C:\AI\indextts2-rust\CLAUDE.md` — Full project overview, architecture, status
- `C:\AI\indextts2-rust\CURRENT_STATUS.md` — Detailed component status
- `C:\AI\indextts2-rust\DEBUGGING.md` — All fixes already attempted
- `C:\AI\indextts2-rust\@fix_weight_architecture.md` — Partially complete weight mapping plan

**Skills and tools:**
- Skill file: `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md` — Candle patterns, weight loading, PyTorch→Candle mapping
- MCP tool: `Context7` — Fetch up-to-date Candle documentation

### New Files
- `C:\AI\indextts2-rust\src\bin\diagnose_weights.rs` — Diagnostic binary to compare checkpoint keys vs Rust expected keys

## Implementation Phases

### Phase 1: Cleanup
Remove ~25 junk/stale files and directories accumulated during months of debugging. Update no code — purely filesystem operations.

### Phase 2: Weight Diagnosis
Create a diagnostic Rust binary that loads each safetensors checkpoint, enumerates every tensor key with shape, and compares against what the Rust model code expects. Produce a structured report showing: (a) tensors successfully loaded, (b) tensors expected but missing, (c) tensors present in checkpoint but not consumed. This report drives all subsequent mapping work.

### Phase 3: Weight Mapping Implementation
For each of the 4 broken components, implement the minimal code changes to correctly map checkpoint tensor names to what the Rust structs expect. This may involve renaming keys before lookup, adjusting prefix handling, or adding tensor reshape/transpose operations. Each component is independent and can be fixed in parallel.

### Phase 4: Integration Validation
Run full end-to-end inference with fixed weights. Verify output WAV contains recognizable speech. Compare mel spectrogram statistics (generated mel mean should be close to speaker mel mean ~-6.5, not -10).

### Phase 5: Documentation Update
Update CLAUDE.md, CURRENT_STATUS.md, and DEBUGGING.md to reflect the weight loading fix and cleanup. Remove references to deleted files. Simplify "Next Steps" section.

## Team Orchestration

- You operate as the team lead and orchestrate the team to execute the plan.
- You're responsible for deploying the right team members with the right context to execute the plan.
- IMPORTANT: You NEVER operate directly on the codebase. You use `Task` and `Task*` tools to deploy team members to to the building, validating, testing, deploying, and other tasks.
  - This is critical. You're job is to act as a high level director of the team, not a builder.
  - You're role is to validate all work is going well and make sure the team is on track to complete the plan.
  - You'll orchestrate this by using the Task* Tools to manage coordination between the team members.
  - Communication is paramount. You'll use the Task* Tools to communicate with the team members and ensure they're on track to complete the plan.
- Take note of the session id of each team member. This is how you'll reference them.

### Team Members

- Builder
  - Name: builder-cleanup
  - Role: Delete junk files and stale documentation from the project root
  - Agent Type: general-purpose
  - Resume: false

- Builder
  - Name: builder-diagnosis
  - Role: Create and run a diagnostic binary that compares checkpoint tensor keys against Rust expected keys for all 4 broken components
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-wav2vec
  - Role: Fix weight loading for the Wav2Vec-BERT semantic encoder component
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-dit
  - Role: Fix weight loading for the DiT (Diffusion Transformer) component
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-conformer
  - Role: Fix weight loading for the Conformer encoder component
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-perceiver
  - Role: Fix weight loading for the Perceiver resampler component
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-integration
  - Role: Run full inference with fixed weights, verify audio output, compare mel statistics
  - Agent Type: general-purpose
  - Resume: true

- Builder
  - Name: builder-docs
  - Role: Update CLAUDE.md, CURRENT_STATUS.md, and DEBUGGING.md after fix and cleanup
  - Agent Type: general-purpose
  - Resume: false

- Builder
  - Name: validator-cleanup
  - Role: Verify all junk files were deleted and no required files were removed
  - Agent Type: general-purpose
  - Resume: false

- Builder
  - Name: validator-weights
  - Role: Verify all 4 components load weights correctly, run full inference, verify audio quality
  - Agent Type: general-purpose
  - Resume: false

## Step by Step Tasks

- IMPORTANT: Execute every step in order, top to bottom. Each task maps directly to a `TaskCreate` call.
- Before you start, run `TaskCreate` to create the initial task list that all team members can see and execute.

### 1. Clean Up Project Directory
- **Task ID**: cleanup-files
- **Depends On**: none
- **Assigned To**: builder-cleanup
- **Agent Type**: general-purpose
- **Parallel**: true (can run alongside diagnosis)
- Read `C:\AI\indextts2-rust\CLAUDE.md` to understand the project structure.
- Delete the following files from `C:\AI\indextts2-rust\`:
  - WAV artifacts: `test_output.wav`, `test_output2.wav`, `test_output3.wav`, `test_output4.wav`, `test_output5.wav`, `test_output6.wav`, `test_output7.wav`, `test_output8.wav`, `test_debug.wav`, `test_final.wav`, `output.wav`, `output_cfg0.wav`, `output_fox.wav`, `output_test.wav`, `default.wav`
  - Debug artifacts: `dit_test_output.txt`, `nul`
  - Stale planning docs: `@AGENT.md`, `@fix_compilation_errors.md`, `@fix_plan.md`, `@fix_plan_phase6.md`, `@fix_plan_phase7.md`, `@fix_plan_phase8.md`, `@fix_plan_phase9.md`, `@fix_weight_architecture.md`, `AGENTS.md`, `GEMINI.md`, `FIXES.md`, `FIX_TOKENIZER.md`, `DEBUG_STRATEGY.md`, `PROMPT.md`
  - Wrong-project files: `DUAL_WAVEFORM_PLAN.md` (if exists), `plan-with-team.md` (if exists), `claude-code-execution-guide.md` (if exists), `VOICES_AND_SKILLS.md` (if exists)
  - Git scripts: `cleanup_github.bat`, `setup_github.bat`, `push_to_github.sh`
  - Directories: `.gemini-clipboard/`, `.planning/`
- Do NOT delete: `CLAUDE.md`, `CURRENT_STATUS.md`, `DEBUGGING.md`, `README.md`, `Cargo.toml`, `Cargo.lock`, `.gitignore`, `speaker.wav`, `speaker_16k.wav`, `src/`, `checkpoints/`, `tests/`, `benches/`, `examples/`, `scripts/`, `specs/`, `.claude/`, `logs/`, `debug/`, `target/`
- After deletion, run `dir C:\AI\indextts2-rust\` to confirm only expected files remain.

### 2. Validate Cleanup
- **Task ID**: validate-cleanup
- **Depends On**: cleanup-files
- **Assigned To**: validator-cleanup
- **Agent Type**: general-purpose
- **Parallel**: false
- List the contents of `C:\AI\indextts2-rust\` and verify:
  - None of the files from the deletion list exist
  - All files from the "keep" list still exist
  - The `.gemini-clipboard/` and `.planning/` directories no longer exist
  - `cargo build --release --bin indextts2` still compiles successfully (deleting docs should not break compilation)

### 3. Diagnose Weight Loading for All 4 Components
- **Task ID**: diagnose-weights
- **Depends On**: none
- **Assigned To**: builder-diagnosis
- **Agent Type**: general-purpose
- **Parallel**: true (can run alongside cleanup)
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST to understand the full architecture and module map.
- Read the skill documentation at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md` to understand Candle patterns and weight loading.
- Use the `Context7` MCP tool: run `Context7:resolve-library-id "candle machine learning"` then `Context7:get-library-docs` with topic `"safetensors VarBuilder load"` to fetch up-to-date Candle documentation on safetensors loading.
- Create a new Rust binary at `C:\AI\indextts2-rust\src\bin\diagnose_weights.rs` that does the following for each checkpoint file:
  1. Load the safetensors file using `candle_core::safetensors::load()` to get a `HashMap<String, Tensor>`.
  2. Print every tensor key and its shape, sorted alphabetically.
  3. For each model component, list which keys the Rust code attempts to look up (trace through the `from_tensors` / `load_weights` functions in the source files).
  4. Categorize each expected key as: FOUND (present in checkpoint), MISSING (not in checkpoint → falls back to random), or SHAPE_MISMATCH.
  5. Print a summary per component: total expected tensors, found, missing, shape mismatches.
- The checkpoint files to diagnose:
  - `C:\AI\indextts2-rust\checkpoints\wav2vec_bert.safetensors` — expected keys come from `src/models/semantic/wav2vec_bert.rs` (the `EncoderLayer::from_tensors`, `FeatureProjection::from_tensors`, `SelfAttention::from_tensors`, `FeedForward::from_tensors`, `ConvModule::from_tensors` functions)
  - `C:\AI\indextts2-rust\checkpoints\gpt.safetensors` — expected keys come from `src/models/gpt/conformer.rs` and `src/models/gpt/perceiver.rs` (trace through `load_weights` methods)
  - `C:\AI\indextts2-rust\checkpoints\s2mel.safetensors` — expected keys come from `src/models/s2mel/dit.rs` (the `DiffusionTransformer::load_weights` method, using prefix `cfm.estimator`)
  - `C:\AI\indextts2-rust\checkpoints\bigvgan.safetensors` — include as WORKING REFERENCE to show what correct loading looks like
- Also cross-reference with the Python source files at `C:\AI\index-tts\indextts\` to identify any tensors in the checkpoint that the Python code uses but the Rust code does not attempt to load.
- Run the diagnostic: `cd C:\AI\indextts2-rust && cargo run --release --bin diagnose_weights`
- Save the full output to `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt`
- CRITICAL: Do NOT modify any model source files in this task. This is diagnosis only.

### 4. Fix Wav2Vec-BERT Weight Loading
- **Task ID**: fix-wav2vec
- **Depends On**: diagnose-weights
- **Assigned To**: builder-wav2vec
- **Agent Type**: general-purpose
- **Parallel**: true (can run in parallel with fix-dit, fix-conformer, fix-perceiver)
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST.
- Read the skill documentation at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md`.
- Read the diagnosis report at `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt` to understand exactly which tensors are missing for Wav2Vec-BERT.
- Read the Python reference at `C:\AI\index-tts\indextts\infer.py` to understand how Wav2Vec-BERT is loaded (look for `Wav2VecFeatureExtractor` or `Wav2VecBert`).
- Read the current Rust implementation at `C:\AI\indextts2-rust\src\models\semantic\wav2vec_bert.rs`.
- Read BigVGAN at `C:\AI\indextts2-rust\src\models\vocoder\bigvgan.rs` as a reference for correct weight loading patterns.
- Based on the diagnosis report, implement the minimal changes needed to correctly load all Wav2Vec-BERT tensors from the checkpoint. This may involve:
  - Adjusting key name patterns in `from_tensors` methods
  - Adding a key mapping/rename step before tensor lookup
  - Removing silent fallback to random init (make missing tensors an error, or at least a loud warning)
- After fixing, verify by running: `cd C:\AI\indextts2-rust && cargo test --lib models::semantic`
- Do NOT change the model architecture (layer dimensions, attention patterns, etc.). Only fix how weights are loaded.
- Do NOT touch BigVGAN — it already works.

### 5. Fix DiT Weight Loading
- **Task ID**: fix-dit
- **Depends On**: diagnose-weights
- **Assigned To**: builder-dit
- **Agent Type**: general-purpose
- **Parallel**: true
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST.
- Read the skill documentation at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md`.
- Read the diagnosis report at `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt` to understand exactly which tensors are missing for DiT.
- Read the Python reference at `C:\AI\index-tts\indextts\s2mel\dit.py` and `C:\AI\index-tts\indextts\s2mel\model.py` for the ground-truth DiT architecture and weight names.
- Read the current Rust implementation at `C:\AI\indextts2-rust\src\models\s2mel\dit.rs`.
- Read the weight loading helper at `C:\AI\indextts2-rust\src\models\s2mel\weights.rs` (contains `load_s2mel_safetensors`).
- Based on the diagnosis report, implement the minimal changes to correctly load all DiT tensors. Pay special attention to:
  - The `cfm.estimator` prefix handling
  - Transformer block loading: `attention_norm`, `attention.wqkv`, `attention.wo`, `ffn_norm`, `feed_forward.w1/w2/w3`
  - WaveNet layers: weight-normalized convolutions (`weight_v`, `weight_g`)
  - `x_embedder`, `cond_embedder`, `cond_projection`, `cond_x_merge_linear`, `skip_linear`
  - `final_layer.adaLN_modulation`, `final_layer.linear` (weight-normalized)
  - `conv2` output projection
- After fixing, verify: `cd C:\AI\indextts2-rust && cargo test --lib models::s2mel`
- Do NOT change model architecture. Only fix weight loading.

### 6. Fix Conformer Weight Loading
- **Task ID**: fix-conformer
- **Depends On**: diagnose-weights
- **Assigned To**: builder-conformer
- **Agent Type**: general-purpose
- **Parallel**: true
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST.
- Read the skill documentation at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md`.
- Read the diagnosis report at `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt` for Conformer-specific missing tensors.
- Read the Python reference at `C:\AI\index-tts\indextts\gpt\model.py` — search for the Conformer / conditioning encoder class to see exact weight names.
- Read the current Rust implementation at `C:\AI\indextts2-rust\src\models\gpt\conformer.rs`.
- Read the GPT weight loading at `C:\AI\indextts2-rust\src\models\gpt\weights.rs` and `C:\AI\indextts2-rust\src\models\gpt\unified_voice.rs` to understand how gpt.safetensors is loaded and passed to the Conformer.
- Based on the diagnosis report, implement minimal changes to load all Conformer tensors correctly. Key areas:
  - `conditioning_encoder.encoders.{i}.*` prefix and sub-component names
  - Self-attention: `linear_q`, `linear_k`, `linear_v`, `linear_out` (or different naming)
  - Feed-forward: `w_1`, `w_2` (or different naming)
  - Conv module: pointwise convolutions, depthwise convolution
  - Layer norms: `norm_mha`, `norm_ff`, `norm_conv`, `norm_final`
- After fixing, verify: `cd C:\AI\indextts2-rust && cargo test --lib models::gpt`
- Do NOT change model architecture.

### 7. Fix Perceiver Weight Loading
- **Task ID**: fix-perceiver
- **Depends On**: diagnose-weights
- **Assigned To**: builder-perceiver
- **Agent Type**: general-purpose
- **Parallel**: true
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST.
- Read the skill documentation at `C:\Users\Henri Smith\.claude-membership\skills\rust-tts\SKILL.md`.
- Read the diagnosis report at `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt` for Perceiver-specific missing tensors.
- Read the Python reference at `C:\AI\index-tts\indextts\gpt\model.py` — search for the Perceiver / `perceiver_encoder` class.
- Read the current Rust implementation at `C:\AI\indextts2-rust\src\models\gpt\perceiver.rs`.
- Based on the diagnosis report, implement minimal changes. Key areas:
  - `perceiver_encoder.latents` — learned latent queries [32, 1280]
  - Cross-attention layers: `perceiver_encoder.layers.{i}.0.to_q`, `to_kv`, `to_out`
  - FFN layers: `perceiver_encoder.layers.{i}.1.0`, `1.2`
  - Norm: `perceiver_encoder.norm.gamma`
  - Context projection: `perceiver_encoder.proj_context`
- After fixing, verify: `cd C:\AI\indextts2-rust && cargo test --lib models::gpt`
- Do NOT change model architecture.

### 8. Integration Test — Full Inference
- **Task ID**: integration-test
- **Depends On**: fix-wav2vec, fix-dit, fix-conformer, fix-perceiver
- **Assigned To**: builder-integration
- **Agent Type**: general-purpose
- **Parallel**: false
- Read `C:\AI\indextts2-rust\CLAUDE.md` FIRST.
- Ensure all 4 component fixes compile together: `cd C:\AI\indextts2-rust && cargo build --release --bin indextts2`
- Run full test suite: `cd C:\AI\indextts2-rust && cargo test`
- Run full inference: `cd C:\AI\indextts2-rust && cargo run --release --bin indextts2 -- --cpu infer --text "Hello world, this is a test of the emergency broadcast system." --speaker "speaker_16k.wav" --output "output_fixed.wav"`
- Examine the output:
  - Check that `output_fixed.wav` was created and is non-empty
  - Check stderr/stdout for any "Missing tensor" or "using random initialization" warnings — there should be NONE (or at most for non-critical optional tensors)
  - Check mel spectrogram statistics in the log output: generated mel mean should be close to speaker mel mean (~-6.5), NOT -10
- If the audio still sounds like noise, examine the debug output to identify which component is still producing bad output, and report findings in the task update.
- If the audio sounds like speech (even imperfect), mark the task as successful.

### 9. Validate Weight Loading and Audio Quality
- **Task ID**: validate-weights
- **Depends On**: integration-test
- **Assigned To**: validator-weights
- **Agent Type**: general-purpose
- **Parallel**: false
- Read `C:\AI\indextts2-rust\CLAUDE.md` and `C:\AI\indextts2-rust\specs\weight_diagnosis_report.txt`.
- Re-run the diagnostic binary to confirm all tensors now load: `cd C:\AI\indextts2-rust && cargo run --release --bin diagnose_weights`
- Verify that for each of the 4 components, the number of MISSING tensors is 0 (or contains only truly optional tensors with documented justification).
- Re-run inference: `cd C:\AI\indextts2-rust && cargo run --release --bin indextts2 -- --cpu infer --text "Testing one two three." --speaker "speaker_16k.wav" --output "output_validation.wav"`
- Run full test suite: `cd C:\AI\indextts2-rust && cargo test`
- Report pass/fail with specifics on:
  - Number of tensors loaded vs missing per component
  - Mel spectrogram mean of generated vs speaker reference
  - Whether any "random initialization" warnings appear in stderr
  - Whether all 131+ tests still pass

### 10. Update Documentation
- **Task ID**: update-docs
- **Depends On**: validate-weights, validate-cleanup
- **Assigned To**: builder-docs
- **Agent Type**: general-purpose
- **Parallel**: false
- Read current `C:\AI\indextts2-rust\CLAUDE.md`, `C:\AI\indextts2-rust\CURRENT_STATUS.md`, and `C:\AI\indextts2-rust\DEBUGGING.md`.
- Update `CLAUDE.md`:
  - Change status banner from "PIPELINE RUNS BUT AUDIO IS NOISE" to reflect current state (either "AUDIO WORKING" or "WEIGHT LOADING FIXED — AUDIO QUALITY UNDER EVALUATION")
  - Remove the "Model Weight Loading Status" section's misleading "✅ Loaded" markers if any were previously incorrect
  - Remove references to all deleted files (the `@fix_plan*.md`, `AGENTS.md`, `FIXES.md`, etc.)
  - Remove stale Ralph Loop commands section (or update with current commands)
  - Update the "Key Files" table to remove deleted files
  - Simplify "Next Steps" to only list remaining work (CUDA support, performance optimization, Python bindings)
  - Update the "Progress Tracker" to reflect weight loading completion
- Update `CURRENT_STATUS.md` with the resolution of the weight loading issue
- Update `DEBUGGING.md`:
  - Add a section documenting the weight name mismatch root cause and fix
  - Merge any key information from the now-deleted `FIXES.md` into this file
  - Document which tensors were missing per component and how they were mapped
- Merge key fix information from `FIXES.md` (which was deleted in cleanup) into `DEBUGGING.md` before the deletion takes effect. If `FIXES.md` was already deleted, use your knowledge of the prior fix history from `CLAUDE.md` and `DEBUGGING.md`.

## Acceptance Criteria

1. **Weight loading**: All 4 broken components (Wav2Vec-BERT, DiT, Conformer, Perceiver) load 100% of their tensors from the checkpoint files with zero fallback to random initialization (excepting truly optional/unused tensors with documented justification).
2. **Audio quality**: Running `cargo run --release --bin indextts2 -- --cpu infer --text "Hello world" --speaker "speaker_16k.wav" --output "output.wav"` produces a WAV file containing recognizable speech (not noise/rumbling).
3. **Mel statistics**: Generated mel spectrogram mean is within 2 dB of the speaker reference mel mean (~-6.5).
4. **Tests pass**: `cargo test` passes all existing tests (131+) with no regressions.
5. **Compilation**: `cargo build --release` completes without errors or new warnings related to weight loading.
6. **Cleanup**: All files listed in the deletion list are removed. All files listed in the keep list remain. Project root contains only expected files.
7. **Documentation**: `CLAUDE.md`, `CURRENT_STATUS.md`, and `DEBUGGING.md` accurately reflect the current project state with no references to deleted files.

## Validation Commands

Execute these commands to validate the task is complete:

- `cd C:\AI\indextts2-rust && cargo build --release --bin indextts2` — Verify clean compilation
- `cd C:\AI\indextts2-rust && cargo test` — Verify all tests pass (expect 131+)
- `cd C:\AI\indextts2-rust && cargo run --release --bin diagnose_weights` — Verify all tensors load (0 MISSING per component)
- `cd C:\AI\indextts2-rust && cargo run --release --bin indextts2 -- --cpu infer --text "Hello world, this is a test of the emergency broadcast system." --speaker "speaker_16k.wav" --output "output_fixed.wav"` — Verify inference runs and produces speech audio
- `dir C:\AI\indextts2-rust\*.md` — Verify only CLAUDE.md, CURRENT_STATUS.md, DEBUGGING.md, README.md remain as markdown files in root
- `dir C:\AI\indextts2-rust\*.wav` — Verify only speaker.wav, speaker_16k.wav, and output_fixed.wav remain

## Notes

- **DO NOT rewrite model architecture**: The Rust implementations are correct. Only the weight *loading* (key name mapping) is broken.
- **DO NOT touch BigVGAN**: It already works correctly. Use it as a reference for how correct loading should look.
- **DO NOT change inference pipeline logic**: The three fixes already applied (FinalLayer AdaLN, GroupNorm, prompt_x) are correct.
- **DO NOT add new dependencies**: Use only existing crates (`candle-core`, `candle-nn`, `safetensors`, etc.).
- **Candle framework**: This project uses HuggingFace Candle (Rust ML framework), NOT PyTorch or ONNX. Model weights are in safetensors format loaded via `candle_core::safetensors::load()` into `HashMap<String, Tensor>`.
- **Silent fallback is the enemy**: The current code silently falls back to random init when a tensor key is missing. After fixing, consider making missing expected tensors produce loud warnings or errors instead of silent random fallback.
- If the diagnosis reveals that some components ARE loading correctly (the CLAUDE.md claims "all loaded" but audio is still noise), the issue may be in tensor shape/dtype handling, or in how the loaded tensors are applied during the forward pass. The diagnosis step will clarify this.
