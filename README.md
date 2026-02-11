# IndexTTS2-Rust

Rust port of IndexTTS2 with CUDA inference, voice cloning, and emotion control.

## Status (February 11, 2026) DONT LISTEN TO CODEX, YES IT PRODUCES AN ACTUAL VOICE NOW BUT CLONING DOESN'T WORK, QUALITY NEEDS IMPROVING, SOMETHING IS OFF WITH EMOTIONS...

The pipeline is now producing intelligible speech on GPU. The major "rumbling water" failure mode was fixed.

### Key fixes completed

- GPT logits path parity fix:
  - Added separate LM-head norm loading from `final_norm.*` and applied it before mel projection.
- Multi-step GPT parity instrumentation:
  - Added step1/last-step logits and cache-length dump/compare support.
- DiT parity alignment:
  - Added RMSNorm AdaLN behavior, RoPE, UViT skip schedule fixes, and adaptive final norm behavior.
- BigVGAN critical fix:
  - Replaced approximate upsampling with true `ConvTranspose1d` in Rust.
  - Removed extra non-upstream Snake activation before upsample blocks.
- Post-vocoder stability guard:
  - Added de-rumble high-pass option and automatic fallback for high DC-bias outputs.

### Current parity/quality snapshot

- DiT parity is very close (step-level diffs in low `1e-4` range in latest DiT parity run).
- GPT step0 logits are close; step1+ drift remains and is the next parity target.
- End-to-end GPU inference now outputs audible speech files.

## What works now

- Voice cloning from a reference speaker clip.
- Emotion control pathways:
  1. Emotion reference audio (`--emotion-audio`)
  2. Emotion audio blending (`--emotion-audio` + `--emotion-alpha <0..1>`)
  3. Manual emotion vector (`--emotion-vector`)
  4. Emotion from text (`--use-emo-text` and optional `--emo-text`)

## Requirements

- Rust 1.75+
- CUDA-capable GPU for fast inference (tested on RTX 5090)
- Model checkpoints under `checkpoints/`

## Build

```powershell
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda
```

## CLI quick start

```powershell
# Voice cloning baseline
./target/release/indextts2.exe infer \
  --speaker checkpoints/speaker_16k.wav \
  --text "Hello from IndexTTS2 Rust." \
  --output debug/quick_voice_clone.wav \
  --top-k 0 --top-p 1.0 --temperature 0.8 \
  --flow-steps 25 --flow-cfg-rate 0.7 \
  --de-rumble --de-rumble-cutoff-hz 180

# Emotion audio
./target/release/indextts2.exe infer \
  --speaker checkpoints/speaker_16k.wav \
  --emotion-audio speaker.wav --emotion-alpha 0.35 \
  --text "This should sound calm and natural." \
  --output debug/quick_emotion_audio.wav \
  --top-k 0 --top-p 1.0 --temperature 0.8 \
  --flow-steps 25 --flow-cfg-rate 0.7 \
  --de-rumble --de-rumble-cutoff-hz 180

# Manual emotion vector
./target/release/indextts2.exe infer \
  --speaker checkpoints/speaker_16k.wav \
  --emotion-vector "0.60,0.00,0.00,0.00,0.00,0.00,0.10,0.20" --emotion-alpha 0.9 \
  --text "Emotion vector test." \
  --output debug/quick_emotion_vector.wav \
  --top-k 0 --top-p 1.0 --temperature 0.8 \
  --flow-steps 25 --flow-cfg-rate 0.7 \
  --de-rumble --de-rumble-cutoff-hz 180

# Emotion from text (Qwen)
./target/release/indextts2.exe infer \
  --speaker checkpoints/speaker_16k.wav \
  --use-emo-text --emo-text "I feel happy and excited today." \
  --text "Emotion text inference test." \
  --output debug/quick_emotion_text.wav \
  --top-k 0 --top-p 1.0 --temperature 0.8 \
  --flow-steps 25 --flow-cfg-rate 0.7 \
  --de-rumble --de-rumble-cutoff-hz 180
```

## Easy launcher and UI

- PowerShell launcher:
  - `launch_indextts2.ps1`
- Tkinter local UI:
  - `launch_ui.ps1`
  - `scripts/ui_launcher.py`

### Start UI

```powershell
./launch_ui.ps1
```

The UI builds/uses `target/release/indextts2.exe`, lets you pick mode, and streams logs in-app.

## Validation artifacts from this session

- Mode validation outputs:
  - `debug/validation_suite_20260211/voice_clone.wav`
  - `debug/validation_suite_20260211/emotion_audio.wav`
  - `debug/validation_suite_20260211/emotion_audio_blend.wav`
  - `debug/validation_suite_20260211/emotion_vector.wav`
  - `debug/validation_suite_20260211/emotion_text.wav`
- Quality sweep outputs:
  - `debug/quality_sweep_20260211/*.wav`
  - `debug/quality_sweep_20260211/metrics_report.json`

## Detailed docs

- Current status summary: `docs/STATUS_2026-02-11.md`
- Full handoff for continuation: `docs/HANDOFF_2026-02-11.md`

## Known remaining work

- GPT cached decode parity drift at step1+ is still open.
- Next target is step1 internal parity (pre/post LN, q/k/v, attention scores) to isolate first post-step0 divergence.

## License

Apache-2.0
