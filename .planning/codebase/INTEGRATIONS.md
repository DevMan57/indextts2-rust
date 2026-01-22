# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**HuggingFace Hub:**
- Purpose: Model weight downloads
- SDK/Client: `hf-hub` crate (0.3)
- Auth: HuggingFace token (optional, for gated models)
- Status: Stubbed (not fully implemented)
- CLI fallback: `hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints`

**No Other External APIs:**
- This is a fully offline inference system
- All processing happens locally after model download

## Data Storage

**Databases:**
- None (file-based only)

**File Storage:**
- Local filesystem for model checkpoints
- Local filesystem for audio I/O

**Caching:**
- None implemented
- Potential: HuggingFace cache at `checkpoints/.cache/`

## Authentication & Identity

**Auth Provider:**
- None required for inference
- HuggingFace token only needed for model downloads (optional)

## Monitoring & Observability

**Error Tracking:**
- None (local CLI application)

**Logs:**
- `tracing` crate with `tracing-subscriber`
- Configurable via `--verbose` flag
- Levels: DEBUG (verbose) or INFO (default)
- Format: Compact terminal output

## CI/CD & Deployment

**Hosting:**
- Self-hosted binary (no cloud deployment)

**CI Pipeline:**
- Not configured (no `.github/workflows/` found)

**Containerization:**
- Not configured (no Dockerfile found)

## Environment Configuration

**Required env vars:**
- None

**Optional env vars:**
- `RUST_LOG` - Override tracing log level
- `HF_TOKEN` - HuggingFace authentication (for gated model downloads)

**Secrets location:**
- Not applicable (no secrets required)

## Webhooks & Callbacks

**Incoming:**
- None (CLI-only, no server mode implemented yet)

**Outgoing:**
- None

## Model Weight Dependencies

**Pre-trained Models Required:**

| Model | Source | Format | Size | Purpose |
|-------|--------|--------|------|---------|
| Wav2Vec-BERT 2.0 | `facebook/w2v-bert-2.0` | SafeTensors | ~600MB | Semantic audio encoding |
| GPT | IndexTTS-2 checkpoint | SafeTensors | ~800MB | Mel code generation |
| S2MEL/DiT | IndexTTS-2 checkpoint | SafeTensors | ~200MB | Flow matching synthesis |
| BigVGAN v2 | `nvidia/bigvgan_v2_22khz_80band_256x` | SafeTensors | ~150MB | Vocoder (mel to audio) |
| Emotion Matrix | IndexTTS-2 checkpoint | SafeTensors | ~10MB | Emotion control |
| Speaker Matrix | IndexTTS-2 checkpoint | SafeTensors | ~10MB | Speaker style |
| Qwen (emotion) | IndexTTS-2 checkpoint | SafeTensors | ~600MB | Emotion embeddings |

**Checkpoint Directory Structure:**
```
checkpoints/
├── config.yaml                    # Model configuration
├── tokenizer_english.json         # BPE tokenizer
├── gpt.safetensors               # GPT model weights
├── s2mel.safetensors             # S2MEL/DiT weights
├── bigvgan.safetensors           # BigVGAN vocoder weights
├── wav2vec_bert.safetensors      # Wav2Vec-BERT weights
├── wav2vec2bert_stats.safetensors # Normalization statistics
├── emotion_matrix.safetensors    # Emotion embeddings
├── speaker_matrix.safetensors    # Speaker embeddings
├── w2v-bert-2.0/                 # Facebook Wav2Vec-BERT
│   ├── config.json
│   └── model.safetensors
├── bigvgan-v2/                   # NVIDIA BigVGAN config
│   └── config.json
└── qwen0.6bemo4-merge/           # Qwen emotion model
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## File Formats

**Audio Input:**
- WAV (via `hound` - preferred, faster)
- MP3 (via `symphonia`)
- FLAC (via `symphonia`)
- OGG/Vorbis (via `symphonia`)

**Audio Output:**
- WAV (22050 Hz, mono, f32/i16)

**Model Weights:**
- SafeTensors (`.safetensors`) - Primary format
- PyTorch (`.pth`) - Legacy support (checkpoint conversion needed)

**Configuration:**
- YAML (`config.yaml`) - Model architecture config
- JSON (`tokenizer.json`) - HuggingFace tokenizer format

**Tokenizer:**
- HuggingFace `tokenizer.json` format
- Supports BPE and Unigram models
- Located at: `checkpoints/tokenizer_english.json`

## Data Flow Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                      LOCAL FILE SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Audio (.wav/.mp3/.flac/.ogg)                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  INFERENCE PIPELINE                      │   │
│  │  (Loads model weights from checkpoints/*.safetensors)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Output Audio (.wav @ 22050 Hz)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Model Download Sources

**Primary Source:**
- HuggingFace Hub: `IndexTeam/IndexTTS-2`

**Pre-trained Model Sources:**
- Facebook Wav2Vec-BERT: `facebook/w2v-bert-2.0`
- NVIDIA BigVGAN: `nvidia/bigvgan_v2_22khz_80band_256x`

**Download Command (manual):**
```bash
# Full IndexTTS-2 checkpoint
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Wav2Vec-BERT 2.0 (if needed separately)
hf download facebook/w2v-bert-2.0 --local-dir=checkpoints/w2v-bert-2.0
```

---

*Integration audit: 2026-01-23*
