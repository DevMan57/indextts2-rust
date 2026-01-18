# ML Port Debugging - Ralph Workflow

## The Strategy: Layer-by-Layer Validation

```
Python (Reference)          Rust (Target)
       │                          │
       ▼                          ▼
   ┌───────┐                  ┌───────┐
   │ Layer │──── Compare ─────│ Layer │
   └───────┘                  └───────┘
       │                          │
       ▼                          ▼
   ┌───────┐                  ┌───────┐
   │ Layer │──── Compare ─────│ Layer │
   └───────┘                  └───────┘
       │                          │
      ...                        ...
```

---

## Phase 0: Generate Golden Reference Data

First, create a Python script that dumps intermediate tensors:

```python
# C:\AI\index-tts\debug_dump.py
import torch
import numpy as np
from pathlib import Path

DUMP_DIR = Path("C:/AI/indextts2-rust/debug/golden")
DUMP_DIR.mkdir(parents=True, exist_ok=True)

def save_tensor(name: str, tensor: torch.Tensor):
    """Save tensor as numpy for Rust comparison"""
    np_array = tensor.detach().cpu().float().numpy()
    np.save(DUMP_DIR / f"{name}.npy", np_array)
    print(f"Saved {name}: {tensor.shape}")

# Hook into model layers
class DebugIndexTTS:
    def __init__(self, model):
        self.model = model
    
    def forward_with_dumps(self, text, speaker_audio):
        # Tokenizer output
        tokens = self.model.tokenize(text)
        save_tensor("01_tokens", tokens)
        
        # Mel spectrogram
        mel = self.model.compute_mel(speaker_audio)
        save_tensor("02_mel_spectrogram", mel)
        
        # Speaker embedding
        spk_emb = self.model.speaker_encoder(speaker_audio)
        save_tensor("03_speaker_embedding", spk_emb)
        
        # Semantic features
        semantic = self.model.semantic_encoder(speaker_audio)
        save_tensor("04_semantic_features", semantic)
        
        # GPT intermediate
        # ... hook into each layer
        
        # Final mel codes
        mel_codes = self.model.gpt(...)
        save_tensor("10_mel_codes", mel_codes)
        
        # Vocoder output
        audio = self.model.vocoder(mel_codes)
        save_tensor("11_final_audio", audio)
        
        return audio
```

---

## Rust Validation Module

```rust
// src/debug/validator.rs
use candle_core::{Tensor, Device};
use std::path::Path;

pub struct Validator {
    golden_dir: PathBuf,
    tolerance: f32,
}

impl Validator {
    pub fn new(golden_dir: &str) -> Self {
        Self {
            golden_dir: PathBuf::from(golden_dir),
            tolerance: 1e-4,
        }
    }
    
    pub fn load_golden(&self, name: &str) -> Result<Tensor> {
        let path = self.golden_dir.join(format!("{}.npy", name));
        Tensor::read_npy(path)
    }
    
    pub fn compare(&self, name: &str, rust_tensor: &Tensor) -> Result<ValidationResult> {
        let golden = self.load_golden(name)?;
        
        // Shape check
        if golden.shape() != rust_tensor.shape() {
            return Ok(ValidationResult::ShapeMismatch {
                expected: golden.shape().to_vec(),
                got: rust_tensor.shape().to_vec(),
            });
        }
        
        // Value check
        let diff = (golden - rust_tensor)?.abs()?;
        let max_diff = diff.max(0)?.to_scalar::<f32>()?;
        let mean_diff = diff.mean(0)?.to_scalar::<f32>()?;
        
        if max_diff > self.tolerance {
            Ok(ValidationResult::ValueMismatch { max_diff, mean_diff })
        } else {
            Ok(ValidationResult::Pass)
        }
    }
    
    pub fn validate_and_report(&self, name: &str, tensor: &Tensor) {
        match self.compare(name, tensor) {
            Ok(ValidationResult::Pass) => {
                println!("✅ {}: PASS", name);
            }
            Ok(ValidationResult::ShapeMismatch { expected, got }) => {
                println!("❌ {}: SHAPE MISMATCH", name);
                println!("   Expected: {:?}", expected);
                println!("   Got:      {:?}", got);
            }
            Ok(ValidationResult::ValueMismatch { max_diff, mean_diff }) => {
                println!("⚠️  {}: VALUE MISMATCH", name);
                println!("   Max diff:  {:.6}", max_diff);
                println!("   Mean diff: {:.6}", mean_diff);
            }
            Err(e) => {
                println!("❌ {}: ERROR - {}", name, e);
            }
        }
    }
}
```

---

## Debug-Focused @fix_plan.md Structure

```markdown
## Phase 6: Validation & Debug [PENDING]

### Stage 1: Generate Reference Data
- [ ] **P6.1** Create Python debug_dump.py script
- [ ] **P6.2** Run Python model with test input, save all intermediates
- [ ] **P6.3** Create standard test inputs (text + audio file)

### Stage 2: Rust Validation Harness  
- [ ] **P6.4** Implement src/debug/validator.rs
- [ ] **P6.5** Add NPY file loading support
- [ ] **P6.6** Create comparison CLI command

### Stage 3: Layer-by-Layer Validation
- [ ] **P6.7** Validate tokenizer output
- [ ] **P6.8** Validate mel spectrogram computation
- [ ] **P6.9** Validate speaker encoder output
- [ ] **P6.10** Validate semantic encoder output
- [ ] **P6.11** Validate GPT layer 0 output
- [ ] **P6.12** Validate GPT layer N output (repeat for key layers)
- [ ] **P6.13** Validate mel code generation
- [ ] **P6.14** Validate vocoder output

### Stage 4: Fix Identified Issues
- [ ] **P6.15** Fix shape mismatches (dynamic task)
- [ ] **P6.16** Fix numerical precision issues
- [ ] **P6.17** Fix weight loading issues
- [ ] **P6.18** End-to-end audio comparison
```

---

## Ralph Debug Loop Commands

```bash
# Stage 1: Setup validation infrastructure
/ralph-loop "Implement P6.1-P6.6 from @fix_plan.md - create Python dump script and Rust validator. Use Context7 for numpy/npy loading in Rust." --max-iterations 20 --completion-promise "DEBUG_INFRA_COMPLETE"

# Stage 2: Run validation on each component
/ralph-loop "Run validation for P6.7-P6.14. For each failing validation: identify the bug, fix it, re-validate. Document each fix in FIXES.md" --max-iterations 50 --completion-promise "VALIDATION_COMPLETE"

# Stage 3: Integration debugging
/ralph-loop "P6.15-P6.18: Fix all remaining issues until end-to-end audio matches reference within tolerance." --max-iterations 40 --completion-promise "DEBUG_COMPLETE"
```

---

## Smart Debug Strategies

### 1. Binary Search for Divergence

```rust
// Find exactly where outputs diverge
fn find_divergence_point(model: &Model, validator: &Validator) {
    let checkpoints = [
        "embedding", "layer_0", "layer_4", "layer_8", 
        "layer_12", "layer_16", "layer_20", "layer_23", "output"
    ];
    
    for (i, checkpoint) in checkpoints.iter().enumerate() {
        if !validator.compare(checkpoint, &tensors[i]).is_pass() {
            println!("🎯 Divergence starts at: {}", checkpoint);
            // Now binary search between this and previous
            break;
        }
    }
}
```

### 2. Common Fix Patterns

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Shape `[B,S,H]` vs `[B,H,S]` | Transpose missing | Add `.transpose()` |
| Values off by ~1e-3 | LayerNorm epsilon | Match Python's `eps=1e-5` |
| Values off by ~0.5 | Normalization | Check mean/std computation |
| All zeros | ReLU/activation wrong | Check activation function |
| NaN/Inf | Overflow | Use F32 instead of F16 |
| Wrong vocab tokens | Tokenizer mismatch | Compare vocab files |
| Audio clicks | Overlap-add window | Check hop_length |

### 3. Differential Testing

```rust
// Run same input through both, compare
fn differential_test(text: &str, audio_path: &str) {
    // Run Python (via subprocess)
    let py_output = Command::new("python")
        .args(["run_inference.py", text, audio_path, "--dump"])
        .output()?;
    
    // Run Rust
    let rs_output = model.synthesize(text, audio_path)?;
    
    // Compare
    validator.compare_audio(py_output, rs_output);
}
```

---

## Quick Debug CLI

Add to Cargo.toml:
```toml
[[bin]]
name = "debug-validate"
path = "src/bin/debug_validate.rs"
```

```rust
// src/bin/debug_validate.rs
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    golden_dir: String,
    
    #[arg(short, long)]
    component: String,  // "tokenizer", "mel", "gpt", etc.
    
    #[arg(short, long)]
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let validator = Validator::new(&args.golden_dir);
    
    match args.component.as_str() {
        "tokenizer" => validate_tokenizer(&validator, &args.input),
        "mel" => validate_mel(&validator, &args.input),
        "speaker" => validate_speaker(&validator, &args.input),
        "gpt" => validate_gpt(&validator, &args.input),
        "vocoder" => validate_vocoder(&validator, &args.input),
        "full" => validate_full_pipeline(&validator, &args.input),
        _ => println!("Unknown component"),
    }
}
```

Usage:
```bash
cargo run --bin debug-validate -- --golden-dir ./debug/golden --component mel --input test.wav
```

---

## Test Data Setup

```bash
# Create standard test inputs
mkdir -p debug/inputs
mkdir -p debug/golden

# Copy a short reference audio (3-5 seconds)
cp "path/to/reference_speaker.wav" debug/inputs/speaker.wav

# Standard test texts
echo "Hello, this is a test." > debug/inputs/test_short.txt
echo "The quick brown fox jumps over the lazy dog." > debug/inputs/test_medium.txt
```
