# Phase 1: Diagnostic Infrastructure - Research

**Researched:** 2026-01-23
**Domain:** Safetensors weight loading diagnostics in Rust (Candle ML framework)
**Confidence:** HIGH

## Summary

This research investigates how to build diagnostic infrastructure for weight loading in a Candle-based Rust ML project. The goal is to expose silent weight loading failures so that subsequent fixes can be validated.

The codebase currently loads weights from multiple safetensors files across 6+ model components (Wav2Vec-BERT, GPT/UnifiedVoice, Conformer, Perceiver, DiT, BigVGAN). Current behavior: when tensor keys are missing, some components silently fall back to random weights via `unwrap_or_else`, others error inconsistently. There is no systematic logging of what tensors exist vs. what the code expects.

**Primary recommendation:** Create a centralized `WeightDiagnostics` struct that wraps `candle_core::safetensors::load()` and tracks expected vs. found tensors per component, with opt-in verbose output via `--verbose` CLI flag.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| candle-core | 0.8 | `safetensors::load()` returns `HashMap<String, Tensor>` | Already in use, Candle's native safetensors API |
| safetensors | (via candle) | Low-level safetensors parsing, `.names()` method | Underlying library Candle uses |
| tracing | 0.1 | Structured diagnostic logging with levels | Already in use for INFO/WARN/DEBUG |
| tracing-subscriber | 0.3 | Filtering by log level | Already configured in main.rs |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| memmap2 | 0.9 | Memory-mapped file access for large files | Already a dependency, enables zero-copy header reading |
| anyhow | 1.0 | Error handling with context | Already in use throughout codebase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tracing | log crate | tracing already integrated, log is simpler but less structured |
| HashMap tracking | safetensors_explorer CLI | External tool, not integrated into pipeline |

**Installation:**
No new dependencies needed - all required libraries already in Cargo.toml.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── debug/
│   ├── mod.rs               # Re-exports
│   ├── validator.rs         # Existing golden data validation
│   ├── npy_loader.rs        # Existing NPY loading
│   └── weight_diagnostics.rs # NEW: Weight loading diagnostics
└── models/
    └── [component]/
        └── *.rs             # Modify load_weights() calls
```

### Pattern 1: Centralized Weight Diagnostics
**What:** A `WeightDiagnostics` struct that wraps safetensors loading and tracks found/missing tensors
**When to use:** All weight loading operations
**Example:**
```rust
// Source: Pattern derived from existing codebase and safetensors docs
use std::collections::{HashMap, HashSet};
use candle_core::{Device, Tensor};
use anyhow::Result;

/// Diagnostic report for a single component's weight loading
#[derive(Debug, Clone)]
pub struct ComponentReport {
    pub component_name: String,
    pub file_path: String,
    pub available_keys: Vec<String>,
    pub expected_keys: HashSet<String>,
    pub found_keys: HashSet<String>,
    pub missing_keys: HashSet<String>,
    pub extra_keys: HashSet<String>,
}

impl ComponentReport {
    pub fn success_rate(&self) -> f32 {
        if self.expected_keys.is_empty() {
            return 1.0;
        }
        self.found_keys.len() as f32 / self.expected_keys.len() as f32
    }

    pub fn print_summary(&self) {
        eprintln!("\n=== {} Weight Loading ===", self.component_name);
        eprintln!("  File: {}", self.file_path);
        eprintln!("  Available in file: {} tensors", self.available_keys.len());
        eprintln!("  Expected: {} | Found: {} | Missing: {}",
            self.expected_keys.len(),
            self.found_keys.len(),
            self.missing_keys.len()
        );
        if !self.missing_keys.is_empty() {
            eprintln!("  MISSING:");
            for key in self.missing_keys.iter().take(10) {
                eprintln!("    - {}", key);
            }
            if self.missing_keys.len() > 10 {
                eprintln!("    ... and {} more", self.missing_keys.len() - 10);
            }
        }
    }
}

/// Weight loading wrapper with diagnostics
pub struct WeightDiagnostics {
    verbose: bool,
    reports: Vec<ComponentReport>,
}

impl WeightDiagnostics {
    pub fn new(verbose: bool) -> Self {
        Self { verbose, reports: Vec::new() }
    }

    /// Load safetensors and enumerate all keys
    pub fn load_safetensors<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        component_name: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();
        let tensors = candle_core::safetensors::load(path, device)?;

        let available_keys: Vec<String> = tensors.keys().cloned().collect();

        if self.verbose {
            eprintln!("\n[{}] Loaded {} tensors from {:?}",
                component_name, available_keys.len(), path);
            eprintln!("  First 5 keys: {:?}", &available_keys[..available_keys.len().min(5)]);
        }

        Ok(tensors)
    }

    /// Record expected vs found keys for a component
    pub fn record_component(
        &mut self,
        component_name: &str,
        file_path: &str,
        available_keys: Vec<String>,
        expected_keys: HashSet<String>,
    ) {
        let found_keys: HashSet<String> = expected_keys
            .intersection(&available_keys.iter().cloned().collect())
            .cloned()
            .collect();
        let missing_keys: HashSet<String> = expected_keys
            .difference(&found_keys)
            .cloned()
            .collect();
        let extra_keys: HashSet<String> = available_keys
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .difference(&expected_keys)
            .cloned()
            .collect();

        let report = ComponentReport {
            component_name: component_name.to_string(),
            file_path: file_path.to_string(),
            available_keys,
            expected_keys,
            found_keys,
            missing_keys,
            extra_keys,
        };

        if self.verbose || !report.missing_keys.is_empty() {
            report.print_summary();
        }

        self.reports.push(report);
    }

    /// Print final summary
    pub fn print_final_summary(&self) {
        eprintln!("\n=== Weight Loading Summary ===");
        for report in &self.reports {
            let status = if report.missing_keys.is_empty() { "OK" } else { "MISSING" };
            eprintln!("  [{}] {}: {:.0}% loaded ({}/{})",
                status,
                report.component_name,
                report.success_rate() * 100.0,
                report.found_keys.len(),
                report.expected_keys.len()
            );
        }
    }
}
```

### Pattern 2: Expected Key Registry per Component
**What:** Each model component declares its expected tensor keys
**When to use:** At the point of loading weights in each component
**Example:**
```rust
// In wav2vec_bert.rs
fn expected_keys_for_layer(layer_idx: usize) -> Vec<String> {
    let prefix = format!("encoder.layers.{}", layer_idx);
    vec![
        format!("{}.self_attn.linear_q.weight", prefix),
        format!("{}.self_attn.linear_q.bias", prefix),
        format!("{}.self_attn.linear_k.weight", prefix),
        format!("{}.self_attn.linear_k.bias", prefix),
        format!("{}.self_attn.linear_v.weight", prefix),
        format!("{}.self_attn.linear_v.bias", prefix),
        format!("{}.self_attn.linear_out.weight", prefix),
        format!("{}.self_attn.linear_out.bias", prefix),
        format!("{}.ffn1.intermediate_dense.weight", prefix),
        format!("{}.ffn1.intermediate_dense.bias", prefix),
        // ... etc
    ]
}
```

### Pattern 3: Verbose Flag Propagation
**What:** Pass `--verbose` flag through inference config to all components
**When to use:** CLI invocation, controls diagnostic output
**Example:**
```rust
// In inference/pipeline.rs
pub struct InferenceConfig {
    // ... existing fields ...
    /// Enable verbose weight loading diagnostics
    pub verbose_weights: bool,
}

// In main.rs
let inference_config = InferenceConfig {
    verbose_weights: cli.verbose,
    // ...
};
```

### Anti-Patterns to Avoid
- **Silent `unwrap_or_else` with random tensors:** Currently done in multiple places - makes debugging impossible
- **Scattered diagnostic prints:** Currently using `eprintln!` ad-hoc - should use structured tracing
- **Hardcoded key expectations:** Better to derive from a single source of truth

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Listing tensor names | Manual safetensors parsing | `HashMap::keys()` from `candle_core::safetensors::load()` | Already returns HashMap, keys() is O(1) |
| Verbose logging | Custom println-based system | `tracing` with `RUST_LOG` env var | Already integrated, industry standard |
| Shape inspection | Manual dims extraction | `Tensor::dims()` method | Built into Candle |
| File existence checking | Manual fs operations | `Path::exists()` | Already used throughout codebase |

**Key insight:** The diagnostic infrastructure is mostly about tracking and reporting, not parsing. Candle already does the heavy lifting.

## Common Pitfalls

### Pitfall 1: Silently Falling Back to Random Weights
**What goes wrong:** Missing tensor keys trigger `unwrap_or_else(|| Tensor::randn(...))` with no warning
**Why it happens:** Quick fix during development to keep code running
**How to avoid:** Always emit a warning when falling back to random weights
**Warning signs:** Model produces random/noisy output despite "successful" loading

### Pitfall 2: Key Name Mismatches Between PyTorch and Rust
**What goes wrong:** PyTorch checkpoint uses `encoder.layer.0.attention.self.query` but Rust expects `encoder.layers.0.self_attn.linear_q`
**Why it happens:** Different naming conventions between frameworks
**How to avoid:** Print available keys first, then compare to expected
**Warning signs:** 0% of expected keys found, but file has hundreds of tensors

### Pitfall 3: Loading from Wrong File
**What goes wrong:** Wav2Vec-BERT component loads from gpt.safetensors instead of w2v-bert-2.0/model.safetensors
**Why it happens:** File path configuration error
**How to avoid:** Log the actual file path being loaded for each component
**Warning signs:** Keys look completely unrelated to component

### Pitfall 4: Shape Mismatches Not Detected Until Forward Pass
**What goes wrong:** Tensor loads successfully but has wrong shape
**Why it happens:** Safetensors loading doesn't validate shapes
**How to avoid:** Add optional shape validation in diagnostics
**Warning signs:** Runtime errors like "dimension mismatch" during forward()

## Code Examples

Verified patterns from existing codebase and safetensors docs:

### Loading and Enumerating Keys
```rust
// Source: Derived from src/models/vocoder/weights.rs and safetensors docs
use candle_core::{safetensors, Device, Tensor};
use std::collections::HashMap;

pub fn load_with_diagnostics<P: AsRef<std::path::Path>>(
    path: P,
    device: &Device,
    verbose: bool,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let path = path.as_ref();

    // Load all tensors
    let tensors = safetensors::load(path, device)?;

    if verbose {
        eprintln!("\n=== Tensors in {:?} ===", path);
        eprintln!("Total: {} tensors", tensors.len());

        // Group by prefix for readability
        let mut by_prefix: HashMap<String, Vec<&str>> = HashMap::new();
        for key in tensors.keys() {
            let prefix = key.split('.').next().unwrap_or("root");
            by_prefix.entry(prefix.to_string()).or_default().push(key);
        }

        for (prefix, keys) in by_prefix.iter() {
            eprintln!("  {}: {} tensors", prefix, keys.len());
        }
    }

    Ok(tensors)
}
```

### Warning on Missing Keys
```rust
// Source: Pattern from existing codebase warning messages
fn load_with_fallback(
    tensors: &HashMap<String, Tensor>,
    key: &str,
    fallback_shape: &[usize],
    device: &Device,
) -> anyhow::Result<Tensor> {
    match tensors.get(key) {
        Some(t) => Ok(t.clone()),
        None => {
            tracing::warn!(
                "Missing tensor '{}', using random initialization [{:?}]",
                key, fallback_shape
            );
            Ok(Tensor::randn(0.0f32, 0.02, fallback_shape, device)?)
        }
    }
}
```

### Integrating with CLI
```rust
// Source: Existing main.rs pattern
#[derive(Parser, Debug)]
struct Cli {
    /// Enable verbose weight loading diagnostics
    #[arg(short, long, global = true)]
    verbose: bool,
    // ...
}

// Then in inference config
let inference_config = InferenceConfig {
    verbose_weights: cli.verbose,
    // ...
};
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `println!` debug statements | `tracing` structured logging | 2023 | Filter by level, structured output |
| Manual safetensors parsing | `candle_core::safetensors::load()` | Candle 0.3+ | Single function call, returns HashMap |
| Panic on missing weights | Graceful fallback with warning | Best practice | Debugging possible |

**Deprecated/outdated:**
- Direct `safetensors` crate use when Candle is involved (use candle's wrapper)
- `eprintln!` for diagnostics (use `tracing::debug!` or `tracing::warn!`)

## Open Questions

Things that couldn't be fully resolved:

1. **Shape validation during load**
   - What we know: Shapes can be checked via `tensor.dims()`
   - What's unclear: Should shape validation be automatic or opt-in?
   - Recommendation: Opt-in, activated with `--verbose` flag to avoid performance impact

2. **Diagnostic output format**
   - What we know: Current code uses mix of `eprintln!` and `tracing`
   - What's unclear: Should diagnostics go to stderr, tracing, or both?
   - Recommendation: Use `tracing::debug!` for verbose, `tracing::warn!` for missing weights

3. **Expected key registry maintenance**
   - What we know: Key names are scattered across load functions
   - What's unclear: How to keep registry in sync with Python model updates?
   - Recommendation: Document expected keys in comments, validate against actual checkpoint

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/models/gpt/weights.rs`, `src/models/vocoder/weights.rs`
- [safetensors docs.rs](https://docs.rs/safetensors/latest/safetensors/tensor/struct.SafeTensors.html) - `names()`, `iter()` methods
- [tracing docs.rs](https://docs.rs/tracing) - Level filtering, structured logging

### Secondary (MEDIUM confidence)
- [safetensors GitHub](https://github.com/huggingface/safetensors) - File format spec
- [Tokio tracing guide](https://tokio.rs/tokio/topics/tracing) - Best practices

### Tertiary (LOW confidence)
- WebSearch results on Rust diagnostic patterns - general guidance only

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in Cargo.toml
- Architecture: HIGH - Patterns derived from existing codebase
- Pitfalls: HIGH - Documented from actual issues seen in code

**Research date:** 2026-01-23
**Valid until:** 2026-02-23 (30 days - stable domain, no breaking changes expected)
