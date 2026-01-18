---
name: implement-phase
description: Implement a specific phase of the IndexTTS2 Rust rewrite
argument-hint: [phase-number] [phase-name]
allowed-tools: Read, Write, Bash(cargo:*), Bash(rustfmt:*), Grep, Context7
---

# Phase $1: $2

You are implementing Phase $1 ($2) of the IndexTTS2 Rust rewrite.

## Current Project State

Read the progress tracker in CLAUDE.md:
```
@CLAUDE.md
```

## Context7 Research

Before writing any code, fetch relevant documentation:

### For Audio Processing:
```
Context7:resolve-library-id("cpal rust audio")
Context7:resolve-library-id("rubato resampler")
Context7:resolve-library-id("symphonia audio decode")
```

### For ML/Tensor Operations:
```
Context7:get-library-docs("/huggingface/candle", topic="transformer inference kv-cache")
```

## Implementation Workflow

1. **List all modules** for this phase from the architecture diagram
2. **For each module:**
   - Read corresponding Python source
   - Research Rust crate equivalents
   - Write Rust implementation
   - Add to `mod.rs` exports
   - Run `cargo check`

3. **Integration:**
   - Wire up modules in parent `mod.rs`
   - Add integration tests in `tests/`
   - Verify with `cargo test`

## Success Output

When phase is complete, output:
```
<promise>PHASE_$1_COMPLETE</promise>
```

## Iteration Guidelines

- Use `cargo check` frequently (not `cargo build`)
- Print tensor shapes during debugging
- Comment complex Candle operations with PyTorch equivalents
- If stuck on a specific operation, move to next module and note it
