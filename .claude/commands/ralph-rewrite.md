---
name: ralph-rewrite
description: Start a Ralph Wiggum autonomous loop to rewrite a Python module to Rust
argument-hint: [python-module] [rust-target]
allowed-tools: Read, Write, Bash(cargo:*), Bash(rustfmt:*), Grep
---

# Ralph Loop: Python → Rust Module Rewrite

You are rewriting **$1** from the original Python IndexTTS2 codebase to Rust.
Target Rust file: **$2**

## Phase 1: Analysis (DO NOT SKIP)

First, read and deeply understand the Python source:
```
@C:\AI\index-tts\indextts\$1
```

### Questions to Answer:
1. What is this module's PRIMARY responsibility?
2. What are the INPUT types (tensors, configs, paths)?
3. What are the OUTPUT types?
4. What PyTorch operations are used?
5. What external dependencies does it have?

## Phase 2: Rust Implementation

Use **Candle** for tensor operations. Before writing code:

```
Context7:get-library-docs("/huggingface/candle", topic="$1 transformer")
```

### Implementation Rules:
1. Create proper Rust struct(s) matching Python class(es)
2. Implement `new()` constructor with config loading
3. Implement `forward()` method matching Python's forward pass
4. Use `anyhow::Result` for error handling
5. Add `#[derive(Debug, Clone)]` where appropriate
6. Document all public APIs with `///` comments

## Phase 3: Verification

After implementation:
1. Run `cargo check --features cuda` to verify compilation
2. Run `cargo clippy --features cuda` for lint warnings
3. Run `rustfmt src/$2` to format code
4. Add unit test stubs in `#[cfg(test)]` module

## Completion Criteria

Output `<promise>MODULE_COMPLETE</promise>` when:
- [ ] Rust code compiles without errors
- [ ] All PyTorch operations mapped to Candle equivalents
- [ ] Type signatures documented
- [ ] Basic tests stubbed out

If blocked after 10 iterations:
- Document what's blocking progress
- List attempted approaches
- Suggest alternative implementations
