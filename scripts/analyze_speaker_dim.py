#!/usr/bin/env python3
"""Analyze what speaker dimension the checkpoint expects."""

from safetensors import safe_open
import torch

# Check s2mel for speaker-related weights
print("=== S2MEL Speaker Conditioning ===")
with safe_open("checkpoints/s2mel.safetensors", framework="pt") as f:
    for key in sorted(f.keys()):
        if "cond" in key.lower() or "style" in key.lower() or "spk" in key.lower():
            tensor = f.get_tensor(key)
            print(f"{key}: {list(tensor.shape)}")

# Check GPT for speaker-related weights
print("\n=== GPT Speaker Conditioning ===")
with safe_open("checkpoints/gpt.safetensors", framework="pt") as f:
    for key in sorted(f.keys()):
        if "cond" in key.lower() or "style" in key.lower() or "spk" in key.lower():
            tensor = f.get_tensor(key)
            print(f"{key}: {list(tensor.shape)}")

# Check if there's a speaker encoder or projection
print("\n=== Looking for speaker projection ===")
with safe_open("checkpoints/s2mel.safetensors", framework="pt") as f:
    # Look at cond_embedder input dimension
    cond_emb = f.get_tensor("cfm.estimator.cond_embedder.weight")
    print(f"cond_embedder.weight shape: {list(cond_emb.shape)}")
    print(f"  -> Expects input dim: {cond_emb.shape[1]}, outputs: {cond_emb.shape[0]}")

    # Check cond_projection
    if "cfm.estimator.cond_projection.weight" in f.keys():
        cond_proj = f.get_tensor("cfm.estimator.cond_projection.weight")
        print(f"cond_projection.weight shape: {list(cond_proj.shape)}")

# Look for any 192-dim related weights (CAMPPlus output)
print("\n=== Checking for 192-dim weights ===")
for checkpoint in ["checkpoints/s2mel.safetensors", "checkpoints/gpt.safetensors"]:
    print(f"\n{checkpoint}:")
    with safe_open(checkpoint, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if 192 in tensor.shape:
                print(f"  {key}: {list(tensor.shape)}")
