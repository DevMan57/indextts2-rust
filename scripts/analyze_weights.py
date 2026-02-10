#!/usr/bin/env python3
"""List tensor names from available safetensors files."""

import sys
from pathlib import Path
try:
    from safetensors import safe_open
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

def list_tensors(filepath: str, prefix_filter: str = None, max_count: int = 50):
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"\n[SKIP] File not found: {filepath}")
        return

    print(f"\n{'='*80}")
    print(f"FILE: {filepath}")
    print(f"{'='*80}")

    try:
        with safe_open(filepath, framework="pt") as f:
            keys = sorted(f.keys())
            count = 0
            for key in keys:
                if prefix_filter and not key.startswith(prefix_filter):
                    continue
                tensor = f.get_tensor(key)
                print(f"{key}: {list(tensor.shape)}")
                count += 1
                if max_count and count >= max_count:
                    print(f"... (truncated, {len(keys)} total keys)")
                    break
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    checkpoints = Path("checkpoints")
    
    # 1. Wav2Vec-BERT
    list_tensors(checkpoints / "w2v-bert-2.0/model.safetensors", max_count=100)
    
    # 2. GPT (Qwen/UnifiedVoice)
    list_tensors(checkpoints / "qwen0.6bemo4-merge/model.safetensors", max_count=100)
    
    # 3. Search for DiT/s2mel weights
    # Try potential locations
    potential_dit = [
        checkpoints / "s2mel.safetensors",
        checkpoints / "s2mel/model.safetensors",
        checkpoints / "dit.safetensors",
    ]
    
    found_dit = False
    for p in potential_dit:
        if p.exists():
            list_tensors(p, max_count=100)
            found_dit = True
            break
            
    if not found_dit:
        print("\n[WARNING] Could not find DiT/s2mel weights in standard locations.")
