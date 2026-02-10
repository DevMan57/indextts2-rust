#!/usr/bin/env python3
"""List tensor names and shapes from safetensors files."""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Installing safetensors...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "-q"])
    from safetensors import safe_open

def list_tensors(filepath: str, prefix_filter: str = None, max_count: int = None):
    """List tensors from a safetensors file."""
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
            print(f"\nTotal keys: {len(keys)}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    checkpoints = Path("checkpoints")

    # List main model files with limited output first
    print("\n" + "="*80)
    print("EXAMINING SAFETENSORS FILES FOR WEIGHT MAPPING")
    print("="*80)

    # GPT weights - likely contains UnifiedVoice, Conformer, Perceiver
    list_tensors(checkpoints / "gpt.safetensors", max_count=100)

    # S2Mel weights - contains DiT
    list_tensors(checkpoints / "s2mel.safetensors", max_count=100)

    # Wav2Vec-BERT weights
    if (checkpoints / "wav2vec_bert.safetensors").exists():
        list_tensors(checkpoints / "wav2vec_bert.safetensors", max_count=100)
    elif (checkpoints / "w2v-bert-2.0" / "model.safetensors").exists():
        list_tensors(checkpoints / "w2v-bert-2.0" / "model.safetensors", max_count=100)

    # BigVGAN weights (for reference - already working)
    list_tensors(checkpoints / "bigvgan.safetensors", max_count=50)
