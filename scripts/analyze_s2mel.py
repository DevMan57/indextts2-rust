#!/usr/bin/env python3
"""Analyze S2Mel safetensors structure."""

from safetensors import safe_open
from collections import defaultdict

path = "checkpoints/s2mel.safetensors"

print(f"Analyzing {path}")
print("="*80)

groups = defaultdict(list)

with safe_open(path, framework="pt") as f:
    for key in sorted(f.keys()):
        tensor = f.get_tensor(key)
        shape = list(tensor.shape)

        # Group by prefix
        parts = key.split(".")
        if parts[0] == "cfm":
            if parts[1] == "estimator":
                if parts[2] == "transformer":
                    prefix = ".".join(parts[:4])  # cfm.estimator.transformer.layers
                else:
                    prefix = ".".join(parts[:3])  # cfm.estimator.*
            else:
                prefix = ".".join(parts[:2])
        else:
            prefix = parts[0]

        groups[prefix].append((key, shape))

# Print grouped summary
for prefix in sorted(groups.keys()):
    items = groups[prefix]
    print(f"\n{prefix}:")
    for key, shape in items[:10]:  # First 10 of each group
        print(f"  {key}: {shape}")
    if len(items) > 10:
        print(f"  ... ({len(items)} total)")
