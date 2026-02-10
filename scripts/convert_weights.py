#!/usr/bin/env python3
"""
Convert PyTorch .pth weights to safetensors format for Rust loading.

Usage:
    python convert_weights.py [--checkpoint-dir checkpoints/] [--output-dir checkpoints/]

This script converts:
    - gpt.pth → gpt.safetensors
    - s2mel.pth → s2mel.safetensors
    - feat1.pt → speaker_matrix.safetensors
    - feat2.pt → emotion_matrix.safetensors
    - wav2vec2bert_stats.pt → wav2vec2bert_stats.safetensors
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def flatten_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Flatten nested state dict into flat key-value pairs."""
    result = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_state_dict(value, full_key))
        elif isinstance(value, torch.Tensor):
            result[full_key] = value
        # Skip non-tensor values
    return result


def convert_gpt_weights(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert GPT weights from gpt.pth to gpt.safetensors."""
    input_path = checkpoint_dir / "gpt.pth"
    output_path = output_dir / "gpt.safetensors"

    if not input_path.exists():
        print(f"  [SKIP] {input_path} not found")
        return

    print(f"  Loading {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    # Handle nested state dict (e.g., {"model": {...}})
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Flatten and ensure all values are tensors
    tensors = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert to contiguous and appropriate dtype
            tensor = value.contiguous()
            # Keep original dtype but ensure it's supported
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            tensors[key] = tensor

    print(f"  Found {len(tensors)} tensors")
    print(f"  Saving to {output_path}...")
    save_file(tensors, str(output_path))

    # Print summary
    total_params = sum(t.numel() for t in tensors.values())
    print(f"  [OK] GPT: {len(tensors)} tensors, {total_params:,} parameters")

    # Print key sample
    print("  Sample keys:")
    for i, key in enumerate(sorted(tensors.keys())[:10]):
        shape = list(tensors[key].shape)
        print(f"    - {key}: {shape}")
    if len(tensors) > 10:
        print(f"    ... and {len(tensors) - 10} more")


def convert_s2mel_weights(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert S2Mel weights from s2mel.pth to s2mel.safetensors.

    S2Mel checkpoint structure:
        - net.cfm: Flow matching (DiT) model
        - net.length_regulator: Duration predictor
        - net.gpt_layer: Projection layers
    """
    input_path = checkpoint_dir / "s2mel.pth"
    output_path = output_dir / "s2mel.safetensors"

    if not input_path.exists():
        print(f"  [SKIP] {input_path} not found")
        return

    print(f"  Loading {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)

    # S2Mel checkpoint has: net, optimizer, scheduler, iters, epoch
    # We only need the 'net' dict containing model weights
    if "net" not in checkpoint:
        print(f"  [ERROR] 'net' key not found in s2mel.pth")
        return

    net = checkpoint["net"]

    # Flatten the nested structure: net.cfm, net.length_regulator, net.gpt_layer
    tensors = {}
    for module_name, module_dict in net.items():
        if isinstance(module_dict, dict):
            for key, value in module_dict.items():
                if isinstance(value, torch.Tensor):
                    # Create full key: cfm.estimator.xxx, length_regulator.xxx, etc.
                    full_key = f"{module_name}.{key}"
                    tensor = value.contiguous()
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float16)
                    tensors[full_key] = tensor
        elif isinstance(module_dict, torch.Tensor):
            tensor = module_dict.contiguous()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            tensors[module_name] = tensor

    print(f"  Found {len(tensors)} tensors")
    print(f"  Saving to {output_path}...")
    save_file(tensors, str(output_path))

    total_params = sum(t.numel() for t in tensors.values())
    print(f"  [OK] S2Mel: {len(tensors)} tensors, {total_params:,} parameters")

    # Print summary by module
    modules = {}
    for key in tensors.keys():
        module = key.split('.')[0]
        if module not in modules:
            modules[module] = 0
        modules[module] += 1
    print("  By module:")
    for module, count in sorted(modules.items()):
        print(f"    - {module}: {count} tensors")

    # Print key sample
    print("  Sample keys:")
    for i, key in enumerate(sorted(tensors.keys())[:10]):
        shape = list(tensors[key].shape)
        print(f"    - {key}: {shape}")
    if len(tensors) > 10:
        print(f"    ... and {len(tensors) - 10} more")


def convert_small_tensors(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert smaller .pt files to safetensors."""
    small_files = [
        ("feat1.pt", "speaker_matrix.safetensors", "Speaker Matrix"),
        ("feat2.pt", "emotion_matrix.safetensors", "Emotion Matrix"),
        ("wav2vec2bert_stats.pt", "wav2vec2bert_stats.safetensors", "Wav2Vec-BERT Stats"),
    ]

    for input_name, output_name, description in small_files:
        input_path = checkpoint_dir / input_name
        output_path = output_dir / output_name

        if not input_path.exists():
            print(f"  [SKIP] {input_path} not found")
            continue

        print(f"  Loading {input_name}...")
        data = torch.load(input_path, map_location="cpu", weights_only=True)

        tensors = {}
        if isinstance(data, torch.Tensor):
            # Single tensor
            tensors["data"] = data.contiguous()
        elif isinstance(data, dict):
            # State dict
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    tensor = value.contiguous()
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float16)
                    tensors[key] = tensor

        if tensors:
            print(f"  Saving to {output_path}...")
            save_file(tensors, str(output_path))
            total_params = sum(t.numel() for t in tensors.values())
            shapes = {k: list(v.shape) for k, v in tensors.items()}
            print(f"  [OK] {description}: {shapes}, {total_params:,} parameters")
        else:
            print(f"  [WARN] No tensors found in {input_name}")


def inspect_checkpoint(checkpoint_path: Path) -> None:
    """Inspect a checkpoint file and print its structure."""
    print(f"\n=== Inspecting {checkpoint_path} ===")

    if not checkpoint_path.exists():
        print(f"  File not found")
        return

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    def print_structure(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            print(f"{prefix}dict with {len(obj)} keys:")
            for key in sorted(obj.keys())[:20]:
                value = obj[key]
                if isinstance(value, torch.Tensor):
                    print(f"{prefix}  - {key}: Tensor{list(value.shape)} dtype={value.dtype}")
                elif isinstance(value, dict):
                    print(f"{prefix}  - {key}: dict with {len(value)} keys")
                else:
                    print(f"{prefix}  - {key}: {type(value).__name__}")
            if len(obj) > 20:
                print(f"{prefix}  ... and {len(obj) - 20} more keys")
        elif isinstance(obj, torch.Tensor):
            print(f"{prefix}Tensor: shape={list(obj.shape)}, dtype={obj.dtype}")
        else:
            print(f"{prefix}{type(obj).__name__}")

    print_structure(data)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to safetensors")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing PyTorch checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for safetensors (default: same as checkpoint-dir)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect checkpoint structure without converting"
    )
    parser.add_argument(
        "--gpt-only",
        action="store_true",
        help="Only convert GPT weights"
    )
    parser.add_argument(
        "--s2mel-only",
        action="store_true",
        help="Only convert S2Mel weights"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        # Inspect mode
        for name in ["gpt.pth", "s2mel.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"]:
            inspect_checkpoint(args.checkpoint_dir / name)
        return

    print("=" * 60)
    print("IndexTTS2 Weight Converter: PyTorch -> Safetensors")
    print("=" * 60)
    print(f"Input:  {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print()

    if args.gpt_only:
        print("[1/1] Converting GPT weights...")
        convert_gpt_weights(args.checkpoint_dir, args.output_dir)
    elif args.s2mel_only:
        print("[1/1] Converting S2Mel weights...")
        convert_s2mel_weights(args.checkpoint_dir, args.output_dir)
    else:
        print("[1/3] Converting GPT weights...")
        convert_gpt_weights(args.checkpoint_dir, args.output_dir)
        print()

        print("[2/3] Converting S2Mel weights...")
        convert_s2mel_weights(args.checkpoint_dir, args.output_dir)
        print()

        print("[3/3] Converting smaller tensors...")
        convert_small_tensors(args.checkpoint_dir, args.output_dir)

    print()
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
