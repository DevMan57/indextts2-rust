#!/usr/bin/env python3
"""
Dump intermediate tensors from Python IndexTTS model for Rust validation.

This script hooks into the Python IndexTTS2 model and saves intermediate
tensors at each layer as .npy files for comparison with the Rust implementation.

Usage:
    python dump_python.py --text "Hello world" --speaker voice.wav --output golden/

Output structure:
    golden/
    ├── input/
    │   ├── text.txt            # Original text
    │   ├── tokens.npy          # Tokenized text
    │   └── speaker_audio.npy   # Speaker audio samples
    ├── encoders/
    │   ├── speaker_emb.npy     # CAMPPlus output (192-dim)
    │   ├── semantic_feat.npy   # Wav2Vec-BERT features
    │   └── semantic_codes.npy  # VQ codes
    ├── gpt/
    │   ├── conditioning.npy    # Conformer + Perceiver output
    │   ├── layer_00.npy        # First decoder layer
    │   ├── layer_12.npy        # Middle layer
    │   ├── layer_23.npy        # Last layer
    │   └── mel_codes.npy       # Generated mel codes
    ├── synthesis/
    │   ├── length_reg.npy      # Length regulator output
    │   ├── dit_step_00.npy     # Flow matching step 0
    │   ├── dit_step_12.npy     # Flow matching step 12
    │   ├── dit_step_24.npy     # Flow matching step 24
    │   └── mel_spec.npy        # Final mel spectrogram
    └── output/
        ├── audio.npy           # Generated audio samples
        └── audio.wav           # Audio file for listening
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# Try to import IndexTTS - adjust path as needed
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "index-tts"))
    from indextts.infer_v2 import IndexTTS
except ImportError:
    print("Warning: IndexTTS not found. Install from: https://github.com/index-tts/index-tts")
    print("Or adjust the path in this script.")
    IndexTTS = None


class TensorDumper:
    """Hook into model layers and dump tensors."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["input", "encoders", "gpt", "synthesis", "output"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)

    def save(self, name: str, tensor, subdir: str = ""):
        """Save a tensor as .npy file."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        if subdir:
            path = self.output_dir / subdir / f"{name}.npy"
        else:
            path = self.output_dir / f"{name}.npy"

        np.save(path, tensor)
        print(f"Saved: {path} shape={tensor.shape} dtype={tensor.dtype}")

    def save_text(self, name: str, text: str, subdir: str = "input"):
        """Save text to file."""
        path = self.output_dir / subdir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"Saved: {path}")


def dump_with_hooks(model, text: str, speaker_audio_path: str, dumper: TensorDumper):
    """Run inference with hooks to capture intermediate tensors."""

    # Save input
    dumper.save_text("text", text)

    # Load and save speaker audio
    audio, sr = sf.read(speaker_audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Mono
    dumper.save("speaker_audio", audio.astype(np.float32), "input")

    # Tokenize text
    tokens = model.tokenizer.encode(text)
    dumper.save("tokens", np.array(tokens, dtype=np.int64), "input")

    # Extract speaker embedding
    speaker_audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    if hasattr(model, 'speaker_encoder'):
        speaker_emb = model.speaker_encoder(speaker_audio_tensor)
        dumper.save("speaker_emb", speaker_emb, "encoders")

    # Extract semantic features
    if hasattr(model, 'wav2vec'):
        semantic_feat = model.wav2vec(speaker_audio_tensor)
        dumper.save("semantic_feat", semantic_feat, "encoders")

    # Get semantic codes
    if hasattr(model, 'semantic_codec'):
        semantic_codes = model.semantic_codec.quantize(semantic_feat)
        dumper.save("semantic_codes", semantic_codes, "encoders")

    # GPT generation with layer hooks
    if hasattr(model, 'gpt'):
        gpt = model.gpt

        # Hook for layer outputs
        layer_outputs = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[name] = output[0]
                else:
                    layer_outputs[name] = output
            return hook

        # Register hooks on specific layers
        hooks = []
        if hasattr(gpt, 'transformer') and hasattr(gpt.transformer, 'h'):
            for i in [0, 12, 23]:
                if i < len(gpt.transformer.h):
                    h = gpt.transformer.h[i].register_forward_hook(make_hook(f"layer_{i:02d}"))
                    hooks.append(h)

        # Run generation
        mel_codes = model.generate_mel_codes(text, speaker_audio_path)
        dumper.save("mel_codes", np.array(mel_codes, dtype=np.int64), "gpt")

        # Save captured layers
        for name, tensor in layer_outputs.items():
            dumper.save(name, tensor, "gpt")

        # Remove hooks
        for h in hooks:
            h.remove()

    # Synthesis with flow matching hooks
    if hasattr(model, 's2mel'):
        s2mel = model.s2mel

        # Length regulator
        if hasattr(s2mel, 'length_regulator'):
            # This would need the actual intermediate tensor
            pass

        # DiT / Flow matching
        if hasattr(s2mel, 'dit'):
            # Capture steps 0, 12, 24
            pass

    # Final synthesis
    if hasattr(model, 'vocoder'):
        # Get mel spectrogram and audio
        audio_output = model.infer(text, speaker_audio_path)
        dumper.save("audio", np.array(audio_output, dtype=np.float32), "output")

        # Save as WAV
        wav_path = dumper.output_dir / "output" / "audio.wav"
        sf.write(wav_path, audio_output, 22050)
        print(f"Saved: {wav_path}")


def dump_mock_data(dumper: TensorDumper, text: str):
    """Generate mock data when IndexTTS is not available."""
    print("Generating mock golden data (IndexTTS not available)")

    # Input
    dumper.save_text("text", text)
    tokens = np.array([ord(c) % 1000 for c in text], dtype=np.int64)
    dumper.save("tokens", tokens, "input")
    dumper.save("speaker_audio", np.random.randn(16000).astype(np.float32), "input")

    # Encoders
    dumper.save("speaker_emb", np.random.randn(1, 192).astype(np.float32), "encoders")
    dumper.save("semantic_feat", np.random.randn(1, 50, 1024).astype(np.float32), "encoders")
    dumper.save("semantic_codes", np.random.randint(0, 8192, (1, 50), dtype=np.int64), "encoders")

    # GPT
    dumper.save("conditioning", np.random.randn(1, 32, 1280).astype(np.float32), "gpt")
    dumper.save("layer_00", np.random.randn(1, 100, 1280).astype(np.float32), "gpt")
    dumper.save("layer_12", np.random.randn(1, 100, 1280).astype(np.float32), "gpt")
    dumper.save("layer_23", np.random.randn(1, 100, 1280).astype(np.float32), "gpt")
    mel_codes = np.random.randint(0, 8192, (100,), dtype=np.int64)
    dumper.save("mel_codes", mel_codes, "gpt")

    # Synthesis
    dumper.save("length_reg", np.random.randn(1, 256, 512).astype(np.float32), "synthesis")
    dumper.save("dit_step_00", np.random.randn(1, 256, 80).astype(np.float32), "synthesis")
    dumper.save("dit_step_12", np.random.randn(1, 256, 80).astype(np.float32), "synthesis")
    dumper.save("dit_step_24", np.random.randn(1, 256, 80).astype(np.float32), "synthesis")
    dumper.save("mel_spec", np.random.randn(1, 256, 80).astype(np.float32), "synthesis")

    # Output
    audio = np.random.randn(256 * 256).astype(np.float32) * 0.1  # 256x upsampling
    dumper.save("audio", audio, "output")

    # Save mock WAV
    wav_path = dumper.output_dir / "output" / "audio.wav"
    sf.write(wav_path, audio, 22050)
    print(f"Saved mock: {wav_path}")


def main():
    parser = argparse.ArgumentParser(description="Dump IndexTTS tensors for Rust validation")
    parser.add_argument("--text", "-t", default="Hello world, this is a test.",
                       help="Text to synthesize")
    parser.add_argument("--speaker", "-s", default=None,
                       help="Path to speaker reference audio")
    parser.add_argument("--output", "-o", default="golden",
                       help="Output directory for tensors")
    parser.add_argument("--model-dir", "-m", default="checkpoints",
                       help="Directory containing model checkpoints")
    parser.add_argument("--mock", action="store_true",
                       help="Generate mock data (no model needed)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    dumper = TensorDumper(output_dir)

    if args.mock or IndexTTS is None:
        dump_mock_data(dumper, args.text)
    else:
        if args.speaker is None:
            print("Error: --speaker is required when not using --mock")
            sys.exit(1)

        print(f"Loading IndexTTS from {args.model_dir}")
        model = IndexTTS(args.model_dir)

        print(f"Running inference on: {args.text}")
        dump_with_hooks(model, args.text, args.speaker, dumper)

    print(f"\nGolden data saved to: {output_dir}")
    print("Use these files to validate the Rust implementation.")


if __name__ == "__main__":
    main()
