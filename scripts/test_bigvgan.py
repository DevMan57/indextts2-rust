#!/usr/bin/env python3
"""Test BigVGAN by encoding/decoding a reference audio."""

import torch
import torchaudio
from safetensors.torch import load_file

# Load reference audio
print("Loading reference audio...")
waveform, sr = torchaudio.load("speaker_16k.wav")
print(f"  Shape: {waveform.shape}, SR: {sr}")

# Resample to 22050 if needed
if sr != 22050:
    resampler = torchaudio.transforms.Resample(sr, 22050)
    waveform = resampler(waveform)
    print(f"  Resampled to 22050 Hz: {waveform.shape}")

# Compute mel spectrogram
print("\nComputing mel spectrogram...")
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0,
    f_max=8000,
)
mel = mel_transform(waveform)
mel = torch.log(torch.clamp(mel, min=1e-5))
print(f"  Mel shape: {mel.shape}")  # Should be [1, 80, time]
print(f"  Mel range: [{mel.min():.2f}, {mel.max():.2f}]")

# Save mel for inspection
torch.save(mel, "debug_mel.pt")
print("  Saved to debug_mel.pt")

# Try to load with our Rust code would need a test binary
# For now, just verify the mel looks reasonable
print("\nMel statistics:")
print(f"  Mean: {mel.mean():.4f}")
print(f"  Std: {mel.std():.4f}")
