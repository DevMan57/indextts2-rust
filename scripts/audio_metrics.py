#!/usr/bin/env python3
"""Compute basic WAV metrics for one file or all WAV files in a directory."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def wav_metrics(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        frames = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(frames)

    if sampwidth != 2:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    data /= 32768.0

    if data.size == 0:
        raise ValueError("Empty WAV")

    duration = data.size / sample_rate
    mean = float(np.mean(data))
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))

    window = np.hanning(data.size)
    spec = np.fft.rfft(data * window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate)

    total = float(np.sum(power) + 1e-12)
    low_180 = float(np.sum(power[freqs < 180]) / total)
    low_300 = float(np.sum(power[freqs < 300]) / total)
    high_6000 = float(np.sum(power[freqs > 6000]) / total)

    return {
        "file": str(path).replace("\\", "/"),
        "sample_rate": int(sample_rate),
        "duration_s": round(duration, 4),
        "mean": round(mean, 6),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "low_ratio_lt180": round(low_180, 6),
        "low_ratio_lt300": round(low_300, 6),
        "high_ratio_gt6000": round(high_6000, 6),
    }


def collect(path: Path) -> list[dict]:
    if path.is_file():
        return [wav_metrics(path)]

    results = [wav_metrics(p) for p in sorted(path.glob("*.wav"))]
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="WAV file or directory")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path not found: {target}")

    results = collect(target)
    payload = {"target": str(target).replace("\\", "/"), "results": results}
    text = json.dumps(payload, indent=2)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
