#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_gen.py
- MusicGen を使用してセグメントのみ生成するスクリプト
- 生成したトラックの連結・ループは別スクリプトで実施
"""

import os
import yaml
import torch
import argparse
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import time
from pathlib import Path


# ---------------------------
# Device Resolution
# ---------------------------
def resolve_device(cfg_device: str | None, cli_device: str | None) -> str:
    """
    Priority:
      1) CLI --device
      2) ENV DEVICE
      3) config.yaml generation.device
      4) auto detect (cuda > mps > cpu)
    """
    requested = None

    if cli_device and cli_device != "auto":
        requested = cli_device.strip().lower()

    if requested is None:
        env = os.getenv("DEVICE", "").strip().lower()
        if env and env != "auto":
            requested = env

    if requested is None and cfg_device:
        cfgd = str(cfg_device).strip().lower()
        if cfgd and cfgd != "auto":
            requested = cfgd

    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("⚠️ CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("⚠️ MPS requested but unavailable. Falling back to CPU.")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------
# Config Loading
# ---------------------------
def load_config(path="config.yaml") -> dict:
    """Load config YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------
# Audio Utils
# ---------------------------
def apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 30) -> np.ndarray:
    """Apply fade-in/out to reduce clicks at boundaries."""
    if fade_ms <= 0:
        return audio

    n = int(sr * (fade_ms / 1000.0))
    if n <= 1 or len(audio) < 2 * n:
        return audio

    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)

    audio = audio.astype(np.float32, copy=True)
    audio[:n] *= fade_in
    audio[-n:] *= fade_out
    return audio


# ---------------------------
# MusicGen
# ---------------------------
@torch.inference_mode()
def generate_segment(model, processor, prompt: str, duration_sec: int, device: str) -> tuple[np.ndarray, int]:
    """Generate one segment of audio."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # MusicGen token heuristic
    max_new_tokens = int(duration_sec * model.config.audio_encoder.frame_rate)

    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    # (batch, channels, time) -> mono float32
    audio = audio_values[0, 0].detach().cpu().numpy().astype(np.float32)
    sr = int(model.config.audio_encoder.sampling_rate)
    return audio, sr


# ---------------------------
# CLI Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MusicGen Lo-fi BGM builder")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--description", type=str, help="Override prompt description")
    p.add_argument("--duration", type=int, help="Override total duration (seconds)")
    p.add_argument("--num-tracks", type=int, help="Override number of tracks to generate")
    p.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device override",
    )
    p.add_argument("--output-dir", type=str, help="Override output directory")
    p.add_argument("--takes-dir", type=str, help="Override takes directory")
    p.add_argument("--music-preset", type=str, help="Use specific music preset from config")
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    start_time = time.perf_counter()
    args = parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    
    cfg = load_config(config_path)

    # Get model and musicgen config
    model_name = cfg.get("musicgen", {}).get("model", "facebook/musicgen-medium")
    segment_duration = int(cfg.get("musicgen", {}).get("duration", 30))

    # Get generation config
    gen_cfg = cfg.get("generation", {})
    prompt = gen_cfg.get("prompt", "lo-fi ambient music")
    num_tracks = int(gen_cfg.get("num_tracks", 4))
    device_cfg = gen_cfg.get("device", "auto")
    fade_ms = int(gen_cfg.get("fade_ms", 30))
    output_dir = gen_cfg.get("output_dir", "outputs")
    takes_dir_cfg = gen_cfg.get("takes_dir", "outputs/_takes")

    # Handle music preset
    if args.music_preset and "music" in cfg and "presets" in cfg["music"]:
        if args.music_preset in cfg["music"]["presets"]:
            music_preset = cfg["music"]["presets"][args.music_preset]
            prompt = music_preset.get("prompt", prompt)

    # CLI overrides
    if args.description:
        prompt = args.description
    if args.num_tracks:
        num_tracks = args.num_tracks
    if args.output_dir:
        output_dir = args.output_dir
    if args.takes_dir:
        takes_dir = args.takes_dir
    elif args.output_dir and takes_dir_cfg == "outputs/_takes":
        takes_dir = os.path.join(output_dir, "_takes")
    else:
        takes_dir = takes_dir_cfg
    

    # Device resolution
    device = resolve_device(cfg_device=device_cfg, cli_device=args.device)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(takes_dir, exist_ok=True)

    print("=" * 60)
    print(f"🎵 Segment length: {segment_duration} sec")
    print(f"🎯 Target duration: {num_tracks * segment_duration} sec ({num_tracks} segments)")
    print(f"🧩 Num segments  : {num_tracks}")
    print(f"🔀 Fade          : {fade_ms} ms (per segment)")
    print(f"🧠 Device        : {device}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("PROMPT")
    print("=" * 60)
    print(prompt.strip())
    print("=" * 60 + "\n")

    # Load model
    print(f"🚀 Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    # Generate segments
    segments: list[np.ndarray] = []
    sr_final: int | None = None

    for i in range(num_tracks):
        print(f"\n🎧 Generating segment {i+1}/{num_tracks}...")
        audio, sr = generate_segment(model, processor, prompt, segment_duration, device)

        if sr_final is None:
            sr_final = sr
        elif sr != sr_final:
            raise RuntimeError(f"Sample rate mismatch: {sr} vs {sr_final}")

        audio = apply_fade(audio, sr_final, fade_ms=fade_ms)

        seg_path = os.path.join(takes_dir, f"track_{i+1:02}.wav")
        sf.write(seg_path, audio, sr_final)
        print(f"   ✅ Saved: {seg_path}")

        segments.append(audio)

        # Memory cleanup
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    assert sr_final is not None

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print(f"⏱️ Generation time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print("=" * 60)

    print("\n🎉 DONE! Segments created:")
    print(f"   Takes dir: {takes_dir}")
    print(f"   Tracks: {num_tracks}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
