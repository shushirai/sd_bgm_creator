#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_gen.py
- MusicGen を使用してBGMを生成
- 複数セグメント生成 → シャッフル → クロスフェード接続 → ループで指定長に拡張
- lofi_batch.py の手法に基づいて実装
"""

import os
import math
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


def crossfade_concat(chunks: list[np.ndarray], sr: int, crossfade_ms: int = 250) -> np.ndarray:
    """Concatenate audio chunks with crossfade."""
    if not chunks:
        raise ValueError("No audio chunks to concatenate.")

    if crossfade_ms <= 0 or len(chunks) == 1:
        return np.concatenate(chunks)

    xf = int(sr * (crossfade_ms / 1000.0))
    if xf <= 1:
        return np.concatenate(chunks)

    out = chunks[0].astype(np.float32)
    for nxt in chunks[1:]:
        nxt = nxt.astype(np.float32)

        if len(out) < xf or len(nxt) < xf:
            out = np.concatenate([out, nxt])
            continue

        a = out[:-xf]
        b1 = out[-xf:]
        b2 = nxt[:xf]
        c = nxt[xf:]

        w = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        blended = b1 * (1.0 - w) + b2 * w
        out = np.concatenate([a, blended, c])

    return out


def loop_to_length_crossfade(
    audio: np.ndarray, sr: int, target_sec: int, crossfade_ms: int = 30
) -> np.ndarray:
    """Loop audio to target length using crossfade."""
    if target_sec <= 0:
        raise ValueError("target_sec must be > 0")
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("audio is empty")

    target_samples = int(target_sec * sr)
    if len(audio) >= target_samples:
        return audio[:target_samples]

    loops = int(math.ceil(target_samples / len(audio)))
    chunks = [audio] * loops
    out = crossfade_concat(chunks, sr, crossfade_ms=crossfade_ms)

    if len(out) < target_samples:
        pad = np.tile(audio, int(math.ceil((target_samples - len(out)) / len(audio))) + 1)
        out = np.concatenate([out, pad], axis=0)

    return out[:target_samples]


def to_safe_stereo(audio_mono: np.ndarray) -> np.ndarray:
    """Convert mono to stereo if needed."""
    a = np.asarray(audio_mono)
    if a.ndim == 1:
        return np.stack([a, a], axis=1).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 2:
        return a.astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {a.shape}")


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
    total_sec = int(gen_cfg.get("total_duration_sec", 3600))
    select_track = gen_cfg.get("select_track", "auto")
    device_cfg = gen_cfg.get("device", "auto")
    crossfade_ms = int(gen_cfg.get("crossfade_ms", 250))
    fade_ms = int(gen_cfg.get("fade_ms", 30))
    output_dir = gen_cfg.get("output_dir", "outputs")
    takes_dir = gen_cfg.get("takes_dir", "outputs/_takes")
    final_dir = gen_cfg.get("final_dir", "outputs/_final")
    final_output = gen_cfg.get("final_output", "bgm.wav")

    # Handle music preset
    if args.music_preset and "music" in cfg and "presets" in cfg["music"]:
        if args.music_preset in cfg["music"]["presets"]:
            music_preset = cfg["music"]["presets"][args.music_preset]
            prompt = music_preset.get("prompt", prompt)

    # CLI overrides
    if args.description:
        prompt = args.description
    if args.duration:
        total_sec = args.duration
    if args.num_tracks:
        num_tracks = args.num_tracks
    if args.output_dir:
        output_dir = args.output_dir

    # Device resolution
    device = resolve_device(cfg_device=device_cfg, cli_device=args.device)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(takes_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print("=" * 60)
    print(f"🎯 Target length : {total_sec} sec")
    print(f"🎵 Segment length: {segment_duration} sec")
    print(f"🧩 Num segments  : {num_tracks}")
    print(f"🔀 Crossfade     : {crossfade_ms} ms | Fade: {fade_ms} ms")
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

    # Shuffle segments
    print(f"\n🔀 Shuffling {num_tracks} tracks...")
    shuffled_indices = list(range(num_tracks))
    np.random.shuffle(shuffled_indices)
    shuffled_segments = [segments[i] for i in shuffled_indices]

    # Concatenate with crossfade
    print(f"🔗 Concatenating all {num_tracks} tracks with crossfade...")
    concatenated_audio = crossfade_concat(shuffled_segments, sr_final, crossfade_ms=crossfade_ms)

    # Save playlist
    playlist_path = os.path.join(output_dir, "playlist_all_tracks.wav")
    sf.write(playlist_path, concatenated_audio, sr_final)
    playlist_duration_sec = len(concatenated_audio) / sr_final
    print(f"✅ Playlist saved: {playlist_path}")
    print(f"   Length: {playlist_duration_sec:.1f} sec ({playlist_duration_sec/60:.2f} min)")

    # Loop to target length
    print(f"\n🔁 Looping to target length ({total_sec} sec)...")
    final_audio = loop_to_length_crossfade(concatenated_audio, sr_final, total_sec, crossfade_ms=crossfade_ms)

    # Convert to stereo
    final_audio_stereo = to_safe_stereo(final_audio)

    # Save final audio as stereo
    final_path = os.path.join(output_dir, final_output)
    sf.write(final_path, final_audio_stereo, sr_final)
    print(f"✅ Final audio saved (stereo): {final_path}")

    # Also save stereo version to _final folder for backup
    final_path_stereo = os.path.join(final_dir, f"{Path(final_output).stem}_stereo.wav")
    sf.write(final_path_stereo, final_audio_stereo, sr_final)
    print(f"✅ Stereo backup saved: {final_path_stereo}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print(f"⏱️ Generation time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print("=" * 60)

    print("\n🎉 DONE! BGM created:")
    print(f"   Raw   : {final_path}")
    print(f"   Stereo: {final_path_stereo}")
    print(f"   Length: {total_sec} sec")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
