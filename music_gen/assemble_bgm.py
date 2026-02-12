#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assemble_bgm.py
- 生成済みのトラック (track_*.wav) をランダムに並べてクロスフェード結合し、指定長にループして出力
- 音源生成は music_gen.py が担当し、本スクリプトは組み立て専用
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def loop_to_length_crossfade(audio: np.ndarray, sr: int, target_sec: int, crossfade_ms: int = 30) -> np.ndarray:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble generated tracks into BGM")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--takes-dir", type=str, help="Directory containing track_*.wav")
    p.add_argument("--output", type=str, help="Output file path for assembled BGM")
    p.add_argument("--output-dir", type=str, help="Override output directory (used with config final_output)")
    p.add_argument("--duration", type=int, help="Target BGM length (seconds)")
    p.add_argument("--num-tracks", type=int, help="Number of tracks to draw (<= available)")
    p.add_argument("--track-ids", type=str, help="Comma-separated 1-based track numbers to use (disables shuffle)")
    p.add_argument("--crossfade", type=int, help="Crossfade length in ms")
    p.add_argument("--seed", type=int, help="Shuffle seed")
    return p.parse_args()


def parse_track_ids(spec: str) -> list[int]:
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    ids: list[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"track id must be positive integer: '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("track id must be >= 1")
        ids.append(v)
    # keep order provided; allow duplicates? better to de-dup while preserving order
    seen = set()
    ordered_unique: list[int] = []
    for v in ids:
        if v in seen:
            continue
        seen.add(v)
        ordered_unique.append(v)
    return ordered_unique


def main() -> int:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    gen_cfg = cfg.get("generation", {})

    crossfade_ms = args.crossfade if args.crossfade is not None else int(gen_cfg.get("crossfade_ms", 250))
    target_sec = args.duration if args.duration is not None else int(gen_cfg.get("total_duration_sec", 3600))
    output_dir_cfg = gen_cfg.get("output_dir", "outputs")
    takes_dir_cfg = gen_cfg.get("takes_dir", "outputs/_takes")
    final_output_name = gen_cfg.get("final_output", "bgm.wav")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(output_dir_cfg).expanduser()

    if args.takes_dir:
        takes_dir = Path(args.takes_dir).expanduser()
    elif args.output_dir and takes_dir_cfg == "outputs/_takes":
        takes_dir = output_dir / "_takes"
    else:
        takes_dir = Path(takes_dir_cfg).expanduser()

    output_path = Path(args.output).expanduser() if args.output else output_dir / final_output_name

    files = sorted(takes_dir.glob("track_*.wav"))
    if not files:
        print(f"❌ No tracks found in {takes_dir}")
        return 1

    if args.track_ids:
        try:
            track_ids = parse_track_ids(args.track_ids)
        except ValueError as e:
            print(f"❌ {e}")
            return 1

        max_id = len(files)
        for tid in track_ids:
            if tid > max_id:
                print(f"❌ track id {tid} exceeds available tracks ({max_id}).")
                return 1
        selected = [files[tid - 1] for tid in track_ids]
        num_tracks = len(selected)
        shuffle_used = False
    else:
        if args.num_tracks:
            num_tracks = args.num_tracks
        elif gen_cfg.get("num_tracks"):
            num_tracks = int(gen_cfg.get("num_tracks"))
        else:
            num_tracks = len(files)

        if num_tracks <= 0:
            print("❌ num_tracks must be > 0")
            return 1

        if num_tracks > len(files):
            print(f"⚠️ Requested {num_tracks} tracks but only {len(files)} available. Using available tracks.")
            num_tracks = len(files)

        rng = random.Random(args.seed)
        shuffled_files = files[:]
        rng.shuffle(shuffled_files)
        selected = shuffled_files[:num_tracks]
        shuffle_used = True

    print("=" * 60)
    print(f"📁 Takes dir : {takes_dir}")
    print(f"🎵 Using     : {num_tracks} tracks (from {len(files)} available)")
    if shuffle_used:
        print(f"🔀 Seed      : {args.seed if args.seed is not None else 'random'}")
    else:
        print(f"🎯 Track ids : {args.track_ids}")
    print(f"🔗 Crossfade : {crossfade_ms} ms")
    print(f"🎯 Target    : {target_sec} sec")
    print(f"💾 Output    : {output_path}")
    print("=" * 60)

    segments: list[np.ndarray] = []
    sr_final: int | None = None

    for path in selected:
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr_final is None:
            sr_final = sr
        elif sr != sr_final:
            raise RuntimeError(f"Sample rate mismatch: {sr} vs {sr_final}")
        segments.append(audio.astype(np.float32))

    assert sr_final is not None

    concatenated = crossfade_concat(segments, sr_final, crossfade_ms=crossfade_ms)
    final_audio = loop_to_length_crossfade(concatenated, sr_final, target_sec, crossfade_ms=crossfade_ms)
    final_stereo = to_safe_stereo(final_audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, final_stereo, sr_final)

    print("\n✅ Assembled BGM saved:")
    print(f"   File : {output_path}")
    print(f"   SR   : {sr_final} Hz")
    print(f"   Time : {target_sec} sec")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
