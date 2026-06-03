#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_gen.py
- ACE-Step または MusicGen でセグメントを生成するスクリプト
- config.yaml の musicgen.model_type で切り替え ("acestep" | "musicgen")
- 生成したトラックの連結は assemble_bgm.py で実施
"""

import os
import yaml
import torch
import argparse
import numpy as np
import soundfile as sf
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
# ACE-Step Backend
# ---------------------------
def load_acestep(model_name: str, bf16: bool, device: str):
    """Load ACE-Step pipeline. Returns pipeline."""
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError:
        raise ImportError(
            "ACE-Step が見つかりません。以下でインストールしてください:\n"
            "  pip install -e vendor/ace-step"
        )

    # dtype: bf16=True -> "bfloat16", False -> "float32"
    # MPS環境ではpipeline内部で自動的にfloat16にフォールバックされる
    dtype = "bfloat16" if bf16 else "float32"

    # device_id: CUDA環境ではGPU番号、MPS/CPUは無視される（pipeline内部で自動検出）
    device_id = int(device.split(":")[-1]) if device.startswith("cuda") else 0

    pipeline = ACEStepPipeline(
        checkpoint_dir=model_name if model_name != "ACE-Step/ACE-Step-v1-3.5B" else None,
        device_id=device_id,
        dtype=dtype,
        torch_compile=False,
        cpu_offload=False,
    )
    return pipeline


def generate_acestep(
    pipeline,
    prompt: str,
    duration_sec: float,
    infer_step: int,
    guidance_scale: float,
    scheduler_type: str,
    cfg_type: str,
    seed: int,
    fade_ms: int,
    save_path: str,
) -> tuple[np.ndarray, int]:
    """Generate one segment with ACE-Step.
    Saves to save_path via pipeline, reads back, applies fade.
    Returns (audio_mono_float32, sample_rate).
    """
    results = pipeline(
        audio_duration=float(duration_sec),
        prompt=prompt.strip(),
        lyrics="[instrumental]",
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=10.0,
        manual_seeds=[seed],
        use_erg_diffusion=True,
        use_erg_tag=True,
        use_erg_lyric=False,
        oss_steps=None,
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        retake_seeds=None,
        retake_variance=0.5,
        save_path=save_path,
    )

    # results = [output_wav_path, ..., input_params_json_dict]
    # First element is the audio file path
    output_wav_path = results[0]

    audio, sr = sf.read(output_wav_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono
    audio = apply_fade(audio, sr, fade_ms=fade_ms)

    # Re-save with fade applied (overwrite in place)
    sf.write(output_wav_path, audio, sr)
    return audio, int(sr)


# ---------------------------
# MusicGen Backend
# ---------------------------
def load_musicgen(model_name: str, device: str):
    """Load MusicGen model. Returns (model, processor)."""
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    print(f"   ⚠️  MusicGen は CC-BY-NC 4.0 ライセンスです。商用・YouTube収益化には使用しないでください。")
    processor = AutoProcessor.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_musicgen(
    model,
    processor,
    prompt: str,
    duration_sec: int,
    device: str,
    fade_ms: int,
) -> tuple[np.ndarray, int]:
    """Generate one segment with MusicGen. Returns (audio_mono_float32, sample_rate)."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    max_new_tokens = int(duration_sec * model.config.audio_encoder.frame_rate)
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    audio = audio_values[0, 0].detach().cpu().numpy().astype(np.float32)
    sr = int(model.config.audio_encoder.sampling_rate)
    audio = apply_fade(audio, sr, fade_ms=fade_ms)
    return audio, sr


# ---------------------------
# CLI Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ACE-Step / MusicGen BGM generator")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--description", type=str, help="Override prompt")
    p.add_argument("--duration", type=int, help="Override total duration (seconds)")
    p.add_argument("--num-tracks", type=int, help="Override number of segments")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--output-dir", type=str)
    p.add_argument("--takes-dir", type=str)
    p.add_argument("--music-preset", type=str, help="Use music preset from config")
    p.add_argument("--seed", type=int, default=None, help="Random seed (-1 for random)")
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    start_time = time.perf_counter()
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    musicgen_cfg = cfg.get("musicgen", {})
    gen_cfg = cfg.get("generation", {})

    # Backend selection
    model_type = musicgen_cfg.get("model_type", "musicgen").strip().lower()
    model_name = musicgen_cfg.get("model", "facebook/musicgen-stereo-large")
    segment_duration = int(musicgen_cfg.get("duration", 30))

    # ACE-Step specific params
    infer_step = int(musicgen_cfg.get("infer_step", 27))
    guidance_scale = float(musicgen_cfg.get("guidance_scale", 15.0))
    scheduler_type = musicgen_cfg.get("scheduler_type", "euler")
    cfg_type = musicgen_cfg.get("cfg_type", "apg")
    bf16 = bool(musicgen_cfg.get("bf16", False))

    # Generation params
    prompt = gen_cfg.get("prompt", "lo-fi ambient music")
    num_tracks = int(gen_cfg.get("num_tracks", 4))
    device_cfg = gen_cfg.get("device", "auto")
    fade_ms = int(gen_cfg.get("fade_ms", 30))
    output_dir = gen_cfg.get("output_dir", "outputs")
    takes_dir_cfg = gen_cfg.get("takes_dir", "outputs/_takes")

    # Music preset override
    if args.music_preset and "music" in cfg and "presets" in cfg["music"]:
        preset = cfg["music"]["presets"].get(args.music_preset)
        if preset:
            prompt = preset.get("prompt", prompt)

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

    seed_base = args.seed if args.seed is not None else -1

    device = resolve_device(cfg_device=device_cfg, cli_device=args.device)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(takes_dir, exist_ok=True)

    total_sec = num_tracks * segment_duration
    print("=" * 60)
    print(f"🎵 Backend       : {model_type.upper()}  ({model_name})")
    print(f"⏱️  Segment length: {segment_duration} sec")
    print(f"🎯 Total duration: {total_sec} sec  ({total_sec/60:.1f} min, {num_tracks} segments)")
    print(f"🔀 Fade          : {fade_ms} ms")
    print(f"🧠 Device        : {device}")
    if model_type == "acestep":
        print(f"⚙️  Steps/CFG     : {infer_step} steps / cfg={guidance_scale} / {scheduler_type}")
    print("=" * 60)
    print("\nPROMPT\n" + "=" * 60)
    print(prompt.strip())
    print("=" * 60 + "\n")

    # Load model
    print(f"🚀 Loading {model_type} model: {model_name}")
    if model_type == "acestep":
        pipeline = load_acestep(model_name, bf16=bf16, device=device)
        model = processor = None
    else:
        model, processor = load_musicgen(model_name, device)
        pipeline = None

    # Generate segments
    sr_final: int | None = None

    for i in range(num_tracks):
        seg_seed = seed_base if seed_base >= 0 else int(time.time() * 1000) % (2**31)
        seg_path = os.path.join(takes_dir, f"track_{i+1:02}.wav")
        print(f"\n🎧 Generating segment {i+1}/{num_tracks}  (seed={seg_seed})...")

        if model_type == "acestep":
            audio, sr = generate_acestep(
                pipeline, prompt, segment_duration,
                infer_step, guidance_scale, scheduler_type, cfg_type,
                seed=seg_seed, fade_ms=fade_ms,
                save_path=seg_path,
            )
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        else:
            audio, sr = generate_musicgen(model, processor, prompt, segment_duration, device, fade_ms)
            sf.write(seg_path, audio, sr)
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        if sr_final is None:
            sr_final = sr
        elif sr != sr_final:
            raise RuntimeError(f"Sample rate mismatch: {sr} vs {sr_final}")

        elapsed_so_far = time.perf_counter() - start_time
        print(f"   ✅ Saved: {seg_path}  (elapsed: {elapsed_so_far:.0f}s)")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"⏱️  Generation time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"🎉 DONE! {num_tracks} segments in: {takes_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())



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
