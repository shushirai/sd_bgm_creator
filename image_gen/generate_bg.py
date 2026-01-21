#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_bg.py
- config.yaml の preset を使って Stable Diffusion 背景画像を量産
- CUDA( NVIDIA ) 前提（CPUでも動くが遅い）
- safetensors 単体ファイル / Hugging Face repo の両対応

依存:
  pip install diffusers transformers accelerate safetensors pyyaml pillow torch

例:
  python3 generate_bg.py --preset japan_rain --count 10
  python3 generate_bg.py --config config.yaml --preset japan_neon --count 6 --seed 1000
  python3 generate_bg.py --preset japan_rain --count 10 --steps 35 --cfg 7.0 --width 1024 --height 576
  python3 generate_bg.py --preset japan_rain --count 10 --fp32   # 黒画像など出る時の切り分け
"""

from __future__ import annotations

import argparse
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline

# -----------------------------
# Utilities
# -----------------------------


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_probably_file_model(model_str: str) -> bool:
    # ざっくり判定: .safetensors / .ckpt / .pt など
    s = model_str.lower()
    return any(s.endswith(ext) for ext in [".safetensors", ".ckpt", ".pt", ".bin"])


def resolve_device(device_arg: str) -> Tuple[torch.device, str]:
    """Resolve device string into (torch.device, device_name).

    Supported: auto, cuda, mps, cpu
    Priority (auto): cuda -> mps -> cpu
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested but CUDA is not available.")
        return torch.device("cuda"), "cuda"

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device=mps was requested but MPS is not available.")
        return torch.device("mps"), "mps"

    if device_arg == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unknown device: {device_arg}")


def pick_torch_dtype(device_name: str, fp32: bool) -> torch.dtype:
    if fp32:
        return torch.float32
    # CUDAなら基本 fp16（高速）。MPS/CPUはまず fp32 が無難（安定優先）
    return torch.float16 if device_name == "cuda" else torch.float32


def build_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def sanitize_filename(s: str) -> str:
    # ファイル名に使いにくい文字を軽く除去
    return "".join(c for c in s if c.isalnum() or c in ["-", "_"])


@dataclass
class CommonConfig:
    model: str
    sampler: str = "DPM++ 2M Karras"  # UI都合の名前。diffusers側はschedulerで扱うが本スクリプトでは簡略化。
    steps: int = 30
    cfg_scale: float = 6.5
    width: int = 768
    height: int = 512
    seed: int = -1
    batch_size: int = 1
    count: int = 1
    negative_prompt: str = ""

    # optional:
    torch_compile: bool = False
    enable_xformers: bool = False


@dataclass
class PresetConfig:
    name: str
    description: str
    output_dir: str
    prompt: str
    negative_prompt: Optional[str] = None


# -----------------------------
# Pipeline Loader
# -----------------------------


def load_pipeline(
    model_str: str,
    device_name: str,
    dtype: torch.dtype,
    enable_xformers: bool = False,
    torch_compile: bool = False,
) -> StableDiffusionPipeline:
    """
    model_str:
      - path to .safetensors/.ckpt (single file) OR
      - HF repo id (e.g. "runwayml/stable-diffusion-v1-5")
    """
    t0 = time.time()

    if is_probably_file_model(model_str):
        model_path = Path(model_str).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        # HF repo
        pipe = StableDiffusionPipeline.from_pretrained(
            model_str,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    if device_name == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()  # VRAM節約（速度を少し犠牲に）
        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[WARN] xformers enable failed: {e}")

        if torch_compile and hasattr(torch, "compile"):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}")
    else:
        pipe = pipe.to("cpu")

    dt = time.time() - t0
    print(f"🚀 Pipeline loadeds: {dt:.2f}s | model={model_str} | dtype={dtype} | device={device_name}")
    return pipe


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stable Diffusion background generator (config/preset based)")

    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--preset", required=True, help="Preset name in config.yaml (presets.<name>)")

    p.add_argument("--count", type=int, default=None, help="Number of images to generate (fallback to config image.common.count)")
    p.add_argument("--seed", type=int, default=None, help="Seed override. If omitted, uses config common.seed (-1=random).")

    p.add_argument("--out", default=None, help="Output dir override. If omitted, uses preset.output_dir")

    # overrides
    p.add_argument("--steps", type=int, default=None, help="Override steps")
    p.add_argument("--cfg", type=float, default=None, help="Override cfg_scale")
    p.add_argument("--width", type=int, default=None, help="Override width")
    p.add_argument("--height", type=int, default=None, help="Override height")

    p.add_argument("--prompt", default=None, help="Append/override prompt (see --prompt_mode)")
    p.add_argument("--negative", default=None, help="Append/override negative prompt (see --neg_mode)")
    p.add_argument("--prompt_mode", choices=["override", "append"], default="override")
    p.add_argument("--neg_mode", choices=["override", "append"], default="override")

    p.add_argument("--device", default="auto", choices=["auto","mps","cuda","cpu"], help="Device: auto|mps|cuda|cpu (auto prefers cuda->mps->cpu)")
    p.add_argument("--fp32", action="store_true", help="Force fp32 (useful if fp16 yields black images)")

    p.add_argument("--xformers", action="store_true", help="Enable xformers attention (if installed)")
    p.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for UNet (if available)")

    p.add_argument("--save_meta", action="store_true", help="Save meta json next to images")
    return p.parse_args()


def merge_prompts(
    base: str,
    extra: Optional[str],
    mode: str,
) -> str:
    if not extra:
        return base.strip()
    if mode == "override":
        return extra.strip()
    # append
    if base.strip() == "":
        return extra.strip()
    return (base.strip() + ", " + extra.strip()).strip()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    cfg = load_yaml(config_path)

    # 新しいconfig.yaml構造: image.common と image.presets
    if "image" not in cfg:
        raise KeyError("config.yaml must contain top-level key: image")
    if "common" not in cfg["image"]:
        raise KeyError("config.yaml must contain key: image.common")
    if "presets" not in cfg["image"] or not isinstance(cfg["image"]["presets"], dict):
        raise KeyError("config.yaml must contain key: image.presets")

    common_raw = cfg["image"]["common"]
    presets_raw = cfg["image"]["presets"]

    if args.preset not in presets_raw:
        raise KeyError(f"Preset not found: {args.preset} (available: {', '.join(presets_raw.keys())})")

    preset_raw = presets_raw[args.preset]

    common = CommonConfig(
        model=str(common_raw.get("model", "")),
        sampler=str(common_raw.get("sampler", "DPM++ 2M Karras")),
        steps=int(common_raw.get("steps", 30)),
        cfg_scale=float(common_raw.get("cfg_scale", 6.5)),
        width=int(common_raw.get("width", 768)),
        height=int(common_raw.get("height", 512)),
        seed=int(common_raw.get("seed", -1)),
        batch_size=int(common_raw.get("batch_size", 1)),
        count=int(common_raw.get("count", 1)),
        negative_prompt=str(common_raw.get("negative_prompt", "")),
        torch_compile=bool(common_raw.get("torch_compile", False)),
        enable_xformers=bool(common_raw.get("enable_xformers", False)),
    )

    preset = PresetConfig(
        name=args.preset,
        description=str(preset_raw.get("description", "")),
        output_dir=str(preset_raw.get("output_dir", f"outputs/{args.preset}")),
        prompt=str(preset_raw.get("prompt", "")),
        negative_prompt=preset_raw.get("negative_prompt", None),
    )

    # CLI overrides
    steps = args.steps if args.steps is not None else common.steps
    cfg_scale = args.cfg if args.cfg is not None else common.cfg_scale
    width = args.width if args.width is not None else common.width
    height = args.height if args.height is not None else common.height

    # prompt merge
    base_neg = preset.negative_prompt if preset.negative_prompt is not None else common.negative_prompt
    prompt = merge_prompts(preset.prompt, args.prompt, args.prompt_mode)
    negative_prompt = merge_prompts(base_neg, args.negative, args.neg_mode)

    # output dir
    out_dir = Path(args.out if args.out else preset.output_dir).expanduser()
    ensure_dir(out_dir)

    # device / dtype
    device, device_name = resolve_device(args.device)
    dtype = pick_torch_dtype(device_name, args.fp32)

    enable_xformers = args.xformers or common.enable_xformers
    torch_compile = args.torch_compile or common.torch_compile

    # seed logic
    # - args.seed があればそれを起点に count 分連番
    # - なければ common.seed を見る（-1ならランダム起点）
    if args.seed is not None:
        base_seed = int(args.seed)
    else:
        if common.seed == -1:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        else:
            base_seed = int(common.seed)

    print("====================================")
    print("🎨 Preset:", preset.name)
    print("📝 Description:", preset.description)
    print("📁 Out:", str(out_dir))
    print("🧠 Device:", device_name, "| dtype:", str(dtype))
    print("⚙️  steps/CFG/size:", steps, cfg_scale, f"{width}x{height}")
    count = args.count if args.count is not None else common.count
    print("🌱 seed:", base_seed, "| count:", count)
    print("------------------------------------")
    print("💬 Prompt:")
    print(prompt)
    print("------------------------------------")
    print("🚫 Negative Prompt:")
    print(negative_prompt)
    print("====================================")

    # load pipe
    pipe = load_pipeline(
        common.model,
        device_name=device_name,
        dtype=dtype,
        enable_xformers=enable_xformers,
        torch_compile=torch_compile,
    )

    # generation loop
    total_t0 = time.time()
    meta_records = []

    for i in range(count):
        seed = base_seed + i
        gen = build_generator(device=torch.device(device_name), seed=seed) if device_name == "cuda" else torch.Generator().manual_seed(seed)

        t0 = time.time()
        # NOTE: sampler/schedulerの厳密再現は環境差が出るため、ここでは簡略化して pipe デフォルトを使用
        # もし scheduler を指定したい場合は、後で使っているモデル/要件に合わせて差し替えできます。
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=gen,
        )
        img: Image.Image = result.images[0]
        dt = time.time() - t0

        fname = f"{sanitize_filename(preset.name)}_seed{seed:09d}_{now_stamp()}.png"
        out_path = out_dir / fname
        img.save(out_path)

        print(f"✅ {i+1:03d}/{count} Saved: {out_path.name} | {dt:.2f}s")

        meta = {
            "preset": preset.name,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "device": device_name,
            "dtype": str(dtype),
            "model": common.model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "filename": out_path.name,
            "seconds": round(dt, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        meta_records.append(meta)

        if args.save_meta:
            meta_path = out_path.with_suffix(".json")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    total_dt = time.time() - total_t0
    print("====================================")
    print(f"✅ Generation completed: {count} images in {total_dt:.2f}s  (avg {total_dt/max(count,1):.2f}s/img)")
    print("====================================")

    # summary meta
    if args.save_meta:
        summary_path = out_dir / f"_summary_{sanitize_filename(preset.name)}_{now_stamp()}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(meta_records, f, ensure_ascii=False, indent=2)
            print(f"📝 Meta summary saved: {summary_path.name}")


if __name__ == "__main__":
    main()
