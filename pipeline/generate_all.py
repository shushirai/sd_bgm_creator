#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_all.py
- 背景画像とBGMを統合的に生成
- タイムスタンプベースの統合フォルダに保存
- config.yaml から統合プリセットを読み込み
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def now_stamp() -> str:
    """現在時刻をYYYYMMDD_HHMMSSフォーマットで取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    """ディレクトリを再帰的に作成"""
    p.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> Dict[str, Any]:
    """config.yaml を読み込み"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_image_generation(
    preset: str,
    count: int = 1,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    """画像生成を実行"""
    print(f"\n{'='*60}")
    print(f"🖼️  画像生成開始: preset={preset}, count={count}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "image_gen/generate_bg.py",
        "--preset", preset,
        "--count", str(count),
    ]
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if steps is not None:
        cmd.extend(["--steps", str(steps)])
    if cfg is not None:
        cmd.extend(["--cfg", str(cfg)])
    if output_dir is not None:
        cmd.extend(["--out", str(output_dir)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_music_generation(
    description: str,
    duration: Optional[int] = None,
    output_dir: Optional[Path] = None,
    num_tracks: Optional[int] = None,
    music_preset: Optional[str] = None,
) -> bool:
    """BGM生成を実行"""
    print(f"\n{'='*60}")
    print(f"🎵 BGM生成開始")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "music_gen/music_gen.py",
    ]
    
    if music_preset:
        cmd.extend(["--music-preset", music_preset])
    else:
        cmd.extend(["--description", description])
    
    if duration is not None:
        cmd.extend(["--duration", str(duration)])
    if num_tracks is not None:
        cmd.extend(["--num-tracks", str(num_tracks)])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0



def main():
    parser = argparse.ArgumentParser(
        description="背景画像とBGMを統合生成"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルパス（デフォルト: config.yaml）"
    )
    
    parser.add_argument(
        "--image-preset",
        type=str,
        help="画像生成プリセット"
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=None,
        help="生成する画像枚数"
    )
    parser.add_argument(
        "--image-seed",
        type=int,
        help="画像生成のシード値"
    )
    
    parser.add_argument(
        "--music-preset",
        type=str,
        help="音楽生成プリセット"
    )
    parser.add_argument(
        "--music-description",
        type=str,
        help="音楽生成の説明文（プリセットの代わり）"
    )
    parser.add_argument(
        "--music-duration",
        type=int,
        help="BGM長（秒）"
    )
    parser.add_argument(
        "--music-num-tracks",
        type=int,
        help="BGM生成セグメント数"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        help="拡散ステップ数"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        help="CFGスケール"
    )
    
    parser.add_argument(
        "--output-base",
        type=str,
        default="outputs",
        help="統合出力フォルダの基本パス（デフォルト: outputs）"
    )
    
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="画像生成のみ実行"
    )
    parser.add_argument(
        "--music-only",
        action="store_true",
        help="BGM生成のみ実行"
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="利用可能なプリセット一覧を表示"
    )
    
    args = parser.parse_args()
    
    # 設定ファイルを読み込み
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    
    # プリセット一覧表示
    if args.list_presets:
        print("\n📋 利用可能な画像プリセット:")
        print("=" * 60)
        if "image" in config and "presets" in config["image"]:
            for preset_name, preset_cfg in config["image"]["presets"].items():
                desc = preset_cfg.get("description", "")
                print(f"  • {preset_name:20s} - {desc}")
        else:
            print("  (画像プリセットが定義されていません)")
        
        print("\n📋 利用可能な音楽プリセット:")
        print("=" * 60)
        if "music" in config and "presets" in config["music"]:
            for preset_name, preset_cfg in config["music"]["presets"].items():
                desc = preset_cfg.get("description", "")
                print(f"  • {preset_name:20s} - {desc}")
        else:
            print("  (音楽プリセットが定義されていません)")
        print()
        return
    
    # パラメータを解析
    image_preset = args.image_preset
    image_count = args.image_count
    image_seed = args.image_seed
    music_preset = args.music_preset
    music_description = args.music_description
    music_duration = args.music_duration
    music_num_tracks = args.music_num_tracks
    
    # config から image count のデフォルト値を取得（CLIで上書き可能）
    if image_count is None and "image" in config and "common" in config["image"]:
        image_count = config["image"]["common"].get("count", 1)
    if image_count is None:
        image_count = 1
    
    # 音楽プリセットが指定されている場合、そこから設定を取得
    if music_preset:
        if "music" not in config or "presets" not in config["music"]:
            print(f"❌ エラー: 音楽プリセットが定義されていません")
            return 1
        
        if music_preset not in config["music"]["presets"]:
            print(f"❌ エラー: 音楽プリセット '{music_preset}' が見つかりません")
            print(f"利用可能なプリセット: {', '.join(config['music']['presets'].keys())}")
            return 1
        
        preset_cfg = config["music"]["presets"][music_preset]
        music_description = preset_cfg.get("prompt")
    
    # バリデーション
    if not image_preset and not music_description:
        print("❌ エラー: 以下のいずれかを指定してください:")
        print("  • --image-preset <プリセット名>")
        print("  • --music-preset <プリセット名> または --music-description <説明文>")
        print("\nプリセット一覧を表示: python generate_all.py --list-presets")
        return 1
    
    # 音楽プリセット名から説明文を取得
    if music_description and "music" in config:
        music_presets = config.get("music", {}).get("presets", {})
        if music_description in music_presets:
            # プリセット名として指定されている場合
            music_preset_cfg = music_presets[music_description]
            actual_description = music_preset_cfg.get("description", music_description)
            actual_duration = music_preset_cfg.get("total_duration_sec", music_duration)
            music_description_display = actual_description
            if not music_duration:
                music_duration = actual_duration
        else:
            music_description_display = music_description
    else:
        music_description_display = music_description
    
    # タイムスタンプベースの統合フォルダを作成
    timestamp = now_stamp()
    base_output = Path(args.output_base).expanduser()
    integrated_folder = base_output / timestamp
    images_folder = integrated_folder / "images"
    music_folder = integrated_folder / "music"
    
    ensure_dir(images_folder)
    ensure_dir(music_folder)
    
    print(f"\n{'='*60}")
    print(f"📁 統合出力フォルダ: {integrated_folder}")
    print(f"{'='*60}\n")
    
    # メタデータ辞書
    metadata: Dict[str, Any] = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "image": {},
        "music": {},
    }
    
    image_success = True
    music_success = True
    
    # 画像生成
    if image_preset and not args.music_only:
        image_success = run_image_generation(
            preset=image_preset,
            count=image_count,
            seed=image_seed,
            steps=args.steps,
            cfg=args.cfg,
            output_dir=images_folder,
        )
        
        metadata["image"] = {
            "preset": image_preset,
            "count": image_count,
            "seed": image_seed,
            "steps": args.steps,
            "cfg": args.cfg,
            "success": image_success,
        }
    
    # BGM生成
    if music_description and not args.image_only:
        music_success = run_music_generation(
            description=music_description,
            duration=music_duration,
            num_tracks=music_num_tracks,
            output_dir=music_folder,
        )
        
        metadata["music"] = {
            "prompt": music_description,
            "duration": music_duration,
            "num_tracks": music_num_tracks,
            "success": music_success,
        }
    
    # メタデータを保存
    metadata_path = integrated_folder / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 結果表示
    print(f"\n{'='*60}")
    print("📊 生成結果")
    print(f"{'='*60}")
    if image_preset and not args.music_only:
        print(f"画像生成: {'✅ 成功' if image_success else '❌ 失敗'}")
    if music_description and not args.image_only:
        print(f"BGM生成: {'✅ 成功' if music_success else '❌ 失敗'}")
    print(f"\n📁 統合フォルダ: {integrated_folder}")
    print(f"📝 メタデータ: {metadata_path.name}")
    print()
    
    return 0 if (image_success and music_success) else 1


if __name__ == "__main__":
    sys.exit(main())
