# 🎨🎵 SD BGM Creator

Stable Diffusion + MusicGen による統合画像・BGM生成システム

## 📋 プロジェクト概要

Stable Diffusion v1.5 で日本風の画像を生成し、MusicGen でその世界観に合わせた背景音楽（BGM）を生成するシステムです。

画像とBGMを同一タイムスタンプのフォルダに出力し、メタデータで実行パラメータを記録します。

### 🎯 主な特徴

- ✅ **統合出力**: `outputs/{TIMESTAMP}/` に画像とBGMを同時保存
- ✅ **高品質BGM**: 複数セグメント生成 → シャッフル → クロスフェード → ループ
- ✅ **柔軟なプリセット**: 画像と音楽を独立して選択可能
- ✅ **処理時間表示**: 画像生成時間、BGM生成時間、合計時間を表示
- ✅ **ステレオ出力**: 最終BGMはYouTube対応のステレオ形式
- ✅ **実行ログ**: metadata.json で全パラメータを記録

## 📁 プロジェクト構造

```
sd_bgm_creator/
├── config.yaml                 # 統合設定ファイル
├── requirements.txt            # パッケージ依存関係
├── README.md                   # このファイル
│
├── image_gen/
│   ├── generate_bg.py          # 画像生成スクリプト
│   └── outputs/                # 画像出力フォルダ（未使用）
│
├── music_gen/
│   ├── music_gen.py            # BGM生成スクリプト
│   └── outputs/                # BGM出力フォルダ（未使用）
│
├── pipeline/
│   └── generate_all.py         # 統合生成オーケストレーター
│
└── outputs/                    # 統合出力ディレクトリ
    ├── {TIMESTAMP}/            # タイムスタンプベースのフォルダ
    │   ├── images/             # 生成画像
    │   ├── music/              # 生成BGM
    │   └── metadata.json       # 実行パラメータログ
    ├── _takes/                 # BGM中間ファイル（セグメント）
    └── _final/                 # BGM最終ファイル（バックアップ）
```

## 🔧 セットアップ

### 1. 環境構築

```bash
cd /Users/shu/code/sd_bgm_creator

# 仮想環境作成
python3 -m venv .venv

# 仮想環境有効化
source .venv/bin/activate

# パッケージインストール
pip install -r requirements.txt
```

### 2. 必要パッケージ

```
pyyaml
torch
torchaudio
transformers
soundfile
Pillow
diffusers
accelerate
safetensors
scipy
```

## ⚙️ config.yaml 構成

```yaml
# 画像生成設定
image:
  common:
    model: "runwayml/stable-diffusion-v1-5"
    sampler: "DPM++ 2M Karras"
    steps: 50
    cfg_scale: 5.8
    width: 768
    height: 512
    seed: -1
    negative_prompt: "..."

  presets:
    sakura_rain_street:
      description: "..."
      prompt: "..."
    sakura_wet_snowy:
      description: "..."
      prompt: "..."

# BGM生成設定
musicgen:
  model: "facebook/musicgen-large"
  duration: 45  # 1セグメント長（秒）

# 共通生成パラメータ（BGM用）
generation:
  num_tracks: 10                # セグメント数
  total_duration_sec: 3600      # 最終BGM長（秒）
  device: "auto"                # auto/cuda/mps/cpu
  crossfade_ms: 250             # クロスフェード長
  fade_ms: 30                    # フェード長
  output_dir: "outputs"

# 音楽プリセット
music:
  presets:
    sakura_ambient:
      description: "..."
      prompt: "..."
    sakura_calm:
      description: "..."
      prompt: "..."
    sakura_lofi:
      description: "..."
      prompt: "..."
```

## 🚀 使用方法

### プリセット一覧表示

```bash
cd /Users/shu/code/sd_bgm_creator
.venv/bin/python pipeline/generate_all.py --list-presets
```

出力：
```
📋 利用可能な画像プリセット:
============================================================
  • sakura_wet_snowy     - 桜×夜×濡れ路面＋雪/みぞれ粒（幻想寄り）
  • sakura_rain_street   - 雨の夜×桜×日本の街角（低彩度／桜ピンク残し／路面反射）

📋 利用可能な音楽プリセット:
============================================================
  • sakura_ambient       - lo-fi ambient music with soft piano and nature sounds
  • sakura_calm          - calm evening ambient with lo-fi vibes
  • sakura_lofi          - lo-fi hip-hop with cherry blossom theme
```

### 画像のみ生成

```bash
.venv/bin/python pipeline/generate_all.py \
  --image-only \
  --image-preset sakura_rain_street \
  --image-count 1
```

### BGMのみ生成

```bash
.venv/bin/python pipeline/generate_all.py \
  --music-only \
  --music-preset sakura_calm \
  --music-num-tracks 2 \
  --music-duration 120
```

### 画像 + BGM 統合生成

```bash
.venv/bin/python pipeline/generate_all.py \
  --image-preset sakura_rain_street \
  --image-count 1 \
  --music-preset sakura_lofi \
  --music-num-tracks 2 \
  --music-duration 120
```

### 詳細オプション

```bash
.venv/bin/python pipeline/generate_all.py \
  --image-preset sakura_wet_snowy \
  --image-count 3 \
  --image-seed 12345 \
  --music-preset sakura_calm \
  --music-num-tracks 5 \
  --music-duration 300 \
  --steps 60 \
  --cfg 6.0 \
  --output-base ./outputs
```

**CLI パラメータ:**
- `--config`: 設定ファイルパス（デフォルト: config.yaml）
- `--image-preset`: 画像プリセット名
- `--image-count`: 生成画像枚数（デフォルト: 1）
- `--image-seed`: 画像シード値
- `--music-preset`: 音楽プリセット名
- `--music-description`: 音楽説明文（プリセットの代わり）
- `--music-num-tracks`: BGMセグメント数
- `--music-duration`: BGM長（秒）
- `--steps`: 拡散ステップ数（画像）
- `--cfg`: CFGスケール（画像）
- `--output-base`: 出力フォルダ基本パス
- `--image-only`: 画像生成のみ実行
- `--music-only`: BGM生成のみ実行
- `--list-presets`: プリセット一覧表示

## 📊 出力構造

### 統合フォルダ例：`outputs/20260121_060450/`

```
20260121_060450/
├── images/
│   └── sakura_wet_snowy_seed1840812939_20260121_060630.png
├── music/
│   ├── bgm.wav                    # ステレオ形式のBGM（最終出力）
│   └── playlist_all_tracks.wav    # 全セグメント連結版
└── metadata.json                  # 実行パラメータ
```

### metadata.json 例

```json
{
  "timestamp": "20260121_060450",
  "datetime": "2026-01-21T06:04:50",
  "image": {
    "preset": "sakura_wet_snowy",
    "count": 1,
    "seed": null,
    "steps": null,
    "cfg": null,
    "success": true,
    "generation_time_sec": 97
  },
  "music": {
    "prompt": "lo-fi hip-hop style background music...",
    "duration": 120,
    "num_tracks": 2,
    "success": true,
    "generation_time_sec": 312
  },
  "total_time_sec": 409
}
```

## 🎵 BGM生成アルゴリズム

1. **マルチセグメント生成**: 指定数のセグメント（各45秒）を個別生成
2. **シャッフル**: セグメント順序をランダムに並べ替え
3. **クロスフェード**: 250msのオーバーラップで滑らかに結合
4. **フェード処理**: 各セグメントの開始・終了に30msのフェード
5. **ループ拡張**: クロスフェードで繋ぎながら目標時間まで拡張
6. **ステレオ変換**: モノラルをステレオ（両チャンネル同一）に変換
7. **出力**: YouTube対応のWAV形式で保存

**効果:**
- 複数セグメントのランダムな組み合わせで「無限ループ感」を軽減
- クロスフェードで滑らかな音の繋がり
- ステレオ出力でYouTube配信に対応

## 💡 実行例と処理時間

### 例1: 画像のみ生成（1枚）
```
⏱️  画像生成時間: 2分 3秒
⏱️  合計処理時間: 2分 3秒
```

### 例2: BGMのみ生成（2セグメント、120秒）
```
⏱️  BGM生成時間: 6分 7秒
⏱️  合計処理時間: 6分 7秒
```

### 例3: 統合生成（画像1枚 + BGM 2セグメント、120秒）
```
⏱️  画像生成時間: 2分 6秒
⏱️  BGM生成時間: 6分 2秒
⏱️  合計処理時間: 8分 8秒
```

## 🔍 トラブルシューティング

### CUDA/MPS デバイス自動選択
- macOS: `mps` (Metal Performance Shaders) 自動利用
- Linux/Windows: `cuda` (NVIDIA GPU) 利用可能なら使用
- デフォルト: `cpu` にフォールバック

### モデルの初回ダウンロード
- Stable Diffusion v1.5: ~4GB
- MusicGen Large: ~3.5GB
- 初回実行時のみ自動ダウンロード（HuggingFace)

### メモリ不足の場合
```bash
# セグメント数を減らす
--music-num-tracks 1

# 時間を短縮する
--music-duration 60

# ステップ数を減らす（画像品質が低下）
--steps 30
```

## 📝 設定のカスタマイズ

### 新しい画像プリセット追加

config.yaml に追加：
```yaml
image:
  presets:
    my_custom_preset:
      description: "カスタムプリセットの説明"
      prompt: "プリセットのプロンプト..."
```

### 新しい音楽プリセット追加

config.yaml に追加：
```yaml
music:
  presets:
    my_music_preset:
      description: "カスタム音楽プリセット"
      prompt: "音楽生成用プロンプト..."
```

## 🛠️ 開発ノート

### 実装日時
- 2026年1月20-21日

### 主要モジュール

#### `pipeline/generate_all.py`
- 画像・BGM生成のオーケストレーター
- タイムスタンプベースのフォルダ管理
- metadata.json の生成
- 処理時間の計測・表示

#### `image_gen/generate_bg.py`
- Stable Diffusion v1.5 による画像生成
- プリセットシステム
- デバイス自動選択（MPS/CUDA/CPU）

#### `music_gen/music_gen.py`
- MusicGen による複数セグメント生成
- シャッフル・クロスフェード・ループ処理
- ステレオ変換
- 音声ファイル出力

### 最新の改修（2026年1月21日）
1. ✅ config.yaml 構造統一（image.common/presets, generation, music.presets）
2. ✅ BGM ステレオ出力（YouTube対応）
3. ✅ 処理時間表示（画像、BGM、合計）
4. ✅ metadata.json に処理時間を記録

### 改修履歴（2026年4月〜6月）

#### BGMモデル移行: MusicGen → ACE-Step（2026年4月〜5月）
1. ✅ **MusicGen廃止・ACE-Step採用**: MusicGen (CC-BY-NC、商用不可) から ACE-Step v1.0 (Apache 2.0、商用可) に移行
2. ✅ **dual-backend 対応**: `music_gen/music_gen.py` を ACE-Step / MusicGen 両対応に書き換え
3. ✅ **ローカルvendor管理**: ACE-Step を `vendor/ace-step/` に配置（editable install）
   - PyPI の spacy==3.8.4 バグを回避するため `spacy>=3.8.7` にパッチ済み
4. ✅ **torchaudio 2.9.1 互換修正**: `vendor/ace-step/acestep/pipeline_ace_step.py` の `save_wav_file` を `soundfile` 使用に変更（torchcodec 不要）
5. ✅ **モデルキャッシュ**: `~/.cache/ace-step/checkpoints/` に自動ダウンロード（約8GB）

#### 画像プリセット追加（2026年4月）
追加プリセット（`image.presets` 以下）:
- `japan_rainy_convenience_store` — 雨夜のコンビニ×ネオン反射
- `nostalgic_japan_train_crossing` — ノスタルジックな踏切と田舎道
- `tokyo_lofi_window_view` — 東京の窓から眺める夜景×Lo-fi
- `kyoto_rain_lantern_alley` — 京都の雨×石畳×提灯
- `japanese_coastal_drive_lofi` — 海岸線ドライブ×夕暮れ
- `tokyo_rain_no_text` — 雨の東京（テキストなし、YouTube サムネ向け）

#### BGMプリセット追加（2026年5月）
追加プリセット（`music.presets` 以下）:
- `rainy_tokyo_cafe` — 雨の東京カフェ × Lo-fi Jazz（最高バズ期待値）
- `dark_academia_piano` — ダークアカデミア × 深夜図書館ピアノ（TikTok/YouTube急成長）
- `synthwave_night_coding` — シンセウェーブ × 深夜コーディング（プログラマー・ゲーマー層）
- `japanese_ambient_countryside` — 日本田舎 × 自然音 × Ghibli感（癒し系）

#### BGMプロンプト最適化（2026年6月）
- **全プロンプトをACE-Stepタグ形式に変換**: MusicGen向けの長文散文 → コンマ区切りタグ形式
  - 例: `lo-fi, jazz piano, upright bass, rain ambience, 74 bpm, Tokyo cafe, nostalgic, study music`
  - ACE-Stepはタグ形式の方が各属性を正確に反映し生成品質が向上する
- 対象: `generation.prompt`（デフォルト）および全6音楽プリセット

#### ACE-Step v1.5 へのアップグレード（準備中）
- v1.0 (score 28.5) → v1.5 (score 39.1、+37%)、License: MIT
- v1.5-XL は score 47.9（Suno v5 超え）
- アップグレード手順は `config.yaml` の `musicgen:` セクションのコメントを参照

## 📄 ライセンス

MIT License

## 🎓 参考モデル

- **Image Generation**: [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Audio Generation**: [ACE-Step v1.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)（Apache 2.0）
  - 旧: [MusicGen Large](https://huggingface.co/facebook/musicgen-large)（CC-BY-NC、商用不可のため廃止）
- **Audio Processing**: マルチセグメント生成 + シャッフル + クロスフェード + ループ処理

---

**Last Updated**: 2026年6月3日
