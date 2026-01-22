# Metaworld VLM annotation

目的: `datasets/Metaworld/data/` の Parquet（HuggingFace Datasets形式）を読み込み、
各 roll-out（= parquet の 1 row）に対して Open VLM で **ゴール/タスク推定**を行い、
`jsonl` 形式のアノテーションデータセットを作る。

出力は以下を同一 episode に対して併存させる:
- **closed-set**: 候補ラベル（例: Metaworld task名）から 1つ選ぶ
- **open-set**: 短い命令文（英語・imperative・1行）

---

## 0. venv (uv)

このリポジトリ直下に uv venv を作る（例）:

```bash
cd /workspace/wm_assignments

uv venv --python python3 .venv-vlm
source .venv-vlm/bin/activate

# 推論に必要な最小セット
uv pip install -U datasets pyarrow pillow imageio[ffmpeg] torch torchvision transformers accelerate

# Qwen3-VL utils（複数画像/動画入力の取り回し）
uv pip install -U qwen-vl-utils

# 速度/VRAM（A100推奨）
# uv pip install -U flash-attn --no-build-isolation
```

注意:
- `Qwen/Qwen3-VL-30B-A3B-Thinking` は transformers >= 4.57 系が前提。
- FP8 checkpoint は Transformers で直接ロードできない（vLLM/SGLang推奨）。

---

## 1. 実行

```bash
source /workspace/wm_assignments/.venv-vlm/bin/activate

python3 -m vlm.annotate_metaworld \
  --data-dir /workspace/wm_assignments/datasets/Metaworld/data \
  --out /workspace/wm_assignments/datasets/Metaworld/annotations_qwen3vl30a3b_thinking.jsonl \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --frames 0,24,49,74,98 \
  --max-new-tokens 96 \
  --batch-size 1
```

再開:
- `--resume` を付けると、既存 jsonl の `episode_idx` を読み取りスキップする。

---

## 2. 出力スキーマ（jsonl）

各行が1 episode。

```json
{
  "episode_idx": 123,
  "task_gt": "mw-reach-v3",
  "frame_indices": [0, 24, 49, 74, 98],
  "closed": {
    "label": "mw-reach-v3",
    "confidence": 0.82,
    "top_k": [{"label": "...", "confidence": 0.82}]
  },
  "open": {
    "instruction": "Reach the target.",
    "confidence": 0.63
  },
  "model": {
    "id": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "prompt_version": "v1"
  }
}
```
