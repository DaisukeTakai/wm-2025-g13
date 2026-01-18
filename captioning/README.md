# Captioning
VLMを使用して、最終フレームからキャプションを生成します。

## ⚙️ 定数の意味

スクリプト内の以下の定数を環境に合わせて変更してください。

| 定数 | 意味 |
| --- | --- |
| `ROOT_DIR` | データセットのベースパス |
| `TASK` | タスクディレクトリ名（例: `pusht_noise`） |
| `TRAIN_VAL` | 処理対象（`train` / `val`） |
| `OVERWRITE_JSONL` | すでに `captions.jsonl` がある場合に上書きするかどうか |
| `MODEL_ID` | 使用するモデル名|
| `PROMPT` | VLMに与える指示文|

## 🛠 実行手順

1. **動画から最終フレームを抽出**  
`save_last_frames()` を実行すると、`last_frames/` ディレクトリに `.jpg` が保存されます。
2. **キャプション生成**  
`generate_captions()`を実行すると、画像ごとに推論を実行します。
3. **結果の確認**  
結果が `captions.jsonl` に保存されます。

## 📂 ディレクトリ構造
```
ROOT_DIR/TASK/TRAIN_VAL/  
├── obses/           # 入力動画 (.mp4)  
├── last_frames/     # 抽出された最終フレーム (.jpg)  
└── captions.jsonl   # 生成されたキャプション  
```

## 📝 その他
- `notebook.ipynb`は`extract_last_frames.ipynb`と`llama_captioning.ipynb`を1つのノートブックにまとめたものです。
---