# テストディレクトリ構成

## ディレクトリ構造

```
tests/
├── integration/     # 統合テスト
│   ├── test_ocr_api.py         # OCR APIの統合テスト
│   └── test_real_image.py      # 実画像でのOCR認識テスト
│
└── debug/          # デバッグ・分析用スクリプト
    ├── analyze_pleasure_image.py    # PLEASURE画像の詳細分析
    ├── analyze_test_image_cnn.py    # CNN認識の分析
    ├── debug_test_image.py          # テスト画像のデバッグ
    ├── test_hash_mapping.py         # ハッシュベース認識のテスト
    ├── test_improved_parameters.py  # パラメータ最適化テスト
    ├── test_integrated_ocr.py       # 統合OCRシステムテスト
    ├── test_similarity_mapping.py   # 類似度ベース認識のテスト
    └── output/                      # デバッグ出力画像
```

## 実行方法

### 統合テスト

```bash
# プロジェクトルートから実行
cd /path/to/granulate-char-OCR

# 実画像でのOCR認識テスト
python tests/integration/test_real_image.py

# API統合テスト（APIサーバー起動が必要）
uv run uvicorn backend.main:app --reload
python tests/integration/test_ocr_api.py
```

### デバッグスクリプト

```bash
# 各種認識手法のテスト
python tests/debug/test_hash_mapping.py
python tests/debug/test_similarity_mapping.py

# 画像分析
python tests/debug/analyze_pleasure_image.py
python tests/debug/analyze_test_image_cnn.py
```

## テスト結果

### 現在の認識精度
- **初期状態**: 12.5% (1/8文字正解)
- **改善後**: 62.5% (5/8文字正解)
- **処理時間**: 約200ms

### 認識手法の精度
- **CNN**: 95% (訓練データ) / 実画像では可変
- **Tesseract**: 61.5% (カスタムモデル)
- **Hash-based**: 28.5%

## 注意事項

- テストスクリプトはプロジェクトルートからの相対パスを想定
- デバッグ画像は `tests/debug/output/` に保存される
- API統合テストは事前にサーバー起動が必要