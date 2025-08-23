# jTessBoxEditorクイックガイド - Granulate文字の.traineddata作成

## 状況説明

コマンドラインでの自動生成を試みましたが、画像形式の問題（黒背景に白文字）により、多くのページが「Empty page」として認識され、適切なトレーニングができませんでした。

そのため、**jTessBoxEditorのGUIを使用**することを推奨します。

## 必要なファイル（準備済み）

`training_data/tesseract/`ディレクトリに以下のファイルが準備されています：

- `gran.font.exp0.tif` - 学習用画像（520ページ、各文字20サンプル）
- `gran.font.exp0.box` - 文字位置情報
- `unicharset` - 文字セット（A-Z）
- `font_properties` - フォント属性

## jTessBoxEditorでの手順

### 1. jTessBoxEditorのダウンロード

```bash
# ダウンロードと起動
curl -L https://github.com/nguyenq/jTessBoxEditor/releases/download/v2.6.0/jTessBoxEditor-2.6.0.zip -o ~/Downloads/jTessBoxEditor.zip
cd ~/Downloads
unzip jTessBoxEditor.zip
cd jTessBoxEditor
java -Xms128m -Xmx1024m -jar jTessBoxEditor.jar &
```

### 2. BOXファイルの確認（推奨）

1. jTessBoxEditorで「Box Editor」タブを選択
2. 「Open」をクリックし、`/Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/gran.font.exp0.tif`を開く
3. 各ページの文字が正しく認識されているか確認
   - 問題：多くのページが空として表示される場合がある
   - 対処：画像が見える少数のページでも学習は可能

### 3. トレーニングの実行

1. 「Trainer」タブに切り替え
2. 以下を設定：
   - **Training Data**: `gran.font.exp0.tif`を選択
   - **Box Files**: 自動的に`gran.font.exp0.box`が選択される
   - **Language**: `gran`と入力
   - **Bootstrap Language**: `eng`を選択（英語ベース）
3. 「Train」ボタンをクリック
4. 処理が完了すると`gran.traineddata`が生成される

### 4. インストール

```bash
# Tessdataディレクトリにコピー
sudo cp ~/Downloads/jTessBoxEditor/gran.traineddata /opt/homebrew/share/tessdata/

# 確認
ls -la /opt/homebrew/share/tessdata/gran.traineddata
```

### 5. テスト

```bash
# OCRテスト（レガシーエンジンを使用）
tesseract test_image.png output -l gran --oem 0
```

## トラブルシューティング

### 画像が表示されない/Empty pageエラー

現在の画像は黒背景に白文字のため、Tesseractが認識しにくい状態です。以下の対処法があります：

1. **少数のページでも学習を実行** - 完全でなくても基本的な認識は可能
2. **画像を変換** - 白背景・黒文字に変換してから再度BOXファイルを生成
3. **新しい学習データを作成** - `training_data/extracted/`の元画像から作り直す

### 精度が低い場合

1. より多くの学習サンプルを追加
2. CNN model（`models/cnn_model_best.pth`）の統合を検討（95%精度）
3. Hash-based認識にフォールバック（28.5%精度）

## 次のステップ

1. `gran.traineddata`を作成後、OCRServiceに統合：

```python
# backend/application/services/ocr_service.pyに追加
import pytesseract

text = pytesseract.image_to_string(
    image, 
    lang='gran',
    config='--oem 0 --psm 8'  # レガシーエンジン、単一文字モード
)
```

2. 精度テストを実施し、必要に応じて学習データを調整

## 参考

- 詳細ガイド: `docs/jtessboxeditor-training-guide.md`
- zutomayo_OCR: https://github.com/geum-ztmy/zutomayo_OCR