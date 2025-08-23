# jTessBoxEditorを使用したGranulate文字の学習ガイド

## 概要

このガイドでは、jTessBoxEditorを使用してGranulate文字（仮面ライダーガヴの架空文字）のTesseractカスタムモデル（.traineddata）を作成する方法を説明します。

## 前提条件

- Java Runtime Environment (JRE) 8以上
- Tesseract OCR 4.0以上（training toolsを含む）
- 既存の学習データ（`training_data/tesseract/`ディレクトリ内）

## 既存の学習データ

プロジェクトには以下の学習用ファイルが既に準備されています：

- **gran.font.exp0.tif** - 学習用画像（マルチページTIFF形式、各文字A-Zが20サンプルずつ）
- **gran.font.exp0.box** - 文字位置情報（各ページの文字と座標情報）
- **unicharset** - 文字セット定義（A-Zの26文字）
- **font_properties** - フォント属性情報

## インストールと環境設定

### 1. jTessBoxEditorのダウンロードと起動

```bash
# jTessBoxEditorをダウンロード
cd ~/Downloads
curl -L https://github.com/nguyenq/jTessBoxEditor/releases/download/v2.6.0/jTessBoxEditor-2.6.0.zip -o jTessBoxEditor.zip

# 解凍
unzip jTessBoxEditor.zip

# ディレクトリに移動
cd jTessBoxEditor

# 起動（メモリを十分に確保）
java -Xms128m -Xmx1024m -jar jTessBoxEditor.jar &
```

### 2. 環境変数の設定

```bash
# macOS (Homebrew)の場合
export TESSDATA_PREFIX="/usr/local/share/tessdata"

# または、Tesseractのインストール場所を確認
brew list tesseract | grep tessdata

# Linux (apt)の場合
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"
```

### 3. 作業ディレクトリへの移動

```bash
cd /Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract
```

## GUI（jTessBoxEditor）での学習手順

### 1. BOXファイルの確認・修正

1. jTessBoxEditorを起動
2. メニューから「Box Editor」タブを選択
3. 「Open」をクリックし、`gran.font.exp0.tif`を選択
4. 各ページの文字と座標が正しいか確認
5. 必要に応じて座標を修正（ドラッグまたは数値入力）
6. 修正後は「Save」で保存

### 2. 学習の実行

1. 「Trainer」タブに切り替え
2. 以下の設定を行う：
   - **Type**: LSTM（推奨）またはLegacy
   - **Language**: gran
   - **Bootstrap Language**: eng（英語をベースに学習）
   - **Training Data**: `gran.font.exp0.tif`を選択
   - **Character Set**: A-Z（既に設定済み）

3. 「Train」ボタンをクリックして学習開始
4. 学習完了後、`gran.traineddata`が生成される

## コマンドラインでの学習手順（代替方法）

jTessBoxEditorが使用できない場合は、以下のコマンドラインで学習を実行できます：

```bash
# 1. BOXファイルの生成（既存のものを使用する場合はスキップ）
tesseract gran.font.exp0.tif gran.font.exp0 batch.nochop makebox

# 2. トレーニングファイルの生成
tesseract gran.font.exp0.tif gran.font.exp0 box.train

# 3. 文字セットの抽出
unicharset_extractor gran.font.exp0.box

# 4. フォント特徴の抽出
shapeclustering -F font_properties -U unicharset -O gran.unicharset gran.font.exp0.tr
mftraining -F font_properties -U unicharset -O gran.unicharset gran.font.exp0.tr
cntraining gran.font.exp0.tr

# 5. traineddataファイルの作成
combine_tessdata gran.

# 生成されるファイル：
# - gran.traineddata
# - gran.unicharset
# - gran.shapetable
# - gran.inttemp
# - gran.pffmtable
# - gran.normproto
```

## 学習済みモデルのインストール

```bash
# Tessdataディレクトリにコピー
sudo cp gran.traineddata $TESSDATA_PREFIX/

# 確認
ls -la $TESSDATA_PREFIX/gran.traineddata

# 権限の設定
sudo chmod 644 $TESSDATA_PREFIX/gran.traineddata
```

## OCRサービスでの使用方法

### Pythonでの実装例

```python
import pytesseract
import cv2
from PIL import Image

def process_with_custom_tesseract(image_path):
    """カスタムTesseractモデルを使用してOCR処理"""
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # カスタム言語（gran）を指定してOCR実行
    # PSM 8: 単一文字として扱う
    # PSM 6: 均一なテキストブロックとして扱う
    text = pytesseract.image_to_string(
        gray, 
        lang='gran',
        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    )
    
    return text.strip()
```

### 既存のOCRServiceへの統合

```python
# backend/application/services/ocr_service.py に追加

def process_with_tesseract(self, char_image: np.ndarray) -> Optional[str]:
    """Tesseractカスタムモデルでの文字認識"""
    try:
        # Tesseractで認識
        text = pytesseract.image_to_string(
            char_image,
            lang='gran',
            config='--psm 8'
        )
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"Tesseract error: {e}")
        return None

def process_image(self, image_bytes: bytes) -> OCRResult:
    # ... 既存のコード ...
    
    # 各文字を認識
    for i, (x, y, w, h) in enumerate(char_regions):
        char_image = preprocessed[y:y+h, x:x+w]
        
        # まずTesseractで試す
        recognized_char = self.process_with_tesseract(char_image)
        
        # Tesseractで認識できない場合はHash-basedにフォールバック
        if not recognized_char:
            recognized_char = self.alphabet.compare_image_to_mapping(char_image)
        
        # ... 残りのコード ...
```

## トラブルシューティング

### 1. jTessBoxEditorが起動しない

```bash
# Javaバージョンを確認
java -version

# Java 8以上が必要。インストールされていない場合：
# macOS
brew install openjdk@11

# Linux
sudo apt-get install openjdk-11-jre
```

### 2. Tesseractコマンドが見つからない

```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr libtesseract-dev
```

### 3. 学習がうまくいかない

- BOXファイルの座標が正しいか確認
- 画像が白背景に黒文字になっているか確認
- unicharsetに全ての文字が含まれているか確認

### 4. 認識精度が低い

- より多くの学習サンプルを追加（現在は各文字20サンプル）
- 画像の前処理を調整（二値化、ノイズ除去）
- PSM（Page Segmentation Mode）を調整

## 期待される精度

- **Hash-based認識**: 約28.5%
- **Tesseractカスタムモデル**: 約60-80%（学習データの質による）
- **CNN model**: 約95%（別途実装が必要）

## 参考リンク

- [jTessBoxEditor GitHub](https://github.com/nguyenq/jTessBoxEditor)
- [Tesseract Training Documentation](https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html)
- [zutomayo_OCR リポジトリ](https://github.com/geum-ztmy/zutomayo_OCR)

## 次のステップ

1. jTessBoxEditorで`gran.traineddata`を作成
2. OCRServiceにTesseract統合を実装
3. 精度をテストし、必要に応じて学習データを調整
4. より高精度が必要な場合は、CNN modelの統合を検討