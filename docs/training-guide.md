# グラニュート文字OCR学習データ準備ガイド

このガイドでは、仮面ライダーガヴのグラニュート文字をOCRで認識するための学習データの準備方法を説明します。

## 目次

1. [概要](#概要)
2. [必要な環境](#必要な環境)
3. [データ収集](#データ収集)
4. [画像の前処理](#画像の前処理)
5. [Tesseract用データ形式への変換](#tesseract用データ形式への変換)
6. [学習の実行](#学習の実行)
7. [プロジェクトへの統合](#プロジェクトへの統合)
8. [トラブルシューティング](#トラブルシューティング)

## 概要

グラニュート文字は架空の文字体系のため、既存のOCRエンジンでは認識できません。そのため、カスタム学習データを作成し、Tesseractを訓練する必要があります。

### 対象文字
- アルファベット: A-Z（26文字）
- 数字: 0-9（10文字）
- 合計: 36文字

## 必要な環境

### ソフトウェア要件
- Python 3.11以上
- Tesseract 4.0以上
- OpenCV 4.0以上
- Git

### インストール
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr libtesseract-dev
sudo apt-get install tesseract-ocr-jpn  # 日本語データ（ベース用）
sudo apt-get install python3-opencv

# macOS
brew install tesseract
brew install tesseract-lang

# Python依存関係
pip install opencv-python pytesseract numpy pillow
```

## データ収集

### 1. ディレクトリ構造の作成

```bash
mkdir -p training_data/{raw,processed,ground_truth,scripts}
mkdir -p training_data/raw/{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}
mkdir -p training_data/raw/{0,1,2,3,4,5,6,7,8,9}
```

### 2. 画像収集のガイドライン

#### 必要なデータ量
- 各文字につき最低100枚の画像
- 理想的には各文字200-300枚

#### 画像の要件
- 解像度: 300DPI以上
- 形式: PNG推奨（JPEGも可）
- サイズ: 最小32x32ピクセル
- 背景: 多様な背景（白、黒、グレー、模様付き）

#### バリエーション
- フォントサイズ: 小（12pt）〜大（72pt）
- 角度: -15°〜+15°の回転
- ノイズ: 軽度のぼかし、ノイズ追加
- 照明: 明暗の変化

### 3. 画像収集スクリプト

```python
# training_data/scripts/generate_synthetic_data.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path

class SyntheticDataGenerator:
    def __init__(self, output_dir="training_data/raw"):
        self.output_dir = Path(output_dir)
        self.backgrounds = self._load_backgrounds()
        
    def generate_character_image(self, char, label, count=100):
        """指定文字の合成画像を生成"""
        char_dir = self.output_dir / label
        char_dir.mkdir(exist_ok=True)
        
        for i in range(count):
            # ランダムパラメータ
            font_size = random.randint(24, 72)
            rotation = random.uniform(-15, 15)
            noise_level = random.uniform(0, 0.1)
            
            # 画像生成
            img = self._create_character_image(
                char, font_size, rotation, noise_level
            )
            
            # 保存
            filename = f"{label}_{i:04d}.png"
            cv2.imwrite(str(char_dir / filename), img)
            
    def _create_character_image(self, char, font_size, rotation, noise_level):
        """文字画像を生成"""
        # 基本画像サイズ
        img_size = (128, 128)
        
        # PIL画像作成
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # フォント設定（グラニュート文字フォントを想定）
        try:
            font = ImageFont.truetype("granulate.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 文字を中央に配置
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # 文字を描画
        draw.text((x, y), char, fill='black', font=font)
        
        # NumPy配列に変換
        img_array = np.array(img)
        
        # 回転
        if rotation != 0:
            center = (img_size[0] // 2, img_size[1] // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img_array = cv2.warpAffine(img_array, M, img_size)
        
        # ノイズ追加
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return img_array

# 使用例
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # グラニュート文字マッピング
    granulate_mapping = {
        'ᐁ': 'A', 'ᐂ': 'B', 'ᐃ': 'C', # ... 続く
    }
    
    for granulate_char, latin_label in granulate_mapping.items():
        generator.generate_character_image(granulate_char, latin_label, count=200)
```

## 画像の前処理

### 1. 前処理スクリプト

```python
# training_data/scripts/preprocess_images.py
import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self, input_dir="training_data/raw", output_dir="training_data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
    def preprocess_all(self):
        """全画像を前処理"""
        image_files = list(self.input_dir.glob("**/*.png"))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(
                executor.map(self._process_single_image, image_files),
                total=len(image_files),
                desc="Processing images"
            ))
    
    def _process_single_image(self, image_path):
        """単一画像の前処理"""
        # 出力パスの生成
        relative_path = image_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 画像読み込み
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 二値化（適応的閾値）
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # モルフォロジー変換（文字を太くする）
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # サイズ正規化（Tesseract推奨サイズ）
        resized = cv2.resize(morphed, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # 保存
        cv2.imwrite(str(output_path), resized)
        
        # Ground truthファイルも生成
        self._create_ground_truth(output_path, relative_path.parent.name)
    
    def _create_ground_truth(self, image_path, label):
        """Ground truthテキストファイルを生成"""
        gt_path = image_path.with_suffix('.gt.txt')
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(label)

# 使用例
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.preprocess_all()
```

### 2. データ検証スクリプト

```python
# training_data/scripts/validate_data.py
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def validate_dataset(data_dir="training_data/processed"):
    """データセットの統計情報を表示"""
    data_dir = Path(data_dir)
    
    # 文字ごとの画像数をカウント
    char_counts = Counter()
    image_sizes = []
    
    for char_dir in data_dir.iterdir():
        if char_dir.is_dir():
            images = list(char_dir.glob("*.png"))
            char_counts[char_dir.name] = len(images)
            
            # 画像サイズチェック
            for img_path in images[:5]:  # サンプルチェック
                img = cv2.imread(str(img_path))
                if img is not None:
                    image_sizes.append(img.shape[:2])
    
    # 統計表示
    print("データセット統計:")
    print(f"総文字数: {len(char_counts)}")
    print(f"総画像数: {sum(char_counts.values())}")
    print(f"平均画像数/文字: {sum(char_counts.values()) / len(char_counts):.1f}")
    print(f"最小画像数: {min(char_counts.values())} ({min(char_counts, key=char_counts.get)})")
    print(f"最大画像数: {max(char_counts.values())} ({max(char_counts, key=char_counts.get)})")
    
    # グラフ表示
    plt.figure(figsize=(12, 6))
    plt.bar(char_counts.keys(), char_counts.values())
    plt.xlabel('Character')
    plt.ylabel('Number of Images')
    plt.title('Training Data Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('training_data/data_distribution.png')
    plt.show()

if __name__ == "__main__":
    validate_dataset()
```

## Tesseract用データ形式への変換

### 1. BOXファイル生成

```python
# training_data/scripts/create_tesseract_files.py
import cv2
from pathlib import Path
import subprocess

class TesseractDataPreparer:
    def __init__(self, processed_dir="training_data/processed"):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path("training_data/tesseract_format")
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_all_data(self):
        """全データをTesseract形式に変換"""
        # 1. TIFFファイルとBOXファイルを生成
        self._create_tiff_and_box_files()
        
        # 2. フォントプロパティファイルを作成
        self._create_font_properties()
        
        # 3. ユニコード文字セットファイルを作成
        self._create_unicharset()
        
        # 4. 学習用リストファイルを作成
        self._create_training_list()
    
    def _create_tiff_and_box_files(self):
        """TIFF画像とBOXファイルのペアを作成"""
        page_number = 0
        
        for char_dir in self.processed_dir.iterdir():
            if not char_dir.is_dir():
                continue
                
            char_label = char_dir.name
            
            for img_path in char_dir.glob("*.png"):
                # TIFF形式に変換
                img = cv2.imread(str(img_path))
                tiff_name = f"granulate.{page_number:04d}.tif"
                tiff_path = self.output_dir / tiff_name
                cv2.imwrite(str(tiff_path), img)
                
                # BOXファイル作成
                h, w = img.shape[:2]
                box_content = f"{char_label} 0 0 {w} {h} {page_number}\n"
                box_path = self.output_dir / f"granulate.{page_number:04d}.box"
                with open(box_path, 'w', encoding='utf-8') as f:
                    f.write(box_content)
                
                page_number += 1
    
    def _create_font_properties(self):
        """フォントプロパティファイルを作成"""
        font_properties_path = self.output_dir / "font_properties"
        with open(font_properties_path, 'w') as f:
            f.write("granulate 0 0 0 0 0\n")
    
    def _create_unicharset(self):
        """文字セットファイルを作成"""
        # 文字セットを収集
        charset = set()
        for char_dir in self.processed_dir.iterdir():
            if char_dir.is_dir():
                charset.add(char_dir.name)
        
        # ファイルに書き込み
        unicharset_path = self.output_dir / "granulate.unicharset"
        with open(unicharset_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(charset)))
    
    def _create_training_list(self):
        """学習用ファイルリストを作成"""
        list_path = self.output_dir / "granulate.training_files.txt"
        with open(list_path, 'w') as f:
            for tif_file in sorted(self.output_dir.glob("*.tif")):
                base_name = tif_file.stem
                f.write(f"{self.output_dir}/{base_name}\n")

# 使用例
if __name__ == "__main__":
    preparer = TesseractDataPreparer()
    preparer.prepare_all_data()
```

### 2. Tesseract学習スクリプト

```bash
#!/bin/bash
# training_data/scripts/train_tesseract.sh

set -e

# 設定
LANG_NAME="granulate"
TESSDATA_DIR="/usr/share/tesseract-ocr/4.00/tessdata"
TRAINING_DIR="training_data/tesseract_format"
OUTPUT_DIR="training_data/output"

# 出力ディレクトリ作成
mkdir -p $OUTPUT_DIR

echo "1. Training data preparation..."
cd $TRAINING_DIR

# 各TIFFファイルに対してBOXファイルを検証
for f in *.tif; do
    base="${f%.tif}"
    if [ ! -f "${base}.box" ]; then
        echo "Error: Missing box file for $f"
        exit 1
    fi
done

echo "2. Extracting unicharset..."
unicharset_extractor *.box
mv unicharset ${LANG_NAME}.unicharset

echo "3. Creating font properties..."
echo "${LANG_NAME} 0 0 0 0 0" > font_properties

echo "4. Clustering..."
shapeclustering -F font_properties -U ${LANG_NAME}.unicharset *.tr

echo "5. MFTraining..."
mftraining -F font_properties -U ${LANG_NAME}.unicharset -O ${LANG_NAME}.unicharset *.tr

echo "6. CNTraining..."
cntraining *.tr

echo "7. Renaming files..."
mv inttemp ${LANG_NAME}.inttemp
mv normproto ${LANG_NAME}.normproto
mv pffmtable ${LANG_NAME}.pffmtable
mv shapetable ${LANG_NAME}.shapetable

echo "8. Combining trained data..."
combine_tessdata ${LANG_NAME}.

echo "9. Moving to output directory..."
mv ${LANG_NAME}.traineddata $OUTPUT_DIR/

echo "Training completed! Output: $OUTPUT_DIR/${LANG_NAME}.traineddata"
```

## 学習の実行

### 1. 学習環境のセットアップ

```bash
# training_data/scripts/setup_training_env.sh
#!/bin/bash

# Tesseract学習ツールのインストール
git clone https://github.com/tesseract-ocr/tesstrain.git
cd tesstrain

# 必要な依存関係をインストール
make tesseract-langdata

# 既存の日本語モデルをベースとして使用
cp /usr/share/tesseract-ocr/4.00/tessdata/jpn.traineddata ./
```

### 2. Fine-tuning実行

```makefile
# training_data/Makefile
# Tesseract Fine-tuning用Makefile

# 基本設定
MODEL_NAME = granulate
START_MODEL = jpn
TESSDATA = /usr/share/tesseract-ocr/4.00/tessdata

# ディレクトリ
GROUND_TRUTH_DIR = ground_truth
OUTPUT_DIR = output
TRAINING_DIR = tesseract_format

# 学習パラメータ
EPOCHS = 100
LEARNING_RATE = 0.0001
NET_SPEC = [1,36,0,1 Ct3,3,16 Mp3,3 Ct3,3,16 Mp3,3 Ct3,3,32 Mp3,3 Ct3,3,64 Fc128 Fc96 Fc$(shell cat $(GROUND_TRUTH_DIR)/unicharset | wc -l)]

# ターゲット
.PHONY: all clean training evaluate

all: training

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -f $(TRAINING_DIR)/*.lstm
	rm -f $(TRAINING_DIR)/*.checkpoint

training:
	@echo "Starting Tesseract fine-tuning..."
	@mkdir -p $(OUTPUT_DIR)
	
	# LSTMトレーニング
	lstmtraining \
		--model_output $(OUTPUT_DIR)/$(MODEL_NAME) \
		--continue_from $(TESSDATA)/$(START_MODEL).lstm \
		--traineddata $(TESSDATA)/$(START_MODEL).traineddata \
		--train_listfile $(TRAINING_DIR)/$(MODEL_NAME).training_files.txt \
		--max_iterations $(EPOCHS) \
		--learning_rate $(LEARNING_RATE) \
		--net_spec "$(NET_SPEC)"
	
	# traineddataファイルの生成
	lstmtraining \
		--stop_training \
		--continue_from $(OUTPUT_DIR)/$(MODEL_NAME)_checkpoint \
		--traineddata $(TESSDATA)/$(START_MODEL).traineddata \
		--model_output $(OUTPUT_DIR)/$(MODEL_NAME).traineddata

evaluate:
	@echo "Evaluating model performance..."
	# テストセットでの評価
	lstmeval \
		--model $(OUTPUT_DIR)/$(MODEL_NAME).traineddata \
		--eval_listfile $(TRAINING_DIR)/$(MODEL_NAME).eval_files.txt
```

### 3. 学習実行コマンド

```bash
# 完全な学習フロー
cd training_data

# 1. 画像の前処理
python scripts/preprocess_images.py

# 2. Tesseractフォーマットへの変換
python scripts/create_tesseract_files.py

# 3. 学習の実行
make -f Makefile training

# 4. モデルの評価
make -f Makefile evaluate

# 5. インストール
sudo cp output/granulate.traineddata $TESSDATA_DIR/
```

## プロジェクトへの統合

### 1. バックエンドでの使用

```python
# backend/infrastructure/ocr/tesseract_with_custom_model.py
import pytesseract
import cv2
from pathlib import Path

class CustomTesseractOCR:
    def __init__(self):
        # カスタムモデルの存在確認
        self.model_path = Path("/usr/share/tesseract-ocr/4.00/tessdata/granulate.traineddata")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Custom model not found: {self.model_path}")
        
        # Tesseract設定
        self.custom_config = r'--oem 3 --psm 6 -l granulate'
    
    def recognize_text(self, image):
        """画像からグラニュート文字を認識"""
        # 画像の前処理
        processed = self._preprocess_image(image)
        
        # OCR実行
        try:
            text = pytesseract.image_to_string(
                processed,
                config=self.custom_config
            )
            
            # 詳細な結果も取得
            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config=self.custom_config
            )
            
            return self._parse_results(text, data)
            
        except Exception as e:
            raise OCRError(f"Recognition failed: {str(e)}")
    
    def _preprocess_image(self, image):
        """認識精度向上のための前処理"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # ノイズ除去
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 二値化
        _, binary = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    def _parse_results(self, text, data):
        """OCR結果をパース"""
        results = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # 信頼度が0より大きい
                results.append({
                    'text': data['text'][i],
                    'confidence': float(data['conf'][i]) / 100.0,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return {
            'full_text': text.strip(),
            'characters': results
        }
```

### 2. フロントエンドでの使用（Tesseract.js）

```javascript
// front/app/services/ocr/tesseract-custom.ts
import Tesseract from 'tesseract.js';

export class CustomTesseractService {
  private worker: Tesseract.Worker | null = null;
  
  async initialize() {
    // カスタムモデルを使用してワーカーを初期化
    this.worker = await Tesseract.createWorker({
      logger: m => console.log(m),
      langPath: '/models', // カスタムモデルのパス
      gzip: false
    });
    
    // カスタム言語データをロード
    await this.worker.loadLanguage('granulate');
    await this.worker.initialize('granulate');
    
    // OCRパラメータ設定
    await this.worker.setParameters({
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
    });
  }
  
  async recognizeImage(imageData: ImageData): Promise<RecognitionResult> {
    if (!this.worker) {
      throw new Error('Worker not initialized');
    }
    
    const result = await this.worker.recognize(imageData);
    
    return {
      text: result.data.text,
      confidence: result.data.confidence,
      characters: result.data.symbols.map(symbol => ({
        text: symbol.text,
        confidence: symbol.confidence,
        bbox: symbol.bbox
      }))
    };
  }
  
  async terminate() {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }
}
```

### 3. モデル配布設定

```javascript
// front/vite.config.ts の追加設定
export default defineConfig({
  // ... 既存の設定
  
  plugins: [
    // ... 既存のプラグイン
    
    // カスタムモデルをpublicディレクトリにコピー
    {
      name: 'copy-tesseract-model',
      buildStart() {
        const source = 'training_data/output/granulate.traineddata';
        const destination = 'public/models/granulate.traineddata';
        
        if (fs.existsSync(source)) {
          fs.copyFileSync(source, destination);
          console.log('Tesseract model copied successfully');
        }
      }
    }
  ]
});
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. 認識精度が低い
```bash
# 解決策: より多くの学習データを追加
python scripts/generate_synthetic_data.py --count=500

# 学習率を調整
make training LEARNING_RATE=0.00001 EPOCHS=200
```

#### 2. 特定の文字が認識されない
```python
# 該当文字のデータを確認
python scripts/validate_data.py --char="X"

# 不足している場合は追加生成
python scripts/generate_synthetic_data.py --char="X" --count=200
```

#### 3. メモリ不足エラー
```bash
# バッチサイズを減らして学習
lstmtraining --max_image_MB 500 ...
```

#### 4. モデルサイズが大きすぎる
```bash
# モデルの圧縮
combine_tessdata -c granulate.traineddata
```

### パフォーマンス最適化

```python
# 並列処理による高速化
import concurrent.futures

def batch_recognize(images):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ocr.recognize_text, img) for img in images]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results
```

## 付録

### A. グラニュート文字マッピング表

```python
# 完全なマッピング辞書
GRANULATE_TO_LATIN = {
    # アルファベット
    'ᐁ': 'A', 'ᐂ': 'B', 'ᐃ': 'C', 'ᐄ': 'D', 'ᐅ': 'E',
    'ᐆ': 'F', 'ᐇ': 'G', 'ᐈ': 'H', 'ᐉ': 'I', 'ᐊ': 'J',
    'ᐋ': 'K', 'ᐌ': 'L', 'ᐍ': 'M', 'ᐎ': 'N', 'ᐏ': 'O',
    'ᐐ': 'P', 'ᐑ': 'Q', 'ᐒ': 'R', 'ᐓ': 'S', 'ᐔ': 'T',
    'ᐕ': 'U', 'ᐖ': 'V', 'ᐗ': 'W', 'ᐘ': 'X', 'ᐙ': 'Y', 'ᐚ': 'Z',
    
    # 数字
    '᐀': '0', 'ᑐ': '1', 'ᑑ': '2', 'ᑒ': '3', 'ᑓ': '4',
    'ᑔ': '5', 'ᑕ': '6', 'ᑖ': '7', 'ᑗ': '8', 'ᑘ': '9'
}
```

### B. 推奨ディレクトリ構造

```
training_data/
├── raw/                    # 元画像
│   ├── A/
│   ├── B/
│   └── ...
├── processed/              # 前処理済み画像
│   ├── A/
│   ├── B/
│   └── ...
├── tesseract_format/       # Tesseract形式
│   ├── granulate.0000.tif
│   ├── granulate.0000.box
│   └── ...
├── output/                 # 学習済みモデル
│   └── granulate.traineddata
├── scripts/                # 各種スクリプト
│   ├── generate_synthetic_data.py
│   ├── preprocess_images.py
│   ├── create_tesseract_files.py
│   └── train_tesseract.sh
└── Makefile               # 学習用Makefile
```

### C. 参考リンク

- [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- [Tesseract Training Wiki](https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract-4.00)
- [Tesseract.js Documentation](https://tesseract.projectnaptha.com/)

---

このガイドに従って学習データを準備し、カスタムOCRモデルを作成することで、グラニュート文字の高精度な認識が可能になります。