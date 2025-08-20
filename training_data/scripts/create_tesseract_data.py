#!/usr/bin/env python3
"""
Tesseract用のカスタム言語データを作成
グラニュート文字をTesseractで認識可能にする
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil
import random
from typing import List, Tuple
import json


class TesseractDataGenerator:
    """Tesseract用の訓練データ生成クラス"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tesseract訓練用ディレクトリ
        self.tesseract_dir = self.output_dir / "tesseract"
        self.tesseract_dir.mkdir(exist_ok=True)
        
        self.classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    def create_box_file(self, image_path: Path, char: str, box_path: Path):
        """BOXファイルを作成（文字の位置情報）"""
        # 画像を読み込み
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        
        h, w = img.shape
        
        # BOXファイルのフォーマット: char left bottom right top page
        # Tesseractは左下が原点
        with open(box_path, 'w', encoding='utf-8') as f:
            f.write(f"{char} 0 0 {w} {h} 0\n")
    
    def create_training_images(self, num_samples_per_char: int = 10):
        """訓練用画像を作成"""
        print("Creating training images...")
        
        # 訓練用画像を収集
        training_images = []
        
        for char in self.classes:
            char_dir = self.data_dir / char
            if not char_dir.exists():
                continue
            
            # 各文字から画像を選択
            images = list(char_dir.glob("*.png"))
            selected = random.sample(images, min(num_samples_per_char, len(images)))
            
            for img_path in selected:
                training_images.append((img_path, char))
        
        # TIFFファイルとBOXファイルを作成
        page_num = 0
        tiff_images = []
        box_content = []
        
        for img_path, char in training_images:
            # 画像を読み込み
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 白背景に黒文字に変換（Tesseractの標準形式）
            img = 255 - img  # 反転
            
            # パディングを追加
            padded = cv2.copyMakeBorder(img, 10, 10, 10, 10, 
                                       cv2.BORDER_CONSTANT, value=255)
            
            h, w = padded.shape
            
            # PIL Imageに変換
            pil_img = Image.fromarray(padded)
            tiff_images.append(pil_img)
            
            # BOX情報を追加
            box_content.append(f"{char} 10 10 {w-10} {h-10} {page_num}")
            page_num += 1
        
        # マルチページTIFFとして保存
        if tiff_images:
            tiff_path = self.tesseract_dir / "gran.font.exp0.tif"
            tiff_images[0].save(
                tiff_path, 
                save_all=True, 
                append_images=tiff_images[1:],
                compression='tiff_lzw'
            )
            print(f"Created TIFF: {tiff_path}")
            
            # BOXファイルを保存
            box_path = self.tesseract_dir / "gran.font.exp0.box"
            with open(box_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(box_content) + '\n')
            print(f"Created BOX: {box_path}")
    
    def create_font_properties(self):
        """フォントプロパティファイルを作成"""
        font_props_path = self.tesseract_dir / "font_properties"
        with open(font_props_path, 'w') as f:
            # フォント名 italic bold fixed serif fraktur
            f.write("font 0 0 0 0 0\n")
        print(f"Created font properties: {font_props_path}")
    
    def create_unicharset(self):
        """文字セットファイルを作成"""
        unicharset_path = self.tesseract_dir / "unicharset"
        
        # Tesseractのunicharsetフォーマット
        with open(unicharset_path, 'w', encoding='utf-8') as f:
            # ヘッダー
            f.write(f"{len(self.classes) + 2}\n")  # 文字数 + NULL + 改行
            
            # NULL文字
            f.write("NULL 0 NULL 0\n")
            
            # 各文字
            for i, char in enumerate(self.classes):
                # properties: isalpha, islower, isupper, isdigit, ispunctuation
                f.write(f"{char} 1 0,255,{i+1} 1 0 1 0 0\n")
            
            # 改行文字
            f.write("\\n 2 10 2 0 0 0 0\n")
        
        print(f"Created unicharset: {unicharset_path}")
    
    def create_wordlist(self):
        """単語リストを作成（グラニュート文字の組み合わせ）"""
        wordlist_path = self.tesseract_dir / "gran.wordlist"
        
        # サンプル単語を生成
        words = []
        
        # 単一文字
        words.extend(self.classes)
        
        # 2-5文字の組み合わせ
        for length in range(2, 6):
            for _ in range(20):
                word = ''.join(random.choices(self.classes, k=length))
                words.append(word)
        
        with open(wordlist_path, 'w') as f:
            f.write('\n'.join(sorted(set(words))) + '\n')
        
        print(f"Created wordlist: {wordlist_path}")
    
    def create_training_script(self):
        """Tesseract訓練用のシェルスクリプトを作成"""
        script_path = self.tesseract_dir / "train_tesseract.sh"
        
        script_content = """#!/bin/bash
# Tesseract訓練スクリプト

echo "Starting Tesseract training for Granulate characters..."

# 作業ディレクトリに移動
cd "$(dirname "$0")"

# 1. 訓練データからlstmfファイルを生成
echo "Creating training data..."
tesseract gran.font.exp0.tif gran.font.exp0 --psm 6 lstm.train

# 2. unicharsetを抽出
echo "Extracting unicharset..."
unicharset_extractor gran.font.exp0.box

# 3. フォントプロパティを設定
echo "Setting font properties..."
set_unicharset_properties -U unicharset -O unicharset --script_dir .

# 4. 訓練用のスターターモデルを作成
echo "Creating starter traineddata..."
combine_lang_model \\
  --input_unicharset unicharset \\
  --script_dir . \\
  --output_dir . \\
  --lang gran

# 5. LSTM訓練を実行
echo "Training LSTM model..."
lstmtraining \\
  --model_output ./gran \\
  --continue_from ./gran.lstm \\
  --traineddata ./gran.traineddata \\
  --train_listfile ./gran.training_files.txt \\
  --max_iterations 400

# 6. 最終的なtraineddataを作成
echo "Creating final traineddata..."
lstmtraining \\
  --stop_training \\
  --continue_from ./gran_checkpoint \\
  --traineddata ./gran.traineddata \\
  --model_output ./gran.traineddata

echo "Training complete! Copy gran.traineddata to your tessdata directory."
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 実行可能にする
        script_path.chmod(0o755)
        
        print(f"Created training script: {script_path}")
    
    def create_config_file(self):
        """Tesseract設定ファイルを作成"""
        config_path = self.tesseract_dir / "gran.config"
        
        config_content = """# Granulate character recognition config
tessedit_char_whitelist ABCDEFGHIJKLMNOPQRSTUVWXYZ
preserve_interword_spaces 0
tessedit_pageseg_mode 8
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Created config: {config_path}")
    
    def create_all_training_data(self):
        """全ての訓練データを作成"""
        print("=== Creating Tesseract Training Data ===\n")
        
        # 訓練用画像とBOXファイルを作成
        self.create_training_images(num_samples_per_char=20)
        
        # フォントプロパティを作成
        self.create_font_properties()
        
        # unicharsetを作成
        self.create_unicharset()
        
        # 単語リストを作成
        self.create_wordlist()
        
        # 設定ファイルを作成
        self.create_config_file()
        
        # 訓練スクリプトを作成
        self.create_training_script()
        
        # 手順を出力
        print("\n=== Next Steps ===")
        print("1. Install Tesseract training tools:")
        print("   brew install tesseract --with-training-tools")
        print("   # or")
        print("   sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-eng")
        print("\n2. Navigate to the training directory:")
        print(f"   cd {self.tesseract_dir}")
        print("\n3. Run the training script:")
        print("   ./train_tesseract.sh")
        print("\n4. Copy the trained data to Tesseract:")
        print("   cp gran.traineddata /usr/local/share/tessdata/")
        print("   # or")
        print("   cp gran.traineddata /usr/share/tesseract-ocr/4.00/tessdata/")
        
        # 簡易訓練手順をREADMEとして保存
        readme_path = self.tesseract_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("""# Tesseract Training for Granulate Characters

## Prerequisites
- Tesseract 4.0+ with training tools
- Python packages: pillow, opencv-python

## Training Steps

1. **Generate training data** (already done):
   - TIFF images with corresponding BOX files
   - Font properties
   - Unicharset
   - Word list

2. **Run Tesseract training**:
   ```bash
   # Generate LSTM training files
   tesseract gran.font.exp0.tif gran.font.exp0 --psm 6 lstm.train
   
   # Extract character set
   unicharset_extractor gran.font.exp0.box
   
   # Train the model (simplified version)
   combine_tessdata -e tessdata/eng.traineddata eng.
   lstmtraining --model_output gran \\
     --continue_from eng.lstm \\
     --traineddata eng.traineddata \\
     --train_listfile train.list \\
     --max_iterations 400
   ```

3. **Install trained data**:
   ```bash
   sudo cp gran.traineddata /usr/local/share/tessdata/
   ```

4. **Test the model**:
   ```python
   import pytesseract
   
   # Use custom language
   text = pytesseract.image_to_string(image, lang='gran')
   ```

## Using with the OCR API

The backend will automatically use the custom Tesseract model if available.
Make sure to set the language parameter to 'gran' in API calls.
""")
        
        print(f"\nTraining data created in: {self.tesseract_dir}")
        print(f"README available at: {readme_path}")


def main():
    # ディレクトリ設定
    data_dir = Path("training_data/augmented")
    output_dir = Path("training_data")
    
    # データ生成器を作成
    generator = TesseractDataGenerator(data_dir, output_dir)
    
    # 全ての訓練データを作成
    generator.create_all_training_data()


if __name__ == "__main__":
    main()