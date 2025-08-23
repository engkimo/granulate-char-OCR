#!/usr/bin/env python3
"""
テスト画像でCNNが使用されない理由を調査
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from backend.application.services.ocr_service import OCRService

# テスト画像
test_image_path = Path("test_data/test.png")

if test_image_path.exists():
    print("=== CNNデバッグ ===\n")
    
    # 画像を読み込み
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    
    # OCRサービスを初期化
    ocr_service = OCRService()
    
    # 画像の前処理
    image = Image.open(BytesIO(image_bytes))
    image_np = np.array(image)
    preprocessed = ocr_service._preprocess_image(image_np)
    
    # 文字領域の抽出
    char_regions = ocr_service._extract_character_regions(preprocessed)
    print(f"検出された文字領域数: {len(char_regions)}")
    
    # 各文字領域でCNNをテスト
    for i, (x, y, w, h) in enumerate(char_regions[:5]):  # 最初の5文字
        print(f"\n=== 文字領域 {i+1} (x={x}, y={y}, w={w}, h={h}) ===")
        
        # 文字画像を切り出し
        char_image = preprocessed[y:y+h, x:x+w]
        print(f"切り出し画像サイズ: {char_image.shape}")
        
        # CNNで処理
        recognized_char, cnn_confidence = ocr_service.process_with_cnn(char_image)
        print(f"CNN結果: '{recognized_char}' (信頼度: {cnn_confidence:.3f})")
        
        if not recognized_char or cnn_confidence < 0.8:
            print("→ CNNの信頼度が低いため、Tesseractを使用")
            
        # デバッグ用に画像を保存
        cv2.imwrite(f'debug_char_{i+1}.png', char_image)
        
        # 反転して再度試す
        char_image_inv = 255 - char_image
        recognized_inv, conf_inv = ocr_service.process_with_cnn(char_image_inv)
        print(f"反転画像のCNN結果: '{recognized_inv}' (信頼度: {conf_inv:.3f})")
    
    # CNNモデルの状態を確認
    print(f"\n\nCNNモデルの状態:")
    print(f"  - モデルロード済み: {ocr_service.cnn_model is not None}")
    if ocr_service.cnn_model:
        print(f"  - デバイス: {ocr_service.device}")
        print(f"  - モデルモード: {'eval' if not ocr_service.cnn_model.training else 'train'}")
else:
    print(f"テスト画像が見つかりません: {test_image_path}")