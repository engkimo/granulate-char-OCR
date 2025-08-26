#!/usr/bin/env python
"""
修正版CRNNサービスでカメラ画像をテスト
"""

import cv2
import numpy as np
from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed
import json

def test_camera_image():
    # 画像を読み込む
    image_path = "20250826_test.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"画像が読み込めません: {image_path}")
        return
    
    print(f"画像サイズ: {image.shape}")
    
    # 修正版OCRサービスを初期化
    print("\n=== 修正版CRNN付きOCRサービスでテスト ===")
    service = OCRServiceWithCRNNFixed()
    
    # 画像をバイト配列に変換
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    
    result = service.process_image(img_bytes)
    
    print(f"\n認識結果: {result.text}")
    print(f"平均信頼度: {result.average_confidence:.3f}")
    print(f"文字数: {len(result.characters)}")
    
    print("\n文字詳細:")
    for i, char in enumerate(result.characters):
        print(f"  [{i}] {char.granulate_symbol} → {char.latin_equivalent} (信頼度: {char.confidence:.3f})")
    
    # 結果を保存
    result_dict = {
        "text": result.text,
        "average_confidence": result.average_confidence,
        "characters": [
            {
                "granulate_symbol": char.granulate_symbol,
                "latin_equivalent": char.latin_equivalent,
                "confidence": char.confidence
            }
            for char in result.characters
        ]
    }
    
    with open("camera_test_result_fixed.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print("\n結果を保存: camera_test_result_fixed.json")
    
    # PLEASUREと比較
    expected = "PLEASURE"
    if result.text == expected:
        print(f"\n✅ 正しく認識されました！")
    else:
        print(f"\n❌ 誤認識: 期待値 '{expected}' → 実際 '{result.text}'")
        
        # 文字ごとの差異を表示
        print("\n文字ごとの比較:")
        for i, (e, a) in enumerate(zip(expected, result.text)):
            if e != a:
                print(f"  位置{i}: '{e}' → '{a}' ❌")
            else:
                print(f"  位置{i}: '{e}' → '{a}' ✅")
    
    return result

if __name__ == "__main__":
    test_camera_image()