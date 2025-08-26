#!/usr/bin/env python
"""
カメラキャプチャ画像でOCRサービスをテスト
"""

import cv2
import numpy as np
from backend.application.services.ocr_service_crnn import OCRServiceWithCRNN
from backend.application.services.ocr_service import OCRService
import json

def test_camera_image():
    # 画像を読み込む
    image_path = "20250826_test.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"画像が読み込めません: {image_path}")
        return
    
    print(f"画像サイズ: {image.shape}")
    
    # OCRサービスを初期化（CRNNあり）
    print("\n=== CRNN付きOCRサービスでテスト ===")
    service_crnn = OCRServiceWithCRNN()
    
    # 画像をバイト配列に変換
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    
    result_crnn = service_crnn.process_image(img_bytes)
    
    print(f"認識結果: {result_crnn.text}")
    print(f"平均信頼度: {result_crnn.average_confidence:.3f}")
    print(f"文字数: {len(result_crnn.characters)}")
    
    print("\n文字詳細:")
    for i, char in enumerate(result_crnn.characters):
        print(f"  [{i}] {char.granulate_symbol} → {char.latin_equivalent} (信頼度: {char.confidence:.3f})")
    
    # 前処理の各ステップを可視化
    print("\n=== 前処理ステップの可視化 ===")
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_1_gray.png", gray)
    print("1. グレースケール画像を保存: debug_1_gray.png")
    
    # 背景判定
    mean_val = np.mean(gray)
    print(f"2. 背景平均値: {mean_val:.1f} ({'白背景' if mean_val > 128 else '黒背景'})")
    
    if mean_val > 128:
        gray_inv = 255 - gray
        cv2.imwrite("debug_2_inverted.png", gray_inv)
        print("   → 白背景なので反転: debug_2_inverted.png")
    else:
        gray_inv = gray
    
    # ノイズ除去
    denoised = cv2.bilateralFilter(gray_inv, 9, 75, 75)
    cv2.imwrite("debug_3_denoised.png", denoised)
    print("3. ノイズ除去後: debug_3_denoised.png")
    
    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    cv2.imwrite("debug_4_enhanced.png", enhanced)
    print("4. コントラスト強調後: debug_4_enhanced.png")
    
    # 二値化
    _, binary = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug_5_binary.png", binary)
    print("5. 二値化後: debug_5_binary.png")
    
    # モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("debug_6_morphology.png", morph)
    print("6. モルフォロジー処理後: debug_6_morphology.png")
    
    # 文字領域の切り出しを可視化
    print("\n=== 文字領域の切り出し ===")
    horizontal_projection = np.sum(morph, axis=0)
    
    # プロジェクションを可視化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(horizontal_projection)
    plt.title("Horizontal Projection")
    plt.xlabel("X position")
    plt.ylabel("Sum of white pixels")
    plt.grid(True)
    plt.savefig("debug_7_projection.png")
    plt.close()
    print("7. 水平プロジェクション: debug_7_projection.png")
    
    # より穏やかな前処理でテスト
    print("\n=== より穏やかな前処理でテスト ===")
    
    # カスタム前処理
    gray_mild = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_mild) > 128:
        gray_mild = 255 - gray_mild
    
    # より穏やかなノイズ除去
    denoised_mild = cv2.bilateralFilter(gray_mild, 5, 50, 50)
    
    # より穏やかな二値化（適応的閾値）
    binary_mild = cv2.adaptiveThreshold(
        denoised_mild, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite("debug_8_mild_binary.png", binary_mild)
    print("8. 穏やかな二値化: debug_8_mild_binary.png")
    
    return result_crnn

if __name__ == "__main__":
    result = test_camera_image()
    
    # 結果をJSONで保存
    if result:
        result_dict = {
            "text": result.text,
            "average_confidence": result.average_confidence,
            "characters": [
                {
                    "granulate_symbol": char.granulate_symbol,
                    "latin_equivalent": char.latin_equivalent,
                    "confidence": char.confidence,
                    "position": char.position
                }
                for char in result.characters
            ]
        }
        
        with open("camera_test_result.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print("\n結果を保存: camera_test_result.json")