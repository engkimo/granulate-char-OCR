#!/usr/bin/env python3
"""
統合OCRサービスの認識精度テスト
CNN + Tesseract + Hash-based の組み合わせをテスト
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

# テスト用の画像を準備
def create_test_image(char: str) -> bytes:
    """テスト用の画像を作成"""
    # 元の画像を読み込み
    img_path = Path(f"training_data/extracted/{char}/{char}_reference.png")
    if not img_path.exists():
        return None
    
    # 画像を読み込んでバイト列に変換
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

# OCRサービスをテスト
def test_ocr_service():
    print("=== 統合OCRサービスのテスト ===\n")
    
    # OCRサービスを初期化
    ocr_service = OCRService()
    
    # 全26文字をテスト
    results = []
    all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    print("文字ごとの認識結果:")
    print("-" * 60)
    print(f"{'文字':^6} | {'認識結果':^10} | {'信頼度':^8} | {'認識手法':^15} | {'状態':^6}")
    print("-" * 60)
    
    for char in all_chars:
        image_bytes = create_test_image(char)
        if image_bytes:
            # OCRを実行
            result = ocr_service.process_image(image_bytes)
            
            if result.characters:
                # 最初の文字の結果を使用
                recognized = result.characters[0]
                recognized_char = recognized.latin_equivalent
                confidence = recognized.confidence
                
                # 認識手法を推定
                if confidence >= 0.8:
                    method = "CNN"
                elif confidence >= 0.7:
                    method = "Tesseract"
                elif confidence >= 0.5:
                    method = "Hash-based"
                else:
                    method = "Unknown"
                
                status = "✓" if recognized_char == char else "✗"
                results.append((char, recognized_char, confidence, method, status))
                
                print(f"{char:^6} | {recognized_char:^10} | {confidence:^8.2f} | {method:^15} | {status:^6}")
            else:
                results.append((char, "なし", 0.0, "失敗", "✗"))
                print(f"{char:^6} | {'なし':^10} | {0.0:^8.2f} | {'失敗':^15} | {'✗':^6}")
        else:
            results.append((char, "画像なし", 0.0, "N/A", "✗"))
            print(f"{char:^6} | {'画像なし':^10} | {0.0:^8.2f} | {'N/A':^15} | {'✗':^6}")
    
    print("-" * 60)
    
    # 統計情報
    successful = sum(1 for _, _, _, _, status in results if status == "✓")
    total = len(results)
    accuracy = successful / total * 100
    
    print(f"\n=== 統計情報 ===")
    print(f"全体精度: {successful}/{total} ({accuracy:.1f}%)")
    
    # 手法別の統計
    method_counts = {}
    method_success = {}
    for char, recognized, conf, method, status in results:
        if method not in method_counts:
            method_counts[method] = 0
            method_success[method] = 0
        method_counts[method] += 1
        if status == "✓":
            method_success[method] += 1
    
    print(f"\n手法別の認識結果:")
    for method in sorted(method_counts.keys()):
        count = method_counts[method]
        success = method_success[method]
        rate = success / count * 100 if count > 0 else 0
        print(f"  {method}: {success}/{count} ({rate:.1f}%)")
    
    # 改善された文字の確認
    print(f"\n=== Tesseractから改善された文字 ===")
    tesseract_failed = ['D', 'F', 'H', 'J', 'K', 'M', 'R', 'S', 'T', 'W']
    improved = []
    for char, recognized, conf, method, status in results:
        if char in tesseract_failed and status == "✓":
            improved.append(f"{char} ({method})")
    
    if improved:
        print(f"改善された文字: {', '.join(improved)}")
    else:
        print("改善された文字はありません")
    
    # 失敗した文字
    failed_chars = []
    for char, recognized, conf, method, status in results:
        if status == "✗":
            failed_chars.append(f"{char}→{recognized}")
    
    if failed_chars:
        print(f"\n認識失敗: {', '.join(failed_chars)}")

if __name__ == "__main__":
    test_ocr_service()