#!/usr/bin/env python3
"""
実際のテスト画像でOCRサービスをテスト
"""
import sys
from pathlib import Path
from PIL import Image
from io import BytesIO

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.application.services.ocr_service import OCRService

# テスト画像のパス
test_image_path = project_root / "test_data" / "test.png"

if test_image_path.exists():
    print("=== グラニュート文字OCR認識テスト ===\n")
    
    # 画像を読み込み
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    
    # OCRサービスを初期化
    ocr_service = OCRService()
    
    # OCRを実行
    print("認識処理中...")
    result = ocr_service.process_image(image_bytes)
    
    print(f"\n処理時間: {result.processing_time:.3f}秒")
    print(f"検出文字数: {len(result.characters)}")
    
    if result.characters:
        print("\n=== 認識結果 ===")
        
        # 位置でソート（左から右へ）
        sorted_chars = sorted(result.characters, key=lambda c: c.granulate_symbol)
        
        # 認識された文字列を構築
        recognized_text = ''.join(c.latin_equivalent for c in sorted_chars)
        print(f"\n認識された文字列: {recognized_text}")
        
        print("\n文字ごとの詳細:")
        print("-" * 60)
        print(f"{'No.':^4} | {'グラニュート':^12} | {'ラテン文字':^10} | {'信頼度':^8} | {'認識手法':^12}")
        print("-" * 60)
        
        for i, char in enumerate(sorted_chars, 1):
            # 信頼度から認識手法を推定
            if char.confidence >= 0.8:
                method = "CNN"
            elif char.confidence >= 0.7:
                method = "Tesseract"
            elif char.confidence >= 0.5:
                method = "Hash-based"
            elif char.confidence >= 0.3:
                method = "CNN"  # 0.3-0.5はCNN
            else:
                method = "CNN(low)"  # 0.3未満もCNN
            
            print(f"{i:^4} | {char.granulate_symbol:^12} | {char.latin_equivalent:^10} | {char.confidence:^8.2f} | {method:^12}")
        
        print("-" * 60)
        
        # 統計情報
        avg_confidence = sum(c.confidence for c in result.characters) / len(result.characters)
        print(f"\n平均信頼度: {avg_confidence:.2f}")
        
        # 手法別の統計
        method_counts = {"CNN": 0, "Tesseract": 0, "Hash-based": 0, "Unknown": 0}
        for char in result.characters:
            if char.confidence >= 0.8:
                method_counts["CNN"] += 1
            elif char.confidence >= 0.7:
                method_counts["Tesseract"] += 1
            elif char.confidence >= 0.5:
                method_counts["Hash-based"] += 1
            else:
                method_counts["Unknown"] += 1
        
        print("\n認識手法の内訳:")
        for method, count in method_counts.items():
            if count > 0:
                print(f"  {method}: {count}文字 ({count/len(result.characters)*100:.1f}%)")
    else:
        print("\n文字が検出されませんでした。")
else:
    print(f"テスト画像が見つかりません: {test_image_path}")