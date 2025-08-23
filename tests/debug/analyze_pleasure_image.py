#!/usr/bin/env python3
"""
PLEASURE画像の詳細分析
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from backend.application.services.ocr_service import OCRService

# テスト画像
test_image_path = Path("test_data/test.png")

if test_image_path.exists():
    print("=== PLEASURE画像の分析 ===\n")
    print("実際の文字列: PLEASURE (8文字)")
    print("認識結果: BCDEEEEPPQTX (12文字)")
    print("\n文字分割の問題を調査中...\n")
    
    # 画像を読み込み
    img = cv2.imread(str(test_image_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OCRサービスの前処理
    ocr_service = OCRService()
    preprocessed = ocr_service._preprocess_image(img_gray)
    
    # 文字領域の抽出
    char_regions = ocr_service._extract_character_regions(preprocessed)
    
    # 結果を可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 元画像
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('元画像 - PLEASURE')
    axes[0].axis('off')
    
    # 2. 前処理後の画像と検出された文字領域
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title(f'前処理後 - {len(char_regions)}個の領域検出（正解は8個）')
    
    # 各文字領域を異なる色で表示
    colors = plt.cm.rainbow(np.linspace(0, 1, len(char_regions)))
    for i, (x, y, w, h) in enumerate(char_regions):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=colors[i], facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x + w/2, y - 5, f'{i+1}', color=colors[i], 
                    ha='center', fontsize=10, weight='bold')
    
    # 3. 改善案：より大きなカーネルでのモルフォロジー処理
    kernel_large = np.ones((5, 5), np.uint8)
    # 膨張処理で近い文字部分を結合
    dilated = cv2.dilate(preprocessed, kernel_large, iterations=2)
    # 収縮処理で元のサイズに近づける
    improved = cv2.erode(dilated, kernel_large, iterations=1)
    
    # 改善後の輪郭検出
    contours_improved, _ = cv2.findContours(improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    axes[2].imshow(improved, cmap='gray')
    axes[2].set_title(f'改善案（モルフォロジー処理強化） - {len(contours_improved)}個の輪郭')
    
    # 改善後の輪郭を表示
    for i, contour in enumerate(contours_improved):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # 小さすぎる輪郭を除外
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('analyze_pleasure_result.png', dpi=150, bbox_inches='tight')
    print("分析結果を 'analyze_pleasure_result.png' に保存しました。")
    
    # 文字領域の詳細情報
    print("\n検出された文字領域の詳細:")
    print("-" * 60)
    print("No. | X座標 | Y座標 | 幅 | 高さ | 推定文字")
    print("-" * 60)
    
    expected_chars = list("PLEASURE")
    for i, (x, y, w, h) in enumerate(char_regions):
        # 切り出した画像でCNN認識
        char_image = preprocessed[y:y+h, x:x+w]
        recognized_char, confidence = ocr_service.process_with_cnn(char_image)
        
        # X座標から推定される正しい文字
        expected_idx = min(int(x / (img.shape[1] / 8)), 7)
        expected_char = expected_chars[expected_idx] if expected_idx < len(expected_chars) else "?"
        
        print(f"{i+1:3} | {x:5} | {y:5} | {w:3} | {h:4} | {expected_char} → {recognized_char or '?'}")
    
    print("-" * 60)
    
    # 問題の分析
    print("\n問題の分析:")
    print("1. 文字の分割が過剰（12個検出、正解は8個）")
    print("2. グラニュート文字の特殊な形状により、1文字が複数部分に分割されている")
    print("3. 特に複雑な文字（P, R, Aなど）が誤分割されている可能性")
    
    print("\n推奨される改善策:")
    print("1. モルフォロジー処理の強化（膨張・収縮）で文字部分を結合")
    print("2. 最小文字サイズの閾値を上げる（現在5ピクセル → 15ピクセル）")
    print("3. 隣接する小さな輪郭を結合するアルゴリズムの実装")
    print("4. 実際のグラニュート文字画像でのCNN再訓練")
    
else:
    print(f"テスト画像が見つかりません: {test_image_path}")