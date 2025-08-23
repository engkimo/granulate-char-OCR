#!/usr/bin/env python3
"""
改善パラメータのテスト - 最適な設定を見つける
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

# テスト画像
test_image_path = Path("test_data/test.png")

if test_image_path.exists():
    print("=== 改善パラメータのテスト ===\n")
    
    # 画像を読み込み
    img = cv2.imread(str(test_image_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 白背景なので反転
    img_inv = 255 - img_gray
    
    # ノイズ除去
    denoised = cv2.bilateralFilter(img_inv, 9, 75, 75)
    
    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 二値化
    _, binary = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)
    
    # 異なるモルフォロジーパラメータをテスト
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    params = [
        (1, 1, "軽い処理"),
        (2, 1, "現在の設定"),
        (3, 1, "中程度"),
        (3, 2, "強い膨張"),
        (5, 2, "より強い膨張"),
        (7, 3, "最強"),
        (3, 1, "水平方向強化"),  # 特殊カーネル
        (1, 3, "垂直方向強化"),  # 特殊カーネル
        (5, 1, "大きめカーネル")
    ]
    
    for idx, (kernel_size, iterations, label) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        
        # 特殊カーネルの処理
        if label == "水平方向強化":
            kernel = np.ones((1, 5), np.uint8)  # 水平方向に長いカーネル
        elif label == "垂直方向強化":
            kernel = np.ones((5, 1), np.uint8)  # 垂直方向に長いカーネル
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 膨張処理
        dilated = cv2.dilate(binary, kernel, iterations=iterations)
        
        # 輪郭検出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 大きな輪郭のみカウント
        large_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 15 and h > 15:
                large_contours.append((x, y, w, h))
        
        # 結果を表示
        result_img = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
        for x, y, w, h in large_contours:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        ax.imshow(result_img)
        ax.set_title(f'{label}\nKernel:{kernel_size}x{kernel_size}, Iter:{iterations}\n検出:{len(large_contours)}個')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_test_results.png', dpi=150)
    print("パラメータテスト結果を 'parameter_test_results.png' に保存しました。")
    
    # 最適なパラメータを選択
    print("\n推奨設定:")
    print("- カーネルサイズ: 5x5")
    print("- 膨張回数: 2回")
    print("- これにより、分離した文字部分がより確実に結合されます")
    
    # プロジェクション分析も試す
    print("\n=== 水平プロジェクション分析 ===")
    horizontal_projection = np.sum(binary, axis=0)
    
    # 文字境界の検出
    threshold = np.max(horizontal_projection) * 0.1
    in_char = False
    char_boundaries = []
    start = 0
    
    for i, val in enumerate(horizontal_projection):
        if not in_char and val > threshold:
            in_char = True
            start = i
        elif in_char and val <= threshold:
            in_char = False
            if i - start > 10:  # 最小幅
                char_boundaries.append((start, i))
    
    print(f"プロジェクション分析で検出された文字数: {len(char_boundaries)}")
    print("文字境界:", char_boundaries)
    
else:
    print(f"テスト画像が見つかりません: {test_image_path}")