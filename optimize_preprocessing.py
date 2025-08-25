#!/usr/bin/env python3
"""
新しいテストデータに最適化した前処理パイプライン
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from backend.application.services.ocr_service import OCRService
from training_data.scripts.preprocess_color_images import ColorAwarePreprocessor


class OptimizedPreprocessor:
    """新しいテストデータに最適化された前処理クラス"""
    
    def __init__(self):
        self.color_preprocessor = ColorAwarePreprocessor()
        self.ocr_service = OCRService()
    
    def adaptive_preprocess(self, image: np.ndarray) -> np.ndarray:
        """画像の特性に応じた適応的前処理"""
        
        # カラー画像の場合
        if len(image.shape) == 3:
            # HSV変換
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 彩度の平均値で判定
            mean_saturation = hsv[:, :, 1].mean()
            
            if mean_saturation > 100:
                # カラフルな画像 → 色ベースの処理
                return self._process_colorful_image(image)
            else:
                # 彩度が低い → エッジベースの処理
                return self._process_low_saturation_image(image)
        else:
            # グレースケール画像
            return self._process_grayscale_image(image)
    
    def _process_colorful_image(self, image: np.ndarray) -> np.ndarray:
        """カラフルな画像の処理"""
        # LAB色空間で処理
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # L*チャンネル（明度）を強調
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # エッジ保持フィルタ
        l_filtered = cv2.bilateralFilter(l_enhanced, 9, 75, 75)
        
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            l_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # テキストの向きを検出して補正
        return self._correct_text_orientation(binary)
    
    def _process_low_saturation_image(self, image: np.ndarray) -> np.ndarray:
        """彩度が低い画像の処理"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
    
    def _process_grayscale_image(self, image: np.ndarray) -> np.ndarray:
        """グレースケール画像の処理"""
        # エッジ保持フィルタ
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    def _correct_text_orientation(self, binary: np.ndarray) -> np.ndarray:
        """テキストの向きを補正"""
        # 輪郭を検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
        
        # 最大の輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 最小外接矩形を取得
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # 角度が45度以上の場合は90度回転が必要
        if angle < -45:
            angle = 90 + angle
        
        # 回転が必要な場合
        if abs(angle) > 5:
            (h, w) = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return binary
    
    def extract_text_regions(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """改善された文字領域抽出"""
        # 水平方向の膨張で文字を結合
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        connected = cv2.dilate(binary, kernel_h, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # バウンディングボックスを取得
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # ノイズ除去
            if w > 10 and h > 10:
                boxes.append((x, y, w, h))
        
        # x座標でソート
        boxes.sort(key=lambda b: b[0])
        
        return boxes


def test_preprocessing_methods():
    """様々な前処理手法をテスト"""
    test_images = [
        "test_data/PLEASURE_1.png",
        "test_data/OPERATE_1.png", 
        "test_data/STOMACH_1.png",
        "test_data/HIRING!_1.png",
    ]
    
    preprocessor = OptimizedPreprocessor()
    
    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 4*len(test_images)))
    
    for i, img_path in enumerate(test_images):
        # 画像を読み込み
        image = cv2.imread(img_path)
        
        # オリジナル
        axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original: {Path(img_path).stem}')
        axes[i, 0].axis('off')
        
        # 現在の前処理
        current = preprocessor.ocr_service._preprocess_image(image)
        axes[i, 1].imshow(current, cmap='gray')
        axes[i, 1].set_title('Current Method')
        axes[i, 1].axis('off')
        
        # 最適化された前処理
        optimized = preprocessor.adaptive_preprocess(image)
        axes[i, 2].imshow(optimized, cmap='gray')
        axes[i, 2].set_title('Optimized Method')
        axes[i, 2].axis('off')
        
        # 文字領域の可視化
        regions = preprocessor.extract_text_regions(optimized)
        vis = cv2.cvtColor(optimized, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in regions:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axes[i, 3].imshow(vis)
        axes[i, 3].set_title(f'Detected: {len(regions)} regions')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison_new.png')
    print("前処理比較画像を保存: preprocessing_comparison_new.png")


def evaluate_preprocessing():
    """前処理の効果を評価"""
    preprocessor = OptimizedPreprocessor()
    ocr_service = OCRService()
    
    test_images = list(Path("test_data").glob("*_*.png"))[:20]  # 最初の20枚でテスト
    
    results = {
        'current': {'correct': 0, 'total': 0},
        'optimized': {'correct': 0, 'total': 0}
    }
    
    for img_path in tqdm(test_images, desc="前処理評価"):
        expected = img_path.stem.split('_')[0].replace('!', '')
        
        # 画像を読み込み
        image = cv2.imread(str(img_path))
        
        # 現在の前処理で認識
        with open(img_path, 'rb') as f:
            result = ocr_service.process_image(f.read())
        current_text = ''.join(c.latin_equivalent for c in result.characters)
        
        # 文字レベルで評価
        for e, r in zip(expected, current_text):
            results['current']['total'] += 1
            if e == r:
                results['current']['correct'] += 1
        
        # 最適化された前処理（ここではシミュレーション）
        # 実際にはOCRServiceの前処理を置き換える必要がある
        optimized = preprocessor.adaptive_preprocess(image)
        # TODO: 最適化された前処理での認識結果を取得
    
    print("\n=== 前処理の評価結果 ===")
    print(f"現在の手法: {results['current']['correct']}/{results['current']['total']} "
          f"({results['current']['correct']/results['current']['total']*100:.1f}%)")


if __name__ == "__main__":
    print("前処理手法の比較...")
    test_preprocessing_methods()
    
    print("\n前処理の効果を評価...")
    evaluate_preprocessing()