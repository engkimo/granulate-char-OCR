#!/usr/bin/env python3
"""
カラー画像や多階調画像に対応した前処理パイプライン
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class ColorAwarePreprocessor:
    """色情報を活用する前処理クラス"""
    
    def __init__(self):
        # グラニュート文字でよく使われる色の範囲（HSV）
        self.color_ranges = {
            'purple': ([120, 50, 50], [150, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 30])
        }
    
    def preprocess(self, image: np.ndarray, mode: str = 'auto') -> np.ndarray:
        """
        画像を前処理
        
        Args:
            image: 入力画像（カラーまたはグレースケール）
            mode: 処理モード
                - 'auto': 自動判定
                - 'color': カラー情報を活用
                - 'binary': 従来の二値化
                - 'multigrade': 多階調を保持
        """
        if mode == 'auto':
            mode = self._detect_best_mode(image)
            print(f"自動検出モード: {mode}")
        
        if mode == 'color':
            return self._preprocess_color(image)
        elif mode == 'multigrade':
            return self._preprocess_multigrade(image)
        else:
            return self._preprocess_binary(image)
    
    def _detect_best_mode(self, image: np.ndarray) -> str:
        """最適な処理モードを自動判定"""
        if len(image.shape) == 2:
            # グレースケール画像
            return 'binary'
        
        # カラー画像の場合、色の分布を分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 色相のヒストグラム
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_h = hist_h.flatten() / hist_h.sum()
        
        # 彩度の平均
        mean_saturation = hsv[:, :, 1].mean()
        
        if mean_saturation > 50:
            # 彩度が高い = カラフルな画像
            return 'color'
        else:
            # 彩度が低い = ほぼグレースケール
            return 'multigrade'
    
    def _preprocess_color(self, image: np.ndarray) -> np.ndarray:
        """カラー情報を活用した前処理"""
        if len(image.shape) == 2:
            # グレースケールの場合は通常処理
            return self._preprocess_binary(image)
        
        # HSV色空間で処理
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 各色のマスクを作成
        masks = []
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if mask.sum() > 0:  # この色が存在する場合
                masks.append(mask)
        
        if masks:
            # 複数の色マスクを結合
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # ノイズ除去
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
        else:
            # 色が検出されない場合は通常処理
            return self._preprocess_binary(image)
    
    def _preprocess_multigrade(self, image: np.ndarray) -> np.ndarray:
        """多階調を保持した前処理"""
        if len(image.shape) == 3:
            # Lab色空間で処理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 明度チャンネルを強調
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # エッジ保持フィルタ
            l_filtered = cv2.bilateralFilter(l_enhanced, 9, 75, 75)
            
            # グレースケールとして返す
            return l_filtered
        else:
            # グレースケール画像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            return filtered
    
    def _preprocess_binary(self, image: np.ndarray) -> np.ndarray:
        """従来の二値化処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # ノイズ除去
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def visualize_modes(self, image_path: Path):
        """各モードの処理結果を視覚化"""
        image = cv2.imread(str(image_path))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # オリジナル
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # 各モードで処理
        modes = ['auto', 'color', 'binary', 'multigrade']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        
        for mode, (row, col) in zip(modes, positions):
            result = self.preprocess(image, mode=mode)
            axes[row, col].imshow(result, cmap='gray')
            axes[row, col].set_title(f'Mode: {mode}')
            axes[row, col].axis('off')
        
        # 最後のセルは空
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        output_path = image_path.parent / f"{image_path.stem}_preprocessing_comparison.png"
        plt.savefig(output_path)
        print(f"比較画像を保存: {output_path}")


def test_with_granulate_images():
    """グラニュート文字画像でテスト"""
    preprocessor = ColorAwarePreprocessor()
    
    # テスト画像のパス
    test_images = [
        Path("test_data/test.png"),  # 現在のPLEASURE画像
        # 追加のテスト画像があればここに
    ]
    
    for img_path in test_images:
        if img_path.exists():
            print(f"\n処理中: {img_path}")
            preprocessor.visualize_modes(img_path)
            
            # 各モードで認識テスト
            image = cv2.imread(str(img_path))
            for mode in ['auto', 'color', 'binary', 'multigrade']:
                processed = preprocessor.preprocess(image, mode=mode)
                print(f"  {mode}モード: 出力shape={processed.shape}, dtype={processed.dtype}")


if __name__ == "__main__":
    test_with_granulate_images()