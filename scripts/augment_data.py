#!/usr/bin/env python3
"""
データ拡張パイプラインの強化
実際のテストデータに近い訓練データを生成
"""
import cv2
import numpy as np
from pathlib import Path
import random
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm


class EnhancedDataAugmentor:
    """強化されたデータ拡張クラス"""
    
    def __init__(self):
        self.thickness_variations = [1, 2, 3, 4]  # 文字の太さバリエーション
        self.background_types = ['black', 'white', 'gradient', 'textured']
        self.noise_levels = [0, 5, 10, 15]
    
    def augment_thickness(self, image: np.ndarray, target_thickness: int) -> np.ndarray:
        """文字の太さを変更"""
        if target_thickness == 1:
            # 細くする（エロージョン）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            return cv2.erode(image, kernel, iterations=1)
        elif target_thickness == 2:
            # そのまま
            return image
        elif target_thickness == 3:
            # 少し太くする
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            return cv2.dilate(image, kernel, iterations=1)
        else:
            # かなり太くする
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.dilate(image, kernel, iterations=1)
    
    def create_background(self, shape: Tuple[int, int], bg_type: str) -> np.ndarray:
        """様々な背景を生成"""
        h, w = shape
        
        if bg_type == 'black':
            return np.zeros((h, w), dtype=np.uint8)
        
        elif bg_type == 'white':
            return np.ones((h, w), dtype=np.uint8) * 255
        
        elif bg_type == 'gradient':
            # グラデーション背景
            gradient = np.linspace(0, 100, w)
            background = np.tile(gradient, (h, 1))
            return background.astype(np.uint8)
        
        elif bg_type == 'textured':
            # テクスチャ背景（ノイズパターン）
            background = np.random.normal(50, 20, (h, w))
            background = np.clip(background, 0, 100)
            return background.astype(np.uint8)
        
        return np.zeros((h, w), dtype=np.uint8)
    
    def add_realistic_noise(self, image: np.ndarray, noise_level: int) -> np.ndarray:
        """リアルなノイズを追加"""
        if noise_level == 0:
            return image
        
        # ガウシアンノイズ
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        
        # ソルト＆ペッパーノイズ
        if noise_level > 10:
            salt_pepper = np.random.random(image.shape)
            noisy[salt_pepper < 0.01] = 0
            noisy[salt_pepper > 0.99] = 255
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def simulate_lighting_variations(self, image: np.ndarray) -> np.ndarray:
        """照明変化をシミュレート"""
        # ランダムな明度調整
        brightness = random.uniform(0.7, 1.3)
        adjusted = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # 部分的な影
        if random.random() < 0.3:
            h, w = image.shape
            shadow = np.ones((h, w), dtype=np.float32)
            
            # ランダムな位置に影を作成
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            shadow[:, x1:x2] = 0.5
            
            adjusted = (adjusted * shadow).astype(np.uint8)
        
        return adjusted
    
    def augment_single_image(self, image: np.ndarray, num_variations: int = 5) -> List[np.ndarray]:
        """単一画像から複数のバリエーションを生成"""
        variations = []
        
        # オリジナルの二値化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        for _ in range(num_variations):
            # ランダムなパラメータを選択
            thickness = random.choice(self.thickness_variations)
            bg_type = random.choice(self.background_types)
            noise_level = random.choice(self.noise_levels)
            
            # 文字の太さを変更
            char_image = self.augment_thickness(binary, thickness)
            
            # 背景を作成
            background = self.create_background(char_image.shape, bg_type)
            
            # 文字と背景を合成
            if bg_type in ['white', 'gradient', 'textured']:
                # 明るい背景の場合は文字を黒に
                result = background.copy()
                result[char_image > 128] = 0
            else:
                # 暗い背景の場合は文字を白に
                result = background.copy()
                result[char_image > 128] = 255
            
            # ノイズを追加
            result = self.add_realistic_noise(result, noise_level)
            
            # 照明変化を追加
            result = self.simulate_lighting_variations(result)
            
            variations.append(result)
        
        return variations


def augment_training_data(input_dir: Path, output_dir: Path):
    """訓練データ全体を拡張"""
    augmentor = EnhancedDataAugmentor()
    output_dir.mkdir(exist_ok=True)
    
    # 各文字ディレクトリを処理
    for char_dir in sorted(input_dir.glob('[A-Z]')):
        if not char_dir.is_dir():
            continue
        
        char = char_dir.name
        char_output_dir = output_dir / char
        char_output_dir.mkdir(exist_ok=True)
        
        # 各画像を処理
        image_files = list(char_dir.glob('*.png'))
        
        for img_path in tqdm(image_files, desc=f"Processing {char}"):
            image = cv2.imread(str(img_path))
            
            # バリエーションを生成
            variations = augmentor.augment_single_image(image, num_variations=5)
            
            # 保存
            base_name = img_path.stem
            for i, var in enumerate(variations):
                output_path = char_output_dir / f"{base_name}_enhanced_{i:03d}.png"
                cv2.imwrite(str(output_path), var)


def visualize_augmentations():
    """拡張結果を可視化"""
    augmentor = EnhancedDataAugmentor()
    
    # テスト画像を読み込み
    test_images = [
        Path("training_data/augmented/P/P_original_000.png"),
        Path("training_data/augmented/L/L_original_000.png"),
        Path("training_data/augmented/E/E_original_000.png"),
    ]
    
    fig, axes = plt.subplots(len(test_images), 6, figsize=(15, 3*len(test_images)))
    
    for i, img_path in enumerate(test_images):
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        
        # オリジナル
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Original {img_path.parent.name}')
        axes[i, 0].axis('off')
        
        # バリエーション
        variations = augmentor.augment_single_image(image, num_variations=5)
        for j, var in enumerate(variations):
            axes[i, j+1].imshow(var, cmap='gray')
            axes[i, j+1].set_title(f'Variation {j+1}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/data_augmentation_examples.png')
    print("拡張例を保存: results/data_augmentation_examples.png")


def main():
    print("=== データ拡張パイプラインの強化 ===")
    
    # 拡張例を可視化
    print("\n拡張例を生成中...")
    visualize_augmentations()
    
    # 訓練データ全体を拡張
    print("\n訓練データを拡張中...")
    input_dir = Path("training_data/augmented")
    output_dir = Path("training_data/enhanced")
    
    augment_training_data(input_dir, output_dir)
    
    print("\nデータ拡張完了！")
    print(f"拡張データ保存先: {output_dir}")


if __name__ == "__main__":
    main()