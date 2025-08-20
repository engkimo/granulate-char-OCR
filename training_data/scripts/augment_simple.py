"""
シンプルなデータ拡張スクリプト（PyTorch不要）
"""
import cv2
import numpy as np
from pathlib import Path
import random
from typing import List


class SimpleAugmentor:
    """シンプルな文字拡張クラス（PyTorch不要）"""
    
    def __init__(self, base_dir: str = "training_data/extracted"):
        self.base_dir = Path(base_dir)
        
    def augment_image(self, image: np.ndarray, num_augmented: int = 50) -> List[np.ndarray]:
        """画像を拡張"""
        augmented_images = []
        h, w = image.shape[:2]
        
        for _ in range(num_augmented):
            aug_img = image.copy()
            
            # 1. 回転（-15度〜+15度）
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=0)
            
            # 2. スケール変更（0.8〜1.2倍）
            scale = random.uniform(0.8, 1.2)
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            scaled = cv2.resize(aug_img, (scaled_w, scaled_h))
            
            # サイズを元に戻す
            if scale > 1:
                # 中央をクロップ
                y_start = (scaled_h - h) // 2
                x_start = (scaled_w - w) // 2
                aug_img = scaled[y_start:y_start+h, x_start:x_start+w]
            else:
                # パディング
                canvas = np.zeros((h, w), dtype=np.uint8)
                y_start = (h - scaled_h) // 2
                x_start = (w - scaled_w) // 2
                canvas[y_start:y_start+scaled_h, x_start:x_start+scaled_w] = scaled
                aug_img = canvas
            
            # 3. 平行移動（-5〜+5ピクセル）
            tx = random.randint(-5, 5)
            ty = random.randint(-5, 5)
            M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M_translate, (w, h), borderValue=0)
            
            # 4. ノイズ追加
            noise_level = random.uniform(0, 0.03)
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, aug_img.shape)
                aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)
            
            # 5. 明度・コントラスト調整
            alpha = random.uniform(0.8, 1.2)  # コントラスト
            beta = random.randint(-20, 20)    # 明度
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
            
            # 6. ぼかし（たまに）
            if random.random() > 0.8:
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            
            # 7. モルフォロジー変換（たまに）
            if random.random() > 0.8:
                kernel = np.ones((2, 2), np.uint8)
                if random.random() > 0.5:
                    aug_img = cv2.erode(aug_img, kernel, iterations=1)
                else:
                    aug_img = cv2.dilate(aug_img, kernel, iterations=1)
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def process_all_characters(self, output_dir: str = "training_data/augmented"):
        """全文字に対してデータ拡張を実行"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_generated = 0
        
        # 各文字ディレクトリを処理
        for char_dir in sorted(self.base_dir.iterdir()):
            if not char_dir.is_dir() or char_dir.name == "debug":
                continue
            
            char_label = char_dir.name
            print(f"\n処理中: {char_label}")
            
            # オリジナル画像を読み込み
            original_images = []
            for img_path in char_dir.glob("*_reference.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    original_images.append(img)
            
            if not original_images:
                print(f"  警告: {char_label} の画像が見つかりません")
                continue
            
            # 出力ディレクトリを作成
            char_output_dir = output_path / char_label
            char_output_dir.mkdir(exist_ok=True)
            
            # オリジナルをコピー
            for i, img in enumerate(original_images):
                cv2.imwrite(str(char_output_dir / f"{char_label}_original_{i:03d}.png"), img)
            
            # データ拡張
            print(f"  データ拡張を実行中...")
            for i, base_img in enumerate(original_images):
                augmented = self.augment_image(base_img, num_augmented=150)  # 各画像から150個生成
                for j, aug_img in enumerate(augmented):
                    filename = f"{char_label}_aug_{i:02d}_{j:03d}.png"
                    cv2.imwrite(str(char_output_dir / filename), aug_img)
            
            # 統計情報
            total_images = len(list(char_output_dir.glob("*.png")))
            total_generated += total_images
            print(f"  完了: {char_label} - 画像数: {total_images}")
        
        print(f"\nデータ拡張完了！")
        print(f"総生成画像数: {total_generated}")
        print(f"1文字あたり平均: {total_generated / 26:.0f}枚")


if __name__ == "__main__":
    augmentor = SimpleAugmentor()
    augmentor.process_all_characters()