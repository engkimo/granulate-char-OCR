import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import List, Tuple
import random


class GranulateDataset(Dataset):
    """グラニュート文字データセット"""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 各文字ディレクトリから画像を収集
        for char_dir in sorted(self.data_dir.iterdir()):
            if char_dir.is_dir():
                char_label = char_dir.name
                for img_path in char_dir.glob("*.png"):
                    if "preview" not in str(img_path):
                        self.images.append(str(img_path))
                        self.labels.append(char_label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 画像を読み込み
        image = Image.open(img_path).convert('L')  # グレースケール
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class StyleGAN2Generator(nn.Module):
    """StyleGAN2風のジェネレーター（簡略版）"""
    
    def __init__(self, z_dim=512, w_dim=512, img_size=64):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_size = img_size
        
        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, w_dim)
        )
        
        # Synthesis network
        self.synthesis = nn.ModuleList([
            self._make_block(w_dim, 512, 4),
            self._make_block(512, 256, 8),
            self._make_block(256, 128, 16),
            self._make_block(128, 64, 32),
            self._make_block(64, 32, 64),
        ])
        
        self.to_rgb = nn.Conv2d(32, 1, 1)  # グレースケール出力
        
    def _make_block(self, in_channels, out_channels, size):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1) if size > 4 else nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, z):
        w = self.mapping(z)
        x = w.view(-1, self.w_dim, 1, 1)
        
        for block in self.synthesis:
            x = block(x)
        
        x = self.to_rgb(x)
        x = torch.tanh(x)
        
        return x


class CharacterAugmentor:
    """文字拡張クラス"""
    
    def __init__(self, base_dir: str = "training_data/extracted"):
        self.base_dir = Path(base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def traditional_augmentation(self, image: np.ndarray, num_augmented: int = 50) -> List[np.ndarray]:
        """伝統的なデータ拡張"""
        augmented_images = []
        
        for _ in range(num_augmented):
            aug_img = image.copy()
            
            # 1. 回転
            angle = random.uniform(-15, 15)
            center = (aug_img.shape[1] // 2, aug_img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))
            
            # 2. スケール変更
            scale = random.uniform(0.8, 1.2)
            scaled = cv2.resize(aug_img, None, fx=scale, fy=scale)
            
            # サイズ調整
            if scale > 1:
                # 中央をクロップ
                h, w = aug_img.shape[:2]
                y_start = (scaled.shape[0] - h) // 2
                x_start = (scaled.shape[1] - w) // 2
                aug_img = scaled[y_start:y_start+h, x_start:x_start+w]
            else:
                # パディング
                h, w = aug_img.shape[:2]
                pad_h = (h - scaled.shape[0]) // 2
                pad_w = (w - scaled.shape[1]) // 2
                aug_img = cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
                aug_img = cv2.resize(aug_img, (w, h))
            
            # 3. 変形
            if random.random() > 0.5:
                # アフィン変換
                pts1 = np.float32([[0, 0], [64, 0], [0, 64]])
                offset = 5
                pts2 = np.float32([
                    [random.uniform(-offset, offset), random.uniform(-offset, offset)],
                    [64 + random.uniform(-offset, offset), random.uniform(-offset, offset)],
                    [random.uniform(-offset, offset), 64 + random.uniform(-offset, offset)]
                ])
                M = cv2.getAffineTransform(pts1, pts2)
                aug_img = cv2.warpAffine(aug_img, M, (64, 64))
            
            # 4. ノイズ追加
            noise_level = random.uniform(0, 0.05)
            noise = np.random.normal(0, noise_level * 255, aug_img.shape)
            aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)
            
            # 5. 明度・コントラスト調整
            alpha = random.uniform(0.8, 1.2)  # コントラスト
            beta = random.uniform(-20, 20)    # 明度
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
            
            # 6. モルフォロジー変換
            if random.random() > 0.7:
                kernel_size = random.choice([3, 5])
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                if random.random() > 0.5:
                    aug_img = cv2.erode(aug_img, kernel, iterations=1)
                else:
                    aug_img = cv2.dilate(aug_img, kernel, iterations=1)
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def gan_augmentation(self, char_label: str, num_samples: int = 50):
        """GANを使用したデータ生成（シンプル版）"""
        # ジェネレーターを初期化
        generator = StyleGAN2Generator().to(self.device)
        
        # 事前学習済みモデルがあればロード（ここではスキップ）
        # generator.load_state_dict(torch.load(f"models/gan_{char_label}.pth"))
        
        generator.eval()
        generated_images = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn(1, generator.z_dim).to(self.device)
                fake_img = generator(z)
                
                # numpy配列に変換
                img = fake_img.squeeze().cpu().numpy()
                img = ((img + 1) * 127.5).astype(np.uint8)
                
                generated_images.append(img)
        
        return generated_images
    
    def diffusion_augmentation(self, base_image: np.ndarray, num_samples: int = 30):
        """拡散モデルを使用したデータ生成（シンプルな実装）"""
        generated_images = []
        
        for _ in range(num_samples):
            # ノイズを段階的に追加してから除去するプロセスをシミュレート
            noisy_img = base_image.copy().astype(np.float32)
            
            # Forward process (ノイズ追加)
            noise_level = random.uniform(0.1, 0.3)
            noise = np.random.normal(0, noise_level * 255, noisy_img.shape)
            noisy_img = noisy_img + noise
            
            # Reverse process (ノイズ除去 - 簡略化)
            # 実際のDDPMではニューラルネットワークを使用
            denoised = cv2.bilateralFilter(noisy_img.astype(np.uint8), 9, 75, 75)
            
            # ランダムな変化を加える
            if random.random() > 0.5:
                kernel = np.ones((3, 3), np.uint8)
                if random.random() > 0.5:
                    denoised = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
            
            generated_images.append(denoised)
        
        return generated_images
    
    def process_all_characters(self, output_dir: str = "training_data/augmented"):
        """全文字に対してデータ拡張を実行"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 各文字ディレクトリを処理
        for char_dir in sorted(self.base_dir.iterdir()):
            if not char_dir.is_dir():
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
            
            # 1. 伝統的な拡張
            print(f"  伝統的拡張を実行中...")
            for i, base_img in enumerate(original_images):
                augmented = self.traditional_augmentation(base_img, num_augmented=100)
                for j, aug_img in enumerate(augmented):
                    filename = f"{char_label}_trad_{i:02d}_{j:03d}.png"
                    cv2.imwrite(str(char_output_dir / filename), aug_img)
            
            # 2. 拡散モデル風拡張
            print(f"  拡散モデル風拡張を実行中...")
            for i, base_img in enumerate(original_images):
                diffusion_augmented = self.diffusion_augmentation(base_img, num_samples=50)
                for j, aug_img in enumerate(diffusion_augmented):
                    filename = f"{char_label}_diff_{i:02d}_{j:03d}.png"
                    cv2.imwrite(str(char_output_dir / filename), aug_img)
            
            # 3. 統計情報
            total_images = len(list(char_output_dir.glob("*.png")))
            print(f"  完了: {char_label} - 総画像数: {total_images}")
        
        print("\nデータ拡張完了")


class MixupAugmentation:
    """文字間でのMixup拡張"""
    
    @staticmethod
    def mixup(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """二つの画像を混合"""
        lambda_param = np.random.beta(alpha, alpha)
        mixed = lambda_param * img1.astype(np.float32) + (1 - lambda_param) * img2.astype(np.float32)
        return np.clip(mixed, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # データ拡張を実行
    augmentor = CharacterAugmentor()
    augmentor.process_all_characters()
    
    print("\nデータ拡張が完了しました。")