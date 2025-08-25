#!/usr/bin/env python3
"""
新しいテストデータでCNNモデルを再訓練
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from backend.application.services.ocr_service import GranulateOCRModel, OCRService


class RealGranulateDataset(Dataset):
    """実際のグラニュート文字画像のデータセット"""
    
    def __init__(self, data_dir: Path, transform=None, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.char_to_idx = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
        
        # OCRサービスのインスタンスを作成（文字分割に使用）
        self.ocr_service = OCRService()
        
        # 画像とラベルを収集
        for img_path in sorted(data_dir.glob("*_*.png")):
            label = img_path.stem.split('_')[0]
            # 特殊文字を除去
            label = label.replace('!', '').replace('.', '')
            
            # 有効な文字のみ処理
            if all(c in self.char_to_idx for c in label):
                self.samples.append((img_path, label))
        
        print(f"データセット作成完了: {len(self.samples)}個のサンプル")
        
        # 文字ごとのサンプル数を表示
        char_counts = {}
        for _, label in self.samples:
            for char in label:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        print("文字ごとのサンプル数:")
        for char in sorted(char_counts.keys()):
            print(f"  {char}: {char_counts[char]}個")
    
    def __len__(self):
        # データ拡張を考慮
        return len(self.samples) * (5 if self.augment else 1)
    
    def __getitem__(self, idx):
        # データ拡張を考慮したインデックス計算
        if self.augment:
            sample_idx = idx // 5
            aug_idx = idx % 5
        else:
            sample_idx = idx
            aug_idx = 0
        
        img_path, text = self.samples[sample_idx]
        
        # 画像を読み込み
        image = cv2.imread(str(img_path))
        
        # データ拡張
        if aug_idx > 0:
            image = self._augment_image(image, aug_idx)
        
        # 前処理
        preprocessed = self.ocr_service._preprocess_image(image)
        
        # 文字領域を抽出
        char_regions = self.ocr_service._extract_character_regions_improved(preprocessed)
        
        # 各文字画像とラベルを収集
        char_images = []
        labels = []
        
        for i, (x, y, w, h) in enumerate(char_regions):
            if i < len(text):
                char_img = preprocessed[y:y+h, x:x+w]
                
                # 画像をリサイズ
                char_img = cv2.resize(char_img, (64, 64))
                
                # テンソルに変換
                if self.transform:
                    char_img = self.transform(char_img)
                else:
                    char_img = torch.from_numpy(char_img).float() / 255.0
                    char_img = char_img.unsqueeze(0)  # チャンネル次元を追加
                
                char_images.append(char_img)
                labels.append(self.char_to_idx[text[i]])
        
        return char_images, labels
    
    def _augment_image(self, image: np.ndarray, aug_idx: int) -> np.ndarray:
        """画像のデータ拡張"""
        h, w = image.shape[:2]
        
        if aug_idx == 1:
            # 明度調整
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(0.7, 1.3)
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif aug_idx == 2:
            # ガウシアンノイズ
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        elif aug_idx == 3:
            # 軽い回転
            angle = np.random.uniform(-5, 5)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        elif aug_idx == 4:
            # コントラスト調整
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image


def custom_collate_fn(batch):
    """バッチ内の可変長データを処理"""
    all_images = []
    all_labels = []
    
    for char_images, labels in batch:
        all_images.extend(char_images)
        all_labels.extend(labels)
    
    if all_images:
        images_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        return images_tensor, labels_tensor
    else:
        return torch.empty(0, 1, 64, 64), torch.empty(0, dtype=torch.long)


def train_model(data_dir: Path, output_path: Path, epochs: int = 20):
    """モデルを訓練"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット作成
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = RealGranulateDataset(data_dir, transform=None, augment=True)
    
    # 訓練用と検証用に分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    # モデルの準備
    model = GranulateOCRModel(num_classes=26)
    
    # 既存のモデルがあれば読み込み
    existing_model_path = Path('models/cnn_model_best.pth')
    if existing_model_path.exists():
        checkpoint = torch.load(existing_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("既存モデルを読み込みました")
    
    model.to(device)
    
    # 損失関数と最適化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # 訓練履歴
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    # 訓練ループ
    for epoch in range(epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 訓練"):
            if len(images) == 0:
                continue
            
            images = images.to(device)
            labels = labels.to(device)
            
            # 正規化を適用
            images = (images - 0.5) / 0.5
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 検証"):
                if len(images) == 0:
                    continue
                
                images = images.to(device)
                labels = labels.to(device)
                
                # 正規化を適用
                images = (images - 0.5) / 0.5
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # 学習率調整
        scheduler.step(val_loss)
        
        # 履歴を記録
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'history': history
            }, output_path)
            print(f"ベストモデルを保存しました (Val Acc: {val_acc:.4f})")
    
    # 学習曲線をプロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig('training_history.png')
    print("学習曲線を保存: training_history.png")
    
    return model, history


def main():
    data_dir = Path("test_data")
    output_path = Path("models/cnn_model_retrained.pth")
    
    print("=== 新しいテストデータでCNNモデルを再訓練 ===")
    
    # モデルを訓練
    model, history = train_model(data_dir, output_path, epochs=20)
    
    print(f"\n訓練完了！")
    print(f"最終訓練精度: {history['train_acc'][-1]*100:.1f}%")
    print(f"最終検証精度: {history['val_acc'][-1]*100:.1f}%")
    print(f"モデルを保存: {output_path}")


if __name__ == "__main__":
    main()