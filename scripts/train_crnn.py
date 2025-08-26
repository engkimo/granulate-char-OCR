#!/usr/bin/env python3
"""
CRNNモデルの訓練スクリプト
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
from pathlib import Path
import sys
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.crnn_model import CRNN, CTCLabelConverter, create_crnn_model


class GranulateTextDataset(Dataset):
    """グラニュート文字の単語画像データセット"""
    
    def __init__(self, data_dir: Path, img_width: int = 256, img_height: int = 64, augment: bool = True):
        """
        Args:
            data_dir: テストデータのディレクトリ
            img_width: リサイズ後の画像幅
            img_height: リサイズ後の画像高さ
            augment: データ拡張を行うか
        """
        self.data_dir = data_dir
        self.img_width = img_width
        self.img_height = img_height
        self.augment = augment
        self.samples = []
        
        # 画像とラベルを収集
        for img_path in sorted(data_dir.glob("*_*.png")):
            label = img_path.stem.split('_')[0]
            # 特殊文字を除去
            label = label.replace('!', '').replace('.', '')
            
            # アルファベットのみ処理
            if label.isalpha() and label.isupper():
                self.samples.append((img_path, label))
        
        print(f"データセット作成完了: {len(self.samples)}個のサンプル")
        
        # 単語長の分布を表示
        word_lengths = [len(label) for _, label in self.samples]
        print(f"単語長の分布: 最小={min(word_lengths)}, 最大={max(word_lengths)}, 平均={np.mean(word_lengths):.1f}")
    
    def __len__(self):
        return len(self.samples) * (3 if self.augment else 1)
    
    def __getitem__(self, idx):
        # データ拡張を考慮したインデックス計算
        if self.augment:
            sample_idx = idx // 3
            aug_idx = idx % 3
        else:
            sample_idx = idx
            aug_idx = 0
        
        img_path, label = self.samples[sample_idx]
        
        # 画像を読み込み
        image = cv2.imread(str(img_path))
        
        # グレースケールに変換
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # データ拡張
        if aug_idx > 0:
            image = self._augment_image(image, aug_idx)
        
        # 前処理
        image = self._preprocess_image(image)
        
        # リサイズ（アスペクト比を保持）
        h, w = image.shape
        aspect_ratio = w / h
        
        if aspect_ratio > self.img_width / self.img_height:
            # 幅に合わせる
            new_width = self.img_width
            new_height = int(self.img_width / aspect_ratio)
        else:
            # 高さに合わせる
            new_height = self.img_height
            new_width = int(self.img_height * aspect_ratio)
        
        # リサイズ
        image = cv2.resize(image, (new_width, new_height))
        
        # パディング
        padded = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        y_offset = (self.img_height - new_height) // 2
        x_offset = 0  # 左寄せ
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image
        
        # 正規化とテンソル変換
        image = padded.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # チャンネル次元を追加
        
        return image, label, new_width
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """画像の前処理"""
        # 背景色を判定
        mean_val = np.mean(image)
        if mean_val > 128:
            # 白背景の場合は反転
            image = 255 - image
        
        # ノイズ除去
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # 二値化
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return image
    
    def _augment_image(self, image: np.ndarray, aug_idx: int) -> np.ndarray:
        """データ拡張"""
        if aug_idx == 1:
            # 軽い回転
            angle = random.uniform(-5, 5)
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        elif aug_idx == 2:
            # ノイズ追加
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image


def collate_fn(batch):
    """バッチ処理用のカスタム関数"""
    images, labels, widths = zip(*batch)
    
    # 画像をパディング（バッチ内で同じサイズにする必要がある）
    images = torch.stack(images)
    
    return images, labels, widths


def train_crnn(data_dir: Path, output_dir: Path, epochs: int = 50):
    """CRNNモデルを訓練"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)
    
    # データセット作成
    dataset = GranulateTextDataset(data_dir, augment=True)
    
    # 訓練用と検証用に分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # モデルとラベル変換器
    model = create_crnn_model().to(device)
    converter = CTCLabelConverter()
    
    # 損失関数（CTC Loss）
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 最適化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 訓練履歴
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0
    
    # 訓練ループ
    for epoch in range(epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, labels, widths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 訓練"):
            images = images.to(device)
            
            # ラベルをエンコード
            targets, target_lengths = converter.encode(labels)
            targets = targets.to(device)
            
            # 予測
            optimizer.zero_grad()
            outputs = model(images)  # (seq_len, batch, num_classes)
            
            # 入力系列の長さを計算
            input_lengths = model.get_input_lengths(images.size(0), images.size(3))
            
            # CTC Loss
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            # バックプロパゲーション
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels, widths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 検証"):
                images = images.to(device)
                
                # ラベルをエンコード
                targets, target_lengths = converter.encode(labels)
                targets = targets.to(device)
                
                # 予測
                outputs = model(images)
                
                # 入力系列の長さを計算
                input_lengths = model.get_input_lengths(images.size(0), images.size(3))
                
                # CTC Loss
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
                val_batches += 1
                
                # 精度計算
                _, preds = outputs.max(2)
                preds = preds.transpose(1, 0).contiguous()  # (batch, seq_len)
                
                # デコード
                pred_texts = converter.decode(preds.transpose(0, 1))
                
                for pred, label in zip(pred_texts, labels):
                    if pred == label:
                        val_correct += 1
                    val_total += 1
        
        val_loss /= val_batches
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # 学習率調整
        scheduler.step(val_loss)
        
        # 履歴を記録
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # ベストモデルを保存
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'history': history
            }, output_dir / 'crnn_model_best.pth')
            print(f"ベストモデルを保存しました (Val Accuracy: {val_accuracy:.4f})")
    
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
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig(output_dir / 'crnn_training_history.png')
    print(f"学習曲線を保存: {output_dir / 'crnn_training_history.png'}")
    
    return model, history


def main():
    data_dir = Path("test_data")
    output_dir = Path("models")
    
    print("=== CRNNモデルの訓練 ===")
    
    # モデルを訓練
    model, history = train_crnn(data_dir, output_dir, epochs=50)
    
    print(f"\n訓練完了！")
    print(f"最終検証精度: {history['val_accuracy'][-1]*100:.1f}%")


if __name__ == "__main__":
    main()