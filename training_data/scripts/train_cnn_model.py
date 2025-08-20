#!/usr/bin/env python3
"""
CNNモデルによるグラニュート文字認識の訓練
高精度なOCRのための深層学習モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class GranulateCharacterDataset(Dataset):
    """グラニュート文字のデータセット"""
    
    def __init__(self, data_dir: Path, transform=None, target_size=(64, 64)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
        # 各画像のパスとラベルを収集
        self.samples = []
        for char in self.classes:
            char_dir = data_dir / char
            if char_dir.exists():
                for img_path in char_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[char]))
        
        # データ拡張
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 画像を読み込み
        img = Image.open(img_path).convert('L')  # グレースケール
        
        # 変換を適用
        if self.transform:
            img = self.transform(img)
        
        return img, label


class GranulateOCRModel(nn.Module):
    """グラニュート文字認識用のCNNモデル"""
    
    def __init__(self, num_classes=26):
        super().__init__()
        
        # 特徴抽出層
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetTransferModel(nn.Module):
    """転移学習を使用したResNetベースのモデル"""
    
    def __init__(self, num_classes=26):
        super().__init__()
        
        # ResNet18をロード（事前学習済み）
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 最初の畳み込み層を1チャンネル入力用に変更
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 最終層を変更
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, device: str, model_name: str):
    """モデルの訓練"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 
                             'acc': train_correct/train_total})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学習率を調整
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f'models/{model_name}_best.pth')
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, 
                   idx_to_class: Dict[int, str]):
    """モデルの評価"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 混同行列を計算
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 分類レポート
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(all_labels, all_predictions, 
                                  target_names=classes, 
                                  output_dict=True)
    
    return cm, report


def plot_results(train_losses, train_accs, val_losses, val_accs, cm, classes, model_name):
    """結果をプロット"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 損失曲線
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 精度曲線
    axes[0, 1].plot(train_accs, label='Train Accuracy')
    axes[0, 1].plot(val_accs, label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 混同行列
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 各クラスの精度
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    axes[1, 1].bar(classes, class_accuracies)
    axes[1, 1].set_xlabel('Character')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Per-Class Accuracy')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_results.png')
    plt.close()


def main():
    # 設定
    batch_size = 32
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセットの準備
    data_dir = Path("training_data/augmented")
    dataset = GranulateCharacterDataset(data_dir)
    
    # 訓練/検証/テストに分割（70:15:15）
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # モデルディレクトリを作成
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. カスタムCNNモデル
    print("\n=== Training Custom CNN Model ===")
    cnn_model = GranulateOCRModel(num_classes=26).to(device)
    
    train_losses, train_accs, val_losses, val_accs = train_model(
        cnn_model, train_loader, val_loader, num_epochs, device, "cnn_model"
    )
    
    # 評価
    cm, report = evaluate_model(cnn_model, test_loader, device, dataset.idx_to_class)
    plot_results(train_losses, train_accs, val_losses, val_accs, cm, 
                dataset.classes, "cnn_model")
    
    # 結果を保存
    with open('models/cnn_model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nCustom CNN Test Accuracy: {report['accuracy']:.4f}")
    
    # 2. ResNet転移学習モデル（オプション）
    if False:  # 必要に応じて有効化
        print("\n=== Training ResNet Transfer Learning Model ===")
        resnet_model = ResNetTransferModel(num_classes=26).to(device)
        
        train_losses, train_accs, val_losses, val_accs = train_model(
            resnet_model, train_loader, val_loader, num_epochs, device, "resnet_model"
        )
        
        cm, report = evaluate_model(resnet_model, test_loader, device, dataset.idx_to_class)
        plot_results(train_losses, train_accs, val_losses, val_accs, cm, 
                    dataset.classes, "resnet_model")
        
        with open('models/resnet_model_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResNet Test Accuracy: {report['accuracy']:.4f}")
    
    print("\nTraining complete! Models saved in 'models/' directory.")


if __name__ == "__main__":
    main()