import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import Dict, List, Tuple
import random
from collections import defaultdict
import json


class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for Few-shot Learning"""
    
    def __init__(self, in_channels=1, hidden_dim=64, out_dim=64):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x
    
    def compute_prototypes(self, support_embeddings, support_labels):
        """各クラスのプロトタイプを計算"""
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes[label.item()] = prototype
        
        return prototypes
    
    def euclidean_distance(self, query_embeddings, prototypes):
        """ユークリッド距離を計算"""
        n_queries = query_embeddings.size(0)
        n_prototypes = len(prototypes)
        
        distances = torch.zeros(n_queries, n_prototypes)
        
        for i, (label, prototype) in enumerate(prototypes.items()):
            distances[:, i] = torch.norm(query_embeddings - prototype, dim=1)
        
        return distances


class SiameseNetwork(nn.Module):
    """Siamese Network for One-shot Learning"""
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
    def forward_once(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML)"""
    
    def __init__(self, base_model, inner_lr=0.01, first_order=True):
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.first_order = first_order
        
    def inner_loop(self, support_x, support_y, num_steps=5):
        """タスク固有の適応"""
        params = list(self.base_model.parameters())
        
        for _ in range(num_steps):
            support_logits = self.base_model(support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # 勾配を計算
            grads = torch.autograd.grad(
                support_loss, params,
                create_graph=not self.first_order
            )
            
            # パラメータを更新
            params = [p - self.inner_lr * g for p, g in zip(params, grads)]
            
            # モデルに新しいパラメータを設定
            for p, new_p in zip(self.base_model.parameters(), params):
                p.data = new_p.data
        
        return params
    
    def forward(self, support_x, support_y, query_x):
        # タスク固有の適応
        adapted_params = self.inner_loop(support_x, support_y)
        
        # クエリセットで評価
        query_logits = self.base_model(query_x)
        
        return query_logits


class GranulateCharacterFewShotDataset(Dataset):
    """Few-shot学習用データセット"""
    
    def __init__(self, data_dir: str, n_way: int = 5, k_shot: int = 5, n_query: int = 10):
        self.data_dir = Path(data_dir)
        self.n_way = n_way  # クラス数
        self.k_shot = k_shot  # サポートセットのサンプル数
        self.n_query = n_query  # クエリセットのサンプル数
        
        # クラスごとに画像を整理
        self.class_images = defaultdict(list)
        for char_dir in sorted(self.data_dir.iterdir()):
            if char_dir.is_dir():
                char_label = char_dir.name
                for img_path in char_dir.glob("*.png"):
                    self.class_images[char_label].append(str(img_path))
        
        self.classes = list(self.class_images.keys())
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return 1000  # エポックあたりのタスク数
    
    def __getitem__(self, idx):
        # N-way K-shotタスクを生成
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for i, class_name in enumerate(selected_classes):
            # サポートセットとクエリセットを選択
            selected_images = random.sample(
                self.class_images[class_name], 
                self.k_shot + self.n_query
            )
            
            # サポートセット
            for j in range(self.k_shot):
                img = Image.open(selected_images[j]).convert('L')
                img_tensor = self.transform(img)
                support_images.append(img_tensor)
                support_labels.append(i)
            
            # クエリセット
            for j in range(self.k_shot, self.k_shot + self.n_query):
                img = Image.open(selected_images[j]).convert('L')
                img_tensor = self.transform(img)
                query_images.append(img_tensor)
                query_labels.append(i)
        
        # Tensorに変換
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }


class FewShotTrainer:
    """Few-shot学習トレーナー"""
    
    def __init__(self, model, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train_prototypical(self, dataloader, num_epochs=100):
        """Prototypical Networksの学習"""
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            
            for batch in dataloader:
                # バッチサイズが1の場合、最初の次元を除去
                support_images = batch['support_images'].squeeze(0).to(self.device)
                support_labels = batch['support_labels'].squeeze(0).to(self.device)
                query_images = batch['query_images'].squeeze(0).to(self.device)
                query_labels = batch['query_labels'].squeeze(0).to(self.device)
                
                # 埋め込みを計算
                support_embeddings = self.model(support_images)
                query_embeddings = self.model(query_images)
                
                # プロトタイプを計算
                prototypes = self.model.compute_prototypes(
                    support_embeddings, support_labels
                )
                
                # 距離を計算
                distances = self.model.euclidean_distance(
                    query_embeddings, prototypes
                )
                
                # 損失を計算
                log_probs = F.log_softmax(-distances, dim=1)
                loss = F.nll_loss(log_probs, query_labels)
                
                # 精度を計算
                predictions = torch.argmax(log_probs, dim=1)
                accuracy = (predictions == query_labels).float().mean()
                
                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_acc += accuracy.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_acc = total_acc / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    def train_siamese(self, dataloader, num_epochs=100):
        """Siamese Networkの学習"""
        self.model.train()
        criterion = nn.BCELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                # ペアを作成
                images = batch['support_images'].to(self.device)
                labels = batch['support_labels'].to(self.device)
                
                # 正のペアと負のペアを作成
                pairs, pair_labels = self._create_pairs(images, labels)
                
                # フォワードパス
                out1, out2 = self.model(pairs[:, 0], pairs[:, 1])
                
                # コサイン類似度
                similarity = F.cosine_similarity(out1, out2)
                
                # 損失を計算
                loss = criterion(similarity, pair_labels.float())
                
                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def _create_pairs(self, images, labels):
        """正負のペアを作成"""
        n = len(images)
        pairs = []
        pair_labels = []
        
        # 各画像に対して
        for i in range(n):
            # 正のペア（同じクラス）
            same_class_indices = (labels == labels[i]).nonzero().squeeze()
            if len(same_class_indices) > 1:
                j = random.choice(same_class_indices[same_class_indices != i])
                pairs.append([images[i], images[j]])
                pair_labels.append(1)
            
            # 負のペア（違うクラス）
            diff_class_indices = (labels != labels[i]).nonzero().squeeze()
            if len(diff_class_indices) > 0:
                j = random.choice(diff_class_indices)
                pairs.append([images[i], images[j]])
                pair_labels.append(0)
        
        return torch.stack(pairs), torch.tensor(pair_labels)


def create_combined_model():
    """統合モデルの作成"""
    
    class CombinedFewShotModel(nn.Module):
        """複数のFew-shot手法を統合したモデル"""
        
        def __init__(self):
            super().__init__()
            self.prototypical = PrototypicalNetwork()
            self.siamese = SiameseNetwork()
            
            # 統合用の全結合層
            self.fusion = nn.Sequential(
                nn.Linear(128, 64),  # 両モデルの出力を結合
                nn.ReLU(),
                nn.Linear(64, 36)  # 36クラス（A-Z + 0-9）
            )
        
        def forward(self, x, mode='inference'):
            if mode == 'prototypical':
                return self.prototypical(x)
            elif mode == 'siamese':
                return self.siamese.forward_once(x)
            else:
                # 両モデルの出力を統合
                proto_out = self.prototypical(x)
                siamese_out = self.siamese.forward_once(x)
                combined = torch.cat([proto_out, siamese_out], dim=1)
                return self.fusion(combined)
    
    return CombinedFewShotModel()


if __name__ == "__main__":
    # データセットを作成
    dataset = GranulateCharacterFewShotDataset(
        data_dir="training_data/augmented",
        n_way=5,
        k_shot=5,
        n_query=10
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Prototypical Networksの学習
    print("Training Prototypical Networks...")
    proto_model = PrototypicalNetwork()
    # CPUで実行（MPSのバグを回避）
    proto_trainer = FewShotTrainer(proto_model, device='cpu')
    proto_trainer.train_prototypical(dataloader, num_epochs=2)  # デモ用に少ないエポック数
    
    # モデルを保存
    torch.save(proto_model.state_dict(), "models/prototypical_network.pth")
    
    print("\nFew-shot学習が完了しました。")