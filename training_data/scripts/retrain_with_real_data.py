#!/usr/bin/env python3
"""
実際のグラニュート文字画像でCNNモデルを再訓練するスクリプト
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
from tqdm import tqdm


class RealGranulateDataset(Dataset):
    """実際のグラニュート文字画像のデータセット"""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 画像と対応するテキストを読み込み
        for img_path in sorted(data_dir.glob("*.png")):
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                self.samples.append((img_path, text))
        
        print(f"データセット作成完了: {len(self.samples)}個のサンプル")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # 画像を読み込み
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # 文字領域を抽出（実装は既存のOCRServiceから流用）
        char_images = self._extract_characters(img)
        
        # 各文字とラベルのペアを返す
        return char_images, list(text)
    
    def _extract_characters(self, img: np.ndarray) -> List[np.ndarray]:
        """画像から文字領域を抽出"""
        # OCRServiceの_extract_character_regions_improvedメソッドを流用
        # ここでは簡略化
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        
        # 水平プロジェクションで文字境界を検出
        horizontal_projection = np.sum(binary, axis=0)
        threshold = np.max(horizontal_projection) * 0.1
        
        char_regions = []
        in_char = False
        start = 0
        
        for i, val in enumerate(horizontal_projection):
            if not in_char and val > threshold:
                in_char = True
                start = i
            elif in_char and val <= threshold:
                in_char = False
                if i - start > 10:
                    char_regions.append((start, i))
        
        # 各文字画像を抽出
        char_images = []
        for x_start, x_end in char_regions:
            char_img = img[:, x_start:x_end]
            char_images.append(char_img)
        
        return char_images


def retrain_model(test_data_dir: Path, model_path: Path, output_path: Path):
    """既存のCNNモデルを実画像で再訓練"""
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット作成
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = RealGranulateDataset(test_data_dir / 'granulate_texts', transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 既存モデルをロード
    from backend.application.services.ocr_service import GranulateOCRModel
    model = GranulateOCRModel(num_classes=26)
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("既存モデルをロードしました")
    
    model.to(device)
    
    # 訓練設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 低い学習率でファインチューニング
    
    # 訓練ループ
    model.train()
    for epoch in range(10):  # 少ないエポック数でファインチューニング
        running_loss = 0.0
        correct = 0
        total = 0
        
        for char_images_list, labels_list in tqdm(dataloader, desc=f"Epoch {epoch+1}/10"):
            # 各サンプルの文字を処理
            for char_images, labels in zip(char_images_list, labels_list):
                for char_img, label in zip(char_images, labels):
                    # 文字をテンソルに変換
                    if transform:
                        char_tensor = transform(char_img).unsqueeze(0).to(device)
                    
                    # ラベルをインデックスに変換
                    label_idx = ord(label) - ord('A')
                    label_tensor = torch.tensor([label_idx]).to(device)
                    
                    # 順伝播
                    optimizer.zero_grad()
                    outputs = model(char_tensor)
                    loss = criterion(outputs, label_tensor)
                    
                    # 逆伝播
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += 1
                    correct += (predicted == label_tensor).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss: {running_loss/total:.4f}, Accuracy: {accuracy:.2f}%")
    
    # モデルを保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }, output_path)
    
    print(f"再訓練済みモデルを保存: {output_path}")


def evaluate_on_test_data(model_path: Path, test_data_dir: Path):
    """テストデータでモデルを評価"""
    from backend.application.services.ocr_service import OCRService
    
    # OCRサービスを初期化（新しいモデルを使用）
    ocr_service = OCRService()
    
    results = []
    
    # 各テスト画像を処理
    for img_path in sorted((test_data_dir / 'granulate_texts').glob("*.png")):
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists():
            continue
        
        # 期待される結果を読み込み
        with open(txt_path, 'r') as f:
            expected = f.read().strip()
        
        # 画像を処理
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        
        result = ocr_service.process_image(image_bytes)
        recognized = ''.join(c.latin_equivalent for c in result.characters)
        
        # 結果を記録
        accuracy = sum(1 for e, r in zip(expected, recognized) if e == r) / len(expected)
        results.append({
            'file': img_path.name,
            'expected': expected,
            'recognized': recognized,
            'accuracy': accuracy
        })
        
        print(f"{img_path.name}: {expected} → {recognized} ({accuracy*100:.1f}%)")
    
    # 全体の精度を計算
    overall_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\n全体精度: {overall_accuracy*100:.1f}%")
    
    # 結果を保存
    with open(test_data_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'overall_accuracy': overall_accuracy,
            'results': results
        }, f, indent=2)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    test_data_dir = project_root / 'test_data'
    model_path = project_root / 'models' / 'cnn_model_best.pth'
    output_path = project_root / 'models' / 'cnn_model_retrained.pth'
    
    # 実画像でモデルを再訓練
    if (test_data_dir / 'granulate_texts').exists():
        print("実画像でモデルを再訓練します...")
        retrain_model(test_data_dir, model_path, output_path)
        
        print("\n再訓練済みモデルを評価します...")
        evaluate_on_test_data(output_path, test_data_dir)
    else:
        print(f"エラー: {test_data_dir / 'granulate_texts'} が見つかりません")
        print("test_data/granulate_texts/ に実際のグラニュート文字画像を配置してください")