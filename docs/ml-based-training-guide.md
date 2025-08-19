# 最新機械学習手法を用いたグラニュート文字OCR学習ガイド

早見表画像から限られたデータで高精度なOCRモデルを構築するための最新手法を解説します。

## 目次

1. [概要](#概要)
2. [早見表からの文字抽出](#早見表からの文字抽出)
3. [データ拡張手法](#データ拡張手法)
4. [Few-shot Learning](#few-shot-learning)
5. [実装手順](#実装手順)
6. [パフォーマンス最適化](#パフォーマンス最適化)

## 概要

### 課題
- グラニュート文字の学習データが早見表1枚のみ
- 各文字につき1サンプルしかない（26文字）
- 実世界での多様な撮影条件に対応する必要がある

### 解決アプローチ
1. **画像処理による文字抽出** - 早見表から各文字を正確に切り出し
2. **高度なデータ拡張** - GAN/Diffusion Modelを使用した合成データ生成
3. **Few-shot Learning** - 少数データでの効率的な学習
4. **転移学習** - 事前学習済みモデルの活用

## 早見表からの文字抽出

### 1. 文字領域の自動検出

```python
# training_data/scripts/extract_from_reference.py の実行
python training_data/scripts/extract_from_reference.py
```

#### 処理フロー
1. HSV色空間での紫色バブル検出
2. 輪郭検出による個別バブルの抽出
3. 各バブル内の白色文字領域の検出
4. 64x64ピクセルへの正規化

#### 期待される出力
```
training_data/extracted/
├── A/
│   ├── A_reference.png
│   └── A_preview.png
├── B/
│   ├── B_reference.png
│   └── B_preview.png
└── ... (全26文字)
```

### 2. 抽出品質の確認

```python
# 抽出結果の視覚化
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def visualize_extracted_chars():
    extracted_dir = Path("training_data/extracted")
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, char_dir in enumerate(sorted(extracted_dir.iterdir())[:26]):
        if char_dir.is_dir():
            img_path = char_dir / f"{char_dir.name}_reference.png"
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"Character: {char_dir.name}")
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("training_data/extracted_characters_grid.png")
    plt.show()

visualize_extracted_chars()
```

## データ拡張手法

### 1. 従来型データ拡張

#### 実装する変換
- **幾何学的変換**
  - 回転: ±15度
  - スケーリング: 0.8-1.2倍
  - せん断変形
  - 弾性変形

- **ピクセルレベル変換**
  - 明度・コントラスト調整
  - ガウシアンノイズ
  - モーションブラー
  - ソルト&ペッパーノイズ

- **形態学的変換**
  - エロージョン（収縮）
  - ダイレーション（膨張）
  - オープニング・クロージング

### 2. GAN/Diffusion Modelによる拡張

#### StyleGAN2ベースの文字生成

```python
# training_data/scripts/augment_with_gan.py の実行
python training_data/scripts/augment_with_gan.py
```

特徴：
- 潜在空間からの新しい文字バリエーション生成
- スタイル混合による多様性確保
- 文字の基本構造を保持しながら変化を加える

#### 簡易Diffusion Model

実装のポイント：
- ノイズ追加→除去プロセスの模擬
- 段階的な品質向上
- 文字のエッジ保持

### 3. 実行結果

期待される生成データ数：
- 各文字あたり約150枚
- 総計: 26文字 × 150枚 = 3,900枚

## Few-shot Learning

### 1. Prototypical Networks

#### アーキテクチャ
```python
Encoder: Conv2d → BatchNorm → ReLU → MaxPool (×4)
         ↓
Global Average Pooling
         ↓
64次元埋め込みベクトル
```

#### 学習プロセス
1. サポートセット（各クラス5枚）から特徴抽出
2. クラスごとのプロトタイプ（平均ベクトル）計算
3. クエリサンプルとプロトタイプ間の距離計算
4. 最近傍分類

### 2. Siamese Networks

#### 用途
- 文字ペアの類似度学習
- ワンショット認識
- 新規文字への汎化

### 3. MAML (Model-Agnostic Meta-Learning)

#### 特徴
- タスク適応的な初期パラメータ学習
- 少数サンプルでの高速適応
- グラニュート文字の構造的特徴を活用

## 実装手順

### 1. 環境構築

```bash
# 必要なライブラリをインストール
pip install torch torchvision opencv-python scikit-learn matplotlib

# CUDA対応（GPU使用時）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. データ準備

```bash
# ディレクトリ構造を作成
mkdir -p training_data/{scripts,extracted,augmented,models}
mkdir -p models

# 早見表から文字を抽出
python training_data/scripts/extract_from_reference.py

# データ拡張を実行
python training_data/scripts/augment_with_gan.py
```

### 3. Few-shot学習

```bash
# Few-shot modelの学習
python training_data/scripts/few_shot_learning.py
```

### 4. Tesseractへの統合

```python
# Few-shotモデルの特徴をTesseract学習データに変換
import torch
import numpy as np
from pathlib import Path

def export_to_tesseract_format():
    # Few-shotモデルをロード
    model = torch.load("models/prototypical_network.pth")
    
    # 各文字の特徴を抽出
    features = {}
    for char_dir in Path("training_data/augmented").iterdir():
        if char_dir.is_dir():
            char_features = extract_character_features(model, char_dir)
            features[char_dir.name] = char_features
    
    # Tesseract形式に変換
    create_tesseract_training_data(features)
```

## パフォーマンス最適化

### 1. データ効率の向上

#### Mixup拡張
```python
def mixup_augmentation(img1, img2, alpha=0.2):
    lambda_param = np.random.beta(alpha, alpha)
    mixed = lambda_param * img1 + (1 - lambda_param) * img2
    return mixed
```

#### CutMix
```python
def cutmix_augmentation(img1, img2):
    h, w = img1.shape
    cut_size = int(h * 0.3)
    
    # ランダムな位置を選択
    x = np.random.randint(0, w - cut_size)
    y = np.random.randint(0, h - cut_size)
    
    # 画像を合成
    mixed = img1.copy()
    mixed[y:y+cut_size, x:x+cut_size] = img2[y:y+cut_size, x:x+cut_size]
    return mixed
```

### 2. アンサンブル学習

```python
class EnsembleOCR:
    def __init__(self):
        self.models = [
            PrototypicalNetwork(),
            SiameseNetwork(),
            TesseractOCR(),
            TransformerOCR()
        ]
    
    def predict(self, image):
        predictions = []
        confidences = []
        
        for model in self.models:
            pred, conf = model.predict(image)
            predictions.append(pred)
            confidences.append(conf)
        
        # 重み付き投票
        return weighted_vote(predictions, confidences)
```

### 3. 後処理の最適化

#### 文字の構造的制約を利用
```python
def apply_structural_constraints(predictions):
    """
    グラニュート文字の構造的特徴を利用した後処理
    """
    # 文字の連続性チェック
    # 文字間の類似度計算
    # 文脈を考慮した修正
    return refined_predictions
```

## 評価とデバッグ

### 1. 評価メトリクス

```python
def evaluate_model(model, test_loader):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrix': None
    }
    
    # 各メトリクスを計算
    # ...
    
    return metrics
```

### 2. エラー分析

```python
def analyze_errors(predictions, ground_truth):
    """
    誤認識パターンの分析
    """
    error_patterns = defaultdict(list)
    
    for pred, gt in zip(predictions, ground_truth):
        if pred != gt:
            error_patterns[gt].append(pred)
    
    # 混同しやすい文字ペアを特定
    confusion_pairs = identify_confusion_pairs(error_patterns)
    
    return confusion_pairs
```

## まとめ

### 推奨アプローチ

1. **初期段階**
   - 早見表からの文字抽出
   - 基本的なデータ拡張（100-200枚/文字）
   - シンプルなCNNでベースライン構築

2. **改善段階**
   - Few-shot学習の導入
   - GAN/Diffusionによる高度な拡張
   - アンサンブル手法の適用

3. **実用化段階**
   - リアルタイムOCRの最適化
   - エッジデバイスへの展開
   - 継続的な学習システムの構築

### 期待される性能

- **初期モデル**: 70-80%精度
- **Few-shot適用後**: 85-90%精度
- **完全なパイプライン**: 90-95%精度

### 次のステップ

1. 実際のカメラ画像での評価
2. リアルタイム処理の実装
3. モバイルアプリへの統合
