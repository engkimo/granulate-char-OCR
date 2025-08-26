#!/usr/bin/env python3
"""
CRNNモデルの評価スクリプト
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import json
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.crnn_model import CRNN, CTCLabelConverter, create_crnn_model
from scripts.train_crnn import GranulateTextDataset


def evaluate_crnn(model_path: Path, data_dir: Path, device: str = 'cpu') -> Dict:
    """CRNNモデルを評価
    
    Args:
        model_path: 訓練済みモデルのパス
        data_dir: テストデータのディレクトリ
        device: 使用デバイス
        
    Returns:
        results: 評価結果の辞書
    """
    # モデルをロード
    model = create_crnn_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # ラベル変換器
    converter = CTCLabelConverter()
    
    # テストデータを準備
    test_images = sorted(data_dir.glob("*_*.png"))
    
    results = {
        'predictions': [],
        'char_accuracy': 0,
        'word_accuracy': 0,
        'error_analysis': defaultdict(int)
    }
    
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0
    
    # 各画像を評価
    for img_path in tqdm(test_images, desc="評価中"):
        # 期待される出力
        expected = img_path.stem.split('_')[0].replace('!', '').replace('.', '')
        
        if not expected.isalpha() or not expected.isupper():
            continue
        
        # 画像を読み込み
        image = cv2.imread(str(img_path))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 前処理
        image = preprocess_for_crnn(image)
        
        # テンソルに変換
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(device)
        
        # 予測
        with torch.no_grad():
            output = model(image_tensor)
            _, preds = output.max(2)
            
            # デコード
            pred_text = converter.decode(preds)[0]
        
        # 結果を記録
        results['predictions'].append({
            'file': img_path.name,
            'expected': expected,
            'predicted': pred_text,
            'correct': pred_text == expected
        })
        
        # 精度計算
        total_words += 1
        if pred_text == expected:
            correct_words += 1
        
        # 文字レベルの精度
        for i, (e, p) in enumerate(zip(expected, pred_text)):
            total_chars += 1
            if e == p:
                correct_chars += 1
            else:
                results['error_analysis'][f"{e}→{p}"] += 1
        
        # 長さが違う場合
        if len(expected) != len(pred_text):
            results['error_analysis'][f"長さ: {len(expected)}→{len(pred_text)}"] += 1
    
    # 全体の精度を計算
    results['char_accuracy'] = correct_chars / total_chars if total_chars > 0 else 0
    results['word_accuracy'] = correct_words / total_words if total_words > 0 else 0
    results['total_words'] = total_words
    results['total_chars'] = total_chars
    
    return results


def preprocess_for_crnn(image: np.ndarray, target_height: int = 64, max_width: int = 256) -> np.ndarray:
    """CRNN用の前処理"""
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
    
    # リサイズ（アスペクト比を保持）
    h, w = image.shape
    aspect_ratio = w / h
    
    new_height = target_height
    new_width = int(target_height * aspect_ratio)
    
    # 最大幅を超える場合は調整
    if new_width > max_width:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    
    image = cv2.resize(image, (new_width, new_height))
    
    # パディング（左寄せ）
    padded = np.zeros((target_height, max_width), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    padded[y_offset:y_offset+new_height, :new_width] = image
    
    # 正規化
    padded = padded.astype(np.float32) / 255.0
    
    return padded


def visualize_predictions(results: Dict, output_path: Path, num_samples: int = 20):
    """予測結果を可視化"""
    predictions = results['predictions']
    
    # 成功例と失敗例を分ける
    correct = [p for p in predictions if p['correct']]
    incorrect = [p for p in predictions if not p['correct']]
    
    # サンプルを選択
    samples = []
    if len(incorrect) > 0:
        samples.extend(incorrect[:min(num_samples//2, len(incorrect))])
    if len(correct) > 0:
        samples.extend(correct[:min(num_samples//2, len(correct))])
    
    # プロット
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 2*len(samples)))
    if len(samples) == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        # 画像を読み込み
        img_path = Path("test_data") / sample['file']
        image = cv2.imread(str(img_path))
        
        # 表示
        axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Expected: {sample['expected']} | Predicted: {sample['predicted']} | {'✓' if sample['correct'] else '✗'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"予測結果の可視化を保存: {output_path}")


def main():
    model_path = Path("models/crnn_model_best.pth")
    data_dir = Path("test_data")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print("=== CRNNモデルの評価 ===")
    
    # モデルが存在しない場合はスキップ
    if not model_path.exists():
        print(f"モデルが見つかりません: {model_path}")
        print("先に train_crnn.py を実行してモデルを訓練してください。")
        return
    
    # 評価を実行
    results = evaluate_crnn(model_path, data_dir)
    
    # 結果を表示
    print(f"\n=== 評価結果 ===")
    print(f"単語レベル精度: {results['word_accuracy']*100:.1f}% ({results['word_accuracy']*results['total_words']:.0f}/{results['total_words']})")
    print(f"文字レベル精度: {results['char_accuracy']*100:.1f}% ({results['char_accuracy']*results['total_chars']:.0f}/{results['total_chars']})")
    
    # エラー分析
    print(f"\n=== よくあるエラー (上位10件) ===")
    error_items = sorted(results['error_analysis'].items(), key=lambda x: x[1], reverse=True)[:10]
    for error, count in error_items:
        print(f"{error}: {count}回")
    
    # 結果をJSONに保存
    output_json = output_dir / "crnn_evaluation_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n評価結果を保存: {output_json}")
    
    # 予測結果を可視化
    visualize_predictions(results, output_dir / "crnn_predictions.png")
    
    # CNNモデルとの比較
    print(f"\n=== モデル比較 ===")
    print(f"CNN (文字分割): 9.1% (文字レベル)")
    print(f"CRNN (エンドツーエンド): {results['char_accuracy']*100:.1f}% (文字レベル), {results['word_accuracy']*100:.1f}% (単語レベル)")


if __name__ == "__main__":
    main()