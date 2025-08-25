#!/usr/bin/env python3
"""
新しいテストデータでOCRシステムの精度を評価
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from backend.application.services.ocr_service import OCRService
from training_data.scripts.preprocess_color_images import ColorAwarePreprocessor


def evaluate_current_system(test_data_dir: Path) -> Dict:
    """現在のOCRシステムでベースライン精度を測定"""
    ocr_service = OCRService()
    results = []
    
    # 各テスト画像を処理
    test_images = sorted(test_data_dir.glob("*_*.png"))
    print(f"テスト画像数: {len(test_images)}")
    
    for img_path in tqdm(test_images, desc="評価中"):
        # ファイル名から期待される結果を抽出
        expected = img_path.stem.split('_')[0]
        
        # 特殊文字を処理
        if expected == "HIRING!":
            expected = "HIRING"
        elif expected == "TEAM!":
            expected = "TEAM"
        elif expected == "WORLD!":
            expected = "WORLD"
        
        # 画像を処理
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        
        try:
            result = ocr_service.process_image(image_bytes)
            recognized = ''.join(c.latin_equivalent for c in result.characters)
            
            # 文字レベルの精度を計算
            correct_chars = sum(1 for e, r in zip(expected, recognized) if e == r)
            total_chars = max(len(expected), len(recognized))
            char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
            
            # 単語レベルの精度
            word_accuracy = 1.0 if recognized == expected else 0.0
            
            results.append({
                'file': img_path.name,
                'expected': expected,
                'recognized': recognized,
                'char_accuracy': char_accuracy,
                'word_accuracy': word_accuracy,
                'char_count': len(expected),
                'processing_time': result.processing_time
            })
            
        except Exception as e:
            print(f"エラー: {img_path.name} - {e}")
            results.append({
                'file': img_path.name,
                'expected': expected,
                'recognized': '',
                'char_accuracy': 0.0,
                'word_accuracy': 0.0,
                'char_count': len(expected),
                'processing_time': 0.0,
                'error': str(e)
            })
    
    # 統計を計算
    char_accuracies = [r['char_accuracy'] for r in results if 'error' not in r]
    word_accuracies = [r['word_accuracy'] for r in results if 'error' not in r]
    
    stats = {
        'total_images': len(test_images),
        'successful': len(char_accuracies),
        'failed': len(results) - len(char_accuracies),
        'char_accuracy_mean': np.mean(char_accuracies) if char_accuracies else 0,
        'char_accuracy_std': np.std(char_accuracies) if char_accuracies else 0,
        'word_accuracy_mean': np.mean(word_accuracies) if word_accuracies else 0,
        'perfect_words': sum(word_accuracies),
        'avg_processing_time': np.mean([r['processing_time'] for r in results if 'error' not in r])
    }
    
    return {'results': results, 'stats': stats}


def analyze_by_word_length(results: List[Dict]) -> Dict:
    """単語の長さ別に精度を分析"""
    length_analysis = {}
    
    for result in results:
        if 'error' in result:
            continue
            
        length = result['char_count']
        if length not in length_analysis:
            length_analysis[length] = {
                'count': 0,
                'char_accuracy_sum': 0,
                'word_accuracy_sum': 0,
                'examples': []
            }
        
        length_analysis[length]['count'] += 1
        length_analysis[length]['char_accuracy_sum'] += result['char_accuracy']
        length_analysis[length]['word_accuracy_sum'] += result['word_accuracy']
        
        if len(length_analysis[length]['examples']) < 3:
            length_analysis[length]['examples'].append({
                'expected': result['expected'],
                'recognized': result['recognized']
            })
    
    # 平均を計算
    for length, data in length_analysis.items():
        data['char_accuracy_avg'] = data['char_accuracy_sum'] / data['count']
        data['word_accuracy_avg'] = data['word_accuracy_sum'] / data['count']
        del data['char_accuracy_sum']
        del data['word_accuracy_sum']
    
    return length_analysis


def analyze_common_errors(results: List[Dict]) -> Dict:
    """よくある誤認識パターンを分析"""
    char_confusion = {}  # 文字の混同マトリックス
    
    for result in results:
        if 'error' in result:
            continue
            
        expected = result['expected']
        recognized = result['recognized']
        
        # 文字レベルで比較
        for i, (e, r) in enumerate(zip(expected, recognized)):
            if e != r:
                key = f"{e}→{r}"
                if key not in char_confusion:
                    char_confusion[key] = 0
                char_confusion[key] += 1
    
    # 頻度順にソート
    sorted_confusion = sorted(char_confusion.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'top_10_confusions': sorted_confusion[:10],
        'total_confusion_patterns': len(char_confusion),
        'total_confused_chars': sum(char_confusion.values())
    }


def main():
    test_data_dir = Path("test_data")
    
    print("=== 新しいテストデータでのOCR評価 ===")
    print(f"評価開始時刻: {datetime.now()}")
    
    # 現在のシステムで評価
    evaluation = evaluate_current_system(test_data_dir)
    
    # 結果を表示
    stats = evaluation['stats']
    print(f"\n=== 評価結果 ===")
    print(f"テスト画像数: {stats['total_images']}")
    print(f"成功: {stats['successful']}, 失敗: {stats['failed']}")
    print(f"\n文字レベル精度: {stats['char_accuracy_mean']*100:.1f}% (±{stats['char_accuracy_std']*100:.1f}%)")
    print(f"単語レベル精度: {stats['word_accuracy_mean']*100:.1f}% ({stats['perfect_words']}/{stats['successful']} 完全一致)")
    print(f"平均処理時間: {stats['avg_processing_time']:.3f}秒")
    
    # 単語長別の分析
    print("\n=== 単語長別の精度 ===")
    length_analysis = analyze_by_word_length(evaluation['results'])
    for length in sorted(length_analysis.keys()):
        data = length_analysis[length]
        print(f"{length}文字: {data['count']}語, 文字精度{data['char_accuracy_avg']*100:.1f}%, 単語精度{data['word_accuracy_avg']*100:.1f}%")
        for ex in data['examples']:
            print(f"  例: {ex['expected']} → {ex['recognized']}")
    
    # エラー分析
    print("\n=== よくある誤認識パターン (Top 10) ===")
    error_analysis = analyze_common_errors(evaluation['results'])
    for pattern, count in error_analysis['top_10_confusions']:
        print(f"{pattern}: {count}回")
    
    # 最も認識が困難な単語
    print("\n=== 認識が困難な単語 (精度0%) ===")
    difficult_words = [r for r in evaluation['results'] if r['char_accuracy'] == 0 and 'error' not in r]
    for word in difficult_words[:10]:
        print(f"{word['expected']} → {word['recognized']}")
    
    # 結果をJSONファイルに保存
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\n詳細な結果を保存: {output_path}")
    
    # 改善のための推奨事項
    print("\n=== 改善の推奨事項 ===")
    if stats['char_accuracy_mean'] < 0.7:
        print("1. 実画像でのCNNモデル再訓練が強く推奨されます")
    if error_analysis['top_10_confusions'][0][1] > 5:
        print(f"2. {error_analysis['top_10_confusions'][0][0]}の誤認識が多いため、この文字ペアに注目した改善が必要です")
    if length_analysis.get(1, {}).get('char_accuracy_avg', 0) < 0.8:
        print("3. 単一文字の認識精度が低いため、文字分割の改善が必要です")


if __name__ == "__main__":
    main()