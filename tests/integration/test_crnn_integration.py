#!/usr/bin/env python3
"""
CRNN統合テスト
APIエンドポイントでCRNNモデルが正しく動作するか確認
"""
import requests
import base64
from pathlib import Path
import json
from typing import Dict, List
import time


def encode_image_to_base64(image_path: Path) -> str:
    """画像をBase64エンコード"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_single_image(image_path: Path, expected_text: str) -> Dict:
    """単一画像のテスト"""
    # APIエンドポイント
    url = "http://localhost:8000/api/v1/ocr/process-base64"
    
    # 画像をBase64エンコード
    image_base64 = encode_image_to_base64(image_path)
    
    # リクエスト送信
    start_time = time.time()
    response = requests.post(
        url,
        json={"image": image_base64},
        headers={"Content-Type": "application/json"}
    )
    request_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        recognized_text = result['text']
        confidence = result['average_confidence']
        processing_time = result['processing_time']
        
        # 精度計算
        correct_chars = sum(1 for e, r in zip(expected_text, recognized_text) if e == r)
        total_chars = len(expected_text)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        word_accuracy = 1.0 if recognized_text == expected_text else 0.0
        
        return {
            'file': image_path.name,
            'expected': expected_text,
            'recognized': recognized_text,
            'confidence': confidence,
            'char_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'processing_time': processing_time,
            'request_time': request_time,
            'success': True
        }
    else:
        return {
            'file': image_path.name,
            'expected': expected_text,
            'recognized': '',
            'error': f"HTTP {response.status_code}: {response.text}",
            'success': False
        }


def test_all_images() -> List[Dict]:
    """すべてのテスト画像を評価"""
    test_data_dir = Path("test_data")
    results = []
    
    # テスト画像を収集
    test_images = []
    for img_path in sorted(test_data_dir.glob("*_*.png")):
        expected = img_path.stem.split('_')[0].replace('!', '').replace('.', '')
        if expected.isalpha() and expected.isupper():
            test_images.append((img_path, expected))
    
    print(f"Testing {len(test_images)} images...")
    
    # 各画像をテスト
    for img_path, expected in test_images:
        print(f"Testing {img_path.name}...", end='', flush=True)
        result = test_single_image(img_path, expected)
        results.append(result)
        
        if result['success']:
            accuracy_str = f"{result['char_accuracy']*100:.1f}%"
            if result['word_accuracy'] == 1.0:
                print(f" ✓ Perfect match! ({result['processing_time']*1000:.0f}ms)")
            else:
                print(f" {accuracy_str} ({result['recognized']}) ({result['processing_time']*1000:.0f}ms)")
        else:
            print(f" ✗ Error: {result['error']}")
    
    return results


def analyze_results(results: List[Dict]):
    """結果を分析"""
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("\n=== No successful results ===")
        return
    
    # 統計を計算
    total_words = len(successful)
    perfect_words = sum(1 for r in successful if r['word_accuracy'] == 1.0)
    
    total_chars = sum(len(r['expected']) for r in successful)
    correct_chars = sum(int(r['char_accuracy'] * len(r['expected'])) for r in successful)
    
    avg_confidence = sum(r['confidence'] for r in successful) / total_words
    avg_processing_time = sum(r['processing_time'] for r in successful) / total_words
    avg_request_time = sum(r['request_time'] for r in successful) / total_words
    
    # エラー分析
    errors = {}
    for r in successful:
        if r['word_accuracy'] < 1.0:
            expected = r['expected']
            recognized = r['recognized']
            for i, (e, rec) in enumerate(zip(expected, recognized)):
                if e != rec:
                    error_key = f"{e}→{rec}"
                    errors[error_key] = errors.get(error_key, 0) + 1
    
    # 結果を表示
    print(f"\n=== CRNN Integration Test Results ===")
    print(f"Total images tested: {total_words}")
    print(f"Word accuracy: {perfect_words}/{total_words} ({perfect_words/total_words*100:.1f}%)")
    print(f"Character accuracy: {correct_chars}/{total_chars} ({correct_chars/total_chars*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Average processing time: {avg_processing_time*1000:.1f}ms")
    print(f"Average request time: {avg_request_time*1000:.1f}ms")
    
    if errors:
        print(f"\n=== Common errors (Top 10) ===")
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:10]
        for error, count in sorted_errors:
            print(f"{error}: {count} times")
    
    # 結果を保存
    output_file = Path("results/crnn_integration_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'summary': {
                'total_images': total_words,
                'word_accuracy': perfect_words/total_words if total_words > 0 else 0,
                'character_accuracy': correct_chars/total_chars if total_chars > 0 else 0,
                'average_confidence': avg_confidence,
                'average_processing_time': avg_processing_time,
                'average_request_time': avg_request_time,
                'errors': errors
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    print("=== Testing CRNN Integration with API ===")
    print("Make sure the API server is running (uv run uvicorn backend.main:app --reload)")
    print()
    
    try:
        # APIが起動しているか確認
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code != 200:
            print("Error: API server is not responding")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API server at http://localhost:8000")
        print("Please start the server with: uv run uvicorn backend.main:app --reload")
        return
    
    # テスト実行
    results = test_all_images()
    
    # 結果を分析
    analyze_results(results)


if __name__ == "__main__":
    main()