#!/usr/bin/env python3
"""
OCR APIのテストスクリプト
"""
import httpx
import base64
from pathlib import Path
import json
import cv2
import numpy as np
from typing import List, Dict


def encode_image_to_base64(image_path: Path) -> str:
    """画像をBase64エンコード"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_single_character(api_url: str, image_path: Path, expected_char: str) -> Dict:
    """単一文字のテスト"""
    # 画像をBase64エンコード
    base64_image = encode_image_to_base64(image_path)
    
    # APIリクエスト
    files = {
        'file': (image_path.name, open(image_path, 'rb'), 'image/png')
    }
    data = {
        'options': json.dumps({
            'enhance': True,
            'language': 'granulate'
        })
    }
    
    try:
        response = httpx.post(f"{api_url}/api/ocr/process", files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        recognized_text = result.get('text', '')
        confidence = result.get('confidence', 0)
        
        return {
            'expected': expected_char,
            'recognized': recognized_text,
            'confidence': confidence,
            'correct': recognized_text == expected_char,
            'error': None
        }
    except Exception as e:
        return {
            'expected': expected_char,
            'recognized': '',
            'confidence': 0,
            'correct': False,
            'error': str(e)
        }


def test_all_characters(api_url: str, test_dir: Path) -> List[Dict]:
    """全文字のテスト"""
    results = []
    
    # A-Zの各文字をテスト
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = test_dir / char
        if not char_dir.exists():
            print(f"警告: {char} のディレクトリが見つかりません")
            continue
        
        # 最初の画像をテスト（オリジナルを優先）
        test_images = list(char_dir.glob("*_original_*.png"))
        if not test_images:
            test_images = list(char_dir.glob("*.png"))
        
        if test_images:
            print(f"テスト中: {char} - {test_images[0].name}")
            result = test_single_character(api_url, test_images[0], char)
            results.append(result)
            
            # 結果を表示
            if result['correct']:
                print(f"  ✓ 正解: {char}")
            else:
                print(f"  ✗ 不正解: 期待={char}, 認識={result['recognized']}")
    
    return results


def create_test_image_grid(test_dir: Path, output_path: Path):
    """テスト画像のグリッドを作成"""
    images = []
    labels = []
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = test_dir / char
        if char_dir.exists():
            # オリジナル画像を探す
            original = list(char_dir.glob("*_original_*.png"))
            if original:
                img = cv2.imread(str(original[0]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(char)
    
    if not images:
        print("テスト画像が見つかりません")
        return
    
    # グリッドを作成（6行×5列を想定）
    rows = 6
    cols = 5
    img_size = images[0].shape[0]
    
    grid = np.zeros((rows * img_size, cols * img_size), dtype=np.uint8)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        y1 = row * img_size
        y2 = (row + 1) * img_size
        x1 = col * img_size
        x2 = (col + 1) * img_size
        
        grid[y1:y2, x1:x2] = img
    
    cv2.imwrite(str(output_path), grid)
    print(f"テストグリッド画像を保存: {output_path}")


def main():
    # APIのURL（ローカル開発環境）
    API_URL = "http://localhost:8000"
    
    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent.parent
    
    # テストデータのディレクトリ
    TEST_DIR = project_root / "training_data" / "augmented"
    
    print("=== OCR API テスト ===")
    print(f"API URL: {API_URL}")
    print(f"テストデータ: {TEST_DIR}")
    print()
    
    # APIが起動しているか確認
    try:
        response = httpx.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("エラー: APIサーバーが応答しません")
            print("以下のコマンドでサーバーを起動してください:")
            print("  uv run uvicorn backend.main:app --reload")
            return
    except:
        print("エラー: APIサーバーに接続できません")
        print("以下のコマンドでサーバーを起動してください:")
        print("  uv run uvicorn backend.main:app --reload")
        return
    
    # テストグリッド画像を作成
    grid_path = project_root / "tests" / "integration" / "test_grid.png"
    create_test_image_grid(TEST_DIR, grid_path)
    
    # 全文字をテスト
    print("\n個別文字のテスト:")
    results = test_all_characters(API_URL, TEST_DIR)
    
    # 結果のサマリー
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n=== テスト結果 ===")
    print(f"正解数: {correct_count}/{total_count}")
    print(f"精度: {accuracy:.1f}%")
    
    # エラーがあった文字を表示
    errors = [r for r in results if not r['correct'] and r['error'] is None]
    if errors:
        print("\n認識エラーの詳細:")
        for e in errors:
            print(f"  {e['expected']} → {e['recognized']} (信頼度: {e['confidence']:.2f})")


if __name__ == "__main__":
    main()