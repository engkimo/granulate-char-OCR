#!/usr/bin/env python3
"""
ハッシュベースのマッピングをテスト
"""
import cv2
import numpy as np
from pathlib import Path
from backend.infrastructure.mapping.granulate_alphabet_generated import GranulateAlphabet


def create_hash(image: np.ndarray, size: int = 8) -> str:
    """画像から知覚的ハッシュを生成"""
    # グレースケールに変換（既にグレースケールの場合はスキップ）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # リサイズ
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    
    # 平均値を計算
    avg = resized.mean()
    
    # ハッシュを生成
    hash_str = ""
    for row in resized:
        for pixel in row:
            hash_str += "1" if pixel > avg else "0"
    
    return hash_str


def test_mapping():
    """マッピングをテスト"""
    mapper = GranulateAlphabet()
    test_dir = Path("training_data/augmented")
    
    print("=== グラニュート文字認識テスト ===\n")
    
    total = 0
    correct = 0
    
    # 各文字をテスト
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = test_dir / char
        if not char_dir.exists():
            continue
        
        # オリジナル画像を探す
        test_images = list(char_dir.glob("*_original_*.png"))[:3]  # 最初の3枚をテスト
        
        for img_path in test_images:
            # 画像を読み込み
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # ハッシュを生成
            hash_value = create_hash(img)
            
            # マッピングで認識
            recognized = mapper.get_latin_from_hash(hash_value)
            
            total += 1
            if recognized == char:
                correct += 1
                print(f"✓ {char}: 正解")
            else:
                print(f"✗ {char}: 不正解 (認識: {recognized})")
                # ハッシュ値を表示（デバッグ用）
                print(f"  ハッシュ: {hash_value}")
    
    # 結果のサマリー
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n=== 結果 ===")
    print(f"正解数: {correct}/{total}")
    print(f"精度: {accuracy:.1f}%")
    
    # 各文字のオリジナルハッシュを確認
    print("\n=== 登録されているハッシュ値 ===")
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        latin_hash = mapper.get_hash_from_latin(char)
        if latin_hash:
            print(f"{char}: {latin_hash[:16]}...")  # 最初の16ビットのみ表示


if __name__ == "__main__":
    test_mapping()