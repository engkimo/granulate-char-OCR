#!/usr/bin/env python3
"""
類似度ベースのマッピングをテスト
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def create_hash(image: np.ndarray, size: int = 8) -> str:
    """画像から知覚的ハッシュを生成"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    
    hash_str = ""
    for row in resized:
        for pixel in row:
            hash_str += "1" if pixel > avg else "0"
    
    return hash_str


def hamming_distance(hash1: str, hash2: str) -> int:
    """ハミング距離を計算"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def find_closest_match(test_hash: str, reference_hashes: dict, threshold: int = 10) -> Tuple[Optional[str], int]:
    """最も近いマッチを見つける"""
    best_match = None
    min_distance = float('inf')
    
    for char, ref_hash in reference_hashes.items():
        distance = hamming_distance(test_hash, ref_hash)
        if distance < min_distance:
            min_distance = distance
            best_match = char
    
    if min_distance <= threshold:
        return best_match, min_distance
    return None, min_distance


def create_reference_hashes(extracted_dir: Path) -> dict:
    """リファレンス画像からハッシュを作成"""
    reference_hashes = {}
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = extracted_dir / char
        if not char_dir.exists():
            continue
        
        # reference画像を探す
        ref_images = list(char_dir.glob("*_reference.png"))
        if ref_images:
            img = cv2.imread(str(ref_images[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                reference_hashes[char] = create_hash(img)
    
    return reference_hashes


def test_similarity_mapping():
    """類似度ベースのマッピングをテスト"""
    extracted_dir = Path("training_data/extracted")
    augmented_dir = Path("training_data/augmented")
    
    print("=== グラニュート文字認識テスト（類似度ベース） ===\n")
    
    # リファレンスハッシュを作成
    print("リファレンスハッシュを作成中...")
    reference_hashes = create_reference_hashes(extracted_dir)
    print(f"作成されたリファレンス: {len(reference_hashes)}文字\n")
    
    total = 0
    correct = 0
    distances = []
    
    # 各文字をテスト
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = augmented_dir / char
        if not char_dir.exists():
            continue
        
        # テスト画像（オリジナルと拡張画像の両方）
        test_images = list(char_dir.glob("*.png"))[:5]  # 最初の5枚をテスト
        
        for img_path in test_images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # ハッシュを生成
            test_hash = create_hash(img)
            
            # 最も近いマッチを見つける
            recognized, distance = find_closest_match(test_hash, reference_hashes)
            distances.append(distance)
            
            total += 1
            if recognized == char:
                correct += 1
                print(f"✓ {char}: 正解 (距離: {distance})")
            else:
                print(f"✗ {char}: 不正解 (認識: {recognized}, 距離: {distance})")
    
    # 結果のサマリー
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_distance = sum(distances) / len(distances) if distances else 0
    
    print(f"\n=== 結果 ===")
    print(f"正解数: {correct}/{total}")
    print(f"精度: {accuracy:.1f}%")
    print(f"平均ハミング距離: {avg_distance:.1f}")
    print(f"最小距離: {min(distances) if distances else 'N/A'}")
    print(f"最大距離: {max(distances) if distances else 'N/A'}")


def test_single_image(image_path: str):
    """単一画像をテスト"""
    extracted_dir = Path("training_data/extracted")
    
    # リファレンスハッシュを作成
    reference_hashes = create_reference_hashes(extracted_dir)
    
    # 画像を読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"画像を読み込めません: {image_path}")
        return
    
    # ハッシュを生成
    test_hash = create_hash(img)
    
    # 最も近いマッチを見つける
    recognized, distance = find_closest_match(test_hash, reference_hashes, threshold=15)
    
    print(f"認識結果: {recognized}")
    print(f"ハミング距離: {distance}")
    
    # 上位3つの候補を表示
    candidates = []
    for char, ref_hash in reference_hashes.items():
        dist = hamming_distance(test_hash, ref_hash)
        candidates.append((char, dist))
    
    candidates.sort(key=lambda x: x[1])
    print("\n上位候補:")
    for char, dist in candidates[:3]:
        print(f"  {char}: 距離 {dist}")


if __name__ == "__main__":
    test_similarity_mapping()
    
    # 特定の画像をテストする場合
    # test_single_image("path/to/test/image.png")