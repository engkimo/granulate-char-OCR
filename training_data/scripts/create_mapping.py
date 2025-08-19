#!/usr/bin/env python3
"""
抽出されたグラニュート文字画像を基に文字マッピングを作成
"""
import json
from pathlib import Path
import cv2
import numpy as np
import base64


def image_to_base64(image_path: Path) -> str:
    """画像をBase64エンコード"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_character_hash(image_path: Path) -> str:
    """画像の特徴的なハッシュを生成"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    
    # 画像を8x8にリサイズして簡易的なハッシュを生成
    resized = cv2.resize(img, (8, 8))
    # 平均値で二値化
    avg = resized.mean()
    hash_str = ''.join(['1' if pixel > avg else '0' for pixel in resized.flatten()])
    
    return hash_str


def create_granulate_mapping():
    """グラニュート文字マッピングを作成"""
    extracted_dir = Path("training_data/extracted")
    
    # マッピング辞書
    granulate_mapping = {}
    
    # A-Zの各文字について
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ref_path = extracted_dir / letter / f"{letter}_reference.png"
        
        if ref_path.exists():
            # 画像のハッシュを生成
            char_hash = get_character_hash(ref_path)
            
            # マッピングに追加
            granulate_mapping[letter] = {
                "hash": char_hash,
                "image_path": str(ref_path),
                "base64_sample": image_to_base64(ref_path)[:100] + "..."  # サンプルのみ
            }
            
            print(f"Processed: {letter}")
        else:
            print(f"Warning: {letter} not found")
    
    # JSONファイルとして保存
    output_path = Path("backend/infrastructure/mapping/granulate_character_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(granulate_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nマッピングファイルを保存: {output_path}")
    
    # Pythonコードも生成
    generate_python_mapping(granulate_mapping)


def generate_python_mapping(mapping: dict):
    """Python用のマッピングコードを生成"""
    
    code = '''"""
自動生成されたグラニュート文字マッピング
Generated from training_data/extracted/
"""
from typing import Dict, Optional


class GranulateAlphabet:
    """グラニュート文字のマッピングクラス"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_mappings()
        return cls._instance
    
    def _initialize_mappings(self):
        # 実際のグラニュート文字のハッシュマッピング
        self._hash_to_latin: Dict[str, str] = {
'''
    
    # ハッシュマッピングを追加
    for letter, data in sorted(mapping.items()):
        code += f'            "{data["hash"]}": "{letter}",\n'
    
    code += '''        }
        
        # 逆引き用マッピング
        self._latin_to_hash: Dict[str, str] = {
            v: k for k, v in self._hash_to_latin.items()
        }
    
    def get_latin_from_hash(self, char_hash: str) -> Optional[str]:
        """ハッシュからラテン文字を取得"""
        return self._hash_to_latin.get(char_hash)
    
    def get_hash_from_latin(self, latin_char: str) -> Optional[str]:
        """ラテン文字からハッシュを取得"""
        return self._latin_to_hash.get(latin_char.upper())
    
    def compare_image_to_mapping(self, image_array) -> Optional[str]:
        """画像配列を既知のグラニュート文字と比較"""
        import cv2
        
        # 画像を8x8にリサイズ
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
            
        resized = cv2.resize(gray, (8, 8))
        
        # ハッシュを生成
        avg = resized.mean()
        hash_str = ''.join(['1' if pixel > avg else '0' for pixel in resized.flatten()])
        
        # マッピングから検索
        return self.get_latin_from_hash(hash_str)
'''
    
    # ファイルに保存
    output_path = Path("backend/infrastructure/mapping/granulate_alphabet_generated.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"Pythonマッピングコードを生成: {output_path}")


if __name__ == "__main__":
    create_granulate_mapping()