#!/usr/bin/env python3
"""
保存されたモデルの構造を確認
"""
import torch
from pathlib import Path

model_path = Path('models/cnn_model_best.pth')
if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("チェックポイントのキー:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    print("\nモデルのstate_dictのキー（最初の10個）:")
    state_dict = checkpoint['model_state_dict']
    for i, (key, value) in enumerate(state_dict.items()):
        if i < 10:
            print(f"  - {key}: {value.shape}")
        else:
            break
    
    print(f"\n総パラメータ数: {len(state_dict)}")
    
    # features層の構造を確認
    print("\nfeatures層の構造:")
    features_keys = [k for k in state_dict.keys() if k.startswith('features.')]
    unique_indices = sorted(set(int(k.split('.')[1]) for k in features_keys))
    print(f"features層のインデックス: {unique_indices}")
else:
    print(f"モデルファイルが見つかりません: {model_path}")