#!/usr/bin/env python3
"""
CRNN (Convolutional Recurrent Neural Network) モデルの実装
エンドツーエンドのテキスト認識を行う
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class BidirectionalLSTM(nn.Module):
    """双方向LSTMモジュール"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    """CRNN モデル
    
    CNN特徴抽出器 + RNN系列認識器の組み合わせ
    文字分割を必要とせずに単語全体を認識可能
    """
    
    def __init__(self, img_height: int = 64, num_classes: int = 27, hidden_size: int = 256):
        """
        Args:
            img_height: 入力画像の高さ
            num_classes: 出力クラス数（26文字 + CTC blank）
            hidden_size: RNNの隠れ層サイズ
        """
        super().__init__()
        
        # CNN特徴抽出器
        # VGGライクなアーキテクチャ
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x? -> 32x?
            
            # Conv Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x? -> 16x?
            
            # Conv Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 16x? -> 8x?
            
            # Conv Block 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 8x? -> 4x?
            
            # Conv Block 5
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # RNN系列認識器
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (batch, channel=1, height, width)
        
        Returns:
            output: (seq_len, batch, num_classes)
        """
        # CNN特徴抽出
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"CNNの出力高さは1である必要があります。現在: {h}"
        
        # RNN用に変形
        conv = conv.squeeze(2)  # (b, c, w)
        conv = conv.permute(2, 0, 1)  # (w, b, c)
        
        # RNN処理
        output = self.rnn(conv)
        
        # Log softmaxを適用（CTC lossで必要）
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def get_input_lengths(self, batch_size: int, img_width: int) -> torch.Tensor:
        """入力画像幅から出力系列長を計算"""
        # CNNによる幅の縮小を計算
        # MaxPool2d: 2回 -> 1/4
        # MaxPool2d with stride (2,1): 2回 -> 1/2
        # 合計: 1/8
        output_width = img_width // 8
        return torch.full((batch_size,), output_width, dtype=torch.long)


class CTCLabelConverter:
    """CTC用のラベル変換器"""
    
    def __init__(self, character_set: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.character_set = character_set
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(character_set)}  # 0はCTC blank
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(character_set)}
        self.blank_idx = 0
    
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """テキストをインデックスに変換
        
        Args:
            texts: 文字列のリスト
            
        Returns:
            encoded: 連結されたラベルテンソル
            lengths: 各テキストの長さ
        """
        encoded_texts = []
        lengths = []
        
        for text in texts:
            encoded = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
            encoded_texts.extend(encoded)
            lengths.append(len(encoded))
        
        return torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor, lengths: torch.Tensor = None) -> List[str]:
        """インデックスをテキストに変換（CTC decodingを含む）
        
        Args:
            indices: 予測されたインデックス (seq_len, batch) or (seq_len,)
            lengths: 各バッチの有効な長さ
            
        Returns:
            decoded_texts: デコードされたテキストのリスト
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        
        texts = []
        for i in range(indices.size(1)):
            seq = indices[:, i]
            if lengths is not None:
                seq = seq[:lengths[i]]
            
            # CTC decoding: 連続する同じ文字を削除し、blankを除去
            decoded = []
            prev_idx = self.blank_idx
            
            for idx in seq:
                idx = idx.item()
                if idx != self.blank_idx and idx != prev_idx:
                    if idx in self.idx_to_char:
                        decoded.append(self.idx_to_char[idx])
                prev_idx = idx
            
            texts.append(''.join(decoded))
        
        return texts


def create_crnn_model(pretrained: bool = False) -> CRNN:
    """CRNNモデルのインスタンスを作成
    
    Args:
        pretrained: 事前学習済みモデルを使用するか（将来の実装用）
        
    Returns:
        model: CRNNモデル
    """
    model = CRNN(img_height=64, num_classes=27, hidden_size=256)
    
    if pretrained:
        # TODO: 事前学習済みモデルのロード
        pass
    
    return model


if __name__ == "__main__":
    # モデルのテスト
    model = create_crnn_model()
    print(f"CRNNモデル作成完了")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ダミー入力でテスト
    batch_size = 4
    img_height = 64
    img_width = 256  # 可変長対応
    dummy_input = torch.randn(batch_size, 1, img_height, img_width)
    
    output = model(dummy_input)
    print(f"入力サイズ: {dummy_input.shape}")
    print(f"出力サイズ: {output.shape}")  # (seq_len, batch, num_classes)
    
    # ラベル変換器のテスト
    converter = CTCLabelConverter()
    texts = ["HELLO", "WORLD", "GRANULATE", "OCR"]
    encoded, lengths = converter.encode(texts)
    print(f"\nエンコード結果:")
    print(f"ラベル: {encoded}")
    print(f"長さ: {lengths}")
    
    # デコードテスト
    # ダミーの予測結果（同じ文字の繰り返しを含む）
    dummy_predictions = torch.tensor([
        [8, 8, 5, 5, 0, 12, 12, 12, 15, 15],  # H H E E _ L L L O O
        [23, 15, 0, 18, 18, 12, 4, 0, 0, 0],  # W O _ R R L D _ _ _
    ]).T  # (seq_len, batch)
    
    decoded = converter.decode(dummy_predictions)
    print(f"\nデコード結果: {decoded}")