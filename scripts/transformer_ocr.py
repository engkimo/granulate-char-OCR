"""
Skeleton for a Transformer-based OCR model (encoder-decoder).
Not integrated yet. Intended for future experimentation.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerOCR(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4, num_decoder_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        # A simple CNN backbone to produce a sequence from image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, d_model, 3, 1, 1), nn.ReLU(True)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.generator = nn.Linear(d_model, num_classes)

    def _image_to_sequence(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 1, H, W) -> features: (W', B, d_model)
        feats = self.cnn(img)
        b, c, h, w = feats.size()
        assert h > 0, "Feature map height must be > 0"
        feats = feats.mean(dim=2)  # global average over height -> (B, C, W)
        feats = feats.permute(2, 0, 1)  # (W, B, C)
        return feats

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        # images: (B, 1, H, W)
        # tgt_seq: (T, B, d_model) - pre-embedded target sequence
        src = self._image_to_sequence(images)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt_seq)
        out = self.transformer(src, tgt)
        logits = self.generator(out)  # (T, B, num_classes)
        return F.log_softmax(logits, dim=-1)


if __name__ == "__main__":
    # quick shape test
    model = TransformerOCR(num_classes=27)
    images = torch.randn(2, 1, 64, 256)
    tgt = torch.randn(10, 2, 256)
    out = model(images, tgt)
    print(out.shape)

