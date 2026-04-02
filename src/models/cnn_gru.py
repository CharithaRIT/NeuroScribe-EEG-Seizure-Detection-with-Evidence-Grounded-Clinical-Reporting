"""
cnn_gru.py
----------
Baseline 1 model: 3-layer 1D CNN + 2-layer Bidirectional GRU + Attention seizure classifier.

Architecture:
    Input  : (batch, n_channels, window_samples)
    CNN    : 3 x (Conv1d → BatchNorm → ReLU → MaxPool)
    GRU    : 2-layer bidirectional GRU
    Attn   : Soft attention over time steps
    Output : (batch,) logits  →  sigmoid → seizure probability

Typical usage:
    from src.models.cnn_gru import CNNGRUClassifier
    model = CNNGRUClassifier(n_channels=23, dropout=0.5)
"""

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """One CNN stage: Conv1d → BatchNorm → ReLU → MaxPool2."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNNGRUClassifier(nn.Module):
    """
    3-layer 1D CNN + 2-layer Bidirectional GRU + Attention seizure detector.

    Args:
        n_channels: Number of EEG channels (default 23 for CHB-MIT bipolar).
        dropout:    Dropout rate applied in GRU and before the final FC layer.
    """
    def __init__(self, n_channels: int = 23, dropout: float = 0.5,
                 cnn_filters: list = None, hidden_size: int = 128):
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [32, 64, 128]
        self.cnn = nn.Sequential(
            CNNBlock(n_channels,       cnn_filters[0], kernel=7),
            CNNBlock(cnn_filters[0],   cnn_filters[1], kernel=5),
            CNNBlock(cnn_filters[1],   cnn_filters[2], kernel=3),
        )
        self.gru = nn.GRU(
            input_size=cnn_filters[2], hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        gru_out = hidden_size * 2
        self.attention = nn.Linear(gru_out, 1)   # soft attention over time steps
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(gru_out, 1)

    def forward(self, x):
        x = self.cnn(x)                                    # (B, 128, T/8)
        x = x.permute(0, 2, 1)                            # (B, T/8, 128)
        x, _ = self.gru(x)                                # (B, T/8, 256)
        attn = torch.softmax(self.attention(x), dim=1)    # (B, T/8, 1)
        x = (x * attn).sum(dim=1)                         # (B, 256)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)                       # (B,) logits
