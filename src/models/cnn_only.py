"""
CNN-Only Baseline for EEG Seizure Detection.

3-layer 1D CNN with global average pooling.
No temporal modeling — uses pure spatial-frequency feature extraction.
"""

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """One stage: Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    """
    CNN-only seizure classifier.

    Architecture:
        Input   : (B, 23, 1024)
        CNN1    : (B, f0,  512)  — kernel 7
        CNN2    : (B, f1,  256)  — kernel 5
        CNN3    : (B, f2,  128)  — kernel 3
        AvgPool : (B, f2,   1)
        FC      : (B,   1)
    """

    def __init__(
        self,
        n_channels: int = 23,
        n_samples: int = 1024,
        dropout: float = 0.5,
        filters: list = None,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]
        self.cnn = nn.Sequential(
            CNNBlock(n_channels, filters[0], kernel_size=7),
            CNNBlock(filters[0], filters[1], kernel_size=5),
            CNNBlock(filters[1], filters[2], kernel_size=3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  — raw EEG windows
        Returns:
            logits: (B,)
        """
        out = self.cnn(x)               # (B, 128, T/8)
        out = self.global_pool(out)     # (B, 128, 1)
        out = out.squeeze(-1)           # (B, 128)
        out = self.dropout(out)
        out = self.fc(out)              # (B, 1)
        return out.squeeze(-1)          # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
