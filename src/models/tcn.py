"""
Temporal Convolutional Network (TCN) for EEG Seizure Detection.

Based on:
    Bai et al., "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling", arXiv:1803.01271, 2018.

Uses dilated causal convolutions with exponentially growing receptive fields
and residual connections for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Remove right-side padding to enforce causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One TCN residual block: two dilated causal conv layers + residual.

    Receptive field per block: 2 * (kernel_size - 1) * dilation + 1 samples
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        # 1×1 conv to match channel dimensions for residual
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.bn1(self.chomp1(self.conv1(x)))))
        out = self.drop2(self.relu2(self.bn2(self.chomp2(self.conv2(out)))))
        return self.final_relu(out + self.residual(x))


class TCNClassifier(nn.Module):
    """
    TCN seizure classifier.

    Architecture:
        Input         : (B, 23, 1024)
        TCN blocks    : 6 temporal blocks, dilation = 2^i for i in 0..5
                        Receptive field ≈ 2*(3-1)*(1+2+4+8+16+32)*2 = 252 samples (~1s)
        GlobalAvgPool : (B, num_filters, 1) → (B, num_filters)
        Dropout + FC  : (B, 1)

    Default ~500K parameters (num_filters=64, num_blocks=6, kernel_size=3).
    """

    def __init__(
        self,
        n_channels: int = 23,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.receptive_field = self._calc_receptive_field(kernel_size, num_blocks)

        blocks = []
        in_ch = n_channels
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(in_ch, num_filters, kernel_size, dilation, dropout)
            )
            in_ch = num_filters

        self.network = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    @staticmethod
    def _calc_receptive_field(kernel_size: int, num_blocks: int) -> int:
        """Total receptive field in time steps."""
        return 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  — raw EEG windows (C=23, T=1024)
        Returns:
            logits: (B,)
        """
        out = self.network(x)           # (B, num_filters, T)
        out = self.global_pool(out)     # (B, num_filters, 1)
        out = out.squeeze(-1)           # (B, num_filters)
        out = self.dropout(out)
        out = self.fc(out)              # (B, 1)
        return out.squeeze(-1)          # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
