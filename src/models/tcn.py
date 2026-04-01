"""
Temporal Convolutional Network (TCN) for EEG Seizure Detection.

Based on:
    Bai et al., "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling", arXiv:1803.01271, 2018.

Key improvements over vanilla TCN:
  1. Input projection layer: mixes 23 EEG channels into 64-dim feature space
     before any temporal modeling (separates spatial from temporal processing).
  2. 8 dilated blocks (dilation 1→128): receptive field = 1021 samples (~4s),
     covering the entire input window at 256 Hz.
  3. Wider filters (128): matches CNN+GRU capacity.
  4. Dual pooling head: concatenates global avg-pool + global max-pool → 256-dim,
     capturing both mean activity and peak activations.
  5. Two-layer MLP classifier with higher dropout (0.5) for regularization.
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
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

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
    Improved TCN seizure classifier.

    Architecture:
        Input         : (B, 23, 1024)
        Input proj    : Conv1d(23→64, k=1) + BN + ReLU   ← spatial mixing
        TCN blocks    : 8 blocks, num_filters=128, dilation=2^i (i=0..7)
                        Receptive field = 1021 samples (~4s) — full window
        Dual pool     : AvgPool(1) || MaxPool(1) → concat → (B, 256)
        MLP head      : Linear(256→64) → ReLU → Dropout(0.5) → Linear(64→1)

    ~1.1M parameters.
    """

    def __init__(
        self,
        n_channels: int = 23,
        proj_channels: int = 64,
        num_filters: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 8,
        dropout: float = 0.5,      # increased from 0.3 — reduces overfitting
    ):
        super().__init__()
        self.receptive_field = self._calc_receptive_field(kernel_size, num_blocks)

        # ── Input projection: mix EEG channels into feature space ──────
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(proj_channels),
            nn.ReLU(inplace=True),
        )

        # ── Dilated temporal blocks ─────────────────────────────────────
        blocks = []
        in_ch = proj_channels
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(in_ch, num_filters, kernel_size, dilation, dropout)
            )
            in_ch = num_filters
        self.network = nn.Sequential(*blocks)

        # ── Dual pooling head ───────────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # ── MLP classifier ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _calc_receptive_field(kernel_size: int, num_blocks: int) -> int:
        return 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  — raw EEG windows (C=23, T=1024)
        Returns:
            logits: (B,)
        """
        out = self.input_proj(x)            # (B, 64, 1024)
        out = self.network(out)             # (B, 128, 1024)

        avg = self.avg_pool(out).squeeze(-1)  # (B, 128)
        mx  = self.max_pool(out).squeeze(-1)  # (B, 128)
        out = torch.cat([avg, mx], dim=1)     # (B, 256)

        out = self.classifier(out)            # (B, 1)
        return out.squeeze(-1)                # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
