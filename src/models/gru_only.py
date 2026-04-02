"""
GRU-Only Baseline for EEG Seizure Detection.

2-layer Bidirectional GRU operating directly on raw EEG channels over time.
No CNN feature extraction — tests pure sequential modeling capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """
    BiGRU-only seizure classifier with soft attention.

    Architecture:
        Input    : (B, 23, 1024)
        Permute  : (B, 1024, 23)   — time-first for GRU
        BiGRU    : (B, 1024, 256)  — hidden=128, bidirectional
        Attention: soft attention over 1024 time steps → (B, 256)
        Dropout  + FC → (B, 1)

    ~320K parameters.
    """

    def __init__(
        self,
        n_channels: int = 23,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_size * (2 if bidirectional else 1)

        # Soft attention
        self.attn_fc = nn.Linear(gru_out_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  — raw EEG windows
        Returns:
            logits: (B,)
        """
        # (B, C, T) → (B, T, C) for GRU
        x = x.permute(0, 2, 1)                 # (B, 1024, 23)
        gru_out, _ = self.gru(x)               # (B, 1024, 256)

        # Soft attention over time steps
        attn_scores = self.attn_fc(gru_out)    # (B, 1024, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 1024, 1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, 256)

        out = self.dropout(context)
        out = self.fc(out)                     # (B, 1)
        return out.squeeze(-1)                 # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
