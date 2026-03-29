"""
losses.py
---------
Loss functions for EEG seizure detection.

FocalLoss addresses the severe class imbalance in CHB-MIT (~1-5% seizure windows)
by down-weighting easy negatives and focusing training on hard examples.

Typical usage:
    from src.models.losses import FocalLoss
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss: FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive (seizure) class.
               alpha=0.75 means seizure windows receive 3x more weight.
        gamma: Focusing parameter. gamma=2 strongly down-weights easy examples.
               gamma=0 reduces to standard cross-entropy.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t   = torch.exp(-bce)
        alpha = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss  = alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()
