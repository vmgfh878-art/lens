"""
Asymmetric Loss
- 상승을 놓치는 것(False Negative)보다 하락을 잘못 예측하는 것(False Positive)에 더 큰 패널티
- alpha > 1이면 실제 상승(y=1)을 놓쳤을 때 더 큰 패널티
"""

import torch
import torch.nn as nn


class AsymmetricBCELoss(nn.Module):
    def __init__(self, alpha: float = 2.0):
        """
        alpha: 실제 상승(y=1)에 대한 가중치.
               alpha=2이면 상승 놓침에 2배 패널티.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        weight = torch.where(target == 1, self.alpha, torch.ones_like(target))
        return (bce * weight).mean()
