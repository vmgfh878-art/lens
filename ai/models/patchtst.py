"""
PatchTST - Transformer 기반 시계열 예측 모델
논문: "A Time Series is Worth 64 Words" (2023)

입력: (batch, seq_len, n_features) = (B, 60, 17)
출력: (batch, n_horizons) = (B, 4)  # 1,3,5,7일 상승확률
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_outputs: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO: PatchTST 구현
        self.placeholder = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        # TODO: 실제 PatchTST forward 구현
        return torch.sigmoid(self.placeholder(x.mean(dim=1)))
