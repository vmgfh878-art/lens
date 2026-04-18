"""
TiDE - Time-series Dense Encoder (MLP 기반)
논문: "Long-term Forecasting with TiDE" (2023)

입력: (batch, seq_len, n_features) = (B, 60, 17)
출력: (batch, n_horizons) = (B, 4)
"""

import torch
import torch.nn as nn


class TiDE(nn.Module):
    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_outputs: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = n_features * seq_len
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        x = x.flatten(start_dim=1)      # (B, seq_len * n_features)
        return torch.sigmoid(self.net(x))
