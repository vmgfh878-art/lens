from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.common import ForecastOutput, MultiHeadForecastModel


class TiDE(MultiHeadForecastModel):
    """MLP 기반 TiDE 스타일 backbone 위에 line/band head를 올린다."""

    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        hidden_dim: int = 256,
        n_layers: int = 4,
        horizon: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, horizon=horizon)
        input_dim = n_features * seq_len
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        hidden = self.backbone(x.flatten(start_dim=1))
        return self.build_output(hidden)
