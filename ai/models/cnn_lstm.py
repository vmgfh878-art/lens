from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.common import ForecastOutput, MultiHeadForecastModel


class CNNLSTM(MultiHeadForecastModel):
    """CNN으로 국소 패턴을 본 뒤 LSTM으로 시계열 의존성을 요약한다."""

    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(hidden_dim=lstm_hidden, horizon=horizon)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        return self.build_output(hidden)
