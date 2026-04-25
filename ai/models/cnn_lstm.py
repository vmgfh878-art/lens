from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.blocks import AttentionPooling1D, init_weights
from ai.models.common import ForecastOutput, MultiHeadForecastModel


class CNNLSTM(MultiHeadForecastModel):
    """Conv residual block과 attention pooling을 포함한 CNN-LSTM."""

    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 252,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
    ) -> None:
        super().__init__(hidden_dim=lstm_hidden, horizon=horizon, band_mode=band_mode)
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(cnn_channels)
        self.conv_residual_proj = nn.Conv1d(n_features, cnn_channels, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.attn_pool = AttentionPooling1D(lstm_hidden)
        self.output_dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        channel_first = x.permute(0, 2, 1)
        residual = self.conv_residual_proj(channel_first)

        hidden = F.relu(self.conv1(channel_first))
        hidden = self.norm1(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = self.conv_dropout(hidden)
        hidden = F.relu(self.conv2(hidden))
        hidden = self.norm2(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = hidden + residual
        hidden = self.conv_dropout(hidden)

        sequence_hidden = hidden.permute(0, 2, 1)
        lstm_out, _ = self.lstm(sequence_hidden)
        lstm_out = self.lstm_norm(lstm_out)
        pooled = self.attn_pool(lstm_out)
        pooled = self.output_dropout(pooled)
        return self.build_output(pooled)
