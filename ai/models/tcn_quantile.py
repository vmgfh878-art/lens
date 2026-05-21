from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.blocks import init_weights
from ai.models.common import BandOutput, ForecastOutput, LineRegimeOutput, LineV2Output, LineWarningOutput, MultiHeadForecastModel


class _CausalConv1d(nn.Module):
    """미래 시점을 보지 않도록 왼쪽에만 padding하는 Conv1d."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.left_padding, 0))
        return self.conv(padded)


class _TCNResidualBlock(nn.Module):
    """causal dilated TCN 블록."""

    def __init__(self, channels: int, dilation: int, dropout: float, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = _CausalConv1d(channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.conv1(x)
        hidden = self.norm1(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = self.dropout(hidden)
        return F.relu(hidden + residual)


class TCNQuantile(MultiHeadForecastModel):
    """line/band 공통 계약을 맞춘 causal direct quantile TCN baseline."""

    def __init__(
        self,
        n_features: int = 36,
        seq_len: int = 252,
        tcn_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
        output_role: str = "legacy",
    ) -> None:
        self.seq_len = seq_len
        self.n_features = n_features
        self.dilations = tuple(dilations)
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (2 * (kernel_size - 1) * sum(self.dilations))
        self.use_ticker_embedding = num_tickers > 0
        ticker_extra = ticker_emb_dim if self.use_ticker_embedding else 0
        if output_role in ("line_v2", "line_regime", "line_warning", "line_distributional_mono"):
            raise ValueError("TCNQuantile은 현재 band 전용 후보로만 사용합니다.")
        super().__init__(
            hidden_dim=tcn_channels + ticker_extra,
            horizon=horizon,
            band_mode=band_mode,
            output_role=output_role,
        )
        self.input_proj = nn.Conv1d(n_features, tcn_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _TCNResidualBlock(tcn_channels, dilation, dropout, kernel_size=kernel_size)
                for dilation in self.dilations
            ]
        )
        self.output_norm = nn.LayerNorm(tcn_channels)
        self.output_dropout = nn.Dropout(dropout)
        self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim) if self.use_ticker_embedding else None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> ForecastOutput | LineV2Output | LineRegimeOutput | LineWarningOutput | BandOutput:
        hidden = self.input_proj(x.permute(0, 2, 1))
        hidden = self.blocks(hidden)
        pooled = hidden.mean(dim=-1)
        pooled = self.output_norm(pooled)
        pooled = self.output_dropout(pooled)
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker embedding이 활성화된 모델은 ticker_id가 필요합니다.")
            pooled = torch.cat((pooled, self.ticker_embedding(ticker_id)), dim=-1)
        return self.build_output(pooled)
