from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.blocks import ResidualBlock, init_weights
from ai.models.common import ForecastOutput, MultiHeadForecastModel


class TiDE(MultiHeadForecastModel):
    """ResidualBlock 기반 encoder/decoder와 lookback skip을 포함한 TiDE."""

    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 252,
        feature_dim: int = 16,
        enc_dim: int = 256,
        dec_dim: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
        band_mode: str = "direct",
        lookback_baseline_idx: int = 0,
    ) -> None:
        super().__init__(hidden_dim=dec_dim, horizon=horizon, band_mode=band_mode)
        self.seq_len = seq_len
        self.horizon = horizon
        self.lookback_baseline_idx = lookback_baseline_idx
        self.feature_proj = nn.Linear(n_features, feature_dim)
        self.enc_input_proj = nn.Linear(seq_len * feature_dim, enc_dim)
        self.encoder = nn.Sequential(*[ResidualBlock(enc_dim, dropout=dropout) for _ in range(n_enc_layers)])
        self.dec_input_proj = nn.Linear(enc_dim, dec_dim * horizon)
        self.decoder = nn.Sequential(*[ResidualBlock(dec_dim, dropout=dropout) for _ in range(n_dec_layers)])
        self.temporal_decoder = nn.Linear(dec_dim, dec_dim)
        self.lookback_skip = nn.Linear(seq_len, horizon)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        batch_size = x.size(0)
        projected = self.feature_proj(x)
        encoded = self.enc_input_proj(projected.flatten(start_dim=1))
        encoded = self.encoder(encoded)
        decoded = self.dec_input_proj(encoded).view(batch_size, self.horizon, -1)
        decoded = self.decoder(decoded)
        decoded = self.temporal_decoder(decoded)
        pooled = decoded.mean(dim=1)
        output = self.build_output(pooled)

        baseline_series = x[..., self.lookback_baseline_idx]
        skip = self.lookback_skip(baseline_series)
        return ForecastOutput(
            line=output.line + skip,
            lower_band=output.lower_band + skip,
            upper_band=output.upper_band + skip,
        )
