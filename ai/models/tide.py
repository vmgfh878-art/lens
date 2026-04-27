from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.blocks import ResidualBlock, init_weights
from ai.models.common import ForecastOutput, _split_band


class TiDE(nn.Module):
    """ResidualBlock 기반 encoder/decoder와 lookback skip을 포함한 TiDE."""

    def __init__(
        self,
        n_features: int = 36,
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
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
        future_cov_dim: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.band_mode = band_mode
        self.lookback_baseline_idx = lookback_baseline_idx
        self.use_ticker_embedding = num_tickers > 0
        self.ticker_emb_dim = ticker_emb_dim if self.use_ticker_embedding else 0
        self.future_cov_dim = future_cov_dim
        self.temporal_input_dim = dec_dim + self.ticker_emb_dim + self.future_cov_dim
        self.temporal_hidden_dim = self.temporal_input_dim
        self.feature_proj = nn.Linear(n_features, feature_dim)
        self.enc_input_proj = nn.Linear(seq_len * feature_dim, enc_dim)
        self.encoder = nn.Sequential(*[ResidualBlock(enc_dim, dropout=dropout) for _ in range(n_enc_layers)])
        self.dec_input_proj = nn.Linear(enc_dim, dec_dim * horizon)
        self.decoder = nn.Sequential(*[ResidualBlock(dec_dim, dropout=dropout) for _ in range(n_dec_layers)])
        self.temporal_decoder = ResidualBlock(self.temporal_input_dim, dropout=dropout)
        self.line_head = nn.Linear(self.temporal_hidden_dim, 1)
        self.band_head = nn.Linear(self.temporal_hidden_dim, 2)
        self.lookback_skip = nn.Linear(seq_len, horizon)
        self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim) if self.use_ticker_embedding else None
        self.last_temporal_hidden_shape: tuple[int, ...] | None = None
        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        ticker_id: torch.Tensor | None = None,
        future_covariate: torch.Tensor | None = None,
    ) -> ForecastOutput:
        batch_size = x.size(0)
        projected = self.feature_proj(x)
        encoded = self.enc_input_proj(projected.flatten(start_dim=1))
        encoded = self.encoder(encoded)
        decoded = self.dec_input_proj(encoded).view(batch_size, self.horizon, -1)
        decoded = self.decoder(decoded)

        temporal_parts = [decoded]
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker embedding이 활성화된 모델은 ticker_id가 필요합니다.")
            ticker_hidden = self.ticker_embedding(ticker_id).unsqueeze(1).expand(-1, self.horizon, -1)
            temporal_parts.append(ticker_hidden)
        if future_covariate is not None:
            if future_covariate.dim() != 3 or future_covariate.size(1) != self.horizon:
                raise ValueError("future_covariate shape는 (B, H, C) 형식이어야 합니다.")
            if self.future_cov_dim and future_covariate.size(-1) != self.future_cov_dim:
                raise ValueError("future_covariate 채널 수가 future_cov_dim과 다릅니다.")
            temporal_parts.append(future_covariate)
        elif self.future_cov_dim > 0:
            raise ValueError("future_cov_dim > 0 인 경우 future_covariate가 필요합니다.")

        temporal_input = torch.cat(temporal_parts, dim=-1)
        temporal_hidden = self.temporal_decoder(temporal_input)
        self.last_temporal_hidden_shape = tuple(temporal_hidden.shape)
        line = self.line_head(temporal_hidden).squeeze(-1)
        band_raw = self.band_head(temporal_hidden)
        lower_band, upper_band = _split_band(band_raw, self.band_mode)

        baseline_series = x[..., self.lookback_baseline_idx]
        skip = self.lookback_skip(baseline_series)
        return ForecastOutput(
            line=line + skip,
            lower_band=lower_band + skip,
            upper_band=upper_band + skip,
        )
