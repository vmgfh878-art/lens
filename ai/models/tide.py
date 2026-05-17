from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.blocks import ResidualBlock, init_weights
from ai.models.common import BandOutput, ForecastOutput, LineDistributionalOutput, LineMonotonicDistributionalOutput, LineRegimeOutput, LineV2Output, LineWarningOutput, MODEL_OUTPUT_ROLES, _split_band


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
        output_role: str = "legacy",
        quantile_count: int = 10,
        quantile_center_index: int = 6,
    ) -> None:
        super().__init__()
        if output_role not in MODEL_OUTPUT_ROLES:
            raise ValueError(f"지원하지 않는 output_role입니다: {output_role}")
        self.seq_len = seq_len
        self.horizon = horizon
        self.band_mode = band_mode
        self.output_role = output_role
        self.quantile_count = int(quantile_count)
        self.quantile_center_index = max(0, min(int(quantile_center_index), self.quantile_count - 1))
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
        if output_role in ("legacy", "line_v2", "line_regime"):
            self.line_head = nn.Linear(self.temporal_hidden_dim, 1)
        if output_role in ("legacy", "band"):
            self.band_head = nn.Linear(self.temporal_hidden_dim, 2)
        if output_role == "line_v2":
            self.downside_risk_head = nn.Linear(self.temporal_hidden_dim, 1)
        if output_role == "line_regime":
            self.regime_head = nn.Linear(self.temporal_hidden_dim, 5)
        if output_role == "line_warning":
            self.warning_head = nn.Linear(self.temporal_hidden_dim, 1)
        if output_role == "line_distributional":
            self.quantile_head = nn.Linear(self.temporal_hidden_dim, self.quantile_count)
        if output_role == "line_distributional_mono":
            self.line_head = nn.Linear(self.temporal_hidden_dim, 1)
            lower_count = self.quantile_center_index
            upper_count = self.quantile_count - self.quantile_center_index - 1
            self.lower_offset_head = nn.Linear(self.temporal_hidden_dim, lower_count) if lower_count > 0 else None
            self.upper_offset_head = nn.Linear(self.temporal_hidden_dim, upper_count) if upper_count > 0 else None
        self.lookback_skip = nn.Linear(seq_len, horizon)
        self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim) if self.use_ticker_embedding else None
        self.last_temporal_hidden_shape: tuple[int, ...] | None = None
        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        ticker_id: torch.Tensor | None = None,
        future_covariate: torch.Tensor | None = None,
    ) -> ForecastOutput | LineV2Output | LineRegimeOutput | LineWarningOutput | LineDistributionalOutput | LineMonotonicDistributionalOutput | BandOutput:
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

        baseline_series = x[..., self.lookback_baseline_idx]
        skip = self.lookback_skip(baseline_series)
        if self.output_role == "line_v2":
            line = self.line_head(temporal_hidden).squeeze(-1)
            risk_logit = self.downside_risk_head(temporal_hidden).squeeze(-1)
            return LineV2Output(line=line + skip, downside_risk_logit=risk_logit)

        if self.output_role == "line_regime":
            line = self.line_head(temporal_hidden).squeeze(-1)
            regime_logits = self.regime_head(temporal_hidden[:, -1, :])
            return LineRegimeOutput(line=line + skip, regime_logits=regime_logits)

        if self.output_role == "line_warning":
            return LineWarningOutput(warning_logit=self.warning_head(temporal_hidden[:, -1, :]).squeeze(-1))

        if self.output_role == "line_distributional":
            quantiles = self.quantile_head(temporal_hidden)
            return LineDistributionalOutput(quantiles=quantiles + skip.unsqueeze(-1))

        if self.output_role == "line_distributional_mono":
            line = self.line_head(temporal_hidden).squeeze(-1) + skip
            pieces: list[torch.Tensor] = []
            lower_count = self.quantile_center_index
            upper_count = self.quantile_count - self.quantile_center_index - 1
            if lower_count > 0:
                lower_steps = torch.nn.functional.softplus(self.lower_offset_head(temporal_hidden)) + 1e-6
                lower_near_to_far = line.unsqueeze(-1) - torch.cumsum(lower_steps, dim=-1)
                pieces.append(torch.flip(lower_near_to_far, dims=(-1,)))
            pieces.append(line.unsqueeze(-1))
            if upper_count > 0:
                upper_steps = torch.nn.functional.softplus(self.upper_offset_head(temporal_hidden)) + 1e-6
                pieces.append(line.unsqueeze(-1) + torch.cumsum(upper_steps, dim=-1))
            return LineMonotonicDistributionalOutput(line=line, quantiles=torch.cat(pieces, dim=-1))

        band_raw = self.band_head(temporal_hidden)
        lower_band, upper_band = _split_band(band_raw, self.band_mode)
        if self.output_role == "band":
            return BandOutput(
                lower_band=lower_band + skip,
                upper_band=upper_band + skip,
            )

        line = self.line_head(temporal_hidden).squeeze(-1)
        return ForecastOutput(
            line=line + skip,
            lower_band=lower_band + skip,
            upper_band=upper_band + skip,
        )
