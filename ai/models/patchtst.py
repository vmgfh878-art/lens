from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.blocks import ChannelAttentionPooling, init_weights
from ai.models.common import ForecastOutput, MultiHeadForecastModel
from ai.models.revin import RevIN


class PatchTST(MultiHeadForecastModel):
    """RevIN과 channel independence를 포함한 PatchTST 계열 모델."""

    def __init__(
        self,
        n_features: int = 36,
        seq_len: int = 252,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        horizon: int = 5,
        dropout: float = 0.1,
        band_mode: str = "direct",
        use_revin: bool = True,
        channel_independent: bool = True,
        target_channel_idx: int = 0,
        ci_aggregate: str = "target",
        ci_target_fast: bool = False,
        num_tickers: int = 0,
        ticker_emb_dim: int = 32,
    ) -> None:
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.channel_independent = channel_independent
        self.use_revin = use_revin
        self.target_channel_idx = target_channel_idx
        self.ci_aggregate = ci_aggregate
        self.ci_target_fast = ci_target_fast
        self.n_patches = self._num_patches(seq_len, patch_len, stride)
        self.use_ticker_embedding = num_tickers > 0
        ticker_extra = ticker_emb_dim if self.use_ticker_embedding else 0
        super().__init__(hidden_dim=d_model * self.n_patches + ticker_extra, horizon=horizon, band_mode=band_mode)
        self.revin = RevIN(n_features) if use_revin else None
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.mixed_patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.ticker_embedding = nn.Embedding(num_tickers + 1, ticker_emb_dim) if self.use_ticker_embedding else None
        self.channel_attn = ChannelAttentionPooling(n_features) if ci_aggregate == "attention" else None
        self.apply(init_weights)

    @staticmethod
    def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
        return ((seq_len - patch_len) // stride) + 1

    def _aggregate_channels(self, hidden_per_channel: torch.Tensor) -> torch.Tensor:
        if self.ci_target_fast and self.ci_aggregate == "target":
            return hidden_per_channel[:, 0, :]
        if self.ci_aggregate == "target":
            return hidden_per_channel[:, self.target_channel_idx, :]
        if self.ci_aggregate == "mean":
            return hidden_per_channel.mean(dim=1)
        if self.ci_aggregate == "attention" and self.channel_attn is not None:
            return self.channel_attn(hidden_per_channel)
        raise ValueError(f"지원하지 않는 ci_aggregate 입니다: {self.ci_aggregate}")

    def _concat_ticker_embedding(self, hidden: torch.Tensor, ticker_id: torch.Tensor | None) -> torch.Tensor:
        if not self.use_ticker_embedding:
            return hidden
        if ticker_id is None:
            raise ValueError("ticker embedding이 활성화된 모델은 ticker_id가 필요합니다.")
        return torch.cat((hidden, self.ticker_embedding(ticker_id)), dim=-1)

    def _maybe_denormalize_output(self, output: ForecastOutput) -> ForecastOutput:
        if not self.use_revin or self.revin is None:
            return output
        return ForecastOutput(
            line=self.revin.denormalize_target(output.line, self.target_channel_idx),
            lower_band=self.revin.denormalize_target(output.lower_band, self.target_channel_idx),
            upper_band=self.revin.denormalize_target(output.upper_band, self.target_channel_idx),
        )

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> ForecastOutput:
        batch_size, seq_len, channel_count = x.shape
        hidden_input = x
        if self.use_revin and self.revin is not None:
            hidden_input = self.revin(hidden_input, mode="norm")

        if self.channel_independent:
            if self.ci_target_fast and self.ci_aggregate == "target":
                hidden_input = hidden_input[..., self.target_channel_idx : self.target_channel_idx + 1]
                channel_count = 1
            channel_major = hidden_input.permute(0, 2, 1).reshape(batch_size * channel_count, seq_len)
            patches = channel_major.unfold(dimension=1, size=self.patch_len, step=self.stride)
            encoded = self.patch_proj(patches)
            encoded = encoded + self.position_embedding[:, : encoded.size(1)]
            encoded = self.input_dropout(encoded)
            encoded = self.encoder(encoded)
            encoded = self.norm(encoded)
            encoded = self.output_dropout(encoded)
            flattened = encoded.reshape(batch_size, channel_count, -1)
            hidden = self._aggregate_channels(flattened)
            hidden = self._concat_ticker_embedding(hidden, ticker_id)
            return self._maybe_denormalize_output(self.build_output(hidden))

        patches = hidden_input.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(batch_size, self.n_patches, self.patch_len * channel_count)
        encoded = self.mixed_patch_proj(patches)
        encoded = encoded + self.position_embedding[:, : encoded.size(1)]
        encoded = self.input_dropout(encoded)
        encoded = self.encoder(encoded)
        encoded = self.norm(encoded)
        encoded = self.output_dropout(encoded)
        hidden = encoded.reshape(batch_size, -1)
        hidden = self._concat_ticker_embedding(hidden, ticker_id)
        return self._maybe_denormalize_output(self.build_output(hidden))
