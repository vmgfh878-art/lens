from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.blocks import init_weights
from ai.models.common import ForecastOutput, MultiHeadForecastModel
from ai.models.revin import RevIN


class PatchTST(MultiHeadForecastModel):
    """RevIN과 channel independence를 포함한 PatchTST 계열 모델."""

    def __init__(
        self,
        n_features: int = 29,
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
    ) -> None:
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.channel_independent = channel_independent
        self.use_revin = use_revin
        self.n_patches = self._num_patches(seq_len, patch_len, stride)
        super().__init__(hidden_dim=d_model * self.n_patches, horizon=horizon, band_mode=band_mode)
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
        self.apply(init_weights)

    @staticmethod
    def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
        return ((seq_len - patch_len) // stride) + 1

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        batch_size, seq_len, channel_count = x.shape
        hidden_input = x
        if self.use_revin and self.revin is not None:
            hidden_input = self.revin(hidden_input, mode="norm")

        if self.channel_independent:
            channel_major = hidden_input.permute(0, 2, 1).reshape(batch_size * channel_count, seq_len)
            patches = channel_major.unfold(dimension=1, size=self.patch_len, step=self.stride)
            encoded = self.patch_proj(patches)
            encoded = encoded + self.position_embedding[:, : encoded.size(1)]
            encoded = self.input_dropout(encoded)
            encoded = self.encoder(encoded)
            encoded = self.norm(encoded)
            encoded = self.output_dropout(encoded)
            flattened = encoded.reshape(batch_size * channel_count, -1)
            per_channel_output = self.build_output(flattened)
            return ForecastOutput(
                line=per_channel_output.line.view(batch_size, channel_count, self.horizon).mean(dim=1),
                lower_band=per_channel_output.lower_band.view(batch_size, channel_count, self.horizon).mean(dim=1),
                upper_band=per_channel_output.upper_band.view(batch_size, channel_count, self.horizon).mean(dim=1),
            )

        patches = hidden_input.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(batch_size, self.n_patches, self.patch_len * channel_count)
        encoded = self.mixed_patch_proj(patches)
        encoded = encoded + self.position_embedding[:, : encoded.size(1)]
        encoded = self.input_dropout(encoded)
        encoded = self.encoder(encoded)
        encoded = self.norm(encoded)
        encoded = self.output_dropout(encoded)
        flattened = encoded.reshape(batch_size, -1)
        return self.build_output(flattened)
