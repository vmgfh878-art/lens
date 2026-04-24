from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.common import ForecastOutput, MultiHeadForecastModel


class PatchTST(MultiHeadForecastModel):
    """간결한 PatchTST 스타일 backbone 위에 line/band head를 얹은 모델."""

    def __init__(
        self,
        n_features: int = 17,
        seq_len: int = 60,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        horizon: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(hidden_dim=d_model, horizon=horizon)
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(n_features * patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self._num_patches(seq_len), d_model) * 0.02)
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

    def _num_patches(self, seq_len: int) -> int:
        return ((seq_len - self.patch_len) // self.stride) + 1

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        # 최근 시계열을 patch 단위로 나눈 뒤 Transformer backbone에 통과시킨다.
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(x.size(0), patches.size(1), -1)
        encoded = self.patch_proj(patches) + self.position_embedding[:, : patches.size(1)]
        hidden = self.encoder(encoded).mean(dim=1)
        hidden = self.norm(hidden)
        return self.build_output(hidden)
