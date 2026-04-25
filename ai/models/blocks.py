from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    """모델 전체에 BERT 스타일 trunc_normal 초기화를 적용한다."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
        return
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.trunc_normal_(param, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)


class AttentionPooling1D(nn.Module):
    """시계열 축 전체를 attention 가중 평균으로 요약한다."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        self.last_weights = weights.detach()
        return (x * weights).sum(dim=1)


class ResidualBlock(nn.Module):
    """TiDE 계열에서 쓰는 선형 residual block."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        inner_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return self.norm(x + hidden)
