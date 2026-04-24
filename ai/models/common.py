from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ForecastOutput:
    """공통 backbone + 다중 head 구조의 출력 형식."""

    line: torch.Tensor
    lower_band: torch.Tensor
    upper_band: torch.Tensor

    @property
    def bands(self) -> torch.Tensor:
        return torch.stack((self.lower_band, self.upper_band), dim=-1)


class MultiHeadForecastModel(nn.Module):
    """공통 hidden 표현에서 예측선과 밴드 출력을 분기한다."""

    def __init__(self, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.line_head = nn.Linear(hidden_dim, horizon)
        self.band_head = nn.Linear(hidden_dim, horizon * 2)

    def build_output(self, hidden: torch.Tensor) -> ForecastOutput:
        line = self.line_head(hidden)
        band = self.band_head(hidden).view(hidden.size(0), self.horizon, 2)
        return ForecastOutput(
            line=line,
            lower_band=band[..., 0],
            upper_band=band[..., 1],
        )
