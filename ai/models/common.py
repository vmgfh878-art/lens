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
    """공통 hidden 표현에서 선 예측과 band 예측을 분기한다."""

    def __init__(self, hidden_dim: int, horizon: int, band_mode: str = "direct") -> None:
        super().__init__()
        self.horizon = horizon
        self.band_mode = band_mode
        self.line_head = nn.Linear(hidden_dim, horizon)
        self.band_head = nn.Linear(hidden_dim, horizon * 2)

    def build_output(self, hidden: torch.Tensor) -> ForecastOutput:
        line = self.line_head(hidden)
        band_raw = self.band_head(hidden).view(hidden.size(0), self.horizon, 2)
        if self.band_mode == "direct":
            lower_band = band_raw[..., 0]
            upper_band = band_raw[..., 1]
        elif self.band_mode == "param":
            center = band_raw[..., 0]
            log_half_width = band_raw[..., 1]
            half_width = torch.exp(log_half_width)
            lower_band = center - half_width
            upper_band = center + half_width
        else:
            raise ValueError(f"지원하지 않는 band_mode입니다: {self.band_mode}")

        return ForecastOutput(
            line=line,
            lower_band=lower_band,
            upper_band=upper_band,
        )
