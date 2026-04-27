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
    direction_logit: torch.Tensor | None = None

    @property
    def bands(self) -> torch.Tensor:
        return torch.stack((self.lower_band, self.upper_band), dim=-1)


def _split_band(band_raw: torch.Tensor, band_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """band_mode 규칙에 따라 밴드 원시 출력을 lower/upper로 분리한다."""
    if band_mode == "direct":
        return band_raw[..., 0], band_raw[..., 1]
    if band_mode == "param":
        center = band_raw[..., 0]
        log_half_width = band_raw[..., 1]
        half_width = torch.exp(log_half_width)
        return center - half_width, center + half_width
    raise ValueError(f"지원하지 않는 band_mode입니다: {band_mode}")


class MultiHeadForecastModel(nn.Module):
    """공통 hidden 표현에서 line 예측과 band 예측을 분기한다."""

    def __init__(self, hidden_dim: int, horizon: int, band_mode: str = "direct") -> None:
        super().__init__()
        self.horizon = horizon
        self.band_mode = band_mode
        self.line_head = nn.Linear(hidden_dim, horizon)
        self.band_head = nn.Linear(hidden_dim, horizon * 2)

    def build_output(self, hidden: torch.Tensor) -> ForecastOutput:
        line = self.line_head(hidden)
        band_raw = self.band_head(hidden).view(hidden.size(0), self.horizon, 2)
        lower_band, upper_band = _split_band(band_raw, self.band_mode)
        return ForecastOutput(
            line=line,
            lower_band=lower_band,
            upper_band=upper_band,
        )
