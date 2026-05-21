from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_OUTPUT_ROLES = ("legacy", "line_v2", "line_regime", "line_warning", "line_distributional", "line_distributional_mono", "band")


@dataclass
class ForecastOutput:
    """레거시 line+band 통합 출력 계약."""

    line: torch.Tensor
    lower_band: torch.Tensor
    upper_band: torch.Tensor
    direction_logit: torch.Tensor | None = None

    @property
    def bands(self) -> torch.Tensor:
        return torch.stack((self.lower_band, self.upper_band), dim=-1)


@dataclass
class LineV2Output:
    """line v2 전용 출력 계약.

    line은 랭킹/예측선 신호이고 downside_risk_logit은 별도 하방 위험 head다.
    밴드 출력은 의도적으로 포함하지 않는다.
    """

    line: torch.Tensor
    downside_risk_logit: torch.Tensor

    @property
    def downside_risk_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.downside_risk_logit)


@dataclass
class LineRegimeOutput:
    """line v2.1 전용 출력 계약.

    line은 기존 h-step 보수적 점추정이고, regime_logits는 h5 누적수익률의
    5단계 구간을 예측하는 단일 분류 head다. band 출력은 포함하지 않는다.
    """

    line: torch.Tensor
    regime_logits: torch.Tensor

    @property
    def regime_prob(self) -> torch.Tensor:
        return torch.softmax(self.regime_logits, dim=-1)


@dataclass
class LineWarningOutput:
    """line warning 전용 출력 계약.

    warning_logit은 line 예측선을 대체하지 않는 독립 guard layer 출력이다.
    line/band payload와 섞이지 않도록 line, lower_band, upper_band를 포함하지 않는다.
    """

    warning_logit: torch.Tensor

    @property
    def warning_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.warning_logit)


@dataclass
class LineDistributionalOutput:
    """line v3 분포형 예측선 출력 계약.

    quantiles는 (batch, horizon, quantile_count) 형태를 유지한다.
    line/band 저장 payload와 섞이지 않도록 lower/upper band 필드는 두지 않는다.
    """

    quantiles: torch.Tensor


@dataclass
class LineMonotonicDistributionalOutput:
    """line v3 monotonic 분포형 출력 계약.

    line은 q50 또는 beta location 역할을 하는 중심선이고,
    quantiles는 softplus 양수 offset 누적으로 생성되어 quantile crossing이
    구조적으로 발생하지 않는다.
    """

    line: torch.Tensor
    quantiles: torch.Tensor


@dataclass
class BandOutput:
    """band 전용 출력 계약. line 예측선은 의도적으로 포함하지 않는다."""

    lower_band: torch.Tensor
    upper_band: torch.Tensor

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
    """공통 hidden 표현에서 역할별 head를 구성한다.

    legacy는 기존 checkpoint 호환을 위해 line과 band를 함께 둔다.
    신규 실험은 line_v2 또는 band 역할을 명시해 head 자체를 분리한다.
    """

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        band_mode: str = "direct",
        output_role: str = "legacy",
        quantile_count: int = 10,
        quantile_center_index: int = 6,
    ) -> None:
        super().__init__()
        if output_role not in MODEL_OUTPUT_ROLES:
            raise ValueError(f"지원하지 않는 output_role입니다: {output_role}")
        self.horizon = horizon
        self.band_mode = band_mode
        self.output_role = output_role
        self.quantile_count = int(quantile_count)
        self.quantile_center_index = max(0, min(int(quantile_center_index), self.quantile_count - 1))
        if output_role in ("legacy", "line_v2", "line_regime"):
            self.line_head = nn.Linear(hidden_dim, horizon)
        if output_role == "line_distributional_mono":
            self.line_head = nn.Linear(hidden_dim, horizon)
        if output_role in ("legacy", "band"):
            self.band_head = nn.Linear(hidden_dim, horizon * 2)
        if output_role == "line_v2":
            self.downside_risk_head = nn.Linear(hidden_dim, horizon)
        if output_role == "line_regime":
            self.regime_head = nn.Linear(hidden_dim, 5)
        if output_role == "line_warning":
            self.warning_head = nn.Linear(hidden_dim, 1)
        if output_role == "line_distributional":
            self.quantile_head = nn.Linear(hidden_dim, horizon * self.quantile_count)
        if output_role == "line_distributional_mono":
            lower_count = self.quantile_center_index
            upper_count = self.quantile_count - self.quantile_center_index - 1
            self.lower_offset_head = nn.Linear(hidden_dim, horizon * lower_count) if lower_count > 0 else None
            self.upper_offset_head = nn.Linear(hidden_dim, horizon * upper_count) if upper_count > 0 else None

    def _build_monotonic_quantiles(self, hidden: torch.Tensor) -> LineMonotonicDistributionalOutput:
        line = self.line_head(hidden)
        pieces: list[torch.Tensor] = []
        lower_count = self.quantile_center_index
        upper_count = self.quantile_count - self.quantile_center_index - 1
        if lower_count > 0:
            lower_raw = self.lower_offset_head(hidden).view(hidden.size(0), self.horizon, lower_count)
            # raw 순서는 중심선에 가까운 하위 quantile부터 두고, 누적 후 뒤집어 낮은 quantile부터 정렬한다.
            lower_steps = F.softplus(lower_raw) + 1e-6
            lower_near_to_far = line.unsqueeze(-1) - torch.cumsum(lower_steps, dim=-1)
            pieces.append(torch.flip(lower_near_to_far, dims=(-1,)))
        pieces.append(line.unsqueeze(-1))
        if upper_count > 0:
            upper_raw = self.upper_offset_head(hidden).view(hidden.size(0), self.horizon, upper_count)
            upper_steps = F.softplus(upper_raw) + 1e-6
            pieces.append(line.unsqueeze(-1) + torch.cumsum(upper_steps, dim=-1))
        return LineMonotonicDistributionalOutput(line=line, quantiles=torch.cat(pieces, dim=-1))

    def _build_band(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        band_raw = self.band_head(hidden).view(hidden.size(0), self.horizon, 2)
        return _split_band(band_raw, self.band_mode)

    def build_output(self, hidden: torch.Tensor) -> ForecastOutput | LineV2Output | LineRegimeOutput | LineWarningOutput | LineDistributionalOutput | LineMonotonicDistributionalOutput | BandOutput:
        if self.output_role == "line_v2":
            return LineV2Output(
                line=self.line_head(hidden),
                downside_risk_logit=self.downside_risk_head(hidden),
            )
        if self.output_role == "line_regime":
            return LineRegimeOutput(
                line=self.line_head(hidden),
                regime_logits=self.regime_head(hidden),
            )
        if self.output_role == "line_warning":
            return LineWarningOutput(warning_logit=self.warning_head(hidden).squeeze(-1))
        if self.output_role == "line_distributional":
            quantiles = self.quantile_head(hidden).view(hidden.size(0), self.horizon, self.quantile_count)
            return LineDistributionalOutput(quantiles=quantiles)
        if self.output_role == "line_distributional_mono":
            return self._build_monotonic_quantiles(hidden)
        if self.output_role == "band":
            lower_band, upper_band = self._build_band(hidden)
            return BandOutput(lower_band=lower_band, upper_band=upper_band)
        line = self.line_head(hidden)
        lower_band, upper_band = self._build_band(hidden)
        return ForecastOutput(
            line=line,
            lower_band=lower_band,
            upper_band=upper_band,
        )
