from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.common import ForecastOutput


class AsymmetricHuberLoss(nn.Module):
    """상방 오차에 더 큰 페널티를 주는 Huber loss."""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, delta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        huber = 0.5 * quadratic.pow(2) + self.delta * linear
        weight = torch.where(error > 0, torch.full_like(error, self.beta), torch.full_like(error, self.alpha))
        return (weight * huber).mean()


class AsymmetricBCELoss(nn.Module):
    """오버슈트 쪽 오차에 더 큰 가중을 주는 BCE."""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clipped = prediction.clamp(self.eps, 1.0 - self.eps)
        positive = -self.alpha * target * torch.log(clipped)
        negative = -self.beta * (1.0 - target) * torch.log(1.0 - clipped)
        return (positive + negative).mean()


class PinballLoss(nn.Module):
    """지정한 quantile 묶음에 대한 pinball loss."""

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float] | float = (0.1, 0.5, 0.9),
        *,
        sort_quantiles: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(quantiles, float):
            quantiles = (quantiles,)
        self.quantiles = tuple(float(quantile) for quantile in quantiles)
        self.sort_quantiles = sort_quantiles

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.dim() == target.dim():
            prediction = prediction.unsqueeze(-1)
        if self.sort_quantiles:
            prediction = torch.sort(prediction, dim=-1).values

        quantile_tensor = prediction.new_tensor(self.quantiles).view(*([1] * (prediction.dim() - 1)), -1)
        expanded_target = target.unsqueeze(-1).expand_as(prediction)
        diff = expanded_target - prediction
        return torch.maximum(quantile_tensor * diff, (quantile_tensor - 1.0) * diff).mean()


class WidthPenaltyLoss(nn.Module):
    """밴드 폭이 음수로 뒤집히지 않도록 relu 기반 폭만 벌점으로 쓴다."""

    def forward(self, lower_band: torch.Tensor, upper_band: torch.Tensor) -> torch.Tensor:
        return F.relu(upper_band - lower_band).mean()


class BandCrossPenaltyLoss(nn.Module):
    """하단 밴드가 상단 밴드를 넘는 경우에만 벌점을 준다."""

    def forward(self, lower_band: torch.Tensor, upper_band: torch.Tensor) -> torch.Tensor:
        return torch.relu(lower_band - upper_band).mean()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    line: torch.Tensor
    band: torch.Tensor
    width: torch.Tensor
    cross: torch.Tensor

    def to_log_dict(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach().cpu()),
            "line_loss": float(self.line.detach().cpu()),
            "band_loss": float(self.band.detach().cpu()),
            "width_loss": float(self.width.detach().cpu()),
            "cross_loss": float(self.cross.detach().cpu()),
        }


class ForecastCompositeLoss(nn.Module):
    """선 예측과 밴드, 폭, 교차 벌점을 묶어 계산한다."""

    def __init__(
        self,
        *,
        q_low: float = 0.1,
        q_high: float = 0.9,
        alpha: float = 1.0,
        beta: float = 2.0,
        delta: float = 1.0,
        lambda_line: float = 1.0,
        lambda_band: float = 1.0,
        lambda_width: float = 0.1,
        lambda_cross: float = 1.0,
        band_mode: str = "direct",
    ) -> None:
        super().__init__()
        self.band_mode = band_mode
        self.line_loss = AsymmetricHuberLoss(alpha=alpha, beta=beta, delta=delta)
        self.low_quantile_loss = PinballLoss((q_low,))
        self.high_quantile_loss = PinballLoss((q_high,))
        self.width_loss = WidthPenaltyLoss()
        self.cross_loss = BandCrossPenaltyLoss()
        self.lambda_line = lambda_line
        self.lambda_band = lambda_band
        self.lambda_width = lambda_width
        self.lambda_cross = lambda_cross

    def forward(
        self,
        prediction: ForecastOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
    ) -> LossBreakdown:
        line_component = self.line_loss(prediction.line, line_target)
        band_component = self.low_quantile_loss(prediction.lower_band, band_target) + self.high_quantile_loss(
            prediction.upper_band,
            band_target,
        )
        width_component = self.width_loss(prediction.lower_band, prediction.upper_band)
        if self.band_mode == "direct":
            cross_component = self.cross_loss(prediction.lower_band, prediction.upper_band)
        else:
            cross_component = torch.zeros_like(width_component)
        total = (
            (self.lambda_line * line_component)
            + (self.lambda_band * band_component)
            + (self.lambda_width * width_component)
            + (self.lambda_cross * cross_component)
        )
        return LossBreakdown(
            total=total,
            line=line_component,
            band=band_component,
            width=width_component,
            cross=cross_component,
        )
