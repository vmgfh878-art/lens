from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ai.models.common import BandOutput, ForecastOutput, LineDistributionalOutput, LineMonotonicDistributionalOutput, LineRegimeOutput, LineV2Output, LineWarningOutput


class AsymmetricHuberLoss(nn.Module):
    """과대 예측에 더 큰 벌점을 주는 Huber loss."""

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
    """오버슈트 쪽에 더 큰 가중치를 주는 BCE."""

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


class AsymmetricBCEWithLogitsLoss(nn.Module):
    """하락 구간 오판을 더 크게 보는 BCEWithLogits 손실."""

    def __init__(self, negative_weight: float = 2.0, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight

    def forward(self, prediction_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            target > 0.5,
            torch.full_like(target, self.positive_weight),
            torch.full_like(target, self.negative_weight),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction_logit,
            target,
            reduction="none",
        )
        return (loss * weights).mean()


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


class BandCrossPenaltyLoss(nn.Module):
    """하단 밴드가 상단 밴드를 넘는 경우에만 벌점을 준다."""

    def forward(self, lower_band: torch.Tensor, upper_band: torch.Tensor) -> torch.Tensor:
        return torch.relu(lower_band - upper_band).mean()


class QuantileCrossingPenaltyLoss(nn.Module):
    """분포형 line quantile 순서가 뒤집힌 경우에만 벌점을 준다."""

    def forward(self, quantiles: torch.Tensor) -> torch.Tensor:
        return torch.relu(quantiles[..., :-1] - quantiles[..., 1:]).mean()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    forecast: torch.Tensor
    line: torch.Tensor
    band: torch.Tensor
    cross: torch.Tensor
    direction: torch.Tensor
    risk: torch.Tensor | None = None
    regime: torch.Tensor | None = None
    quantile: torch.Tensor | None = None

    def to_log_dict(self) -> dict[str, float]:
        risk = self.risk if self.risk is not None else torch.zeros_like(self.total)
        regime = self.regime if self.regime is not None else torch.zeros_like(self.total)
        quantile = self.quantile if self.quantile is not None else torch.zeros_like(self.total)
        return {
            "total_loss": float(self.total.detach().cpu()),
            "forecast_loss": float(self.forecast.detach().cpu()),
            "line_loss": float(self.line.detach().cpu()),
            "band_loss": float(self.band.detach().cpu()),
            "cross_loss": float(self.cross.detach().cpu()),
            "direction_loss": float(self.direction.detach().cpu()),
            "risk_loss": float(risk.detach().cpu()),
            "regime_loss": float(regime.detach().cpu()),
            "quantile_loss": float(quantile.detach().cpu()),
        }


class ForecastCompositeLoss(nn.Module):
    """레거시 line+band 출력용 손실.

    lambda_width는 과거 checkpoint/config 호환을 위해 인자로 받지만,
    Phase 1 현재 손실 계산에는 사용하지 않는다.
    """

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
        lambda_direction: float = 0.1,
        band_mode: str = "direct",
        lower_band_loss_weight: float = 1.0,
        upper_band_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.band_mode = band_mode
        self.line_loss = AsymmetricHuberLoss(alpha=alpha, beta=beta, delta=delta)
        self.low_quantile_loss = PinballLoss((q_low,))
        self.high_quantile_loss = PinballLoss((q_high,))
        self.cross_loss = BandCrossPenaltyLoss()
        self.direction_loss = AsymmetricBCEWithLogitsLoss(negative_weight=2.0, positive_weight=1.0)
        self.lambda_line = lambda_line
        self.lambda_band = lambda_band
        # 레거시 width loss 호환용 설정값이다. 현재 손실 계산에는 사용하지 않는다.
        self.lambda_width = lambda_width
        self.lambda_cross = lambda_cross
        self.lambda_direction = lambda_direction
        self.lower_band_loss_weight = lower_band_loss_weight
        self.upper_band_loss_weight = upper_band_loss_weight

    def forward(
        self,
        prediction: ForecastOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        if not isinstance(prediction, ForecastOutput):
            raise TypeError("ForecastCompositeLoss는 ForecastOutput만 받을 수 있습니다.")
        line_component = self.line_loss(prediction.line, line_target)
        lower_band_component = self.low_quantile_loss(prediction.lower_band, band_target)
        upper_band_component = self.high_quantile_loss(
            prediction.upper_band,
            band_target,
        )
        band_component = (
            (self.lower_band_loss_weight * lower_band_component)
            + (self.upper_band_loss_weight * upper_band_component)
        )
        if self.band_mode == "direct":
            cross_component = self.cross_loss(prediction.lower_band, prediction.upper_band)
        else:
            cross_component = torch.zeros_like(line_component)
        forecast_component = (
            (self.lambda_line * line_component)
            + (self.lambda_band * band_component)
            + (self.lambda_cross * cross_component)
        )
        if prediction.direction_logit is not None and raw_future_returns is not None:
            direction_target = (raw_future_returns > 0.0).to(dtype=prediction.direction_logit.dtype)
            direction_component = self.direction_loss(prediction.direction_logit, direction_target)
        else:
            direction_component = torch.zeros_like(line_component)
        total = forecast_component + (self.lambda_direction * direction_component)
        return LossBreakdown(
            total=total,
            forecast=forecast_component,
            line=line_component,
            band=band_component,
            cross=cross_component,
            direction=direction_component,
        )


class LineV2Loss(nn.Module):
    """line v2용 손실.

    예측선 손실과 하방 위험 BCE를 분리해 계산한다.
    밴드 pinball 손실은 포함하지 않는다.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        delta: float = 1.0,
        lambda_line: float = 1.0,
        lambda_risk: float = 0.5,
        downside_threshold: float = -0.05,
        risk_positive_weight: float = 1.0,
        risk_negative_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.line_loss = AsymmetricHuberLoss(alpha=alpha, beta=beta, delta=delta)
        self.lambda_line = lambda_line
        self.lambda_risk = lambda_risk
        self.downside_threshold = downside_threshold
        self.risk_positive_weight = risk_positive_weight
        self.risk_negative_weight = risk_negative_weight

    def forward(
        self,
        prediction: LineV2Output,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del band_target
        if not isinstance(prediction, LineV2Output):
            raise TypeError("LineV2Loss는 LineV2Output만 받을 수 있습니다.")
        if raw_future_returns is None:
            raise ValueError("LineV2Loss에는 raw_future_returns가 필요합니다.")
        line_component = self.line_loss(prediction.line, line_target)
        risk_target = (raw_future_returns <= float(self.downside_threshold)).to(dtype=prediction.downside_risk_logit.dtype)
        risk_weight = torch.where(
            risk_target > 0.5,
            torch.full_like(risk_target, float(self.risk_positive_weight)),
            torch.full_like(risk_target, float(self.risk_negative_weight)),
        )
        risk_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.downside_risk_logit,
            risk_target,
            reduction="none",
        )
        risk_component = (risk_raw * risk_weight).mean()
        zero = torch.zeros_like(line_component)
        forecast_component = (self.lambda_line * line_component) + (self.lambda_risk * risk_component)
        return LossBreakdown(
            total=forecast_component,
            forecast=forecast_component,
            line=line_component,
            band=zero,
            cross=zero,
            direction=zero,
            risk=risk_component,
        )


class LineRegimeLoss(nn.Module):
    """line v2.1 손실.

    line은 기존 비대칭 Huber 손실을 그대로 쓰고, regime은 h5 누적수익률
    구간을 맞히는 cross entropy와 ordinal distance penalty를 함께 쓴다.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        delta: float = 1.0,
        lambda_line: float = 1.0,
        lambda_regime: float = 0.5,
        lambda_ordinal: float = 0.1,
        regime_thresholds: tuple[float, float, float, float] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.line_loss = AsymmetricHuberLoss(alpha=alpha, beta=beta, delta=delta)
        self.lambda_line = float(lambda_line)
        self.lambda_regime = float(lambda_regime)
        self.lambda_ordinal = float(lambda_ordinal)
        self.regime_thresholds = tuple(float(value) for value in (regime_thresholds or (-0.05, -0.01, 0.01, 0.05)))
        if len(self.regime_thresholds) != 4:
            raise ValueError("line_regime regime_thresholds는 q10/q35/q65/q90 네 개가 필요합니다.")

    def _regime_target(self, raw_future_returns: torch.Tensor) -> torch.Tensor:
        h5_return = raw_future_returns[:, -1]
        thresholds = h5_return.new_tensor(self.regime_thresholds)
        target = torch.zeros_like(h5_return, dtype=torch.long)
        target = target + (h5_return > thresholds[0]).to(torch.long)
        target = target + (h5_return > thresholds[1]).to(torch.long)
        target = target + (h5_return > thresholds[2]).to(torch.long)
        target = target + (h5_return > thresholds[3]).to(torch.long)
        return target

    def forward(
        self,
        prediction: LineRegimeOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del band_target
        if not isinstance(prediction, LineRegimeOutput):
            raise TypeError("LineRegimeLoss는 LineRegimeOutput만 받을 수 있습니다.")
        if raw_future_returns is None:
            raise ValueError("LineRegimeLoss에는 raw_future_returns가 필요합니다.")
        line_component = self.line_loss(prediction.line, line_target)
        regime_target = self._regime_target(raw_future_returns)
        regime_component = torch.nn.functional.cross_entropy(prediction.regime_logits, regime_target)
        probability = torch.softmax(prediction.regime_logits, dim=-1)
        class_index = torch.arange(5, device=prediction.regime_logits.device, dtype=probability.dtype).view(1, 5)
        distance_sq = (class_index - regime_target.to(probability.dtype).view(-1, 1)).pow(2)
        ordinal_component = (probability * distance_sq).sum(dim=-1).mean()
        zero = torch.zeros_like(line_component)
        forecast_component = (
            (self.lambda_line * line_component)
            + (self.lambda_regime * regime_component)
            + (self.lambda_ordinal * ordinal_component)
        )
        return LossBreakdown(
            total=forecast_component,
            forecast=forecast_component,
            line=line_component,
            band=zero,
            cross=zero,
            direction=zero,
            risk=zero,
            regime=regime_component + ordinal_component,
        )


class LineWarningLoss(nn.Module):
    """line warning 전용 손실.

    line_score를 입력으로 쓰지 않고, h5 실제 수익률이 downside threshold 이하인지
    독립 warning logit으로 학습한다. line/band 손실과 섞지 않는다.
    """

    def __init__(
        self,
        *,
        downside_threshold: float = -0.03,
        positive_weight: float = 5.0,
        gamma: float = 2.0,
        use_focal: bool = True,
    ) -> None:
        super().__init__()
        self.downside_threshold = float(downside_threshold)
        self.positive_weight = float(positive_weight)
        self.gamma = float(gamma)
        self.use_focal = bool(use_focal)

    def forward(
        self,
        prediction: LineWarningOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del line_target, band_target
        if not isinstance(prediction, LineWarningOutput):
            raise TypeError("LineWarningLoss는 LineWarningOutput만 받을 수 있습니다.")
        if raw_future_returns is None:
            raise ValueError("LineWarningLoss에는 raw_future_returns가 필요합니다.")
        target = (raw_future_returns[:, -1] <= self.downside_threshold).to(dtype=prediction.warning_logit.dtype)
        logit = prediction.warning_logit
        if logit.dim() > target.dim():
            logit = logit.squeeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
        weights = torch.where(
            target > 0.5,
            torch.full_like(target, self.positive_weight),
            torch.ones_like(target),
        )
        if self.use_focal and self.gamma > 0:
            prob = torch.sigmoid(logit)
            pt = torch.where(target > 0.5, prob, 1.0 - prob).clamp_min(1e-6)
            bce = bce * (1.0 - pt).pow(self.gamma)
        warning_component = (bce * weights).mean()
        zero = torch.zeros_like(warning_component)
        return LossBreakdown(
            total=warning_component,
            forecast=warning_component,
            line=zero,
            band=zero,
            cross=zero,
            direction=zero,
            risk=warning_component,
        )


class LineDistributionalLoss(nn.Module):
    """line v3 분포형 예측선 전용 손실.

    pinball loss를 기본으로 쓰고, quantile crossing penalty와 선택적 beta 보조 line loss를 더한다.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float],
        *,
        lambda_cross: float = 1.0,
        beta_aux: float | None = None,
        lambda_beta_aux: float = 0.1,
        display_quantile_index: int = 5,
    ) -> None:
        super().__init__()
        self.pinball_loss = PinballLoss(tuple(float(value) for value in quantiles), sort_quantiles=False)
        self.cross_loss = QuantileCrossingPenaltyLoss()
        self.lambda_cross = float(lambda_cross)
        self.beta_aux = beta_aux
        self.lambda_beta_aux = float(lambda_beta_aux)
        self.display_quantile_index = int(display_quantile_index)
        self.beta_loss = AsymmetricHuberLoss(alpha=1.0, beta=float(beta_aux), delta=1.0) if beta_aux is not None else None

    def forward(
        self,
        prediction: LineDistributionalOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del band_target
        if not isinstance(prediction, LineDistributionalOutput):
            raise TypeError("LineDistributionalLoss는 LineDistributionalOutput만 받을 수 있습니다.")
        target = raw_future_returns if raw_future_returns is not None else line_target
        quantile_component = self.pinball_loss(prediction.quantiles, target)
        cross_component = self.cross_loss(prediction.quantiles)
        beta_component = torch.zeros_like(quantile_component)
        if self.beta_loss is not None:
            display_line = prediction.quantiles[..., self.display_quantile_index]
            beta_component = self.beta_loss(display_line, target)
        total = quantile_component + (self.lambda_cross * cross_component) + (self.lambda_beta_aux * beta_component)
        zero = torch.zeros_like(total)
        return LossBreakdown(
            total=total,
            forecast=total,
            line=beta_component,
            band=zero,
            cross=cross_component,
            direction=zero,
            quantile=quantile_component,
        )


class LineMonotonicDistributionalLoss(nn.Module):
    """line v3 monotonic 분포형 예측선 전용 손실.

    quantile 순서는 모델 출력 구조가 보장하므로 crossing penalty를 쓰지 않는다.
    beta_aux가 있으면 중심선(line)에 기존 보수 line loss를 함께 적용한다.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float],
        *,
        beta_aux: float | None = None,
        lambda_quantile: float = 1.0,
        lambda_beta_aux: float = 0.0,
        display_quantile_index: int = 5,
        detach_location_for_quantile: bool = False,
    ) -> None:
        super().__init__()
        self.pinball_loss = PinballLoss(tuple(float(value) for value in quantiles), sort_quantiles=False)
        self.beta_aux = beta_aux
        self.lambda_quantile = float(lambda_quantile)
        self.lambda_beta_aux = float(lambda_beta_aux)
        self.display_quantile_index = int(display_quantile_index)
        self.detach_location_for_quantile = bool(detach_location_for_quantile)
        self.beta_loss = AsymmetricHuberLoss(alpha=1.0, beta=float(beta_aux), delta=1.0) if beta_aux is not None else None

    def forward(
        self,
        prediction: LineMonotonicDistributionalOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del band_target
        if not isinstance(prediction, LineMonotonicDistributionalOutput):
            raise TypeError("LineMonotonicDistributionalLoss는 LineMonotonicDistributionalOutput만 받을 수 있습니다.")
        target = raw_future_returns if raw_future_returns is not None else line_target
        quantiles = prediction.quantiles
        if self.detach_location_for_quantile:
            location = prediction.line.unsqueeze(-1)
            quantiles = location.detach() + (prediction.quantiles - location)
        quantile_component = self.pinball_loss(quantiles, target)
        beta_component = torch.zeros_like(quantile_component)
        if self.beta_loss is not None:
            beta_component = self.beta_loss(prediction.line, target)
        total = (self.lambda_quantile * quantile_component) + (self.lambda_beta_aux * beta_component)
        zero = torch.zeros_like(total)
        return LossBreakdown(
            total=total,
            forecast=total,
            line=beta_component,
            band=zero,
            cross=zero,
            direction=zero,
            quantile=quantile_component,
        )


class BandOnlyLoss(nn.Module):
    """band 전용 손실. line 예측선 손실은 계산하지 않는다."""

    def __init__(
        self,
        *,
        q_low: float = 0.1,
        q_high: float = 0.9,
        lambda_band: float = 1.0,
        lambda_cross: float = 1.0,
        band_mode: str = "direct",
        lower_band_loss_weight: float = 1.0,
        upper_band_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.low_quantile_loss = PinballLoss((q_low,))
        self.high_quantile_loss = PinballLoss((q_high,))
        self.cross_loss = BandCrossPenaltyLoss()
        self.lambda_band = lambda_band
        self.lambda_cross = lambda_cross
        self.band_mode = band_mode
        self.lower_band_loss_weight = lower_band_loss_weight
        self.upper_band_loss_weight = upper_band_loss_weight

    def forward(
        self,
        prediction: BandOutput,
        line_target: torch.Tensor,
        band_target: torch.Tensor,
        raw_future_returns: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del raw_future_returns
        if not isinstance(prediction, BandOutput):
            raise TypeError("BandOnlyLoss는 BandOutput만 받을 수 있습니다.")
        lower_band_component = self.low_quantile_loss(prediction.lower_band, band_target)
        upper_band_component = self.high_quantile_loss(prediction.upper_band, band_target)
        band_component = (
            (self.lower_band_loss_weight * lower_band_component)
            + (self.upper_band_loss_weight * upper_band_component)
        )
        cross_component = (
            self.cross_loss(prediction.lower_band, prediction.upper_band)
            if self.band_mode == "direct"
            else torch.zeros_like(band_component)
        )
        zero = torch.zeros_like(band_component)
        forecast_component = (self.lambda_band * band_component) + (self.lambda_cross * cross_component)
        return LossBreakdown(
            total=forecast_component,
            forecast=forecast_component,
            line=zero,
            band=band_component,
            cross=cross_component,
            direction=zero,
            risk=zero,
        )
