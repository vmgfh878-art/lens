from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """채널별 인스턴스 정규화와 역정규화를 제공한다."""

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, n_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"지원하지 않는 RevIN 모드입니다: {mode}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        self._std = torch.sqrt(variance + self.eps).detach()
        normalized = (x - self._mean) / self._std
        if self.affine:
            normalized = normalized * self.gamma + self.beta
        return normalized

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("denorm 호출 전에 norm 모드가 먼저 실행되어야 합니다.")
        denormalized = x
        if self.affine:
            denormalized = (denormalized - self.beta) / (self.gamma + self.eps)
        return denormalized * self._std + self._mean

    def denormalize_target(self, y: torch.Tensor, target_channel_idx: int) -> torch.Tensor:
        """지정한 타깃 채널의 통계량으로 예측값을 역정규화한다."""
        if self._mean is None or self._std is None:
            raise RuntimeError("denormalize_target 호출 전에 norm 모드가 먼저 실행되어야 합니다.")
        mean_target = self._mean[..., target_channel_idx]
        std_target = self._std[..., target_channel_idx]
        output = y
        if self.affine:
            beta_target = self.beta[..., target_channel_idx]
            gamma_target = self.gamma[..., target_channel_idx]
            output = (output - beta_target) / (gamma_target + self.eps)
        return output * std_target + mean_target
