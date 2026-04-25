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
            raise RuntimeError("denorm 호출 전에 norm이 먼저 실행되어야 합니다.")
        denormalized = x
        if self.affine:
            denormalized = (denormalized - self.beta) / (self.gamma + self.eps)
        return denormalized * self._std + self._mean
