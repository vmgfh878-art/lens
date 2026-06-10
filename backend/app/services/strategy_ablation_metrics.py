"""CP253 — ablation 통계 헬퍼 (부트스트랩 CI · Wilcoxon).

CP252(v2) 발굴 파이프라인(`strategy_ablation_v2_select`)이 쓰는 순수 통계 유틸만 남긴다.
CP248~251(v1) ablation 코어(`strategy_ablation.py` + 본 모듈의 AblationConfig 의존 함수들)는
CP252/253 이 대체하여 제거됨. 본 모듈은 v1 의존이 없는 두 함수만 보존.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

BOOTSTRAP_B = 2000
ALPHA = 0.05


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, b: int = BOOTSTRAP_B, alpha: float = ALPHA
) -> tuple[float, float]:
    """티커 델타 분포의 평균에 대한 부트스트랩 (1-alpha) CI. seed 는 호출측 rng 로 고정."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(b, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def _wilcoxon_p(values: np.ndarray) -> float:
    """0 대칭성 Wilcoxon 부호순위 p-value (순위 기반, 이상치 robust). 표본 부족 시 NaN."""
    nz = values[np.isfinite(values)]
    nz = nz[nz != 0.0]
    if nz.size < 6:
        return float("nan")
    try:
        return float(wilcoxon(nz, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


__all__ = ["_bootstrap_ci", "_wilcoxon_p"]
