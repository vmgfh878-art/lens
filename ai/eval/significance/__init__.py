"""CP216 — v1 운영 3모델 통계 유의성 검정 패키지.

DM(HAC) + Bootstrap CI(cluster + block × 2 길이) + GW conditional test + Bonferroni.
운영 parquet read-only. 학습/calibration 없음.
"""

from .config import (
    OPS_LINE_1D,
    OPS_BAND_1D,
    OPS_BAND_1W,
    CP175_FROZEN,
    PRICES_1D,
    PRICES_1W,
    INDICATORS_1D,
    BASELINE_DIR,
    SIG_DIR,
    RUNS_DIR,
    Q_LOW,
    Q_HIGH,
    HISTORICAL_WINDOW,
    BOLLINGER_WINDOW,
    BOLLINGER_SIGMA,
    BLOCK_SIZES,
    VIX_THRESHOLD,
    DD_WINDOW,
    DD_THRESHOLD,
)

__all__ = [
    "OPS_LINE_1D",
    "OPS_BAND_1D",
    "OPS_BAND_1W",
    "CP175_FROZEN",
    "PRICES_1D",
    "PRICES_1W",
    "INDICATORS_1D",
    "BASELINE_DIR",
    "SIG_DIR",
    "RUNS_DIR",
    "Q_LOW",
    "Q_HIGH",
    "HISTORICAL_WINDOW",
    "BOLLINGER_WINDOW",
    "BOLLINGER_SIGMA",
    "BLOCK_SIZES",
    "VIX_THRESHOLD",
    "DD_WINDOW",
    "DD_THRESHOLD",
]
