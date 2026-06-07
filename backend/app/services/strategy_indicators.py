"""
전략 규칙 / 지표 / 상태머신 — 순수 계산.

CP226 분리 산출. parquet I/O / 캐시 / 백테스트 엔진은 다른 모듈로 분리.
의존: strategy_rules (StrategyRule dataclass) 만.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from app.strategies.strategy_rules import StrategyRule
from fastapi import HTTPException


def _align_date_dtype(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """CP214 — merge 직전 호출. `col` 을 flat `datetime64[ns]` 로 강제.

    근본 원인은 `parquet_store._compress_strings` 에서 fix (날짜 컬럼은 categorical 제외)
    했지만, 이 헬퍼는 idempotent 방어선이다.
    - `pd.to_datetime(Categorical[str])` 가 결과를 `Categorical[datetime]` 로 유지하는
      pandas 동작 때문에 datetime64 와의 merge 가 실패하는 케이스가 다시 들어와도
      이 함수를 머지 직전에 호출하면 평탄화된다.
    - object / Categorical[str] / datetime64 / Categorical[datetime] 모두 안전.
    """
    if col not in df.columns:
        return df
    series = df[col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="coerce")
    return df.assign(**{col: series.astype("datetime64[ns]")})


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _normalize_rsi(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric > 1.0, numeric * 100.0)


def _safe(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_col(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """전략이 안 쓰는 예측 컬럼(line/band)이 frame 에 없을 때 default Series 반환.

    band 를 안 쓰는 전략(indicator_balance_v2)은 _load_frame 이 band 를 머지하지
    않으므로 band_* 컬럼이 frame 에 없다. 이때 KeyError 대신 NaN(또는 default)으로
    안전 처리한다. 해당 전략은 이 컬럼을 entry/exit 로직에서 쓰지 않으므로 결과 불변.
    """
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return _safe(frame[col], default)


def _raw_target(frame: pd.DataFrame, rule: StrategyRule) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma60 = _safe(frame["ma_60_ratio"])
    ma20 = _safe(frame["ma_20_ratio"])
    macd = _safe(frame["macd_ratio"], 0.0)
    rsi = _safe(frame["rsi_norm"], 50.0)
    atr = _safe(frame["atr_ratio_calc"], 0.0)
    bb = _safe(frame["bb_position"], 1.0)
    line = _safe_col(frame, "line_score")
    lower = _safe_col(frame, "band_lower_return")
    width_expansion = _safe_col(frame, "band_width_expansion", 1.0)

    if rule.id == "indicator_balance_v2":
        trend_entry = (ma60 >= 0.02) & (ma20 >= -0.02) & (macd >= 0.0) & (rsi < 75.0)
        pullback_entry = (ma60 >= 0.02) & (bb <= 0.35) & (rsi < 55.0)
        entry = trend_entry | pullback_entry
        exit_signal = (ma60 <= -0.05) | (ma20 <= -0.05) | ((atr >= 0.07) & (ma20 < 0.0))
        risk_signal = exit_signal
        return entry, exit_signal, risk_signal

    if rule.id == "ai_balance_v2":
        entry = (
            (line >= -0.02)
            & (ma60 >= 0.0)
            & (ma20 >= -0.04)
            & ((lower >= -0.06) | (width_expansion < 1.25))
        )
        line_weak = line < -0.06
        band_risk = (lower < -0.06) | (width_expansion > 1.25)
        price_break = ma20 < -0.10
        volatility_break = (atr > 0.12) & (ma20 < 0.0)
        exit_signal = (line_weak & band_risk) | price_break | volatility_break
        risk_signal = band_risk | price_break | volatility_break
        return entry, exit_signal, risk_signal

    if rule.id == "ai_band_defense_v1":
        indicator_trend = (ma60 >= 0.02) & (ma20 >= -0.03) & (rsi < 82.0)
        indicator_pullback = (ma60 >= 0.02) & (bb <= 0.45) & (rsi < 60.0)
        band_clear = (lower >= -0.08) | (width_expansion < 1.60)
        entry = (indicator_trend | indicator_pullback) & band_clear
        band_stress = (lower < -0.08) & (width_expansion > 1.60)
        trend_break = (ma60 < -0.05) | (ma20 < -0.08)
        volatility_break = (atr > 0.12) & (ma20 < 0.0)
        exit_signal = band_stress | trend_break | volatility_break
        risk_signal = band_stress | trend_break | volatility_break
        return entry, exit_signal, risk_signal

    raise HTTPException(status_code=404, detail=f"지원하지 않는 전략입니다: {rule.id}")


def _reason(rule: StrategyRule, position: int, target: int, risk: bool) -> tuple[str, str, str]:
    if target == 1 and position == 0:
        return "buy", "매수 후보", f"{rule.label} 기준 진입 조건을 확인 중입니다."
    if position == 1 and not risk:
        return "hold", "보유 유지", f"{rule.label} 기준 보유 조건을 유지합니다."
    if position == 1 and risk:
        return "risk", "위험 확대", f"{rule.label} 기준 위험 조건이 감지되어 청산 확인 중입니다."
    if risk:
        return (
            "risk",
            "위험 확대",
            f"{rule.label} 기준 위험 조건이 강해서 신규 진입에 부적합합니다.",
        )
    return "watch", "관망", f"{rule.label} 기준 신규 진입 조건이 아직 충분하지 않습니다."


def _compute_signal_frame(ticker_frame: pd.DataFrame, rule: StrategyRule) -> pd.DataFrame:
    frame = ticker_frame.sort_values("date").copy()
    entry, exit_signal, risk_signal = _raw_target(frame, rule)
    current = 0
    entry_streak = 0
    exit_streak = 0
    positions: list[int] = []
    targets: list[int] = []
    groups: list[str] = []
    labels: list[str] = []
    reasons: list[str] = []

    for wants_entry, wants_exit, risky in zip(
        entry.to_numpy(bool), exit_signal.to_numpy(bool), risk_signal.to_numpy(bool), strict=False
    ):
        if current == 0:
            target = 1 if wants_entry else 0
            entry_streak = entry_streak + 1 if wants_entry else 0
            if entry_streak >= rule.entry_confirm_days:
                current = 1
                exit_streak = 0
        else:
            target = 0 if wants_exit else 1
            exit_streak = exit_streak + 1 if wants_exit else 0
            if exit_streak >= rule.exit_confirm_days:
                current = 0
                entry_streak = 0
        group, label, reason = _reason(rule, current, target, bool(risky))
        positions.append(current)
        targets.append(target)
        groups.append(group)
        labels.append(label)
        reasons.append(reason)

    frame["position"] = positions
    frame["target_position"] = targets
    frame["signal_group"] = groups
    frame["signal_label"] = labels
    frame["reason"] = reasons
    return frame


__all__ = [
    "_align_date_dtype",
    "_jsonable",
    "_normalize_rsi",
    "_safe",
    "_raw_target",
    "_reason",
    "_compute_signal_frame",
]
