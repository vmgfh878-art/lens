"""
전략 백테스트 엔진 + 지표 산출 — 순수 계산.

CP226 분리 산출. 수익률·드로우다운·샤프·소르티노·시뮬레이션 지표.
의존: strategy_indicators (_jsonable), strategy_rules (StrategyRule) 만.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from app.services.strategy_indicators import _jsonable
from app.strategies.strategy_rules import StrategyRule

FEE_RATE = 0.001


def _total_return(returns: np.ndarray) -> float:
    usable = np.nan_to_num(returns, nan=0.0)
    return float(np.prod(1.0 + usable) - 1.0) if usable.size else 0.0


def _max_drawdown(returns: np.ndarray) -> float:
    usable = np.nan_to_num(returns, nan=0.0)
    if usable.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + usable)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _sharpe(returns: np.ndarray) -> float:
    usable = returns[np.isfinite(returns)]
    if usable.size < 2:
        return 0.0
    std = float(np.std(usable, ddof=1))
    return float(np.mean(usable) / std * math.sqrt(252.0)) if std > 0 else 0.0


def _sortino(returns: np.ndarray) -> float:
    usable = returns[np.isfinite(returns)]
    downside = usable[usable < 0]
    if downside.size < 2:
        return 0.0
    std = float(np.std(downside, ddof=1))
    return float(np.mean(usable) / std * math.sqrt(252.0)) if std > 0 else 0.0


def _large_loss_threshold(returns: np.ndarray) -> float | None:
    usable = returns[np.isfinite(returns)]
    if usable.size == 0:
        return None
    return min(-0.02, float(np.nanpercentile(usable, 20)))


def _signal_row(row: pd.Series, rule: StrategyRule) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "position": int(row["position"]),
        "targetPosition": int(row["target_position"]),
        "conservativeReturn": _jsonable(row.get("line_score")) if rule.uses_line else None,
        "lowerBandReturn": _jsonable(row.get("band_lower_return")) if rule.uses_band else None,
        "bandWidthReturn": _jsonable(row.get("band_width_return")) if rule.uses_band else None,
        "bandWidthExpansion": _jsonable(row.get("band_width_expansion"))
        if rule.uses_band
        else None,
        "bandWidthPercentile": _jsonable(row.get("band_width_percentile"))
        if rule.uses_band
        else None,
        "ma60Ratio": _jsonable(row.get("ma_60_ratio")),
        "ma20Ratio": _jsonable(row.get("ma_20_ratio")),
        "macdRatio": _jsonable(row.get("macd_ratio")),
        "rsi": _jsonable(row.get("rsi_norm")),
        "atrRatio": _jsonable(row.get("atr_ratio_calc")),
        "reason": str(row.get("reason") or ""),
    }


def _ticker_metrics(signal_frame: pd.DataFrame) -> dict[str, Any]:
    frame = signal_frame.sort_values("date").copy()
    returns = frame["daily_return"].to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=float)
    shifted = np.concatenate([[0.0], positions[:-1]])
    trades = np.abs(np.diff(shifted, prepend=0.0))
    strategy_returns = shifted * returns - trades * FEE_RATE
    buy_hold_returns = returns
    large_loss_threshold = _large_loss_threshold(buy_hold_returns)
    if large_loss_threshold is not None:
        large_loss_mask = buy_hold_returns <= large_loss_threshold
        large_loss_days = int(large_loss_mask.sum())
        avoided_large_loss_days = int(((shifted == 0.0) & large_loss_mask).sum())
    else:
        large_loss_days = 0
        avoided_large_loss_days = 0
    return {
        "strategyReturnPct": _total_return(strategy_returns) * 100.0,
        "buyHoldReturnPct": _total_return(buy_hold_returns) * 100.0,
        "buyHoldReturnRatio": (
            _total_return(strategy_returns) / _total_return(buy_hold_returns)
            if abs(_total_return(buy_hold_returns)) > 1e-12
            else None
        ),
        "excessReturnPct": (_total_return(strategy_returns) - _total_return(buy_hold_returns))
        * 100.0,
        "maxDrawdownPct": _max_drawdown(strategy_returns) * 100.0,
        "buyHoldMaxDrawdownPct": _max_drawdown(buy_hold_returns) * 100.0,
        "maxDrawdownImprovementPct": (
            _max_drawdown(strategy_returns) - _max_drawdown(buy_hold_returns)
        )
        * 100.0,
        "feeAdjustedReturnPct": _total_return(strategy_returns) * 100.0,
        "feeAdjustedSharpe": _sharpe(strategy_returns),
        "buyHoldSharpe": _sharpe(buy_hold_returns),
        "strategySortino": _sortino(strategy_returns),
        "buyHoldSortino": _sortino(buy_hold_returns),
        "tradeCount": int(trades.sum()),
        "cashWaitRatio": float(np.mean(shifted == 0.0)),
        "marketParticipationRate": float(np.mean(shifted > 0.0)),
        "worstTradeLossPct": None,
        "averageHoldingDays": _average_holding_days(positions),
        "avoidedLargeLossDays": avoided_large_loss_days,
        "largeLossDays": large_loss_days,
        "largeLossAvoidanceRate": avoided_large_loss_days / large_loss_days
        if large_loss_days
        else None,
        "largeLossThresholdPct": large_loss_threshold * 100.0
        if large_loss_threshold is not None
        else None,
    }


def _average_holding_days(positions: np.ndarray) -> float | None:
    durations: list[int] = []
    current = 0
    start: int | None = None
    for index, position in enumerate(positions.astype(int)):
        if current == 0 and position == 1:
            start = index
        if current == 1 and position == 0 and start is not None:
            durations.append(index - start)
            start = None
        current = position
    if current == 1 and start is not None:
        durations.append(len(positions) - start)
    return float(np.mean(durations)) if durations else None


def _trade_events(signal_frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = signal_frame.sort_values("date").copy()
    positions = frame["position"].to_numpy(dtype=int)
    previous = np.concatenate([[0], positions[:-1]])
    changed = np.where(positions != previous)[0]
    events = []
    for index in changed:
        row = frame.iloc[index]
        events.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "kind": "entry" if int(row["position"]) == 1 else "exit",
                "price": _jsonable(row.get("close")),
                "reason": str(row.get("reason") or ""),
            }
        )
    return events


def _points(signal_frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = signal_frame.sort_values("date").copy()
    returns = frame["daily_return"].to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=float)
    shifted = np.concatenate([[0.0], positions[:-1]])
    trades = np.abs(np.diff(shifted, prepend=0.0))
    strategy_returns = shifted * returns - trades * FEE_RATE
    strategy_equity = np.cumprod(1.0 + np.nan_to_num(strategy_returns, nan=0.0))
    buy_hold_equity = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
    result = []
    for index, row in frame.iterrows():
        offset = len(result)
        result.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "price": _jsonable(row.get("close")),
                "strategyEquity": _jsonable(strategy_equity[offset]),
                "buyHoldEquity": _jsonable(buy_hold_equity[offset]),
                "position": int(row["position"]),
            }
        )
    return result


__all__ = [
    "FEE_RATE",
    "_total_return",
    "_max_drawdown",
    "_sharpe",
    "_sortino",
    "_large_loss_threshold",
    "_signal_row",
    "_ticker_metrics",
    "_average_holding_days",
    "_trade_events",
    "_points",
]
