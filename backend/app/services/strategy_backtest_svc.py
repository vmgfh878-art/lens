from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.services import parquet_store
from app.services.strategy_indicators import (
    _align_date_dtype,
    _compute_signal_frame,
    _jsonable,
    _normalize_rsi,
    _raw_target,  # noqa: F401 (re-export)
    _reason,  # noqa: F401 (re-export)
    _safe,  # noqa: F401 (re-export)
)

# 전략 정의 (StrategyRule + STRATEGIES) 는 strategy_rules.py 에서 단일 관리한다 (모델 급 관리).
# 이 파일은 전략의 "실행" (pandas 조건 계산 + 백테스트) 만 담당한다.
from app.strategies.strategy_rules import STRATEGIES, StrategyRule
from fastapi import HTTPException

FEE_RATE = 0.001
MIN_EVAL_DAYS = 120


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "v1"


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


@lru_cache(maxsize=1)
def _load_frame() -> pd.DataFrame:
    base = _data_dir()
    price = pd.read_parquet(base / "market_prices_1d.parquet")
    close_column = "adjusted_close" if "adjusted_close" in price.columns else "close"
    price = price[["ticker", "date", "open", "high", "low", close_column, "volume"]].rename(
        columns={close_column: "close"}
    )
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price = price.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])
    price = price.drop_duplicates(["ticker", "date"], keep="last")
    price["daily_return"] = price.groupby("ticker")["close"].pct_change().fillna(0.0)
    price["previous_close"] = price.groupby("ticker")["close"].shift(1)
    true_range = pd.concat(
        [
            (price["high"] - price["low"]).abs(),
            (price["high"] - price["previous_close"]).abs(),
            (price["low"] - price["previous_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price["atr_ratio_calc"] = (
        true_range.groupby(price["ticker"]).transform(
            lambda values: values.rolling(14, min_periods=5).mean()
        )
        / price["close"]
    )

    indicators = pd.read_parquet(base / "market_indicators_1d.parquet")
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"])
    indicators = indicators.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    for column in [
        "ma_5_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "macd_ratio",
        "bb_position",
        "vol_change",
    ]:
        if column in indicators.columns:
            indicators[column] = pd.to_numeric(indicators[column], errors="coerce")
    indicators["rsi_norm"] = (
        _normalize_rsi(indicators["rsi"]) if "rsi" in indicators.columns else np.nan
    )

    # Use shared parquet_store to avoid loading a second copy of these files
    # (predictions.py already holds them; store ensures only one in-process copy).
    _raw_line = parquet_store.get_raw("line_1d")
    if _raw_line is None:
        raise FileNotFoundError("predictions_line_1d.parquet not found in parquet_store")
    line = _raw_line.copy()
    line["ticker"] = line["ticker"].astype(str).str.upper()
    line["date"] = pd.to_datetime(line["asof_date"])
    for column in ["line_score", "safe_line_score", "line_rank_by_date", "safe_line_rank_by_date"]:
        if column in line.columns:
            line[column] = pd.to_numeric(line[column], errors="coerce")
    line = line.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    _raw_band = parquet_store.get_raw("band_1d")
    if _raw_band is None:
        raise FileNotFoundError("predictions_band_1d.parquet not found in parquet_store")
    band = _raw_band[pd.to_numeric(_raw_band["horizon_step"], errors="coerce") == 5].copy()
    band["ticker"] = band["ticker"].astype(str).str.upper()
    band["date"] = pd.to_datetime(band["asof_date"])
    for column in ["band_lower", "band_upper"]:
        band[column] = pd.to_numeric(band[column], errors="coerce")
    band = band.groupby(["ticker", "date"], as_index=False, observed=True).agg(
        band_lower=("band_lower", "min"), band_upper=("band_upper", "max")
    )

    # CP214 — 머지 직전 date dtype 평탄화 (방어). 근본 fix 는 parquet_store 에서 했지만
    # 향후 다른 source 가 추가돼도 머지가 깨지지 않게 idempotent helper 적용.
    price = _align_date_dtype(price)
    indicators = _align_date_dtype(indicators)
    line = _align_date_dtype(line)
    band = _align_date_dtype(band)

    frame = price.merge(
        indicators[
            [
                "ticker",
                "date",
                "ma_5_ratio",
                "ma_20_ratio",
                "ma_60_ratio",
                "macd_ratio",
                "bb_position",
                "vol_change",
                "rsi_norm",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )
    frame = frame.merge(
        line[
            [
                "ticker",
                "date",
                "line_score",
                "safe_line_score",
                "line_rank_by_date",
                "safe_line_rank_by_date",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )
    frame = frame.merge(band, on=["ticker", "date"], how="left")

    frame["ma_20_ratio"] = frame["ma_20_ratio"].fillna(
        frame.groupby("ticker")["close"].transform(
            lambda values: values / values.rolling(20, min_periods=15).mean() - 1.0
        )
    )
    frame["ma_60_ratio"] = frame["ma_60_ratio"].fillna(
        frame.groupby("ticker")["close"].transform(
            lambda values: values / values.rolling(60, min_periods=40).mean() - 1.0
        )
    )
    frame["band_lower_return"] = frame["band_lower"] / frame["close"] - 1.0
    frame["band_upper_return"] = frame["band_upper"] / frame["close"] - 1.0
    frame["band_width_return"] = frame["band_upper_return"] - frame["band_lower_return"]
    width_reference = frame.groupby("ticker")["band_width_return"].transform(
        lambda values: values.rolling(60, min_periods=20).median().shift(1)
    )
    frame["band_width_expansion"] = (frame["band_width_return"] / width_reference).replace(
        [np.inf, -np.inf], np.nan
    )
    frame["band_width_expansion"] = frame["band_width_expansion"].fillna(1.0)
    frame["band_width_percentile"] = frame.groupby("ticker")["band_width_return"].rank(pct=True)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _sector_map() -> dict[str, str]:
    try:
        stock_info = pd.read_parquet(_data_dir() / "market_stock_info.parquet")
    except Exception:
        return {}
    stock_info["ticker"] = stock_info["ticker"].astype(str).str.upper()
    return {
        str(row["ticker"]).upper(): str(row["sector"]) if pd.notna(row.get("sector")) else "Unknown"
        for _, row in stock_info.iterrows()
    }


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


@lru_cache(maxsize=16)
def _strategy_results(strategy_id: str) -> dict[str, Any]:
    if strategy_id not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 전략입니다: {strategy_id}")
    rule = STRATEGIES[strategy_id]
    frame = _load_frame()
    if rule.uses_line:
        frame = frame[frame["line_score"].notna()].copy()
    if rule.uses_band:
        frame = frame[
            frame["band_lower_return"].notna() & frame["band_width_return"].notna()
        ].copy()
    if frame.empty:
        raise HTTPException(
            status_code=404, detail=f"{rule.label}에 사용할 로컬 데이터가 없습니다."
        )

    end_date = frame["date"].max()
    start_date = end_date - pd.Timedelta(days=365)
    frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()

    by_ticker: dict[str, pd.DataFrame] = {}
    metrics_rows = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        if len(ticker_frame) < MIN_EVAL_DAYS:
            continue
        signal_frame = _compute_signal_frame(ticker_frame, rule)
        by_ticker[str(ticker)] = signal_frame
        metrics = _ticker_metrics(signal_frame)
        metrics_rows.append({"ticker": str(ticker), **metrics})

    if not metrics_rows:
        raise HTTPException(
            status_code=404, detail=f"{rule.label}에 필요한 평가 가능 티커가 없습니다."
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    pass_mask = (
        (metrics_frame["strategyReturnPct"] >= metrics_frame["buyHoldReturnPct"])
        & (metrics_frame["maxDrawdownPct"] >= metrics_frame["buyHoldMaxDrawdownPct"])
        & (metrics_frame["marketParticipationRate"].between(0.2, 0.95))
    )
    aggregate = {
        "strategyReturnPct": float(metrics_frame["strategyReturnPct"].mean()),
        "buyHoldReturnPct": float(metrics_frame["buyHoldReturnPct"].mean()),
        "excessReturnPct": float(metrics_frame["excessReturnPct"].mean()),
        "maxDrawdownPct": float(metrics_frame["maxDrawdownPct"].mean()),
        "buyHoldMaxDrawdownPct": float(metrics_frame["buyHoldMaxDrawdownPct"].mean()),
        "maxDrawdownImprovementPct": float(metrics_frame["maxDrawdownImprovementPct"].mean()),
        "feeAdjustedSharpe": float(metrics_frame["feeAdjustedSharpe"].mean()),
        "buyHoldSharpe": float(metrics_frame["buyHoldSharpe"].mean()),
        "strategySortino": float(metrics_frame["strategySortino"].mean()),
        "buyHoldSortino": float(metrics_frame["buyHoldSortino"].mean()),
        "marketParticipationRate": float(metrics_frame["marketParticipationRate"].mean()),
        "avgSelectedCount": None,
        "avgTradeCount": float(metrics_frame["tradeCount"].mean()),
        "largeLossAvoidanceRate": float(metrics_frame["largeLossAvoidanceRate"].dropna().mean()),
        "passTickerRatio": float(pass_mask.mean()),
    }
    return {
        "rule": rule,
        "start_date": start_date,
        "end_date": end_date,
        "by_ticker": by_ticker,
        "metrics": metrics_frame,
        "aggregate": aggregate,
    }


def get_strategy_scan(strategy_id: str, limit: int = 500) -> dict[str, Any]:
    result = _strategy_results(strategy_id)
    rule: StrategyRule = result["rule"]
    sectors = _sector_map()
    cards = []
    for ticker, signal_frame in result["by_ticker"].items():
        row = signal_frame.iloc[-1]
        has_usable_signal = bool(pd.notna(row.get("close")))
        cards.append(
            {
                "ticker": ticker,
                "sector": sectors.get(ticker),
                "group": str(row["signal_group"]),
                "signalLabel": str(row["signal_label"]),
                "reason": str(row["reason"]),
                "asofDate": pd.Timestamp(row["date"]).date().isoformat(),
                "conservativeReturn": _jsonable(row.get("line_score")) if rule.uses_line else None,
                "lowerBandReturn": _jsonable(row.get("band_lower_return"))
                if rule.uses_band
                else None,
                "bandWidthReturn": _jsonable(row.get("band_width_return"))
                if rule.uses_band
                else None,
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
                "strategyScore": _jsonable(row.get("line_score"))
                if rule.uses_line
                else _jsonable(row.get("ma_60_ratio")),
                "hasUsableSignal": has_usable_signal,
            }
        )
    group_order = {"buy": 0, "hold": 1, "risk": 2, "watch": 3}
    cards.sort(
        key=lambda row: (
            group_order.get(row["group"], 9),
            -(row["strategyScore"] or -9999.0),
            row["ticker"],
        )
    )
    return {
        "strategyId": rule.id,
        "strategyLabel": rule.label,
        "timeframe": "1D",
        "asofDate": pd.Timestamp(result["end_date"]).date().isoformat(),
        "scopeTickerCount": len(result["by_ticker"]),
        "usableSignalCount": sum(1 for card in cards if card["hasUsableSignal"]),
        "latestValidTickerCount": sum(1 for card in cards if card["hasUsableSignal"]),
        "cards": cards[:limit],
        "portfolioMetrics": result["aggregate"],
        "aggregateMetrics": result["aggregate"],
        "contract": "single_ticker_long_cash_average",
    }


def clear_strategy_cache() -> None:
    """admin/reload 에서 호출. parquet 갱신 후 전략 frame/결과/sector cache 를 비운다.
    이게 안 되면 새 데이터를 받아도 전략은 옛 결과를 계속 노출한다."""
    _load_frame.cache_clear()
    _strategy_results.cache_clear()
    _sector_map.cache_clear()


def get_strategy_backtest(strategy_id: str, ticker: str) -> dict[str, Any]:
    result = _strategy_results(strategy_id)
    rule: StrategyRule = result["rule"]
    normalized = ticker.upper()
    if normalized not in result["by_ticker"]:
        raise HTTPException(
            status_code=404,
            detail=f"{normalized}에는 {rule.label} 백테스트에 필요한 로컬 데이터가 없습니다.",
        )

    signal_frame = result["by_ticker"][normalized]
    metrics = _ticker_metrics(signal_frame)
    return {
        "points": _points(signal_frame),
        "signals": [_signal_row(row, rule) for _, row in signal_frame.iterrows()],
        "tradeEvents": _trade_events(signal_frame),
        **metrics,
    }
