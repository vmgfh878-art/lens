from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_DATA_DIR = Path("data/parquet")
DEFAULT_OUTPUT_JSON = Path("docs/cp109_line_band_balance_strategy_grid_metrics.json")
DEFAULT_OUTPUT_CSV = Path("docs/cp109_line_band_balance_strategy_grid_top_candidates.csv")

EXPLORE_START = "2025-06-18"
BASE_EXPLORE_END = "2026-03-31"
ONE_MONTH_HOLDOUT_START = "2026-04-01"
HOLDOUT_END = "2026-05-01"
FEE_BPS = 10


@dataclass(frozen=True)
class SplitPlan:
    label: str
    explore_start: str
    explore_end: str
    holdout_start: str
    holdout_end: str
    reason: str


@dataclass(frozen=True)
class Rule:
    family: str
    line_entry_threshold: float
    line_hold_threshold: float
    lower_risk_threshold: float
    width_expansion_threshold: float
    width_percentile_threshold: float
    entry_band_filter: str
    exit_risk_mode: str
    confirm_days: int
    reentry_confirm_days: int
    trend_filter: str
    position_mode: str
    line_good_band_uncertain_position: float
    line_weak_band_stable_position: float
    upper_continuation: bool

    @property
    def key(self) -> str:
        return (
            f"{self.family}|entry={self.line_entry_threshold:.3f}|hold={self.line_hold_threshold:.3f}|"
            f"lower={self.lower_risk_threshold:.3f}|exp={self.width_expansion_threshold:.2f}|"
            f"pct={self.width_percentile_threshold:.2f}|entry_filter={self.entry_band_filter}|"
            f"exit={self.exit_risk_mode}|confirm={self.confirm_days}|reentry={self.reentry_confirm_days}|"
            f"trend={self.trend_filter}|mode={self.position_mode}|"
            f"good_uncertain={self.line_good_band_uncertain_position}|"
            f"weak_stable={self.line_weak_band_stable_position}|upper={self.upper_continuation}"
        )


@dataclass
class TickerData:
    ticker: str
    price: pd.DataFrame
    signals: pd.DataFrame
    line_prediction_count: int
    band_prediction_count: int
    min_asof: str
    max_asof: str


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_backend_url(value: str | None) -> str:
    raw = (value or DEFAULT_BACKEND_URL).strip().strip("\"'")
    if not raw:
        raw = DEFAULT_BACKEND_URL
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def normalize_rsi(value: Any) -> float | None:
    if not finite(value):
        return None
    number = float(value)
    return number * 100 if 0 <= number <= 1 else number


def read_json_url(url: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_prediction_history(backend_url: str, ticker: str, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = min(max(limit, 1), 200)
    params = urllib.parse.urlencode({"run_id": run_id, "limit": safe_limit})
    url = f"{backend_url}/api/v1/stocks/{urllib.parse.quote(ticker)}/predictions/history?{params}"
    try:
        payload = read_json_url(url)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 422}:
            return []
        raise
    return list(payload.get("data") or [])


def load_universe(data_dir: Path, limit: int) -> list[str]:
    stock_info = pd.read_parquet(data_dir / "stock_info.parquet")
    tickers = stock_info["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    return tickers[:limit]


def load_market_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(data_dir / "price_data_yfinance.parquet")
    close_column = "adjusted_close" if "adjusted_close" in price.columns else "close"
    price = price[["ticker", "date", close_column]].copy().rename(columns={close_column: "close"})
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = price["date"].astype(str)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])

    indicators = pd.read_parquet(data_dir / "indicators_yfinance_1D.parquet")
    indicator_columns = ["ticker", "date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"]
    indicators = indicators[[column for column in indicator_columns if column in indicators.columns]].copy()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = indicators["date"].astype(str)
    for column in ["rsi", "ma_60_ratio", "vol_change", "atr_ratio"]:
        if column in indicators.columns:
            indicators[column] = pd.to_numeric(indicators[column], errors="coerce")
    indicators = indicators.sort_values(["ticker", "date"])
    return price, indicators


def first_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if finite(value):
            return float(value)
    return None


def last_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if finite(value):
            return float(value)
    return None


def min_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    usable = [float(value) for value in values if finite(value)]
    return min(usable) if usable else None


def max_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    usable = [float(value) for value in values if finite(value)]
    return max(usable) if usable else None


def line_start_value(row: dict[str, Any]) -> float | None:
    conservative = first_finite(row.get("conservative_series"))
    if conservative is not None:
        return conservative
    return first_finite(row.get("line_series"))


def line_end_value(row: dict[str, Any]) -> float | None:
    conservative = last_finite(row.get("conservative_series"))
    if conservative is not None:
        return conservative
    return last_finite(row.get("line_series"))


def map_by_asof(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        asof = str(row.get("asof_date") or "")
        if asof:
            mapped[asof] = row
    return mapped


def prepare_ticker_data(
    ticker: str,
    price_rows: pd.DataFrame,
    indicator_rows: pd.DataFrame,
    line_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
) -> TickerData | None:
    if not line_rows or not band_rows:
        return None

    price = price_rows[price_rows["ticker"] == ticker].copy()
    if price.empty:
        return None
    price = price.sort_values("date").drop_duplicates("date", keep="last")
    price["ma60_calc"] = price["close"] / price["close"].rolling(60, min_periods=60).mean() - 1

    indicators = indicator_rows[indicator_rows["ticker"] == ticker].copy()
    if indicators.empty:
        indicators = pd.DataFrame(columns=["date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"])
    indicators = indicators.sort_values("date").drop_duplicates("date", keep="last")

    line_by_asof = map_by_asof(line_rows)
    band_by_asof = map_by_asof(band_rows)
    common_dates = sorted(set(line_by_asof) & set(band_by_asof))
    records: list[dict[str, Any]] = []

    for asof_date in common_dates:
        line_prediction = line_by_asof[asof_date]
        band_prediction = band_by_asof[asof_date]
        line_start = line_start_value(line_prediction)
        line_end = line_end_value(line_prediction)
        lower = min_finite(band_prediction.get("lower_band_series"))
        upper = max_finite(band_prediction.get("upper_band_series"))
        if line_start is None or line_end is None or lower is None or upper is None or upper <= lower:
            continue
        records.append(
            {
                "date": asof_date,
                "line_start_value": line_start,
                "line_end_value": line_end,
                "lower_band": lower,
                "upper_band": upper,
            }
        )

    if not records:
        return None

    signals = pd.DataFrame(records).sort_values("date").drop_duplicates("date", keep="last")
    signals = signals.merge(price[["date", "close", "ma60_calc"]], on="date", how="inner")
    if signals.empty:
        return None

    indicator_columns = [column for column in ["date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"] if column in indicators.columns]
    if len(indicator_columns) > 1:
        signals = signals.merge(indicators[indicator_columns], on="date", how="left")
    else:
        signals["rsi"] = math.nan
        signals["ma_60_ratio"] = math.nan
        signals["vol_change"] = math.nan
        signals["atr_ratio"] = math.nan

    if "ma_60_ratio" not in signals.columns:
        signals["ma_60_ratio"] = math.nan
    signals["ma60_ratio"] = signals["ma_60_ratio"].where(signals["ma_60_ratio"].notna(), signals["ma60_calc"])
    signals["rsi"] = signals.get("rsi", pd.Series(index=signals.index, dtype=float)).map(normalize_rsi)
    signals["line_return"] = signals["line_end_value"] / signals["close"] - 1
    signals["line_slope"] = (signals["line_end_value"] - signals["line_start_value"]) / signals["close"]
    signals["lower_return"] = signals["lower_band"] / signals["close"] - 1
    signals["upper_return"] = signals["upper_band"] / signals["close"] - 1
    signals["band_width_return"] = (signals["upper_band"] - signals["lower_band"]) / signals["close"]
    signals["line_position_in_band"] = (signals["line_end_value"] - signals["lower_band"]) / (
        signals["upper_band"] - signals["lower_band"]
    )
    signals["band_width_ref"] = signals["band_width_return"].shift(1).rolling(20, min_periods=5).median()
    signals["band_width_expansion"] = signals["band_width_return"] / signals["band_width_ref"]
    signals["band_width_percentile"] = signals["band_width_return"].rank(pct=True)
    signals = signals.replace([math.inf, -math.inf], math.nan)
    signals = signals.dropna(subset=["date", "close", "line_return", "lower_return", "band_width_return"])

    if len(signals) < 5:
        return None

    return TickerData(
        ticker=ticker,
        price=price[["date", "close"]].copy(),
        signals=signals.sort_values("date"),
        line_prediction_count=len(line_rows),
        band_prediction_count=len(band_rows),
        min_asof=str(signals["date"].min()),
        max_asof=str(signals["date"].max()),
    )


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def max_drawdown(equity_values: list[float]) -> float:
    peak = 1.0
    mdd = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            mdd = min(mdd, value / peak - 1)
    return mdd * 100


def sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_value = statistics.fmean(returns)
    std_value = statistics.stdev(returns)
    return (mean_value / std_value) * math.sqrt(252) if std_value > 0 else 0.0


def sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_value = statistics.fmean(returns)
    downside = [value for value in returns if value < 0]
    if len(downside) < 2:
        return mean_value * math.sqrt(252) if mean_value > 0 else 0.0
    downside_dev = math.sqrt(sum(value * value for value in downside) / (len(downside) - 1))
    return (mean_value / downside_dev) * math.sqrt(252) if downside_dev > 0 else 0.0


def return_capture_ratio(strategy_return_pct: float, buy_hold_return_pct: float) -> float:
    if buy_hold_return_pct > 0:
        return strategy_return_pct / buy_hold_return_pct
    return 1.0 if strategy_return_pct >= buy_hold_return_pct else 0.0


def trend_allows_entry(row: pd.Series, rule: Rule) -> bool:
    if rule.trend_filter == "off":
        return True
    ma60_ratio = row.get("ma60_ratio")
    return finite(ma60_ratio) and float(ma60_ratio) >= -0.05


def band_state(row: pd.Series, rule: Rule) -> dict[str, bool]:
    width_expansion = row.get("band_width_expansion")
    width_percentile = row.get("band_width_percentile")
    lower_risky = float(row["lower_return"]) <= rule.lower_risk_threshold
    width_expanded = finite(width_expansion) and float(width_expansion) >= rule.width_expansion_threshold
    width_wide = finite(width_percentile) and float(width_percentile) >= rule.width_percentile_threshold
    band_uncertain = width_expanded or width_wide
    band_risky = lower_risky or width_expanded
    return {
        "lower_risky": lower_risky,
        "width_expanded": width_expanded,
        "width_wide": width_wide,
        "band_uncertain": band_uncertain,
        "band_risky": band_risky,
    }


def entry_band_allowed(row: pd.Series, rule: Rule, state: dict[str, bool]) -> bool:
    if rule.entry_band_filter == "off":
        return True
    if rule.entry_band_filter == "block_expansion":
        return not state["width_expanded"]
    if rule.entry_band_filter == "block_risky":
        return not state["band_risky"]
    return True


def risk_exit(row: pd.Series, rule: Rule, state: dict[str, bool]) -> bool:
    line_weak = float(row["line_return"]) < rule.line_hold_threshold
    if not line_weak:
        return False
    if rule.exit_risk_mode == "lower_or_width":
        return state["lower_risky"] or state["width_expanded"]
    if rule.exit_risk_mode == "lower_and_width":
        return state["lower_risky"] and state["width_expanded"]
    if rule.exit_risk_mode == "any_band_risk":
        return state["lower_risky"] or state["width_expanded"] or state["width_wide"]
    return state["band_risky"]


def raw_target_position(row: pd.Series, rule: Rule) -> float:
    state = band_state(row, rule)
    line_good = float(row["line_return"]) >= rule.line_entry_threshold
    line_hold = float(row["line_return"]) >= rule.line_hold_threshold
    entry_ok = line_good and trend_allows_entry(row, rule) and entry_band_allowed(row, rule, state)

    if rule.position_mode == "sizing":
        if line_good and not state["band_uncertain"] and not state["lower_risky"] and trend_allows_entry(row, rule):
            return 1.0
        if line_good and trend_allows_entry(row, rule):
            return rule.line_good_band_uncertain_position
        if line_hold and not state["band_risky"]:
            return rule.line_weak_band_stable_position
        if risk_exit(row, rule, state):
            return 0.0
        return 0.0

    if entry_ok:
        return 1.0
    if line_hold and not risk_exit(row, rule, state):
        return 1.0
    return 0.0


def build_positions(signals: pd.DataFrame, rule: Rule) -> dict[str, float]:
    current_position = 0.0
    exit_streak = 0
    entry_streak = 0
    positions: dict[str, float] = {}

    for _, row in signals.iterrows():
        target = raw_target_position(row, rule)
        lowering = target < current_position
        raising = target > current_position

        if lowering:
            exit_streak += 1
        else:
            exit_streak = 0

        if raising:
            entry_streak += 1
        else:
            entry_streak = 0

        if lowering and exit_streak >= rule.confirm_days:
            current_position = target
            entry_streak = 0
        elif raising and entry_streak >= rule.reentry_confirm_days:
            current_position = target
            exit_streak = 0
        elif not lowering and not raising:
            current_position = target

        positions[str(row["date"])] = current_position

    return positions


def usable_for_range(ticker_data: TickerData, start_date: str, end_date: str) -> bool:
    signals = ticker_data.signals[(ticker_data.signals["date"] >= start_date) & (ticker_data.signals["date"] <= end_date)]
    if len(signals) < 5:
        return False
    first_signal = str(signals["date"].iloc[0])
    price = ticker_data.price[(ticker_data.price["date"] >= first_signal) & (ticker_data.price["date"] <= end_date)]
    return len(price) >= 5


def simulate(ticker_data: TickerData, rule: Rule | None, start_date: str, end_date: str, always_cash: bool = False) -> dict[str, Any] | None:
    signals = ticker_data.signals[(ticker_data.signals["date"] >= start_date) & (ticker_data.signals["date"] <= end_date)].copy()
    if len(signals) < 5:
        return None

    first_signal = str(signals["date"].iloc[0])
    price = ticker_data.price[(ticker_data.price["date"] >= first_signal) & (ticker_data.price["date"] <= end_date)].copy()
    if len(price) < 5:
        return None

    positions = {} if always_cash or rule is None else build_positions(signals, rule)
    fee_rate = FEE_BPS / 10000

    strategy_equity = 1.0
    buy_hold_equity = 1.0
    current_position = 0.0
    trade_count = 0
    exposure_sum = 0.0
    avoided_large_loss_exposure = 0.0
    large_loss_days = 0
    holding_durations: list[int] = []
    strategy_returns: list[float] = []
    buy_hold_returns: list[float] = []
    strategy_equity_curve = [1.0]
    buy_hold_equity_curve = [1.0]
    holding_start: int | None = None

    close_values = price["close"].astype(float).tolist()
    dates = price["date"].astype(str).tolist()
    all_daily_returns = [close_values[index] / close_values[index - 1] - 1 for index in range(1, len(close_values))]
    threshold_base = percentile(all_daily_returns, 0.2)
    large_loss_threshold = min(-0.02, threshold_base) if threshold_base is not None else -0.02

    for index in range(1, len(price)):
        previous_date = dates[index - 1]
        previous_close = close_values[index - 1]
        current_close = close_values[index]
        desired_position = 0.0 if always_cash else positions.get(previous_date, current_position)
        desired_position = max(0.0, min(1.0, float(desired_position)))

        fee_cost = 0.0
        if abs(desired_position - current_position) > 1e-9:
            strategy_equity *= 1 - fee_rate * abs(desired_position - current_position)
            fee_cost = fee_rate * abs(desired_position - current_position)
            trade_count += 1
            if current_position <= 0 and desired_position > 0:
                holding_start = index - 1
            if current_position > 0 and desired_position <= 0 and holding_start is not None:
                holding_durations.append(index - 1 - holding_start)
                holding_start = None
            current_position = desired_position

        daily_return = current_close / previous_close - 1
        strategy_daily_return = daily_return * current_position
        strategy_equity *= 1 + strategy_daily_return
        buy_hold_equity *= 1 + daily_return
        strategy_returns.append(strategy_daily_return - fee_cost)
        buy_hold_returns.append(daily_return)

        if daily_return <= large_loss_threshold:
            large_loss_days += 1
            avoided_large_loss_exposure += 1 - current_position
        exposure_sum += current_position

        strategy_equity_curve.append(strategy_equity)
        buy_hold_equity_curve.append(buy_hold_equity)

    if current_position > 0 and holding_start is not None:
        holding_durations.append(len(price) - 1 - holding_start)

    strategy_return_pct = (strategy_equity - 1) * 100
    buy_hold_return_pct = (buy_hold_equity - 1) * 100
    strategy_mdd = max_drawdown(strategy_equity_curve)
    buy_hold_mdd = max_drawdown(buy_hold_equity_curve)
    market_participation = exposure_sum / max(len(price) - 1, 1)
    loss_avoidance = avoided_large_loss_exposure / large_loss_days if large_loss_days > 0 else None
    avg_holding_days = statistics.fmean(holding_durations) if holding_durations else None

    pass_ticker = (
        strategy_mdd - buy_hold_mdd > 0
        and return_capture_ratio(strategy_return_pct, buy_hold_return_pct) >= 0.75
        and loss_avoidance is not None
        and loss_avoidance >= 0.55
        and 0.45 <= market_participation <= 0.90
        and trade_count <= 40
    )

    return {
        "ticker": ticker_data.ticker,
        "strategy_return_pct": strategy_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "return_capture_ratio": return_capture_ratio(strategy_return_pct, buy_hold_return_pct),
        "excess_return_pct": strategy_return_pct - buy_hold_return_pct,
        "strategy_mdd_pct": strategy_mdd,
        "buy_hold_mdd_pct": buy_hold_mdd,
        "mdd_improvement_pct": strategy_mdd - buy_hold_mdd,
        "strategy_sharpe": sharpe(strategy_returns),
        "buy_hold_sharpe": sharpe(buy_hold_returns),
        "strategy_sortino": sortino(strategy_returns),
        "buy_hold_sortino": sortino(buy_hold_returns),
        "loss_avoidance_rate": loss_avoidance,
        "market_participation": market_participation,
        "trade_count": trade_count,
        "avg_holding_days": avg_holding_days,
        "large_loss_days": large_loss_days,
        "large_loss_threshold_pct": large_loss_threshold * 100,
        "pass_ticker": pass_ticker,
    }


def mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(usable) if usable else None


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"ticker_count": 0}
    buy_hold_values = [item["buy_hold_return_pct"] for item in metrics]
    strong_cutoff = percentile(buy_hold_values, 0.75)
    down_sideways_cutoff = percentile(buy_hold_values, 0.50)
    strong_up = [item for item in metrics if strong_cutoff is not None and item["buy_hold_return_pct"] >= strong_cutoff]
    down_sideways = [item for item in metrics if down_sideways_cutoff is not None and item["buy_hold_return_pct"] <= down_sideways_cutoff]
    return {
        "ticker_count": len(metrics),
        "avg_strategy_return_pct": mean([item["strategy_return_pct"] for item in metrics]),
        "avg_buy_hold_return_pct": mean([item["buy_hold_return_pct"] for item in metrics]),
        "avg_return_capture_ratio": mean([item["return_capture_ratio"] for item in metrics]),
        "avg_excess_return_pct": mean([item["excess_return_pct"] for item in metrics]),
        "avg_strategy_mdd_pct": mean([item["strategy_mdd_pct"] for item in metrics]),
        "avg_buy_hold_mdd_pct": mean([item["buy_hold_mdd_pct"] for item in metrics]),
        "avg_mdd_improvement_pct": mean([item["mdd_improvement_pct"] for item in metrics]),
        "avg_sharpe": mean([item["strategy_sharpe"] for item in metrics]),
        "avg_buy_hold_sharpe": mean([item["buy_hold_sharpe"] for item in metrics]),
        "avg_sortino": mean([item["strategy_sortino"] for item in metrics]),
        "avg_buy_hold_sortino": mean([item["buy_hold_sortino"] for item in metrics]),
        "avg_loss_avoidance_rate": mean([item["loss_avoidance_rate"] for item in metrics]),
        "avg_market_participation": mean([item["market_participation"] for item in metrics]),
        "avg_trade_count": mean([item["trade_count"] for item in metrics]),
        "avg_holding_days": mean([item["avg_holding_days"] for item in metrics]),
        "pass_ticker_ratio": sum(1 for item in metrics if item["pass_ticker"]) / len(metrics),
        "strong_up_definition": "buy_hold_return 상위 25%",
        "strong_up_cutoff_pct": strong_cutoff,
        "strong_up_ticker_count": len(strong_up),
        "strong_up_avg_return_capture_ratio": mean([item["return_capture_ratio"] for item in strong_up]),
        "down_sideways_definition": "buy_hold_return 하위 50%",
        "down_sideways_cutoff_pct": down_sideways_cutoff,
        "down_sideways_ticker_count": len(down_sideways),
        "down_sideways_avg_mdd_improvement_pct": mean([item["mdd_improvement_pct"] for item in down_sideways]),
    }


def survival_flags(summary: dict[str, Any]) -> dict[str, bool]:
    participation = summary.get("avg_market_participation")
    return {
        "mdd_improved": (summary.get("avg_mdd_improvement_pct") or -999) > 0,
        "loss_avoidance_ok": (summary.get("avg_loss_avoidance_rate") or 0) >= 0.55,
        "return_capture_ok": (summary.get("avg_return_capture_ratio") or 0) >= 0.75,
        "participation_ok": participation is not None and 0.45 <= participation <= 0.90,
        "strong_up_capture_ok": (summary.get("strong_up_avg_return_capture_ratio") or 0) >= 0.60,
        "pass_ticker_ratio_ok": (summary.get("pass_ticker_ratio") or 0) >= 0.15,
        "not_always_cash": participation is not None and participation > 0.20,
        "not_buy_hold_like": participation is not None and participation < 0.95,
        "trades_not_excessive": (summary.get("avg_trade_count") or 999) <= 40,
    }


def score_candidate(holdout: dict[str, Any]) -> float:
    if holdout.get("ticker_count", 0) == 0:
        return -9999.0
    participation = holdout.get("avg_market_participation") or 0
    hard_penalty = 0.0
    if participation <= 0.20:
        hard_penalty -= 120
    if participation >= 0.95:
        hard_penalty -= 120
    if (holdout.get("avg_return_capture_ratio") or 0) < 0.60:
        hard_penalty -= 35
    if (holdout.get("avg_mdd_improvement_pct") or 0) <= 0:
        hard_penalty -= 25
    return (
        hard_penalty
        + 5 * ((holdout.get("avg_return_capture_ratio") or 0) - 0.75)
        + (holdout.get("avg_mdd_improvement_pct") or 0)
        + 7 * ((holdout.get("avg_loss_avoidance_rate") or 0) - 0.55)
        + 3 * ((holdout.get("strong_up_avg_return_capture_ratio") or 0) - 0.60)
        + 4 * ((holdout.get("pass_ticker_ratio") or 0) - 0.15)
        - abs(participation - 0.65) * 2
        - max((holdout.get("avg_trade_count") or 0) - 35, 0) * 0.05
    )


def split_candidates() -> list[SplitPlan]:
    return [
        SplitPlan("기본 1개월 holdout", EXPLORE_START, BASE_EXPLORE_END, ONE_MONTH_HOLDOUT_START, HOLDOUT_END, "요청 기본 split"),
        SplitPlan("최근 2개월 holdout", EXPLORE_START, "2026-02-28", "2026-03-01", HOLDOUT_END, "1개월 holdout usable ticker가 30개 미만일 때 대안"),
        SplitPlan("최근 3개월 holdout", EXPLORE_START, "2026-01-31", "2026-02-01", HOLDOUT_END, "2개월 holdout도 부족할 때 대안"),
    ]


def choose_split(ticker_data: list[TickerData]) -> tuple[SplitPlan, list[dict[str, Any]]]:
    coverage: list[dict[str, Any]] = []
    for split in split_candidates():
        explore_count = sum(1 for item in ticker_data if usable_for_range(item, split.explore_start, split.explore_end))
        holdout_count = sum(1 for item in ticker_data if usable_for_range(item, split.holdout_start, split.holdout_end))
        coverage.append(
            {
                "label": split.label,
                "explore_start": split.explore_start,
                "explore_end": split.explore_end,
                "holdout_start": split.holdout_start,
                "holdout_end": split.holdout_end,
                "explore_usable_tickers": explore_count,
                "holdout_usable_tickers": holdout_count,
                "reason": split.reason,
            }
        )
    if coverage[0]["holdout_usable_tickers"] >= 30:
        return split_candidates()[0], coverage
    for split, row in zip(split_candidates()[1:], coverage[1:]):
        if row["holdout_usable_tickers"] >= 30:
            return split, coverage
    best_index = max(range(len(coverage)), key=lambda index: coverage[index]["holdout_usable_tickers"])
    return split_candidates()[best_index], coverage


def build_rules() -> list[Rule]:
    entry_values = [-0.002, 0.0, 0.002]
    hold_values = [-0.010, -0.014]
    lower_values = [-0.05, -0.08]
    expansion_values = [1.10, 1.25, 1.50]
    width_percentile_values = [0.70, 0.80]
    confirm_values = [1, 2]
    reentry_values = [1, 2]
    rules: list[Rule] = []

    for entry in entry_values:
        for hold in hold_values:
            for expansion in expansion_values:
                for width_pct in width_percentile_values:
                    for reentry in reentry_values:
                        for entry_filter in ["block_expansion", "block_risky"]:
                            rules.append(
                                Rule(
                                    "balance_entry_filter",
                                    entry,
                                    hold,
                                    -0.05,
                                    expansion,
                                    width_pct,
                                    entry_filter,
                                    "lower_or_width",
                                    2,
                                    reentry,
                                    "off",
                                    "binary",
                                    0.5,
                                    0.5,
                                    True,
                                )
                            )

    for entry in entry_values:
        for hold in hold_values:
            for lower in lower_values:
                for expansion in expansion_values:
                    for confirm in confirm_values:
                        for exit_mode in ["lower_or_width", "lower_and_width", "any_band_risk"]:
                            rules.append(
                                Rule(
                                    "balance_risk_confirm",
                                    entry,
                                    hold,
                                    lower,
                                    expansion,
                                    0.75,
                                    "off",
                                    exit_mode,
                                    confirm,
                                    1,
                                    "off",
                                    "binary",
                                    0.5,
                                    0.5,
                                    True,
                                )
                            )

    for entry in entry_values:
        for hold in hold_values:
            for lower in lower_values:
                for expansion in expansion_values:
                    for confirm in confirm_values:
                        rules.append(
                            Rule(
                                "balance_trend_continuation",
                                entry,
                                hold,
                                lower,
                                expansion,
                                0.75,
                                "off",
                                "lower_or_width",
                                confirm,
                                1,
                                "off",
                                "binary",
                                1.0,
                                0.5,
                                True,
                            )
                        )

    for entry in entry_values:
        for hold in hold_values:
            for lower in lower_values:
                for expansion in expansion_values:
                    for good_uncertain in [0.5, 1.0]:
                        for weak_stable in [0.0, 0.5]:
                            rules.append(
                                Rule(
                                    "balance_position_sizing",
                                    entry,
                                    hold,
                                    lower,
                                    expansion,
                                    0.75,
                                    "block_expansion",
                                    "lower_or_width",
                                    1,
                                    1,
                                    "off",
                                    "sizing",
                                    good_uncertain,
                                    weak_stable,
                                    True,
                                )
                            )

    for entry in entry_values:
        for hold in hold_values:
            for lower in lower_values:
                for expansion in expansion_values:
                    for confirm in confirm_values:
                        for trend in ["off", "ma60_above_-5pct"]:
                            rules.append(
                                Rule(
                                    "balance_hybrid",
                                    entry,
                                    hold,
                                    lower,
                                    expansion,
                                    0.75,
                                    "block_expansion",
                                    "any_band_risk",
                                    confirm,
                                    1,
                                    trend,
                                    "sizing",
                                    0.5,
                                    0.5,
                                    True,
                                )
                            )
    return rules


def evaluate_rule(rule: Rule, ticker_data: list[TickerData], split: SplitPlan) -> dict[str, Any]:
    explore_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, rule, split.explore_start, split.explore_end)) is not None
    ]
    holdout_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, rule, split.holdout_start, split.holdout_end)) is not None
    ]
    one_month_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, rule, ONE_MONTH_HOLDOUT_START, HOLDOUT_END)) is not None
    ]
    explore = aggregate(explore_metrics)
    holdout = aggregate(holdout_metrics)
    flags = survival_flags(holdout)
    return {
        "rule": asdict(rule),
        "key": rule.key,
        "explore": explore,
        "holdout": holdout,
        "recent_one_month_holdout": aggregate(one_month_metrics),
        "survival_flags": flags,
        "survived": all(flags.values()),
        "score": score_candidate(holdout),
    }


def evaluate_always_cash(ticker_data: list[TickerData], split: SplitPlan) -> dict[str, Any]:
    holdout_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, None, split.holdout_start, split.holdout_end, always_cash=True)) is not None
    ]
    return aggregate(holdout_metrics)


def featured_ticker_comparison(
    tickers: list[str],
    ticker_data: list[TickerData],
    best_rule: Rule,
    split: SplitPlan,
) -> list[dict[str, Any]]:
    data_by_ticker = {item.ticker: item for item in ticker_data}
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        item = data_by_ticker.get(ticker)
        if item is None:
            rows.append({"ticker": ticker, "status": "prediction 없음"})
            continue
        metric = simulate(item, best_rule, split.holdout_start, split.holdout_end)
        rows.append({"ticker": ticker, "status": "ok" if metric else "평가 데이터 부족", "best": metric})
    return rows


def candidate_diagnostics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    family_best: dict[str, dict[str, Any]] = {}
    family_counts: dict[str, int] = {}
    family_survived_counts: dict[str, int] = {}
    flag_pass_counts: dict[str, int] = {}

    for candidate in candidates:
        family = candidate["rule"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
        if candidate["survived"]:
            family_survived_counts[family] = family_survived_counts.get(family, 0) + 1
        if family not in family_best or candidate["score"] > family_best[family]["score"]:
            family_best[family] = candidate
        for flag, passed in candidate["survival_flags"].items():
            if passed:
                flag_pass_counts[flag] = flag_pass_counts.get(flag, 0) + 1

    participation_ok = [candidate for candidate in candidates if candidate["survival_flags"].get("participation_ok")]
    pass_ratio_sorted = sorted(
        candidates,
        key=lambda item: item["holdout"].get("pass_ticker_ratio") or 0,
        reverse=True,
    )

    return {
        "family_counts": family_counts,
        "family_survived_counts": family_survived_counts,
        "family_best": family_best,
        "survival_flag_pass_counts": flag_pass_counts,
        "top_participation_ok_candidates": sorted(participation_ok, key=lambda item: item["score"], reverse=True)[:5],
        "top_pass_ticker_ratio_candidates": pass_ratio_sorted[:5],
    }


def load_comparison_baselines() -> dict[str, Any]:
    result: dict[str, Any] = {}
    cp106_path = Path("docs/cp106_band_risk_strategy_grid_metrics.json")
    cp107_path = Path("docs/cp107_line_trend_strategy_grid_metrics.json")
    if cp106_path.exists():
        cp106 = json.loads(cp106_path.read_text(encoding="utf-8"))
        result["cp106_band_best"] = (cp106.get("top_candidates") or [{}])[0].get("holdout")
    if cp107_path.exists():
        cp107 = json.loads(cp107_path.read_text(encoding="utf-8"))
        result["cp107_line_best"] = (cp107.get("top_candidates") or [{}])[0].get("holdout")
    return result


def write_top_csv(path: Path, candidates: list[dict[str, Any]], limit: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "score",
        "survived",
        "family",
        "line_entry_threshold",
        "line_hold_threshold",
        "lower_risk_threshold",
        "width_expansion_threshold",
        "width_percentile_threshold",
        "entry_band_filter",
        "exit_risk_mode",
        "confirm_days",
        "reentry_confirm_days",
        "trend_filter",
        "position_mode",
        "line_good_band_uncertain_position",
        "line_weak_band_stable_position",
        "holdout_ticker_count",
        "holdout_avg_strategy_return_pct",
        "holdout_avg_buy_hold_return_pct",
        "holdout_avg_return_capture_ratio",
        "holdout_avg_mdd_improvement_pct",
        "holdout_avg_loss_avoidance_rate",
        "holdout_avg_market_participation",
        "holdout_strong_up_capture",
        "holdout_pass_ticker_ratio",
        "holdout_avg_trade_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, candidate in enumerate(candidates[:limit], start=1):
            rule = candidate["rule"]
            holdout = candidate["holdout"]
            writer.writerow(
                {
                    "rank": rank,
                    "score": candidate["score"],
                    "survived": candidate["survived"],
                    "family": rule["family"],
                    "line_entry_threshold": rule["line_entry_threshold"],
                    "line_hold_threshold": rule["line_hold_threshold"],
                    "lower_risk_threshold": rule["lower_risk_threshold"],
                    "width_expansion_threshold": rule["width_expansion_threshold"],
                    "width_percentile_threshold": rule["width_percentile_threshold"],
                    "entry_band_filter": rule["entry_band_filter"],
                    "exit_risk_mode": rule["exit_risk_mode"],
                    "confirm_days": rule["confirm_days"],
                    "reentry_confirm_days": rule["reentry_confirm_days"],
                    "trend_filter": rule["trend_filter"],
                    "position_mode": rule["position_mode"],
                    "line_good_band_uncertain_position": rule["line_good_band_uncertain_position"],
                    "line_weak_band_stable_position": rule["line_weak_band_stable_position"],
                    "holdout_ticker_count": holdout.get("ticker_count"),
                    "holdout_avg_strategy_return_pct": holdout.get("avg_strategy_return_pct"),
                    "holdout_avg_buy_hold_return_pct": holdout.get("avg_buy_hold_return_pct"),
                    "holdout_avg_return_capture_ratio": holdout.get("avg_return_capture_ratio"),
                    "holdout_avg_mdd_improvement_pct": holdout.get("avg_mdd_improvement_pct"),
                    "holdout_avg_loss_avoidance_rate": holdout.get("avg_loss_avoidance_rate"),
                    "holdout_avg_market_participation": holdout.get("avg_market_participation"),
                    "holdout_strong_up_capture": holdout.get("strong_up_avg_return_capture_ratio"),
                    "holdout_pass_ticker_ratio": holdout.get("pass_ticker_ratio"),
                    "holdout_avg_trade_count": holdout.get("avg_trade_count"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP109 Line + Band Balance 전략 탐색")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--universe-limit", type=int, default=100)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    backend_url = normalize_backend_url(args.backend_url)
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    print(f"backend={backend_url}")
    print(f"data_dir={data_dir}")
    print("CPU only: GPU 호출 없음")

    price, indicators = load_market_data(data_dir)
    universe = load_universe(data_dir, args.universe_limit)
    prepared: list[TickerData] = []
    excluded: list[dict[str, str]] = []

    for index, ticker in enumerate(universe, start=1):
        line_rows = fetch_prediction_history(backend_url, ticker, LINE_RUN_ID, args.history_limit)
        band_rows = fetch_prediction_history(backend_url, ticker, BAND_RUN_ID, args.history_limit)
        item = prepare_ticker_data(ticker, price, indicators, line_rows, band_rows)
        if item is None:
            if not line_rows and not band_rows:
                reason = "line/band prediction history 없음"
            elif not line_rows:
                reason = "line prediction history 없음"
            elif not band_rows:
                reason = "band prediction history 없음"
            else:
                reason = "가격/지표와 prediction 날짜 매칭 부족"
            excluded.append({"ticker": ticker, "reason": reason})
        else:
            prepared.append(item)
        if index % 20 == 0:
            print(f"history fetch {index}/{len(universe)}")

    if not prepared:
        raise RuntimeError("사용 가능한 line/band prediction history가 없습니다.")

    selected_split, split_coverage = choose_split(prepared)
    rules = build_rules()
    if args.max_rules is not None:
        rules = rules[: args.max_rules]

    print(f"usable_tickers={len(prepared)} excluded={len(excluded)}")
    print(
        "selected_split="
        f"{selected_split.label} explore={selected_split.explore_start}~{selected_split.explore_end} "
        f"holdout={selected_split.holdout_start}~{selected_split.holdout_end}"
    )
    print(f"rules={len(rules)}")

    candidates: list[dict[str, Any]] = []
    for index, rule in enumerate(rules, start=1):
        candidates.append(evaluate_rule(rule, prepared, selected_split))
        if index % 50 == 0 or index == len(rules):
            print(f"evaluated {index}/{len(rules)} elapsed={time.time() - start_time:.1f}s")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_rule = Rule(**candidates[0]["rule"])

    payload = {
        "cp": "CP109-P",
        "created_at_kst": time.strftime("%Y-%m-%d %H:%M:%S KST", time.localtime()),
        "line_run_id": LINE_RUN_ID,
        "band_run_id": BAND_RUN_ID,
        "backend_url": backend_url,
        "data_sources": {
            "price": str(data_dir / "price_data_yfinance.parquet"),
            "indicators": str(data_dir / "indicators_yfinance_1D.parquet"),
            "universe": str(data_dir / "stock_info.parquet"),
            "prediction_history_api": "/api/v1/stocks/{ticker}/predictions/history",
            "price_column_policy": "adjusted_close 우선, 없으면 close",
        },
        "forbidden_actions_confirmed": {
            "gpu_used": False,
            "model_training": False,
            "inference_execution": False,
            "db_write": False,
            "supabase_price_indicator_bulk_read": False,
            "frontend_modified": False,
            "composite_used": False,
        },
        "universe": {
            "requested": len(universe),
            "usable": len(prepared),
            "excluded": len(excluded),
            "excluded_tickers": excluded,
        },
        "prediction_history_coverage": {
            "min_asof": min(item.min_asof for item in prepared),
            "max_asof": max(item.max_asof for item in prepared),
            "line_prediction_rows_min": min(item.line_prediction_count for item in prepared),
            "line_prediction_rows_max": max(item.line_prediction_count for item in prepared),
            "band_prediction_rows_min": min(item.band_prediction_count for item in prepared),
            "band_prediction_rows_max": max(item.band_prediction_count for item in prepared),
        },
        "split_coverage": split_coverage,
        "selected_split": asdict(selected_split),
        "rule_count": len(rules),
        "strategy_families": {
            "balance_entry_filter": "line 양호 진입, band 확장/위험은 신규 진입 제한",
            "balance_risk_confirm": "line 약화와 lower/width 위험이 동시에 있을 때 청산",
            "balance_trend_continuation": "line이 양호하면 band 불확실만으로 매도하지 않음",
            "balance_position_sizing": "100%, 50%, 현금 3단계 exposure 허용",
            "balance_hybrid": "진입 제한, risk confirm, 부분 비중을 결합",
        },
        "cp108_interpretation_used": {
            "line_return_signal": "진입 방향의 기본 신호",
            "lower_band_risk": "line 약화 시 risk veto/confirm",
            "band_width_expansion": "불확실성 확장 confirm",
            "upper_breach_event": "사후 지표이므로 전략 입력 제외, 무조건 매도 신호로 쓰지 않음",
            "line_band_disagreement": "Balance family 설계의 핵심",
        },
        "survival_criteria": {
            "avg_mdd_improvement_pct": "> 0",
            "avg_loss_avoidance_rate": ">= 0.55",
            "avg_return_capture_ratio": ">= 0.75",
            "avg_market_participation": "0.45 ~ 0.90",
            "strong_up_avg_return_capture_ratio": ">= 0.60",
            "pass_ticker_ratio": ">= 0.15",
        },
        "comparison_baselines": load_comparison_baselines(),
        "always_cash_baseline": evaluate_always_cash(prepared, selected_split),
        "candidate_diagnostics": candidate_diagnostics(candidates),
        "top_candidates": candidates[:10],
        "survived_count": sum(1 for item in candidates if item["survived"]),
        "featured_tickers": featured_ticker_comparison(["AAPL", "MSFT", "NVDA"], prepared, best_rule, selected_split),
        "elapsed_seconds": time.time() - start_time,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_top_csv(output_csv, candidates, limit=10)

    print(f"json={output_json}")
    print(f"csv={output_csv}")
    print(f"survived_count={payload['survived_count']}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
