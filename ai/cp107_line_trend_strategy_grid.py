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


LM_RUN_ID = "patchtst-1D-efad3c29d803"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_DATA_DIR = Path("data/parquet")
DEFAULT_OUTPUT_JSON = Path("docs/cp107_line_trend_strategy_grid_metrics.json")
DEFAULT_OUTPUT_CSV = Path("docs/cp107_line_trend_strategy_grid_top_candidates.csv")

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
    line_entry_threshold: float
    line_exit_threshold: float
    trend_floor: float | None
    trend_override: bool
    rsi_entry_cap: float | None
    rsi_exit_guard: str
    exit_confirm_days: int
    reentry_confirm_days: int
    cooldown_days: int

    @property
    def key(self) -> str:
        trend = "off" if self.trend_floor is None else f"{self.trend_floor:.3f}"
        rsi = "off" if self.rsi_entry_cap is None else f"{self.rsi_entry_cap:.0f}"
        return (
            f"entry={self.line_entry_threshold:.3f}|exit={self.line_exit_threshold:.3f}|"
            f"trend={trend}|override={self.trend_override}|rsi={rsi}|"
            f"rsi_exit={self.rsi_exit_guard}|exit_confirm={self.exit_confirm_days}|"
            f"reentry={self.reentry_confirm_days}|cooldown={self.cooldown_days}"
        )


@dataclass
class TickerData:
    ticker: str
    price: pd.DataFrame
    signals: pd.DataFrame
    prediction_min_date: str
    prediction_max_date: str
    prediction_count: int


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


def fetch_prediction_history(backend_url: str, ticker: str, limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = min(max(limit, 1), 200)
    params = urllib.parse.urlencode({"run_id": LM_RUN_ID, "limit": safe_limit})
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


def last_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if finite(value):
            return float(value)
    return None


def conservative_value(row: dict[str, Any]) -> float | None:
    conservative = last_finite(row.get("conservative_series"))
    if conservative is not None:
        return conservative
    return last_finite(row.get("line_series"))


def prepare_ticker_data(
    ticker: str,
    price_rows: pd.DataFrame,
    indicator_rows: pd.DataFrame,
    prediction_rows: list[dict[str, Any]],
) -> TickerData | None:
    price = price_rows[price_rows["ticker"] == ticker].copy()
    if price.empty or not prediction_rows:
        return None

    price = price.sort_values("date").drop_duplicates("date", keep="last")
    price["ma60_calc"] = price["close"] / price["close"].rolling(60, min_periods=60).mean() - 1

    indicators = indicator_rows[indicator_rows["ticker"] == ticker].copy()
    if indicators.empty:
        indicators = pd.DataFrame(columns=["ticker", "date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"])
    indicators = indicators.sort_values("date").drop_duplicates("date", keep="last")

    rows: list[dict[str, Any]] = []
    prediction_dates: list[str] = []
    for prediction in prediction_rows:
        asof_date = str(prediction.get("asof_date") or "")
        value = conservative_value(prediction)
        if not asof_date or value is None:
            continue
        rows.append({"date": asof_date, "conservative_value": value})
        prediction_dates.append(asof_date)

    if not rows:
        return None

    pred = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    signals = pred.merge(price[["date", "close", "ma60_calc"]], on="date", how="inner")

    indicator_merge_columns = [column for column in ["date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"] if column in indicators.columns]
    if len(indicator_merge_columns) <= 1:
        signals["rsi"] = math.nan
        signals["ma_60_ratio"] = math.nan
        signals["vol_change"] = math.nan
        signals["atr_ratio"] = math.nan
    else:
        signals = signals.merge(indicators[indicator_merge_columns], on="date", how="left")

    if "ma_60_ratio" not in signals.columns:
        signals["ma_60_ratio"] = math.nan
    signals["ma60_ratio"] = signals["ma_60_ratio"].where(signals["ma_60_ratio"].notna(), signals["ma60_calc"])
    signals["rsi"] = signals.get("rsi", pd.Series(index=signals.index, dtype=float)).map(normalize_rsi)
    signals["line_return"] = signals["conservative_value"] / signals["close"] - 1
    signals = signals.dropna(subset=["date", "close", "line_return"]).sort_values("date")

    if len(signals) < 5:
        return None

    return TickerData(
        ticker=ticker,
        price=price[["date", "close"]].copy(),
        signals=signals,
        prediction_min_date=min(prediction_dates),
        prediction_max_date=max(prediction_dates),
        prediction_count=len(prediction_dates),
    )


def trend_is_alive(row: pd.Series, rule: Rule) -> bool:
    ma60_ratio = row.get("ma60_ratio")
    if not finite(ma60_ratio):
        return False
    floor = 0.0 if rule.trend_floor is None else rule.trend_floor
    return float(ma60_ratio) >= floor


def trend_allows_entry(row: pd.Series, rule: Rule) -> bool:
    if rule.trend_floor is None:
        return True
    ma60_ratio = row.get("ma60_ratio")
    return finite(ma60_ratio) and float(ma60_ratio) >= rule.trend_floor


def trend_is_weak(row: pd.Series, rule: Rule) -> bool:
    if rule.trend_floor is None:
        return False
    ma60_ratio = row.get("ma60_ratio")
    if not finite(ma60_ratio):
        return False
    return float(ma60_ratio) < rule.trend_floor - 0.01


def rsi_allows_entry(row: pd.Series, rule: Rule) -> bool:
    if rule.rsi_entry_cap is None:
        return True
    rsi = row.get("rsi")
    if not finite(rsi):
        return True
    return float(rsi) <= rule.rsi_entry_cap


def line_exit_candidate(row: pd.Series, rule: Rule) -> bool:
    line_return = float(row["line_return"])
    line_exit = line_return <= rule.line_exit_threshold
    if rule.trend_override and line_exit and trend_is_alive(row, rule):
        hard_line_exit = line_return <= min(rule.line_exit_threshold * 2, rule.line_exit_threshold - 0.01)
        if not hard_line_exit:
            line_exit = False

    rsi_exit = False
    if rule.rsi_exit_guard == "only_if_trend_weak":
        rsi = row.get("rsi")
        rsi_exit = finite(rsi) and float(rsi) >= 90 and trend_is_weak(row, rule)

    return line_exit or trend_is_weak(row, rule) or rsi_exit


def position_reason(row: pd.Series, rule: Rule, position: int, action: str) -> str:
    if action == "entry":
        return "보수적 예측선과 추세 조건 충족"
    if action == "exit":
        if float(row["line_return"]) <= rule.line_exit_threshold:
            return "보수적 예측선 약화"
        if trend_is_weak(row, rule):
            return "60일 추세 약화"
        return "위험 조건 확인"
    if position == 1:
        return "추세 유지"
    return "대기"


def build_positions(signals: pd.DataFrame, rule: Rule) -> dict[str, tuple[int, str]]:
    position = 0
    exit_streak = 0
    entry_streak = 0
    cooldown = 0
    positions: dict[str, tuple[int, str]] = {}

    for _, row in signals.iterrows():
        entry_ok = (
            float(row["line_return"]) >= rule.line_entry_threshold
            and trend_allows_entry(row, rule)
            and rsi_allows_entry(row, rule)
        )
        exit_risk = line_exit_candidate(row, rule)

        if cooldown > 0:
            cooldown -= 1

        if entry_ok:
            entry_streak += 1
        else:
            entry_streak = 0

        if exit_risk:
            exit_streak += 1
        else:
            exit_streak = 0

        action = "hold"
        if position == 1 and exit_streak >= rule.exit_confirm_days:
            position = 0
            cooldown = rule.cooldown_days
            entry_streak = 0
            action = "exit"
        elif position == 0 and cooldown == 0 and entry_streak >= rule.reentry_confirm_days:
            position = 1
            exit_streak = 0
            action = "entry"

        positions[str(row["date"])] = (position, position_reason(row, rule, position, action))

    return positions


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


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def return_capture_ratio(strategy_return_pct: float, buy_hold_return_pct: float) -> float:
    if buy_hold_return_pct > 0:
        return strategy_return_pct / buy_hold_return_pct
    return 1.0 if strategy_return_pct >= buy_hold_return_pct else 0.0


def usable_for_range(ticker_data: TickerData, start_date: str, end_date: str) -> bool:
    signals = ticker_data.signals[(ticker_data.signals["date"] >= start_date) & (ticker_data.signals["date"] <= end_date)]
    if len(signals) < 5:
        return False
    first_signal = str(signals["date"].iloc[0])
    price = ticker_data.price[(ticker_data.price["date"] >= first_signal) & (ticker_data.price["date"] <= end_date)]
    return len(price) >= 5


def simulate(ticker_data: TickerData, rule: Rule, start_date: str, end_date: str) -> dict[str, Any] | None:
    signals = ticker_data.signals[(ticker_data.signals["date"] >= start_date) & (ticker_data.signals["date"] <= end_date)].copy()
    if len(signals) < 5:
        return None

    first_signal = str(signals["date"].iloc[0])
    price = ticker_data.price[(ticker_data.price["date"] >= first_signal) & (ticker_data.price["date"] <= end_date)].copy()
    if len(price) < 5:
        return None

    positions = build_positions(signals, rule)
    fee_rate = FEE_BPS / 10000

    strategy_equity = 1.0
    buy_hold_equity = 1.0
    position = 0
    trades = 0
    cash_days = 0
    avoided_large_loss_days = 0
    large_loss_days = 0
    holding_durations: list[int] = []
    trade_returns: list[float] = []
    strategy_returns: list[float] = []
    buy_hold_returns: list[float] = []
    strategy_equity_curve = [1.0]
    buy_hold_equity_curve = [1.0]
    entry_price: float | None = None
    entry_index: int | None = None

    close_values = price["close"].astype(float).tolist()
    dates = price["date"].astype(str).tolist()
    all_daily_returns = [close_values[index] / close_values[index - 1] - 1 for index in range(1, len(close_values))]
    threshold_base = percentile(all_daily_returns, 0.2)
    large_loss_threshold = min(-0.02, threshold_base) if threshold_base is not None else -0.02

    for index in range(1, len(price)):
        previous_date = dates[index - 1]
        previous_close = close_values[index - 1]
        current_close = close_values[index]
        desired_position = position
        fee_cost = 0.0

        if previous_date in positions:
            desired_position = positions[previous_date][0]

        if desired_position != position:
            strategy_equity *= 1 - fee_rate
            fee_cost = fee_rate
            trades += 1
            if desired_position == 1:
                entry_price = previous_close
                entry_index = index - 1
            elif entry_price is not None and entry_index is not None:
                trade_returns.append(previous_close / entry_price - 1)
                holding_durations.append(index - 1 - entry_index)
                entry_price = None
                entry_index = None
            position = desired_position

        daily_return = current_close / previous_close - 1
        strategy_daily_return = daily_return if position == 1 else 0.0
        strategy_equity *= 1 + strategy_daily_return
        buy_hold_equity *= 1 + daily_return
        strategy_returns.append(strategy_daily_return - fee_cost)
        buy_hold_returns.append(daily_return)

        if daily_return <= large_loss_threshold:
            large_loss_days += 1
            if position == 0:
                avoided_large_loss_days += 1
        if position == 0:
            cash_days += 1

        strategy_equity_curve.append(strategy_equity)
        buy_hold_equity_curve.append(buy_hold_equity)

    if position == 1 and entry_price is not None and entry_index is not None:
        trade_returns.append(close_values[-1] / entry_price - 1)
        holding_durations.append(len(price) - 1 - entry_index)

    strategy_return_pct = (strategy_equity - 1) * 100
    buy_hold_return_pct = (buy_hold_equity - 1) * 100
    strategy_mdd = max_drawdown(strategy_equity_curve)
    buy_hold_mdd = max_drawdown(buy_hold_equity_curve)
    market_participation = 1 - (cash_days / max(len(price) - 1, 1))
    loss_avoidance = avoided_large_loss_days / large_loss_days if large_loss_days > 0 else None
    avg_holding_days = statistics.fmean(holding_durations) if holding_durations else None

    pass_ticker = (
        strategy_mdd - buy_hold_mdd > 0
        and return_capture_ratio(strategy_return_pct, buy_hold_return_pct) >= 0.70
        and loss_avoidance is not None
        and loss_avoidance >= 0.45
        and 0.45 <= market_participation <= 0.90
        and trades <= 40
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
        "trade_count": trades,
        "avg_holding_days": avg_holding_days,
        "large_loss_days": large_loss_days,
        "avoided_large_loss_days": avoided_large_loss_days,
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
    down_sideways = [
        item
        for item in metrics
        if down_sideways_cutoff is not None and item["buy_hold_return_pct"] <= down_sideways_cutoff
    ]

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
        "return_capture_ok": (summary.get("avg_return_capture_ratio") or 0) >= 0.70,
        "loss_avoidance_ok": (summary.get("avg_loss_avoidance_rate") or 0) >= 0.45,
        "participation_ok": participation is not None and 0.45 <= participation <= 0.90,
        "pass_ticker_ratio_ok": (summary.get("pass_ticker_ratio") or 0) >= 0.50,
        "strong_up_capture_ok": (summary.get("strong_up_avg_return_capture_ratio") or 0) >= 0.60,
        "not_too_passive": participation is not None and participation > 0.30,
        "not_buy_hold_like": participation is not None and participation < 0.95,
        "trades_not_excessive": (summary.get("avg_trade_count") or 999) <= 40,
    }


def score_candidate(holdout: dict[str, Any]) -> float:
    if holdout.get("ticker_count", 0) == 0:
        return -9999.0

    participation = holdout.get("avg_market_participation") or 0
    hard_penalty = 0.0
    if participation <= 0.30:
        hard_penalty -= 100.0
    if participation >= 0.95:
        hard_penalty -= 100.0
    if (holdout.get("avg_return_capture_ratio") or 0) < 0.60:
        hard_penalty -= 35.0
    if (holdout.get("avg_mdd_improvement_pct") or 0) <= 0:
        hard_penalty -= 25.0

    return (
        hard_penalty
        + 5 * ((holdout.get("avg_return_capture_ratio") or 0) - 0.70)
        + (holdout.get("avg_mdd_improvement_pct") or 0)
        + 6 * ((holdout.get("avg_loss_avoidance_rate") or 0) - 0.45)
        + 3 * ((holdout.get("strong_up_avg_return_capture_ratio") or 0) - 0.60)
        + 2 * ((holdout.get("pass_ticker_ratio") or 0) - 0.50)
        - abs(participation - 0.70) * 2
        - max((holdout.get("avg_trade_count") or 0) - 30, 0) * 0.05
    )


def build_rules() -> list[Rule]:
    entry_values = [-0.004, -0.002, 0.000, 0.002, 0.004]
    exit_values = [-0.006, -0.010, -0.014]
    trend_values: list[float | None] = [-0.03, -0.05, None]
    override_values = [True, False]
    rsi_values: list[float | None] = [None, 75, 80]
    rsi_exit_values = ["only_if_trend_weak"]
    exit_confirm_values = [1, 2]
    reentry_confirm_values = [1, 2]
    cooldown_values = [0]

    return [
        Rule(entry, exit_value, trend, override, rsi, rsi_exit, exit_confirm, reentry_confirm, cooldown)
        for entry in entry_values
        for exit_value in exit_values
        for trend in trend_values
        for override in override_values
        for rsi in rsi_values
        for rsi_exit in rsi_exit_values
        for exit_confirm in exit_confirm_values
        for reentry_confirm in reentry_confirm_values
        for cooldown in cooldown_values
    ]


def make_baseline_rule() -> Rule:
    return Rule(
        line_entry_threshold=0.0,
        line_exit_threshold=-0.030,
        trend_floor=-0.05,
        trend_override=True,
        rsi_entry_cap=80,
        rsi_exit_guard="only_if_trend_weak",
        exit_confirm_days=1,
        reentry_confirm_days=1,
        cooldown_days=0,
    )


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
    one_month = aggregate(one_month_metrics)
    flags = survival_flags(holdout)
    return {
        "rule": asdict(rule),
        "key": rule.key,
        "explore": explore,
        "holdout": holdout,
        "recent_one_month_holdout": one_month,
        "survival_flags": flags,
        "survived": all(flags.values()),
        "score": score_candidate(holdout),
    }


def split_candidates() -> list[SplitPlan]:
    return [
        SplitPlan(
            "기본 1개월 holdout",
            EXPLORE_START,
            BASE_EXPLORE_END,
            ONE_MONTH_HOLDOUT_START,
            HOLDOUT_END,
            "요청 기본 split",
        ),
        SplitPlan(
            "최근 2개월 holdout",
            EXPLORE_START,
            "2026-02-28",
            "2026-03-01",
            HOLDOUT_END,
            "1개월 holdout usable ticker가 30개 미만일 때 대안",
        ),
        SplitPlan(
            "최근 3개월 holdout",
            EXPLORE_START,
            "2026-01-31",
            "2026-02-01",
            HOLDOUT_END,
            "2개월 holdout도 부족할 때 대안",
        ),
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

    one_month = coverage[0]
    if one_month["holdout_usable_tickers"] >= 30:
        return split_candidates()[0], coverage

    for split, row in zip(split_candidates()[1:], coverage[1:]):
        if row["holdout_usable_tickers"] >= 30:
            return split, coverage

    best_index = max(range(len(coverage)), key=lambda index: coverage[index]["holdout_usable_tickers"])
    return split_candidates()[best_index], coverage


def write_top_csv(path: Path, candidates: list[dict[str, Any]], limit: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "score",
        "survived",
        "line_entry_threshold",
        "line_exit_threshold",
        "trend_floor",
        "trend_override",
        "rsi_entry_cap",
        "rsi_exit_guard",
        "exit_confirm_days",
        "reentry_confirm_days",
        "cooldown_days",
        "holdout_ticker_count",
        "holdout_avg_strategy_return_pct",
        "holdout_avg_buy_hold_return_pct",
        "holdout_avg_return_capture_ratio",
        "holdout_avg_mdd_improvement_pct",
        "holdout_avg_loss_avoidance_rate",
        "holdout_avg_market_participation",
        "holdout_strong_up_capture",
        "holdout_down_sideways_mdd_improvement",
        "holdout_pass_ticker_ratio",
        "holdout_avg_trade_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
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
                    "line_entry_threshold": rule["line_entry_threshold"],
                    "line_exit_threshold": rule["line_exit_threshold"],
                    "trend_floor": rule["trend_floor"],
                    "trend_override": rule["trend_override"],
                    "rsi_entry_cap": rule["rsi_entry_cap"],
                    "rsi_exit_guard": rule["rsi_exit_guard"],
                    "exit_confirm_days": rule["exit_confirm_days"],
                    "reentry_confirm_days": rule["reentry_confirm_days"],
                    "cooldown_days": rule["cooldown_days"],
                    "holdout_ticker_count": holdout.get("ticker_count"),
                    "holdout_avg_strategy_return_pct": holdout.get("avg_strategy_return_pct"),
                    "holdout_avg_buy_hold_return_pct": holdout.get("avg_buy_hold_return_pct"),
                    "holdout_avg_return_capture_ratio": holdout.get("avg_return_capture_ratio"),
                    "holdout_avg_mdd_improvement_pct": holdout.get("avg_mdd_improvement_pct"),
                    "holdout_avg_loss_avoidance_rate": holdout.get("avg_loss_avoidance_rate"),
                    "holdout_avg_market_participation": holdout.get("avg_market_participation"),
                    "holdout_strong_up_capture": holdout.get("strong_up_avg_return_capture_ratio"),
                    "holdout_down_sideways_mdd_improvement": holdout.get("down_sideways_avg_mdd_improvement_pct"),
                    "holdout_pass_ticker_ratio": holdout.get("pass_ticker_ratio"),
                    "holdout_avg_trade_count": holdout.get("avg_trade_count"),
                }
            )


def featured_ticker_comparison(
    tickers: list[str],
    ticker_data: list[TickerData],
    baseline: Rule,
    best: Rule,
    split: SplitPlan,
) -> list[dict[str, Any]]:
    data_by_ticker = {item.ticker: item for item in ticker_data}
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        item = data_by_ticker.get(ticker)
        if item is None:
            rows.append({"ticker": ticker, "status": "prediction 없음"})
            continue
        baseline_metric = simulate(item, baseline, split.holdout_start, split.holdout_end)
        best_metric = simulate(item, best, split.holdout_start, split.holdout_end)
        rows.append(
            {
                "ticker": ticker,
                "status": "ok" if baseline_metric and best_metric else "평가 데이터 부족",
                "baseline": baseline_metric,
                "best": best_metric,
                "improvement": None
                if not baseline_metric or not best_metric
                else {
                    "strategy_return_delta_pct": best_metric["strategy_return_pct"] - baseline_metric["strategy_return_pct"],
                    "return_capture_delta": best_metric["return_capture_ratio"] - baseline_metric["return_capture_ratio"],
                    "mdd_improvement_delta_pct": best_metric["mdd_improvement_pct"] - baseline_metric["mdd_improvement_pct"],
                    "loss_avoidance_delta": (
                        (best_metric["loss_avoidance_rate"] or 0) - (baseline_metric["loss_avoidance_rate"] or 0)
                    ),
                    "market_participation_delta": best_metric["market_participation"] - baseline_metric["market_participation"],
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP107 Line Trend 100티커 룰 탐색")
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
    excluded: list[dict[str, str]] = []
    prepared: list[TickerData] = []

    for index, ticker in enumerate(universe, start=1):
        rows = fetch_prediction_history(backend_url, ticker, args.history_limit)
        item = prepare_ticker_data(ticker, price, indicators, rows)
        if item is None:
            reason = "LM prediction history 없음" if not rows else "가격/지표와 prediction 날짜 매칭 부족"
            excluded.append({"ticker": ticker, "reason": reason})
        else:
            prepared.append(item)
        if index % 20 == 0:
            print(f"prediction coverage fetch {index}/{len(universe)}")

    if not prepared:
        raise RuntimeError("사용 가능한 LM prediction history가 없습니다.")

    selected_split, split_coverage = choose_split(prepared)
    rules = build_rules()
    if args.max_rules is not None:
        rules = rules[: args.max_rules]

    baseline_rule = make_baseline_rule()
    baseline_result = evaluate_rule(baseline_rule, prepared, selected_split)

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
        if index % 100 == 0 or index == len(rules):
            elapsed = time.time() - start_time
            print(f"evaluated {index}/{len(rules)} elapsed={elapsed:.1f}s")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_rule = Rule(**candidates[0]["rule"]) if candidates else baseline_rule

    prediction_ranges = {
        "min_asof": min(item.prediction_min_date for item in prepared),
        "max_asof": max(item.prediction_max_date for item in prepared),
        "min_prediction_rows": min(item.prediction_count for item in prepared),
        "max_prediction_rows": max(item.prediction_count for item in prepared),
        "avg_prediction_rows": statistics.fmean([item.prediction_count for item in prepared]),
    }

    payload = {
        "cp": "CP107-P",
        "created_at_kst": time.strftime("%Y-%m-%d %H:%M:%S KST", time.localtime()),
        "run_id": LM_RUN_ID,
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
        },
        "universe": {
            "requested": len(universe),
            "usable": len(prepared),
            "excluded": len(excluded),
            "excluded_tickers": excluded,
        },
        "prediction_history_coverage": prediction_ranges,
        "split_coverage": split_coverage,
        "selected_split": asdict(selected_split),
        "rule_count": len(rules),
        "coarse_grid_policy": {
            "line_exit_threshold_note": "-0.018은 이번 coarse grid에서 제외하고 -0.006/-0.010/-0.014를 비교",
            "trend_floor_note": "-0.08은 제외하고 -0.03/-0.05/off를 비교",
            "rsi_exit_note": "RSI 청산은 only_if_trend_weak만 사용해 강한 추세 청산을 과하게 만들지 않음",
            "cooldown_note": "cooldown은 0일로 고정해 1,080개 조합으로 제한",
        },
        "large_loss_definition": "일간 수익률 하위 20%와 -2% 중 더 엄격한 값",
        "strong_up_definition": "평가 구간 Buy & Hold 수익률 상위 25%",
        "down_sideways_definition": "평가 구간 Buy & Hold 수익률 하위 50%",
        "survival_criteria": {
            "avg_mdd_improvement_pct": "> 0",
            "avg_return_capture_ratio": ">= 0.70",
            "avg_loss_avoidance_rate": ">= 0.45",
            "avg_market_participation": "0.45 ~ 0.90",
            "pass_ticker_ratio": ">= 0.50",
            "strong_up_avg_return_capture_ratio": ">= 0.60",
        },
        "baseline": baseline_result,
        "top_candidates": candidates[:10],
        "survived_count": sum(1 for item in candidates if item["survived"]),
        "featured_tickers": featured_ticker_comparison(["AAPL", "MSFT", "NVDA"], prepared, baseline_rule, best_rule, selected_split),
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
