from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BM_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_DATA_DIR = Path("data/parquet")
DEFAULT_OUTPUT_JSON = Path("docs/cp106_band_risk_strategy_grid_metrics.json")
DEFAULT_OUTPUT_CSV = Path("docs/cp106_band_risk_strategy_grid_top_candidates.csv")

EXPLORE_START = "2025-06-18"
EXPLORE_END = "2026-03-31"
HOLDOUT_START = "2026-04-01"
HOLDOUT_END = "2026-05-01"
FEE_BPS = 10


@dataclass(frozen=True)
class Rule:
    lower_risk_threshold: float
    width_risk_threshold: float
    width_expansion_ratio: float
    risk_confirm_days: int
    reentry_confirm_days: int
    trend_filter: str
    rsi_filter: str

    @property
    def key(self) -> str:
        return (
            f"lower={self.lower_risk_threshold}|width={self.width_risk_threshold}|"
            f"exp={self.width_expansion_ratio}|risk={self.risk_confirm_days}|"
            f"reentry={self.reentry_confirm_days}|trend={self.trend_filter}|rsi={self.rsi_filter}"
        )


@dataclass
class TickerData:
    ticker: str
    price: pd.DataFrame
    signals: pd.DataFrame


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
    params = urllib.parse.urlencode({"run_id": BM_RUN_ID, "limit": limit})
    url = f"{backend_url}/api/v1/stocks/{urllib.parse.quote(ticker)}/predictions/history?{params}"
    try:
        payload = read_json_url(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    return list(payload.get("data") or [])


def load_universe(data_dir: Path, limit: int) -> list[str]:
    stock_info_path = data_dir / "stock_info.parquet"
    stock_info = pd.read_parquet(stock_info_path)
    tickers = stock_info["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    return tickers[:limit]


def load_market_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(data_dir / "price_data_yfinance.parquet")
    indicators = pd.read_parquet(data_dir / "indicators_yfinance_1D.parquet")

    price = price[["ticker", "date", "close"]].copy()
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = price["date"].astype(str)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])

    indicator_columns = ["ticker", "date", "rsi", "ma_60_ratio", "atr_ratio"]
    indicators = indicators[[column for column in indicator_columns if column in indicators.columns]].copy()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = indicators["date"].astype(str)
    for column in ["rsi", "ma_60_ratio", "atr_ratio"]:
        if column in indicators.columns:
            indicators[column] = pd.to_numeric(indicators[column], errors="coerce")
    indicators = indicators.sort_values(["ticker", "date"])
    return price, indicators


def worst_lower(row: dict[str, Any]) -> float | None:
    values = [float(value) for value in row.get("lower_band_series") or [] if finite(value)]
    return min(values) if values else None


def highest_upper(row: dict[str, Any]) -> float | None:
    values = [float(value) for value in row.get("upper_band_series") or [] if finite(value)]
    return max(values) if values else None


def prepare_ticker_data(
    ticker: str,
    price_rows: pd.DataFrame,
    indicator_rows: pd.DataFrame,
    prediction_rows: list[dict[str, Any]],
) -> TickerData | None:
    if price_rows.empty or not prediction_rows:
        return None

    price = price_rows[price_rows["ticker"] == ticker].copy()
    if price.empty:
        return None
    price = price.sort_values("date").drop_duplicates("date", keep="last")
    price["ma60_calc"] = price["close"] / price["close"].rolling(60, min_periods=60).mean() - 1

    indicators = indicator_rows[indicator_rows["ticker"] == ticker].copy()
    if indicators.empty:
        indicators = pd.DataFrame(columns=["ticker", "date", "rsi", "ma_60_ratio", "atr_ratio"])
    indicators = indicators.sort_values("date").drop_duplicates("date", keep="last")

    rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        asof_date = str(prediction.get("asof_date") or "")
        lower = worst_lower(prediction)
        upper = highest_upper(prediction)
        if not asof_date or lower is None or upper is None:
            continue
        rows.append({"date": asof_date, "lower_band": lower, "upper_band": upper})

    if not rows:
        return None

    pred = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    signals = pred.merge(price[["date", "close", "ma60_calc"]], on="date", how="inner")
    if indicators.empty:
        signals["rsi"] = math.nan
        signals["ma_60_ratio"] = math.nan
        signals["atr_ratio"] = math.nan
    else:
        signals = signals.merge(indicators[["date", "rsi", "ma_60_ratio", "atr_ratio"]], on="date", how="left")
    signals["ma60_ratio"] = signals["ma_60_ratio"].where(signals["ma_60_ratio"].notna(), signals["ma60_calc"])
    signals["rsi"] = signals["rsi"].map(normalize_rsi)
    signals["lower_return"] = signals["lower_band"] / signals["close"] - 1
    signals["width_return"] = (signals["upper_band"] - signals["lower_band"]) / signals["close"]
    signals["width_ref"] = signals["width_return"].shift(1).rolling(20, min_periods=5).median()
    signals = signals.dropna(subset=["date", "close", "lower_return", "width_return"]).sort_values("date")

    if len(signals) < 5:
        return None

    return TickerData(ticker=ticker, price=price[["date", "close"]].copy(), signals=signals)


def trend_allows_entry(rule: Rule, row: pd.Series) -> bool:
    if rule.trend_filter == "off":
        return True
    ma60_ratio = row.get("ma60_ratio")
    if not finite(ma60_ratio):
        return False
    if rule.trend_filter == "price_above_ma60":
        return float(ma60_ratio) >= 0
    if rule.trend_filter == "ma60_ratio_above_-3pct":
        return float(ma60_ratio) >= -0.03
    return True


def rsi_allows_entry(rule: Rule, row: pd.Series) -> bool:
    if rule.rsi_filter == "off":
        return True
    rsi = row.get("rsi")
    if not finite(rsi):
        return False
    threshold = 75 if rule.rsi_filter == "avoid_entry_if_rsi_above_75" else 80
    return float(rsi) <= threshold


def risk_reason(rule: Rule, row: pd.Series) -> str:
    reasons: list[str] = []
    if float(row["lower_return"]) <= rule.lower_risk_threshold:
        reasons.append("밴드 하단 위험")
    if float(row["width_return"]) >= rule.width_risk_threshold:
        reasons.append("밴드 폭 확대")
    width_ref = row.get("width_ref")
    if finite(width_ref) and float(width_ref) > 0 and float(row["width_return"]) >= float(width_ref) * rule.width_expansion_ratio:
        reasons.append("밴드 폭 최근 대비 확대")
    return ", ".join(reasons) if reasons else "위험 완화"


def build_positions(signals: pd.DataFrame, rule: Rule) -> dict[str, tuple[int, str]]:
    position = 0
    risk_streak = 0
    safe_streak = 0
    positions: dict[str, tuple[int, str]] = {}

    for _, row in signals.iterrows():
        lower_risk = float(row["lower_return"]) <= rule.lower_risk_threshold
        width_risk = float(row["width_return"]) >= rule.width_risk_threshold
        width_ref = row.get("width_ref")
        expansion_risk = finite(width_ref) and float(width_ref) > 0 and float(row["width_return"]) >= float(width_ref) * rule.width_expansion_ratio
        risk = lower_risk or width_risk or expansion_risk
        entry_allowed = trend_allows_entry(rule, row) and rsi_allows_entry(rule, row)

        if risk:
            risk_streak += 1
            safe_streak = 0
        elif entry_allowed:
            safe_streak += 1
            risk_streak = 0
        else:
            safe_streak = 0
            risk_streak = 0

        reason = "이전 상태 유지"
        if position == 1 and risk_streak >= rule.risk_confirm_days:
            position = 0
            reason = risk_reason(rule, row)
        elif position == 0 and safe_streak >= rule.reentry_confirm_days:
            position = 1
            reason = "밴드 위험 완화"
        elif position == 0 and not entry_allowed:
            reason = "재진입 필터 대기"
        elif position == 0 and risk:
            reason = risk_reason(rule, row)
        elif position == 1:
            reason = "밴드 위험 허용 범위"

        positions[str(row["date"])] = (position, reason)

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
    mean = statistics.fmean(returns)
    std = statistics.stdev(returns)
    return (mean / std) * math.sqrt(252) if std > 0 else 0.0


def sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.fmean(returns)
    downside = [value for value in returns if value < 0]
    if len(downside) < 2:
        return mean * math.sqrt(252) if mean > 0 else 0.0
    downside_dev = math.sqrt(sum(value * value for value in downside) / (len(downside) - 1))
    return (mean / downside_dev) * math.sqrt(252) if downside_dev > 0 else 0.0


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * ratio)))
    return ordered[index]


def return_capture_ratio(strategy_return_pct: float, buy_hold_return_pct: float) -> float:
    if buy_hold_return_pct > 0:
        return strategy_return_pct / buy_hold_return_pct
    return 1.0 if strategy_return_pct >= buy_hold_return_pct else 0.0


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
    trade_returns: list[float] = []
    holding_durations: list[int] = []
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
        and loss_avoidance is not None
        and loss_avoidance >= 0.50
        and return_capture_ratio(strategy_return_pct, buy_hold_return_pct) >= 0.70
        and 0.45 <= market_participation <= 0.85
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

    strong_up = [item for item in metrics if item["buy_hold_return_pct"] >= 5]
    down_sideways = [item for item in metrics if item["buy_hold_return_pct"] <= 2]

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
        "strong_up_ticker_count": len(strong_up),
        "strong_up_avg_return_capture_ratio": mean([item["return_capture_ratio"] for item in strong_up]),
        "down_sideways_ticker_count": len(down_sideways),
        "down_sideways_avg_mdd_improvement_pct": mean([item["mdd_improvement_pct"] for item in down_sideways]),
    }


def survival_flags(summary: dict[str, Any]) -> dict[str, bool]:
    participation = summary.get("avg_market_participation")
    return {
        "mdd_improved": (summary.get("avg_mdd_improvement_pct") or -999) > 0,
        "loss_avoidance_ok": (summary.get("avg_loss_avoidance_rate") or 0) >= 0.50,
        "return_capture_ok": (summary.get("avg_return_capture_ratio") or 0) >= 0.70,
        "participation_ok": participation is not None and 0.45 <= participation <= 0.85,
        "pass_ticker_ratio_ok": (summary.get("pass_ticker_ratio") or 0) >= 0.50,
        "not_buy_hold_like": participation is not None and participation < 0.95,
        "not_too_passive": participation is not None and participation > 0.30,
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
    if (holdout.get("avg_mdd_improvement_pct") or 0) <= 0:
        hard_penalty -= 25.0
    if (holdout.get("avg_loss_avoidance_rate") or 0) <= 0.30:
        hard_penalty -= 25.0
    return (
        hard_penalty
        +
        (holdout.get("avg_mdd_improvement_pct") or 0)
        + 8 * ((holdout.get("avg_loss_avoidance_rate") or 0) - 0.5)
        + 4 * ((holdout.get("avg_return_capture_ratio") or 0) - 0.7)
        + 2 * ((holdout.get("pass_ticker_ratio") or 0) - 0.5)
        - abs(participation - 0.65) * 3
        - max((holdout.get("avg_trade_count") or 0) - 25, 0) * 0.05
    )


def build_rules() -> list[Rule]:
    lower_values = [-0.03, -0.05, -0.08, -0.10]
    width_values = [0.04, 0.06, 0.08, 0.10]
    expansion_values = [1.10, 1.25, 1.50, 1.75]
    confirm_values = [1, 2, 3]
    trend_values = ["off", "price_above_ma60", "ma60_ratio_above_-3pct"]
    rsi_values = ["off", "avoid_entry_if_rsi_above_75", "avoid_entry_if_rsi_above_80"]
    return [
        Rule(lower, width, expansion, risk_days, reentry_days, trend, rsi)
        for lower in lower_values
        for width in width_values
        for expansion in expansion_values
        for risk_days in confirm_values
        for reentry_days in confirm_values
        for trend in trend_values
        for rsi in rsi_values
    ]


def evaluate_rule(rule: Rule, ticker_data: list[TickerData]) -> dict[str, Any]:
    explore_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, rule, EXPLORE_START, EXPLORE_END)) is not None
    ]
    holdout_metrics = [
        metric
        for item in ticker_data
        if (metric := simulate(item, rule, HOLDOUT_START, HOLDOUT_END)) is not None
    ]
    explore = aggregate(explore_metrics)
    holdout = aggregate(holdout_metrics)
    flags = survival_flags(holdout)
    return {
        "rule": asdict(rule),
        "key": rule.key,
        "explore": explore,
        "holdout": holdout,
        "survival_flags": flags,
        "survived": all(flags.values()),
        "score": score_candidate(holdout),
    }


def make_legacy_baseline_rule() -> Rule:
    return Rule(
        lower_risk_threshold=-0.10,
        width_risk_threshold=0.25,
        width_expansion_ratio=999.0,
        risk_confirm_days=1,
        reentry_confirm_days=1,
        trend_filter="off",
        rsi_filter="off",
    )


def write_top_csv(path: Path, candidates: list[dict[str, Any]], limit: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "score",
        "survived",
        "lower_risk_threshold",
        "width_risk_threshold",
        "width_expansion_ratio",
        "risk_confirm_days",
        "reentry_confirm_days",
        "trend_filter",
        "rsi_filter",
        "holdout_ticker_count",
        "holdout_avg_strategy_return_pct",
        "holdout_avg_buy_hold_return_pct",
        "holdout_avg_return_capture_ratio",
        "holdout_avg_mdd_improvement_pct",
        "holdout_avg_loss_avoidance_rate",
        "holdout_avg_market_participation",
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
                    "lower_risk_threshold": rule["lower_risk_threshold"],
                    "width_risk_threshold": rule["width_risk_threshold"],
                    "width_expansion_ratio": rule["width_expansion_ratio"],
                    "risk_confirm_days": rule["risk_confirm_days"],
                    "reentry_confirm_days": rule["reentry_confirm_days"],
                    "trend_filter": rule["trend_filter"],
                    "rsi_filter": rule["rsi_filter"],
                    "holdout_ticker_count": holdout.get("ticker_count"),
                    "holdout_avg_strategy_return_pct": holdout.get("avg_strategy_return_pct"),
                    "holdout_avg_buy_hold_return_pct": holdout.get("avg_buy_hold_return_pct"),
                    "holdout_avg_return_capture_ratio": holdout.get("avg_return_capture_ratio"),
                    "holdout_avg_mdd_improvement_pct": holdout.get("avg_mdd_improvement_pct"),
                    "holdout_avg_loss_avoidance_rate": holdout.get("avg_loss_avoidance_rate"),
                    "holdout_avg_market_participation": holdout.get("avg_market_participation"),
                    "holdout_pass_ticker_ratio": holdout.get("pass_ticker_ratio"),
                    "holdout_avg_trade_count": holdout.get("avg_trade_count"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP106 Band Risk 100티커 룰 탐색")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--universe-limit", type=int, default=100)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--max-rules", type=int, default=None, help="디버그용 룰 후보 수 제한")
    return parser.parse_args()


def main() -> int:
    start_time = time.time()
    args = parse_args()
    backend_url = normalize_backend_url(args.backend_url)
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    print("CP106 Band Risk grid 시작")
    print(f"backend_url={backend_url}")
    print(f"data_dir={data_dir}")
    print("CPU only: GPU/학습/inference/DB write 없음")

    universe = load_universe(data_dir, args.universe_limit)
    price, indicators = load_market_data(data_dir)

    prepared: list[TickerData] = []
    exclusions: list[dict[str, str]] = []
    prediction_min_dates: list[str] = []
    prediction_max_dates: list[str] = []

    for index, ticker in enumerate(universe, start=1):
        try:
            predictions = fetch_prediction_history(backend_url, ticker, limit=args.history_limit)
        except Exception as exc:  # noqa: BLE001
            exclusions.append({"ticker": ticker, "reason": f"prediction 조회 실패: {exc}"})
            continue

        if not predictions:
            exclusions.append({"ticker": ticker, "reason": "BM prediction history 없음"})
            continue

        ticker_data = prepare_ticker_data(ticker, price, indicators, predictions)
        if ticker_data is None:
            exclusions.append({"ticker": ticker, "reason": "가격/지표/prediction 날짜 매칭 부족"})
            continue

        prepared.append(ticker_data)
        prediction_min_dates.append(str(ticker_data.signals["date"].min()))
        prediction_max_dates.append(str(ticker_data.signals["date"].max()))
        if index % 10 == 0:
            print(f"데이터 준비 {index}/{len(universe)} tickers, usable={len(prepared)}")

    rules = build_rules()
    if args.max_rules is not None:
        rules = rules[: args.max_rules]

    print(f"usable_tickers={len(prepared)}, excluded={len(exclusions)}, rules={len(rules)}")
    results: list[dict[str, Any]] = []
    for index, rule in enumerate(rules, start=1):
        results.append(evaluate_rule(rule, prepared))
        if index % 250 == 0 or index == len(rules):
            print(f"룰 평가 {index}/{len(rules)}")

    top_candidates = sorted(results, key=lambda item: item["score"], reverse=True)
    baseline_rule = make_legacy_baseline_rule()
    baseline = evaluate_rule(baseline_rule, prepared)
    survived_count = sum(1 for item in results if item["survived"])

    payload = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "bm_run_id": BM_RUN_ID,
            "backend_url": backend_url,
            "data_dir": str(data_dir),
            "universe_count": len(universe),
            "usable_ticker_count": len(prepared),
            "excluded_ticker_count": len(exclusions),
            "rule_count": len(rules),
            "survived_candidate_count": survived_count,
            "explore_period": {"start": EXPLORE_START, "end": EXPLORE_END},
            "holdout_period": {"start": HOLDOUT_START, "end": HOLDOUT_END},
            "actual_prediction_range": {
                "min": min(prediction_min_dates) if prediction_min_dates else None,
                "max": max(prediction_max_dates) if prediction_max_dates else None,
            },
            "large_loss_definition": "일간 수익률 하위 20%와 -2% 중 더 엄격한 값",
            "cpu_only": True,
            "db_write": False,
            "model_training": False,
            "inference": False,
            "supabase_price_indicator_bulk_read": False,
            "elapsed_seconds": round(time.time() - start_time, 3),
        },
        "excluded_tickers": exclusions,
        "baseline_band_risk_v1_like": baseline,
        "top_candidates": top_candidates[:20],
        "all_candidates": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_top_csv(output_csv, top_candidates, limit=10)

    best = top_candidates[0] if top_candidates else None
    print(f"metrics_json={output_json}")
    print(f"top_csv={output_csv}")
    if best:
        print(f"best_score={best['score']:.4f}, survived={best['survived']}, key={best['key']}")
    print(f"elapsed_seconds={time.time() - start_time:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
