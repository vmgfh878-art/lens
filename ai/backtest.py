from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference_contract import resolve_execution_market_data_provider
from ai.preprocessing import normalize_ai_timeframe, resolved_market_data_provider
from ai.storage import fetch_run_evaluations, fetch_run_predictions, get_model_run, save_backtest_results
from backend.collector.repositories.base import fetch_frame

import pandas as pd

ANCHOR_MAX_GAP_DAYS = {
    "1D": 0,
    "1W": 7,
    "1M": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저장된 예측 결과로 규칙 기반 백테스트를 실행한다")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--market-data-provider", default=None)
    parser.add_argument("--strategy-name", default="band_breakout_v1")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def _filter_price_frame_by_provider(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "source" not in frame.columns:
        return frame.copy() if provider == "eodhd" else frame.iloc[0:0].copy()
    source = frame["source"].astype("string")
    if provider == "eodhd":
        return frame[source.isna() | (source.str.lower() == "eodhd")].copy()
    return frame[source.str.lower() == provider].copy()


def _max_anchor_gap_days(timeframe: str) -> int:
    return ANCHOR_MAX_GAP_DAYS.get(normalize_ai_timeframe(timeframe), 0)


def _anchor_gap_distribution(gaps: list[int]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for gap in gaps:
        key = str(int(gap))
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _resolve_anchor_rows(frame: pd.DataFrame, price_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    price = price_frame.copy()
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["ticker", "date"]).reset_index(drop=True)

    exact_lookup = {
        (str(row["ticker"]).upper(), row["date"].strftime("%Y-%m-%d")): (float(row["anchor_close"]), row["date"])
        for _, row in price.iterrows()
    }
    grouped_prices = {
        ticker: group.reset_index(drop=True)
        for ticker, group in price.groupby("ticker", sort=False)
    }

    resolved_closes: list[float | None] = []
    resolved_dates: list[str | None] = []
    resolved_gaps: list[int | None] = []
    resolved_modes: list[str] = []
    exact_count = 0
    resolved_count = 0
    missing_count = 0
    observed_gaps: list[int] = []

    for _, row in frame.iterrows():
        ticker = str(row["ticker"]).upper()
        asof_label = str(row["asof_date"])
        asof_ts = pd.Timestamp(asof_label)
        exact = exact_lookup.get((ticker, asof_label))
        if exact is not None:
            anchor_close, anchor_date = exact
            resolved_closes.append(anchor_close)
            resolved_dates.append(anchor_date.strftime("%Y-%m-%d"))
            resolved_gaps.append(0)
            resolved_modes.append("exact")
            exact_count += 1
            observed_gaps.append(0)
            continue

        timeframe = normalize_ai_timeframe(str(row["timeframe"]))
        max_gap = _max_anchor_gap_days(timeframe)
        if timeframe not in {"1W", "1M"}:
            resolved_closes.append(None)
            resolved_dates.append(None)
            resolved_gaps.append(None)
            resolved_modes.append("missing")
            missing_count += 1
            continue

        ticker_prices = grouped_prices.get(ticker)
        if ticker_prices is None or ticker_prices.empty:
            resolved_closes.append(None)
            resolved_dates.append(None)
            resolved_gaps.append(None)
            resolved_modes.append("missing")
            missing_count += 1
            continue

        candidates = ticker_prices[ticker_prices["date"] <= asof_ts]
        if candidates.empty:
            resolved_closes.append(None)
            resolved_dates.append(None)
            resolved_gaps.append(None)
            resolved_modes.append("missing")
            missing_count += 1
            continue

        candidate = candidates.iloc[-1]
        gap_days = int((asof_ts - candidate["date"]).days)
        if gap_days > max_gap:
            resolved_closes.append(None)
            resolved_dates.append(candidate["date"].strftime("%Y-%m-%d"))
            resolved_gaps.append(gap_days)
            resolved_modes.append("gap_exceeded")
            missing_count += 1
            continue

        resolved_closes.append(float(candidate["anchor_close"]))
        resolved_dates.append(candidate["date"].strftime("%Y-%m-%d"))
        resolved_gaps.append(gap_days)
        resolved_modes.append("resolved_prior_trading_date")
        resolved_count += 1
        observed_gaps.append(gap_days)

    resolved = frame.copy()
    resolved["anchor_close"] = resolved_closes
    resolved["anchor_resolved_date"] = resolved_dates
    resolved["anchor_gap_days"] = resolved_gaps
    resolved["anchor_resolution"] = resolved_modes
    metrics = {
        "anchor_exact_count": exact_count,
        "anchor_resolved_count": resolved_count,
        "anchor_missing_count": missing_count,
        "anchor_gap_days_distribution": _anchor_gap_distribution(observed_gaps),
    }
    return resolved, metrics


def build_backtest_frame(
    run_id: str,
    timeframe: str | None = None,
    *,
    market_data_provider: str | None = None,
) -> pd.DataFrame:
    provider = resolved_market_data_provider(market_data_provider)
    predictions = fetch_run_predictions(run_id, timeframe)
    evaluations = fetch_run_evaluations(run_id, timeframe)
    if predictions.empty or evaluations.empty:
        raise ValueError("백테스트 대상 예측 또는 평가 데이터가 없습니다. 먼저 추론 저장을 실행하세요.")

    frame = predictions.merge(
        evaluations,
        on=["run_id", "ticker", "timeframe", "asof_date"],
        suffixes=("_pred", "_eval"),
    )
    frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.strftime("%Y-%m-%d")

    tickers = sorted(frame["ticker"].astype(str).str.upper().unique().tolist())
    min_date = frame["asof_date"].min()
    max_date = frame["asof_date"].max()
    timeframes = [normalize_ai_timeframe(str(value)) for value in frame["timeframe"].dropna().unique().tolist()]
    max_anchor_gap = max((_max_anchor_gap_days(value) for value in timeframes), default=0)
    price_start_date = (pd.Timestamp(min_date) - pd.Timedelta(days=max_anchor_gap)).strftime("%Y-%m-%d")
    price_frame = fetch_frame(
        "price_data",
        columns="ticker,date,close,adjusted_close,source,provider",
        filters=[("in", "ticker", tickers), ("gte", "date", price_start_date), ("lte", "date", max_date)],
        order_by="date",
    )
    price_frame = _filter_price_frame_by_provider(price_frame, provider)
    if price_frame.empty:
        raise ValueError("anchor close를 찾을 수 없습니다. price_data를 확인하세요.")

    if "adjusted_close" not in price_frame.columns:
        price_frame["adjusted_close"] = price_frame["close"]
    price_frame["anchor_close"] = price_frame["adjusted_close"].fillna(price_frame["close"])
    frame, anchor_metrics = _resolve_anchor_rows(frame, price_frame)
    frame = frame.dropna(subset=["anchor_close"]).copy()
    frame["line_last"] = frame["line_series"].apply(lambda values: float(values[-1]) if values else None)
    frame["actual_last"] = frame["actual_series"].apply(lambda values: float(values[-1]) if values else None)
    frame = frame.dropna(subset=["line_last", "actual_last"]).copy()

    frame["line_return"] = (frame["line_last"] / frame["anchor_close"]) - 1.0
    frame["realized_return"] = (frame["actual_last"] / frame["anchor_close"]) - 1.0
    result = frame.sort_values("asof_date").reset_index(drop=True)
    result.attrs["anchor_resolution_metrics"] = anchor_metrics
    return result


def run_rule_based_backtest(
    prediction_frame: pd.DataFrame,
    *,
    strategy_name: str = "band_breakout_v1",
    fee_bps: float = 10.0,
) -> dict[str, Any]:
    """규칙 기반 시그널의 방향성과 손익 특성을 빠르게 비교한다."""
    if prediction_frame.empty:
        raise ValueError("백테스트 입력 데이터가 비어 있습니다.")

    frame = prediction_frame.sort_values(["asof_date", "ticker"]).copy()
    anchor_resolution_metrics = prediction_frame.attrs.get("anchor_resolution_metrics")
    signal_to_position = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
    fee_rate = float(fee_bps) / 10000.0

    date_returns: list[float] = []
    gross_returns: list[float] = []
    turnover_values: list[float] = []
    trade_counts: list[int] = []
    previous_weights: dict[str, float] = {}

    for asof_date, group in frame.groupby("asof_date", sort=True):
        del asof_date
        raw_positions = group.set_index("ticker")["signal"].map(signal_to_position).fillna(0.0).astype(float)
        exposure = float(raw_positions.abs().sum())
        weights = (
            {str(ticker): float(position / exposure) for ticker, position in raw_positions.items() if position != 0.0}
            if exposure > 0.0
            else {}
        )
        realized_returns = group.set_index("ticker")["realized_return"].astype(float).to_dict()
        gross_return = sum(weights.get(str(ticker), 0.0) * float(return_value) for ticker, return_value in realized_returns.items())
        all_tickers = set(weights) | set(previous_weights)
        turnover = sum(abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in all_tickers)

        gross_returns.append(gross_return)
        turnover_values.append(turnover)
        trade_counts.append(sum(1 for ticker in all_tickers if abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) > 0.0))
        date_returns.append(gross_return - (turnover * fee_rate))
        previous_weights = weights

    strategy_return = pd.Series(date_returns, dtype="float64")
    gross_strategy_return = pd.Series(gross_returns, dtype="float64")
    turnover_series = pd.Series(turnover_values, dtype="float64")
    cumulative = (1.0 + strategy_return).cumprod()
    gross_cumulative = (1.0 + gross_strategy_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0

    wins = strategy_return[strategy_return > 0]
    losses = strategy_return[strategy_return < 0].abs()
    profit_factor = float(wins.sum() / losses.sum()) if losses.sum() > 0 else None
    strategy_std = float(strategy_return.std(ddof=0))
    gross_std = float(gross_strategy_return.std(ddof=0))
    fee_adjusted_return_pct = float((cumulative.iloc[-1] - 1.0) * 100.0)
    gross_return_pct = float((gross_cumulative.iloc[-1] - 1.0) * 100.0)
    fee_adjusted_sharpe = float(strategy_return.mean() / strategy_std) if strategy_std else 0.0

    return {
        "strategy_name": strategy_name,
        "return_pct": fee_adjusted_return_pct,
        "mdd": float(drawdown.min()),
        "sharpe": fee_adjusted_sharpe,
        "win_rate": float((strategy_return > 0).mean()),
        "profit_factor": profit_factor,
        "num_trades": int(sum(trade_counts)),
        "fee_adjusted_return_pct": fee_adjusted_return_pct,
        "fee_adjusted_sharpe": fee_adjusted_sharpe,
        "avg_turnover": float(turnover_series.mean()) if not turnover_series.empty else 0.0,
        "meta": {
            "rows": len(frame),
            "portfolio_dates": int(len(strategy_return)),
            "position_contract": "date_equal_abs_exposure",
            "avg_realized_return": float(frame["realized_return"].mean()),
            "avg_line_return": float(frame["line_return"].mean()),
            "fee_bps": float(fee_bps),
            "gross_return_pct": gross_return_pct,
            "gross_sharpe": float(gross_strategy_return.mean() / gross_std) if gross_std else 0.0,
            "anchor_resolution_metrics": anchor_resolution_metrics,
        },
    }


def persist_backtest_result(run_id: str, timeframe: str, result: dict[str, Any]) -> None:
    save_backtest_results(
        [
            {
                "run_id": run_id,
                "strategy_name": result["strategy_name"],
                "timeframe": timeframe,
                "return_pct": result["return_pct"],
                "mdd": result["mdd"],
                "sharpe": result["sharpe"],
                "win_rate": result["win_rate"],
                "profit_factor": result["profit_factor"],
                "num_trades": result["num_trades"],
                "meta": result["meta"],
            }
        ]
    )


def run_backtest(
    *,
    run_id: str,
    timeframe: str | None = None,
    strategy_name: str,
    fee_bps: float = 10.0,
    save: bool = False,
    market_data_provider: str | None = None,
) -> dict[str, Any]:
    if timeframe is not None:
        normalize_ai_timeframe(timeframe)
    # CP12: NaN으로 실패한 run의 backtest 결과 저장을 차단한다.
    model_run = get_model_run(run_id)
    if model_run is None:
        raise ValueError(f"run_id={run_id}에 해당하는 model_runs 기록이 없습니다.")
    run_status = str(model_run.get("status") or "completed")
    if run_status != "completed":
        raise ValueError(
            f"run_id={run_id} status={run_status}: completed 상태의 run에서만 backtest를 실행할 수 있습니다."
        )
    provider, provider_contract = resolve_execution_market_data_provider(
        model_run.get("config") or {},
        requested_provider=market_data_provider,
        run_config=model_run,
    )
    frame = build_backtest_frame(run_id, timeframe, market_data_provider=provider)
    result = run_rule_based_backtest(frame, strategy_name=strategy_name, fee_bps=fee_bps)
    resolved_timeframe = timeframe or str(frame["timeframe"].iloc[0])
    normalize_ai_timeframe(resolved_timeframe)
    if save:
        persist_backtest_result(run_id, resolved_timeframe, result)
    return {
        "run_id": run_id,
        "timeframe": resolved_timeframe,
        "market_data_provider": provider,
        "contract_metrics": provider_contract,
        "anchor_resolution_metrics": result["meta"].get("anchor_resolution_metrics"),
        **result,
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_backtest(
        run_id=args.run_id,
        timeframe=args.timeframe,
        strategy_name=args.strategy_name,
        fee_bps=args.fee_bps,
        save=args.save,
        market_data_provider=args.market_data_provider,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
