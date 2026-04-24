from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing import build_dataset_plan, fetch_feature_index_frame  # noqa: E402
from ai.splits import FUNDAMENTAL_INSUFFICIENT_TICKERS, MAX_HORIZON_BY_TIMEFRAME  # noqa: E402


def _percentiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "p10": ordered[int((len(ordered) - 1) * 0.1)],
        "p50": ordered[int((len(ordered) - 1) * 0.5)],
        "p90": ordered[int((len(ordered) - 1) * 0.9)],
    }


def _split_stats(counts_by_ticker: dict[str, int]) -> dict[str, object]:
    counts = list(counts_by_ticker.values())
    return {
        "total": sum(counts),
        "ticker_count": len(counts_by_ticker),
        "per_ticker": _percentiles(counts),
    }


def summarize_timeframe(timeframe: str, seq_len: int, horizon: int) -> dict[str, object]:
    feature_df = fetch_feature_index_frame(timeframe=timeframe)
    plan = build_dataset_plan(
        feature_df,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
    )

    reason_counts: dict[str, int] = {}
    for reason in plan.excluded_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    train_counts = {ticker: spec.train.count for ticker, spec in plan.split_specs.items()}
    val_counts = {ticker: spec.val.count for ticker, spec in plan.split_specs.items()}
    test_counts = {ticker: spec.test.count for ticker, spec in plan.split_specs.items()}

    return {
        "timeframe": timeframe,
        "seq_len": seq_len,
        "h_max": MAX_HORIZON_BY_TIMEFRAME[timeframe],
        "input_universe": len(set(feature_df["ticker"]) - FUNDAMENTAL_INSUFFICIENT_TICKERS),
        "eligible": len(plan.eligible_tickers),
        "excluded": len(plan.excluded_reasons),
        "excluded_reasons": reason_counts,
        "excluded_tickers": sorted(plan.excluded_reasons.keys()),
        "train": _split_stats(train_counts),
        "val": _split_stats(val_counts),
        "test": _split_stats(test_counts),
    }


def main() -> None:
    report = {
        "1D": summarize_timeframe("1D", seq_len=252, horizon=5),
        "1W": summarize_timeframe("1W", seq_len=104, horizon=4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
