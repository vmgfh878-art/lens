import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.jobs.compute_market_breadth import _calculate_stats
from backend.collector.jobs.compute_indicators import run as run_compute_indicators
from backend.collector.jobs.sync_macro import resolve_macro_start_date
from backend.collector.jobs.sync_prices import _validate_price_frame
from backend.collector.readiness import summarize_indicator_counts


class CollectorJobTestCase(unittest.TestCase):
    def test_validate_price_frame_rejects_invalid_ohlc_and_zero_volume(self):
        frame = pd.DataFrame(
            [
                {"Open": 10, "High": 12, "Low": 9, "Close": 11, "Adj Close": 11, "Volume": 100},
                {"Open": 13, "High": 12, "Low": 11, "Close": 11.5, "Adj Close": 11.5, "Volume": 100},
                {"Open": 11, "High": 11.5, "Low": 10.5, "Close": 11.2, "Adj Close": 11.2, "Volume": 0},
            ],
            index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
        )

        validated, summary = _validate_price_frame("AAPL", frame)

        self.assertEqual(len(validated), 1)
        self.assertEqual(summary["invalid_ohlc"], 1)
        self.assertEqual(summary["invalid_volume"], 1)

    def test_validate_price_frame_allows_split_like_adjusted_move(self):
        frame = pd.DataFrame(
            [
                {"Open": 100, "High": 102, "Low": 99, "Close": 100, "Adj Close": 100, "Volume": 100},
                {"Open": 52, "High": 53, "Low": 50, "Close": 51, "Adj Close": 102, "Volume": 100},
            ],
            index=pd.to_datetime(["2026-04-01", "2026-04-02"]),
        )

        validated, summary = _validate_price_frame("AAPL", frame)

        self.assertEqual(len(validated), 2)
        self.assertEqual(summary["extreme_jump"], 0)

    def test_calculate_stats_returns_empty_when_not_enough_history(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=10, freq="D"),
                "ticker": ["AAPL"] * 10,
                "close": [100 + idx for idx in range(10)],
            }
        )

        stats = _calculate_stats(frame, min_ticker_count=1)
        self.assertTrue(stats.empty)

    def test_resolve_macro_start_date_supports_repair_mode(self):
        start_date = resolve_macro_start_date(
            730,
            pd.Timestamp("2026-04-17").date(),
            repair_mode=True,
            full_start_date="2015-01-01",
        )

        self.assertEqual(start_date, "2015-01-01")

    def test_summarize_indicator_counts_marks_pending_tickers(self):
        frame = pd.DataFrame(
            [
                {"ticker": "AAPL", "timeframe": "1D", "date": "2026-01-01"},
                {"ticker": "AAPL", "timeframe": "1W", "date": "2026-01-03"},
                {"ticker": "AAPL", "timeframe": "1M", "date": "2026-01-31"},
            ]
        )

        summary = summarize_indicator_counts(
            frame,
            ["AAPL", "MSFT"],
            minimum_rows_by_timeframe={"1D": 1, "1W": 1, "1M": 1},
        )

        self.assertEqual(summary["ready_by_timeframe"]["1D"], 1)
        self.assertIn("MSFT", summary["pending_tickers"])

    def test_summarize_indicator_counts_uses_realistic_monthly_threshold(self):
        frame = pd.DataFrame(
            [{"ticker": "AAPL", "timeframe": "1M", "date": f"2024-{month:02d}-28"} for month in range(1, 13)]
            + [{"ticker": "AAPL", "timeframe": "1M", "date": f"2025-{month:02d}-28"} for month in range(1, 13)]
        )

        summary = summarize_indicator_counts(frame, ["AAPL"], minimum_rows_by_timeframe={"1M": 24})

        self.assertEqual(summary["ready_by_timeframe"]["1M"], 1)
        self.assertEqual(summary["pending_tickers"], [])

    def test_compute_indicators_loads_fundamentals_without_date_cutoff(self):
        price_frame = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2026-04-24"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "adjusted_close": [100.0],
                "volume": [1000],
                "amount": [100000.0],
                "per": [10.0],
                "pbr": [2.0],
            }
        )
        fundamentals_frame = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2025-12-31"],
                "filing_date": ["2026-01-31"],
                "revenue": [100.0],
                "net_income": [10.0],
                "total_liabilities": [40.0],
                "equity": [50.0],
                "eps": [1.0],
            }
        )

        def fake_fetch_frame(table, **kwargs):
            if table == "price_data":
                return price_frame.copy()
            if table == "company_fundamentals":
                return fundamentals_frame.copy()
            return pd.DataFrame()

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame) as fetch_mock, patch(
            "backend.collector.jobs.compute_indicators.build_features",
            return_value=pd.DataFrame(),
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(lookback_days=14, tickers=["AAPL"])

        fundamentals_calls = [
            call.kwargs
            for call in fetch_mock.call_args_list
            if call.args and call.args[0] == "company_fundamentals"
        ]

        self.assertEqual(len(fundamentals_calls), 3)
        for kwargs in fundamentals_calls:
            filters = kwargs.get("filters") or []
            self.assertFalse(any(filter_item[0] == "gte" and filter_item[1] == "date" for filter_item in filters))


if __name__ == "__main__":
    unittest.main()
