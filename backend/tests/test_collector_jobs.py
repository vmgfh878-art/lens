import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.jobs.compute_market_breadth import _calculate_stats
from backend.collector.jobs.compute_indicators import run as run_compute_indicators
from backend.collector.jobs.sync_macro import resolve_macro_start_date
from backend.collector.jobs.sync_prices import _validate_price_frame
from backend.collector.pipelines.daily_market_sync import _build_price_coverage
from backend.collector.readiness import summarize_indicator_counts
from backend.collector.sources.market_data_providers import MarketDataFetchResult
from scripts.cp134_local_daily_update_rehearsal import (
    classify_fetch_errors,
    completed_price_update_dry_run,
    validate_yfinance_price_snapshot,
)


class CollectorJobTestCase(unittest.TestCase):
    def test_daily_append_gate_classifies_yahoo_429(self):
        state = classify_fetch_errors(["yfinance:Exception:HTTP 429 Too Many Requests"])

        self.assertEqual(state, "BLOCKED_YAHOO_429")

    def test_daily_append_gate_validates_yfinance_source_contract(self):
        frame = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2026-05-04"]),
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "adjusted_close": [101.0],
                "volume": [1000],
                "source": ["yfinance"],
                "provider": ["yfinance"],
            }
        )

        result = validate_yfinance_price_snapshot(frame)

        self.assertTrue(result["passed"])
        self.assertEqual(result["duplicate_ticker_date_source"], 0)
        self.assertEqual(result["adjusted_ohlc_violation_count"], 0)

    def test_daily_append_gate_blocks_duplicate_ticker_date_source(self):
        frame = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2026-05-04", "2026-05-04"]),
                "open": [100.0, 100.0],
                "high": [102.0, 102.0],
                "low": [99.0, 99.0],
                "close": [101.0, 101.0],
                "adjusted_close": [101.0, 101.0],
                "volume": [1000, 1000],
                "source": ["yfinance", "yfinance"],
                "provider": ["yfinance", "yfinance"],
            }
        )

        result = validate_yfinance_price_snapshot(frame)

        self.assertFalse(result["passed"])
        self.assertEqual(result["duplicate_ticker_date_source"], 1)

    def test_daily_append_gate_allows_partial_append_ready(self):
        local_price = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "date": pd.to_datetime(["2026-05-04", "2026-05-04"]),
                "open": [100.0, 200.0],
                "high": [101.0, 201.0],
                "low": [99.0, 199.0],
                "close": [100.0, 200.0],
                "adjusted_close": [100.0, 200.0],
                "volume": [1000, 1000],
                "source": ["yfinance", "yfinance"],
                "provider": ["yfinance", "yfinance"],
            }
        )
        provider_frame = pd.DataFrame(
            {
                "Open": [101.0],
                "High": [102.0],
                "Low": [100.0],
                "Close": [101.5],
                "Adj Close": [101.5],
                "Volume": [1200],
                "Amount": [121800.0],
            },
            index=pd.to_datetime(["2026-05-05"]),
        )
        provider_frame.attrs["fetch_method"] = "yahoo_chart"

        def fake_fetch_market_data(ticker, **kwargs):
            if ticker == "AAPL":
                return MarketDataFetchResult("AAPL", "yfinance", "yfinance", provider_frame)
            return MarketDataFetchResult("MSFT", "yfinance", None, pd.DataFrame(), errors=["yfinance:empty"])

        with patch("scripts.cp134_local_daily_update_rehearsal.load_parquet", return_value=local_price), patch(
            "scripts.cp134_local_daily_update_rehearsal.fetch_market_data",
            side_effect=fake_fetch_market_data,
        ):
            completed_rows, metrics = completed_price_update_dry_run(["AAPL", "MSFT"], date(2026, 5, 6), 2)

        self.assertEqual(metrics["status"], "APPEND_READY_PARTIAL")
        self.assertEqual(metrics["append_candidate_tickers"], ["AAPL"])
        self.assertEqual(metrics["failed_tickers"], ["MSFT"])
        self.assertEqual(len(completed_rows), 1)

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

    def test_compute_indicators_does_not_full_backfill_all_tickers_when_partial_state_missing(self):
        def fake_fetch_frame(table, **kwargs):
            filters = kwargs.get("filters") or []
            if table == "price_data":
                tickers = next((item[2] for item in filters if item[0] == "in" and item[1] == "ticker"), [])
                target_tickers = tickers or ["AAPL"]
                row_count = len(target_tickers)
                return pd.DataFrame(
                    {
                        "ticker": target_tickers,
                        "date": ["2026-04-24"] * row_count,
                        "open": [100.0] * row_count,
                        "high": [101.0] * row_count,
                        "low": [99.0] * row_count,
                        "close": [100.0] * row_count,
                        "adjusted_close": [100.0] * row_count,
                        "volume": [1000] * row_count,
                        "amount": [100000.0] * row_count,
                        "per": [10.0] * row_count,
                        "pbr": [2.0] * row_count,
                    }
                )
            if table == "company_fundamentals":
                return pd.DataFrame(
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
            return pd.DataFrame()

        state_map = {"AAPL": {"last_cursor_date": "2026-04-20"}}
        build_features_mock = Mock(return_value=pd.DataFrame())

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame) as fetch_mock, patch(
            "backend.collector.jobs.compute_indicators.build_features",
            build_features_mock,
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value=state_map,
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(lookback_days=14, tickers=["AAPL", "MSFT"])

        price_calls = [
            call.kwargs
            for call in fetch_mock.call_args_list
            if call.args and call.args[0] == "price_data"
        ]

        self.assertTrue(price_calls)
        full_group_calls = [
            kwargs
            for kwargs in price_calls
            if ("gte", "date", "2015-01-01") in (kwargs.get("filters") or [])
            and any(
                filter_item[0] == "in" and filter_item[1] == "ticker" and filter_item[2] == ["AAPL", "MSFT"]
                for filter_item in (kwargs.get("filters") or [])
            )
        ]
        self.assertEqual(full_group_calls, [])

    def test_compute_indicators_can_limit_timeframes_for_backfill(self):
        def fake_fetch_frame(table, **kwargs):
            if table == "price_data":
                return pd.DataFrame(
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
            return pd.DataFrame()

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.compute_indicators.build_features",
            return_value=pd.DataFrame(),
        ) as build_features_mock, patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            result = run_compute_indicators(
                lookback_days=14,
                tickers=["AAPL"],
                force_full_backfill=True,
                full_start_date="2024-01-01",
                timeframes=["1D"],
            )

        self.assertEqual(list(result["timeframes"].keys()), ["1D"])
        build_features_mock.assert_called_once()

    def test_compute_indicators_prunes_full_backfill_rows_before_upsert(self):
        class FakeDeleteQuery:
            def __init__(self, calls, table):
                self.calls = calls
                self.table = table
                self.filters = []

            def delete(self):
                return self

            def eq(self, column, value):
                self.filters.append(("eq", column, value))
                return self

            def gte(self, column, value):
                self.filters.append(("gte", column, value))
                return self

            def lt(self, column, value):
                self.filters.append(("lt", column, value))
                return self

            def gt(self, column, value):
                self.filters.append(("gt", column, value))
                return self

            def execute(self):
                self.calls.append({"table": self.table, "filters": list(self.filters)})
                return Mock(data=[])

        class FakeClient:
            def __init__(self):
                self.delete_calls = []

            def table(self, table):
                return FakeDeleteQuery(self.delete_calls, table)

        fake_client = FakeClient()
        features = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
                "timeframe": ["1D", "1D"],
                "open_ratio": [0.01, 0.02],
            }
        )

        def fake_fetch_frame(table, **kwargs):
            if table == "price_data":
                return pd.DataFrame(
                    {
                        "ticker": ["AAPL"],
                        "date": ["2024-02-02"],
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
            return pd.DataFrame()

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.compute_indicators.build_features",
            return_value=features,
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.compute_indicators.get_client",
            return_value=fake_client,
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_records",
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(
                lookback_days=14,
                tickers=["AAPL"],
                force_full_backfill=True,
                full_start_date="2024-01-01",
                timeframes=["1D"],
            )

        self.assertEqual(len(fake_client.delete_calls), 1)
        self.assertIn(("gte", "date", "2024-01-01"), fake_client.delete_calls[0]["filters"])
        self.assertIn(("eq", "source", "eodhd"), fake_client.delete_calls[0]["filters"])

    def test_compute_indicators_filters_yfinance_price_source_and_stores_source(self):
        mixed_price_frame = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "date": ["2026-04-22", "2026-04-23", "2026-04-24"],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "adjusted_close": [100.0, 101.0, 102.0],
                "volume": [1000, 1000, 1000],
                "amount": [100000.0, 101000.0, 102000.0],
                "per": [10.0, 10.0, 10.0],
                "pbr": [2.0, 2.0, 2.0],
                "source": ["eodhd", "yfinance", "yfinance"],
                "provider": ["eodhd", "yfinance", "yfinance"],
            }
        )

        def fake_fetch_frame(table, **kwargs):
            if table == "price_data":
                return mixed_price_frame.copy()
            return pd.DataFrame()

        def fake_build_features(price_df, **kwargs):
            self.assertEqual(set(price_df["source"].dropna().tolist()), {"yfinance"})
            return pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": [pd.Timestamp("2026-04-24")],
                    "timeframe": ["1D"],
                    "open_ratio": [0.01],
                }
            )

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.compute_indicators.build_features",
            side_effect=fake_build_features,
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_records",
        ) as upsert_mock, patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(
                lookback_days=14,
                tickers=["AAPL"],
                timeframes=["1D"],
                provider="yfinance",
            )

        upsert_mock.assert_called_once()
        self.assertEqual(upsert_mock.call_args.kwargs["on_conflict"], "ticker,timeframe,date,source")
        records = upsert_mock.call_args.args[1]
        self.assertEqual(records[0]["source"], "yfinance")
        self.assertEqual(records[0]["provider"], "yfinance")

    def test_compute_indicators_keeps_eodhd_legacy_boundary_for_weekly_monthly(self):
        observed_sources: list[set[object]] = []
        mixed_price_frame = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "date": ["2026-04-10", "2026-04-17", "2026-04-24"],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "adjusted_close": [100.0, 101.0, 102.0],
                "volume": [1000, 1000, 1000],
                "amount": [100000.0, 101000.0, 102000.0],
                "per": [10.0, 10.0, 10.0],
                "pbr": [2.0, 2.0, 2.0],
                "source": [None, "eodhd", "yfinance"],
                "provider": [None, "eodhd", "yfinance"],
            }
        )

        def fake_fetch_frame(table, **kwargs):
            if table == "price_data":
                return mixed_price_frame.copy()
            return pd.DataFrame()

        def fake_build_features(price_df, **kwargs):
            observed_sources.append(set(price_df["source"].tolist()))
            return pd.DataFrame()

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.compute_indicators.build_features",
            side_effect=fake_build_features,
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(
                lookback_days=14,
                tickers=["AAPL"],
                timeframes=["1W", "1M"],
                provider="eodhd",
            )

        self.assertEqual(len(observed_sources), 2)
        for sources in observed_sources:
            self.assertNotIn("yfinance", sources)
            self.assertIn("eodhd", sources)

    def test_compute_indicators_splits_large_ticker_queries_into_batches(self):
        observed_batches: list[list[str]] = []

        def fake_fetch_frame(table, **kwargs):
            filters = kwargs.get("filters") or []
            if table == "price_data":
                tickers = next((item[2] for item in filters if item[0] == "in" and item[1] == "ticker"), [])
                observed_batches.append(list(tickers))
                return pd.DataFrame(
                    {
                        "ticker": tickers,
                        "date": ["2026-04-24"] * len(tickers),
                        "open": [100.0] * len(tickers),
                        "high": [101.0] * len(tickers),
                        "low": [99.0] * len(tickers),
                        "close": [100.0] * len(tickers),
                        "adjusted_close": [100.0] * len(tickers),
                        "volume": [1000] * len(tickers),
                        "amount": [100000.0] * len(tickers),
                        "per": [10.0] * len(tickers),
                        "pbr": [2.0] * len(tickers),
                    }
                )
            return pd.DataFrame()

        with patch("backend.collector.jobs.compute_indicators.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.compute_indicators.build_features",
            return_value=pd.DataFrame(),
        ), patch(
            "backend.collector.jobs.compute_indicators.get_job_state_map",
            return_value={ticker: {"last_cursor_date": "2026-04-24"} for ticker in ["AAPL", "MSFT", "NVDA"]},
        ), patch(
            "backend.collector.jobs.compute_indicators.TIMEFRAME_BATCH_SIZE",
            {"1D": 2, "1W": 2, "1M": 2},
        ), patch(
            "backend.collector.jobs.compute_indicators.upsert_job_state",
        ):
            run_compute_indicators(
                lookback_days=14,
                tickers=["AAPL", "MSFT", "NVDA"],
                timeframes=["1D"],
            )

        self.assertEqual(observed_batches, [["AAPL", "MSFT"], ["NVDA"]])

    def test_daily_market_sync_coverage_uses_provider_source(self):
        def fake_get_latest_date(table, filters=None, **kwargs):
            self.assertEqual(table, "price_data")
            if ("eq", "source", "yfinance") in (filters or []):
                return date(2026, 4, 24)
            return None

        def fake_fetch_frame(table, **kwargs):
            self.assertEqual(table, "price_data")
            filters = kwargs.get("filters") or []
            self.assertIn(("eq", "source", "yfinance"), filters)
            return pd.DataFrame({"ticker": ["AAPL"], "date": ["2026-04-24"], "source": ["yfinance"]})

        with patch("backend.collector.pipelines.daily_market_sync.get_latest_date", side_effect=fake_get_latest_date), patch(
            "backend.collector.pipelines.daily_market_sync.fetch_frame",
            side_effect=fake_fetch_frame,
        ):
            coverage = _build_price_coverage(["AAPL", "MSFT"], provider="yfinance")

        self.assertEqual(coverage["provider"], "yfinance")
        self.assertEqual(coverage["covered"], 1)
        self.assertEqual(coverage["missing_tickers"], ["MSFT"])


if __name__ == "__main__":
    unittest.main()
