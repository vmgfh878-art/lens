import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.config import get_settings
from backend.collector.jobs import sync_prices
from backend.collector.jobs.sync_prices import _build_price_records
from backend.collector.pipelines.yfinance_price_sync import load_baseline_price_frame, run_dry_run_compare
from backend.collector.sources.market_data_providers import (
    MarketDataFetchResult,
    YFinancePriceProvider,
    fetch_market_data,
    provider_adjustment_policy,
)
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract


class MarketDataProviderTestCase(unittest.TestCase):
    def _sample_provider_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [100.0, 49.5, 50.5],
                "High": [102.0, 51.0, 52.0],
                "Low": [98.0, 48.0, 50.0],
                "Close": [100.0, 50.0, 51.0],
                "Adj Close": [50.0, 50.0, 51.0],
                "Volume": [1000, 2000, 2100],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )

    def test_yfinance_provider_uses_raw_ohlc_and_adjusted_close(self):
        source_frame = self._sample_provider_frame()
        with patch("backend.collector.sources.market_data_providers.yf.download", return_value=source_frame) as mock_download:
            frame = YFinancePriceProvider(enable_direct_chart_fallback=False).fetch_daily(
                "AAPL",
                start_date="2026-01-01",
                end_date="2026-01-06",
            )

        self.assertFalse(frame.empty)
        self.assertIn("Adj Close", frame.columns)
        self.assertIn("Amount", frame.columns)
        kwargs = mock_download.call_args.kwargs
        self.assertFalse(kwargs["auto_adjust"])
        self.assertFalse(kwargs["actions"])
        self.assertFalse(kwargs["threads"])

    def test_yfinance_provider_retries_empty_response(self):
        source_frame = self._sample_provider_frame()
        with patch(
            "backend.collector.sources.market_data_providers.yf.download",
            side_effect=[pd.DataFrame(), source_frame],
        ) as mock_download, patch("backend.collector.sources.market_data_providers.time.sleep") as sleep_mock:
            frame = YFinancePriceProvider(max_retries=1, retry_sleep_seconds=0, enable_direct_chart_fallback=False).fetch_daily(
                "AAPL",
                start_date="2026-01-01",
                end_date="2026-01-06",
            )

        self.assertFalse(frame.empty)
        self.assertEqual(mock_download.call_count, 2)
        sleep_mock.assert_called_once()

    def test_yfinance_provider_uses_direct_chart_when_download_empty(self):
        source_frame = self._sample_provider_frame()
        with patch(
            "backend.collector.sources.market_data_providers.yf.download",
            return_value=pd.DataFrame(),
        ), patch(
            "backend.collector.sources.market_data_providers.fetch_yahoo_chart_frame",
            return_value=source_frame.assign(Amount=source_frame["Close"] * source_frame["Volume"]),
        ) as direct_mock, patch("backend.collector.sources.market_data_providers.time.sleep"):
            frame = YFinancePriceProvider(max_retries=0).fetch_daily(
                "AAPL",
                start_date="2026-01-01",
                end_date="2026-01-06",
            )

        self.assertFalse(frame.empty)
        direct_mock.assert_called_once()
        self.assertEqual(frame.attrs["fetch_method"], "yahoo_chart")
        self.assertEqual(frame.attrs["market_data_provider"], "yfinance")

    def test_yfinance_direct_chart_frame_satisfies_adjusted_contract(self):
        source_frame = self._sample_provider_frame()
        source_frame.attrs["fetch_method"] = "yahoo_chart"

        result = validate_adjusted_ohlc_contract("AAPL", source_frame)

        self.assertTrue(result.passed)

    def test_yfinance_fetch_without_fallback_keeps_eodhd_unused(self):
        source_frame = self._sample_provider_frame()
        source_frame.attrs["fetch_method"] = "yahoo_chart"
        with patch(
            "backend.collector.sources.market_data_providers.YFinancePriceProvider.fetch_daily",
            return_value=source_frame,
        ), patch("backend.collector.sources.market_data_providers.EodhdPriceProvider.fetch_daily") as eodhd_mock:
            result = fetch_market_data(
                "AAPL",
                start_date="2026-01-01",
                end_date="2026-01-06",
                provider_name="yfinance",
                fallback_provider_name=None,
                eodhd_api_key=None,
            )

        self.assertFalse(result.frame.empty)
        self.assertFalse(result.fallback_used)
        eodhd_mock.assert_not_called()

    def test_adjusted_ohlc_contract_accepts_split_like_raw_move(self):
        result = validate_adjusted_ohlc_contract("AAPL", self._sample_provider_frame())

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["adjusted_high_violation_count"], 0)
        self.assertEqual(result.metrics["adjusted_low_violation_count"], 0)
        self.assertEqual(result.metrics["adjusted_factor_invalid_count"], 0)

    def test_adjusted_ohlc_contract_rejects_duplicate_dates(self):
        frame = self._sample_provider_frame()
        frame.index = pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-06"])

        result = validate_adjusted_ohlc_contract("AAPL", frame)

        self.assertFalse(result.passed)
        self.assertIn("duplicate_date", result.violations)

    def test_dry_run_compare_output_passes_for_matching_baseline(self):
        provider_frame = self._sample_provider_frame()
        baseline = pd.DataFrame(
            {
                "ticker": ["AAPL"] * len(provider_frame),
                "date": provider_frame.index.strftime("%Y-%m-%d"),
                "open": provider_frame["Open"].tolist(),
                "high": provider_frame["High"].tolist(),
                "low": provider_frame["Low"].tolist(),
                "close": provider_frame["Close"].tolist(),
                "adjusted_close": provider_frame["Adj Close"].tolist(),
                "volume": provider_frame["Volume"].tolist(),
            }
        )
        fetch_result = MarketDataFetchResult(
            ticker="AAPL",
            requested_provider="yfinance",
            provider="yfinance",
            frame=provider_frame,
            fallback_provider="eodhd",
            fallback_used=False,
        )

        with patch(
            "backend.collector.pipelines.yfinance_price_sync.load_baseline_price_frame",
            return_value=baseline,
        ), patch(
            "backend.collector.pipelines.yfinance_price_sync.fetch_market_data",
            return_value=fetch_result,
        ):
            metrics = run_dry_run_compare(
                ["AAPL"],
                start_date="2026-01-01",
                end_date="2026-01-06",
                provider="yfinance",
                fallback_provider="eodhd",
                eodhd_api_key=None,
            )

        self.assertTrue(metrics["overall_pass"])
        self.assertTrue(metrics["tickers"]["AAPL"]["contract"]["passed"])
        self.assertEqual(metrics["tickers"]["AAPL"]["comparison"]["date_coverage"], 1.0)

    def test_yfinance_compare_baseline_uses_local_snapshot_when_required(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir)
            baseline = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": ["2026-01-02"],
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "adjusted_close": [101.0],
                    "volume": [1000],
                    "source": ["eodhd"],
                    "provider": ["eodhd"],
                }
            )
            try:
                baseline.to_parquet(snapshot_dir / "price_data_eodhd.parquet", index=False)
            except Exception as exc:
                self.skipTest(f"parquet 엔진을 사용할 수 없어 local snapshot 테스트를 건너뜁니다: {exc}")

            with patch.dict(
                os.environ,
                {
                    "LENS_DATA_BACKEND": "local",
                    "LENS_LOCAL_SNAPSHOT_DIR": str(snapshot_dir),
                },
                clear=False,
            ), patch(
                "backend.collector.pipelines.yfinance_price_sync.fetch_frame",
                side_effect=AssertionError("Supabase REST 조회 금지"),
            ):
                frame = load_baseline_price_frame(["AAPL"], "2026-01-01", "2026-01-06")

            self.assertEqual(frame["ticker"].tolist(), ["AAPL"])

    def test_provider_config_defaults_yfinance_fallback_to_eodhd(self):
        with patch.dict(os.environ, {"MARKET_DATA_PROVIDER": "yfinance"}, clear=True):
            settings = get_settings()

        self.assertEqual(settings.market_data_provider, "yfinance")
        self.assertEqual(settings.market_data_fallback_provider, "eodhd")

    def test_provider_config_empty_fallback_disables_eodhd_fallback(self):
        with patch.dict(
            os.environ,
            {"MARKET_DATA_PROVIDER": "yfinance", "MARKET_DATA_FALLBACK_PROVIDER": ""},
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.market_data_provider, "yfinance")
        self.assertIsNone(settings.market_data_fallback_provider)

    def test_yfinance_price_records_include_source_provenance(self):
        records = _build_price_records(
            "AAPL",
            self._sample_provider_frame(),
            source="yfinance",
            adjustment_policy=provider_adjustment_policy("yfinance"),
        )

        self.assertTrue(records)
        self.assertEqual(records[0]["source"], "yfinance")
        self.assertEqual(records[0]["provider"], "yfinance")
        self.assertEqual(
            records[0]["provider_adjustment_policy"],
            "yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc",
        )
        self.assertIn("updated_at", records[0])

    def test_sync_prices_upsert_conflict_includes_source(self):
        fetch_result = MarketDataFetchResult(
            ticker="AAPL",
            requested_provider="yfinance",
            provider="yfinance",
            frame=self._sample_provider_frame(),
            fallback_provider="eodhd",
            fallback_used=False,
        )

        def fake_fetch_frame(table, **kwargs):
            if table == "stock_info":
                return pd.DataFrame({"ticker": ["AAPL"]})
            if table == "company_fundamentals":
                return pd.DataFrame({"ticker": []})
            return pd.DataFrame()

        with patch("backend.collector.jobs.sync_prices.fetch_frame", side_effect=fake_fetch_frame), patch(
            "backend.collector.jobs.sync_prices.fetch_market_data",
            return_value=fetch_result,
        ), patch(
            "backend.collector.jobs.sync_prices.get_job_state_map",
            return_value={},
        ), patch(
            "backend.collector.jobs.sync_prices.attach_trailing_valuation",
            side_effect=lambda frame, fundamentals: frame,
        ), patch(
            "backend.collector.jobs.sync_prices.upsert_records",
        ) as upsert_mock, patch(
            "backend.collector.jobs.sync_prices.upsert_job_state",
        ), patch(
            "backend.collector.jobs.sync_prices.time.sleep",
        ):
            sync_prices.run(
                ["AAPL"],
                default_start="2026-01-01",
                provider="yfinance",
                fallback_provider="eodhd",
                force_start_date=True,
                sleep_seconds=0,
                batch_limit=1,
            )

        upsert_mock.assert_called_once()
        self.assertEqual(upsert_mock.call_args.kwargs["on_conflict"], "ticker,date,source")

    def test_schema_declares_source_aware_unique_keys(self):
        schema_text = (ROOT_DIR / "backend" / "db" / "schema.sql").read_text(encoding="utf-8")

        self.assertIn("UNIQUE (ticker, date, source)", schema_text)
        self.assertIn("UNIQUE (ticker, timeframe, date, source)", schema_text)

    def test_eodhd_legacy_null_start_date_avoids_overlap_until_backfill(self):
        legacy_rows = pd.DataFrame(
            {
                "date": ["2026-05-01"],
                "source": [None],
            }
        )

        with patch("backend.collector.jobs.sync_prices.fetch_frame", return_value=legacy_rows):
            start_date = sync_prices._get_ticker_start_date(
                "AAPL",
                default_start="2015-01-01",
                lookback_days=7,
                repair_mode=False,
                provider="eodhd",
            )

        self.assertEqual(start_date, "2026-05-02")


if __name__ == "__main__":
    unittest.main()
