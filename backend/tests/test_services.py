import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app.core.exceptions import InvalidRunStatusError, TimeframeDisabledError
from app.repositories.market_repo import fetch_indicator_rows, fetch_price_rows, fetch_stocks
from app.services.api_service import aggregate_prices, get_latest_prediction_data, resolve_price_window


class ServiceTestCase(unittest.TestCase):
    def test_aggregate_prices_supports_daily_weekly_monthly(self):
        rows = [
            {"date": "2026-03-31", "open": 9.0, "high": 10.0, "low": 8.0, "close": 9.5, "volume": 90},
            {"date": "2026-04-01", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100},
            {"date": "2026-04-02", "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 120},
            {"date": "2026-04-10", "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 130},
        ]

        daily = aggregate_prices(rows, "1D")
        weekly = aggregate_prices(rows, "1W")
        monthly = aggregate_prices(rows, "1M")

        self.assertEqual(len(daily), 4)
        self.assertEqual(len(weekly), 2)
        self.assertEqual(len(monthly), 1)
        self.assertEqual(monthly[0]["close"], 9.5)

    def test_aggregate_prices_drops_partial_week_and_month(self):
        rows = [
            {"date": "2026-04-06", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100},
            {"date": "2026-04-07", "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 120},
            {"date": "2026-04-08", "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 130},
        ]

        weekly = aggregate_prices(rows, "1W")
        monthly = aggregate_prices(rows, "1M")

        self.assertEqual(weekly, [])
        self.assertEqual(monthly, [])

    def test_resolve_price_window_defaults_to_recent_year(self):
        start, end = resolve_price_window("2026-01-01", "2026-04-01")
        self.assertEqual(start, "2026-01-01")
        self.assertEqual(end, "2026-04-01")

    def test_market_repo_uses_local_snapshot_in_local_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir)
            price_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": ["2026-01-02"],
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1000],
                    "source": ["yfinance"],
                    "provider": ["yfinance"],
                }
            )
            indicator_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "timeframe": ["1D"],
                    "date": ["2026-01-02"],
                    "rsi": [55.0],
                    "macd_ratio": [0.01],
                    "source": ["yfinance"],
                    "provider": ["yfinance"],
                }
            )
            stock_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "sector": ["Technology"],
                    "industry": ["Consumer Electronics"],
                    "market_cap": [1000],
                }
            )
            try:
                price_frame.to_parquet(snapshot_dir / "price_data_yfinance.parquet", index=False)
                indicator_frame.to_parquet(snapshot_dir / "indicators_yfinance_1D.parquet", index=False)
                stock_frame.to_parquet(snapshot_dir / "stock_info.parquet", index=False)
            except Exception as exc:
                self.skipTest(f"parquet 엔진을 사용할 수 없어 local snapshot 테스트를 건너뜁니다: {exc}")

            with patch.dict(
                os.environ,
                {
                    "LENS_DATA_BACKEND": "local",
                    "LENS_LOCAL_SNAPSHOT_DIR": str(snapshot_dir),
                    "MARKET_DATA_PROVIDER": "yfinance",
                },
                clear=False,
            ), patch(
                "app.repositories.market_repo.get_supabase",
                side_effect=AssertionError("Supabase REST 조회 금지"),
            ):
                price_rows = fetch_price_rows("AAPL", start="2026-01-01", market_data_provider="yfinance")
                indicator_rows = fetch_indicator_rows("AAPL", timeframe="1D", market_data_provider="yfinance")
                stock_rows = fetch_stocks(search="AAPL", limit=5, market_data_provider="yfinance")

            self.assertEqual(price_rows[0]["close"], 101.0)
            self.assertEqual(indicator_rows[0]["rsi"], 55.0)
            self.assertEqual(stock_rows[0]["ticker"], "AAPL")

    def test_prediction_arrays_default_to_empty_lists(self):
        def fake_fetch(*args, **kwargs):
            return {
                "ticker": "AAPL",
                "model_name": "patchtst",
                "timeframe": "1D",
                "horizon": 5,
                "asof_date": "2026-04-18",
                "decision_time": "2026-04-18T00:00:00Z",
                "run_id": "run-1",
                "model_ver": "v1",
                "signal": "HOLD",
                "forecast_dates": None,
                "line_series": None,
                "upper_band_series": None,
                "lower_band_series": None,
                "conservative_series": None,
            }

        with patch("app.services.api_service.fetch_latest_prediction", side_effect=fake_fetch):
            payload = get_latest_prediction_data("AAPL")

        self.assertEqual(payload["forecast_dates"], [])
        self.assertEqual(payload["line_series"], [])
        self.assertEqual(payload["upper_band_series"], [])
        self.assertEqual(payload["lower_band_series"], [])
        self.assertEqual(payload["conservative_series"], [])

    def test_prediction_latest_with_run_id_uses_run_prediction(self):
        completed_run = {
            "run_id": "run-2",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
        }
        prediction = {
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-04-18",
            "decision_time": "2026-04-18T00:00:00Z",
            "run_id": "run-2",
            "model_ver": "v1",
            "signal": "BUY",
            "forecast_dates": ["2026-04-19"],
            "line_series": [104.0],
            "upper_band_series": [110.0],
            "lower_band_series": [100.0],
            "conservative_series": [104.0],
        }

        with patch("app.services.api_service.fetch_model_run", return_value=completed_run), patch(
            "app.services.api_service.fetch_prediction_by_run",
            return_value=prediction,
        ) as fetch_by_run, patch("app.services.api_service.fetch_latest_prediction") as fetch_latest:
            payload = get_latest_prediction_data("AAPL", run_id="run-2")

        self.assertEqual(payload["run_id"], "run-2")
        fetch_by_run.assert_called_once_with("AAPL", run_id="run-2")
        fetch_latest.assert_not_called()

    def test_prediction_latest_with_composite_run_id_preserves_meta(self):
        completed_run = {
            "run_id": "composite-1D-demo",
            "status": "completed",
            "model_name": "line_band_composite",
            "timeframe": "1D",
            "horizon": 5,
            "feature_version": "v3_adjusted_ohlc",
            "config": {
                "line_model_run_id": "patchtst-run",
                "band_model_run_id": "cnn-lstm-run",
                "composition_policy": "risk_first_lower_preserve",
                "band_calibration_method": "scalar_width",
            },
        }
        prediction = {
            "ticker": "AAPL",
            "model_name": "patchtst_line__cnn_lstm_calibrated_band",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-04-18",
            "decision_time": "2026-04-18T00:00:00Z",
            "run_id": "composite-1D-demo",
            "model_ver": "line_band_v1",
            "signal": "HOLD",
            "forecast_dates": ["2026-04-19"],
            "line_series": [104.0],
            "upper_band_series": [110.0],
            "lower_band_series": [100.0],
            "conservative_series": [100.0],
            "meta": {"line_model_name": "patchtst", "band_model_name": "cnn_lstm"},
        }

        with patch("app.services.api_service.fetch_model_run", return_value=completed_run), patch(
            "app.services.api_service.fetch_prediction_by_run",
            return_value=prediction,
        ):
            payload = get_latest_prediction_data("AAPL", run_id="composite-1D-demo")

        self.assertEqual(payload["model_name"], "patchtst_line__cnn_lstm_calibrated_band")
        self.assertEqual(payload["meta"]["feature_contract"], "v3_adjusted_ohlc")
        self.assertEqual(payload["meta"]["line_model_run_id"], "patchtst-run")
        self.assertEqual(payload["meta"]["composition_policy"], "risk_first_lower_preserve")

    def test_prediction_latest_rejects_failed_nan_run_id(self):
        failed_run = {
            "run_id": "run-failed",
            "status": "failed_nan",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
        }

        with patch("app.services.api_service.fetch_model_run", return_value=failed_run), patch(
            "app.services.api_service.fetch_prediction_by_run"
        ) as fetch_by_run:
            with self.assertRaises(InvalidRunStatusError):
                get_latest_prediction_data("AAPL", run_id="run-failed")

        fetch_by_run.assert_not_called()

    def test_prediction_timeframe_disabled_for_monthly(self):
        with self.assertRaises(TimeframeDisabledError):
            get_latest_prediction_data("AAPL", timeframe="1M")

    def test_indicator_rows_fill_missing_optional_columns(self):
        class FakeIndicatorTable:
            def __init__(self):
                self.selected_columns = []
                self.current_columns = ""

            def select(self, columns):
                self.current_columns = columns
                self.selected_columns.append(columns)
                return self

            def eq(self, *args):
                return self

            def order(self, *args, **kwargs):
                return self

            def limit(self, *args):
                return self

            def execute(self):
                if "atr_ratio" in self.current_columns:
                    raise Exception("{'message': 'column indicators.atr_ratio does not exist'}")
                if "volume" in self.current_columns:
                    raise Exception("{'message': 'column indicators.volume does not exist'}")
                return SimpleNamespace(data=[{"date": "2026-04-18", "rsi": 55.0}])

        fake_table = FakeIndicatorTable()
        fake_client = SimpleNamespace(table=lambda name: fake_table)

        with patch("app.repositories.market_repo.get_supabase", return_value=fake_client):
            rows = fetch_indicator_rows("aapl", timeframe="1D", limit=2)

        self.assertEqual(rows[0]["date"], "2026-04-18")
        self.assertEqual(rows[0]["rsi"], 55.0)
        self.assertIsNone(rows[0]["atr_ratio"])
        self.assertIsNone(rows[0]["volume"])
        self.assertIn("atr_ratio", fake_table.selected_columns[0])
        self.assertNotIn("atr_ratio", fake_table.selected_columns[-2])
        self.assertNotIn("volume", fake_table.selected_columns[-2])
        self.assertEqual(fake_table.selected_columns[-1], "date, volume")

    def test_price_rows_are_source_aware(self):
        class FakeTable:
            def __init__(self):
                self.filters = []
                self.or_filter = None

            def select(self, *args):
                return self

            def eq(self, column, value):
                self.filters.append(("eq", column, value))
                return self

            def gte(self, column, value):
                self.filters.append(("gte", column, value))
                return self

            def lte(self, column, value):
                self.filters.append(("lte", column, value))
                return self

            def order(self, *args, **kwargs):
                return self

            def or_(self, value):
                self.or_filter = value
                return self

            def execute(self):
                rows = [
                    {"date": "2026-04-01", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100, "source": "yfinance"},
                    {"date": "2026-04-01", "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.5, "volume": 200, "source": "eodhd"},
                    {"date": "2026-04-02", "open": 30.0, "high": 31.0, "low": 29.0, "close": 30.5, "volume": 300, "source": None},
                ]
                for _, column, value in self.filters:
                    if column == "source":
                        rows = [row for row in rows if row.get("source") == value]
                if self.or_filter == "source.eq.eodhd,source.is.null":
                    rows = [row for row in rows if row.get("source") in ("eodhd", None)]
                return SimpleNamespace(data=rows)

        fake_client = SimpleNamespace(table=lambda name: FakeTable())

        with patch("app.repositories.market_repo.get_supabase", return_value=fake_client):
            yfinance_rows = fetch_price_rows(
                "aapl",
                start="2026-04-01",
                market_data_provider="yfinance",
            )
            eodhd_rows = fetch_price_rows(
                "aapl",
                start="2026-04-01",
                market_data_provider="eodhd",
            )

        self.assertEqual([row["source"] for row in yfinance_rows], ["yfinance"])
        self.assertEqual([row["source"] for row in eodhd_rows], ["eodhd", None])

    def test_stock_search_falls_back_to_price_data_when_stock_info_fails(self):
        class FakeTable:
            def __init__(self, name):
                self.name = name

            def select(self, *args):
                return self

            def order(self, *args, **kwargs):
                return self

            def limit(self, *args):
                return self

            def ilike(self, *args):
                return self

            def eq(self, *args):
                return self

            def execute(self):
                if self.name == "stock_info":
                    raise Exception("stock_info unavailable")
                return SimpleNamespace(data=[{"ticker": "AAPL"}, {"ticker": "AAPL"}])

        fake_client = SimpleNamespace(table=lambda name: FakeTable(name))

        with patch("app.repositories.market_repo.get_supabase", return_value=fake_client):
            rows = fetch_stocks(search="AAPL", limit=6)

        self.assertEqual(rows, [{"ticker": "AAPL", "sector": None, "industry": None, "market_cap": None}])

    def test_stock_search_falls_back_to_price_data_when_stock_info_empty(self):
        class FakeTable:
            def __init__(self, name):
                self.name = name

            def select(self, *args):
                return self

            def order(self, *args, **kwargs):
                return self

            def limit(self, *args):
                return self

            def ilike(self, *args):
                return self

            def execute(self):
                if self.name == "stock_info":
                    return SimpleNamespace(data=[])
                return SimpleNamespace(data=[{"ticker": "MSFT"}, {"ticker": "MSFT"}, {"ticker": "AAPL"}])

        fake_client = SimpleNamespace(table=lambda name: FakeTable(name))

        with patch("app.repositories.market_repo.get_supabase", return_value=fake_client):
            rows = fetch_stocks(search=None, limit=2)

        self.assertEqual(
            rows,
            [
                {"ticker": "MSFT", "sector": None, "industry": None, "market_cap": None},
                {"ticker": "AAPL", "sector": None, "industry": None, "market_cap": None},
            ],
        )


if __name__ == "__main__":
    unittest.main()
