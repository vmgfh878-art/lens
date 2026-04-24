import unittest
from unittest.mock import patch

from app.core.exceptions import TimeframeDisabledError
from app.services.api_service import aggregate_prices, get_latest_prediction_data, resolve_price_window


class ServiceTestCase(unittest.TestCase):
    def test_aggregate_prices_supports_daily_weekly_monthly(self):
        rows = [
            {"date": "2026-04-01", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100},
            {"date": "2026-04-02", "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 120},
            {"date": "2026-04-10", "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 130},
        ]

        daily = aggregate_prices(rows, "1D")
        weekly = aggregate_prices(rows, "1W")
        monthly = aggregate_prices(rows, "1M")

        self.assertEqual(len(daily), 3)
        self.assertEqual(len(weekly), 2)
        self.assertEqual(len(monthly), 1)
        self.assertEqual(monthly[0]["close"], 12.5)

    def test_resolve_price_window_defaults_to_recent_year(self):
        start, end = resolve_price_window("2026-01-01", "2026-04-01")
        self.assertEqual(start, "2026-01-01")
        self.assertEqual(end, "2026-04-01")

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

    def test_prediction_timeframe_disabled_for_monthly(self):
        with self.assertRaises(TimeframeDisabledError):
            get_latest_prediction_data("AAPL", timeframe="1M")


if __name__ == "__main__":
    unittest.main()
