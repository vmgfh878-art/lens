import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.exceptions import ResourceNotFoundError, TimeframeDisabledError, UpstreamUnavailableError
from app.db import reset_supabase_client
from app.main import app


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        reset_supabase_client()
        self.client = TestClient(app)

    def test_live_health_returns_wrapped_response(self):
        response = self.client.get("/api/v1/health/live")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["status"], "ok")
        self.assertEqual(body["data"]["service"], "lens-backend")
        self.assertIn("request_id", body["meta"])

    def test_ready_health_returns_config_error_when_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/api/v1/health/ready")

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error"]["code"], "CONFIG_ERROR")
        self.assertIn("request_id", body["meta"])

    def test_stock_prices_success(self):
        payload = {
            "ticker": "AAPL",
            "timeframe": "1D",
            "start": "2025-04-18",
            "end": "2026-04-18",
            "data": [
                {
                    "date": "2026-04-18",
                    "open": 100.0,
                    "high": 110.0,
                    "low": 99.0,
                    "close": 105.0,
                    "volume": 1000,
                }
            ],
        }
        with patch("app.routers.v1.stocks.get_price_response_data", return_value=payload):
            response = self.client.get("/api/v1/stocks/AAPL/prices")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["ticker"], "AAPL")
        self.assertEqual(body["data"]["data"][0]["close"], 105.0)
        self.assertEqual(body["data"]["start"], "2025-04-18")
        self.assertEqual(response.headers["cache-control"], "public, max-age=3600")
        self.assertIn("etag", response.headers)
        self.assertIn("request_id", body["meta"])

    def test_stock_prices_returns_not_modified_when_etag_matches(self):
        payload = {
            "ticker": "AAPL",
            "timeframe": "1D",
            "start": "2025-04-18",
            "end": "2026-04-18",
            "data": [
                {
                    "date": "2026-04-18",
                    "open": 100.0,
                    "high": 110.0,
                    "low": 99.0,
                    "close": 105.0,
                    "volume": 1000,
                }
            ],
        }
        with patch("app.routers.v1.stocks.get_price_response_data", return_value=payload):
            first = self.client.get("/api/v1/stocks/AAPL/prices")
            second = self.client.get(
                "/api/v1/stocks/AAPL/prices",
                headers={"If-None-Match": first.headers["etag"]},
            )

        self.assertEqual(second.status_code, 304)
        self.assertEqual(second.text, "")

    def test_stock_prices_not_found(self):
        with patch(
            "app.routers.v1.stocks.get_price_response_data",
            side_effect=ResourceNotFoundError("데이터 없음"),
        ):
            response = self.client.get("/api/v1/stocks/MSFT/prices")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_NOT_FOUND")

    def test_stock_prices_invalid_timeframe(self):
        response = self.client.get("/api/v1/stocks/AAPL/prices", params={"timeframe": "2D"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_prediction_latest_success(self):
        payload = {
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-04-18",
            "decision_time": "2026-04-18T00:00:00Z",
            "run_id": "run-1",
            "model_ver": "v1",
            "signal": "BUY",
            "forecast_dates": ["2026-04-19"],
            "line_series": [104.0],
            "upper_band_series": [110.0],
            "lower_band_series": [100.0],
            "conservative_series": [104.0],
            "band_quantile_low": 0.1,
            "band_quantile_high": 0.9,
        }
        with patch("app.routers.v1.stocks.get_latest_prediction_data", return_value=payload):
            response = self.client.get("/api/v1/stocks/AAPL/predictions/latest")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["signal"], "BUY")
        self.assertEqual(body["data"]["forecast_dates"], ["2026-04-19"])
        self.assertEqual(body["data"]["line_series"], [104.0])
        self.assertEqual(response.headers["cache-control"], "public, max-age=3600")

    def test_prediction_latest_not_found(self):
        with patch(
            "app.routers.v1.stocks.get_latest_prediction_data",
            side_effect=ResourceNotFoundError("예측 없음"),
        ):
            response = self.client.get("/api/v1/stocks/AAPL/predictions/latest")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_NOT_FOUND")

    def test_prediction_timeframe_disabled(self):
        with patch(
            "app.routers.v1.stocks.get_latest_prediction_data",
            side_effect=TimeframeDisabledError("월봉 AI 예측은 Phase 1에서 지원하지 않습니다."),
        ):
            response = self.client.get("/api/v1/stocks/AAPL/predictions/latest", params={"timeframe": "1M"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "TIMEFRAME_DISABLED")

    def test_prediction_upstream_error(self):
        with patch(
            "app.routers.v1.stocks.get_latest_prediction_data",
            side_effect=UpstreamUnavailableError("DB 오류"),
        ):
            response = self.client.get("/api/v1/stocks/AAPL/predictions/latest")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "UPSTREAM_UNAVAILABLE")

    def test_legacy_predict_returns_disabled_status_for_monthly(self):
        response = self.client.get("/predict/AAPL", params={"timeframe": "1M"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "timeframe_disabled")
        self.assertEqual(response.json()["supported"], ["1D", "1W"])


if __name__ == "__main__":
    unittest.main()
