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

    def test_prediction_latest_with_run_id_uses_specific_run(self):
        payload = {
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-04-18",
            "decision_time": "2026-04-18T00:00:00Z",
            "run_id": "run-2",
            "model_ver": "v1",
            "signal": "HOLD",
            "forecast_dates": ["2026-04-19"],
            "line_series": [104.0],
            "upper_band_series": [110.0],
            "lower_band_series": [100.0],
            "conservative_series": [104.0],
            "band_quantile_low": 0.1,
            "band_quantile_high": 0.9,
        }
        with patch("app.routers.v1.stocks.get_latest_prediction_data", return_value=payload) as fetch:
            response = self.client.get("/api/v1/stocks/AAPL/predictions/latest", params={"run_id": "run-2"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["run_id"], "run-2")
        fetch.assert_called_once_with("AAPL", model="patchtst", timeframe="1D", horizon=None, run_id="run-2")

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

    def test_ai_runs_default_query_returns_completed_runs(self):
        rows = [
            {
                "run_id": "run-completed",
                "status": "completed",
                "model_name": "patchtst",
                "timeframe": "1D",
                "horizon": 5,
                "created_at": "2026-04-18T00:00:00Z",
                "config": {"model_ver": "v2", "line_target_type": "raw_future_return"},
                "val_metrics": {"best_epoch": 3, "best_val_total": 0.12},
                "test_metrics": {},
                "checkpoint_path": "ai/artifacts/run.pt",
            }
        ]
        with patch("app.routers.v1.ai.fetch_model_runs", return_value=rows) as fetch:
            response = self.client.get("/api/v1/ai/runs")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"][0]["run_id"], "run-completed")
        self.assertEqual(body["data"][0]["status"], "completed")
        fetch.assert_called_once_with(
            model_name="patchtst",
            timeframe=None,
            status="completed",
            limit=20,
            offset=0,
        )

    def test_ai_runs_can_query_failed_nan_when_status_is_explicit(self):
        rows = [
            {
                "run_id": "run-failed",
                "status": "failed_nan",
                "model_name": "patchtst",
                "timeframe": "1D",
                "horizon": 5,
                "created_at": "2026-04-18T00:00:00Z",
                "config": {},
                "val_metrics": {},
                "test_metrics": {},
                "checkpoint_path": None,
            }
        ]
        with patch("app.routers.v1.ai.fetch_model_runs", return_value=rows) as fetch:
            response = self.client.get("/api/v1/ai/runs", params={"status": "failed_nan"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["status"], "failed_nan")
        fetch.assert_called_once_with(
            model_name="patchtst",
            timeframe=None,
            status="failed_nan",
            limit=20,
            offset=0,
        )

    def test_ai_run_detail_not_found(self):
        with patch("app.routers.v1.ai.fetch_model_run", return_value=None):
            response = self.client.get("/api/v1/ai/runs/missing-run")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_NOT_FOUND")

    def test_ai_run_detail_builds_safe_config_summary(self):
        row = {
            "run_id": "run-1",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "created_at": "2026-04-18T00:00:00Z",
            "wandb_run_id": "wandb-1",
            "checkpoint_path": "ai/artifacts/run.pt",
            "val_metrics": {"best_epoch": 4, "best_val_total": 0.23, "bad_metric": float("nan")},
            "test_metrics": {"mae": 0.11},
            "config": {
                "model_ver": "v2-multihead",
                "target": "close",
                "seq_len": 64,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 3,
                "dropout": 0.1,
                "lr": 0.001,
                "weight_decay": 0.01,
                "batch_size": 32,
                "epochs": 10,
                "seed": 42,
                "ci_aggregate": "mean",
                "ci_target_fast": True,
                "band_mode": "direct",
                "line_target_type": "raw_future_return",
                "band_target_type": "raw_future_return",
            },
        }
        with patch("app.routers.v1.ai.fetch_model_run", return_value=row):
            response = self.client.get("/api/v1/ai/runs/run-1")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["model_ver"], "v2-multihead")
        self.assertEqual(data["best_epoch"], 4)
        self.assertIsNone(data["val_metrics"]["bad_metric"])
        self.assertEqual(data["config_summary"]["seq_len"], 64)
        self.assertEqual(data["config_summary"]["ci_target_fast"], True)
        self.assertIsNone(data["config"])

    def test_ai_run_detail_can_include_raw_config(self):
        row = {
            "run_id": "run-1",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "created_at": "2026-04-18T00:00:00Z",
            "wandb_run_id": None,
            "checkpoint_path": None,
            "val_metrics": {},
            "test_metrics": {},
            "config": {"seq_len": 64, "secret_debug": "kept-only-when-requested"},
        }
        with patch("app.routers.v1.ai.fetch_model_run", return_value=row):
            response = self.client.get("/api/v1/ai/runs/run-1", params={"include_config": "true"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["config"]["secret_debug"], "kept-only-when-requested")

    def test_ai_run_evaluations_filter_by_run_id(self):
        rows = [
            {
                "run_id": "run-1",
                "ticker": "AAPL",
                "timeframe": "1D",
                "asof_date": "2026-04-18",
                "coverage": 0.9,
                "avg_band_width": 0.2,
                "direction_accuracy": 0.6,
                "mae": 0.1,
                "smape": 0.05,
            }
        ]
        with patch("app.routers.v1.ai.fetch_run_evaluations", return_value=rows) as fetch:
            response = self.client.get("/api/v1/ai/runs/run-1/evaluations", params={"ticker": "aapl"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["run_id"], "run-1")
        fetch.assert_called_once_with("run-1", ticker="aapl", timeframe=None, limit=100)

    def test_ai_run_backtests_filter_by_run_id(self):
        rows = [
            {
                "run_id": "run-1",
                "strategy_name": "band_breakout_v1",
                "timeframe": "1D",
                "return_pct": 4.2,
                "mdd": -0.1,
                "sharpe": 1.3,
                "win_rate": 0.55,
                "profit_factor": 1.8,
                "num_trades": 12,
                "meta": {"fee_adjusted_return_pct": 4.0, "fee_adjusted_sharpe": 1.2, "avg_turnover": 0.3},
                "created_at": "2026-04-18T00:00:00Z",
            }
        ]
        with patch("app.routers.v1.ai.fetch_run_backtests", return_value=rows) as fetch:
            response = self.client.get(
                "/api/v1/ai/runs/run-1/backtests",
                params={"strategy_name": "band_breakout_v1"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"][0]
        self.assertEqual(data["run_id"], "run-1")
        self.assertEqual(data["fee_adjusted_return_pct"], 4.0)
        fetch.assert_called_once_with("run-1", strategy_name="band_breakout_v1", timeframe=None, limit=50)


if __name__ == "__main__":
    unittest.main()
