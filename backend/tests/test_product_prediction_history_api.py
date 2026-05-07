import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.db import reset_supabase_client
from app.main import app


LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"


def _row(ticker: str, role: str, run_id: str, asof_date: str, *, line=None, lower=None, upper=None):
    return {
        "ticker": ticker,
        "timeframe": "1D",
        "role": role,
        "run_id": run_id,
        "asof_date": asof_date,
        "display_horizon": 5,
        "display_date": asof_date,
        "line_value": line,
        "lower_value": lower,
        "upper_value": upper,
        "source": "rolling_replay",
        "model_feature_hash": "hash-test",
        "created_at": "2026-05-06T00:00:00Z",
    }


class ProductPredictionHistoryApiTestCase(unittest.TestCase):
    def setUp(self):
        reset_supabase_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshot_dir = Path(self.temp_dir.name)
        rows = [
            _row("AAPL", "line", LINE_RUN_ID, "2026-05-04", line=101.0),
            _row("AAPL", "band", BAND_RUN_ID, "2026-05-04", lower=95.0, upper=108.0),
            _row("AAPL", "line", LINE_RUN_ID, "2026-05-03", line=100.0),
            _row("AAPL", "band", BAND_RUN_ID, "2026-05-03", lower=94.0, upper=107.0),
            _row("MSFT", "line", LINE_RUN_ID, "2026-05-04", line=201.0),
            _row("MSFT", "band", BAND_RUN_ID, "2026-05-04", lower=190.0, upper=215.0),
            _row("NVDA", "line", LINE_RUN_ID, "2026-05-04", line=301.0),
            _row("NVDA", "band", BAND_RUN_ID, "2026-05-04", lower=280.0, upper=330.0),
        ]
        pd.DataFrame(rows).to_parquet(self.snapshot_dir / "product_prediction_history_1D.parquet", index=False)
        (self.snapshot_dir / "product_prediction_history_1D.manifest.json").write_text(
            """
{
  "line_run_id": "patchtst-1D-efad3c29d803",
  "band_run_id": "cnn_lstm-1D-d0c780dee5e8",
  "asof_start": "2026-05-03",
  "asof_end": "2026-05-04",
  "row_count": 8
}
""".strip(),
            encoding="utf-8",
        )
        self.env_patch = patch.dict(os.environ, {"LENS_LOCAL_SNAPSHOT_DIR": str(self.snapshot_dir)})
        self.env_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_product_history_returns_line_and_band_from_local_parquet(self):
        response = self.client.get("/api/v1/stocks/AAPL/predictions/product-history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        data = body["data"]
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["timeframe"], "1D")
        self.assertEqual(data["source"], "product_rolling_replay")
        self.assertEqual(data["latest_asof_date"], "2026-05-04")
        self.assertEqual([row["asof_date"] for row in data["line_history"]], ["2026-05-03", "2026-05-04"])
        self.assertEqual([row["asof_date"] for row in data["band_history"]], ["2026-05-03", "2026-05-04"])
        self.assertEqual(data["line_history"][-1]["value"], 101.0)
        self.assertEqual(data["band_history"][-1]["lower"], 95.0)
        self.assertEqual(data["band_history"][-1]["upper"], 108.0)
        self.assertEqual(data["manifest_summary"]["line_run_id"], LINE_RUN_ID)
        self.assertEqual(data["manifest_summary"]["band_run_id"], BAND_RUN_ID)
        self.assertEqual(data["manifest_summary"]["date_range"], {"start": "2026-05-03", "end": "2026-05-04"})
        self.assertEqual(body["meta"]["total"], 4)

    def test_product_history_can_filter_role_and_limit(self):
        response = self.client.get(
            "/api/v1/stocks/AAPL/predictions/product-history",
            params={"roles": "line", "limit": 1},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(len(data["line_history"]), 1)
        self.assertEqual(data["line_history"][0]["asof_date"], "2026-05-04")
        self.assertEqual(data["band_history"], [])

    def test_product_history_can_filter_by_run_id(self):
        response = self.client.get(
            "/api/v1/stocks/AAPL/predictions/product-history",
            params={"run_id": BAND_RUN_ID},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["line_history"], [])
        self.assertEqual(len(data["band_history"]), 2)
        self.assertTrue(all(row["run_id"] == BAND_RUN_ID for row in data["band_history"]))

    def test_product_history_unsupported_ticker_returns_empty_state(self):
        response = self.client.get("/api/v1/stocks/T/predictions/product-history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["ticker"], "T")
        self.assertIsNone(body["data"]["latest_asof_date"])
        self.assertEqual(body["data"]["line_history"], [])
        self.assertEqual(body["data"]["band_history"], [])
        self.assertEqual(body["meta"]["total"], 0)
        self.assertEqual(body["data"]["empty_reason"], "ticker_or_filter_has_no_product_history")

    def test_product_history_rejects_invalid_roles(self):
        response = self.client.get(
            "/api/v1/stocks/AAPL/predictions/product-history",
            params={"roles": "line,composite"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
