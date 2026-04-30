import unittest
from unittest.mock import patch

from ai.storage import save_model_run, save_predictions


class StorageContractsTestCase(unittest.TestCase):
    def test_save_model_run_upserts_by_run_id(self):
        record = {
            "run_id": "run-1",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "feature_version": "v3_adjusted_ohlc",
            "band_mode": "direct",
            "status": "completed",
        }

        with patch("ai.storage.upsert_records") as upsert:
            save_model_run(record)

        upsert.assert_called_once_with("model_runs", [record], on_conflict="run_id")

    def test_save_predictions_upsert_key_keeps_run_history(self):
        records = [
            {
                "run_id": "run-1",
                "ticker": "AAPL",
                "model_name": "patchtst",
                "timeframe": "1D",
                "horizon": 5,
                "asof_date": "2026-04-01",
            }
        ]

        with patch("ai.storage.upsert_records") as upsert:
            save_predictions(records)

        upsert.assert_called_once_with(
            "predictions",
            records,
            on_conflict="run_id,ticker,model_name,timeframe,horizon,asof_date",
        )

    def test_save_predictions_preserves_composition_meta(self):
        records = [
            {
                "run_id": "composite-run-1",
                "ticker": "AAPL",
                "model_name": "patchtst_line__cnn_lstm_calibrated_band",
                "timeframe": "1D",
                "horizon": 5,
                "asof_date": "2026-04-01",
                "meta": {
                    "line_model_run_id": "line-run",
                    "band_model_run_id": "band-run",
                    "band_calibration_method": "scalar_width",
                    "prediction_composition_version": "line_band_v1",
                },
            }
        ]

        with patch("ai.storage.upsert_records") as upsert:
            save_predictions(records)

        saved_records = upsert.call_args.args[1]
        self.assertEqual(saved_records[0]["meta"]["line_model_run_id"], "line-run")
        self.assertEqual(saved_records[0]["meta"]["band_model_run_id"], "band-run")


if __name__ == "__main__":
    unittest.main()
