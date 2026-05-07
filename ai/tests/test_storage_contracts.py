import unittest
from unittest.mock import patch

from ai.storage import (
    STORAGE_CONTRACT_EVALUATION_BULK,
    STORAGE_CONTRACT_PRODUCT_LATEST_ONLY,
    save_model_run,
    save_predictions,
    save_product_latest_predictions,
    with_prediction_storage_contract,
)


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

    def test_save_product_latest_predictions_guards_rows_and_composite(self):
        prediction = {
            "run_id": "line-run",
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-05-01",
            "meta": {"layer": "line", "composite": False},
        }
        evaluation = {
            "run_id": "line-run",
            "ticker": "AAPL",
            "timeframe": "1D",
            "asof_date": "2026-05-01",
        }

        with patch("ai.storage.upsert_records") as upsert:
            save_product_latest_predictions([prediction], [evaluation], max_prediction_rows=1, max_evaluation_rows=1)

        self.assertEqual(upsert.call_count, 2)
        self.assertEqual(upsert.call_args_list[0].args[0], "predictions")
        self.assertEqual(upsert.call_args_list[1].args[0], "prediction_evaluations")
        saved_prediction = upsert.call_args_list[0].args[1][0]
        self.assertEqual(saved_prediction["meta"]["storage_contract"], STORAGE_CONTRACT_PRODUCT_LATEST_ONLY)

        composite = {**prediction, "model_name": "line_band_composite", "meta": {"layer": "composite", "composite": True}}
        with self.assertRaises(ValueError):
            save_product_latest_predictions([composite], [evaluation], max_prediction_rows=1, max_evaluation_rows=1)

        with self.assertRaises(ValueError):
            save_product_latest_predictions([prediction, {**prediction, "ticker": "MSFT"}], [evaluation], max_prediction_rows=1)

    def test_save_product_latest_predictions_requires_single_asof_date(self):
        first = {
            "run_id": "line-run",
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-05-01",
            "meta": {"layer": "line", "composite": False},
        }
        second = {**first, "ticker": "MSFT", "asof_date": "2026-05-04"}
        with self.assertRaises(ValueError):
            save_product_latest_predictions([first, second], [])

    def test_with_prediction_storage_contract_does_not_mutate_input(self):
        records = [{"ticker": "AAPL", "meta": {"layer": "line"}}]

        annotated = with_prediction_storage_contract(records, STORAGE_CONTRACT_EVALUATION_BULK)

        self.assertEqual(records[0]["meta"], {"layer": "line"})
        self.assertEqual(annotated[0]["meta"]["layer"], "line")
        self.assertEqual(annotated[0]["meta"]["storage_contract"], STORAGE_CONTRACT_EVALUATION_BULK)


if __name__ == "__main__":
    unittest.main()
