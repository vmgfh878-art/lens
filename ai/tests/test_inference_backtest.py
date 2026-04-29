import unittest
from unittest.mock import patch

import torch
import pandas as pd

from ai.backtest import build_backtest_frame, run_rule_based_backtest
from ai.inference import decode_return_forecasts, resolve_checkpoint_ticker_registry, run_inference


class InferenceBacktestTestCase(unittest.TestCase):
    def test_decode_return_forecasts_converts_returns_to_prices(self):
        line, lower, upper = decode_return_forecasts(
            line_returns=torch.tensor([[0.1, 0.2]]),
            lower_returns=torch.tensor([[0.0, 0.05]]),
            upper_returns=torch.tensor([[0.2, 0.3]]),
            anchor_closes=torch.tensor([100.0]),
        )

        self.assertAlmostEqual(line[0][0], 110.0, places=4)
        self.assertAlmostEqual(line[0][1], 120.0, places=4)
        self.assertAlmostEqual(lower[0][0], 100.0, places=4)
        self.assertAlmostEqual(lower[0][1], 105.0, places=4)
        self.assertAlmostEqual(upper[0][0], 120.0, places=4)
        self.assertAlmostEqual(upper[0][1], 130.0, places=4)

    def test_rule_based_backtest_uses_realized_return(self):
        frame = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "asof_date": ["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-02"],
                "signal": ["BUY", "SELL", "HOLD", "BUY"],
                "realized_return": [0.1, -0.05, 0.02, 0.03],
                "line_return": [0.08, -0.03, 0.01, 0.02],
            }
        )

        result = run_rule_based_backtest(frame)

        self.assertEqual(result["meta"]["position_contract"], "date_equal_abs_exposure")
        self.assertEqual(result["meta"]["portfolio_dates"], 2)
        self.assertEqual(result["num_trades"], 4)
        self.assertGreater(result["return_pct"], 0.0)
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertIn("fee_adjusted_return_pct", result)
        self.assertIn("fee_adjusted_sharpe", result)
        self.assertIn("avg_turnover", result)
        self.assertIn("gross_return_pct", result["meta"])

    def test_build_backtest_frame_uses_adjusted_anchor(self):
        predictions = pd.DataFrame(
            {
                "run_id": ["run-1"],
                "ticker": ["AAPL"],
                "timeframe": ["1D"],
                "asof_date": ["2026-04-01"],
                "line_series": [[55.0]],
                "signal": ["BUY"],
            }
        )
        evaluations = pd.DataFrame(
            {
                "run_id": ["run-1"],
                "ticker": ["AAPL"],
                "timeframe": ["1D"],
                "asof_date": ["2026-04-01"],
                "actual_series": [[60.0]],
            }
        )
        prices = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2026-04-01"],
                "close": [100.0],
                "adjusted_close": [50.0],
            }
        )

        with patch("ai.backtest.fetch_run_predictions", return_value=predictions):
            with patch("ai.backtest.fetch_run_evaluations", return_value=evaluations):
                with patch("ai.backtest.fetch_frame", return_value=prices):
                    frame = build_backtest_frame("run-1", "1D")

        self.assertAlmostEqual(float(frame["anchor_close"].iloc[0]), 50.0)
        self.assertAlmostEqual(float(frame["line_return"].iloc[0]), 0.1)
        self.assertAlmostEqual(float(frame["realized_return"].iloc[0]), 0.2)

    def test_checkpoint_registry_mismatch_raises(self):
        with patch(
            "ai.inference.load_registry",
            return_value={"timeframe": "1D", "mapping": {"AAPL": 0}, "num_tickers": 1},
        ):
            with self.assertRaises(ValueError):
                resolve_checkpoint_ticker_registry(
                    {"num_tickers": 2, "ticker_registry_path": "registry.json"},
                    "1D",
                )

    def test_run_inference_passes_checkpoint_registry_to_bundle(self):
        registry = {"timeframe": "1D", "mapping": {"AAPL": 0, "MSFT": 1}, "num_tickers": 2}
        model_run = {
            "run_id": "run-1",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 2,
            "band_quantile_low": 0.1,
            "band_quantile_high": 0.9,
            "checkpoint_path": "checkpoint.pt",
            "config": {
                "seq_len": 16,
                "tickers": ["MSFT"],
                "limit_tickers": None,
                "use_future_covariate": False,
                "line_target_type": "raw_future_return",
                "band_target_type": "raw_future_return",
                "model_ver": "v2",
            },
        }
        checkpoint_config = {
            "num_tickers": 2,
            "ticker_registry_path": "registry.json",
        }

        with patch("ai.inference.get_model_run", return_value=model_run):
            with patch("ai.inference.load_checkpoint_config", return_value=checkpoint_config):
                with patch("ai.inference.resolve_checkpoint_ticker_registry", return_value=registry):
                    with patch("ai.inference.resolve_bundle", return_value=object()) as resolve_bundle:
                        with patch(
                            "ai.inference.infer_bundle",
                            return_value=([{"ticker": "MSFT"}], [{"ticker": "MSFT"}], {"coverage": 1.0}),
                        ):
                            with patch("ai.inference.save_predictions") as save_predictions:
                                with patch("ai.inference.save_prediction_evaluations") as save_evaluations:
                                    result = run_inference(run_id="run-1", split_name="test", save=True)

        self.assertEqual(result["prediction_count"], 1)
        self.assertIs(resolve_bundle.call_args.kwargs["ticker_registry"], registry)
        self.assertEqual(resolve_bundle.call_args.kwargs["ticker_registry_path"], "registry.json")
        save_predictions.assert_called_once()
        save_evaluations.assert_called_once()


if __name__ == "__main__":
    unittest.main()
