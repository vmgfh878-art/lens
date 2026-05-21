import unittest
from unittest.mock import patch

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch(cpu_only=True)

import pandas as pd

from ai.backtest import build_backtest_frame, run_rule_based_backtest
from ai.inference import _select_bundle_features, decode_return_forecasts, resolve_checkpoint_ticker_registry, run_inference
from ai.preprocessing import MODEL_FEATURE_COLUMNS, SequenceDatasetBundle


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

    def test_build_backtest_frame_filters_price_source(self):
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
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "date": ["2026-04-01", "2026-04-01", "2026-04-01"],
                "close": [100.0, 100.0, 100.0],
                "adjusted_close": [50.0, 80.0, 90.0],
                "source": ["yfinance", "eodhd", None],
            }
        )

        with patch("ai.backtest.fetch_run_predictions", return_value=predictions):
            with patch("ai.backtest.fetch_run_evaluations", return_value=evaluations):
                with patch("ai.backtest.fetch_frame", return_value=prices):
                    yfinance_frame = build_backtest_frame("run-1", "1D", market_data_provider="yfinance")
                    eodhd_frame = build_backtest_frame("run-1", "1D", market_data_provider="eodhd")

        self.assertAlmostEqual(float(yfinance_frame["anchor_close"].iloc[0]), 50.0)
        self.assertIn(float(eodhd_frame["anchor_close"].iloc[0]), {80.0, 90.0})

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
                "market_data_provider": "eodhd",
            },
        }
        checkpoint_config = {
            "num_tickers": 2,
            "ticker_registry_path": "registry.json",
            "market_data_provider": "eodhd",
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
                                    result = run_inference(
                                        run_id="run-1",
                                        split_name="test",
                                        save=True,
                                        allow_bulk_evaluation_save=True,
                                    )

        self.assertEqual(result["prediction_count"], 1)
        self.assertIs(resolve_bundle.call_args.kwargs["ticker_registry"], registry)
        self.assertEqual(resolve_bundle.call_args.kwargs["ticker_registry_path"], "registry.json")
        save_predictions.assert_called_once()
        save_evaluations.assert_called_once()
        saved_prediction = save_predictions.call_args.args[0][0]
        self.assertEqual(saved_prediction["meta"]["storage_contract"], "evaluation_bulk")

    def test_run_inference_blocks_bulk_save_without_explicit_flag(self):
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
                "market_data_provider": "eodhd",
            },
        }
        prediction = {"run_id": "run-1", "ticker": "MSFT", "asof_date": "2026-05-04", "meta": {"layer": "line"}}
        evaluation = {"run_id": "run-1", "ticker": "MSFT", "asof_date": "2026-05-04"}

        with patch("ai.inference.get_model_run", return_value=model_run):
            with patch("ai.inference.load_checkpoint_config", return_value={"num_tickers": 0, "market_data_provider": "eodhd"}):
                with patch("ai.inference.resolve_checkpoint_ticker_registry", return_value=None):
                    with patch("ai.inference.resolve_bundle", return_value=object()):
                        with patch("ai.inference.infer_bundle", return_value=([prediction], [evaluation], {})):
                            with patch("ai.inference.save_predictions") as save_predictions:
                                with self.assertRaisesRegex(ValueError, "bulk predictions"):
                                    run_inference(run_id="run-1", split_name="test", save=True)

        save_predictions.assert_not_called()

    def test_run_inference_uses_product_latest_helper_when_requested(self):
        model_run = {
            "run_id": "product-line-run",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "band_quantile_low": 0.1,
            "band_quantile_high": 0.9,
            "checkpoint_path": "checkpoint.pt",
            "config": {
                "seq_len": 252,
                "tickers": ["AAPL"],
                "limit_tickers": None,
                "use_future_covariate": False,
                "line_target_type": "raw_future_return",
                "band_target_type": "raw_future_return",
                "model_ver": "v2",
                "role": "line_model",
                "market_data_provider": "eodhd",
            },
        }
        prediction = {
            "run_id": "product-line-run",
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-05-04",
            "meta": {"layer": "line", "composite": False},
        }
        evaluation = {"run_id": "product-line-run", "ticker": "AAPL", "timeframe": "1D", "asof_date": "2026-05-04"}

        with patch("ai.inference.get_model_run", return_value=model_run):
            with patch("ai.inference.load_checkpoint_config", return_value={"num_tickers": 0, "market_data_provider": "eodhd"}):
                with patch("ai.inference.resolve_checkpoint_ticker_registry", return_value=None):
                    with patch("ai.inference.resolve_bundle", return_value=object()):
                        with patch("ai.inference.infer_bundle", return_value=([prediction], [evaluation], {})):
                            with patch("ai.inference.save_product_latest_predictions") as product_save:
                                result = run_inference(
                                    run_id="product-line-run",
                                    split_name="test",
                                    save=True,
                                    save_product_latest_only=True,
                                )

        product_save.assert_called_once_with([prediction], [evaluation])
        self.assertEqual(result["storage_contract"], "product_latest_only")

    def test_run_inference_blocks_product_run_bulk_even_with_bulk_flag(self):
        model_run = {
            "run_id": "product-line-run",
            "status": "completed",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "band_quantile_low": 0.1,
            "band_quantile_high": 0.9,
            "checkpoint_path": "checkpoint.pt",
            "config": {
                "seq_len": 252,
                "tickers": ["AAPL"],
                "limit_tickers": None,
                "use_future_covariate": False,
                "line_target_type": "raw_future_return",
                "band_target_type": "raw_future_return",
                "model_ver": "v2",
                "role": "line_model",
                "market_data_provider": "eodhd",
            },
        }
        prediction = {
            "run_id": "product-line-run",
            "ticker": "AAPL",
            "model_name": "patchtst",
            "timeframe": "1D",
            "horizon": 5,
            "asof_date": "2026-05-04",
            "meta": {"layer": "line", "composite": False},
        }
        evaluation = {"run_id": "product-line-run", "ticker": "AAPL", "timeframe": "1D", "asof_date": "2026-05-04"}

        with patch("ai.inference.get_model_run", return_value=model_run):
            with patch("ai.inference.load_checkpoint_config", return_value={"num_tickers": 0, "market_data_provider": "eodhd"}):
                with patch("ai.inference.resolve_checkpoint_ticker_registry", return_value=None):
                    with patch("ai.inference.resolve_bundle", return_value=object()):
                        with patch("ai.inference.infer_bundle", return_value=([prediction], [evaluation], {})):
                            with patch("ai.inference.save_predictions") as save_predictions:
                                with self.assertRaisesRegex(ValueError, "제품 run"):
                                    run_inference(
                                        run_id="product-line-run",
                                        split_name="test",
                                        save=True,
                                        allow_bulk_evaluation_save=True,
                                    )

        save_predictions.assert_not_called()

    def test_select_bundle_features_uses_checkpoint_columns(self):
        features = torch.arange(2 * 3 * len(MODEL_FEATURE_COLUMNS), dtype=torch.float32).reshape(
            2,
            3,
            len(MODEL_FEATURE_COLUMNS),
        )
        bundle = SequenceDatasetBundle(
            features=features,
            line_targets=torch.zeros(2, 1),
            band_targets=torch.zeros(2, 1),
            raw_future_returns=torch.zeros(2, 1),
            anchor_closes=torch.ones(2),
            ticker_ids=torch.zeros(2, dtype=torch.long),
            future_covariates=torch.zeros(2, 1, 0),
            metadata=pd.DataFrame({"ticker": ["AAPL", "MSFT"], "asof_date": ["2026-01-01", "2026-01-02"]}),
        )
        selected = _select_bundle_features(bundle, ["log_return", "low_ratio", "bb_position"])

        self.assertEqual(selected.features.shape[2], 3)
        self.assertTrue(torch.equal(selected.features[:, :, 0], bundle.features[:, :, 0]))


if __name__ == "__main__":
    unittest.main()
