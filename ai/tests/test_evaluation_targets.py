import unittest

import torch
import pandas as pd

from ai.evaluation import summarize_forecast_metrics
from ai.preprocessing import build_lazy_sequence_dataset
from ai.targets import build_target_array


def _build_feature_rows(length: int = 20, ticker: str = "AAPL") -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    feature_rows = []
    price_rows = []
    for idx, date_value in enumerate(dates):
        feature_rows.append(
            {
                "ticker": ticker,
                "timeframe": "1D",
                "date": date_value,
                "log_return": 0.01,
                "open_ratio": 0.01,
                "high_ratio": 0.02,
                "low_ratio": -0.01,
                "vol_change": 0.03,
                "ma_5_ratio": 0.01,
                "ma_20_ratio": 0.01,
                "ma_60_ratio": 0.01,
                "rsi": 0.5,
                "macd_ratio": 0.01,
                "bb_position": 0.5,
                "us10y": 4.0,
                "yield_spread": 1.0,
                "vix_close": 20.0,
                "credit_spread_hy": 3.0,
                "nh_nl_index": 0.2,
                "ma200_pct": 0.7,
                "regime_calm": 0.0,
                "regime_neutral": 1.0,
                "regime_stress": 0.0,
                "revenue": 100.0,
                "net_income": 10.0,
                "equity": 50.0,
                "eps": 1.0,
                "roe": 0.2,
                "debt_ratio": 0.4,
                "has_macro": True,
                "has_breadth": True,
                "has_fundamentals": True,
            }
        )
        price_rows.append(
            {
                "ticker": ticker,
                "date": date_value,
                "close": 100.0 + idx,
                "adjusted_close": 100.0 + idx,
                "open": 100.0 + idx,
                "high": 101.0 + idx,
                "low": 99.0 + idx,
                "volume": 1000 + idx,
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(price_rows)


class EvaluationTargetsTestCase(unittest.TestCase):
    def test_summarize_forecast_metrics_reports_investment_metrics(self):
        metadata = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "asof_date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
            }
        )
        line = torch.tensor([[0.10], [-0.10], [0.20], [-0.20]], dtype=torch.float32)
        lower = line - 0.02
        upper = line + 0.02
        actual = torch.tensor([[0.08], [-0.09], [0.19], [-0.18]], dtype=torch.float32)

        metrics = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=line,
            lower_predictions=lower,
            upper_predictions=upper,
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            line_target_type="raw_future_return",
            band_target_type="raw_future_return",
            top_k_frac=0.5,
            fee_bps=10.0,
        )

        self.assertAlmostEqual(float(metrics["spearman_ic"]), 1.0, places=6)
        self.assertGreater(float(metrics["top_k_long_spread"]), 0.0)
        self.assertLess(float(metrics["top_k_short_spread"]), 0.0)
        self.assertGreater(float(metrics["long_short_spread"]), 0.0)
        self.assertIsNotNone(metrics["fee_adjusted_return"])
        self.assertAlmostEqual(float(metrics["lower_breach_rate"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["upper_breach_rate"]), 0.0, places=6)

    def test_summarize_forecast_metrics_reports_band_breach_rates(self):
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32),
            lower_predictions=torch.tensor([[-0.1], [-0.1], [-0.1]], dtype=torch.float32),
            upper_predictions=torch.tensor([[0.1], [0.1], [0.1]], dtype=torch.float32),
            line_targets=torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32),
            band_targets=torch.tensor([[-0.2], [0.0], [0.2]], dtype=torch.float32),
            raw_future_returns=torch.tensor([[-0.2], [0.0], [0.2]], dtype=torch.float32),
        )

        self.assertAlmostEqual(float(metrics["coverage"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(metrics["lower_breach_rate"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(metrics["upper_breach_rate"]), 1.0 / 3.0, places=6)

    def test_build_target_array_supports_direction_and_vol_normalized(self):
        future = build_target_array([0.1, -0.2], history_returns=[0.01, 0.02], target_type="raw_future_return")
        direction = build_target_array([0.1, -0.2], history_returns=[0.01, 0.02], target_type="direction_label")
        vol_norm = build_target_array([0.1, -0.2], history_returns=[0.1, -0.1, 0.2], target_type="volatility_normalized_return")

        self.assertAlmostEqual(float(future[0]), 0.1, places=6)
        self.assertAlmostEqual(float(future[1]), -0.2, places=6)
        self.assertEqual(direction.tolist(), [1.0, 0.0])
        self.assertEqual(len(vol_norm), 2)

    def test_lazy_dataset_supports_direction_target_plumbing(self):
        feature_df, price_df = _build_feature_rows()
        dataset = build_lazy_sequence_dataset(
            feature_df,
            price_df,
            timeframe="1D",
            seq_len=10,
            horizon=2,
            line_target_type="direction_label",
            band_target_type="raw_future_return",
        )
        _, line_target, band_target, _raw_future_returns, _, _ = dataset[0]
        self.assertTrue(set(line_target.tolist()).issubset({0.0, 1.0}))
        self.assertEqual(band_target.shape[0], 2)

    def test_direction_label_uses_threshold_half_and_investment_metrics_use_raw_return(self):
        metadata = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "asof_date": ["2026-01-05", "2026-01-05"],
            }
        )
        metrics = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=torch.tensor([[0.6], [0.4]], dtype=torch.float32),
            lower_predictions=torch.tensor([[0.2], [0.1]], dtype=torch.float32),
            upper_predictions=torch.tensor([[0.8], [0.9]], dtype=torch.float32),
            line_targets=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            band_targets=torch.tensor([[0.7], [0.2]], dtype=torch.float32),
            raw_future_returns=torch.tensor([[0.05], [-0.03]], dtype=torch.float32),
            line_target_type="direction_label",
            band_target_type="raw_future_return",
        )
        self.assertAlmostEqual(float(metrics["direction_accuracy"]), 1.0, places=6)
        self.assertGreater(float(metrics["long_short_spread"]), 0.0)

    def test_conservative_line_metrics_are_reported(self):
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=torch.tensor([[0.01, -0.04, -0.05, 0.04, -0.02]], dtype=torch.float32),
            lower_predictions=torch.full((1, 5), -0.10, dtype=torch.float32),
            upper_predictions=torch.full((1, 5), 0.10, dtype=torch.float32),
            line_targets=torch.tensor([[-0.10, -0.08, -0.01, 0.06, 0.08]], dtype=torch.float32),
            band_targets=torch.tensor([[-0.10, -0.08, -0.01, 0.06, 0.08]], dtype=torch.float32),
            raw_future_returns=torch.tensor([[-0.10, -0.08, -0.01, 0.06, 0.08]], dtype=torch.float32),
            line_target_type="raw_future_return",
            band_target_type="raw_future_return",
        )

        self.assertIn("underprediction_rate", metrics)
        self.assertIn("false_safe_rate", metrics)
        self.assertIn("downside_capture_rate", metrics)
        self.assertIn("severe_downside_recall", metrics)
        self.assertIn("conservative_bias", metrics)
        self.assertIn("upside_sacrifice", metrics)
        self.assertGreater(float(metrics["false_safe_rate"]), 0.0)
        self.assertLess(float(metrics["conservative_bias"]), 0.0)

    def test_horizon_segment_metrics_are_reported(self):
        metadata = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "asof_date": ["2026-01-05", "2026-01-05", "2026-01-05"],
            }
        )
        actual = torch.tensor(
            [
                [0.03] * 20,
                [0.01] * 20,
                [-0.02] * 20,
            ],
            dtype=torch.float32,
        )
        line = actual.clone()
        metrics = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=line,
            lower_predictions=line - 0.10,
            upper_predictions=line + 0.10,
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            line_target_type="raw_future_return",
            band_target_type="raw_future_return",
        )

        self.assertAlmostEqual(float(metrics["h1_h5_spearman_ic"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["h6_h10_spearman_ic"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["h11_h20_spearman_ic"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["all_horizon_mae"]), 0.0, places=6)
        self.assertGreater(float(metrics["h1_h5_long_short_spread"]), 0.0)

    def test_band_calibration_and_interval_score_are_reported(self):
        actual = torch.tensor([[-0.20], [0.00], [0.05], [0.20]], dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=torch.zeros((4, 1), dtype=torch.float32),
            lower_predictions=torch.full((4, 1), -0.10, dtype=torch.float32),
            upper_predictions=torch.full((4, 1), 0.10, dtype=torch.float32),
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            q_low=0.20,
            q_high=0.80,
        )

        self.assertAlmostEqual(float(metrics["nominal_coverage"]), 0.60, places=6)
        self.assertAlmostEqual(float(metrics["empirical_coverage"]), 0.50, places=6)
        self.assertAlmostEqual(float(metrics["coverage_error"]), -0.10, places=6)
        self.assertAlmostEqual(float(metrics["coverage_abs_error"]), 0.10, places=6)
        self.assertAlmostEqual(float(metrics["median_band_width"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics["p90_band_width"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics["interval_width_component"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics["interval_lower_penalty"]), 0.25, places=6)
        self.assertAlmostEqual(float(metrics["interval_upper_penalty"]), 0.1250, places=6)
        self.assertAlmostEqual(float(metrics["interval_score"]), 0.575, places=6)

    def test_dynamic_width_ic_and_squeeze_metrics_are_reported(self):
        raw = torch.tensor([[-0.01], [-0.02], [-0.04], [-0.08]], dtype=torch.float32)
        widths = torch.tensor([[0.02], [0.04], [0.08], [0.16]], dtype=torch.float32)
        line = torch.zeros((4, 1), dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=line,
            lower_predictions=-(widths / 2.0),
            upper_predictions=widths / 2.0,
            line_targets=raw,
            band_targets=raw,
            raw_future_returns=raw,
        )

        self.assertAlmostEqual(float(metrics["band_width_ic"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["downside_width_ic"]), 1.0, places=6)
        self.assertIn("width_bucket_high_realized_vol", metrics)
        self.assertIn("width_bucket_downside_rate_ratio", metrics)
        self.assertIn("squeeze_breakout_rate", metrics)
        self.assertIsNotNone(metrics["squeeze_breakout_rate"])

    def test_band_horizon_bucket_metrics_are_reported(self):
        actual = torch.tensor([[0.01] * 5 + [0.03] * 5 + [-0.02] * 10], dtype=torch.float32)
        line = torch.zeros((1, 20), dtype=torch.float32)
        lower = torch.full((1, 20), -0.05, dtype=torch.float32)
        upper = torch.full((1, 20), 0.05, dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=line,
            lower_predictions=lower,
            upper_predictions=upper,
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            q_low=0.15,
            q_high=0.85,
        )

        self.assertIn("all_horizon_band_empirical_coverage", metrics)
        self.assertIn("h1_h5_band_interval_score", metrics)
        self.assertIn("h6_h10_band_interval_score", metrics)
        self.assertIn("h11_h20_band_interval_score", metrics)
        self.assertAlmostEqual(float(metrics["h1_h5_band_empirical_coverage"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["h6_h10_band_empirical_coverage"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["h11_h20_band_empirical_coverage"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
