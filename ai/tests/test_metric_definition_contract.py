import unittest

import pandas as pd
import torch

from ai.evaluation import summarize_forecast_metrics


class MetricDefinitionContractTest(unittest.TestCase):
    def test_false_safe_rates_are_split_by_definition(self):
        metadata = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "asof_date": ["2026-01-05"] * 4,
            }
        )
        actual = torch.tensor([[-0.04], [-0.02], [0.01], [0.03]], dtype=torch.float32)
        score = torch.tensor([[0.01], [-0.01], [0.02], [0.04]], dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=score,
            lower_predictions=torch.full((4, 1), -0.10, dtype=torch.float32),
            upper_predictions=torch.full((4, 1), 0.10, dtype=torch.float32),
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            severe_downside_threshold=-0.03,
        )

        self.assertAlmostEqual(float(metrics["false_safe_negative_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["false_safe_tail_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["false_safe_severe_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["severe_downside_recall"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["downside_capture_rate"]), 0.0, places=6)

    def test_line_stability_metrics_use_date_distribution(self):
        metadata = pd.DataFrame(
            {
                "ticker": ["A", "B", "A", "B"],
                "asof_date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
            }
        )
        actual = torch.tensor([[0.02], [-0.01], [-0.01], [0.02]], dtype=torch.float32)
        score = torch.tensor([[0.03], [-0.02], [0.03], [-0.02]], dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=score,
            lower_predictions=score - 0.10,
            upper_predictions=score + 0.10,
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
        )

        self.assertAlmostEqual(float(metrics["ic_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["ic_std"]), 1.41421356237, places=6)
        self.assertAlmostEqual(float(metrics["ic_ir"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["ic_t_stat"]), 0.0, places=6)
        self.assertIn("spread_mean", metrics)
        self.assertIn("spread_t_stat", metrics)

    def test_quantile_calibration_keys_follow_available_bounds(self):
        actual = torch.tensor([[-0.20], [0.00], [0.05], [0.20]], dtype=torch.float32)
        metrics = summarize_forecast_metrics(
            metadata=None,
            line_predictions=torch.zeros((4, 1), dtype=torch.float32),
            lower_predictions=torch.full((4, 1), -0.10, dtype=torch.float32),
            upper_predictions=torch.full((4, 1), 0.10, dtype=torch.float32),
            line_targets=actual,
            band_targets=actual,
            raw_future_returns=actual,
            q_low=0.25,
            q_high=0.75,
        )

        self.assertAlmostEqual(float(metrics["empirical_q_low"]), 0.25, places=6)
        self.assertAlmostEqual(float(metrics["empirical_q_high"]), 0.75, places=6)
        self.assertAlmostEqual(float(metrics["empirical_p25"]), 0.25, places=6)
        self.assertAlmostEqual(float(metrics["empirical_p75"]), 0.75, places=6)
        self.assertIsNone(metrics["empirical_p10"])
        self.assertIsNone(metrics["empirical_p50"])
        self.assertIsNone(metrics["empirical_p90"])


if __name__ == "__main__":
    unittest.main()
