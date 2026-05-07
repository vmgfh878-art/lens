import unittest

from ai.cp136_lm_1w_line_horizon_feature_expansion import compare_metric, _product_ok, _h8_watch


class CP136LMLineHorizonFeatureExpansionTest(unittest.TestCase):
    def test_product_ok_uses_validation_metrics(self):
        candidate = {
            "status": "PASS",
            "config": {"horizon": 4},
            "gate_status": {"line_gate_pass": True},
            "validation_line_metrics": {
                "ic_mean": 0.01,
                "long_short_spread": 0.02,
                "fee_adjusted_return": 0.03,
                "false_safe_tail_rate": 0.20,
                "severe_downside_recall": 0.75,
                "conservative_bias": -0.10,
            },
        }

        self.assertTrue(_product_ok(candidate))

    def test_h8_watch_accepts_weak_rank_when_risk_metrics_hold(self):
        candidate = {
            "status": "PASS",
            "config": {"horizon": 8},
            "gate_status": {"line_gate_pass": True},
            "validation_line_metrics": {
                "ic_mean": -0.01,
                "long_short_spread": -0.02,
                "false_safe_tail_rate": 0.24,
                "severe_downside_recall": 0.76,
            },
        }

        self.assertTrue(_h8_watch(candidate))

    def test_false_safe_delta_lower_is_improved(self):
        result = compare_metric("false_safe_tail_rate", 0.18, 0.22)

        self.assertEqual(result["direction"], "improved")
        self.assertAlmostEqual(result["delta"], -0.04)


if __name__ == "__main__":
    unittest.main()
