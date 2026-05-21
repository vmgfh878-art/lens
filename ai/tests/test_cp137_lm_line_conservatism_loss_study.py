import unittest

from ai.cp137_lm_1w_line_conservatism_loss_study import _product_ok, _risk_variant, compare_metric


class CP137LMLineConservatismLossStudyTest(unittest.TestCase):
    def test_product_ok_requires_validation_return_and_risk(self):
        candidate = {
            "status": "PASS",
            "gate_status": {"line_gate_pass": True},
            "validation_line_metrics": {
                "ic_mean": 0.02,
                "long_short_spread": 0.01,
                "fee_adjusted_return": 0.10,
                "false_safe_tail_rate": 0.10,
                "severe_downside_recall": 0.86,
                "conservative_bias": -0.12,
                "upside_sacrifice": 0.20,
            },
        }

        self.assertTrue(_product_ok(candidate))

    def test_risk_variant_rejects_collapsed_rank(self):
        candidate = {
            "status": "PASS",
            "gate_status": {"line_gate_pass": True},
            "validation_line_metrics": {
                "ic_mean": -0.02,
                "long_short_spread": -0.01,
                "false_safe_tail_rate": 0.05,
                "severe_downside_recall": 0.90,
            },
        }

        self.assertFalse(_risk_variant(candidate))

    def test_false_safe_lower_is_improved(self):
        result = compare_metric("false_safe_tail_rate", 0.08, 0.12)

        self.assertEqual(result["direction"], "improved")
        self.assertAlmostEqual(result["delta"], -0.04)


if __name__ == "__main__":
    unittest.main()
