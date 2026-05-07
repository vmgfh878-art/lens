import unittest

from ai.cp113_lm_1w_line_rescue import classify_candidate, compare_metric


class CP113LMLineRescueTest(unittest.TestCase):
    def test_return_direction_candidate_requires_positive_rank_metrics(self):
        result = classify_candidate(
            {
                "ic_mean": 0.01,
                "long_short_spread": 0.02,
                "false_safe_tail_rate": 0.20,
                "severe_downside_recall": 0.50,
                "conservative_bias": -0.05,
            },
            line_gate_pass=True,
        )

        self.assertEqual(result, "return_direction_line_candidate")

    def test_risk_candidate_allows_negative_rank_metrics(self):
        result = classify_candidate(
            {
                "ic_mean": -0.01,
                "long_short_spread": -0.02,
                "false_safe_tail_rate": 0.20,
                "severe_downside_recall": 0.75,
                "conservative_bias": -0.10,
            },
            line_gate_pass=True,
        )

        self.assertEqual(result, "risk_conservative_line_candidate")

    def test_compare_metric_false_safe_lower_is_improved(self):
        result = compare_metric("false_safe_tail_rate", 0.20, 0.30)

        self.assertEqual(result["direction"], "improved")
        self.assertAlmostEqual(result["delta"], -0.10)


if __name__ == "__main__":
    unittest.main()
