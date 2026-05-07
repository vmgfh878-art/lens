import unittest

from ai.cp114_lm_1w_line_candidate_expansion import classify_candidate, compare_metric


class CP114LMLineCandidateExpansionTest(unittest.TestCase):
    def test_product_line_candidate_requires_rank_and_risk(self):
        result = classify_candidate(
            {
                "ic_mean": 0.01,
                "long_short_spread": 0.02,
                "false_safe_tail_rate": 0.19,
                "severe_downside_recall": 0.76,
                "conservative_bias": -0.10,
            },
            line_gate_pass=True,
            horizon=4,
        )

        self.assertEqual(result, "product_line_candidate")

    def test_risk_only_candidate_allows_weak_rank(self):
        result = classify_candidate(
            {
                "ic_mean": -0.01,
                "long_short_spread": -0.02,
                "false_safe_tail_rate": 0.18,
                "severe_downside_recall": 0.82,
                "conservative_bias": -0.10,
            },
            line_gate_pass=True,
            horizon=4,
        )

        self.assertEqual(result, "risk_only_candidate")

    def test_h8_is_recorded_as_feasibility_watch(self):
        result = classify_candidate(
            {
                "ic_mean": 0.02,
                "long_short_spread": 0.02,
                "false_safe_tail_rate": 0.19,
                "severe_downside_recall": 0.80,
                "conservative_bias": -0.10,
            },
            line_gate_pass=True,
            horizon=8,
        )

        self.assertEqual(result, "h8_feasibility_product_watch")

    def test_compare_metric_lower_false_safe_improves(self):
        result = compare_metric("false_safe_tail_rate", 0.18, 0.20)

        self.assertEqual(result["direction"], "improved")
        self.assertAlmostEqual(result["delta"], -0.02)


if __name__ == "__main__":
    unittest.main()
