import unittest

from ai.cp75_lm_1d_h5_full_line_product_candidate import (
    classify_line_candidate,
    extract_required_line_metrics,
)


class CP75LineProductCandidateTestCase(unittest.TestCase):
    def test_extract_required_line_metrics_prefers_nested_line_metrics(self):
        metrics = {
            "ic_mean": -1.0,
            "line_metrics": {
                "ic_mean": 0.05,
                "long_short_spread": 0.02,
                "false_safe_tail_rate": 0.2,
                "severe_downside_recall": 0.8,
            },
        }

        result = extract_required_line_metrics(metrics)

        self.assertEqual(result["ic_mean"], 0.05)
        self.assertEqual(result["long_short_spread"], 0.02)
        self.assertEqual(result["false_safe_tail_rate"], 0.2)
        self.assertEqual(result["severe_downside_recall"], 0.8)

    def test_classify_product_ready_requires_storage_and_line_risk_metrics(self):
        metrics = {
            "ic_mean": 0.03,
            "long_short_spread": 0.04,
            "false_safe_tail_rate": 0.25,
            "severe_downside_recall": 0.72,
        }

        self.assertEqual(classify_line_candidate(metrics, True), "product_candidate_ready")
        self.assertEqual(classify_line_candidate(metrics, False), "storage_incomplete")

    def test_classify_watch_when_ranking_survives_but_risk_threshold_is_weak(self):
        metrics = {
            "ic_mean": 0.03,
            "long_short_spread": 0.04,
            "false_safe_tail_rate": 0.34,
            "severe_downside_recall": 0.62,
        }

        self.assertEqual(classify_line_candidate(metrics, True), "completed_line_watch")


if __name__ == "__main__":
    unittest.main()
