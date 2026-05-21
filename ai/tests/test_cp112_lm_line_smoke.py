import unittest

from ai.cp112_lm_1w_line_smoke import classify_smoke_result, extract_line_metrics


class CP112LMLineSmokeTest(unittest.TestCase):
    def test_classify_smoke_pass_does_not_judge_performance(self):
        status, failure = classify_smoke_result(
            preflight_pass=True,
            exit_code=0,
            line_metrics_present=True,
            h4_shape_ok=True,
        )

        self.assertEqual(status, "PASS")
        self.assertIsNone(failure)

    def test_classify_smoke_separates_data_failure(self):
        status, failure = classify_smoke_result(
            preflight_pass=False,
            exit_code=0,
            line_metrics_present=True,
            h4_shape_ok=True,
        )

        self.assertEqual(status, "FAIL")
        self.assertEqual(failure, "split_data_failure")

    def test_extract_line_metrics_prefers_nested_line_metrics(self):
        metrics = {
            "ic_mean": -1.0,
            "line_metrics": {
                "ic_mean": 0.12,
                "long_short_spread": 0.03,
                "false_safe_tail_rate": 0.5,
            },
        }

        extracted = extract_line_metrics(metrics)

        self.assertEqual(extracted["ic_mean"], 0.12)
        self.assertEqual(extracted["long_short_spread"], 0.03)
        self.assertEqual(extracted["false_safe_tail_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
