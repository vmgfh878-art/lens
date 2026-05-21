import unittest

from ai.cp70_lm_h20_display_calibration_policy import _display_decision


def _policy(false_safe, severe, h11_false_safe, ic=0.05, spread=0.01, upside=0.12):
    return {
        "test": {
            "line_metrics": {
                "ic_mean": ic,
                "long_short_spread": spread,
                "false_safe_tail_rate": false_safe,
                "severe_downside_recall": severe,
                "upside_sacrifice": upside,
            },
            "bucket_line_metrics": {
                "h11_h20": {
                    "false_safe_tail_rate": h11_false_safe,
                }
            },
        }
    }


class CP70DisplayPolicyTests(unittest.TestCase):
    def test_default_off_candidate_when_all_risk_and_signal_criteria_pass(self):
        raw = _policy(0.39, 0.62, 0.38, upside=0.11)
        candidate = _policy(0.28, 0.72, 0.26, upside=0.12)

        decision = _display_decision(candidate, raw)

        self.assertEqual(decision["verdict"], "default_off_candidate")
        self.assertTrue(decision["criteria"]["false_safe_tail_lt_0_30"])
        self.assertTrue(decision["criteria"]["severe_recall_ge_0_70"])
        self.assertTrue(decision["criteria"]["h11_false_safe_lt_0_35"])
        self.assertTrue(decision["criteria"]["ic_and_spread_positive"])

    def test_fail_when_h11_false_safe_collapses(self):
        raw = _policy(0.39, 0.62, 0.38, upside=0.11)
        candidate = _policy(0.28, 0.72, 0.36, upside=0.12)

        decision = _display_decision(candidate, raw)

        self.assertEqual(decision["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
