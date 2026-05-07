import unittest

import torch

from ai.cp68_lm_h20_conservative_line_rescue import _apply_bucket_offsets, _fit_min_offset


class CP68ConservativeCalibrationTests(unittest.TestCase):
    def test_fit_min_offset_returns_zero_when_target_already_met(self):
        result = _fit_min_offset(
            metric_at_offset=lambda offset: 0.25,
            high=1.0,
            target=0.30,
        )

        self.assertEqual(result["offset"], 0.0)
        self.assertTrue(result["target_met"])

    def test_fit_min_offset_finds_minimal_monotonic_offset(self):
        result = _fit_min_offset(
            metric_at_offset=lambda offset: max(0.0, 0.50 - float(offset)),
            high=1.0,
            target=0.30,
        )

        self.assertTrue(result["target_met"])
        self.assertGreaterEqual(float(result["offset"]), 0.20)
        self.assertLess(float(result["offset"]), 0.201)

    def test_apply_bucket_offsets_only_shifts_matching_horizon_ranges(self):
        line = torch.zeros((2, 20), dtype=torch.float32)
        adjusted = _apply_bucket_offsets(
            line,
            {"h1_h5": 0.1, "h6_h10": 0.2, "h11_h20": 0.3},
        )

        self.assertTrue(torch.allclose(adjusted[:, 0:5], torch.full((2, 5), -0.1)))
        self.assertTrue(torch.allclose(adjusted[:, 5:10], torch.full((2, 5), -0.2)))
        self.assertTrue(torch.allclose(adjusted[:, 10:20], torch.full((2, 10), -0.3)))
        self.assertTrue(torch.allclose(line, torch.zeros((2, 20), dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
