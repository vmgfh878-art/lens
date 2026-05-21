import unittest

import torch

from ai.loss import ForecastCompositeLoss
from ai.models.tcn_quantile import TCNQuantile, _CausalConv1d


class TCNQuantileTest(unittest.TestCase):
    def test_causal_conv_does_not_change_earlier_outputs(self):
        conv = _CausalConv1d(channels=1, kernel_size=3, dilation=2)
        with torch.no_grad():
            conv.conv.weight.fill_(1.0)
            conv.conv.bias.zero_()

        features = torch.zeros(1, 1, 10)
        baseline = conv(features)
        perturbed = features.clone()
        perturbed[..., 7] = 10.0
        changed = conv(perturbed)

        self.assertTrue(torch.allclose(baseline[..., :7], changed[..., :7]))
        self.assertFalse(torch.allclose(baseline[..., 7:], changed[..., 7:]))

    def test_receptive_field_matches_dilation_schedule(self):
        model = TCNQuantile(n_features=36, seq_len=252, horizon=5, dilations=(1, 2, 4, 8))

        self.assertEqual(model.receptive_field, 61)
        self.assertEqual(model.dilations, (1, 2, 4, 8))

    def test_forward_shape_matches_forecast_contract(self):
        model = TCNQuantile(n_features=36, seq_len=60, horizon=5, num_tickers=3, ticker_emb_dim=8)
        features = torch.randn(4, 60, 36)
        ticker_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        output = model(features, ticker_id=ticker_ids)

        self.assertEqual(tuple(output.line.shape), (4, 5))
        self.assertEqual(tuple(output.lower_band.shape), (4, 5))
        self.assertEqual(tuple(output.upper_band.shape), (4, 5))

    def test_forecast_loss_connects_to_direct_quantile_output(self):
        model = TCNQuantile(n_features=11, seq_len=45, horizon=4, band_mode="direct")
        features = torch.randn(3, 45, 11)
        target = torch.randn(3, 4)
        criterion = ForecastCompositeLoss(q_low=0.15, q_high=0.85, lambda_width=0.0)

        output = model(features)
        losses = criterion(output, target, target, target)

        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(torch.isfinite(losses.band))


if __name__ == "__main__":
    unittest.main()
