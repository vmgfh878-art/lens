import unittest

import torch

from ai.loss import AsymmetricHuberLoss, ForecastCompositeLoss, PinballLoss
from ai.models.common import ForecastOutput


class LossTestCase(unittest.TestCase):
    def test_asymmetric_huber_penalizes_over_prediction_more(self):
        loss_fn = AsymmetricHuberLoss(alpha=1.0, beta=2.0, delta=1.0)
        over = loss_fn(torch.tensor([[2.0]]), torch.tensor([[1.0]])).item()
        under = loss_fn(torch.tensor([[0.0]]), torch.tensor([[1.0]])).item()
        self.assertGreater(over, under)
        self.assertAlmostEqual(over, 1.0, places=6)
        self.assertAlmostEqual(under, 0.5, places=6)

    def test_asymmetric_huber_uses_prediction_minus_target_error(self):
        loss_fn = AsymmetricHuberLoss(alpha=1.0, beta=2.0, delta=1.0)

        over_loss = loss_fn(torch.tensor([[1.25]]), torch.tensor([[1.0]])).item()
        under_loss = loss_fn(torch.tensor([[0.75]]), torch.tensor([[1.0]])).item()

        self.assertAlmostEqual(over_loss, 2.0 * 0.5 * (0.25**2), places=6)
        self.assertAlmostEqual(under_loss, 1.0 * 0.5 * (0.25**2), places=6)
        self.assertAlmostEqual(over_loss / under_loss, 2.0, places=6)

    def test_pinball_loss_is_non_negative(self):
        loss_fn = PinballLoss(0.1)
        loss = loss_fn(torch.tensor([[0.2, 0.3]]), torch.tensor([[0.1, 0.5]])).item()
        self.assertGreaterEqual(loss, 0.0)

    def test_composite_loss_returns_all_components(self):
        loss_fn = ForecastCompositeLoss()
        self.assertIsInstance(loss_fn.line_loss, AsymmetricHuberLoss)
        prediction = ForecastOutput(
            line=torch.tensor([[0.01, 0.02]]),
            lower_band=torch.tensor([[-0.01, 0.0]]),
            upper_band=torch.tensor([[0.03, 0.04]]),
        )
        breakdown = loss_fn(
            prediction,
            line_target=torch.tensor([[0.02, 0.01]]),
            band_target=torch.tensor([[0.015, 0.025]]),
        )
        self.assertGreaterEqual(breakdown.total.item(), 0.0)
        self.assertGreaterEqual(breakdown.band.item(), 0.0)
        self.assertGreaterEqual(breakdown.forecast.item(), 0.0)
        self.assertEqual(float(breakdown.direction.item()), 0.0)

    def test_band_loss_weights_adjust_lower_quantile_component(self):
        base_loss = ForecastCompositeLoss(q_low=0.1, q_high=0.9)
        guarded_loss = ForecastCompositeLoss(q_low=0.1, q_high=0.9, lower_band_loss_weight=2.0)
        prediction = ForecastOutput(
            line=torch.tensor([[0.0]]),
            lower_band=torch.tensor([[0.0]]),
            upper_band=torch.tensor([[1.0]]),
        )
        line_target = torch.tensor([[0.0]])
        band_target = torch.tensor([[1.0]])

        base_breakdown = base_loss(prediction, line_target=line_target, band_target=band_target)
        guarded_breakdown = guarded_loss(prediction, line_target=line_target, band_target=band_target)

        self.assertGreater(guarded_breakdown.band.item(), base_breakdown.band.item())
        self.assertGreater(guarded_breakdown.total.item(), base_breakdown.total.item())

    def test_width_loss_removed_from_composite(self):
        loss_fn = ForecastCompositeLoss()
        prediction = ForecastOutput(
            line=torch.tensor([[0.01, 0.02]]),
            lower_band=torch.tensor([[-0.01, 0.0]]),
            upper_band=torch.tensor([[0.03, 0.04]]),
        )
        breakdown = loss_fn(
            prediction,
            line_target=torch.tensor([[0.02, 0.01]]),
            band_target=torch.tensor([[0.015, 0.025]]),
        )
        self.assertFalse(hasattr(breakdown, "width"))
        self.assertNotIn("width_loss", breakdown.to_log_dict())

    def test_direction_loss_added_only_when_direction_head_exists(self):
        loss_fn = ForecastCompositeLoss(lambda_direction=0.1)
        prediction = ForecastOutput(
            line=torch.tensor([[0.01, -0.02]]),
            lower_band=torch.tensor([[-0.01, -0.03]]),
            upper_band=torch.tensor([[0.03, 0.01]]),
            direction_logit=torch.tensor([[0.5, 0.5]]),
        )
        breakdown = loss_fn(
            prediction,
            line_target=torch.tensor([[0.02, -0.01]]),
            band_target=torch.tensor([[0.015, -0.005]]),
            raw_future_returns=torch.tensor([[0.01, -0.02]]),
        )
        self.assertGreater(breakdown.direction.item(), 0.0)
        self.assertGreater(breakdown.total.item(), breakdown.forecast.item())


if __name__ == "__main__":
    unittest.main()
