import unittest

import torch

from ai.loss import AsymmetricBCELoss, PinballLoss


class PinballAndBCELossTestCase(unittest.TestCase):
    def test_pinball_loss_matches_hand_calculation(self):
        loss_fn = PinballLoss((0.1, 0.5, 0.9), sort_quantiles=True)
        prediction = torch.tensor([[[0.3, 0.1, 0.2]]], dtype=torch.float32)
        target = torch.tensor([[0.25]], dtype=torch.float32)

        loss = loss_fn(prediction, target).item()

        expected = (0.015 + 0.025 + 0.005) / 3.0
        self.assertAlmostEqual(loss, expected, places=6)

    def test_asymmetric_bce_defaults_penalize_over_prediction_more(self):
        loss_fn = AsymmetricBCELoss()
        self.assertEqual(loss_fn.alpha, 1.0)
        self.assertEqual(loss_fn.beta, 2.0)

        over = loss_fn(torch.tensor([[0.9]], dtype=torch.float32), torch.tensor([[0.0]], dtype=torch.float32)).item()
        under = loss_fn(torch.tensor([[0.1]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float32)).item()
        self.assertGreater(over, under)


if __name__ == "__main__":
    unittest.main()
