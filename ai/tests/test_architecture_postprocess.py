import unittest

import torch

from ai.loss import ForecastCompositeLoss, WidthPenaltyLoss
from ai.models.blocks import AttentionPooling1D, ResidualBlock
from ai.models.common import MultiHeadForecastModel
from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.models.revin import RevIN
from ai.models.tide import TiDE
from ai.postprocess import apply_band_postprocess


class _DummyHead(MultiHeadForecastModel):
    def forward(self, hidden: torch.Tensor):
        return self.build_output(hidden)


class ArchitecturePostprocessTestCase(unittest.TestCase):
    def test_postprocess_preserves_line_head(self):
        line = torch.tensor([[0.1, -0.2]])
        lower = torch.tensor([[0.3, -0.1]])
        upper = torch.tensor([[0.2, -0.3]])
        line_pp, _, _ = apply_band_postprocess(line, lower, upper)
        self.assertTrue(torch.equal(line_pp, line))

    def test_postprocess_band_sorted(self):
        line = torch.tensor([[0.1, 0.2]])
        lower = torch.tensor([[0.3, -0.1]])
        upper = torch.tensor([[0.2, -0.3]])
        _, lower_pp, upper_pp = apply_band_postprocess(line, lower, upper)
        self.assertTrue(torch.all(lower_pp <= upper_pp))

    def test_train_val_inference_postprocess_identical(self):
        line = torch.randn(2, 5)
        lower = torch.randn(2, 5)
        upper = torch.randn(2, 5)
        first = apply_band_postprocess(line, lower, upper)
        second = apply_band_postprocess(line, lower, upper)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_width_loss_nonnegative(self):
        loss = WidthPenaltyLoss()
        value = loss(torch.tensor([[1.0, 0.5]]), torch.tensor([[0.5, -0.5]]))
        self.assertGreaterEqual(value.item(), 0.0)

    def test_band_mode_param_lower_le_upper(self):
        model = _DummyHead(hidden_dim=8, horizon=5, band_mode="param")
        hidden = torch.randn(1000, 8)
        output = model(hidden)
        self.assertTrue(torch.all(output.lower_band <= output.upper_band))

    def test_patchtst_forward_shape(self):
        model = PatchTST(n_features=29, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        output = model(torch.randn(4, 32, 29))
        self.assertEqual(output.line.shape, (4, 5))
        self.assertEqual(output.lower_band.shape, (4, 5))
        self.assertEqual(output.upper_band.shape, (4, 5))

    def test_patchtst_revin_normalization(self):
        revin = RevIN(29)
        x = torch.randn(8, 32, 29)
        normalized = revin(x, mode="norm")
        mean = normalized.mean(dim=1)
        std = normalized.std(dim=1, unbiased=False)
        self.assertLess(mean.abs().mean().item(), 1e-4)
        self.assertLess((std - 1.0).abs().mean().item(), 1e-3)

    def test_patchtst_channel_independent_shape(self):
        model = PatchTST(
            n_features=29,
            seq_len=32,
            patch_len=8,
            stride=4,
            d_model=16,
            n_heads=4,
            n_layers=2,
            horizon=5,
            channel_independent=True,
        )
        output = model(torch.randn(2, 32, 29))
        self.assertEqual(output.line.shape, (2, 5))

    def test_cnn_lstm_forward_shape(self):
        model = CNNLSTM(n_features=29, seq_len=32, horizon=5, band_mode="direct")
        output = model(torch.randn(4, 32, 29))
        self.assertEqual(output.line.shape, (4, 5))

    def test_cnn_lstm_attention_pooling_sums_to_one(self):
        pooling = AttentionPooling1D(16)
        _ = pooling(torch.randn(4, 12, 16))
        weights = pooling.last_weights
        self.assertIsNotNone(weights)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)), atol=1e-5))

    def test_tide_forward_shape(self):
        model = TiDE(n_features=29, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        output = model(torch.randn(4, 32, 29))
        self.assertEqual(output.line.shape, (4, 5))

    def test_residual_block_preserves_dim(self):
        block = ResidualBlock(dim=32, dropout=0.2)
        x = torch.randn(4, 32)
        y = block(x)
        self.assertEqual(x.shape, y.shape)

    def test_tide_lookback_skip_contributes(self):
        torch.manual_seed(0)
        model = TiDE(n_features=29, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        base = torch.randn(2, 32, 29)
        shifted = base.clone()
        shifted[..., 0] = shifted[..., 0] + 10.0
        output_base = model(base)
        output_shifted = model(shifted)
        self.assertFalse(torch.allclose(output_base.line, output_shifted.line))

    def test_init_weights_trunc_normal(self):
        model = PatchTST(n_features=29, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        stds = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                stds.append(float(module.weight.std().item()))
        self.assertGreater(len(stds), 0)
        avg_std = sum(stds) / len(stds)
        self.assertGreater(avg_std, 0.015)
        self.assertLess(avg_std, 0.025)

    def test_param_mode_composite_loss_disables_cross_penalty(self):
        criterion = ForecastCompositeLoss(band_mode="param")
        prediction = _DummyHead(hidden_dim=8, horizon=5, band_mode="param")(torch.randn(2, 8))
        breakdown = criterion(prediction, torch.randn(2, 5), torch.randn(2, 5))
        self.assertEqual(float(breakdown.cross.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
