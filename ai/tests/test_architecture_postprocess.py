import unittest

import torch

from ai.loss import BandCrossPenaltyLoss, ForecastCompositeLoss
from ai.models.blocks import AttentionPooling1D, ChannelAttentionPooling, ResidualBlock
from ai.models.common import MultiHeadForecastModel
from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.models.revin import RevIN
from ai.models.tide import TiDE
from ai.postprocess import apply_band_postprocess

N_FEATURES = 36
FUTURE_COV_DIM = 7


class _DummyHead(MultiHeadForecastModel):
    def forward(self, hidden: torch.Tensor):
        return self.build_output(hidden)


def _combined_linear_stats(model: torch.nn.Module) -> tuple[float, float]:
    weights = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.numel() >= 64:
            weights.append(module.weight.flatten())
    combined = torch.cat(weights)
    return float(combined.mean().item()), float(combined.std().item())


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

    def test_cross_loss_nonnegative(self):
        loss = BandCrossPenaltyLoss()
        value = loss(torch.tensor([[1.0, 0.5]]), torch.tensor([[0.5, -0.5]]))
        self.assertGreaterEqual(value.item(), 0.0)

    def test_band_mode_param_lower_le_upper(self):
        model = _DummyHead(hidden_dim=8, horizon=5, band_mode="param")
        hidden = torch.randn(1000, 8)
        output = model(hidden)
        self.assertTrue(torch.all(output.lower_band <= output.upper_band))

    def test_patchtst_forward_shape(self):
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        output = model(torch.randn(4, 32, N_FEATURES))
        self.assertEqual(output.line.shape, (4, 5))
        self.assertEqual(output.lower_band.shape, (4, 5))
        self.assertEqual(output.upper_band.shape, (4, 5))

    def test_patchtst_revin_normalization(self):
        revin = RevIN(N_FEATURES)
        x = torch.randn(8, 32, N_FEATURES)
        normalized = revin(x, mode="norm")
        mean = normalized.mean(dim=1)
        std = normalized.std(dim=1, unbiased=False)
        self.assertLess(mean.abs().mean().item(), 1e-4)
        self.assertLess((std - 1.0).abs().mean().item(), 1e-3)

    def test_revin_normalize_denormalize_roundtrip(self):
        revin = RevIN(3)
        x = torch.randn(4, 16, 3)
        normalized = revin(x, mode="norm")
        restored = revin.denormalize_target(normalized[..., 1], 1)
        self.assertTrue(torch.allclose(restored, x[..., 1], atol=1e-5))

    def test_revin_target_channel_selection_changes_output(self):
        revin = RevIN(2)
        x = torch.stack(
            (torch.full((2, 8), 1.0), torch.full((2, 8), 10.0)),
            dim=-1,
        )
        revin(x, mode="norm")
        y = torch.zeros(2, 4)
        restored_first = revin.denormalize_target(y, 0)
        restored_second = revin.denormalize_target(y, 1)
        self.assertFalse(torch.allclose(restored_first, restored_second))

    def test_patchtst_revin_full_pipeline_output_in_target_scale(self):
        torch.manual_seed(0)
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        x = torch.randn(4, 32, N_FEATURES) * 0.02
        output = model(x)
        self.assertTrue(torch.isfinite(output.line).all())
        self.assertLess(output.line.abs().mean().item(), 1.0)

    def test_patchtst_channel_independent_shape(self):
        model = PatchTST(
            n_features=N_FEATURES,
            seq_len=32,
            patch_len=8,
            stride=4,
            d_model=16,
            n_heads=4,
            n_layers=2,
            horizon=5,
            channel_independent=True,
        )
        output = model(torch.randn(2, 32, N_FEATURES))
        self.assertEqual(output.line.shape, (2, 5))

    def test_ci_aggregate_target_equals_target_channel_only(self):
        model = PatchTST(n_features=3, seq_len=16, patch_len=4, stride=4, d_model=8, n_heads=2, n_layers=1, horizon=2, ci_aggregate="target", target_channel_idx=1, use_revin=False)
        hidden = torch.randn(2, 3, 4)
        self.assertTrue(torch.equal(model._aggregate_channels(hidden), hidden[:, 1, :]))

    def test_ci_aggregate_mean_matches_legacy_behavior(self):
        model = PatchTST(n_features=3, seq_len=16, patch_len=4, stride=4, d_model=8, n_heads=2, n_layers=1, horizon=2, ci_aggregate="mean", use_revin=False)
        hidden = torch.randn(2, 3, 4)
        self.assertTrue(torch.allclose(model._aggregate_channels(hidden), hidden.mean(dim=1)))

    def test_ci_aggregate_attention_weights_sum_to_one(self):
        pooling = ChannelAttentionPooling(5)
        _ = pooling(torch.randn(4, 5, 8))
        self.assertIsNotNone(pooling.last_weights)
        self.assertTrue(torch.allclose(pooling.last_weights.sum(dim=1), torch.ones(4), atol=1e-5))

    def test_cnn_lstm_forward_shape(self):
        model = CNNLSTM(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct")
        output = model(torch.randn(4, 32, N_FEATURES))
        self.assertEqual(output.line.shape, (4, 5))
        self.assertIsNone(output.direction_logit)

    def test_cnn_lstm_direction_head_shape(self):
        model = CNNLSTM(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct", use_direction_head=True)
        output = model(torch.randn(4, 32, N_FEATURES))
        self.assertIsNotNone(output.direction_logit)
        self.assertEqual(output.direction_logit.shape, (4, 5))

    def test_cnn_lstm_attention_pooling_sums_to_one(self):
        pooling = AttentionPooling1D(16)
        _ = pooling(torch.randn(4, 12, 16))
        weights = pooling.last_weights
        self.assertIsNotNone(weights)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)), atol=1e-5))

    def test_cnn_lstm_receptive_field(self):
        torch.manual_seed(0)
        model = CNNLSTM(n_features=2, seq_len=64, cnn_channels=4, lstm_hidden=8, horizon=3, dropout=0.0)
        model.eval()

        base = torch.zeros(1, 64, 2)
        near = base.clone()
        far = base.clone()
        near[:, 15, 0] = 1.0
        far[:, 16, 0] = 1.0

        base_conv = model._forward_conv_stack(base)
        near_conv = model._forward_conv_stack(near)
        far_conv = model._forward_conv_stack(far)

        self.assertFalse(torch.allclose(base_conv[:, :, 0], near_conv[:, :, 0]))
        self.assertTrue(torch.allclose(base_conv[:, :, 0], far_conv[:, :, 0], atol=1e-6))
        self.assertEqual(model.receptive_field, 31)

    def test_tide_forward_shape(self):
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        output = model(torch.randn(4, 32, N_FEATURES))
        self.assertEqual(output.line.shape, (4, 5))
        self.assertEqual(model.last_temporal_hidden_shape, (4, 5, 32))

    def test_residual_block_preserves_dim(self):
        block = ResidualBlock(dim=32, dropout=0.2)
        x = torch.randn(4, 32)
        y = block(x)
        self.assertEqual(x.shape, y.shape)

    def test_tide_lookback_skip_contributes(self):
        torch.manual_seed(0)
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        base = torch.randn(2, 32, N_FEATURES)
        shifted = base.clone()
        shifted[..., 0] = shifted[..., 0] + 10.0
        output_base = model(base)
        output_shifted = model(shifted)
        self.assertFalse(torch.allclose(output_base.line, output_shifted.line))

    def test_tide_per_horizon_path(self):
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1, num_tickers=3, ticker_emb_dim=8)
        _ = model(torch.randn(4, 32, N_FEATURES), ticker_id=torch.tensor([0, 1, 2, 3]))
        self.assertEqual(model.last_temporal_hidden_shape, (4, 5, 40))

    def test_tide_temporal_decoder_affects_output(self):
        torch.manual_seed(0)
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, band_mode="direct", enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        x = torch.randn(2, 32, N_FEATURES)
        before = model(x).line
        for parameter in model.temporal_decoder.parameters():
            parameter.data = torch.randn_like(parameter)
        after = model(x).line
        self.assertFalse(torch.allclose(before, after))

    def test_tide_future_covariate_affects_output(self):
        torch.manual_seed(0)
        model = TiDE(
            n_features=N_FEATURES,
            seq_len=32,
            horizon=5,
            enc_dim=64,
            dec_dim=32,
            n_enc_layers=2,
            n_dec_layers=1,
            future_cov_dim=FUTURE_COV_DIM,
        )
        x = torch.randn(2, 32, N_FEATURES)
        future_cov = torch.randn(2, 5, FUTURE_COV_DIM)
        before = model(x, future_covariate=future_cov).line
        for parameter in model.temporal_decoder.parameters():
            parameter.data = torch.randn_like(parameter)
        after = model(x, future_covariate=future_cov).line
        self.assertFalse(torch.allclose(before, after))

    def test_tide_without_future_covariate_still_works(self):
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1, future_cov_dim=0)
        output = model(torch.randn(2, 32, N_FEATURES))
        self.assertEqual(output.line.shape, (2, 5))

    def test_n_features_36_forward_all_models(self):
        patch = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        cnn = CNNLSTM(n_features=N_FEATURES, seq_len=32, horizon=5)
        tide = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, future_cov_dim=FUTURE_COV_DIM)
        patch_out = patch(torch.randn(2, 32, N_FEATURES))
        cnn_out = cnn(torch.randn(2, 32, N_FEATURES))
        tide_out = tide(torch.randn(2, 32, N_FEATURES), future_covariate=torch.randn(2, 5, FUTURE_COV_DIM))
        self.assertEqual(patch_out.line.shape, (2, 5))
        self.assertEqual(cnn_out.line.shape, (2, 5))
        self.assertEqual(tide_out.line.shape, (2, 5))

    def test_patchtst_ticker_embedding_changes_output(self):
        torch.manual_seed(0)
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5, num_tickers=3, ticker_emb_dim=8, use_revin=False)
        x = torch.randn(2, 32, N_FEATURES)
        out_a = model(x, ticker_id=torch.tensor([0, 0]))
        out_b = model(x, ticker_id=torch.tensor([1, 1]))
        self.assertFalse(torch.allclose(out_a.line, out_b.line))

    def test_cnn_lstm_ticker_embedding_changes_output(self):
        torch.manual_seed(0)
        model = CNNLSTM(n_features=N_FEATURES, seq_len=32, horizon=5, num_tickers=3, ticker_emb_dim=8)
        x = torch.randn(2, 32, N_FEATURES)
        out_a = model(x, ticker_id=torch.tensor([0, 0]))
        out_b = model(x, ticker_id=torch.tensor([1, 1]))
        self.assertFalse(torch.allclose(out_a.line, out_b.line))

    def test_tide_ticker_embedding_changes_output(self):
        torch.manual_seed(0)
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1, num_tickers=3, ticker_emb_dim=8)
        x = torch.randn(2, 32, N_FEATURES)
        out_a = model(x, ticker_id=torch.tensor([0, 0]))
        out_b = model(x, ticker_id=torch.tensor([1, 1]))
        self.assertFalse(torch.allclose(out_a.line, out_b.line))

    def test_ticker_embedding_oov_uses_unknown_id(self):
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5, num_tickers=3, ticker_emb_dim=8, use_revin=False)
        x = torch.randn(1, 32, N_FEATURES)
        output = model(x, ticker_id=torch.tensor([3]))
        self.assertEqual(output.line.shape, (1, 5))

    def test_model_without_ticker_embedding_backward_compat(self):
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5, num_tickers=0, use_revin=False)
        x = torch.randn(2, 32, N_FEATURES)
        output = model(x)
        self.assertEqual(output.line.shape, (2, 5))

    def test_init_weights_trunc_normal_patchtst(self):
        model = PatchTST(n_features=N_FEATURES, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=4, n_layers=2, horizon=5)
        mean, std = _combined_linear_stats(model)
        self.assertLess(abs(mean), 0.005)
        self.assertLess(abs(std - 0.02) / 0.02, 0.05)

    def test_init_weights_trunc_normal_cnn_lstm(self):
        model = CNNLSTM(n_features=N_FEATURES, seq_len=32, horizon=5)
        mean, std = _combined_linear_stats(model)
        self.assertLess(abs(mean), 0.005)
        self.assertLess(abs(std - 0.02) / 0.02, 0.05)

    def test_init_weights_trunc_normal_tide(self):
        model = TiDE(n_features=N_FEATURES, seq_len=32, horizon=5, enc_dim=64, dec_dim=32, n_enc_layers=2, n_dec_layers=1)
        mean, std = _combined_linear_stats(model)
        self.assertLess(abs(mean), 0.005)
        self.assertLess(abs(std - 0.02) / 0.02, 0.05)

    def test_param_mode_composite_loss_disables_cross_penalty(self):
        criterion = ForecastCompositeLoss(band_mode="param")
        prediction = _DummyHead(hidden_dim=8, horizon=5, band_mode="param")(torch.randn(2, 8))
        breakdown = criterion(prediction, torch.randn(2, 5), torch.randn(2, 5))
        self.assertEqual(float(breakdown.cross.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
