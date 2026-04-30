import unittest

from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.preprocessing import FUTURE_COVARIATE_DIM
from ai.train import TrainConfig, build_model


def _config(**overrides):
    base = dict(
        model="patchtst",
        timeframe="1D",
        horizon=5,
        seq_len=252,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=1.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.1,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=42,
        device="cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-ai",
        model_ver="v2-multihead",
        early_stop_patience=0,
        early_stop_min_delta=1e-4,
    )
    base.update(overrides)
    return TrainConfig(**base)


class PatchTSTCliConfigTestCase(unittest.TestCase):
    def test_patchtst_defaults_match_model_defaults(self):
        model = build_model(_config())
        self.assertIsInstance(model, PatchTST)
        self.assertEqual(model.patch_len, 16)
        self.assertEqual(model.stride, 8)
        self.assertEqual(model.n_patches, 30)

    def test_patchtst_geometry_args_are_applied(self):
        model = build_model(
            _config(
                patch_len=32,
                patch_stride=8,
                patchtst_d_model=64,
                patchtst_n_heads=4,
                patchtst_n_layers=2,
            )
        )
        self.assertEqual(model.patch_len, 32)
        self.assertEqual(model.stride, 8)
        self.assertEqual(model.n_patches, 28)
        self.assertEqual(model.patch_proj.out_features, 64)
        self.assertEqual(len(model.encoder.layers), 2)

    def test_patchtst_revin_flag_is_applied(self):
        model = build_model(_config(use_revin=False))
        self.assertIsInstance(model, PatchTST)
        self.assertFalse(model.use_revin)
        self.assertIsNone(model.revin)

    def test_patchtst_args_do_not_affect_cnn_lstm(self):
        model = build_model(
            _config(
                model="cnn_lstm",
                patch_len=32,
                patch_stride=4,
                patchtst_d_model=64,
                patchtst_n_heads=4,
                patchtst_n_layers=2,
            )
        )
        self.assertIsInstance(model, CNNLSTM)
        self.assertFalse(hasattr(model, "patch_len"))


if __name__ == "__main__":
    unittest.main()
