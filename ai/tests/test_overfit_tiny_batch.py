import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd

from ai.loss import ForecastCompositeLoss
from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.models.tide import TiDE
from ai.preprocessing import DatasetPlan, FUTURE_COVARIATE_DIM, SequenceDatasetBundle
from ai.train import TrainConfig, build_lr_lambda, compute_grad_norm, train


def _make_tiny_batch():
    torch.manual_seed(0)
    features = torch.randn(8, 12, 3)
    base_target = features[:, -1, 0].unsqueeze(-1)
    line_target = torch.cat((base_target, base_target * 0.5), dim=1)
    band_target = line_target.clone()
    future_covariates = torch.zeros(8, 2, FUTURE_COVARIATE_DIM)
    return features, line_target, band_target, future_covariates


def _fit_model(model, *, future_covariate=None):
    features, line_target, band_target, future_covariates = _make_tiny_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = ForecastCompositeLoss()
    start_loss = None
    end_loss = None

    for _ in range(200):
        optimizer.zero_grad()
        if future_covariate is None:
            output = model(features)
        else:
            output = model(features, future_covariate=future_covariates)
        losses = criterion(output, line_target, band_target)
        if start_loss is None:
            start_loss = float(losses.total.item())
        losses.total.backward()
        optimizer.step()
        end_loss = float(losses.total.item())

    return start_loss, end_loss


def _dummy_bundle() -> SequenceDatasetBundle:
    features = torch.randn(8, 16, 36)
    line_targets = torch.randn(8, 2)
    band_targets = line_targets.clone()
    anchor_closes = torch.full((8,), 100.0)
    ticker_ids = torch.zeros(8, dtype=torch.long)
    future_covariates = torch.zeros(8, 2, FUTURE_COVARIATE_DIM)
    metadata = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 8,
            "timeframe": ["1D"] * 8,
            "asof_date": [f"2026-01-{idx + 1:02d}" for idx in range(8)],
            "forecast_dates": [[f"2026-02-{idx + 1:02d}", f"2026-02-{idx + 2:02d}"] for idx in range(8)],
            "sample_index": list(range(8)),
        }
    )
    return SequenceDatasetBundle(
        features=features,
        line_targets=line_targets,
        band_targets=band_targets,
        raw_future_returns=band_targets.clone(),
        anchor_closes=anchor_closes,
        ticker_ids=ticker_ids,
        future_covariates=future_covariates,
        metadata=metadata,
    )


class OverfitTinyBatchTestCase(unittest.TestCase):
    def test_lr_scheduler_warmup_then_cosine(self):
        lr_lambda = build_lr_lambda(total_steps=100, warmup_frac=0.1, schedule="cosine")
        values = [lr_lambda(step) for step in range(100)]
        self.assertLess(values[0], values[5])
        self.assertLess(values[5], values[9])
        self.assertAlmostEqual(values[9], 1.0, places=5)
        self.assertLess(values[-1], values[20])
        self.assertGreaterEqual(values[-1], 0.01)

    def test_grad_clip_caps_norm(self):
        model = torch.nn.Linear(4, 2)
        x = torch.full((8, 4), 100.0)
        y = model(x).sum()
        y.backward()
        unclipped = compute_grad_norm(model.parameters())
        self.assertGreater(unclipped, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        clipped = compute_grad_norm(model.parameters())
        self.assertLessEqual(clipped, 1.00001)

    def test_overfit_tiny_batch_patchtst(self):
        model = PatchTST(
            n_features=3,
            seq_len=12,
            patch_len=4,
            stride=2,
            d_model=8,
            n_heads=2,
            n_layers=1,
            horizon=2,
            dropout=0.0,
            use_revin=False,
        )
        start_loss, end_loss = _fit_model(model)
        self.assertLess(end_loss, start_loss)
        self.assertLess(end_loss, 0.1)

    def test_overfit_tiny_batch_cnn_lstm(self):
        model = CNNLSTM(
            n_features=3,
            seq_len=12,
            cnn_channels=8,
            lstm_hidden=12,
            n_layers=1,
            horizon=2,
            dropout=0.0,
        )
        start_loss, end_loss = _fit_model(model)
        self.assertLess(end_loss, start_loss)
        self.assertLess(end_loss, 0.1)

    def test_overfit_tiny_batch_tide(self):
        model = TiDE(
            n_features=3,
            seq_len=12,
            feature_dim=4,
            enc_dim=16,
            dec_dim=8,
            n_enc_layers=1,
            n_dec_layers=1,
            horizon=2,
            dropout=0.0,
            future_cov_dim=FUTURE_COVARIATE_DIM,
        )
        start_loss, end_loss = _fit_model(model, future_covariate=torch.zeros(8, 2, FUTURE_COVARIATE_DIM))
        self.assertLess(end_loss, start_loss)
        self.assertLess(end_loss, 0.1)

    def test_grad_norm_logging_present(self):
        bundle = _dummy_bundle()
        plan = DatasetPlan(
            timeframe="1D",
            seq_len=16,
            horizon=2,
            h_max=20,
            min_fold_samples=1,
            input_ticker_count=1,
            eligible_tickers=["AAPL"],
            excluded_reasons={},
            split_specs={},
            ticker_registry_path="ai/cache/ticker_id_map_1d.json",
            num_tickers=1,
        )
        config = TrainConfig(
            model="patchtst",
            timeframe="1D",
            horizon=2,
            seq_len=16,
            epochs=1,
            batch_size=4,
            lr=1e-3,
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
        dropout=0.0,
            band_mode="direct",
            num_tickers=1,
            ticker_emb_dim=32,
            ci_aggregate="target",
            target_channel_idx=0,
            future_cov_dim=FUTURE_COVARIATE_DIM,
            use_future_covariate=False,
            line_target_type="raw_future_return",
            band_target_type="raw_future_return",
            ticker_registry_path="ai/cache/ticker_id_map_1d.json",
            tickers=["AAPL"],
            limit_tickers=None,
            seed=42,
            device="cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        use_wandb=False,
            wandb_project="lens-ai",
            model_ver="v2-multihead",
            early_stop_patience=0,
            early_stop_min_delta=1e-4,
        )
        captured = {}

        def _fake_save_checkpoint(model, config, run_id, metrics, feature_mean, feature_std):
            captured["metrics"] = metrics
            return Path(tempfile.gettempdir()) / f"{run_id}.pt"

        with patch("ai.train.prepare_dataset_splits", return_value=(bundle, bundle, bundle, torch.zeros(36), torch.ones(36), plan)):
            with patch("ai.train.save_checkpoint", side_effect=_fake_save_checkpoint):
                with patch("ai.train.maybe_compile_model", side_effect=lambda model, device: model):
                    result = train(config, save_run=False)

        self.assertIn("grad_norm_history", result["best_metrics"])
        self.assertIn("best_grad_norm_mean", result["best_metrics"])
        self.assertIn("grad_norm_history", captured["metrics"])
        self.assertEqual(len(captured["metrics"]["grad_norm_history"]), 1)


if __name__ == "__main__":
    unittest.main()
