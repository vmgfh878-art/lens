import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd

from ai.preprocessing import DatasetPlan, FUTURE_COVARIATE_DIM, SequenceDatasetBundle
from ai.sweep import parse_args, run_sweep
from ai.train import TrainConfig, run_training


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


def _dummy_plan() -> DatasetPlan:
    return DatasetPlan(
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


def _dummy_config() -> TrainConfig:
    return TrainConfig(
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


def _base_args(**overrides):
    base = argparse.Namespace(
        study_name="test_sweep_cache",
        storage_url="sqlite:///:memory:",
        n_trials=5,
        model="patchtst",
        timeframe="1D",
        seq_len=16,
        horizon=2,
        max_epoch=2,
        batch_size=4,
        seed=42,
        limit_tickers=50,
        device="auto",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_wandb=False,
        wandb_project="lens-ai",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class SweepCachingTestCase(unittest.TestCase):
    def test_run_training_uses_precomputed_bundles(self):
        bundle = _dummy_bundle()
        plan = _dummy_plan()
        config = _dummy_config()
        precomputed = (bundle, bundle, bundle, torch.zeros(36), torch.ones(36), plan)

        with patch("ai.train.prepare_dataset_splits", side_effect=AssertionError("호출되면 안 됩니다.")):
            with patch("ai.train.maybe_compile_model", side_effect=lambda model, device: model):
                with patch(
                    "ai.train.save_checkpoint",
                    return_value=Path(tempfile.gettempdir()) / "checkpoint.pt",
                ):
                    result = run_training(config, save_run=False, precomputed_bundles=precomputed)

        self.assertEqual(result["dataset_plan"]["eligible_ticker_count"], 1)

    def test_sweep_builds_dataset_once(self):
        bundle = _dummy_bundle()
        plan = _dummy_plan()
        precomputed = (bundle, bundle, bundle, torch.zeros(36), torch.ones(36), plan)

        with patch("ai.sweep.prepare_dataset_splits", return_value=precomputed) as prepare_mock:
            with patch(
                "ai.sweep.run_training",
                return_value={
                    "best_metrics": {"best_val_total": 0.25},
                    "run_id": "run-1",
                    "checkpoint_path": "checkpoint.pt",
                    "test_metrics": {"coverage": 0.5},
                },
            ) as run_training_mock:
                with patch("ai.sweep.save_study_plots", return_value=[]):
                    with patch("ai.sweep.maybe_log_summary_to_wandb"):
                        summary = run_sweep(_base_args())

        self.assertEqual(prepare_mock.call_count, 1)
        self.assertEqual(run_training_mock.call_count, 5)
        for call in run_training_mock.call_args_list:
            self.assertIs(call.kwargs["precomputed_bundles"], precomputed)
        self.assertIn("dataset_build_seconds", summary)

    def test_failed_trial_propagates_exception(self):
        bundle = _dummy_bundle()
        plan = _dummy_plan()
        precomputed = (bundle, bundle, bundle, torch.zeros(36), torch.ones(36), plan)

        with patch("ai.sweep.prepare_dataset_splits", return_value=precomputed):
            with patch("ai.sweep.run_training", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    run_sweep(_base_args(n_trials=1))

    def test_limit_tickers_default_100(self):
        with patch.object(sys, "argv", ["ai.sweep", "--study-name", "sample"]):
            args = parse_args()
        self.assertEqual(args.limit_tickers, 100)


if __name__ == "__main__":
    unittest.main()
