import argparse
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna

from ai.preprocessing import FUTURE_COVARIATE_DIM
from ai.sweep import objective_lr_sweep
from ai.train import (
    TrainConfig,
    WANDB_STATUS_DISABLED_BY_ENV,
    WANDB_STATUS_DISABLED_INIT_FAILED,
    WANDB_STATUS_DISABLED_MISSING_KEY,
    WANDB_STATUS_ONLINE_OK,
    WANDB_STATUS_PACKAGE_MISSING,
    init_wandb_for_training,
    init_wandb_run,
)


def _missing_dotenv_path() -> Path:
    return Path(__file__).with_name("missing_wandb_test.env")


def _config(**overrides):
    base = dict(
        model="cnn_lstm",
        timeframe="1D",
        horizon=5,
        seq_len=60,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=0.15,
        q_high=0.85,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
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
        use_wandb=True,
        wandb_project="lens-ai",
        model_ver="v2-multihead",
        early_stop_patience=0,
        early_stop_min_delta=1e-4,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _sweep_args(**overrides):
    base = dict(
        study_name="wandb_required_test",
        storage_url="sqlite:///:memory:",
        n_trials=1,
        model="patchtst",
        timeframe="1D",
        seq_len=60,
        horizon=5,
        max_epoch=1,
        batch_size=4,
        seed=42,
        limit_tickers=10,
        device="cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_wandb=True,
        wandb_project="lens-ai",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class WandbOptionalFallbackTestCase(unittest.TestCase):
    def test_missing_key_returns_none_without_exception(self):
        fake_wandb = SimpleNamespace(init=Mock())
        with patch("ai.train.wandb", fake_wandb), patch.dict("os.environ", {}, clear=True):
            outcome = init_wandb_for_training(_config(), "run-1", dotenv_path=_missing_dotenv_path())

        self.assertIsNone(outcome.run)
        self.assertEqual(outcome.status, WANDB_STATUS_DISABLED_MISSING_KEY)
        fake_wandb.init.assert_not_called()

    def test_package_missing_status(self):
        with patch("ai.train.wandb", None), patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
            outcome = init_wandb_for_training(_config(), "run-1", dotenv_path=_missing_dotenv_path())

        self.assertIsNone(outcome.run)
        self.assertEqual(outcome.status, WANDB_STATUS_PACKAGE_MISSING)

    def test_wandb_mode_disabled_status(self):
        fake_wandb = SimpleNamespace(init=Mock())
        with patch("ai.train.wandb", fake_wandb), patch.dict(
            "os.environ",
            {"WANDB_MODE": "disabled", "WANDB_API_KEY": "secret"},
            clear=True,
        ):
            outcome = init_wandb_for_training(_config(), "run-1", dotenv_path=_missing_dotenv_path())

        self.assertIsNone(outcome.run)
        self.assertEqual(outcome.status, WANDB_STATUS_DISABLED_BY_ENV)
        fake_wandb.init.assert_not_called()

    def test_init_exception_falls_back_and_redacts_key(self):
        secret = "super-secret-key"
        fake_wandb = SimpleNamespace(init=Mock(side_effect=RuntimeError(f"bad key {secret}")))
        buffer = io.StringIO()
        with patch("ai.train.wandb", fake_wandb), patch.dict("os.environ", {"WANDB_API_KEY": secret}, clear=True):
            with contextlib.redirect_stdout(buffer):
                outcome = init_wandb_for_training(_config(), "run-1", dotenv_path=_missing_dotenv_path())

        self.assertIsNone(outcome.run)
        self.assertEqual(outcome.status, WANDB_STATUS_DISABLED_INIT_FAILED)
        self.assertNotIn(secret, outcome.message or "")
        self.assertNotIn(secret, buffer.getvalue())

    def test_init_success_returns_online_ok(self):
        run = SimpleNamespace(id="abc123", url="https://wandb.ai/example/run")
        fake_wandb = SimpleNamespace(init=Mock(return_value=run))
        with patch("ai.train.wandb", fake_wandb), patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
            outcome = init_wandb_for_training(_config(), "run-1", dotenv_path=_missing_dotenv_path())

        self.assertIs(outcome.run, run)
        self.assertEqual(outcome.status, WANDB_STATUS_ONLINE_OK)
        self.assertEqual(outcome.run_id, "abc123")
        self.assertEqual(outcome.run_url, "https://wandb.ai/example/run")

    def test_required_sweep_mode_raises_clear_error(self):
        fake_wandb = SimpleNamespace(init=Mock())
        with patch("ai.train.wandb", fake_wandb), patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "W&B required for sweep, set WANDB_API_KEY or use --no-wandb"):
                init_wandb_run(
                    use_wandb=True,
                    project="lens-ai",
                    config_payload={},
                    required=True,
                    dotenv_path=_missing_dotenv_path(),
                )

    def test_sweep_passes_wandb_required_when_enabled(self):
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        dummy_result = {
            "best_metrics": {"best_val_total": 0.1},
            "run_id": "run",
            "checkpoint_path": "checkpoint.pt",
            "test_metrics": {},
        }
        with patch("ai.sweep.run_training", return_value=dummy_result) as run_training_mock:
            value = objective_lr_sweep(
                trial,
                _sweep_args(use_wandb=True),
                precomputed_bundles=("train", "val", "test", "mean", "std", "plan"),
            )

        self.assertEqual(value, 0.1)
        self.assertTrue(run_training_mock.call_args.kwargs["wandb_required"])


if __name__ == "__main__":
    unittest.main()
