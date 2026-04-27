import argparse
import math
import os
import unittest
from unittest.mock import patch

import optuna

from ai.sweep import build_study, objective_lr_sweep


def _base_args(**overrides):
    base = argparse.Namespace(
        study_name="test_study",
        storage_url="sqlite:///:memory:",
        n_trials=1,
        model="patchtst",
        timeframe="1D",
        seq_len=252,
        horizon=5,
        max_epoch=2,
        batch_size=8,
        seed=42,
        limit_tickers=2,
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


class SweepTestCase(unittest.TestCase):
    def test_build_study_returns_optuna_study(self):
        study = build_study("sample", storage_url="sqlite:///:memory:", max_resource=10)
        self.assertIsInstance(study, optuna.Study)
        self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)
        self.assertIsInstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_objective_runs_with_dummy_args(self):
        study = build_study("dummy_objective", storage_url="sqlite:///:memory:", max_resource=2)
        trial = study.ask()
        with patch(
            "ai.sweep.run_training",
            return_value={
                "best_metrics": {"best_val_total": 0.123},
                "run_id": "run-1",
                "checkpoint_path": "checkpoint.pt",
                "test_metrics": {"coverage": 0.5},
            },
        ):
            value = objective_lr_sweep(trial, _base_args())
        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, 0.123)

    def test_pruner_kills_bad_trial(self):
        study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=1e5), direction="minimize")
        trial = study.ask()
        trial.report(1e6, 5)
        self.assertTrue(trial.should_prune())

    def test_sweep_reproducible_with_seed(self):
        params_list = []

        def _objective(trial: optuna.Trial):
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            }
            params_list.append(params)
            return params["lr"]

        study_a = build_study("repro_a", storage_url="sqlite:///:memory:", max_resource=2)
        study_b = build_study("repro_b", storage_url="sqlite:///:memory:", max_resource=2)
        study_a.optimize(_objective, n_trials=1)
        first = params_list[-1]
        study_b.optimize(_objective, n_trials=1)
        second = params_list[-1]
        self.assertEqual(first, second)

    def test_wandb_disabled_in_test_env(self):
        study = build_study("wandb_off", storage_url="sqlite:///:memory:", max_resource=2)
        trial = study.ask()
        with patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
            with patch(
                "ai.sweep.run_training",
                return_value={
                    "best_metrics": {"best_val_total": 0.456},
                    "run_id": "run-2",
                    "checkpoint_path": "checkpoint.pt",
                    "test_metrics": {"coverage": 0.7},
                },
            ):
                value = objective_lr_sweep(trial, _base_args(use_wandb=True))
        self.assertEqual(value, 0.456)


if __name__ == "__main__":
    unittest.main()
