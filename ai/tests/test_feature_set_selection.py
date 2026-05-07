import json
from pathlib import Path
import tempfile
import unittest

import torch
import pandas as pd

from ai.models.cnn_lstm import CNNLSTM
from ai.models.tide import TiDE
from ai.preprocessing import FUTURE_COVARIATE_DIM, MODEL_FEATURE_COLUMNS, SequenceDatasetBundle
from ai.train import (
    TrainConfig,
    apply_feature_columns_to_splits,
    build_model,
    resolve_feature_columns,
)


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
        use_wandb=False,
        wandb_project="lens-ai",
        model_ver="v2-multihead",
        early_stop_patience=0,
        early_stop_min_delta=1e-4,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _plan_file(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    path = Path(handle.name)
    handle.close()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class FeatureSetSelectionTestCase(unittest.TestCase):
    def test_resolve_feature_columns_from_plan(self):
        payload = {
            "feature_sets": {
                "price_volatility": {
                    "columns": ["log_return", "high_ratio", "low_ratio"],
                    "requires_contract_change": False,
                }
            }
        }
        path = _plan_file(payload)

        self.assertEqual(
            resolve_feature_columns("price_volatility", plan_path=path),
            ["log_return", "high_ratio", "low_ratio"],
        )

    def test_indicator_expanded_set_is_rejected(self):
        payload = {
            "feature_sets": {
                "candidate_indicator_expanded": {
                    "columns": ["log_return"],
                    "indicator_only_candidates": ["atr_ratio"],
                    "requires_contract_change": True,
                }
            }
        }
        path = _plan_file(payload)

        with self.assertRaises(ValueError):
            resolve_feature_columns("candidate_indicator_expanded", plan_path=path)

    def test_default_full_features_does_not_require_plan_file(self):
        missing_path = Path(tempfile.gettempdir()) / "missing_feature_set_plan.json"

        self.assertEqual(resolve_feature_columns("full_features", plan_path=missing_path), list(MODEL_FEATURE_COLUMNS))

    def test_build_model_uses_selected_feature_count(self):
        columns = ["log_return", "high_ratio", "low_ratio", "rsi"]
        cnn = build_model(_config(feature_columns=columns, n_features=len(columns)))
        tide = build_model(_config(model="tide", feature_columns=columns, n_features=len(columns), use_future_covariate=True))

        self.assertIsInstance(cnn, CNNLSTM)
        self.assertEqual(cnn.conv_layers[0].in_channels, len(columns))
        self.assertIsInstance(tide, TiDE)
        self.assertEqual(tide.feature_proj.in_features, len(columns))

    def test_apply_feature_columns_to_eager_bundle(self):
        features = torch.arange(2 * 3 * len(MODEL_FEATURE_COLUMNS), dtype=torch.float32).reshape(
            2,
            3,
            len(MODEL_FEATURE_COLUMNS),
        )
        bundle = SequenceDatasetBundle(
            features=features,
            line_targets=torch.zeros(2, 1),
            band_targets=torch.zeros(2, 1),
            raw_future_returns=torch.zeros(2, 1),
            anchor_closes=torch.ones(2),
            ticker_ids=torch.zeros(2, dtype=torch.long),
            future_covariates=torch.zeros(2, 1, 0),
            metadata=pd.DataFrame({"ticker": ["A", "B"], "asof_date": ["2024-01-01", "2024-01-02"]}),
        )
        mean = torch.arange(len(MODEL_FEATURE_COLUMNS), dtype=torch.float32)
        std = torch.ones(len(MODEL_FEATURE_COLUMNS), dtype=torch.float32)
        columns = ["log_return", "low_ratio", "rsi"]

        selected_train, selected_val, selected_test, selected_mean, selected_std = apply_feature_columns_to_splits(
            bundle,
            bundle,
            bundle,
            mean,
            std,
            columns,
        )

        self.assertEqual(selected_train.features.shape[2], len(columns))
        self.assertEqual(selected_val.features.shape[2], len(columns))
        self.assertEqual(selected_test.features.shape[2], len(columns))
        self.assertEqual(selected_mean.shape[0], len(columns))
        self.assertEqual(selected_std.shape[0], len(columns))


if __name__ == "__main__":
    unittest.main()
