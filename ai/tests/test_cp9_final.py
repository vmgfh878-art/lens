import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import torch
import pandas as pd

from ai.models.common import ForecastOutput
from ai.models.patchtst import PatchTST
from ai.preprocessing import (
    CALENDAR_FEATURE_COLUMNS,
    FUTURE_COVARIATE_DIM,
    MODEL_N_FEATURES,
    SequenceDataset,
    SequenceDatasetBundle,
    _STALE_CACHE_WARNED,
    build_lazy_sequence_dataset,
    build_sequence_dataset,
    resolve_feature_cache_path,
)
from ai.inference import infer_bundle
from ai.sweep import objective_lr_sweep
from ai.train import evaluate_bundle, make_loader


def _build_feature_rows(length: int = 80, tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_list = tickers or ["AAPL", "MSFT"]
    feature_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()

    for ticker_idx, ticker in enumerate(ticker_list):
        base_price = 100.0 + ticker_idx * 10.0
        for idx, date_value in enumerate(dates):
            feature_rows.append(
                {
                    "ticker": ticker,
                    "timeframe": "1D",
                    "date": date_value,
                    "log_return": 0.01,
                    "open_ratio": 0.01,
                    "high_ratio": 0.02,
                    "low_ratio": -0.01,
                    "vol_change": 0.03,
                    "ma_5_ratio": 0.01,
                    "ma_20_ratio": 0.01,
                    "ma_60_ratio": 0.01,
                    "rsi": 0.5,
                    "macd_ratio": 0.01,
                    "bb_position": 0.5,
                    "us10y": 4.0,
                    "yield_spread": 1.0,
                    "vix_close": 20.0,
                    "credit_spread_hy": 3.0,
                    "nh_nl_index": 0.2,
                    "ma200_pct": 0.7,
                    "regime_calm": 0.0,
                    "regime_neutral": 1.0,
                    "regime_stress": 0.0,
                    "revenue": 100.0,
                    "net_income": 10.0,
                    "equity": 50.0,
                    "eps": 1.0,
                    "roe": 0.2,
                    "debt_ratio": 0.4,
                    "has_macro": True,
                    "has_breadth": True,
                    "has_fundamentals": True,
                }
            )
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": date_value,
                    "close": base_price + idx,
                    "adjusted_close": base_price + idx,
                    "open": base_price + idx,
                    "high": base_price + idx + 1,
                    "low": base_price + idx - 1,
                    "volume": 1000 + idx,
                }
            )

    return pd.DataFrame(feature_rows), pd.DataFrame(price_rows)


def _build_dummy_bundle() -> SequenceDatasetBundle:
    features = torch.randn(8, 16, MODEL_N_FEATURES)
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


def _build_base_args(**overrides) -> argparse.Namespace:
    base = argparse.Namespace(
        study_name="test_cp9",
        storage_url="sqlite:///:memory:",
        n_trials=1,
        model="patchtst",
        timeframe="1D",
        seq_len=16,
        horizon=2,
        max_epoch=2,
        batch_size=4,
        seed=42,
        limit_tickers=100,
        device="cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_wandb=False,
        wandb_project="lens-ai",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class _FakeForecastModel(torch.nn.Module):
    def forward(self, x, ticker_id=None, future_covariate=None):
        batch, _, _ = x.shape
        horizon = future_covariate.shape[1] if future_covariate is not None and future_covariate.numel() else 2
        line = torch.zeros(batch, horizon, dtype=x.dtype, device=x.device)
        return ForecastOutput(
            line=line,
            lower_band=line - 0.1,
            upper_band=line + 0.1,
        )


class CP9FinalTestCase(unittest.TestCase):
    def test_make_loader_windows_falls_back_to_zero_workers(self):
        bundle = _build_dummy_bundle()
        calls: list[int] = []

        class FakeLoader:
            def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
                del dataset, batch_size, shuffle, pin_memory, persistent_workers
                self.num_workers = num_workers
                calls.append(num_workers)

            def __iter__(self):
                if self.num_workers > 0:
                    raise PermissionError(5, "access denied")
                return iter([None])

        with patch("ai.train.DataLoader", FakeLoader):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                loader = make_loader(bundle, batch_size=4, shuffle=False, device=torch.device("cuda"), num_workers=2)

        self.assertEqual(calls, [2, 0])
        self.assertEqual(loader.num_workers, 0)
        self.assertIn("num_workers=0", buffer.getvalue())

    def test_lazy_dataset_memory_single_features_copy(self):
        feature_df, price_df = _build_feature_rows(length=40, tickers=["AAPL", "MSFT", "NVDA"])
        dataset = build_lazy_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=10, horizon=2)

        self.assertIsInstance(dataset, SequenceDataset)
        self.assertEqual(len(dataset.ticker_arrays), 3)
        feature_ptrs = {
            ticker_arrays["features"].__array_interface__["data"][0]
            for ticker_arrays in dataset.ticker_arrays.values()
        }
        self.assertEqual(len(feature_ptrs), 3)
        self.assertEqual(len(dataset.sample_refs), len(dataset.metadata))

    def test_lazy_dataset_correctness_vs_eager(self):
        feature_df, price_df = _build_feature_rows(length=40, tickers=["AAPL", "MSFT"])
        eager = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=10, horizon=2)
        lazy = build_lazy_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=10, horizon=2)

        self.assertEqual(len(eager), len(lazy))
        for index in range(3):
            lazy_x, lazy_line, lazy_band, lazy_raw, lazy_ticker, lazy_future = lazy[index]
            self.assertTrue(torch.allclose(lazy_x, eager.features[index]))
            self.assertTrue(torch.allclose(lazy_line, eager.line_targets[index]))
            self.assertTrue(torch.allclose(lazy_band, eager.band_targets[index]))
            self.assertTrue(torch.allclose(lazy_raw, eager.raw_future_returns[index]))
            self.assertEqual(int(lazy_ticker.item()), int(eager.ticker_ids[index].item()))
            self.assertTrue(torch.allclose(lazy_future, eager.future_covariates[index]))

    def test_evaluate_uses_dataloader_batch_loop(self):
        feature_df, price_df = _build_feature_rows(length=40, tickers=["AAPL"])
        bundle = build_lazy_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=10, horizon=2)
        model = _FakeForecastModel()

        with patch("ai.train.make_loader", wraps=make_loader) as loader_mock:
            metrics = evaluate_bundle(
                model,
                bundle,
                torch.device("cpu"),
                batch_size=4,
                num_workers=0,
            )

        self.assertEqual(loader_mock.call_count, 1)
        self.assertIn("coverage", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("smape", metrics)

    def test_infer_batch_processes_split(self):
        feature_df, price_df = _build_feature_rows(length=40, tickers=["AAPL"])
        bundle = build_lazy_sequence_dataset(
            feature_df,
            price_df,
            timeframe="1D",
            seq_len=10,
            horizon=2,
            include_future_covariate=False,
        ).subset([0, 1, 2])
        checkpoint = {
            "config": {
                "device": "cpu",
                "batch_size": 2,
                "num_workers": 0,
                "use_future_covariate": False,
            }
        }

        with patch("ai.inference.load_checkpoint", return_value=(_FakeForecastModel(), checkpoint)):
            with patch("ai.inference.make_loader", wraps=make_loader) as loader_mock:
                predictions, evaluations, summary = infer_bundle(
                    bundle,
                    checkpoint_path="dummy.pt",
                    model_name="patchtst",
                    timeframe="1D",
                    horizon=2,
                    run_id="run-1",
                    model_ver="v2",
                    q_low=0.1,
                    q_high=0.9,
                )

        self.assertEqual(loader_mock.call_count, 1)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(evaluations), 3)
        self.assertIn("mae", summary)

    def test_patchtst_ci_target_fast_input_channels(self):
        model = PatchTST(
            n_features=MODEL_N_FEATURES,
            seq_len=12,
            patch_len=4,
            stride=2,
            d_model=8,
            n_heads=2,
            n_layers=1,
            horizon=2,
            dropout=0.0,
            use_revin=False,
            ci_target_fast=True,
        )
        captured: dict[str, torch.Size] = {}

        def _capture(module, inputs):
            del module
            captured["shape"] = inputs[0].shape

        hook = model.patch_proj.register_forward_pre_hook(_capture)
        try:
            model(torch.randn(2, 12, MODEL_N_FEATURES))
        finally:
            hook.remove()

        self.assertEqual(captured["shape"][0], 2)
        self.assertEqual(captured["shape"][-1], 4)

    def test_future_cov_skipped_for_patchtst(self):
        feature_df, price_df = _build_feature_rows(length=40, tickers=["AAPL"])
        dataset = build_lazy_sequence_dataset(
            feature_df,
            price_df,
            timeframe="1D",
            seq_len=10,
            horizon=2,
            include_future_covariate=False,
        )

        _, _, _, _, _, future_cov = dataset[0]
        self.assertEqual(future_cov.shape, (2, 0))

    def test_feature_cache_invalidates_on_data_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cache = Path(temp_dir)
            old_path = temp_cache / "features_1D_oldhash_deadbeef.pt"
            old_path.write_bytes(b"stale")
            new_path_a = temp_cache / resolve_feature_cache_path(timeframe="1D", data_hash="aaaaaaaa").name
            new_path_b = temp_cache / resolve_feature_cache_path(timeframe="1D", data_hash="bbbbbbbb").name

            self.assertNotEqual(new_path_a.name, new_path_b.name)
            _STALE_CACHE_WARNED.clear()
            buffer = io.StringIO()
            with patch("ai.preprocessing.CACHE_DIR", temp_cache):
                with contextlib.redirect_stdout(buffer):
                    from ai.preprocessing import _maybe_warn_stale_cache

                    _maybe_warn_stale_cache("features_1D", new_path_b)
                    _maybe_warn_stale_cache("features_1D", new_path_b)
            self.assertEqual(len([line for line in buffer.getvalue().splitlines() if line.strip()]), 1)

    def test_compile_disabled_in_sweep(self):
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        base_args = _build_base_args()
        dummy_result = {
            "best_metrics": {"best_val_total": 0.2},
            "run_id": "run-1",
            "checkpoint_path": "checkpoint.pt",
            "test_metrics": {"coverage": 0.5},
        }

        with patch("ai.sweep.run_training", return_value=dummy_result) as run_training_mock:
            value = objective_lr_sweep(trial, base_args, precomputed_bundles=("train", "val", "test", "mean", "std", "plan"))

        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertFalse(run_training_mock.call_args.kwargs["enable_compile"])


if __name__ == "__main__":
    unittest.main()
