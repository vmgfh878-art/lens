"""
모든 ticker 에 대한 AI runs mock JSON 생성.

Output: backend/data/v1/ai_runs_mock.json
- runs: 3 (CP175 line, CP153 1D band, CP178 1W band)
- evaluations: per-ticker, per-run (실제 parquet 에서 계산)
- backtests: per-run 요약

run_id 는 frontend 가 하드코딩한 거에 맞춤:
  - patchtst-1D-efad3c29d803 (line)
  - cnn_lstm-1D-d0c780dee5e8 (1D band)
  - tide-1W-walk-forward-lower (1W band; frontend 미사용이지만 API 응답용)

Run:
    cd C:\\Users\\user\\lens
    python backend/scripts/build_ai_runs_mock.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "backend" / "data" / "v1"
OUT = V1 / "ai_runs_mock.json"

NOW_ISO = datetime.now(timezone.utc).isoformat()

# Frontend 하드코딩 run_ids (StockView, TrainingView, BacktestView)
LINE_RUN_ID = "cp210_F4_b4_ensemble_mean"
BAND_1D_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
BAND_1W_RUN_ID = "tide-1W-walk-forward-lower"


def _safe_float(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def build_line_evaluations() -> list[dict]:
    df = pd.read_parquet(V1 / "predictions_line_1d.parquet")
    df = df.dropna(subset=["line_score", "actual_h5_return"])
    out = []
    for ticker, g in df.groupby("ticker"):
        if len(g) < 5:
            continue
        # 직접 계산
        # coverage = mean(actual > predicted)
        coverage = float((g["actual_h5_return"] > g["line_score"]).mean())
        # direction_accuracy = mean(sign(predicted) == sign(actual))
        direction = float((np.sign(g["line_score"]) == np.sign(g["actual_h5_return"])).mean())
        mae = float((g["actual_h5_return"] - g["line_score"]).abs().mean())
        # spearman IC (per ticker → 평균 actual vs 평균 predicted is meaningless,
        # 그냥 daily rank corr 대신 pearson over time)
        try:
            ic = float(g[["line_score", "actual_h5_return"]].corr().iloc[0, 1])
        except Exception:
            ic = None
        # top decile spread
        rank_threshold = g["line_score"].quantile(0.9)
        top = g[g["line_score"] >= rank_threshold]
        bot = g[g["line_score"] < rank_threshold]
        long_spread = float(top["actual_h5_return"].mean() - bot["actual_h5_return"].mean()) if len(top) and len(bot) else None
        latest_date = g["asof_date"].max() if "asof_date" in g.columns else None
        out.append({
            "run_id": LINE_RUN_ID,
            "ticker": str(ticker),
            "timeframe": "1D",
            "asof_date": str(latest_date) if latest_date is not None else None,
            "coverage": _safe_float(coverage),
            "avg_band_width": None,
            "lower_breach_rate": None,
            "upper_breach_rate": None,
            "direction_accuracy": _safe_float(direction),
            "mae": _safe_float(mae),
            "smape": _safe_float(mae * 0.95),  # 대략적 근사
            "spearman_ic": _safe_float(ic),
            "top_k_long_spread": _safe_float(long_spread / 2 if long_spread else None),
            "top_k_short_spread": _safe_float(-long_spread / 2 if long_spread else None),
            "long_short_spread": _safe_float(long_spread),
            "fee_adjusted_return": _safe_float(long_spread * 0.85 if long_spread else None),
            "fee_adjusted_sharpe": _safe_float(long_spread * 12 if long_spread else None),
            "fee_adjusted_turnover": 0.15,
        })
    print(f"  line evaluations: {len(out)} tickers")
    return out


def build_band_1d_evaluations() -> list[dict]:
    """
    band_lower/upper 는 가격 단위, actual_return 은 수익률 단위라
    직접 비교 불가. 대신 CP202 audit 의 global metric (coverage 0.70 등) 을
    base 로 per-ticker 변동성 더해 plausible mock 생성.
    avg_band_width 는 가격 단위 그대로 (frontend 차트와 일관).
    """
    df = pd.read_parquet(V1 / "predictions_band_1d.parquet")
    df = df.dropna(subset=["band_lower", "band_upper"])
    df = df[df["horizon_step"] == 5]
    rng = np.random.default_rng(42)
    out = []
    for ticker, g in df.groupby("ticker"):
        if len(g) < 5:
            continue
        avg_width = float((g["band_upper"] - g["band_lower"]).mean())
        # CP153 global: coverage ~0.70, lower ~0.14, upper ~0.16
        coverage = float(np.clip(0.70 + rng.normal(0, 0.04), 0.55, 0.85))
        lower_breach = float(np.clip(0.14 + rng.normal(0, 0.03), 0.05, 0.25))
        upper_breach = float(np.clip(1.0 - coverage - lower_breach, 0.05, 0.30))
        latest = g["asof_date"].max() if "asof_date" in g.columns else None
        out.append({
            "run_id": BAND_1D_RUN_ID,
            "ticker": str(ticker),
            "timeframe": "1D",
            "asof_date": str(latest) if latest is not None else None,
            "coverage": _safe_float(coverage),
            "avg_band_width": _safe_float(avg_width),
            "lower_breach_rate": _safe_float(lower_breach),
            "upper_breach_rate": _safe_float(upper_breach),
            "direction_accuracy": None,
            "mae": None,
            "smape": None,
            "spearman_ic": None,
            "top_k_long_spread": None,
            "top_k_short_spread": None,
            "long_short_spread": None,
            "fee_adjusted_return": None,
            "fee_adjusted_sharpe": None,
            "fee_adjusted_turnover": None,
        })
    print(f"  1D band evaluations: {len(out)} tickers")
    return out


def build_band_1w_evaluations() -> list[dict]:
    """
    1W band 의 actual_return 은 수익률, band_lower/upper 도 수익률 단위
    (CP178 spec). 직접 비교 가능.
    """
    df = pd.read_parquet(V1 / "predictions_band_1w.parquet")
    df = df.dropna(subset=["band_lower", "band_upper", "actual_return"])
    out = []
    for ticker, g in df.groupby("ticker"):
        if len(g) < 3:
            continue
        coverage = float(((g["actual_return"] >= g["band_lower"]) & (g["actual_return"] <= g["band_upper"])).mean())
        lower_breach = float((g["actual_return"] < g["band_lower"]).mean())
        upper_breach = float((g["actual_return"] > g["band_upper"]).mean())
        avg_width = float((g["band_upper"] - g["band_lower"]).mean())
        latest = g["asof_date"].max() if "asof_date" in g.columns else None
        out.append({
            "run_id": BAND_1W_RUN_ID,
            "ticker": str(ticker),
            "timeframe": "1W",
            "asof_date": str(latest) if latest is not None else None,
            "coverage": _safe_float(coverage),
            "avg_band_width": _safe_float(avg_width),
            "lower_breach_rate": _safe_float(lower_breach),
            "upper_breach_rate": _safe_float(upper_breach),
            "direction_accuracy": None,
            "mae": None,
            "smape": None,
            "spearman_ic": None,
            "top_k_long_spread": None,
            "top_k_short_spread": None,
            "long_short_spread": None,
            "fee_adjusted_return": None,
            "fee_adjusted_sharpe": None,
            "fee_adjusted_turnover": None,
        })
    print(f"  1W band evaluations: {len(out)} tickers")
    return out


def line_run_metadata() -> dict:
    return {
        "run_id": LINE_RUN_ID,
        "wandb_run_id": None,
        "model_name": "patchtst",
        "timeframe": "1D",
        "horizon": 5,
        "status": "completed",
        "created_at": "2026-05-25T10:22:32Z",
        "model_ver": "cp212-f4-beta4-5seed-ensemble-line",
        "feature_version": "v3_adjusted_ohlc",
        "band_mode": None,
        "checkpoint_path": "ai/artifacts/checkpoints/cp209/seed_stability + ai/artifacts/checkpoints/cp208z",
        "best_epoch": None,
        "best_val_total": None,
        "line_target_type": "raw_future_return",
        "band_target_type": None,
        "role": "line_model",
        "feature_set": "full_features",
        "checkpoint_selection": "5_seed_mean_ensemble",
        "wandb_status": None,
        "deprecated_for_phase1_product_contract": False,
        "indicator_layer_replacement": None,
        "val_metrics": {
            "wf_nonnegative_ic_folds": 4,
            "wf_ic_range": 0.04569981992725139,
        },
        "test_metrics": {
            "ic_mean": 0.03249013025981457,
            "line_top_decile_false_safe_rate": 0.20480795799944737,
            "severe_downside_recall_line_negative": 0.7727485928705441,
            "long_short_spread": 0.005476876825395852,
            "fee_adjusted_return": 0.004476876825395852,
            "wf_nonnegative_ic_folds": 4,
            "wf_ic_range": 0.04569981992725139,
        },
        "config": {
            "target": "raw_future_return",
            "seq_len": 60,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 50,
            "seed": "ensemble[7,13,23,42,71]",
            "feature_set": "full_features",
            "checkpoint_selection": "5_seed_mean_ensemble",
            "q_low": 0.5,
            "q_high": 0.5,
            "role": "line_model",
            "model_role": "line",
            "line_model_name": "patchtst_f4_beta4_ensemble",
            "alpha": 1.0,
            "beta": 4.0,
            "loss": "asymmetric_mse",
            "source_cp": "CP208Z_CP209_F4B4",
            "serving_contract": "cp212_v1_line_serving",
            "wf_stability": "moderate",
            "wf_range_exceeded": True,
        },
    }


def band_1d_run_metadata() -> dict:
    return {
        "run_id": BAND_1D_RUN_ID,
        "wandb_run_id": None,
        "model_name": "cnn_lstm",
        "timeframe": "1D",
        "horizon": 5,
        "status": "completed",
        "created_at": "2026-05-08T15:00:00Z",
        "model_ver": "cp153-1d-band-v1-primary",
        "feature_version": "v3_adjusted_ohlc",
        "band_mode": "param",
        "checkpoint_path": "ai/artifacts/checkpoints/tide_1D_tide-1D-ea54dcae654d.pt",
        "best_epoch": 3,
        "best_val_total": 0.0156,
        "line_target_type": None,
        "band_target_type": "raw_future_return",
        "role": "band_model",
        "feature_set": "price_volatility_volume",
        "checkpoint_selection": "band_gate",
        "wandb_status": None,
        "deprecated_for_phase1_product_contract": False,
        "indicator_layer_replacement": None,
        "val_metrics": {
            "coverage": 0.7010,
            "lower_breach_rate": 0.1419,
            "upper_breach_rate": 0.1570,
            "coverage_abs_error_overall": 0.0010,
            "width_future_severe_auc": 0.518,
            "width_realized_vol_spearman": 0.702,
        },
        "test_metrics": {
            "coverage": 0.7010,
            "lower_breach_rate": 0.1419,
            "upper_breach_rate": 0.1570,
            "coverage_abs_error_overall": 0.0010,
            "coverage_abs_error_stress": 0.1032,
            "width_future_severe_auc": 0.518,
            "width_realized_vol_spearman": 0.702,
            "best_val_total": 0.0156,
        },
        "config": {
            "target": "raw_future_return",
            "seq_len": 60,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 3,
            "seed": 42,
            "feature_set": "price_volatility_volume",
            "checkpoint_selection": "band_gate",
            "q_low": 0.15,
            "q_high": 0.85,
            "lambda_band": 2.0,
            "band_mode": "param",
            "role": "band_model",
            "model_role": "band",
            "band_model_name": "tide_s60_q15_param_lower_focused",
            "band_calibration_method": "lower_focused",
            "band_calibration_params": {"lower_scale": 1.05, "upper_scale": 1.0},
        },
    }


def band_1w_run_metadata() -> dict:
    return {
        "run_id": BAND_1W_RUN_ID,
        "wandb_run_id": None,
        "model_name": "tide",
        "timeframe": "1W",
        "horizon": 4,
        "status": "completed",
        "created_at": "2026-05-05T12:00:00Z",
        "model_ver": "cp178-wflock-1w-band-v1-primary",
        "feature_version": "v3_adjusted_ohlc",
        "band_mode": "param",
        "checkpoint_path": "ai/artifacts/checkpoints/tide_1W (9-checkpoint ensemble)",
        "best_epoch": None,
        "best_val_total": 0.0263,
        "line_target_type": None,
        "band_target_type": "raw_future_return",
        "role": "band_model",
        "feature_set": "price_volatility_volume",
        "checkpoint_selection": "walk_forward_lower",
        "wandb_status": None,
        "deprecated_for_phase1_product_contract": False,
        "indicator_layer_replacement": None,
        "val_metrics": {
            "coverage": 0.7657,
            "lower_breach_rate": 0.1180,
            "upper_breach_rate": 0.1164,
            "coverage_abs_error_overall": 0.0343,
            "width_future_severe_auc": 0.5262,
            "asymmetry_ratio": 1.014,
        },
        "test_metrics": {
            "coverage": 0.7657,
            "lower_breach_rate": 0.0363,
            "upper_breach_rate": 0.2611,
            "coverage_abs_error_overall": 0.0343,
            "coverage_abs_error_stress": 0.0974,
            "width_future_severe_auc": 0.5262,
            "asymmetry_ratio": 1.014,
            "best_val_total": 0.0263,
        },
        "config": {
            "target": "raw_future_return",
            "seq_len": 60,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 50,
            "seed": "ensemble[7,42,123] x fold[1,2,3]",
            "feature_set": "price_volatility_volume",
            "checkpoint_selection": "walk_forward_lower",
            "q_low": 0.10,
            "q_high": 0.90,
            "lambda_band": 2.0,
            "band_mode": "param",
            "role": "band_model",
            "model_role": "band",
            "band_model_name": "tide_s60_q10_q90_param_walk_forward_lower",
            "band_calibration_method": "walk_forward_lower_calibration",
            "band_calibration_params": {"global_shift": 0.004, "calibration_source": "cold_start_current_validation"},
        },
    }


def backtests_per_run() -> dict[str, list[dict]]:
    return {
        LINE_RUN_ID: [
            {
                "run_id": LINE_RUN_ID,
                "strategy_name": "lens_conservative_stop_loss",
                "timeframe": "1D",
                "return_pct": 8.4,
                "sharpe": 0.62,
                "mdd": -7.3,
                "win_rate": 0.547,
                "profit_factor": 1.31,
                "num_trades": 142,
                "fee_adjusted_return_pct": 7.1,
                "fee_adjusted_sharpe": 0.54,
                "avg_turnover": 0.18,
                "meta": {"period_start": "2024-10-30", "period_end": "2026-05-01", "universe": "yfinance_500", "fee_bps": 5},
                "created_at": "2026-05-12T10:00:00Z",
            }
        ],
        BAND_1D_RUN_ID: [
            {
                "run_id": BAND_1D_RUN_ID,
                "strategy_name": "lens_band_breach_signal",
                "timeframe": "1D",
                "return_pct": 6.2,
                "sharpe": 0.48,
                "mdd": -9.1,
                "win_rate": 0.521,
                "profit_factor": 1.18,
                "num_trades": 168,
                "fee_adjusted_return_pct": 4.9,
                "fee_adjusted_sharpe": 0.39,
                "avg_turnover": 0.22,
                "meta": {"period_start": "2024-09-10", "period_end": "2026-05-08", "universe": "yfinance_500", "fee_bps": 5},
                "created_at": "2026-05-12T10:00:00Z",
            }
        ],
        BAND_1W_RUN_ID: [
            {
                "run_id": BAND_1W_RUN_ID,
                "strategy_name": "lens_band_weekly_breach",
                "timeframe": "1W",
                "return_pct": 5.7,
                "sharpe": 0.45,
                "mdd": -8.5,
                "win_rate": 0.518,
                "profit_factor": 1.14,
                "num_trades": 56,
                "fee_adjusted_return_pct": 5.0,
                "fee_adjusted_sharpe": 0.40,
                "avg_turnover": 0.11,
                "meta": {"period_start": "2024-11-01", "period_end": "2026-04-17", "universe": "yfinance_500", "fee_bps": 5},
                "created_at": "2026-05-12T10:00:00Z",
            }
        ],
    }


def main():
    print("Building full ai_runs_mock from parquets...")
    line_evals = build_line_evaluations()
    band_1d_evals = build_band_1d_evaluations()
    band_1w_evals = build_band_1w_evaluations()

    payload = {
        "comment": "v1 학교 데모용 mock AI runs. Supabase 미설정 시 ai_repo 가 이 파일을 읽음. 모든 ticker per-run evaluations 포함.",
        "built_at": NOW_ISO,
        "runs": [
            line_run_metadata(),
            band_1d_run_metadata(),
            band_1w_run_metadata(),
        ],
        "evaluations": {
            LINE_RUN_ID: line_evals,
            BAND_1D_RUN_ID: band_1d_evals,
            BAND_1W_RUN_ID: band_1w_evals,
        },
        "backtests": backtests_per_run(),
    }

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"\n→ {OUT} ({size_kb:.1f} KB)")
    print(f"  total evaluations: {sum(len(v) for v in payload['evaluations'].values())}")


if __name__ == "__main__":
    main()
