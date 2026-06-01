"""CP216 — CLI 진입점.

Example:
    .venv/Scripts/python -m ai.eval.significance.cli \\
        --line-baselines naive_zero,historical_mean,cp175_beta5 \\
        --band-baselines bollinger,historical_quantile,garch_p_q_1_1 \\
        --timeframes 1D,1W \\
        --n-bootstrap 1000 --device cuda \\
        --output-dir docs/cp216_significance
"""

from __future__ import annotations

import os
# Windows mkl/openmp 충돌 방지 — pandas/scipy 이전에 환경변수 박고 torch eager preload.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch as _torch_preload  # noqa: F401
except Exception:
    _torch_preload = None  # CPU only 환경에서는 무시

import argparse
import logging
import sys
import warnings
from pathlib import Path

# 작은 표본에서 모든 점수가 같아 spearmanr 미정의 경우의 경고 억제.
warnings.filterwarnings(
    "ignore",
    message="An input array is constant",
    category=Warning,
)

from . import config
from .pipeline import run_significance


def _csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CP216 v1 운영 3모델 통계 유의성 검정")
    p.add_argument("--line-baselines", type=_csv_list,
                   default=["naive_zero", "historical_mean", "cp175_beta5"])
    p.add_argument("--band-baselines", type=_csv_list,
                   default=["bollinger", "historical_quantile", "garch_p_q_1_1"])
    p.add_argument("--timeframes", type=_csv_list, default=["1D", "1W"])
    p.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP_DEFAULT)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--tickers", type=_csv_list, default=None,
                   help="ticker subset (dry-run / smoke 용).")
    p.add_argument("--output-dir", type=Path, default=config.SIG_DIR)
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="베이스라인 parquet 저장 디렉토리. 미지정 시 config.BASELINE_DIR.")
    p.add_argument("--rebuild-baselines", action="store_true",
                   help="cache 무시 후 baseline 재빌드.")
    p.add_argument("--progress", action="store_true", help="tqdm progress bar 표시 (GARCH 등).")
    p.add_argument("--dry-run", action="store_true",
                   help="ticker 부분집합 + n_bootstrap 100 으로 smoke.")
    p.add_argument("--mode", choices=["cp216", "cp216_2"], default="cp216",
                   help="cp216_2: 1D q15/q85, 1W q10/q90, walk-forward GARCH, 500 ticker 1W price.")
    p.add_argument("--price-1w-parquet", type=Path, default=None,
                   help="1W price parquet override (예: price_data_yfinance_1W_500.parquet).")
    p.add_argument("--backfilled-band-1d", type=Path, default=None,
                   help="1D backfill prediction parquet (운영 영역 이전 asof 합쳐서 검정).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    n_bootstrap = args.n_bootstrap
    if args.dry_run:
        n_bootstrap = min(n_bootstrap, 100)
        if not args.tickers:
            args.tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    # CP216.2 mode 자동 lock
    if args.mode == "cp216_2":
        quantiles_by_timeframe = {"1D": (0.15, 0.85), "1W": (0.10, 0.90)}
        # 기본 1W price 500 → 사용자가 별도 override 안 했으면 자동
        if args.price_1w_parquet is None and config.PRICES_1W_500.exists():
            args.price_1w_parquet = config.PRICES_1W_500
        if args.baseline_dir is None:
            args.baseline_dir = config.BASELINE_DIR_CP216_2
        if args.output_dir == config.SIG_DIR:
            args.output_dir = config.SIG_DIR_CP216_2
    else:
        quantiles_by_timeframe = {"1D": (config.Q_LOW, config.Q_HIGH), "1W": (config.Q_LOW, config.Q_HIGH)}

    out = run_significance(
        line_baselines=args.line_baselines,
        band_baselines=args.band_baselines,
        timeframes=args.timeframes,
        n_bootstrap=n_bootstrap,
        device=args.device,
        dry_run=args.dry_run,
        tickers=args.tickers,
        output_dir=args.output_dir,
        rebuild_baselines=args.rebuild_baselines,
        progress=args.progress,
        baseline_dir=args.baseline_dir,
        quantiles_by_timeframe=quantiles_by_timeframe,
        price_1w_parquet=args.price_1w_parquet,
        backfilled_band_1d_parquet=args.backfilled_band_1d,
    )
    print(f"[CP216] summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
