"""CP216 — 전체 유의성 검정 파이프라인.

흐름:
  1. ops parquet load (line/band 1D/1W)
  2. baseline parquet build / load → docs/cp216_significance_baselines/ 저장
  3. regime label (vix, drawdown, combined)
  4. (ops, baseline) 쌍 × metric × test 종류 결과 수집
  5. Bonferroni 보정 후 metrics.json + summary.csv 작성
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .baselines import (
    build_band_bollinger,
    build_band_garch,
    build_band_historical_quantile,
    build_line_historical_mean,
    build_line_naive_zero,
    convert_band_price_to_return,
    intersect_band_panel,
    intersect_line_panel,
    load_indicators_1d,
    load_line_cp175_beta5,
    load_ops_band,
    load_ops_line,
    load_prices_1d,
    load_prices_1w,
)
from .garch_walkforward import build_band_garch_walkforward
from .bootstrap import block_bootstrap_ci, block_size_sqrt_t, cluster_bootstrap_ci
from .dm_test import dm_test
from .gw_test import gw_test
from .metrics import (
    band_pinball_daily,
    band_pinball_panel,
    line_ic_series_pair,
)
from .regime import build_regime_labels

LOG = logging.getLogger(__name__)


# ----------------------------- 베이스라인 빌드 + 캐시 -----------------------------


def _baseline_path(name: str, timeframe: str, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or config.BASELINE_DIR
    return base / f"{name}_{timeframe}.parquet"


def get_or_build_line_baseline(
    name: str,
    ops_line: pd.DataFrame,
    price_1d: pd.DataFrame,
    force: bool = False,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    path = _baseline_path(name, "1D", base_dir)
    if path.exists() and not force:
        LOG.info("line baseline cache hit: %s", path.name)
        return pd.read_parquet(path)
    LOG.info("building line baseline %s ...", name)
    if name == "naive_zero":
        out = build_line_naive_zero(ops_line)
    elif name == "historical_mean":
        out = build_line_historical_mean(ops_line, price_1d)
    elif name == "cp175_beta5":
        out = load_line_cp175_beta5()
    else:
        raise ValueError(f"unknown line baseline: {name}")
    (base_dir or config.BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def get_or_build_band_baseline(
    name: str,
    ops_band: pd.DataFrame,
    price_df: pd.DataFrame,
    timeframe: str,
    force: bool = False,
    progress: bool = False,
    base_dir: Optional[Path] = None,
    q_low: float = config.Q_LOW,
    q_high: float = config.Q_HIGH,
) -> pd.DataFrame:
    path = _baseline_path(name, timeframe, base_dir)
    if path.exists() and not force:
        LOG.info("band baseline cache hit: %s", path.name)
        return pd.read_parquet(path)
    LOG.info("building band baseline %s (%s) q=%.2f/%.2f ...", name, timeframe, q_low, q_high)
    if name == "bollinger":
        out = build_band_bollinger(ops_band, price_df, timeframe)
    elif name == "historical_quantile":
        out = build_band_historical_quantile(ops_band, price_df, timeframe, q_low=q_low, q_high=q_high)
    elif name == "garch_p_q_1_1":
        out = build_band_garch(ops_band, price_df, timeframe, q_low=q_low, q_high=q_high, progress=progress)
    elif name == "garch_walkforward":
        out = build_band_garch_walkforward(ops_band, price_df, timeframe, q_low=q_low, q_high=q_high, progress=progress)
    else:
        raise ValueError(f"unknown band baseline: {name}")
    (base_dir or config.BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# ----------------------------- 효과 크기 -----------------------------


def cohens_d_paired(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(d.mean() / sd)


# ----------------------------- 검정 한 쌍 -----------------------------


@dataclass
class PairResult:
    family: str        # "line" or "band"
    timeframe: str
    ops_model_id: str
    baseline_id: str
    metric: str        # "ic_daily" or "pinball_daily"
    n_obs: int
    n_tickers: int
    mean_diff: float
    cohens_d: float
    dm_stat: float
    dm_p_two_sided: float
    dm_p_a_better: float
    cluster_n: int
    cluster_ci_lower: float
    cluster_ci_upper: float
    block_sqrt_t_n: int
    block_sqrt_t_size: int
    block_sqrt_t_ci_lower: float
    block_sqrt_t_ci_upper: float
    block_22_n: int
    block_22_ci_lower: float
    block_22_ci_upper: float
    gw_vix_wald_p: float
    gw_vix_beta: float
    gw_vix_beta_p: float
    gw_dd_wald_p: float
    gw_dd_beta: float
    gw_dd_beta_p: float
    gw_combined_wald_p: float
    gw_combined_beta: float
    gw_combined_beta_p: float


def _to_dict(r: PairResult) -> Dict:
    return asdict(r)


def _safe_align(series_a: pd.Series, series_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = series_a.index.intersection(series_b.index)
    return series_a.reindex(common), series_b.reindex(common)


def _per_ticker_pinball_diff(panel_with_pinball: pd.DataFrame) -> np.ndarray:
    grp = panel_with_pinball.groupby("ticker")
    diffs = grp.apply(lambda g: g["pinball_a"].mean() - g["pinball_b"].mean(), include_groups=False)
    return diffs.values


def _regime_aligned(d_series: pd.Series, regime_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    common = d_series.index.intersection(regime_series.index)
    return d_series.reindex(common).values, regime_series.reindex(common).values


def evaluate_line_pair(
    ops_line: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_id: str,
    regime: pd.DataFrame,
    n_bootstrap: int,
    device: str,
) -> PairResult:
    merged = intersect_line_panel(ops_line, baseline)
    if merged.empty:
        raise RuntimeError(f"line panel intersection empty for {baseline_id}")
    ic_a, ic_b = line_ic_series_pair(merged)
    d_series = ic_a - ic_b  # (negative = baseline better; positive = ops better since IC higher is better)
    # NOTE: 라인 metric (IC) 는 높을수록 좋음. d = IC_a - IC_b. d>0 면 A=ops 우위.
    # DM 의 "loss" 컨벤션은 낮을수록 좋음이라 부호를 뒤집어 (loss = -IC) 입력.
    loss_a = (-ic_a).values
    loss_b = (-ic_b).values
    dm = dm_test(loss_a, loss_b)
    # cluster bootstrap: 라인 IC 는 cross-sectional 평균이라 per-ticker 분해 불가 → NA
    cluster = type("Empty", (), {"n_iter": 0, "ci_lower": float("nan"), "ci_upper": float("nan"), "point": float("nan")})()
    # block bootstrap on d_series
    b_sqrt = block_size_sqrt_t(len(d_series))
    block_sqrt = block_bootstrap_ci(d_series.values, b_sqrt, n_iter=n_bootstrap, device=device, seed=2025)
    block_22 = block_bootstrap_ci(d_series.values, 22, n_iter=n_bootstrap, device=device, seed=2025)
    # GW (3 regimes) on d_series
    gw_results = {}
    for regime_name in ["vix_high", "drawdown_low", "combined"]:
        r_arr = regime[regime_name] if regime_name in regime.columns else pd.Series(dtype=int)
        d_arr, h_arr = _regime_aligned(d_series, r_arr)
        gw_results[regime_name] = gw_test(loss_a=-d_arr, loss_b=np.zeros_like(d_arr), regime=h_arr)
        # 위 호출 idea: GW 의 z_t = L_A - L_B = -d_arr (loss 컨벤션) → coefs 의 부호도 그에 맞춤.

    return PairResult(
        family="line",
        timeframe="1D",
        ops_model_id=str(ops_line["model_id"].iloc[0]) if "model_id" in ops_line.columns else "ops_line",
        baseline_id=baseline_id,
        metric="ic_daily",
        n_obs=int(len(d_series)),
        n_tickers=int(merged["ticker"].nunique()),
        mean_diff=float(d_series.mean()),
        cohens_d=cohens_d_paired(d_series.values),
        dm_stat=dm.dm_stat,
        dm_p_two_sided=dm.p_value_two_sided,
        dm_p_a_better=dm.p_value_a_better,
        cluster_n=0,
        cluster_ci_lower=float("nan"),
        cluster_ci_upper=float("nan"),
        block_sqrt_t_n=block_sqrt.n_iter,
        block_sqrt_t_size=b_sqrt,
        block_sqrt_t_ci_lower=block_sqrt.ci_lower,
        block_sqrt_t_ci_upper=block_sqrt.ci_upper,
        block_22_n=block_22.n_iter,
        block_22_ci_lower=block_22.ci_lower,
        block_22_ci_upper=block_22.ci_upper,
        gw_vix_wald_p=gw_results["vix_high"].wald_p_value,
        gw_vix_beta=gw_results["vix_high"].beta,
        gw_vix_beta_p=gw_results["vix_high"].beta_p_value_two_sided,
        gw_dd_wald_p=gw_results["drawdown_low"].wald_p_value,
        gw_dd_beta=gw_results["drawdown_low"].beta,
        gw_dd_beta_p=gw_results["drawdown_low"].beta_p_value_two_sided,
        gw_combined_wald_p=gw_results["combined"].wald_p_value,
        gw_combined_beta=gw_results["combined"].beta,
        gw_combined_beta_p=gw_results["combined"].beta_p_value_two_sided,
    )


def evaluate_band_pair(
    ops_band: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_id: str,
    timeframe: str,
    regime: pd.DataFrame,
    n_bootstrap: int,
    device: str,
    q_low: float = config.Q_LOW,
    q_high: float = config.Q_HIGH,
) -> PairResult:
    merged = intersect_band_panel(ops_band, baseline)
    if merged.empty:
        raise RuntimeError(f"band panel intersection empty for {baseline_id} ({timeframe})")
    panel = band_pinball_panel(merged, q_low=q_low, q_high=q_high)
    pa, pb = band_pinball_daily(panel)
    d_series = pa - pb  # loss diff. negative = ops 더 좋음.
    dm = dm_test(pa.values, pb.values)
    # cluster bootstrap: per-ticker mean pinball diff
    cluster_diff = _per_ticker_pinball_diff(panel)
    cluster = cluster_bootstrap_ci(cluster_diff, n_iter=n_bootstrap, device=device, seed=2025)
    b_sqrt = block_size_sqrt_t(len(d_series))
    block_sqrt = block_bootstrap_ci(d_series.values, b_sqrt, n_iter=n_bootstrap, device=device, seed=2025)
    block_22 = block_bootstrap_ci(d_series.values, 22, n_iter=n_bootstrap, device=device, seed=2025)
    gw_results = {}
    for regime_name in ["vix_high", "drawdown_low", "combined"]:
        r_arr = regime[regime_name] if regime_name in regime.columns else pd.Series(dtype=int)
        d_arr, h_arr = _regime_aligned(d_series, r_arr)
        gw_results[regime_name] = gw_test(loss_a=d_arr, loss_b=np.zeros_like(d_arr), regime=h_arr)

    return PairResult(
        family="band",
        timeframe=timeframe,
        ops_model_id=str(ops_band["model_id"].iloc[0]) if "model_id" in ops_band.columns else f"ops_band_{timeframe}",
        baseline_id=baseline_id,
        metric="pinball_daily",
        n_obs=int(len(d_series)),
        n_tickers=int(panel["ticker"].nunique()),
        mean_diff=float(d_series.mean()),
        cohens_d=cohens_d_paired(d_series.values),
        dm_stat=dm.dm_stat,
        dm_p_two_sided=dm.p_value_two_sided,
        dm_p_a_better=dm.p_value_a_better,
        cluster_n=cluster.n_iter,
        cluster_ci_lower=cluster.ci_lower,
        cluster_ci_upper=cluster.ci_upper,
        block_sqrt_t_n=block_sqrt.n_iter,
        block_sqrt_t_size=b_sqrt,
        block_sqrt_t_ci_lower=block_sqrt.ci_lower,
        block_sqrt_t_ci_upper=block_sqrt.ci_upper,
        block_22_n=block_22.n_iter,
        block_22_ci_lower=block_22.ci_lower,
        block_22_ci_upper=block_22.ci_upper,
        gw_vix_wald_p=gw_results["vix_high"].wald_p_value,
        gw_vix_beta=gw_results["vix_high"].beta,
        gw_vix_beta_p=gw_results["vix_high"].beta_p_value_two_sided,
        gw_dd_wald_p=gw_results["drawdown_low"].wald_p_value,
        gw_dd_beta=gw_results["drawdown_low"].beta,
        gw_dd_beta_p=gw_results["drawdown_low"].beta_p_value_two_sided,
        gw_combined_wald_p=gw_results["combined"].wald_p_value,
        gw_combined_beta=gw_results["combined"].beta,
        gw_combined_beta_p=gw_results["combined"].beta_p_value_two_sided,
    )


# ----------------------------- Bonferroni -----------------------------


P_COLUMNS = [
    "dm_p_two_sided",
    "dm_p_a_better",
    "gw_vix_wald_p",
    "gw_vix_beta_p",
    "gw_dd_wald_p",
    "gw_dd_beta_p",
    "gw_combined_wald_p",
    "gw_combined_beta_p",
]


def add_bonferroni(df: pd.DataFrame) -> pd.DataFrame:
    n_tests = len(df)
    for col in P_COLUMNS:
        if col in df.columns:
            df[f"{col}_bonferroni"] = (df[col] * n_tests).clip(upper=1.0)
    df["n_tests"] = n_tests
    return df


# ----------------------------- 메인 진입점 -----------------------------


def run_significance(
    line_baselines: List[str],
    band_baselines: List[str],
    timeframes: List[str],
    n_bootstrap: int = 1000,
    device: str = "cpu",
    dry_run: bool = False,
    tickers: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    rebuild_baselines: bool = False,
    progress: bool = False,
    baseline_dir: Optional[Path] = None,
    quantiles_by_timeframe: Optional[Dict[str, tuple]] = None,
    price_1w_parquet: Optional[Path] = None,
    backfilled_band_1d_parquet: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = config.SIG_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if quantiles_by_timeframe is None:
        quantiles_by_timeframe = {"1D": (config.Q_LOW, config.Q_HIGH), "1W": (config.Q_LOW, config.Q_HIGH)}

    LOG.info("loading ops + price + indicators ...")
    ops_line = load_ops_line()
    ops_band_1d_raw = load_ops_band("1D") if "1D" in timeframes else pd.DataFrame()
    ops_band_1w = load_ops_band("1W") if "1W" in timeframes else pd.DataFrame()
    price_1d = load_prices_1d()
    # 1W price: override 옵션 (CP216.2 의 500 ticker weekly resample)
    if price_1w_parquet is not None and Path(price_1w_parquet).exists():
        LOG.info("using 1W price parquet override: %s", price_1w_parquet)
        price_1w = pd.read_parquet(price_1w_parquet)
        price_1w["date"] = pd.to_datetime(price_1w["date"]).dt.date.astype(str)
    else:
        price_1w = load_prices_1w()
    ind_1d = load_indicators_1d()
    # 1D backfill 옵션: 운영 parquet 과 backfill prediction 합쳐서 검정
    if backfilled_band_1d_parquet is not None and Path(backfilled_band_1d_parquet).exists():
        LOG.info("merging 1D backfill: %s", backfilled_band_1d_parquet)
        bf = pd.read_parquet(backfilled_band_1d_parquet)
        bf["asof_date"] = pd.to_datetime(bf["asof_date"]).dt.date.astype(str)
        # 운영 영역은 운영 우선, backfill은 신영역만 (운영 asof 최소 이전)
        if not ops_band_1d_raw.empty:
            ops_min_asof = ops_band_1d_raw["asof_date"].min()
            bf_new = bf[bf["asof_date"] < ops_min_asof]
            ops_band_1d_raw = pd.concat([bf_new, ops_band_1d_raw], axis=0, ignore_index=True)
    # 1D ops band (CP153) 는 band_lower/upper 가 가격 단위 → asof_close 로 수익률 환산.
    # 1W ops band (CP178) 는 이미 수익률 단위 → no-op.
    ops_band_1d = convert_band_price_to_return(ops_band_1d_raw, price_1d) if not ops_band_1d_raw.empty else ops_band_1d_raw

    if tickers is not None:
        tickers_set = set(tickers)
        ops_line = ops_line[ops_line["ticker"].isin(tickers_set)]
        ops_band_1d = ops_band_1d[ops_band_1d["ticker"].isin(tickers_set)] if not ops_band_1d.empty else ops_band_1d
        ops_band_1w = ops_band_1w[ops_band_1w["ticker"].isin(tickers_set)] if not ops_band_1w.empty else ops_band_1w
        price_1d = price_1d[price_1d["ticker"].isin(tickers_set)]
        price_1w = price_1w[price_1w["ticker"].isin(tickers_set)]
        LOG.info("ticker subset n=%d", len(tickers_set))

    regime = build_regime_labels(ind_1d, price_1d)
    LOG.info("regime: n=%d vix_on=%d dd_on=%d combined_on=%d",
             len(regime), int(regime["vix_high"].sum()), int(regime["drawdown_low"].sum()),
             int(regime["combined"].sum()))

    results: List[PairResult] = []

    # ---- LINE ----
    if line_baselines:
        for name in line_baselines:
            baseline = get_or_build_line_baseline(name, ops_line, price_1d,
                                                  force=rebuild_baselines, base_dir=baseline_dir)
            if tickers is not None:
                baseline = baseline[baseline["ticker"].isin(tickers_set)]
            LOG.info("LINE pair: ops vs %s (rows=%d)", name, len(baseline))
            r = evaluate_line_pair(ops_line, baseline, name, regime, n_bootstrap=n_bootstrap, device=device)
            results.append(r)

    # ---- BAND 1D ----
    if "1D" in timeframes and band_baselines:
        q_low_1d, q_high_1d = quantiles_by_timeframe.get("1D", (config.Q_LOW, config.Q_HIGH))
        for name in band_baselines:
            baseline = get_or_build_band_baseline(name, ops_band_1d, price_1d, "1D",
                                                  force=rebuild_baselines, progress=progress,
                                                  base_dir=baseline_dir,
                                                  q_low=q_low_1d, q_high=q_high_1d)
            if tickers is not None:
                baseline = baseline[baseline["ticker"].isin(tickers_set)]
            LOG.info("BAND 1D pair: ops vs %s (rows=%d) q=%.2f/%.2f", name, len(baseline), q_low_1d, q_high_1d)
            r = evaluate_band_pair(ops_band_1d, baseline, name, "1D", regime,
                                   n_bootstrap=n_bootstrap, device=device,
                                   q_low=q_low_1d, q_high=q_high_1d)
            results.append(r)

    # ---- BAND 1W ----
    if "1W" in timeframes and band_baselines:
        q_low_1w, q_high_1w = quantiles_by_timeframe.get("1W", (config.Q_LOW, config.Q_HIGH))
        for name in band_baselines:
            baseline = get_or_build_band_baseline(name, ops_band_1w, price_1w, "1W",
                                                  force=rebuild_baselines, progress=progress,
                                                  base_dir=baseline_dir,
                                                  q_low=q_low_1w, q_high=q_high_1w)
            if tickers is not None:
                baseline = baseline[baseline["ticker"].isin(tickers_set)]
            LOG.info("BAND 1W pair: ops vs %s (rows=%d) q=%.2f/%.2f", name, len(baseline), q_low_1w, q_high_1w)
            r = evaluate_band_pair(ops_band_1w, baseline, name, "1W", regime,
                                   n_bootstrap=n_bootstrap, device=device,
                                   q_low=q_low_1w, q_high=q_high_1w)
            results.append(r)

    summary = pd.DataFrame([_to_dict(r) for r in results])
    summary = add_bonferroni(summary)

    summary_path = output_dir / "cp216_significance_summary.csv"
    metrics_path = output_dir / "cp216_significance_metrics.json"
    summary.to_csv(summary_path, index=False)
    metrics_payload = {
        "config": {
            "n_bootstrap": n_bootstrap,
            "device": device,
            "dry_run": dry_run,
            "tickers": list(tickers) if tickers else None,
            "line_baselines": line_baselines,
            "band_baselines": band_baselines,
            "timeframes": timeframes,
            "quantiles_by_timeframe": {tf: list(q) for tf, q in quantiles_by_timeframe.items()},
            "price_1w_parquet": str(price_1w_parquet) if price_1w_parquet else None,
            "backfilled_band_1d_parquet": str(backfilled_band_1d_parquet) if backfilled_band_1d_parquet else None,
            "vix_threshold": config.VIX_THRESHOLD,
            "dd_window": config.DD_WINDOW,
            "dd_threshold": config.DD_THRESHOLD,
        },
        "results": [_to_dict(r) for r in results],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=float, ensure_ascii=False)

    LOG.info("wrote %s and %s (rows=%d)", summary_path.name, metrics_path.name, len(summary))
    return summary_path
