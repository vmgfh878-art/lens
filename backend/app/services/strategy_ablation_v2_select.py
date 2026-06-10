"""CP252 — v2 발굴 결과 집계 + 목적 3트랙 선정 + deflated/selection-inflation.

- aggregate_rows: config 별 (절대 metric 평균 + none 대비 델타 평균/중위, 옵션 CI/Wilcoxon).
- select_tracks: 방어/공격/균형 **따로** 랭킹 (목적별 기준·허용치).
- selection_inflation: 넓은 탐색 과적합 벤치마크(E[max] under null) — dev best noise 가능성 표기.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from app.services.strategy_ablation_metrics import _bootstrap_ci, _wilcoxon_p
from app.services.strategy_ablation_v2 import ABS_METRICS

# 트랙별 허용치 (CP252 P0 사용자 확정: 기본).
DEFENSIVE_RETURN_FLOOR = -10.0  # d_totalReturnPct ≥ (수익 포기 한도)
OFFENSIVE_DMAXDD_FLOOR = -5.0  # d_maxDrawdownPct ≥ (AI 가 낙폭 악화 한도)
OFFENSIVE_MAXDD_FLOOR = -35.0  # 절대 maxDrawdownPct ≥ (파국 배제)
BOOTSTRAP_SEED = 252
ALPHA = 0.05


def aggregate_rows(
    rows: pd.DataFrame, n_tested: int, with_ci: bool = False, ci_metrics: tuple[str, ...] = ()
) -> pd.DataFrame:
    """config 별 집계. 절대 metric 평균(퇴화 제외) + 델타 평균/중위. with_ci 면 CI/Wilcoxon.

    returns DataFrame indexed by config_id.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict[str, Any]] = []
    delta_cols = [c for c in rows.columns if c.startswith("d_")]
    for config_id, grp in rows.groupby("config_id", sort=False):
        kept = grp[~grp["degenerate"]]
        rec: dict[str, Any] = {
            "config_id": config_id,
            "base_key": grp["base_key"].iloc[0],
            "archetype": grp["archetype"].iloc[0],
            "toggle": grp["toggle"].iloc[0],
            "n_tickers": int(len(kept)),
            "n_degenerate": int(grp["degenerate"].sum()),
        }
        for m in ABS_METRICS:
            vals = pd.to_numeric(kept[m], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[m] = float(np.mean(vals)) if vals.size else np.nan
        for m in delta_cols:
            vals = pd.to_numeric(kept[m], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[m + "_mean"] = float(np.mean(vals)) if vals.size else np.nan
            rec[m + "_median"] = float(np.median(vals)) if vals.size else np.nan
            if with_ci and m in ci_metrics and vals.size:
                lo, hi = _bootstrap_ci(vals, rng)
                p = _wilcoxon_p(vals)
                rec[m + "_ci_low"], rec[m + "_ci_high"] = lo, hi
                rec[m + "_p"] = None if np.isnan(p) else float(p)
                rec[m + "_ci_excl0"] = bool(lo > 0 or hi < 0)
                rec[m + "_bonf"] = bool((not np.isnan(p)) and p < ALPHA / max(n_tested, 1))
        out.append(rec)
    return pd.DataFrame(out).set_index("config_id")


def _avg_rank(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # 큰 값이 좋음 → ascending=False. 평균 랭크(작을수록 좋음).
    ranks = [df[c].rank(ascending=False, method="min") for c in cols]
    return sum(ranks) / len(ranks)


def select_tracks(agg: pd.DataFrame, top_k: int = 10) -> dict[str, pd.DataFrame]:
    """방어/공격/균형 따로 랭킹. 각 top_k."""
    variants = agg[agg["toggle"] != "none"]

    # 🛡 방어형: d_maxDrawdownPct·d_lla 최대, 수익 포기 한도 내.
    dfn = variants[variants["d_totalReturnPct_mean"] >= DEFENSIVE_RETURN_FLOOR].copy()
    if len(dfn):
        dfn["avg_rank"] = _avg_rank(dfn, ["d_maxDrawdownPct_mean", "d_lla_mean"])
        defensive = dfn.sort_values("avg_rank").head(top_k)
    else:
        defensive = dfn

    # ⚔ 공격형: 절대 excess·총수익 최대 (none 포함), 파국·AI 낙폭악화 배제.
    off = agg[agg["maxDrawdownPct"] >= OFFENSIVE_MAXDD_FLOOR].copy()
    off = off[(off["toggle"] == "none") | (off["d_maxDrawdownPct_mean"] >= OFFENSIVE_DMAXDD_FLOOR)]
    if len(off):
        off["avg_rank"] = _avg_rank(off, ["excessReturnPct", "totalReturnPct"])
        offensive = off.sort_values("avg_rank").head(top_k)
    else:
        offensive = off

    # ⚖ 균형형: d_calmar·d_sortino 최대 (둘 다 양수 우대).
    bal = variants.copy()
    bal["avg_rank"] = _avg_rank(bal, ["d_calmar_mean", "d_sortino_mean"])
    bal["both_pos"] = (bal["d_calmar_mean"] > 0) & (bal["d_sortino_mean"] > 0)
    balanced = bal.sort_values(["both_pos", "avg_rank"], ascending=[False, True]).head(top_k)

    return {"defensive": defensive, "offensive": offensive, "balanced": balanced}


def selection_inflation(agg: pd.DataFrame, metric_mean_col: str) -> dict[str, float]:
    """넓은 탐색 과적합 벤치마크. metric 의 config 간 분포에서 E[max under null] 추정.

    dev best 가 이 벤치마크를 크게 안 넘으면 noise 가능성 — held-out 이 진짜 판정.
    E[max of N] ≈ mean + std·sqrt(2 ln N) (gaussian 근사).
    """
    vals = pd.to_numeric(agg[metric_mean_col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n < 2:
        return {}
    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
    expected_max = mean + std * float(np.sqrt(2.0 * np.log(max(n, 2))))
    return {
        "n_configs": n,
        "metric": metric_mean_col,
        "mean": mean,
        "std": std,
        "observed_max": float(np.max(vals)),
        "expected_max_under_null": expected_max,
        "exceeds_null_benchmark": bool(np.max(vals) > expected_max),
    }


__all__ = [
    "DEFENSIVE_RETURN_FLOOR",
    "OFFENSIVE_DMAXDD_FLOOR",
    "OFFENSIVE_MAXDD_FLOOR",
    "aggregate_rows",
    "select_tracks",
    "selection_inflation",
]
