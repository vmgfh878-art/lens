"""CP252 — 전략 발굴 실험 하니스 (다양·대규모, find-only).

CP248~250 한계(밴드만 살아남음, 라인·공격형 미탐색, config 15개)를 넘어 **넓게** 찾는다:
- base archetype 8계열(추세/눌림목/균형 + 모멘텀/돌파/평균회귀/저변동/추세모멘텀) × 파라미터.
- AI 토글에 **라인 공정 기회**: gate(자기분위) · momentum · confirm · sizing(분수 포지션).
- band 컷×모드(lower/width/both) 그리드.
- **분수 포지션 엔진**(라인 사이징): 이진 long/cash 상태머신 위에 size∈[floor,1] 곱.

엄밀성(CP248 유지): 2축 OOS(티커 dev/heldout + 시간 val/test) · 부트스트랩 CI · Wilcoxon ·
Bonferroni · deflated · selection-bias 가드. 컷은 dev/val frozen(`v2_cuts.json`, 누수 0).

제약: 라이브 `ai_band_defense_v2`·CP248 경로 **불변**(병렬 모듈, 운영 _raw_target/엔진 미수정).
새 학습·DB write 0, parquet read-only. 부착은 별도 CP(이건 find-only).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.services.strategy_indicators import _run_state_machine, _safe, _safe_col

FEE_RATE = 0.001
TRADING_DAYS = 252.0
DD_FLOOR = 0.01


@lru_cache(maxsize=1)
def v2_cuts() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "data" / "ablation" / "v2_cuts.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# 컬럼 추출 (safe). archetype/toggle 가 공유.
# ─────────────────────────────────────────────────────────────────────────────


def _extract(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ma60": _safe(frame["ma_60_ratio"]),
        "ma20": _safe(frame["ma_20_ratio"]),
        "ma5": _safe_col(frame, "ma_5_ratio", 0.0),
        "macd": _safe(frame["macd_ratio"], 0.0),
        "macd_accel": _safe_col(frame, "macd_accel", 0.0),
        "rsi": _safe(frame["rsi_norm"], 50.0),
        "atr": _safe(frame["atr_ratio_calc"], 0.0),
        "bb": _safe(frame["bb_position"], 1.0),
        "roc20": _safe_col(frame, "roc_20", 0.0),
        "new_high": _safe_col(frame, "high_20", np.nan),
        "close": _safe(frame["close"]),
        "close_z": _safe_col(frame, "close_z_20", 0.0),
        "vol_surge": _safe_col(frame, "vol_surge", 1.0),
        "vol_change": _safe_col(frame, "vol_change", 0.0),
        "line": _safe_col(frame, "line_score"),
        "line_mom": _safe_col(frame, "line_mom", 0.0),
        "lower": _safe_col(frame, "band_lower_return"),
        "width_exp": _safe_col(frame, "band_width_expansion", 1.0),
    }


def _common_exit(
    c: dict[str, pd.Series],
    ma60_stop: float = -0.05,
    ma20_stop: float = -0.06,
    atr_stop: float = 0.10,
) -> pd.Series:
    return (
        (c["ma60"] <= ma60_stop)
        | (c["ma20"] <= ma20_stop)
        | ((c["atr"] > atr_stop) & (c["ma20"] < 0.0))
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8 archetype. 각 (c, p) → (entry, exit, risk). p = 파라미터 dict.
# ─────────────────────────────────────────────────────────────────────────────


def _arch_trend(c, p):
    entry = (
        (c["ma60"] >= p["ma60"]) & (c["ma20"] >= -0.02) & (c["macd"] >= 0.0) & (c["rsi"] < p["rsi"])
    )
    ex = _common_exit(c)
    return entry, ex, ex


def _arch_pullback(c, p):
    entry = (c["ma60"] >= 0.02) & (c["bb"] <= p["bb"]) & (c["rsi"] < p["rsi"])
    ex = _common_exit(c)
    return entry, ex, ex


def _arch_balance(c, p):
    trend = (c["ma60"] >= p["ma60"]) & (c["ma20"] >= -0.02) & (c["macd"] >= 0.0) & (c["rsi"] < 75.0)
    pull = (c["ma60"] >= p["ma60"]) & (c["bb"] <= 0.35) & (c["rsi"] < 55.0)
    ex = _common_exit(c)
    return (trend | pull), ex, ex


def _arch_momentum(c, p):
    entry = (
        (c["roc20"] >= p["roc"])
        & (c["macd_accel"] >= 0.0)
        & (c["ma5"] >= p["ma5"])
        & (c["rsi"] < 80.0)
    )
    ex = (c["roc20"] < 0.0) | _common_exit(c)
    return entry, ex, ex


def _arch_breakout(c, p):
    # 당일 종가가 직전 20일 고가의 near 배 이상(돌파/근접) + 거래량 surge + 단기 추세.
    near_high = c["close"] >= (c["new_high"] * p["near"])
    entry = near_high & (c["vol_surge"] >= p["vol_surge"]) & (c["ma20"] >= 0.0)
    ex = (c["ma20"] < -0.04) | _common_exit(c)
    return entry, ex, ex


def _arch_meanrev(c, p):
    # counter-trend: 과매도/저z 진입, 반등하면 청산.
    entry = (c["close_z"] <= p["z"]) & (c["rsi"] < p["rsi"]) & (c["ma60"] >= -0.10)
    ex = (c["rsi"] > 55.0) | (c["close_z"] > 0.0) | (c["ma20"] < -0.10)
    risk = (c["ma20"] < -0.10) | ((c["atr"] > 0.12) & (c["ma20"] < 0.0))
    return entry, ex, risk


def _arch_lowvol(c, p):
    entry = (
        (c["atr"] <= p["atr"])
        & (c["vol_change"] <= p["vol_change"])
        & (c["ma60"] >= 0.0)
        & (c["ma20"] >= -0.02)
    )
    ex = (c["atr"] > p["atr"] * 2.0) | _common_exit(c)
    return entry, ex, ex


def _arch_trendmom(c, p):
    entry = (
        (c["ma60"] >= p["ma60"])
        & (c["ma20"] >= 0.0)
        & (c["roc20"] >= p["roc"])
        & (c["macd"] >= 0.0)
    )
    ex = (c["roc20"] < 0.0) | _common_exit(c)
    return entry, ex, ex


ARCHETYPES: dict[str, Callable] = {
    "trend": _arch_trend,
    "pullback": _arch_pullback,
    "balance": _arch_balance,
    "momentum": _arch_momentum,
    "breakout": _arch_breakout,
    "meanrev": _arch_meanrev,
    "lowvol": _arch_lowvol,
    "trendmom": _arch_trendmom,
}

# 파라미터 그리드 (archetype 별). 곱집합으로 base 인스턴스 생성.
ARCHETYPE_GRID: dict[str, dict[str, list]] = {
    "trend": {"ma60": [0.01, 0.02, 0.03], "rsi": [70.0, 75.0, 80.0]},
    "pullback": {"bb": [0.30, 0.35, 0.40], "rsi": [50.0, 55.0, 60.0]},
    "balance": {"ma60": [0.01, 0.02, 0.03]},
    "momentum": {"roc": [0.02, 0.05, 0.08], "ma5": [0.0, 0.005]},
    "breakout": {"vol_surge": [1.2, 1.5, 2.0], "near": [0.99, 1.0]},
    "meanrev": {"z": [-2.0, -1.5, -1.0], "rsi": [25.0, 30.0, 35.0]},
    "lowvol": {"atr": [0.03, 0.05, 0.07], "vol_change": [0.0, 0.25]},
    "trendmom": {"ma60": [0.01, 0.02], "roc": [0.03, 0.06]},
}


def _param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    out = [{}]
    for k in keys:
        out = [{**d, k: v} for d in out for v in grid[k]]
    return out


def _ptag(params: dict) -> str:
    return "-".join(f"{k}{params[k]}" for k in sorted(params)) if params else "base"


# ─────────────────────────────────────────────────────────────────────────────
# AI 토글. band(컷×모드) + line(gate/momentum/confirm/sizing).
# toggle 은 (c, base_entry, base_exit, base_risk) → (entry, exit, risk, size).
# size 는 None(=1.0) 또는 [floor,1] Series (분수 사이징).
# ─────────────────────────────────────────────────────────────────────────────


def _band_toggle(c, entry, ex, risk, *, q: float, mode: str):
    cuts = v2_cuts()
    lower_cut = cuts["band_lower"][str(q)]
    width_cut = cuts["band_width"][str(round(1.0 - q, 2))]
    deep = c["lower"] <= lower_cut
    wide = c["width_exp"] >= width_cut
    if mode == "lower":
        band_risk = deep
    elif mode == "width":
        band_risk = wide
    else:
        band_risk = deep | wide
    return (entry & ~band_risk), (ex | band_risk), (risk | band_risk)


def _line_size(c, floor: float) -> pd.Series:
    cuts = v2_cuts()
    lo, hi = cuts["line_score"]["0.1"], cuts["line_score"]["0.9"]
    raw = (c["line"] - lo) / (hi - lo)
    return raw.clip(lower=0.0, upper=1.0).fillna(1.0) * (1.0 - floor) + floor


def _line_toggle(c, entry, ex, risk, *, kind: str, q: float = 0.5, floor: float = 0.3):
    cuts = v2_cuts()
    size = None
    if kind == "gate":  # 자기분위 상대 게이트: line 이 dev/val q분위 이상이어야 진입
        thr = cuts["line_score"][str(q)]
        entry = entry & (c["line"] >= thr)
    elif kind == "gate_exit":  # 게이트 + line 약세(p10) 청산 가산
        thr = cuts["line_score"][str(q)]
        weak = c["line"] < cuts["line_score"]["0.1"]
        entry = entry & (c["line"] >= thr)
        ex = ex | weak
        risk = risk | weak
    elif kind == "momentum":  # line 상승(line_mom>0) 진입 확인
        entry = entry & (c["line_mom"] > 0.0)
    elif kind == "confirm":  # line ≥ median & base
        entry = entry & (c["line"] >= cuts["line_score"]["0.5"])
    elif kind == "sizing":  # 진입은 base 그대로, 포지션 크기를 line 으로 조절
        size = _line_size(c, floor)
    return entry, ex, risk, size


# ─────────────────────────────────────────────────────────────────────────────
# config + 신호 + 분수 metric
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class V2Config:
    archetype: str
    params: dict = field(default_factory=dict)
    band: dict | None = None  # {q, mode}
    line: dict | None = None  # {kind, q?, floor?}
    entry_confirm_days: int = 2
    exit_confirm_days: int = 3

    @property
    def toggle_tag(self) -> str:
        parts = []
        if self.line:
            lk = self.line["kind"]
            tag = f"line.{lk}"
            if lk in ("gate", "gate_exit"):
                tag += f"q{self.line.get('q', 0.5)}"
            if lk == "sizing":
                tag += f"f{self.line.get('floor', 0.3)}"
            parts.append(tag)
        if self.band:
            parts.append(f"band.{self.band['mode']}q{self.band['q']}")
        return "_".join(parts) if parts else "none"

    @property
    def id(self) -> str:
        return f"{self.archetype}|{_ptag(self.params)}|{self.toggle_tag}"

    @property
    def label(self) -> str:
        return self.id


def v2_raw(frame: pd.DataFrame, cfg: V2Config):
    c = _extract(frame)
    entry, ex, risk = ARCHETYPES[cfg.archetype](c, cfg.params)
    size = None
    if cfg.line:
        entry, ex, risk, size = _line_toggle(c, entry, ex, risk, **cfg.line)
    if cfg.band:
        entry, ex, risk = _band_toggle(c, entry, ex, risk, **cfg.band)
    return entry, ex, risk, size


def v2_signal_frame(ticker_slice: pd.DataFrame, cfg: V2Config) -> pd.DataFrame:
    frame = ticker_slice.sort_values("date").copy()
    entry, ex, risk, size = v2_raw(frame, cfg)
    frame = _run_state_machine(frame, entry, ex, risk, cfg)
    frame["size"] = size.to_numpy() if size is not None else np.ones(len(frame), dtype=float)
    return frame


def _max_drawdown(returns: np.ndarray) -> float:
    usable = np.nan_to_num(returns, nan=0.0)
    if usable.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + usable)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _ann_metric(returns: np.ndarray, downside_only: bool) -> float:
    usable = returns[np.isfinite(returns)]
    ref = usable[usable < 0] if downside_only else usable
    if ref.size < 2:
        return 0.0
    std = float(np.std(ref, ddof=1))
    return float(np.mean(usable) / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def v2_metrics(signal_frame: pd.DataFrame) -> dict[str, Any]:
    """분수 포지션(effective = position × size) full metric suite."""
    frame = signal_frame.sort_values("date")
    returns = frame["daily_return"].to_numpy(dtype=float)
    pos = frame["position"].to_numpy(dtype=float) * frame["size"].to_numpy(dtype=float)
    shifted = np.concatenate([[0.0], pos[:-1]])
    trades = np.abs(np.diff(shifted, prepend=0.0))
    sret = shifted * returns - trades * FEE_RATE
    n = len(frame)

    def total(a):
        a = np.nan_to_num(a, nan=0.0)
        return float(np.prod(1.0 + a) - 1.0) if a.size else 0.0

    strat_tot = total(sret)
    bh_tot = total(returns)
    mdd = _max_drawdown(sret)
    ann = max(1.0 + strat_tot, 1e-6) ** (TRADING_DAYS / max(n, 1)) - 1.0
    # 대형손실 회피
    finite = returns[np.isfinite(returns)]
    if finite.size:
        thr = min(-0.02, float(np.nanpercentile(finite, 20)))
        ll_mask = returns <= thr
        ll_days = int(ll_mask.sum())
        avoided = int(((shifted <= 0.0) & ll_mask).sum())
    else:
        ll_days = avoided = 0
    win_days = sret[shifted > 0.0]
    return {
        "calmar": ann / max(abs(mdd), DD_FLOOR),
        "sortino": _ann_metric(sret, True),
        "sharpe": _ann_metric(sret, False),
        "maxDrawdownPct": mdd * 100.0,
        "totalReturnPct": strat_tot * 100.0,
        "buyHoldReturnPct": bh_tot * 100.0,
        "excessReturnPct": (strat_tot - bh_tot) * 100.0,
        "largeLossAvoidanceRate": (avoided / ll_days) if ll_days else None,
        "winRate": float(np.mean(win_days > 0)) if win_days.size else None,
        "tradeCount": int(trades.sum()),
        "avgHoldDays": _avg_hold(frame["position"].to_numpy(dtype=int)),
        "marketParticipationRate": float(np.mean(shifted > 0.0)),
        "cashWaitRatio": float(np.mean(shifted <= 0.0)),
        "exposed": bool(np.any(shifted > 0.0)),
        "nDays": n,
    }


def _avg_hold(positions: np.ndarray) -> float | None:
    durations, start, current = [], None, 0
    for i, p in enumerate(positions):
        if current == 0 and p == 1:
            start = i
        if current == 1 and p == 0 and start is not None:
            durations.append(i - start)
            start = None
        current = p
    if current == 1 and start is not None:
        durations.append(len(positions) - start)
    return float(np.mean(durations)) if durations else None


# ─────────────────────────────────────────────────────────────────────────────
# config 그리드 생성 (~수천)
# ─────────────────────────────────────────────────────────────────────────────


def _toggle_space() -> list[dict]:
    """base 당 토글 변형 (none/band/line/line&band)."""
    toggles: list[dict] = [{"band": None, "line": None}]  # none
    band_variants = [
        {"q": q, "mode": m} for q in [0.05, 0.10, 0.15] for m in ["lower", "width", "both"]
    ]
    line_variants = (
        [{"kind": "gate", "q": q} for q in [0.4, 0.5, 0.6]]
        + [{"kind": "gate_exit", "q": q} for q in [0.4, 0.5, 0.6]]
        + [{"kind": "momentum"}, {"kind": "confirm"}]
        + [{"kind": "sizing", "floor": f} for f in [0.3, 0.5]]
    )
    for b in band_variants:
        toggles.append({"band": b, "line": None})
    for ln in line_variants:
        toggles.append({"band": None, "line": ln})
    # line&band 대표 조합 (전부 곱하면 폭주 → 대표만)
    combo_lines = [
        {"kind": "gate", "q": 0.5},
        {"kind": "sizing", "floor": 0.3},
        {"kind": "momentum"},
    ]
    combo_bands = [{"q": 0.10, "mode": "both"}, {"q": 0.10, "mode": "lower"}]
    for ln in combo_lines:
        for b in combo_bands:
            toggles.append({"band": b, "line": ln})
    return toggles


def generate_grid() -> list[V2Config]:
    configs: list[V2Config] = []
    toggles = _toggle_space()
    for arch, grid in ARCHETYPE_GRID.items():
        for params in _param_grid(grid):
            for tg in toggles:
                configs.append(
                    V2Config(archetype=arch, params=params, band=tg["band"], line=tg["line"])
                )
    return configs


def n_tested(configs: list[V2Config]) -> int:
    """Bonferroni 분모 = none 아닌 config 수."""
    return sum(1 for cfg in configs if cfg.band or cfg.line)


# ─────────────────────────────────────────────────────────────────────────────
# per-ticker 델타/절대 metric row (목적별 선정 입력)
# ─────────────────────────────────────────────────────────────────────────────

ABS_METRICS = [
    "calmar",
    "sortino",
    "sharpe",
    "maxDrawdownPct",
    "totalReturnPct",
    "excessReturnPct",
    "largeLossAvoidanceRate",
    "winRate",
    "tradeCount",
    "avgHoldDays",
    "marketParticipationRate",
    "cashWaitRatio",
]
DELTA_METRICS = ["calmar", "sortino", "maxDrawdownPct", "totalReturnPct", "excessReturnPct"]


def base_key(cfg: V2Config) -> str:
    return f"{cfg.archetype}|{_ptag(cfg.params)}"


def group_by_base(configs: list[V2Config]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        bk = base_key(cfg)
        g = groups.setdefault(bk, {"none": None, "variants": []})
        if cfg.band or cfg.line:
            g["variants"].append(cfg)
        else:
            g["none"] = cfg
    return groups


def _row(ticker, bk, cfg, m, none_m, degenerate) -> dict[str, Any]:
    row = {
        "ticker": ticker,
        "base_key": bk,
        "config_id": cfg.id,
        "archetype": cfg.archetype,
        "toggle": cfg.toggle_tag,
        "degenerate": degenerate,
    }
    for k in ABS_METRICS:
        row[k] = m[k]
    for k in DELTA_METRICS:
        a, b = none_m[k], m[k]
        row["d_" + k] = (b - a) if (a is not None and b is not None) else None
    la_n, la_v = none_m["largeLossAvoidanceRate"], m["largeLossAvoidanceRate"]
    row["d_lla"] = (la_v - la_n) if (la_n is not None and la_v is not None) else None
    return row


def v2_ticker_rows(ticker: str, ticker_slice: pd.DataFrame, grouped: dict[str, dict]) -> list[dict]:
    """한 ticker 의 전 config row (절대 metric + none 대비 델타). none 자체도 1 row(델타 0)."""
    rows: list[dict] = []
    for bk, grp in grouped.items():
        none_m = v2_metrics(v2_signal_frame(ticker_slice, grp["none"]))
        rows.append(_row(ticker, bk, grp["none"], none_m, none_m, not none_m["exposed"]))
        for cfg in grp["variants"]:
            vm = v2_metrics(v2_signal_frame(ticker_slice, cfg))
            degen = (not none_m["exposed"]) and (not vm["exposed"])
            rows.append(_row(ticker, bk, cfg, vm, none_m, degen))
    return rows


__all__ = [
    "v2_cuts",
    "ARCHETYPES",
    "ARCHETYPE_GRID",
    "V2Config",
    "v2_raw",
    "v2_signal_frame",
    "v2_metrics",
    "generate_grid",
    "n_tested",
    "ABS_METRICS",
    "DELTA_METRICS",
    "base_key",
    "group_by_base",
    "v2_ticker_rows",
]
