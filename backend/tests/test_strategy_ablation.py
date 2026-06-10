"""CP253 — 발굴 방어 전략 라이브 포팅 + v2 하니스 검증.

핵심 가드:
1. **포팅 정확성** — 라이브 A/B/C(`lineband_risk_guard`/`lineband_defense`/`line_defense`)가
   CP252 발굴 v2 config 와 포지션·신호 byte-identical (동일 프레임). 운영 _raw_target == v2_raw.
2. frozen 컷 일관성 — strategy_rules 상수 == v2_cuts.json.
3. v2 하니스 basics — 그리드 수, full metric suite, 분수 사이징.
4. indicator_balance_v2(no-AI 대조군) 신호 계약 불변.

(CP248~251 v1 ablation 코어는 CP252/253 이 대체하여 제거됨 — 본 테스트는 v2/라이브만.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from app.services.strategy_ablation_split import load_ablation_frame_v2
from app.services.strategy_ablation_v2 import (
    V2Config,
    generate_grid,
    v2_cuts,
    v2_metrics,
    v2_signal_frame,
)
from app.services.strategy_indicators import _compute_signal_frame
from app.strategies import strategy_rules
from app.strategies.strategy_rules import STRATEGIES

# 라이브 전략 ↔ CP252 발굴 v2 config (CP253 부착 매핑).
LIVE_TO_V2 = {
    "lineband_risk_guard": V2Config(
        "momentum",
        {"roc": 0.02, "ma5": 0.0},
        line={"kind": "gate", "q": 0.5},
        band={"q": 0.10, "mode": "both"},
    ),
    "line_defense": V2Config(
        "momentum",
        {"roc": 0.02, "ma5": 0.0},
        line={"kind": "gate", "q": 0.6},
    ),
}
_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO", "PG", "TSLA"]
_FUNC_COLS = ["position", "target_position", "signal_group", "signal_label"]


@pytest.fixture(scope="module")
def frame():
    return load_ablation_frame_v2()


# ── 1. 포팅 정확성 (라이브 == v2) ────────────────────────────────────────────


@pytest.mark.parametrize("live_id", list(LIVE_TO_V2.keys()))
def test_live_reproduces_v2(frame, live_id):
    cfg = LIVE_TO_V2[live_id]
    rule = STRATEGIES[live_id]
    checked = 0
    for tk in _TICKERS:
        tf = frame[frame["ticker"] == tk].sort_values("date").copy()
        if len(tf) < 120:
            continue
        checked += 1
        live = _compute_signal_frame(tf, rule)
        v2 = v2_signal_frame(tf, cfg)
        assert (
            live[_FUNC_COLS].reset_index(drop=True).equals(v2[_FUNC_COLS].reset_index(drop=True))
        ), f"{live_id}/{tk} 신호가 v2 config 와 다름 (포팅 오류)"
    assert checked >= 5


def test_strategies_registry():
    # CP253(+후속) — 대조군 1 + 발굴 방어 2 = 3. lineband_defense 는 참여 6%로 제거됨.
    assert set(STRATEGIES) == {
        "indicator_balance_v2",
        "lineband_risk_guard",
        "line_defense",
    }


# ── 2. frozen 컷 일관성 ──────────────────────────────────────────────────────


def test_frozen_cuts_match_v2_cuts():
    cuts = v2_cuts()
    assert cuts["line_score"]["0.5"] == strategy_rules.LINE_GATE_Q50
    assert cuts["line_score"]["0.6"] == strategy_rules.LINE_GATE_Q60
    assert cuts["band_lower"]["0.1"] == strategy_rules.BAND_LOWER_P10
    assert cuts["band_width"]["0.9"] == strategy_rules.BAND_WIDTH_P90


# ── 3. indicator_balance_v2 계약 불변 ───────────────────────────────────────


def test_control_strategy_contract(frame):
    tf = frame[frame["ticker"] == "AAPL"].sort_values("date").copy()
    sf = _compute_signal_frame(tf, STRATEGIES["indicator_balance_v2"])
    assert set(np.unique(sf["position"])) <= {0, 1}
    assert sf["signal_group"].isin(["buy", "hold", "risk", "watch"]).all()
    assert len(sf) == len(tf)


# ── 4. v2 하니스 basics ──────────────────────────────────────────────────────


def test_v2_grid_shape():
    grid = generate_grid()
    assert len(grid) == 1352
    archetypes = {c.archetype for c in grid}
    assert archetypes == {
        "trend",
        "pullback",
        "balance",
        "momentum",
        "breakout",
        "meanrev",
        "lowvol",
        "trendmom",
    }


def test_v2_metrics_full_suite_and_sizing(frame):
    tf = frame[frame["ticker"] == "AAPL"].sort_values("date").copy()
    # 분수 사이징: size < 1 인 날이 존재.
    cfg = V2Config("balance", {"ma60": 0.02}, line={"kind": "sizing", "floor": 0.3})
    sig = v2_signal_frame(tf, cfg)
    assert sig["size"].min() < 1.0 and sig["size"].min() >= 0.3
    m = v2_metrics(sig)
    for key in [
        "calmar",
        "sortino",
        "sharpe",
        "maxDrawdownPct",
        "totalReturnPct",
        "excessReturnPct",
        "winRate",
        "marketParticipationRate",
    ]:
        assert key in m


def test_v2_cuts_artifact_present():
    path = Path("backend/data/ablation/v2_cuts.json")
    assert path.exists()
    cuts = json.loads(path.read_text(encoding="utf-8"))
    assert "band_lower" in cuts and "line_score" in cuts
