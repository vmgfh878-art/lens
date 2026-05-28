"""
전략 정의 단일 관리 (모델 급 관리).

주식 모델 (line / band) 과 동등한 중요도로 전략을 관리한다.
- 메타 (StrategyRule): id, 라벨, 사용 데이터 (ai/line/band), 진입/청산 확인일.
- 사람이 읽는 명세 (entry_desc / exit_desc / risk_desc): 각 전략이 무엇을 보고
  진입/청산/위험 판정하는지 한국어로 기록. 면접 / 문서 / UI 가이드의 단일 출처.
- 실제 pandas 조건 로직은 strategy_backtest_svc._raw_target 에 strategy id 별 구현.
  여기 description 과 거기 코드가 어긋나면 안 된다 (변경 시 양쪽 같이).

새 전략 추가 절차:
1. 이 파일 STRATEGIES 에 StrategyRule 추가.
2. strategy_backtest_svc._raw_target 에 id 분기 + pandas 조건 추가.
3. (선택) docs/strategy_card_<id>.md 작성.
이 흐름은 자동 갱신 파이프라인 (run_v1_unified_refresh_local.ps1) 과 독립이라
전략을 바꿔도 데이터 갱신 흐름은 변하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyRule:
    id: str
    label: str
    short_label: str
    uses_ai: bool
    uses_line: bool
    uses_band: bool
    entry_confirm_days: int
    exit_confirm_days: int
    # 사람이 읽는 명세 — _raw_target 의 코드 로직과 동기 유지.
    entry_desc: str = ""
    exit_desc: str = ""
    risk_desc: str = ""


STRATEGIES: dict[str, StrategyRule] = {
    "indicator_balance_v2": StrategyRule(
        id="indicator_balance_v2",
        label="지표 균형 v2",
        short_label="지표 균형",
        uses_ai=False,
        uses_line=False,
        uses_band=False,
        entry_confirm_days=2,
        exit_confirm_days=3,
        entry_desc=(
            "추세 진입: MA60 ≥ +2% & MA20 ≥ -2% & MACD ≥ 0 & RSI < 75. "
            "또는 눌림목 진입: MA60 ≥ +2% & BB위치 ≤ 0.35 & RSI < 55."
        ),
        exit_desc="MA60 ≤ -5% 또는 MA20 ≤ -5% 또는 (ATR ≥ 7% & MA20 < 0).",
        risk_desc="청산 조건과 동일 (추세 붕괴 / 변동성 급등).",
    ),
    "ai_balance_v2": StrategyRule(
        id="ai_balance_v2",
        label="AI 균형 v2",
        short_label="AI 균형",
        uses_ai=True,
        uses_line=True,
        uses_band=True,
        entry_confirm_days=2,
        exit_confirm_days=3,
        entry_desc=(
            "line_score ≥ -2% & MA60 ≥ 0 & MA20 ≥ -4% & "
            "(밴드 하단 ≥ -6% 또는 밴드폭 확장 < 1.25x)."
        ),
        exit_desc=(
            "(line 약함 < -6% & 밴드 위험) 또는 MA20 < -10% 또는 (ATR > 12% & MA20 < 0)."
        ),
        risk_desc="밴드 하단 < -6% 또는 밴드폭 확장 > 1.25x 또는 가격/변동성 붕괴.",
    ),
    "ai_band_defense_v1": StrategyRule(
        id="ai_band_defense_v1",
        label="AI 밴드 방어 v1",
        short_label="밴드 방어",
        uses_ai=True,
        uses_line=False,
        uses_band=True,
        entry_confirm_days=2,
        exit_confirm_days=3,
        entry_desc=(
            "(추세: MA60 ≥ +2% & MA20 ≥ -3% & RSI < 82 / 또는 눌림목: MA60 ≥ +2% & "
            "BB ≤ 0.45 & RSI < 60) & 밴드 안정 (하단 ≥ -8% 또는 밴드폭 < 1.60x)."
        ),
        exit_desc=(
            "(밴드 하단 < -8% & 밴드폭 > 1.60x) 또는 (MA60 < -5% 또는 MA20 < -8%) "
            "또는 (ATR > 12% & MA20 < 0)."
        ),
        risk_desc="밴드 스트레스 / 추세 붕괴 / 변동성 급등.",
    ),
}


def list_strategies() -> list[StrategyRule]:
    return list(STRATEGIES.values())
