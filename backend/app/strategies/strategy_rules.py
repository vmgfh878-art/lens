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

# CP253 — 발굴(CP252) 에서 held-out 271 test·Bonferroni(1300) 통과한 방어 전략의 frozen 컷.
# 출처 backend/data/ablation/v2_cuts.json (dev/val 분위, 누수 0). 라인/밴드를 절대값이 아니라
# "이 종목 기준 상대 분위"로 읽어 위험 국면에서 노출을 줄인다. 운영 1D 라인/밴드 모델(CP153/210)
# 분포 기준 — 재학습으로 분포가 바뀌면 재보정 필요(v2 과제, 구 절대임계가 죽은 드리프트와 동일).
LINE_GATE_Q50 = -0.03742746263742447  # line_score dev/val p50 (gate: 이 이상이어야 진입)
LINE_GATE_Q60 = -0.026474294364452363  # line_score dev/val p60 (더 엄격한 gate)
BAND_LOWER_P10 = -0.027661561347678353  # band_lower_return dev/val p10 (깊은 하단 = 꼬리위험)
BAND_WIDTH_P90 = 1.0452357822024705  # band_width_expansion dev/val p90 (넓은 폭 = 고변동)


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
    # CP253 — CP252 발굴에서 held-out 271 test·엄격 Bonferroni(1300) 통과한 방어 전략 3개.
    # 구 ai_band_defense_v2(CP251, +2% 방어)를 이들(2~4배 강함)로 교체. 모두 **보수적 리스크
    # 가드**: 위험 국면엔 현금으로 빠져 낙폭·대형손실을 줄이되 평균 참여율 24~33%(현금 多).
    # ★정직: AI 는 수익엔 기여 못함(공격형 OOS null, 상승장 caveat) — 방어에만 효과.
    "lineband_risk_guard": StrategyRule(
        id="lineband_risk_guard",
        label="라인밴드 리스크 가드",
        short_label="리스크 가드",
        uses_ai=True,
        uses_line=True,
        uses_band=True,
        entry_confirm_days=2,
        exit_confirm_days=3,
        entry_desc=(
            "모멘텀 진입(ROC20 ≥ +2% & MACD 가속 ≥ 0 & MA5 ≥ 0 & RSI < 80) & "
            "AI 라인 ≥ 중간분위(p50) & 밴드 위험-상태 아님."
        ),
        exit_desc=(
            "ROC20 < 0 또는 MA60 ≤ -5% 또는 MA20 ≤ -6% 또는 (ATR > 10% & MA20 < 0) "
            "또는 밴드 위험-상태."
        ),
        risk_desc=(
            "밴드 위험-상태(하단이 이 종목 p10 이하로 깊거나 밴드폭이 p90 이상으로 넓음) 또는 "
            "추세·변동성 붕괴. 위험하면 현금 — 보수적 방어(낙폭 −6%, 수익 비용 작음)."
        ),
    ),
    # CP253 후속 — lineband_defense(눌림목+라인모멘텀+밴드)는 참여율 6%로 너무 선별적이라
    # 화면에서 볼 게 없어 제거(사용자 결정). 나머지 둘은 ~23-25% 참여.
    "line_defense": StrategyRule(
        id="line_defense",
        label="라인 방어",
        short_label="라인 방어",
        uses_ai=True,
        uses_line=True,
        uses_band=False,
        entry_confirm_days=2,
        exit_confirm_days=3,
        entry_desc=(
            "모멘텀 진입(ROC20 ≥ +2% & MACD 가속 ≥ 0 & MA5 ≥ 0 & RSI < 80) & "
            "AI 라인 ≥ p60(밴드 미사용 — 라인 단독 게이트)."
        ),
        exit_desc="ROC20 < 0 또는 MA60 ≤ -5% 또는 MA20 ≤ -6% 또는 (ATR > 10% & MA20 < 0).",
        risk_desc=(
            "추세·변동성 붕괴. 밴드 없이 AI 라인만으로 위험 국면 진입을 막는 보수적 방어 "
            "(낙폭 −4.5%). '라인 부활' 쇼케이스 — 옛 라인 임계가 망가졌던 것을 교정."
        ),
    ),
}
