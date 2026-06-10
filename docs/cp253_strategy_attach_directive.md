# CP253 지시서 — 발굴 전략 프론트 부착 (attach)

작성 2026-06-10 · 상태: 발주 · 선행: CP252(완료, 방어 후보 검증) · 담당: 구현+자가리뷰
> CP252에서 OOS·엄격 통계로 검증된 **방어 후보**를 라이브 전략으로 등록(기존 약한 AI 전략 교체). 프론트 형식은 **현재와 동일** 유지. 공격/균형은 null이라 미부착(정직 기록).

---

## 0. 목적
CP252 `v2_candidates.json` 방어 후보를 운영 전략으로 이식 → UI 노출. 기존 약한 AI 전략 교체. **per-ticker 계산 + 4단계 signal_group(buy/hold/risk/watch) 분리 + full metric** = 현 프론트 형식 그대로.

## 1. 부착 대상 — **4개 활성 확정** (3 신규 + 대조군)
held-out 271 test·Bonferroni(1300) 통과 방어 후보 전부 부착. 임시 라벨(§4에서 확정):
- **A `라인밴드 리스크 가드`** `momentum|ma50.0-roc0.02|line.gateq0.5_band.bothq0.1` — 라인+밴드, ΔMaxDD +6.0%(헤드라인), 커버리지 255/271.
- **B `라인밴드 방어`** `pullback|bb0.3-rsi50.0|line.momentum_band.bothq0.1` — 눌림목+라인모멘텀+밴드, ΔMaxDD +7.9%(최강), 커버리지 98/271(선별적).
- **C `라인 방어`** `momentum|ma50.0-roc0.02|line.gateq0.6` — **라인 단독**(밴드 없음), +4.5%, 커버리지 255/271. "라인 부활" 쇼케이스.
- **유지 `지표 균형`** `indicator_balance_v2` (no-AI 대조군 — "재래 vs +AI 방어" 직접 비교축).
- 이름 규칙: **괄호 AI 표기 대신 사용 지표(라인/밴드)를 이름에 직접 노출.**
- **미부착**: 공격형·균형형 = OOS null(보고서 §8에 정직 기록).

### 1.1 ★ 정직성 (UI·보고서 표현의 근거)
held-out 분해상 방어는 **① 위험 종목 아예 회피(현금)** + **② 거래하며 낙폭 방어** 두 형태. 헤드라인 ΔMaxDD(+6~8%)는 ①(회피) 비중이 크고, **실제 거래 중 방어는 +1.5~3%(수익 −1~2% 비용), 평균 참여율 24~33%**. → 이들은 **매우 보수적(현금 多)인 "리스크 가드"**다. UI는 이 성격을 정직히("위험할 때 현금으로 빠져 낙폭을 줄이는 보수 전략") 표현. 과대광고 금지.

## 2. 등록 방식
- 각 후보 config(archetype 규칙 + AI 토글 + **frozen 컷** `v2_cuts.json`)를 `strategy_rules.STRATEGIES` + `strategy_backtest_svc._raw_target`로 이식. CP252 `strategy_ablation_v2*.py` 로직을 운영 경로로 *포팅*(재현 일치 검증).
- **기존 약한 AI 전략 교체/제거**: `ai_band_defense_v2`(A로 대체·3배 강함), `ai_balance_v2`·`ai_band_defense_v1`(구 부실). `indicator_balance_v2` 유지.
- frozen 컷은 현 밴드/라인 모델 분포 기준 — 재학습 시 재보정 필요(주석 명시, v2 과제).

## 2.5 겹치는 CP248~251(v1) 산출물 제거 ← 사용자 지시
CP252(v2)가 CP248~251(v1, 아쉬운 첫 시도)을 대체. **v2가 v1 일부를 import하므로 의존 확인 후 superseded만 제거:**
- **유지(필수, v2 런타임 의존)**: `strategy_ablation_metrics.py`(`_bootstrap_ci`/`_wilcoxon_p`), `strategy_ablation_split.py`(`load_split`), `data/ablation/ticker_split.json`(공유 분할).
- **제거(v1 전용·v2/v2_* 가 대체, 라이브·런타임 import 0 확인)**:
  - 코드: `strategy_ablation.py`(v1 base — v2는 `strategy_ablation_v2.py`에 자체 8 archetype), `strategy_ablation_report.py`(importer 0 확인 후), `run_ablation_screening.py`(→ `run_v2_discovery.py` 대체).
  - 데이터: `data/ablation/cp249_dev_screening.*` · `cp250_candidates.json` · `cp250_heldout_validation.*`(→ `v2_*` 대체).
  - 교체 전략: 라이브 `ai_band_defense_v2`(CP251 승급) → 신규 4전략으로 대체·제거 + 구 `ai_balance_v2`/`ai_band_defense_v1` 제거.
  - 문서: `cp248`~`cp251` directive/report 는 cp252/253 이 대체 → 제거(중복 narrative). cp252_report 가 첫 시도 한계를 이미 인용함.
  - `test_strategy_ablation.py` 가 v1 심볼 가리키면 v2 로 갱신 또는 제거.
- **제거 전 필수**: 각 파일별 `grep` importer/참조 0(라이브·테스트·v2) 확인 → 안전한 것만. 깨지면 되돌림.

## 3. 프론트 형식 (현재와 동일 — 변경 0)
- **strategy scan**: 티커별 카드 + `signal_group` 4단계(buy/hold/risk/watch) 분리 유지 + 기존 카드 필드(conservativeReturn/lowerBandReturn/ma·rsi·strategyScore 등) full 계산.
- **백테스트**: 기존 단일 종목 long/cash 계약(`single_ticker_long_cash_average`) + full metric(Calmar/Sortino/MDD/대형손실회피/수익/excess 등).
- 응답 스키마·카드 구조 변경 0 — 전략 id/규칙만 교체.

## 4. 표현 (라벨·배지·카피 — 사용자 확인 후)
- **이름 = 사용 지표 노출**(괄호 AI 표기 금지). 확정: A=`라인밴드 리스크 가드` / B=`라인밴드 방어` / C=`라인 방어` / 대조군=`지표 균형`. (배지·설명 카피만 AskUserQuestion으로 추가 확정)
- **정직성 카피**: UI·보고서에 "AI는 **수익엔 기여 못함**(공격형 null, 상승장 caveat), **위험할 때 현금으로 빠져 낙폭을 줄이는 보수적 방어**에만 효과(참여율 24~33%)" 명시. 과대광고 금지.

## 5. 검증
- **출력 회귀0**: `indicator_balance_v2`(no-AI) characterization 스냅샷 불변.
- 신규 3전략 scan/backtest 200 + 형식(4단계·카드 필드) 현재와 동일.
- CP252 산출값과 운영 포팅값 **수치 일치**(이식 정확성).
- **메모리 가드(CP246)**: 전략 4개(3신규+대조)로 scan RSS 재측정 → **<512MB** 확인. 신규는 lazy 유지.

## 6. 제약
새 학습·calibration·DB write 금지 · 운영 parquet read-only · facade 계약 유지 · 표현은 사용자 확인 후 · selection-bias/공격 null 정직 기록.

## 7. 산출물 / 완료 기준 + 푸쉬
- 신규 4 전략(라인밴드 리스크가드 / 라인밴드 방어 / 라인 방어 + 지표균형 대조군) 라이브 등록 + UI + 구 약전략 제거.
- §2.5 v1 겹침 산출물 제거.
- 보고서 §8 갱신: **라인 부활(방어 3배) + 공격 null** 정직 기록.
- RSS<512 확인 + `indicator_balance_v2` 스냅샷 회귀0 + 신규 형식(4단계·카드) 동일.
- **검증 전부 통과 후 main 커밋+푸쉬** ← 사용자 지시. 신규 전략 등록 + v1 정리 + 보고서를 묶어 설명적 커밋 → `git push origin main`. (스냅샷 회귀0·RSS<512 확인 *후에만* push; 실패 시 push 금지.)
