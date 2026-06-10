# CP252 지시서 — AI 보조지표 전략 발굴 실험 (다양·대규모, find-only)

작성 2026-06-09 · 상태: 발주(빡센 재설계) · 선행: CP248~251(완료, 결과 빈약) · 담당: 구현+자가리뷰
> **find-only CP.** 방어형·공격형·균형형 후보를 다양·엄격하게 *찾아오기*까지. 프론트 부착은 **확인 후 별도 CP**.

---

## 0. 왜 다시 (CP248~251 실측 한계)
- 후보가 **전부 밴드 기반**(base_balance/trend/pullback __band), **라인 살아남은 것 0**.
- config **15개**·base **4계열**(balance/trend/pullback/balance_loose)뿐 = **탐색 너무 얇음**.
- 헤드라인 게이트(ΔCalmar&Sortino 단일) **0/3** → MDD 방어 1개(ai_band_defense_v2)만 승급. **공격형(수익추종)·라인 가치 미탐색.**
- 결론: 방법론(티커+시간 OOS, selection-bias 가드)은 **견고**(pullback 노이즈 정확 적발). **탐색 폭과 목적을 확장**하면 됨.

## 1. 재사용 자산 (확장 대상)
- `backend/app/services/strategy_ablation.py`(base 정의), `strategy_ablation_metrics.py`·`_report.py`·`_split.py`, `backend/scripts/run_ablation_screening.py`, `backend/data/ablation/`, `test_strategy_ablation.py`.
- rigor(2축 OOS·Bonferroni·deflated) 유지, **폭만 키운다**.

## 2. 확장 설계 (빡세게)

### 2.1 base archetype 대폭 확장 (4 → 8+ 계열)
| 계열 | 코어 신호 | 비고 |
|---|---|---|
| 추세추종 (기존) | MA60/MA20 정렬·MACD | 유지 |
| 눌림목 (기존) | BB하단·RSI | 유지 |
| 균형 (기존) | 추세+눌림목 | 유지 |
| **모멘텀** (신규) | MACD/ROC 가속·신고가 추종 | 공격형 핵심 |
| **돌파** (신규) | N일 고점·거래량 동반 돌파 | 공격형 |
| **순수 평균회귀** (신규) | RSI 극단·가격 zscore | |
| **저변동 진입** (신규) | ATR·vol_change 게이트 | |
| **추세+모멘텀 복합** (신규) | 추세 위 모멘텀 가속 | 공격형 |
각 archetype = 진입/청산 규칙 + 파라미터.

### 2.2 AI 토글 — 라인에 공정한 기회
- 전 조합: `none / +line / +band / +line&band`.
- **라인 메커니즘 강화**(단순 임계 1개 아님): line_score 진입 필터 + 추세 확인 + line_score **순위/모멘텀** + (선택) line 기반 사이징. 라인을 *진입 타이밍·강도*에 쓰는 변형을 명시 포함 — 이전에 라인이 죽은 건 메커니즘이 얇아서일 수 있음.

### 2.3 config 그리드 대규모 (15 → 수백~수천)
- 임계값을 K단계씩 × archetype × 토글. **총 config 수 기록**(Bonferroni 분모·deflated). 계산 큼 → 워크플로 병렬.

### 2.4 목적 3트랙 — **목적별로 따로 선정** (핵심 수정)
이전엔 Calmar&Sortino 단일 게이트라 전부 0/3. 목적별 기준을 분리한다:
- 🛡 **방어형**: ΔMaxDD·ΔLLA 최대 (수익은 −허용범위 내). [이미 band 작동 확인]
- ⚔ **공격형**: 총수익·excess(vs buy&hold)·수익추종 최대 (MDD 허용범위 내). [**라인·모멘텀·돌파 기회**]
- ⚖ **균형형**: ΔCalmar·ΔSortino 최고 (위험조정). [엄격 게이트 유지]
각 트랙별 OOS 통과 후보를 *따로* 산출 (한 게이트로 뭉뚱그리지 않음).

### 2.5 metric full (절대값 + AIon−off 델타)
Calmar·Sortino·Sharpe·MaxDD·총수익·excess·대형손실회피(LLA)·승률·거래수·평균보유일·시장참여율·현금대기율.

### 2.6 OOS (rigor 유지)
- 티커축: dev **200** / held-out **271** (겹침0).
- 시간축: 티커 내 val(앞70%, 탐색·선정) / test(뒤30%, 보고 1회·peeking 금지).
- 통계: 티커 분포 → 부트스트랩 CI + Wilcoxon + **Bonferroni(전체 config 수)** + deflated. **selection-bias 가드** 유지(dev best가 노이즈면 OOS에서 탈락).
- 데이터 현실: 공통창 ~250거래일, `regime_label`(VIX/DD 없음, proxy), stress 6%→aggregate.

## 3. Phase
- **P0** archetype·그리드·목적·선정기준 스펙 확정 — **착수 전 사용자 확인**.
- **P1** 하니스 확장: `strategy_ablation.py`에 신규 archetype + `run_ablation_screening.py` 그리드 확대 + `strategy_ablation_metrics/report`에 **목적 3트랙 선정**. 기존 `ai_band_defense_v2` 출력 회귀0(스냅샷).
- **P2** dev 200 대규모 탐색 (전 archetype×토글×그리드) → val 성능 + 목적별 dev 후보.
- **P3** 목적별 후보 → held-out 271 test 1회 + 통계 → 목적별 OOS 후보.
- **P4** 산출: **방어/공격/균형 목적별 후보 랭킹표(full metric+통계)** + 정직성 평결(라인·공격형이 실제 되나/안 되나를 *넓은* 탐색으로 결론).
- **★ 게이트**: 사용자가 목적별 N개 선정 → 부착은 다음 CP.

## 4. 제약
새 학습·calibration·DB write 금지 · 운영 parquet read-only · **기존 승급 전략(`ai_band_defense_v2`) 출력 회귀0** · facade 계약 유지 · **selection-bias 정직 보고**(config 총수·2축 OOS·Bonferroni 명시) · 1W밴드는 데이터 얇아 옵션 트랙.

## 5. 산출물 / 완료 기준
- **목적별(방어·공격·균형) 후보 전략 + full metric + 통계 평결** 리포트.
- "라인/공격형이 넓은 탐색에서도 되는가"에 대한 정직한 OOS 결론.
- 부착(라이브 등록·UI)은 **별도 CP**. 이 CP는 다양·대규모로 **찾아오기**까지.
