# CP252 보고서 — AI 보조지표 전략 발굴 실험 (다양·대규모, find-only)

작성 2026-06-10 · 상태: 완료(find-only, ★게이트 대기) · 선행: CP248~251 · 담당: 구현+자가리뷰

> 넓고 엄격한 탐색으로 방어/공격/균형 후보를 **찾기**까지. 부착(라이브 등록·UI)은 별도 CP.

---

## 0. 한 줄 결과

8 archetype × AI 토글(라인 게이트/모멘텀/확인/**분수 사이징** + 밴드 컷×모드) **1352 config**을
dev 200(val) 전면 탐색 → 목적 3트랙 후보 → held-out 271(test) 1회 검증. **핵심 발견 2개**:
- **라인이 방어용으로 부활** — recalibrated line gate/momentum 이 낙폭을 OOS·엄격 Bonferroni 로 깎는다.
  최강 방어는 **라이브 ai_band_defense_v2(+2%)의 3배**(+6~8% MaxDD, 수익비용 ~0).
- **공격형은 넓은 탐색에서도 null** — 0/1352 가 buy-hold 못 이김(불장 caveat). 균형형도 OOS collapse.

---

## 1. 설계 (CP248 하니스 확장)

- **archetype 8계열**(52 base 인스턴스): 추세·눌림목·균형(기존) + 모멘텀·돌파·평균회귀·저변동·추세+모멘텀(신규).
- **AI 토글 26종/base**: 밴드 컷{p5/p10/p15}×모드{lower/width/both} · 라인 gate{q.4/.5/.6}·gate_exit·momentum·
  confirm·**sizing(분수 포지션)** · 대표 line&band 조합.
- **분수 엔진**: 이진 long/cash 위 size∈[floor,1] 곱(라인 사이징). full metric 12종.
- **컷 dev/val frozen**(`v2_cuts.json`, 누수 0). 라이브 `ai_band_defense_v2`·CP248 경로 **불변**(병렬 모듈).
- 실행: `run_v2_discovery.py` 멀티프로세싱(14워커). dev 270,400행 148초. n_tested=1300.

## 2. dev 탐색 (200티커 val) — selection-inflation 가드

| 트랙 | dev best | observed_max | E[max under null] | 초과? |
|---|---|--:|--:|:--:|
| 방어 ΔMaxDD | pullback+line.mom+band +8.7 | 8.74 | 11.50 | ✗ |
| 공격 excess | (전부 음수) −5.06 | −5.06 | +21.18 | ✗ |
| 균형 ΔCalmar | breakout+AI +0.36 | 0.36 | 1.19 | ✗ |

→ 세 트랙 모두 dev-best 가 noise 벤치마크 미달 = **과적합 위험 높음, held-out 이 진짜 판정**(정직 표기).
정직성: 탐색 config **1352**(Bonferroni 분모 1300, α_strict=3.8e-5). 선정은 dev=val 한정.

## 3. ★ held-out 검증 (271티커 test, 1회) — 목적별 평결

### 🛡 방어형 — **강력, OOS 재현 + 엄격 Bonferroni(1300) 통과**

| config | ΔMaxDD (CI) | Wilcoxon p | strict | ΔLLA | Δ수익 | n |
|---|---|--:|:--:|--:|--:|--:|
| momentum·line.gateq0.5·band.both | **+6.03** [+5.24,+6.87] | ≈0 | ✅ | +0.22 | **−0.02** | 255 |
| momentum·line.gateq0.5·band.both(ma5.005) | +6.14 [+5.31,+6.99] | ≈0 | ✅ | +0.22 | −0.02 | 248 |
| pullback·line.momentum·band.both | **+7.89** [+6.87,+9.01] | ≈0 | ✅ | +0.37 | −1.29 | 101 |
| momentum·line.gateq0.6 (라인 단독) | +4.51 [+3.76,+5.27] | ≈0 | ✅ | — | −1.1 | 255 |

→ **낙폭 −6~8% 감소를 OOS·100~255티커에서 재현, 전체 1300 비교 Bonferroni 도 통과.** dev 에서 컸던
수익 비용(−8%)이 변동성 큰 held-out 에선 **~0**(방어가 실제 낙폭이 있을 때 값을 함). **라이브
ai_band_defense_v2(+2.0%)의 3배.** **라인이 핵심**(line.gate/momentum) → CP248-251 "라인 죽음"은
망가진 절대임계(-0.02, 63% 차단) 탓이었음을 넓은 탐색이 교정. breakout 방어(n=9-14)는 strict 미달(fragile).

### ⚔ 공격형 — **null 확정 (OOS)**

전 후보 excess **음수**(−2.2~−3.8%, breakout −35%). best offensive = none 토글 재래(가장 많이 투자),
그래도 buy-hold 에 −2%. d_excess(AI 기여) ~0. **8 archetype·모멘텀·돌파·분수 다 동원해도 수익 우위 0.**
caveat: 평가창(2025-06~2026-06)이 net 상승 → long/cash 구조적 불리(현금 보유 = 기회비용). 하락/횡보장은
다를 수 있으나 **현 데이터에선 수익추종 전략 부재**.

### ⚖ 균형형 — **null, collapse 확인**

dev 양수 ΔCalmar(breakout, +0.36)가 OOS 에서 무너짐(breakout+band.lower +0.60 but n=14 p=0.74 ns;
다른 breakout −0.08~−0.14). 견고(n=245)한 건 ΔCalmar +0.01(≈0). **risk-adjusted 개선 후보 0** —
dev 외관 승자는 selection 노이즈였고 **두 축 OOS 가드가 정확히 적발**(CP250 pullback 붕괴와 동일 패턴).

## 4. 정직한 결론 (find-only)

1. **라인 = 방어 신호로 부활.** recalibrated line gate/momentum 이 낙폭·대형손실을 OOS·엄격 통계로 깎음.
   최강 방어 후보가 라이브 전략의 3배 → **차기 CP 에서 ai_band_defense_v2 교체 후보**.
2. **공격형 = 넓은 탐색에서도 null.** 수익 우위 전략 부재(불장 caveat). AI 가 수익에 기여 못함 — CP248~250
   결론을 대규모로 재확인.
3. **균형형 = null.** risk-adjusted 개선 후보 없음. selection-bias 가드 견고.

## 5. ★ 게이트 (사용자 판단 — 목적별 N개 선정 → 부착은 다음 CP)

- 방어형 강력 후보(전부 strict 통과): `momentum·line.gateq0.5·band.both`(공짜 방어, n255) /
  `pullback·line.momentum·band.both`(최강 −8% MaxDD, n101) / `momentum·line.gateq0.6`(라인 단독·단순, n255).
- 공격·균형: 승급 후보 없음(정직 null) — 보고서 기록.

산출물: `v2_dev_discovery.*`(rows/agg/json) · `v2_candidates.json`(고정 후보) · `v2_heldout_validation.*` ·
`v2_cuts.json`. 코드 `strategy_ablation_v2*.py` · `run_v2_discovery.py`. 라이브·CP248 경로 회귀 0.
