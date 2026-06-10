# CP253 보고서 — 발굴 방어 전략 프론트 부착 (attach)

작성 2026-06-10 · 상태: 완료 · 선행: CP252(발굴·검증) · 담당: 구현+자가리뷰

> CP252에서 OOS·엄격 Bonferroni(1300) 통과한 방어 후보 3개를 라이브 전략으로 등록(구 약전략 교체).
> 프론트 형식 동일 유지. 공격/균형은 OOS null이라 미부착(정직 기록). CP248~251(v1) 겹침 산출물 제거.

---

## 0. 한 줄 결과

CP252 검증 방어 전략 3개를 운영 경로(`_raw_target`)로 **byte-identical 포팅**해 라이브 등록(+대조군 유지),
구 `ai_band_defense_v2` 교체, CP248~251(v1) ablation 코어·데이터·문서 제거. **AI는 수익엔 기여 못함을
정직히 표기**하고 방어 전용으로만 노출. 회귀 0, RSS 327MB(<512).

## 1. 라이브 등록 — 신규 방어 3 + 대조군

| id | 라벨 | 구성 | held-out ΔMaxDD | 참여율 | 손실회피 |
|---|---|---|--:|--:|--:|
| `lineband_risk_guard` | 라인밴드 리스크 가드 | 모멘텀+라인gate(p50)+밴드(both) | +6.0%p (n255) | 23% | 76.8% |
| `lineband_defense` | 라인밴드 방어 | 눌림목+라인모멘텀+밴드(both) | +7.8%p (n98) | 6% | 93.8% |
| `line_defense` | 라인 방어 | 모멘텀+라인gate(p60), 밴드無 | +4.5%p (n255) | 25% | 73.7% |
| `indicator_balance_v2` | 지표 균형 | no-AI 대조군 (유지) | — | 63% | — |

- 이름 = 사용 지표 노출(괄호 AI 표기 금지, 사용자 확정). 배지 없음·정직 카피는 카드 설명/범위주(사용자 확정).
- ★정직(§1.1): 헤드라인 ΔMaxDD는 **위험 종목 회피(현금)** 비중이 큼. 평균 참여율 6~25%(현금 多) =
  매우 보수적 "리스크 가드". 카드에 "수익엔 기여 못함(공격형 OOS null·상승장 caveat), 위험할 때 현금으로
  빠져 낙폭을 줄이는 보수 전략" 명시. 과대광고 없음.

## 2. 포팅 방식 + 정확성

- `strategy_ablation_v2` 로직(momentum/pullback archetype + 라인 gate/momentum + 밴드 both)을
  운영 `strategy_indicators._raw_target` 로 포팅. frozen 컷은 `strategy_rules` 상수(v2_cuts.json 출처).
- 파생 피처(roc_20·macd_accel·ma_5_ratio·line_mom)를 라이브 프레임 로더(`strategy_scan._load_base_frame`/
  `_merge_line`)에 v2 와 동일 계산으로 추가(누수 없는 당일까지 shift). 기존 컬럼 불변.
- **포팅 정확성 검증**: 라이브 `_compute_signal_frame` == v2 `v2_signal_frame` 포지션·신호 **byte-identical**
  (8티커, `test_strategy_ablation.py`). frozen 컷 == v2_cuts.json (테스트 가드).

## 3. 제거 — 구 약전략 + CP248~251(v1) 겹침 (§2.5)

- 라이브: `ai_band_defense_v2`(CP251, +2% 방어 → 신규 3개가 2~4배) 교체. (ai_balance_v2/ai_band_defense_v1 은
  CP251에서 이미 제거.)
- v1 코드: `strategy_ablation.py` · `strategy_ablation_report.py` · `run_ablation_screening.py` 삭제.
  - `strategy_ablation_metrics.py` 는 v2 가 쓰는 `_bootstrap_ci`/`_wilcoxon_p` 만 남기고 strip(v1 의존 제거).
  - `strategy_ablation_split.py` 의 `band_toggle_from_split`(v1 의존) 제거. **유지**: load_split·
    load_ablation_frame_v2·iter_ticker_slices·ticker_split.json(v2 공유).
- v1 데이터: `cp249_dev_screening.*` · `cp250_candidates.json` · `cp250_heldout_validation.*` 삭제.
- v1 문서: `cp248`~`cp251` directive/report 삭제(narrative 는 cp252_report 가 인용). cp252/253 만 유지.
- `test_strategy_ablation.py` 는 CP253 포팅 가드 + v2 basics 로 재작성(구 v1 REPRO 제거).
- **제거 전 grep importer 0 확인** 완료(라이브·테스트·v2 무참조). import sanity·전체 테스트 통과.

## 4. 프론트 (형식 변경 0)

- `constants.ts`/`types.ts`: 신규 3 카드(정직 카피·검증숫자) + 대조군 유지, `ai_band_defense_v2` 제거.
  `strategyNeedsLine`(3개 다)·`strategyNeedsBand`(A·B) 갱신. signal_group 4단계·카드 필드·응답 스키마 불변.
- tsc PASS · Vitest 166 passed · 잔여 참조 0.

## 5. 검증 (회귀 0)

| 항목 | 결과 |
|---|---|
| 포팅 정확성(라이브==v2) | byte-identical (3전략 × 8티커) |
| frozen 컷 == v2_cuts.json | 테스트 통과 |
| indicator_balance_v2 스냅샷 | **9/9 불변** |
| 신규 3전략 scan/backtest | 정상(scope 471, 4단계·full metric) |
| 백엔드 전체 | 156 passed / 12 pre-existing fail(무관) |
| frontend tsc·Vitest | PASS · 166 passed |
| ruff(변경 전부) | All checks passed |
| RSS(4전략 scan+backtest) | **327MB < 512** |

## 6. 정직성 / §8 결론

- **라인 부활**: CP248~251 "라인 죽음"은 옛 절대임계(-2%, 진입 63% 차단) 탓 — CP252 상대 분위 게이트로
  교정하니 방어에 유효(라이브 3전략 모두 라인 사용, 라이브 ai_band_defense_v2 대비 2~4배 방어).
- **공격형 null**: 1352 config·8 archetype·모멘텀·돌파·분수 사이징 다 동원해도 buy-hold 못 이김 →
  미부착, 정직 기록(상승장 caveat). 균형형도 OOS collapse(selection-bias 가드 작동).
- 한계: frozen 컷은 현 라인/밴드 모델 분포 기준 — 재학습 시 재보정 필요(주석 명시, v2 과제).

## 7. 커밋

검증 전부 통과(스냅샷 회귀0·RSS<512·포팅 일치) 후 신규 전략 등록 + v1 정리 + 보고서를 묶어 main 커밋·푸쉬.
