# CP175-LM 1D Conservative Line Learning Revisit

## 한 줄 결론

최종 판정은 **WARN**이다. beta를 키우면 line 자체가 더 보수적으로 움직이고 false-safe가 줄어드는 신호는 확인됐다. 다만 PASS 기준인 false-safe 절대 감소 3.0pp에는 못 미쳤다. 가장 균형이 좋은 후보는 **beta_5**이고, 더 공격적으로 보면 **beta_7**도 Pareto 후보지만 fee retention이 0.80 아래로 내려간다.

## 범위 준수

- product save-run, DB write, inference 저장, live fetch, EODHD fallback은 수행하지 않았다.
- band/composite 실험은 수행하지 않았다. runtime output은 legacy head를 썼지만 loss/evaluation은 line head만 사용했고 lower/upper band는 학습/평가하지 않았다.
- CP153 band artifact와 CP164/169 line artifact는 수정하지 않았다.
- seed sweep, walk-forward는 수행하지 않았다.

## Stage 0 Preflight

- source_data_hash: `90666b44cbfb8e5c`
- split_mode: `calendar_aligned`
- cross_split_date_overlap_count: `0`
- feature_version: `v3_adjusted_ohlc`
- feature_set: `price_volatility_volume`
- base feature count: `11`
- `atr_ratio` in indicator parquet: `True`
- `atr_ratio` in MODEL_FEATURE_COLUMNS: `False`
- ATR 실험 방식: 전역 contract 변경 없이 CP175 runner 내부에서만 1개 확장 feature로 표준화해 추가
- train/validation/test rows: `793864` / `175419` / `177095`

## CP164 reference와 현재 beta2 baseline

| 기준 | IC | spread | fee | false-safe | severe recall(line<0) | top actual return |
|---|---:|---:|---:|---:|---:|---:|
| CP164 calendar line_regime reference | 0.0436 | 0.0079 | 0.0069 | 0.2056 | 0.6732 | 0.0099 |
| CP175 beta_2_baseline line-only | 0.0444 | 0.0081 | 0.0071 | 0.2126 | 0.6152 | 0.0106 |

주의: CP164는 line_regime 구조의 reference이고, CP175 trial은 line-only smoke다. 그래서 직접 우열보다 방향성을 보는 기준으로 쓴다.

## Test 결과표

| trial | beta | ATR | IC | spread | fee | false-safe | FS 감소 vs beta2 | severe recall | IC 유지 | spread 유지 | fee 유지 | std 유지 | 판정 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| beta_2_baseline | 2.0 | no | 0.0444 | 0.0081 | 0.0071 | 0.2126 | 0.00pp | 0.6152 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | FAIL |
| beta_3 | 3.0 | no | 0.0447 | 0.0079 | 0.0069 | 0.2120 | 0.06pp | 0.6495 | 1.0064 | 0.9731 | 0.9693 | 1.0005 | FAIL |
| beta_5 | 5.0 | no | 0.0420 | 0.0069 | 0.0059 | 0.1972 | 1.54pp | 0.7921 | 0.9458 | 0.8409 | 0.8187 | 1.0152 | WARN_PARETO |
| beta_7 | 7.0 | no | 0.0403 | 0.0067 | 0.0057 | 0.1929 | 1.97pp | 0.8348 | 0.9079 | 0.8165 | 0.7908 | 1.0277 | WARN_PARETO |
| beta_2_plus_atr_ratio | 2.0 | yes | 0.0427 | 0.0076 | 0.0066 | 0.2059 | 0.67pp | 0.7076 | 0.9610 | 0.9364 | 0.9275 | 1.0004 | FAIL |
| beta_3_plus_atr_ratio | 3.0 | yes | 0.0345 | 0.0061 | 0.0051 | 0.2042 | 0.84pp | 0.7168 | 0.7759 | 0.7505 | 0.7156 | 1.0435 | FAIL |
| beta_5_plus_atr_ratio | 5.0 | yes | 0.0389 | 0.0062 | 0.0052 | 0.1942 | 1.85pp | 0.8343 | 0.8767 | 0.7621 | 0.7288 | 1.0135 | WARN_PARETO |
| beta_7_plus_atr_ratio | 7.0 | yes | 0.0373 | 0.0055 | 0.0045 | 0.1893 | 2.33pp | 0.8660 | 0.8397 | 0.6754 | 0.6300 | 1.0306 | FAIL |

## 질문별 답변

1. beta를 키우는 것이 line을 더 보수적으로 만들었는가?

그렇다. beta_2의 severe recall(line<0)은 0.6152였고 beta_5는 0.7921, beta_7은 0.8348까지 올라갔다. 예측 score std도 baseline 대비 1.02 수준이라 collapse는 아니었다.

2. 그 보수성이 false-safe 감소로 이어졌는가?

이어졌다. beta_5는 false-safe를 1.54pp, beta_7은 1.97pp 줄였다. 다만 PASS 기준 3.0pp에는 못 미친다.

3. 수익성 alpha가 얼마나 깎였는가?

beta_5는 IC retention 0.9458, spread retention 0.8409, fee retention 0.8187이라 비용 대비 가장 균형적이다. beta_7은 false-safe를 더 줄였지만 fee retention 0.7908로 0.80 아래다.

4. ATR 재포함은 효과가 있었는가?

단독 효과는 약했다. beta_2_plus_atr_ratio는 false-safe를 0.67pp 줄이고 severe recall을 0.7076으로 올렸지만, 목표치에는 부족했다.

5. beta와 ATR의 결합이 단독보다 나았는가?

false-safe만 보면 beta_7_plus_atr_ratio가 2.33pp 감소로 가장 좋다. 하지만 spread/fee retention이 0.6754/0.6300까지 내려가 FAIL이다. 즉 결합은 risk를 더 줄이지만 alpha 비용이 커졌다.

6. line 자체를 고치는 길이 warning overlay보다 가능성 있어 보이는가?

일부 가능성은 있다. CP169 two-tier warning은 no-warning FS 0.2065에서 unwarned FS 0.1990로 약 0.76pp 줄였고, spread/fee retention은 0.8543/0.8348이었다. CP175 beta_5는 beta2 대비 1.54pp 줄였으므로 개선폭은 더 크다. 다만 CP170 recall-first wide rule은 recall을 0.5202까지 올렸지만 warning share 0.5181, spread/fee retention 0.6405/0.5886으로 제품형 비용이 컸다. 결론적으로 line 자체 fix는 연구 가치가 있지만, 아직 warning보다 명확히 우월하다고 말하기는 이르다.

7. 효과가 있다면 다음 CP는 seed stability인가, beta/ATR narrow sweep인가?

seed stability로 바로 가기보다는 **beta narrow sweep**이 먼저다. beta_5와 beta_7 사이에서 trade-off가 생겼고, beta_5는 비용이 적고 beta_7은 risk 개선이 크다. 다음 후보는 beta 5/6/7, ATR 포함 여부, epochs 1~3을 좁게 비교하는 방식이 맞다.

8. 효과가 없다면 외부 데이터/밴드 중심 전략으로 가는 근거가 되는가?

완전 실패는 아니다. line 학습 보수성 강화는 신호가 있다. 다만 3pp 이상의 false-safe 개선과 alpha retention 0.80 이상을 동시에 만족하지 못했으므로, 1D risk primary는 여전히 band 쪽이 더 안정적이고 line은 수익성 보조선으로 두는 판단 근거는 유지된다.

## Pareto 판단

- `beta_5`: 현재 최선의 WARN_PARETO. false-safe 1.54pp 감소, IC/spread/fee 유지율 0.9458/0.8409/0.8187.
- `beta_7`: risk 개선은 더 크지만 fee retention 0.7908로 제품형 기준에는 부담.
- `beta_5_plus_atr_ratio`: beta_5보다 false-safe는 더 줄지만 spread/fee 비용이 커져 beta_5보다 덜 균형적.
- `beta_7_plus_atr_ratio`: 최대 risk 개선 후보지만 alpha 비용 때문에 FAIL.

## 다음 제안

CP176을 한다면 `beta_5`, `beta_6`, `beta_7` 중심의 narrow sweep을 권한다. ATR은 정식 feature contract 후보로 바로 올리기보다 beta-only sweet spot을 먼저 좁힌 뒤, ATR을 보조 비교로 다시 붙이는 순서가 낫다.

## 산출물

- `docs/cp175_lm_1d_conservative_line_learning_revisit_metrics.json`
- `docs/cp175_lm_1d_conservative_line_learning_revisit_summary.csv`
- `docs/cp175_lm_1d_beta_only_sweep.csv`
- `docs/cp175_lm_1d_atr_only_result.csv`
- `docs/cp175_lm_1d_beta_atr_combo.csv`
- `docs/cp175_lm_1d_pareto_frontier.csv`
- `docs/cp175_lm_1d_line_collapse_diagnostic.csv`
- `ai/cp175_lm_1d_conservative_line_learning_revisit.py`
