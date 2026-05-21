# CP148-LM-1D Stage 4-4G overextension feature preflight 보고서

- 작성 시각: 2026-05-10T21:08:45
- 범위: Stage 4-4 기존 6개 checkpoint, validation/test split, forward-only 진단
- 금지 작업 준수: 새 학습, Optuna, product save-run, DB write, inference 저장, live fetch, band/composite, core feature contract 변경 모두 미실행

## 1. 핵심 결론

다음 학습 실험에 넣을 후보는 `runup_20d_xs_z, runup_20d, ma60_extension_pos`로 제한한다. 이 후보들은 missed severe가 caught severe보다 높게 나타나는 방향성이 상대적으로 일관적이며, Stage 4-4F의 quiet-tail/overextension blind spot을 직접 겨냥한다.

## 2. 후보 피처 판정표

| feature | 판정 | 방향성 | val방향 | test방향 | median diff | max corr | D계열 | 이유 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runup_20d | recommend | 1.000000 | 1.000000 | 1.000000 | 0.052181 | 0.804723 | N | seed/candidate/split 전반에서 missed severe가 더 높고 해석이 명확함 |
| runup_20d_xs_z | recommend | 1.000000 | 1.000000 | 1.000000 | 0.524735 | 0.711103 | N | seed/candidate/split 전반에서 missed severe가 더 높고 해석이 명확함 |
| runup_20d_xs_rank | hold | 1.000000 | 1.000000 | 1.000000 | 0.142346 | 0.706769 | N | runup_20d_xs_z와 같은 cross-sectional runup 계열이므로 둘 다 넣지 않고 xs_z를 우선함 |
| ma60_extension_pos | recommend | 1.000000 | 1.000000 | 1.000000 | 0.050090 | 0.886133 | N | seed/candidate/split 전반에서 missed severe가 더 높고 해석이 명확함 |
| ma20_extension_pos | hold | 1.000000 | 1.000000 | 1.000000 | 0.016240 | 0.854699 | N | ma60_extension_pos보다 단기/기존 ma_20_ratio 중복성이 커서 보조 후보로만 둠 |
| max_down_day_20d | exclude | 0.500000 | 1.000000 | 0.000000 | 0.002534 | 0.533647 | Y | missed severe가 caught severe보다 높다는 방향성이 약하거나 반대임 |
| pullback_from_20d_high | exclude | 0.000000 | 0.000000 | 0.000000 | -0.012402 | 0.721781 | Y | missed severe가 caught severe보다 높다는 방향성이 약하거나 반대임 |

## 3. 추천 피처 2~3개

- `runup_20d_xs_z`
- `runup_20d`
- `ma60_extension_pos`

## 4. 보류/제외 피처

- 보류 `runup_20d_xs_rank`: runup_20d_xs_z와 같은 cross-sectional runup 계열이므로 둘 다 넣지 않고 xs_z를 우선함
- 보류 `ma20_extension_pos`: ma60_extension_pos보다 단기/기존 ma_20_ratio 중복성이 커서 보조 후보로만 둠
- 제외 `max_down_day_20d`: missed severe가 caught severe보다 높다는 방향성이 약하거나 반대임
- 제외 `pullback_from_20d_high`: missed severe가 caught severe보다 높다는 방향성이 약하거나 반대임

## 5. 과적합 위험 평가

- `max_down_day_20d`와 `pullback_from_20d_high`는 D_stock_fragility와 가까운 계열이라, 분리력이 있어도 바로 주력 pack에 넣기보다 보류한다.
- `runup_20d_xs_z`와 `runup_20d_xs_rank`는 같은 원천의 상대 상승 피처라 둘 다 넣으면 중복될 수 있다. 둘 중 하나만 선택한다.
- `ma20_extension_pos`와 `ma60_extension_pos`는 기존 ma ratio의 양수 부분만 분리한 값이라 상관이 높으면 설명력 추가가 제한된다.

## 6. 다음 feature pack 제안

- 기준: C_stress_delta + `runup_20d_xs_z, runup_20d, ma60_extension_pos`
- beta=2.0, line_gate 의미, 모델 구조는 유지한다.
- 3-seed 단일 후보로 먼저 확인하고, product save-run 표현은 쓰지 않는다.

## 7. 학습 실행 판단 근거

이 preflight에서 추천 후보가 2개 이상 나오면 다음 단계 학습을 돌릴 근거가 있다. 추천 후보가 0~1개면 feature blind를 피처로 해결하기 어렵다고 보고 selector 또는 horizon/target 재검토가 낫다.

## 8. 산출물

- metrics: `docs\cp148_lm_1d_stage4_4g_overextension_feature_preflight_metrics.json`
- summary: `docs\cp148_lm_1d_stage4_4g_overextension_feature_preflight_summary.csv`
- script: `ai/cp148_lm_1d_stage4_4g_overextension_feature_preflight.py`
