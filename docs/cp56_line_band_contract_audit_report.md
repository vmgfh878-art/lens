# CP56-M Line/Band 분리 계약 전수 감사 보고서

CP56은 성능 개선 실험이 아니라, PatchTST 예측선과 CNN-LSTM/TiDE 밴드 모델을 조합하는 데이터·타겟·모델·평가·추론·저장 계약 감사다.

## 1. Executive Summary

| 항목 | 판정 | 요약 |
|---|---|---|
| 데이터 계약 | 대부분 통과, P1 보강 필요 | 모든 주 후보가 `feature_version=v3_adjusted_ohlc`, `raw_future_return`, 1D h5를 사용한다. 다만 line/band 후보의 ticker registry가 다르면 composite가 교집합만 조용히 사용한다. |
| 타겟 계약 | 통과 | `ai.inference`와 `ai.composite_inference`는 raw-return checkpoint만 가격 decode와 composite 저장을 허용한다. |
| 모델 출력 계약 | 부분 통과 | PatchTST/CNN-LSTM/TiDE 모두 `ForecastOutput(line, lower, upper)` 계약은 맞다. 그러나 CNN-LSTM band calibration은 CNN-LSTM 자체 line을 band center로 사용해 PatchTST line과 위치 불일치가 생긴다. |
| 평가 계약 | 통과 | `line_gate`, `band_gate`, `combined_gate`는 분리되어 있고 `coverage_gate`는 deprecated alias로 남아 있다. |
| inference/composite 저장 계약 | 부분 통과, P1 존재 | `run_id`별 prediction 보존과 meta 저장은 맞다. 그러나 `series_length_all_5` 하드코딩, `risk_first_lower_preserve`와 `include_line_clamp` 동일 구현, ticker/date 교집합 silent 처리 문제가 있다. |
| 다음 실험 전 결론 | 수정 필요 | 새 학습 전에 P1 5건을 먼저 닫아야 CP57 이후 band/composite 실험 해석이 안전하다. |

핵심 결론은 단순하다. 현재 병목은 세 모델 자체만의 문제가 아니라, line 모델과 band 모델이 서로 다른 registry·seq_len·중심선을 가진 상태에서 composite가 이를 너무 조용히 합치는 계약 문제다. PatchTST는 line 후보로 유지 가능하고, CNN-LSTM/TiDE는 band 후보로 계속 볼 수 있다. 다만 composite 정책과 공통 sample 계약을 명시하지 않으면 실험 결과가 제품 정책 효과인지 모델 성능인지 섞인다.

## 2. 데이터 계약 표

| 점검 항목 | 현재 상태 | 근거 | 리스크 | 판정 |
|---|---|---|---|---|
| feature version | `v3_adjusted_ohlc` 사용 | `ai/preprocessing.py`의 `FEATURE_CONTRACT_VERSION`, 주요 checkpoint config | 낮음 | 통과 |
| 모델 입력 feature 수 | `MODEL_N_FEATURES=36` | `SOURCE_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS` | 낮음 | 통과 |
| adjusted OHLC 적용 | price-derived feature repair 경로 존재 | `fetch_training_frames`, `_repair_price_feature_contract` | 낮음 | 통과 |
| raw future return 기준 | dataset이 `raw_future_returns`를 별도 반환 | `SequenceDataset.__getitem__` | 낮음 | 통과 |
| ticker registry | checkpoint별 registry 저장 | PatchTST 48 ticker, CNN-LSTM 188 ticker, TiDE 48 ticker | 교집합만 사용되면 coverage가 조용히 축소됨 | P1 |
| train/val/test split | h_max gap 기반 split | `ai/splits.py`에서 1D h_max=20 | 낮음. horizon별 leakage 방어는 보수적 | 통과 |
| seq_len 차이 | line 252, band 60/252 허용 | composite는 seq_len 일치를 요구하지 않음 | sample anchor가 같아도 모델이 본 과거 구간은 다름 | P2 |
| horizon 차이 | composite에서 horizon 일치 요구 | `_assert_compatible` | 낮음 | 통과 |

주요 후보 config 확인:

| 후보 | model | role 의도 | seq_len | ticker 수 | registry | feature_version |
|---|---|---|---:|---:|---|---|
| `h5_longer_context_seq252_p32_s16` | PatchTST | line | 252 | 48 | `ai\cache\ticker_id_map_1d_c3a8729f2b24.json` | `v3_adjusted_ohlc` |
| `s60_q15_b2_direct_188` | CNN-LSTM | band | 60 | 188 | `ai\cache\ticker_id_map_1d_023a14244cfc.json` | `v3_adjusted_ohlc` |
| `tide_param_scalar_width` | TiDE | band | 252 | 48 | `ai\cache\ticker_id_map_1d_c3a8729f2b24.json` | `v3_adjusted_ohlc` |
| `tide_direct_original` | TiDE | band | 252 | 48 | `ai\cache\ticker_id_map_1d_c3a8729f2b24.json` | `v3_adjusted_ohlc` |

데이터 계약의 가장 큰 병목은 CNN-LSTM 188 ticker band 후보와 PatchTST 48 ticker line 후보를 합칠 때 발생한다. 현재 composite는 공통 ticker/asof_date/forecast_dates만 선택하고, 얼마나 버려졌는지 제품 판단용 지표로 크게 드러내지 않는다.

## 3. target 계약 표

| 점검 항목 | 현재 상태 | 근거 | 리스크 | 판정 |
|---|---|---|---|---|
| line target type | 주요 후보 모두 `raw_future_return` | checkpoint config | 낮음 | 통과 |
| band target type | 주요 후보 모두 `raw_future_return` | checkpoint config | 낮음 | 통과 |
| raw future returns 별도 전달 | 평가 함수에 별도 전달 | `summarize_forecast_metrics(... raw_future_returns=...)` | 낮음 | 통과 |
| 비 raw target 저장 차단 | inference save 차단 | `ai/inference.py`의 `save and not raw_return_mode` guard | 낮음 | 통과 |
| composite raw target guard | raw-return만 허용 | `ai/composite_inference.py::_assert_compatible` | 낮음 | 통과 |
| q_low/q_high 전달 | train/eval/checkpoint에는 저장 | checkpoint config | line q와 band q가 다를 경우 composite 기준 q가 불명확해질 수 있음 | P2 |
| direction/rank target 혼입 위험 | 저장 경로에서 차단 | `market_excess_return`, `rank_target`은 미구현 또는 score-only | 낮음 | 통과 |

타겟 계약은 전반적으로 안전하다. 특히 composite 저장은 `line_target_type=raw_future_return`과 `band_target_type=raw_future_return`만 허용하므로, volatility-normalized target이나 direction label checkpoint가 저장/가격 decode 경로에 들어갈 위험은 현재 낮다.

다만 q 계약은 더 명확히 해야 한다. line 모델의 q는 line 평가에는 본질적이지 않지만, composite band 평가에서는 band 모델의 q가 nominal coverage 기준이다. 이 차이를 meta와 보고서에 명시해야 한다.

## 4. 모델별 출력 계약 표

| 모델 | 출력 계약 | 현재 상태 | 리스크 | 판정 |
|---|---|---|---|---|
| PatchTST | `ForecastOutput(line, lower, upper)` | RevIN denorm이 line/lower/upper에 모두 적용됨 | line 전용 run인데 band output도 같이 저장될 수 있음 | P1 |
| PatchTST | patch geometry 복원 | `patch_len`, `patch_stride`, d_model/head/layer config 저장/복원 | 낮음 | 통과 |
| PatchTST | channel independence | `ci_aggregate=target/mean/attention`, `ci_target_fast` 지원 | 낮음 | 통과 |
| CNN-LSTM | `ForecastOutput(line, lower, upper, direction_logit optional)` | fp32_modules와 cuDNN off 경로 존재 | 낮음 | 통과 |
| CNN-LSTM | band center | scalar width calibration이 CNN-LSTM 자체 line을 center로 사용 | PatchTST line과 band center 위치가 다르면 composite upper breach가 커짐 | P1 |
| CNN-LSTM | CUDA 안정화 | `fp32_modules=lstm,heads`와 cuDNN off | 낮음 | 통과 |
| TiDE | per-horizon output | pooled collapse 제거, per-step head 사용 | 낮음 | 통과 |
| TiDE | future covariate | `use_future_covariate=True`, `future_cov_dim=7` config 복원 | future cov config가 composite 호환성 검증 항목은 아님 | P2 |
| 공통 | band mode | `direct`, `param` 모두 `_split_band` 사용 | 낮음 | 통과 |

모델 출력 계약에서 가장 중요한 병목은 “band 모델의 중심선”이다. 현재 CNN-LSTM band는 line/lower/upper를 동시에 출력하고, scalar width calibration은 CNN-LSTM line을 중심으로 폭을 조정한다. 이후 composite에서 PatchTST line을 끼워 넣으면, band center와 display line이 달라진다. 이 구조에서는 band 단독 평가는 통과하지만 composite upper breach가 나빠지는 현상이 자연스럽다.

PatchTST도 line 전용 후보라면 band 출력이 제품 band로 오해되지 않게 해야 한다. `role=line_model`인 run의 standalone inference 저장에서는 lower/upper를 저장하지 않거나 meta에 “line-only band ignored”를 남기는 정책이 필요하다.

## 5. evaluation 계약 표

| 점검 항목 | 현재 상태 | 근거 | 리스크 | 판정 |
|---|---|---|---|---|
| line gate | IC/spread/MAE/SMAPE 기준 | `line_gate_eligible`, `line_gate_sort_key` | 낮음 | 통과 |
| band gate | coverage/breach/width/band_loss 기준 | `band_gate_eligible`, `band_gate_sort_key` | 낮음 | 통과 |
| combined gate | line+band 동시 조건 | `combined_gate_eligible` | 낮음 | 통과 |
| coverage_gate | `combined_gate` alias | `normalize_checkpoint_selection_mode` | deprecated 혼동 가능 | P2 |
| CP52 metric set | line/band/composite 분리 | `ai/evaluation.py` | 낮음 | 통과 |
| line_inside_band | 제품 표시 보조 지표 | metric으로 존재하나 selector에는 섞이지 않음 | 낮음 | 통과 |
| composite metric | 제품 정책 지표 | composite 결과에 별도 기록 | band 모델 성능으로 해석하면 안 됨 | P2 |
| interval score | asymmetric lower penalty 2.0 | CP52 계약 반영 | 낮음 | 통과 |

평가 계약은 CP36 이후 의도대로 분리되어 있다. band 후보는 IC가 음수여도 band 조건을 통과할 수 있고, line 후보는 coverage가 과보수여도 line 조건을 통과할 수 있다.

주의할 점은 `coverage_gate`가 여전히 CLI choice에 남아 있다는 것이다. backward compatibility 때문에 유지하는 것은 맞지만, 새 실험 문서에서는 `combined_gate`만 쓰도록 정리해야 한다.

## 6. inference/composite 계약 표

| 점검 항목 | 현재 상태 | 근거 | 리스크 | 판정 |
|---|---|---|---|---|
| completed run만 inference | `model_runs.status == completed` 요구 | `ai/inference.py`, `ai/composite_inference.py` | 낮음 | 통과 |
| run_id별 prediction 보존 | conflict key에 run_id 포함 | `ai/storage.py` | 낮음 | 통과 |
| prediction meta | line/band run id, policy, calibration params 저장 | `ai/composite_inference.py` | 낮음 | 통과 |
| forecast_dates 정렬 | common key에 forecast_dates 포함 | `_select_latest_common_indices` | 낮음 | 통과 |
| 공통 ticker/date 선택 | 교집합만 사용 | `_select_latest_common_indices` | silent subset 발생 | P1 |
| composition policy | 3개 정책 지원 | `raw_composite`, `include_line_clamp`, `risk_first_lower_preserve` | 두 정책 구현이 동일 | P1 |
| series length check | `series_length_all_5` 하드코딩 | h5 외 horizon에서 오탐 | P1 |
| scalar calibration | lower/upper width scale 저장 | `_apply_scalar_width` | band model line 중심에 종속 | P1 |
| DB meta column 확인 문구 | 결과에 최소 변경 경로 설명 | `contract_options` | stale 문구 가능 | P2 |

Composite 저장 계약은 큰 틀에서는 닫혀 있다. `predictions.meta`에는 `composition_policy`, `line_model_run_id`, `band_model_run_id`, `line_model_name`, `band_model_name`, `band_calibration_method`, `band_calibration_params`, `prediction_composition_version`이 저장된다.

그러나 composite 정책 경로에는 즉시 손봐야 할 부분이 있다. 현재 `risk_first_lower_preserve`는 `include_line_clamp`와 동일하게 `lower=min(lower,line)`, `upper=max(upper,line)`을 반환한다. 이름은 다르지만 결과가 같아서 CP43 이후 정책 비교의 의미가 약해진다.

또한 `series_length_all_5`는 h20 composite smoke에서 실제 series 길이가 20이어도 contract check 이름과 조건이 h5에 고정되는 문제를 만든다. 다음 horizon 실험 전에 반드시 `series_length_all_horizon`으로 바꿔야 한다.

## 7. line/band 분리 병목 후보

| 질문 | 답 | 병목 후보 |
|---|---|---|
| PatchTST line과 CNN-LSTM band가 같은 sample을 보고 있는가? | 부분적으로만 그렇다. composite는 ticker/asof_date/forecast_dates 교집합만 사용한다. | registry mismatch와 silent subset |
| line과 band의 스케일이 맞는가? | target은 raw return으로 같지만, band center는 CNN-LSTM line이고 제품 line은 PatchTST line이다. | center mismatch |
| line이 band 밖으로 나가는 원인은 무엇인가? | line 자체 과대/과소, band 위치, band 폭이 모두 가능하다. 현 구조상 band 위치 문제가 특히 크다. | center mismatch + 후처리 clamp |
| composite 후처리가 dynamic width 지표를 망가뜨리는가? | 가능성이 높다. line 포함을 위해 upper/lower가 늘어나면 width IC가 모델 원 신호가 아니라 정책 산물이 된다. | composite metric과 band metric 혼합 |
| band 단독은 괜찮은데 composite에서 나빠지는 이유는? | band 단독은 CNN-LSTM center 기준이고, composite는 PatchTST line 기준으로 표시된다. | line/band center 분리 부재 |
| TiDE를 band 2안으로 살릴 계약상 장애가 있는가? | 큰 장애는 없다. TiDE future covariate 복원은 되며, registry도 PatchTST line과 같은 48 ticker 후보가 있다. | calibration과 baseline 대비 성능 문제 |
| PatchTST를 band 후보로 다시 평가할 때 막힌 부분은? | 코드상 막힌 부분은 없다. 다만 line_gate checkpoint의 band 출력은 line용 훈련 산물이라 band 후보로 쓰려면 band_gate 재실험이 필요하다. | 역할 혼동 |

병목의 핵심은 모델별 feature/target이 아니라 composite의 정렬과 중심선 계약이다. 지금 구조에서는 “CNN-LSTM이 예측한 위험 폭”과 “PatchTST가 표시하는 중심선”이 분리되어 있지만, 이를 명시적으로 모델링하지 않고 clamp로 합친다.

## 8. 즉시 고쳐야 할 P1

| 우선순위 | 항목 | 이유 | 권장 수정 |
|---|---|---|---|
| P1-1 | `risk_first_lower_preserve`와 `include_line_clamp` 동일 구현 분리 | 정책 비교와 보고서 해석이 무의미해질 수 있음 | risk-first는 하방 보수성 유지와 상단 보정 규칙을 별도 함수로 분리하고, 정책별 width 증가율을 meta에 저장 |
| P1-2 | `series_length_all_5` 하드코딩 제거 | h10/h20 contract check 오탐 | checkpoint horizon을 읽어 `series_length_all_horizon`으로 검증 |
| P1-3 | composite common sample coverage 기록 | ticker registry mismatch가 조용히 숨겨짐 | line ticker 수, band ticker 수, common ticker 수, dropped ticker 목록/비율을 result와 meta에 저장 |
| P1-4 | scalar width center 계약 명시 | CNN-LSTM band center와 PatchTST line이 달라 composite upper breach 발생 | band width만 추출하는 `width_only` 계약 또는 line-centered calibration 평가 경로 추가 |
| P1-5 | role별 inference 저장 정책 | line run의 band output이 제품 band로 오해될 수 있음 | `role=line_model`이면 standalone prediction lower/upper 저장 차단 또는 `band_output_ignored=true` meta 추가 |

P1은 모두 성능 개선이 아니라 계약 안정화다. 이 5개를 닫아야 다음 band sweep 또는 composite 200 ticker 저장 smoke가 의미 있다.

## 9. 다음 실험 전에 확인할 P2

| 항목 | 리스크 | 권장 처리 |
|---|---|---|
| `coverage_gate` alias | 새 실험에서 combined gate가 의도치 않게 섞일 수 있음 | 문서와 CLI help에서 deprecated 명시 |
| `lambda_width` legacy config | 제거된 width penalty가 살아 있는 것처럼 보임 | report/config에 `legacy_unused` 표기 |
| q 계약 | line q와 band q가 다를 때 nominal coverage 기준 혼동 | composite 평가는 band checkpoint q 기준이라고 meta에 명시 |
| TiDE future covariate | future covariate config는 복원되지만 composite compatibility 항목은 아님 | 모델별 내부 계약으로 유지하되 report에 사용 여부 기록 |
| CP52 metric completeness | 기존 후보 일부는 full metric set이 부족 | 재실험 필요와 탈락을 구분 |
| DB meta 확인 문구 | CP40/41 이후 stale 문구가 남아 있을 수 있음 | schema 변경 없이 문구만 최신화 |
| h20 branch | h20은 forecast length 20이지만 일부 check가 h5 중심 | P1-2 후 별도 branch로 재검증 |

P2는 다음 실험을 막지는 않지만, 결과 해석을 흐릴 수 있다. 특히 `coverage_gate`와 `lambda_width`는 과거 로그를 읽을 때 착시를 만든다.

## 10. 3모델 강화 관점의 다음 추천

| 모델 | 현재 역할 | 추천 |
|---|---|---|
| PatchTST | h5 line 주력 후보 | line 전용으로 유지한다. band 출력은 제품 밴드로 쓰지 말고, `line_gate`와 보수적 line 지표를 중심으로 100/200 ticker 재확인한다. |
| CNN-LSTM | band 후보 1 | 모델 구조를 바꾸기 전에 center 계약을 분리한다. band model은 lower/upper 자체보다 downside/upside width를 예측하는 후보로 재해석하는 편이 맞다. |
| TiDE | band 후보 2 | future covariate가 있는 band 후보로 유지한다. registry가 PatchTST line과 같아 composite sample mismatch가 적은 장점이 있다. 다만 baseline 대비 interval/dynamic width는 재검증 필요하다. |

다음 CP 추천 순서:

1. CP57: composite 계약 P1 수정. 학습 금지, 저장 smoke만 허용.
2. CP58: CNN-LSTM/TiDE band 후보를 `width_only` 또는 line-centered calibration 기준으로 재채점.
3. CP59: PatchTST line 100/200 ticker 재확인. band와 분리된 line-only report.
4. CP60: 통계 baseline 위 residual/scale band 설계 검토. 단, 세 모델 강화 본류를 유지하되 baseline-aware band를 비교군으로 둔다.

이번 감사 기준으로 보면, Phase 1 본류를 PatchTST/TiDE/CNN-LSTM 세 모델 강화로 유지하는 것은 가능하다. 다만 “line 모델”과 “band 모델”을 하나의 모델처럼 평가하는 경로는 더 이상 쓰면 안 된다. line은 line 지표로, band는 nominal calibration과 interval/dynamic width 지표로, composite는 제품 표시 정책으로 따로 닫아야 한다.
