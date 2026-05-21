# CP148-LM 1D 실패 Taxonomy

이 문서는 1D 보수적 예측선 실험 결과가 실패했을 때 원인을 분류하고 다음 액션을 정하기 위한 기준이다. 성능이 안 나오면 단순히 "나쁨"으로 끝내지 않고, 어느 축이 깨졌는지 분리한다.

| failure_type | 정의 | 진단 신호 | 다음 액션 |
|---|---|---|---|
| ranking_fail | 종목 간 순위 신호가 없음 | IC <= 0 또는 spearman_ic 음수 | model family 변경, feature_set 단순화, seq_len 재검토 |
| spread_fail | 상위/하위 포트폴리오 실제 수익 차이가 없음 | long_short_spread <= 0 | ranking objective/feature 점검, baseline 대비 비교 |
| false_safe_fail | 위험한 하락을 안전하게 보는 오류가 많음 | false_safe_tail_rate > 0.20 | pvv/no_fundamentals 비교, regularization 강화, model family 변경 |
| severe_recall_fail | 큰 하락 포착률이 낮음 | severe_downside_recall < 0.75 | downside sample weighting은 다음 CP 후보, 이번 CP에서는 기록 |
| over_conservative_fail | 지나치게 낮은 line으로 전부 위험 처리 | conservative_bias가 과도하게 음수, direction/return 악화 | beta 조정은 이번 CP 금지, 후보 제외 또는 risk-only |
| upside_sacrifice_fail | 상승 기회를 과도하게 희생 | upside_sacrifice 과다, fee_adjusted_return 약화 | feature/model 단순화, regularization 조정 |
| overfit_val_test_gap | validation에서만 좋고 test에서 무너짐 | val pass, test false_safe 또는 IC 급락 | seed stability 전 제품 후보 제외, split/regime 분석 |
| seed_unstable | seed에 따라 결론이 바뀜 | top 후보 평균 차이보다 seed 표준편차가 큼 | seed 수 확대, 동률권 판정, 제품 저장 보류 |
| feature_noise_fail | feature 확장 후 성능/위험 지표 악화 | full_features/no_fundamentals가 pvv보다 악화 | feature 줄이기, context_light 별도 설계 |
| cache_or_contract_fail | provider/source/cache/feature contract가 불명확 | source_data_hash 누락, yfinance/generic cache 혼입 | 실험 폐기, Stage 0 재실행 |
| explainability_fail | 제품 설명이 어려움 | full_features가 이기지만 fundamentals coverage 설명 불가 | 기본 후보 금지, WARN 비교 후보로만 기록 |
| metric_generation_fail | 학습은 끝났지만 metric 생성 실패 | line_metrics 누락 또는 NaN/Inf | 코드/평가 경로 별도 CP에서 수정, 결과 미사용 |
| process_runtime_fail | 학습 프로세스가 멈추거나 잘못된 Python 사용 | GPU compute 없음, system Python, progress 정지 | 잔여 프로세스 정리, .venv runner 확인, 병목 수정 후 재실행 |

## 분류 규칙

한 후보는 여러 failure_type을 동시에 가질 수 있다. 최종 판정은 가장 제품 위험이 큰 실패를 우선한다.

우선순위:

1. cache_or_contract_fail
2. metric_generation_fail
3. ranking_fail 또는 spread_fail
4. false_safe_fail 또는 severe_recall_fail
5. overfit_val_test_gap
6. seed_unstable
7. feature_noise_fail 또는 explainability_fail
8. upside_sacrifice_fail 또는 over_conservative_fail

## 판정 예시

CP146 1D PatchTST pvv 후보는 IC와 spread가 양수라 ranking_fail은 아니다. 그러나 validation false_safe_tail_rate 0.3118, severe_downside_recall 0.6842로 false_safe_fail과 severe_recall_fail에 해당한다. 따라서 제품 v1 후보가 아니라 rejected이며, CP148에서는 model family와 feature set을 넓혀 같은 실패가 반복되는지 확인해야 한다.

CP146 1D PatchTST dense 후보는 IC/spread가 더 강했지만 false_safe_tail_rate가 0.3524로 더 나빠졌다. 이는 ranking signal 강화가 보수적 위험선 개선으로 이어지지 않는 대표 사례이므로, CP148 objective에서 false_safe penalty를 별도 기록해야 한다.

## 실패별 연구 방향

- ranking_fail: pvv가 너무 제한적이면 no_fundamentals로 확장하고, model family를 PatchTST에서 TiDE/TCN으로 바꾼다.
- spread_fail: IC가 양수인데 spread가 약하면 bucket 구성, fee proxy, turnover를 확인한다.
- false_safe_fail: feature 확장보다 model family와 regularization을 먼저 본다. beta 변경은 CP148 범위 밖이다.
- severe_recall_fail: severe downside sample weighting은 다음 CP 후보로만 제안한다.
- over_conservative_fail: line이 전반적으로 낮아져 risk metric만 좋아지는 후보는 제품 기본 후보가 아니라 risk_conservative_variant로 분류한다.
- overfit_val_test_gap: validation 선택 기준은 유지하되, 제품 저장 전 seed stability와 regime split을 추가한다.
- feature_noise_fail: full_features가 실패하면 fundamentals coverage WARN을 원인 후보로 기록하고 pvv/no_fundamentals를 우선한다.
- explainability_fail: 성능이 좋아도 사용자에게 설명하기 어려운 feature 조합은 제품 기본 후보로 올리지 않는다.
