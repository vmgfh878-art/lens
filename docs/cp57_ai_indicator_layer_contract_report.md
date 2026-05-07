# CP57-M AI Indicator Layer 계약 분리 보고서

CP57은 “line과 band를 더 잘 합치는 법”이 아니라, 억지로 합치지 않는 계약을 세우는 작업이다.

## 1. 왜 composite 모델 계약을 Phase 1 본류에서 내리는가

CP56 감사 이후 처음에는 composite 계약을 더 안전하게 수리하는 방향을 검토했다. 하지만 실제 병목은 composite를 더 정교하게 고치는 문제가 아니라, 서로 다른 역할의 산출물을 하나의 `line/lower/upper` prediction으로 강제 결합하는 해석 자체였다.

Phase 1에서는 다음 결정을 고정한다.

| 결정 | 내용 |
|---|---|
| composite model | deprecated |
| composite prediction | Phase 1 본류에서 deprecated |
| line_inside_band | 탈락 기준도 보정 목표도 아님 |
| clamp / upper buffer | Phase 1 본류 실험에서 제외 |
| 제품 표시 | line layer와 band layer를 화면에서 overlay |

따라서 `include_line_clamp`, `risk_first_lower_preserve`, `risk_first_upper_buffer_1.10` 같은 정책은 모델 성능 개선 수단이 아니다. 과거 CP 기록 보존을 위한 legacy/experimental utility로만 남긴다.

## 2. AI line layer와 AI band layer의 차이

| layer | 목적 | 대표 후보 | 제품 사용 출력 | 평가 지표 |
|---|---|---|---|---|
| AI line layer | 하방 보수적 예측선 또는 순위/방향 score | PatchTST h5/h10/h20, TiDE line 후보 | `line_series` 또는 score | IC, spread, false-safe, severe downside recall |
| AI band layer | calibrated risk interval | CNN-LSTM band, TiDE band, PatchTST band 재실험 | `lower_band_series`, `upper_band_series` | nominal calibration, interval score, dynamic width, breach |
| overlay bundle | 화면에서 여러 layer를 함께 표시 | latest completed line + latest completed band | layer 묶음과 provenance | horizon/asof/date mismatch 표시 |

AI line layer와 AI band layer는 같은 ticker와 horizon을 가질 수 있지만, 하나의 모델 산출물이 아니다. line layer가 band 안에 있어야 한다는 조건도 없다. 두 layer가 다르면 화면에서 “서로 다른 AI 지표”임을 provenance로 보여주는 것이 맞다.

## 3. role별 제품 사용/무시 output

| role | 제품에 쓰는 output | 제품에서 무시하는 output | 비고 |
|---|---|---|---|
| `line_model` | `line_series` | model이 부수적으로 낸 `lower_band`, `upper_band` | 부수 band는 evaluation 내부 참고용 |
| `band_model` | `lower_band_series`, `upper_band_series` | model이 부수적으로 낸 `line` | 부수 line은 band center 또는 내부 참고용 |
| `composite_model` | Phase 1 본류 사용 안 함 | 전체 composite row | legacy demo artifact |

즉 PatchTST line run의 lower/upper를 제품 band로 쓰지 않는다. 반대로 CNN-LSTM/TiDE band run의 line을 제품 예측선으로 쓰지 않는다.

## 4. 기존 composite run의 지위

기존 composite saved run은 삭제하지 않는다. 다만 지위는 다음처럼 고정한다.

| 항목 | 상태 |
|---|---|
| 기존 composite run | legacy demo artifact |
| composite CLI | legacy experimental utility |
| composite policy 결과 | 모델 성능 지표가 아니라 과거 조합 실험 기록 |
| 신규 Phase 1 실험 | composite run 기준으로 성능 판단 금지 |

`ai/composite_inference.py`와 `ai/composite_policy_eval.py`에는 legacy 안내 문구를 추가했다. DB schema 변경은 하지 않았다.

## 5. 저장/조회 계약

DB schema를 바꾸지 않는 범위에서 Phase 1 저장/조회 계약을 아래처럼 분리한다.

| 계약 | 내용 |
|---|---|
| line prediction row | line model run_id 기준으로 독립 저장 |
| band prediction row | band model run_id 기준으로 독립 저장 |
| composite row | Phase 1 본류 신규 저장 금지 |
| 조회 | API/프론트 레이어에서 최신 completed line run과 최신 completed band run을 각각 조회 |
| overlay | 같은 ticker/asof/horizon이면 함께 표시, 다르면 mismatch 표시 |

`line_model_run_id`와 `band_model_run_id`를 하나의 prediction row에 합성하는 구조는 Phase 1 본류에서 중단한다.

## 6. 공통 sample coverage는 계속 기록

line과 band를 합치지 않더라도 화면 overlay에는 정렬 정보가 필요하다.

| 항목 | 용도 |
|---|---|
| line layer ticker/asof_date | line provenance |
| band layer ticker/asof_date | band provenance |
| horizon 일치 여부 | overlay 가능 여부 |
| forecast_dates 일치 여부 | 같은 미래 구간인지 확인 |
| common ticker/date 여부 | 같은 시점 지표인지 표시 |
| mismatch | 모델 탈락 사유가 아니라 화면 표시 경고 |

이 정보는 모델 성능 평가 지표가 아니다. overlay/product contract 지표다.

## 7. 평가 지표 재분류

| 지표 | 새 분류 | 처리 |
|---|---|---|
| `line_inside_band_ratio` | overlay 참고 | 모델 탈락 기준에서 제외 |
| `line_inside_band_point_ratio` | overlay 참고 | 모델 탈락 기준에서 제외 |
| `product_display_warning_rate` | 기술 오류 표시 | lower > upper, length mismatch 같은 표시 오류에만 사용 |
| `conservative_series_false_safe_rate` | layer별로 분리 필요 | line 기준인지 band 기준인지 명시 전까지 composite 판단 금지 |
| `include_line_clamp` | legacy policy | Phase 1 본류 제외 |
| `risk_first_lower_preserve` | legacy policy | Phase 1 본류 제외 |
| `risk_first_upper_buffer_1.10` | legacy policy | Phase 1 본류 제외 |

CP52 line/band metric set은 유지한다. 바뀐 것은 composite metric의 지위다.

## 8. 앞으로 모델 실험을 나누는 기준

### Line experiments

| 후보 | 평가 |
|---|---|
| PatchTST h5/h10/h20 | IC, spread, false-safe, severe recall |
| TiDE line 2안 | PatchTST 대비 line 보조 후보 |
| CNN-LSTM line | 현재 주력 아님, 필요 시 참고 |

### Band experiments

| 후보 | 평가 |
|---|---|
| CNN-LSTM band | nominal calibration, interval score, dynamic width, breach |
| TiDE band | future covariate 기반 band 후보 |
| PatchTST band 재실험 | line과 분리된 band_gate 후보로만 평가 |
| 통계 baseline | rolling quantile, Bollinger, constant width와 비교 |

### Overlay/product experiments

| 항목 | 평가 |
|---|---|
| provenance 표시 | model_run_id, model_name, config, horizon, asof_date |
| mismatch 표시 | 서로 다른 모델/시점/forecast_dates 여부 |
| 화면 overlay | 합성 prediction이 아니라 layer 동시 표시 |

## 9. 코드 정리 사항

| 파일 | 변경 |
|---|---|
| `ai/composite_inference.py` | Phase 1 legacy utility 안내와 result/config meta 추가 |
| `ai/composite_policy_eval.py` | CP43 composite policy 비교 도구를 legacy로 표시 |
| backend/frontend | 이번 CP에서 수정 안 함 |
| DB schema | 수정 안 함 |

이전 CP57 방향으로 진행하던 composite 수리 코드는 되돌렸다. 즉 `risk_first_upper_buffer_1.10` 신규 정책 추가, common sample 함수 확장, h20 length check 코드 변경, 신규 composite contract 테스트 파일은 남기지 않았다.

## 10. 다음 추천 CP

| 다음 CP | 목적 |
|---|---|
| CP58 | CNN-LSTM/TiDE/PatchTST band 후보를 band layer 기준으로 재실험 |
| CP59 | PatchTST/TiDE line 후보를 line layer 기준으로 재실험 |
| CP60 | 프론트/API overlay bundle 계약 설계. 단 UI 수정은 별도 CP에서 진행 |

최종 결론은 명확하다. Phase 1에서는 line과 band를 하나의 composite prediction으로 합치지 않는다. line은 line layer, band는 band layer로 독립 저장·평가하고, 제품에서는 provenance가 붙은 indicator overlay로 보여준다.
