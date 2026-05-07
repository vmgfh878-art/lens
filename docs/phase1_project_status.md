# Lens Phase 1 프로젝트 상태

작성일: 2026-05-07

이 문서는 Lens Phase 1의 현재 기준, 닫힌 것, 아직 열려 있는 것을 한곳에서 보기 위한 프로젝트 기록이다. 개별 모델 실험 설계서는 이 문서에 섞지 않는다.

## 현재 기준

Phase 1의 제품 모델 최소 기준은 500티커 데이터셋으로 올린다. 100티커 모델은 초기 검증과 데모 후보로는 유지할 수 있지만, 최종 제품 후보라고 부르지 않는다.

제품 모델 슬롯은 네 개로 고정한다.

| 슬롯 | 목표 | 현재 상태 |
|---|---|---|
| 1D line | 1D 보수적 예측선 v1 | 500티커 기준 재실험 필요 |
| 1D band | 1D AI 밴드 v1 | 500티커 기준 재실험 필요 |
| 1W line | 1W 보수적 예측선 v1 | 500티커 기준 가능성 있음, 추가 검증 필요 |
| 1W band | 1W AI 밴드 v1 | 가장 약한 슬롯, 모델군 확장 필요 |

1M은 Phase 1 모델 실험에서 제외한다. 데이터 길이와 split gate가 충분하지 않아 가격 전용 화면으로 유지한다.

## 데이터 기준

현재 학습용 기준 데이터는 EODHD 500 local parquet다.

| 항목 | 상태 |
|---|---|
| EODHD 500 price | 503티커, 2015-01-02부터 2026-05-05까지 생성 완료 |
| EODHD 500 1D indicators | 503티커, 2026-05-05까지 생성 완료 |
| EODHD 500 1W indicators | 502티커, 2026-05-01 완료 주까지 생성 완료 |
| yfinance | 무료 운영 전환용 parallel dataset. 현재 429와 최신 coverage 이슈로 운영 primary 확정 전 |
| Supabase | 원천 저장소가 아니라 제품 표시용 thin DB |
| local parquet | 학습, 실험, 백테스트, scanner의 원천 저장소 |

EODHD와 yfinance는 같은 시계열에 이어 붙이지 않는다. provider/source/hash/cache/manifest를 분리한다.

## 데이터 가드

다음 가드는 모델 full run 전 기준으로 고정한다.

- 1D absolute minimum history: 450 rows
- 1W absolute minimum history: 78 rows
- 1M full run: Phase 1 제외
- provider/source는 실험 CP마다 명시한다.
- generic latest cache를 믿지 않고 manifest provider/source/hash를 기록한다.
- feature contract는 `v3_adjusted_ohlc`를 유지한다.
- `atr_ratio`는 indicator에는 존재하지만 현재 모델 feature에는 포함하지 않는다.

CP148-0과 CP148-0-DG2에서 EODHD 500 가격, indicator, feature finite, target finite, split, tiny forward shape는 통과했다. 남은 WARN은 fundamentals coverage가 낮다는 점과 legacy cache 후보가 존재한다는 점이다. 이는 실험 차단이 아니라 해석 주의다.

## 모델 기준

제품 후보 run은 다음 조건을 만족해야 한다.

| 모델 | 필수 조건 |
|---|---|
| line model | `checkpoint_selection=line_gate`, `line_gate_pass=true` |
| band model | `checkpoint_selection=band_gate`, `band_gate_pass=true` |
| 공통 | provider/source/hash/feature_set/checkpoint_path가 명확해야 함 |

`val_total` 기본값으로 completed가 된 run은 제품 후보로 자동 승격하지 않는다. calibration 결과가 실제 inference 저장 경로에 적용되지 않았다면 calibrated 성능을 제품 성능으로 말하지 않는다.

## 현재 제품 연결 상태

현재 프론트에 연결된 1D/1W 모델은 Phase 1 최종 500티커 제품 후보가 아니라 초기 제품 데모 상태로 본다.

| 화면 슬롯 | 현재 연결 | 해석 |
|---|---|---|
| 1D 보수적 예측선 | 기존 100티커/초기 run | 데모/초기 후보 |
| 1D AI 밴드 | 기존 100티커/초기 run | 데모/초기 후보 |
| 1W 보수적 예측선 | 저장된 1W line run | 교체 검토 후보 |
| 1W AI 밴드 | 없음 | 준비 중 |
| 1M | 없음 | 가격 전용 |

제품 history는 `/predictions/history`가 아니라 local parquet 기반 product history API를 기준으로 한다. latest-only row는 미래 forecast 전용이고, test split bulk row는 제품 차트 history로 쓰지 않는다.

## Supabase 기준

Supabase는 전체 가격/지표 저장소가 아니다. 비용과 egress를 줄이기 위해 다음 원칙을 유지한다.

- 전체 price/indicator/prediction history는 local parquet에 둔다.
- Supabase에는 제품 화면에 필요한 최신 결과와 선택 결과만 얇게 저장한다.
- 500티커 scanner는 전체 raw prediction을 Supabase에 올리지 않고, top-k 결과와 선택 ticker latest만 저장한다.
- row pruning은 당장 실행하지 않고 보류한다. 로그인/구독형/상위 조회 기능을 설계한 뒤 보존 정책을 확정한다.
- 개발 편의를 위해 Supabase Pro를 한 달 쓰는 선택은 가능하지만, 구조는 Free/slim 운영이 가능하도록 유지한다.

## 다음 집중 작업

프론트, Supabase pruning, 백테스트 UX는 잠시 뒤로 미룬다. 현재 우선순위는 네 모델 슬롯의 500티커 v1을 만드는 것이다.

1. CP148-LM-1D: 500티커 1D 보수적 예측선 v1
2. CP149-BM-1D: 500티커 1D AI 밴드 v1
3. CP150-LM-1W: 500티커 1W 보수적 예측선 v1
4. CP151-BM-1W: 500티커 1W AI 밴드 v1

각 CP는 성능이 나올 때까지 반복한다. 실패한 모델도 기록하지만, 제품 화면에는 기준을 넘은 모델만 노출한다.

## 실험 문서 분리 원칙

프로젝트 상태 문서는 전체 방향과 결정만 기록한다. 개별 실험의 연구 질문, 실험 matrix, metric dictionary, failure taxonomy는 해당 실험 계획서에만 둔다.

사용자가 읽어야 할 1D line 실험 기준 문서는 단일 계획서로 통합한다. 여러 부록 문서는 원자료로만 둔다.
