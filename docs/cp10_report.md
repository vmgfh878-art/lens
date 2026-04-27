# CP10 보고서

## 목적
이번 CP의 목적은 모델 구조를 바꾸는 것이 아니라, PatchTST, CNN-LSTM, TiDE가 같은 투자 판단 기준으로 비교될 수 있도록 공통 평가판과 target 배선 구조를 만드는 것이다.

## 공통 평가 지표
다음 지표를 공통 요약 함수로 정리했다.

- `direction_accuracy`: 마지막 horizon 방향 일치율
- `spearman_ic`: 날짜별 점수-실현수익률 스피어만 상관
- `top_k_long_spread`: 상위 종목 평균 실현수익률
- `top_k_short_spread`: 하위 종목 평균 실현수익률
- `long_short_spread`: 상위 평균 - 하위 평균
- `coverage`: 예측 밴드 안에 실현값이 들어온 비율
- `avg_band_width`: 예측 밴드 평균 폭
- `mae`: 절대오차 평균
- `smape`: 대칭 평균 절대 퍼센트 오차
- `fee_adjusted_return`: turnover와 수수료를 반영한 누적 수익률
- `fee_adjusted_sharpe`: 수수료 반영 수익률 기준 샤프
- `fee_adjusted_turnover`: 리밸런싱 turnover 평균

## target plumbing
`ai/targets.py`를 추가해 target 타입을 코드 구조상 분리했다.

- `raw_future_return`
- `volatility_normalized_return`
- `direction_label`
- `market_excess_return` 준비만 완료, 계산 경로는 추후 연결
- `rank_target` 준비만 완료, 단면 랭킹 경로는 추후 연결

현재 `prepare_dataset_splits()`와 lazy dataset 경로에서 `line_target_type`, `band_target_type`을 각각 전달받아 학습/추론 양쪽에 동일하게 반영한다.

## 반영 위치
- `ai/evaluation.py`
  - 공통 요약 지표 계산
  - 샘플 단위 평가 계산
- `ai/preprocessing.py`
  - target 타입별 라벨 생성 배선
- `ai/train.py`
  - validation/test 공통 요약 지표 사용
- `ai/inference.py`
  - 예측 저장과 함께 공통 요약 지표 반환
- `ai/baselines.py`
  - baseline도 같은 요약 지표 포맷 사용
- `ai/backtest.py`
  - 수수료 반영 수익률, 샤프, turnover를 명시적으로 계산

## MAPE 정리
코드 경로의 `MAPE`는 제거했다.

- `ai/inference.py`: per-sample 평가에서 제거
- `ai/baselines.py`: 요약 지표에서 제거
- `ai/train.py`: validation/test 로그에서 제거
- `backend/db/schema.sql`: 신규 스키마에서 제거
- `backend/db/scripts/ensure_runtime_schema.py`: 신규 생성 경로에서 제거

기존 체크포인트 아티팩트 안에 남아 있는 `mape` 문자열은 과거 저장 결과라서 이번 CP 범위에서는 유지한다.
이미 운영 중인 DB에 남아 있는 `prediction_evaluations.mape` 컬럼은 레거시 호환용으로 유지하고, 새 코드에서는 더 이상 쓰지 않는다.

## 레거시 설정 정리
`lambda_width`는 더 이상 loss total에 참여하지 않는다. 다만 과거 설정 파일, 체크포인트, 실험 비교를 깨지 않기 위해 CLI와 config에는 레거시 호환용으로만 남겨두었다.

## baseline 고정
현재 line + band 출력 구조를 baseline 포맷으로 유지했다. 새 목적 함수나 새 target 실험도 기존 저장 포맷(`model_runs`, `predictions`, `prediction_evaluations`)을 그대로 재사용할 수 있다.

## 남은 일
- `market_excess_return`: 시장 벤치마크 수익률 조인 경로 연결
- `rank_target`: 같은 날짜 단면 랭킹 생성 경로 연결
- 필요 시 `prediction_evaluations`에 집계 지표를 별도 적재하는 경로 추가

이번 CP에서 중요한 것은 학습 목적 함수 튜닝이 아니라 비교 가능한 평가판을 먼저 고정한 점이다.
