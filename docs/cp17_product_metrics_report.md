# CP17-P 백테스트/평가 패널 정리 보고서

## 1. 변경 요약

- 백테스트 화면의 latest completed run 조회 기준을 선택한 차트 단위 기준으로 수정했다.
- 백테스트 비용/수수료 반영 지표와 수수료 전 gross 지표를 구분해 표시했다.
- latest completed PatchTST run의 `val_metrics` / `test_metrics` 품질 패널을 추가했다.
- 모델 학습 화면의 상태 라벨을 원문 병기 방식으로 정리했다.
- 주식 보기 화면의 단순함은 유지했고, target type 관련 용어는 계속 노출하지 않는다.

## 2. fetchAiRuns timeframe 전달 여부

수정했다.

백테스트 화면은 이제 아래 형태로 run 목록을 조회한다.

```ts
fetchAiRuns({
  status: "completed",
  modelName: "patchtst",
  timeframe: nextTimeframe,
  limit: 20,
});
```

따라서 전체 latest run을 먼저 고른 뒤 백테스트만 timeframe으로 거르는 문제가 줄었다.

## 3. 백테스트 비용/수수료 지표 필드

- `fee_adjusted_return_pct`
  - 우선 `BacktestSummary.fee_adjusted_return_pct`에서 읽는다.
  - 없으면 기존 `return_pct`를 fallback으로 사용한다.
- `fee_adjusted_sharpe`
  - 우선 `BacktestSummary.fee_adjusted_sharpe`에서 읽는다.
  - 없으면 기존 `sharpe`를 fallback으로 사용한다.
- `avg_turnover`
  - 우선 `BacktestSummary.avg_turnover`에서 읽는다.
  - 없으면 `meta.avg_turnover`를 fallback으로 사용한다.
- `fee_bps`
  - `meta.fee_bps`에서 읽는다.
- `gross_return_pct`
  - `meta.gross_return_pct`에서 읽는다.
- `gross_sharpe`
  - `meta.gross_sharpe`에서 읽는다.

화면에서는 `수수료 반영 수익률/샤프`와 `수수료 전 수익률/샤프`를 분리해서 표시한다.

## 4. val/test metric 패널 표시 지표

백테스트 화면과 모델 학습 화면 모두 아래 지표를 `val_metrics` / `test_metrics`에서 표시한다.

- `coverage`
- `avg_band_width`
- `mae`
- `smape`
- `spearman_ic`
- `top_k_long_spread`
- `long_short_spread`
- `fee_adjusted_return`
- `fee_adjusted_sharpe`
- `fee_adjusted_turnover`
- `direction_accuracy`

밴드 품질이 중요하므로 `coverage`, `avg_band_width`를 가장 위에 배치했고, `direction_accuracy`는 보조 위치로 뒤쪽에 배치했다.

## 5. completed / failed_nan 라벨 처리

모델 학습 화면에서는 원문 상태값을 병기한다.

- `completed` -> `완료(completed)`
- `failed_nan` -> `NaN 실패(failed_nan)`

`run_id`, `target type`, `model_name` 같은 연구/개발 용어는 모델 학습 화면에서 유지한다.

## 6. empty state 동작

- 선택한 차트 단위에 completed run이 없으면 백테스트 화면은 `선택한 차트 단위의 완료 run이 없어서 평가 지표를 표시할 수 없습니다.`를 표시한다.
- 선택한 run에 백테스트 결과가 없으면 `최근 완료 run에 저장된 백테스트 결과가 없습니다.`를 표시한다.
- 모델 학습 화면에서 선택한 상태의 run 목록이 없으면 `선택한 상태의 run이 없습니다.`를 표시한다.
- 선택한 run의 평가 테이블이 없으면 기존처럼 빈 평가 테이블 안내를 표시한다.

## 7. npm run build 결과

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 실패했다. 승인된 실행에서는 정상 통과했다.

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
Route (app) / 78.9 kB, First Load JS 166 kB
```

## 8. 남은 TODO

- 백테스트 결과에 equity curve와 drawdown series가 저장되면 현재 placeholder 영역을 실제 차트로 교체한다.
- metric 값의 단위가 DB에서 더 명확해지면 퍼센트/소수 표시 규칙을 지표별로 세분화한다.
