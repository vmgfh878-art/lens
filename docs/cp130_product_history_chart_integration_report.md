# CP130-P 주식 보기 product-history 차트 연결 보고서

## 1. 목표

주식 보기 1D 차트의 과거 예측 이력을 기존 `/api/v1/stocks/{ticker}/predictions/history`가 아니라 신규 `/api/v1/stocks/{ticker}/predictions/product-history` 응답으로 교체했다.

최신 미래 예측 구간은 기존 `/api/v1/stocks/{ticker}/predictions/latest`를 그대로 사용한다.

## 2. 변경 파일

- `frontend/src/api/client.ts`
  - product-history 응답 타입 추가
  - `fetchProductPredictionHistory()` 추가
- `frontend/src/components/StockView.tsx`
  - 1D rolling history 조회를 product-history API로 변경
  - latest line/band forecast 조회는 기존 latest prediction API 유지
  - product-history empty 응답이면 과거 예측 이력만 숨김
  - AI 밴드 폭 보조지표 계산도 product `band_history` 기반으로 변경
- `frontend/src/components/Chart.tsx`
  - 과거 예측 이력 props 타입을 product-history 전용 타입으로 변경
  - line history는 `line_history[].value`를 사용
  - band history는 `band_history[].lower`, `band_history[].upper`를 사용
  - latest forecast와 rolling history를 별도 series로 유지

## 3. API 연결 내용

추가 client 함수:

```ts
fetchProductPredictionHistory(ticker, {
  timeframe: "1D",
  roles: "all",
  lookbackDays: 370,
})
```

응답 사용 필드:

- `line_history[].asof_date`
- `line_history[].display_horizon`
- `line_history[].value`
- `band_history[].asof_date`
- `band_history[].display_horizon`
- `band_history[].lower`
- `band_history[].upper`
- `latest_asof_date`
- `empty_reason`

제품 차트에서는 더 이상 `/predictions/history`를 사용하지 않는다. 다만 백테스트 화면은 이번 CP 범위가 아니므로 기존 호출을 유지했다.

## 4. 차트 계약

- 과거 보수적 예측선
  - product-history `line_history`만 사용한다.
  - 각 `asof_date`에 대표 horizon 값인 `value` 하나만 표시한다.
  - h1~h5 전체를 이어 붙이지 않는다.
- 과거 AI 밴드
  - product-history `band_history`만 사용한다.
  - 각 `asof_date`에 대표 horizon의 `lower`와 `upper`만 표시한다.
- 최신 미래 forecast
  - 기존 latest prediction의 `forecast_dates` 기준으로 표시한다.
  - 최신 가격일 또는 모델 기준일보다 과거인 forecast point는 표시하지 않는다.
- history와 future forecast
  - 서로 다른 series로 렌더링한다.
  - history 선과 future 점선이 서로 이어지지 않는다.

## 5. 1W/1M 처리

- 1D
  - product-history 조회를 수행한다.
  - 과거 예측 이력과 최신 forecast를 함께 표시한다.
- 1W
  - product-history를 조회하지 않는다.
  - 과도한 예측 history 표시가 나오지 않는다.
- 1M
  - 기존 정책대로 가격 전용 화면을 유지한다.
  - AI history와 latest forecast를 모두 표시하지 않는다.

## 6. Empty State

`T`는 product-history API가 정상 200을 반환하지만 `line_history`, `band_history`가 비어 있고 `empty_reason`은 `ticker_or_filter_has_no_product_history`였다.

프론트에서는 이 경우 가격 차트는 유지하고, 과거 예측 이력 legend와 AI 위험 범위 history만 숨긴다. fake data는 만들지 않았다.

## 7. API 확인

직접 확인한 product-history 응답:

- AAPL: `source=product_rolling_replay`, `date_range=2025-05-05~2026-05-04`, `line_history`/`band_history` 존재
- NVDA: `source=product_rolling_replay`, `line_history`/`band_history` 존재
- T: `source=product_rolling_replay`, history 없음, `empty_reason=ticker_or_filter_has_no_product_history`

latest forecast 날짜 계약 확인:

- AAPL: 최신 가격일 `2026-05-04`, 첫 forecast `2026-05-05`
- MSFT: 최신 가격일 `2026-05-04`, 첫 forecast `2026-05-05`
- NVDA: 최신 가격일 `2026-05-04`, 첫 forecast `2026-05-05`

따라서 latest forecast는 최신 캔들 이후 구간에만 표시 가능한 상태다.

## 8. 검증 결과

실행 명령:

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 샌드박스 내부 첫 실행은 Next child process `spawn EPERM`으로 실패
- 권한 밖 재실행에서 build 통과

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과:

- backend health OK
- CORS OK
- frontend 200 OK
- frontend-static CSS 200 OK
- AAPL 1D 가격 OK
- indicators OK
- 1M 가격 OK
- stock search OK
- LM/BM run, prediction, evaluation, history OK

브라우저 확인:

- AAPL 1D: 가격 차트, 보수적 예측선, AI 위험 범위, 과거 예측 이력 표시
- MSFT 1D: 가격 차트, 보수적 예측선, AI 위험 범위, 과거 예측 이력 표시
- NVDA 1D: 가격 차트, 보수적 예측선, AI 위험 범위, 과거 예측 이력 표시
- T 1D: 가격 차트 유지, product-history 없음 상태로 과거 예측 이력 숨김
- 1D → 1W → 1M → 1D 전환: 런타임 오류 없음
- 브라우저 console error/warn: 0건
- frontend static CSS: 200 확인

## 9. 서버 상태

- 백엔드가 꺼져 있어 `127.0.0.1:8000`으로 재기동했다.
- 프론트는 `127.0.0.1:3000`에서 200 응답 및 CSS 200 상태를 확인했다.
- `scripts/restart_frontend_dev.ps1`는 기존 `logs/frontend_dev.out.log` 파일 잠금으로 실패했지만, 최종 프론트 root/CSS 상태는 정상이다.

## 10. 남은 리스크

- `scripts/check_demo_readiness.ps1`의 `LM history`/`BM history` 항목은 아직 legacy `/predictions/history` 기준 문구를 사용한다. 이번 CP에서는 제품 차트 연결만 바꿨고 readiness 문구 교체는 별도 정리 대상으로 남겼다.
- product-history가 없는 티커는 가격 차트만 보이고 과거 예측 이력은 숨긴다. 이는 의도된 동작이다.
- 브라우저에서 canvas 내부 픽셀 단위의 선 연결 여부는 자동 판독하지 않았고, DOM legend/데이터 계약/콘솔 상태 중심으로 검증했다.

## 11. 금지 사항 준수

- 모델 학습 없음
- inference 재실행 없음
- DB write/delete 없음
- prediction 값 수정 없음
- fake data 생성 없음
- 전략 룰 변경 없음
- composite 사용 없음
