# CP122-4-DP NVDA 예측 history 단절 감사 및 차트 계약 수정

생성일: 2026-05-06

## 1. 요약

최종 판정: `FIXED`

NVDA 1D 차트의 2026-03-26 전후 예측선/AI 밴드 깨짐은 모델 값 문제가 아니라 차트 series 계약 문제였다. 저장된 prediction history에는 과거 bulk test row와 5월 product latest-only thin upload row가 같은 run_id/ticker/history 응답에 섞여 있었다. 프론트 차트는 이 둘을 하나의 rolling history line으로 연결해 `2026-04-01 -> 2026-05-01` 같은 큰 공백을 직선으로 이어 그렸다.

수정:
- product latest-only thin upload row는 rolling history series에서 제외
- rolling history 내부에 `10`일 초과 공백이 있으면 최신 contiguous segment만 사용
- latest prediction의 future forecast는 기존처럼 `forecast_dates` 기준으로 별도 표시

## 2. 대상과 범위

- ticker: `NVDA`, `AAPL`, `MSFT`
- timeframe: `1D`
- line run: `patchtst-1D-efad3c29d803`
- band run: `cnn_lstm-1D-d0c780dee5e8`
- 감사 기간: `2026-03-01` ~ `2026-05-06`
- history endpoint 모사 limit: `90`

## 3. 저장 row 요약

| ticker | layer | rows | bulk max asof | thin asof | 수정 전 max gap days | 수정 후 max gap days |
|---|---|---:|---|---|---:|---:|
| NVDA | line | 25 | 2026-04-01 | 2026-05-01,2026-05-04 | 30 | 0 |
| NVDA | band | 25 | 2026-04-01 | 2026-05-01,2026-05-04 | 30 | 0 |
| AAPL | line | 26 | 2026-04-02 | 2026-05-01,2026-05-04 | 29 | 0 |
| AAPL | band | 26 | 2026-04-02 | 2026-05-01,2026-05-04 | 29 | 0 |
| MSFT | line | 25 | 2026-04-01 | 2026-05-01,2026-05-04 | 30 | 0 |
| MSFT | band | 25 | 2026-04-01 | 2026-05-01,2026-05-04 | 30 | 0 |

핵심 관찰:
- line과 band의 asof coverage는 세 ticker 모두 동일했다.
- 모든 row의 `forecast_dates`, `line_series`, `conservative_series`, `upper_band_series`, `lower_band_series` 길이는 5로 맞았다.
- 과거 bulk row는 `meta={}`이며 `created_at`은 2026-05-01 계열이다.
- product latest-only row는 `meta.thin_upload=true`, `meta.layer=line/band`, `source=yfinance`를 가진다.

## 4. 단절 원인

원인:
1. `/predictions/history`는 run_id 기준 최근 row를 그대로 반환한다.
2. 해당 응답에 bulk test history와 product latest-only row가 섞인다.
3. 차트는 rolling history를 `asof_date`에 h5 대표값으로 찍는다.
4. latest-only row까지 rolling history에 포함되면서 과거 bulk 마지막 지점에서 5월 latest 지점까지 긴 대각선이 생겼다.

NVDA 기준 대표 사례:
- bulk history 마지막: `2026-04-01`
- latest-only row: `2026-05-01`, `2026-05-04`
- 수정 전 max gap: `30`일
- 수정 후 max gap: `0`일

## 5. 차트 계약

수정 후 계약:
- rolling history: thin upload row 제외
- rolling history 시간축: `asof_date`
- rolling history 값: h5 대표값. 길이가 짧으면 마지막 horizon
- future forecast: latest prediction 1건의 `forecast_dates`와 h1~h5 series
- 큰 공백: `10`일 초과 gap은 한 line series로 잇지 않음

주의:
- h5 대표 history를 `asof_date`에 찍는 계약은 유지했다.
- 다만 product latest-only row는 rolling history가 아니라 latest future forecast layer로만 해석한다.

## 6. History endpoint 확인

현재 endpoint는 run_id, ticker, limit 기준만 사용한다. run_id가 제품 run으로 고정되어 있으므로 timeframe/horizon 혼합은 이번 사례의 주원인이 아니었다. 주원인은 storage contract가 다른 row를 같은 history로 반환한 점이다.

장기 개선:
- API에 `history_kind=rolling_only|include_latest` 또는 `exclude_thin_upload=true` 옵션 추가
- `PredictionData.meta.storage_contract`를 명시적으로 `bulk_backtest_history` 또는 `product_latest_only`로 저장
- rolling AI band history를 Supabase latest-only와 분리해 local parquet 또는 별도 thin history table로 제공

## 7. 파일 변경

- `frontend/src/components/Chart.tsx`

## 8. 금지 작업 확인

| 금지 항목 | 발생 |
|---|---|
| 모델 재학습 | false |
| inference 재실행 | false |
| prediction DB write | false |
| DB row delete | false |
| Supabase price/indicator 대량 read | false |

## 9. 검증

metrics JSON에 수정 전/후 chart history gap이 기록되어 있다.

실행한 검증:
- `python -m py_compile scripts/cp122_4_prediction_history_discontinuity_audit.py`: PASS
- `python -m json.tool docs/cp122_4_prediction_history_discontinuity_metrics.json`: PASS
- 핵심 assertion: `decision.status=FIXED`, NVDA line/band 수정 전 max gap `30`, 수정 후 max gap `0`, line/band coverage 동일, 금지 작업 flag 없음: PASS
- `npm run build` in `frontend`: PASS

참고:
- 첫 `npm run build`는 sandbox child process 제한으로 `spawn EPERM`이 발생했다.
- 동일 빌드를 권한 상승으로 재실행했을 때 Next.js production build, lint, type check가 통과했다.
