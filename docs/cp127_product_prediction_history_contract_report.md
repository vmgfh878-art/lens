# CP127-DG 제품용 Rolling Prediction History 계약 재설계

생성일: 2026-05-06  
상태: PASS  
범위: 데이터 계약 감사와 설계. 프론트 수정, 모델 학습, inference 재실행, DB write/delete는 하지 않았다.

## 1. Executive Summary

결론: 기존 `predictions/history` endpoint를 제품 차트의 과거 예측 이력으로 계속 쓰면 안 된다. 이 endpoint는 `run_id`, `ticker`, `limit` 기준으로 prediction row를 가져오며, row의 목적이 평가용 test split bulk인지 product latest-only인지 제품 rolling history인지 분리하지 않는다.

제품 차트는 세 데이터를 분리해야 한다.

| 데이터 | 용도 | 시간축 | 제품 차트 기본 history |
|---|---|---|---|
| test split bulk prediction | 평가, 리포트, 백테스트 | forecast_date 중심 | 금지 |
| product latest-only prediction | 최신 미래 forecast | forecast_dates h1~h5 | future series만 허용 |
| product rolling replay history | 과거 1년 예측 이력 | asof_date | 허용 |

최종 설계:
- 과거 1년 history는 제품 checkpoint로 별도 rolling replay해서 만든다.
- 기본 저장소는 local parquet다.
- Supabase에는 필요할 때 선택 ticker 또는 최근 N개만 thin upload한다.
- composite는 사용하지 않는다.

## 2. 기존 Endpoint 재점검

대상:
- `backend/app/routers/v1/stocks.py`
- `backend/app/services/api_service.py`
- `backend/app/repositories/prediction_repo.py`

현재 `/api/v1/stocks/{ticker}/predictions/history` 계약:

| 항목 | 현재 |
|---|---|
| 필수 인자 | `run_id` |
| 선택 인자 | `limit`, 기본 90, 최대 200 |
| ticker 필터 | path ticker |
| 정렬 | `asof_date desc`, `decision_time desc` |
| 반환 순서 | 최근 row를 가져온 뒤 reverse |
| storage contract 필터 | 없음 |
| thin upload 제외 필터 | 없음 |
| rolling replay 전용 필터 | 없음 |

코드 근거:
- `backend/app/routers/v1/stocks.py:105`: `/predictions/history` route
- `backend/app/routers/v1/stocks.py:113`: `run_id` query
- `backend/app/routers/v1/stocks.py:114`: `limit` query
- `backend/app/repositories/prediction_repo.py:70`: `fetch_prediction_history_by_run`
- `backend/app/repositories/prediction_repo.py:82`: `.eq("run_id", run_id)`
- `backend/app/repositories/prediction_repo.py:83`: `.order("asof_date", desc=True)`
- `backend/app/repositories/prediction_repo.py:85`: `.limit(limit)`

판단:
- 이 endpoint는 run history 조회용으로는 쓸 수 있다.
- 제품 표시용 rolling history endpoint로는 부적합하다.

## 3. CP122-4 관찰 요약

CP122-4에서 NVDA/AAPL/MSFT, 1D 제품 line/band run을 확인했다.

| ticker | layer | rows | bulk max asof | latest-only asof | 수정 전 gap |
|---|---|---:|---|---|---:|
| NVDA | line | 25 | 2026-04-01 | 2026-05-01, 2026-05-04 | 30일 |
| NVDA | band | 25 | 2026-04-01 | 2026-05-01, 2026-05-04 | 30일 |
| AAPL | line | 26 | 2026-04-02 | 2026-05-01, 2026-05-04 | 29일 |
| AAPL | band | 26 | 2026-04-02 | 2026-05-01, 2026-05-04 | 29일 |
| MSFT | line | 25 | 2026-04-01 | 2026-05-01, 2026-05-04 | 30일 |
| MSFT | band | 25 | 2026-04-01 | 2026-05-01, 2026-05-04 | 30일 |

의미:
- product latest-only row는 `2026-05-01`, `2026-05-04`에 있다.
- bulk test history는 4월 초까지만 이어진다.
- 두 종류를 하나의 rolling history로 연결하면 큰 공백이 대각선처럼 보인다.
- CP122-4는 프론트에서 thin row 제외와 gap 차단으로 증상을 막았지만, 데이터 계약 자체는 아직 분리되지 않았다.

## 4. Row 구분 가능성

현재 product latest-only row는 대체로 구분 가능하다.

구분 신호:
- `meta.thin_upload=true`
- `meta.product_latest_only=true`
- `meta.storage_contract='product_latest_only'`
- `meta.layer='line'` 또는 `meta.layer='band'`
- `source/provider=yfinance`

하지만 test split bulk row는 안전하게 구분하기 어렵다.

문제:
- 기존 bulk row는 `meta={}`인 경우가 있다.
- `storage_contract='evaluation_bulk'` 같은 명시 계약이 없다.
- created_at이나 asof range로 추정할 수는 있지만 제품 계약으로 쓰기에는 약하다.

판단:
- latest-only row는 어느 정도 제외 가능하다.
- test split bulk row는 “제품 history가 아니다”라고 명시할 수는 있지만, 기존 endpoint에서 완전히 안전하게 필터링하는 계약은 없다.
- 따라서 제품 history는 별도 데이터셋으로 생성해야 한다.

## 5. 제품 차트가 필요한 History 계약

제품 차트는 “이 모델이 각 과거 기준일에 5일 뒤를 어떻게 봤는가”를 보여줘야 한다.

1D history 계약:
- 기간: 최근 1년
- 기준일: `asof_date`
- 표시일: `display_date=asof_date`
- 대표 horizon: h5
- line 값: 제품 line run의 h5 대표값
- band 값: 제품 band run의 h5 lower/upper
- h1~h5 전체를 과거 구간에서 한 선으로 이어 붙이지 않음
- latest-only row와 섞지 않음
- test split bulk row와 섞지 않음

제품 run:
- line: `patchtst-1D-efad3c29d803`
- band: `cnn_lstm-1D-d0c780dee5e8`
- composite: 금지

## 6. Product Prediction History 데이터셋

SoT 문서:
- `docs/product_prediction_history_contract.md`

기본 저장소:
- local parquet

제안 경로:

```text
data/product_prediction_history/timeframe=1D/source=yfinance/asof_start={start}/asof_end={end}/product_prediction_history_1D.parquet
data/product_prediction_history/timeframe=1D/source=yfinance/asof_start={start}/asof_end={end}/manifest.json
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `ticker` | 티커 |
| `timeframe` | `1D` |
| `role` | `line` 또는 `band` |
| `run_id` | 제품 run id |
| `asof_date` | 모델 기준일 |
| `display_horizon` | 1D history는 `5` |
| `display_date` | history는 `asof_date` |
| `line_value` | line role 표시값 |
| `lower_value` | band role 하단 |
| `upper_value` | band role 상단 |
| `source` | `rolling_replay` 또는 `latest_live` |
| `model_feature_hash` | 입력 feature/hash |
| `created_at` | 생성 시각 |

## 7. asof_date와 forecast_date 역할 분리

| 용도 | 기준 날짜 | 값 |
|---|---|---|
| 제품 과거 history | `asof_date` | h5 대표값 |
| 제품 미래 forecast | `forecast_dates` | latest prediction h1~h5 |
| 평가/metric | `forecast_date`와 actual | 실제값 비교 |

중요:
- 과거 history의 `display_date`는 `asof_date`다.
- 미래 forecast의 `display_date`는 `forecast_date`다.
- 두 series는 시각적으로 연결하지 않는다.

## 8. 1년 History 생성 방식

기존 prediction row 재사용 금지. rolling replay로 생성한다.

절차:
1. local yfinance price/indicator parquet를 읽는다.
2. 최근 1년 거래일을 replay asof_date로 잡는다.
3. 각 asof_date에서 그 날짜까지의 feature만 사용한다.
4. 제품 checkpoint로 forward-only prediction을 만든다.
5. line role은 h5 `line_value` 1개를 저장한다.
6. band role은 h5 `lower_value`, `upper_value`를 저장한다.
7. full forecast arrays는 local artifact에만 둘 수 있다.
8. 차트용 history parquet에는 scalar display point만 저장한다.

검증 gate:
- source/provider `yfinance`
- feature_version `v3_adjusted_ohlc`
- model_feature_hash 기록
- ticker coverage 기록
- asof range 기록
- line/band row count 일치 확인
- lower <= upper 확인
- NaN/Inf 0
- composite 0

## 9. Local Parquet vs Supabase Thin 비용 비교

가정:
- 1년 거래일 252
- role 2개
- ticker당 1년 row 수 504

| 저장 방식 | row 수 | 장점 | 리스크 | 판단 |
|---|---:|---|---|---|
| local parquet 100티커 | 50,400 | Supabase egress 없음, 원천 구조와 일치 | backend가 local file serving 필요 | 권장 |
| local parquet 500티커 | 252,000 | 용량 작고 archive 쉬움 | 최초 replay 시간 필요 | 권장 |
| Supabase 선택 5티커 | 2,520 | 데모 검증 쉬움 | row pruning 필요 | 선택 허용 |
| Supabase 선택 50티커 | 25,200 | 제품 top-k history 가능 | egress/용량 관리 필요 | 선택 허용 |
| Supabase 500티커 전체 | 252,000 | API 단순 | Free DB slim 구조와 충돌 | 비권장 |

추정:
- local parquet는 수 MB~수십 MB 수준으로 충분하다.
- Supabase scalar row는 index 포함 대략 800~2,000 bytes로 보면 100티커 1년 history가 약 40~100 MB, 500티커는 약 200~500 MB 이상이 될 수 있다.
- 제품 화면은 한 번에 한 ticker history만 필요하므로 Supabase 전체 저장 이점이 작다.

## 10. 프론트가 봐야 할 API/파일

이번 CP에서는 프론트 수정 금지이므로 구현하지 않았다.

다음 구현 CP의 목표 계약:

```text
GET /api/v1/stocks/{ticker}/product-predictions/history?timeframe=1D&window=1y
GET /api/v1/stocks/{ticker}/predictions/latest?run_id={product_run_id}
```

응답 구조:
- `history`: rolling replay scalar points
- `latest_forecast`: 기존 latest prediction 또는 horizon별 future display points

프론트 표시:
- `history`는 `display_date=asof_date` 기준으로 과거 구간에 표시
- `latest_forecast`는 `forecast_dates` 기준으로 미래 구간에 표시
- history와 latest forecast를 한 선으로 연결하지 않음

## 11. 기존 predictions row를 제품 history로 계속 써도 되는가

판정: 안 된다.

이유:
1. 기존 `predictions`는 평가 저장소와 제품 최신 저장소가 섞여 있다.
2. `predictions/history`는 storage contract 필터가 없다.
3. test split bulk row는 제품 live replay가 아니다.
4. product latest-only row는 과거 history가 아니라 미래 forecast용이다.
5. CP122-4에서 실제로 NVDA/AAPL/MSFT의 29~30일 gap 문제가 확인됐다.

따라서 제품 history는 별도 rolling replay dataset으로 만든다.

## 12. PASS/WARN/FAIL 판정

| 항목 | 판정 |
|---|---|
| 평가용 prediction과 제품 표시용 rolling history 분리 | PASS |
| 1년 history 생성 방식 정의 | PASS |
| 프론트가 봐야 할 API/파일 계약 | PASS |
| local parquet 우선 정책 | PASS |
| Supabase 비용 영향 정리 | PASS |
| 기존 predictions row를 제품 history로 계속 사용 | FAIL |

최종 판정: PASS

단, 구현 관점에서는 다음 CP가 필요하다.
- rolling replay 생성 runner
- local parquet manifest
- product-predictions history API
- 프론트가 기존 predictions/history 대신 신규 product history를 보도록 연결

## 13. 읽기 전용 명령 목록

- `Get-Content backend/app/repositories/prediction_repo.py`
- `Get-Content backend/app/services/api_service.py`
- `Get-Content backend/app/routers/v1/stocks.py`
- `Get-Content docs/cp122_4_prediction_history_discontinuity_metrics.json`
- `Select-String`으로 endpoint, chart h5 대표값, thin upload 필터 근거 확인

이번 CP에서는 Supabase 대량 read, DB write/delete, 모델 학습, inference 재실행, 프론트 수정, fake data 생성을 하지 않았다.
