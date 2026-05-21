# 제품용 Prediction History 계약

작성일: 2026-05-06  
상태: CP127 기준 SoT  
범위: 주식 보기 차트의 제품용 과거 예측 이력과 최신 미래 forecast를 평가용 prediction row에서 분리한다.

## 1. 핵심 원칙

제품 차트는 기존 `predictions/history`를 과거 예측 이력으로 사용하지 않는다.

이유:
- `predictions`에는 평가용 test split bulk row와 product latest-only row가 함께 들어 있다.
- 기존 `/api/v1/stocks/{ticker}/predictions/history` endpoint는 `ticker`, `run_id`, `limit`만으로 row를 가져온다.
- row의 목적이 평가용인지, 제품 최신 forecast인지, 제품 rolling replay인지 endpoint 계약으로 분리되지 않는다.
- 따라서 이 endpoint를 제품 chart history로 쓰면 다시 긴 공백이나 대각선 연결 문제가 생길 수 있다.

계약:
- 평가용 prediction: 성능 비교와 백테스트용
- product latest-only prediction: 최신 asof 기준 미래 forecast용
- product rolling history: 제품 차트의 과거 1년 history용

## 2. 데이터 역할

| 데이터 | 시간축 | 용도 | 제품 차트 기본 history 사용 |
|---|---|---|---|
| 평가용 test split bulk prediction | `forecast_date` 중심 평가 | metric, report, backtest | 금지 |
| product latest-only prediction | `forecast_dates` h1~h5 | 최신 미래 forecast | history에는 금지, future series만 허용 |
| product rolling replay history | `asof_date` | 과거 1년 예측 이력 | 허용 |

## 3. 저장 대상

기본 저장소:
- local parquet

권장 경로:

```text
data/product_prediction_history/timeframe=1D/source=yfinance/asof_start={start}/asof_end={end}/product_prediction_history_1D.parquet
data/product_prediction_history/timeframe=1D/source=yfinance/asof_start={start}/asof_end={end}/manifest.json
```

Supabase 저장:
- 기본은 저장하지 않는다.
- 필요하면 선택 ticker 또는 최근 N개 asof만 thin upload한다.
- 전체 500티커 1년 history를 Supabase에 저장하는 것은 기본 금지 방향이다.

## 4. 필수 컬럼

`product_prediction_history_1D` 또는 local parquet equivalent는 아래 컬럼을 가진다.

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `ticker` | string | 티커 |
| `timeframe` | string | `1D` |
| `role` | string | `line` 또는 `band` |
| `run_id` | string | 제품 model run id |
| `asof_date` | date | 모델 입력 기준일 |
| `display_horizon` | int | 차트 대표 horizon. 1D history는 `5` |
| `display_date` | date | 차트에 찍을 날짜 |
| `line_value` | float null | line role의 표시값 |
| `lower_value` | float null | band role의 lower 표시값 |
| `upper_value` | float null | band role의 upper 표시값 |
| `source` | string | `rolling_replay` 또는 `latest_live` |
| `model_feature_hash` | string | replay 입력 feature/hash |
| `created_at` | timestamp | 생성 시각 |

권장 추가 컬럼:
- `provider`
- `feature_version`
- `price_snapshot_hash`
- `indicator_snapshot_hash`
- `display_policy`
- `model_version`

## 5. Role별 값 계약

### line

| 필드 | 값 |
|---|---|
| `role` | `line` |
| `run_id` | `patchtst-1D-efad3c29d803` |
| `display_horizon` | `5` |
| `display_date` | `asof_date` |
| `line_value` | `conservative_series[h5]` 우선, 없으면 `line_series[h5]` |
| `lower_value` | null |
| `upper_value` | null |
| `source` | `rolling_replay` |

### band

| 필드 | 값 |
|---|---|
| `role` | `band` |
| `run_id` | `cnn_lstm-1D-d0c780dee5e8` |
| `display_horizon` | `5` |
| `display_date` | `asof_date` |
| `line_value` | null |
| `lower_value` | `lower_band_series[h5]` |
| `upper_value` | `upper_band_series[h5]` |
| `source` | `rolling_replay` |

## 6. 표시 계약

### 과거 1년 history

계약:
- source는 `rolling_replay`
- 최근 1년 asof_date만 사용
- 차트 시간축은 `asof_date`
- 1D 대표값은 h5
- h1~h5를 하나의 선으로 이어 붙이지 않는다
- product latest-only row와 섞지 않는다
- 평가용 test split bulk row와 섞지 않는다

line:
- `role=line`
- `display_date=asof_date`
- `line_value`만 사용

band:
- `role=band`
- `display_date=asof_date`
- `lower_value`, `upper_value`만 사용

### 최신 미래 forecast

계약:
- latest prediction 1건만 사용
- source는 `latest_live`
- 시간축은 `forecast_dates`
- h1~h5만 표시
- latest price date 이후 날짜만 표시
- 과거 history와 별도 series로 그린다

주의:
- future forecast는 `display_date=forecast_date`, `display_horizon=1..5` 형태로 변환할 수 있다.
- 그래도 화면에서는 history series와 같은 선으로 연결하지 않는다.

### 평가용 데이터

계약:
- forecast_date 기준 actual 비교에 사용한다.
- metric, backtest, report에서만 사용한다.
- 제품 차트 기본 history에는 사용하지 않는다.

## 7. Rolling Replay 생성 방식

제품 history는 기존 test split 저장 row를 재활용하지 않고, 제품 모델 checkpoint로 별도 rolling replay해서 만든다.

기준:
- line run: `patchtst-1D-efad3c29d803`
- band run: `cnn_lstm-1D-d0c780dee5e8`
- timeframe: `1D`
- provider/source: `yfinance`
- feature version: `v3_adjusted_ohlc`
- display horizon: `5`
- window: 최근 1년
- composite 사용 금지

절차:
1. local yfinance price/indicator parquet를 읽는다.
2. replay 대상 asof_date를 최근 1년 거래일로 잡는다.
3. 각 asof_date에서 해당 날짜까지의 feature만 사용한다.
4. 제품 checkpoint로 forward-only prediction을 만든다.
5. line/band 각각 h5 scalar만 차트 history row로 저장한다.
6. full prediction array는 필요하면 local artifact에만 둔다.
7. manifest에 run_id, hash, ticker_count, asof range, row_count를 기록한다.

중요:
- 새 학습이 아니다.
- DB write가 아니다.
- inference 결과를 기존 `predictions` history에 대량 저장하지 않는다.

## 8. Manifest 계약

manifest 필수 필드:
- `line_run_id`
- `band_run_id`
- `timeframe`
- `source`
- `provider`
- `feature_version`
- `model_feature_hash`
- `ticker_count`
- `asof_start`
- `asof_end`
- `display_horizon`
- `row_count`
- `created_at`
- `input_price_hash`
- `input_indicator_hash`

load gate:
- run_id가 제품 run과 다르면 실패
- source/provider가 `yfinance`가 아니면 실패
- feature hash가 현재 local snapshot과 맞지 않으면 stale로 보고 재생성
- display_horizon이 5가 아니면 1D 제품 history로 사용 금지

## 9. API 계약 제안

기존 endpoint:
- `/api/v1/stocks/{ticker}/predictions/history`
- 평가/legacy 조회용으로 남길 수 있지만 제품 chart history 기본 endpoint로 쓰지 않는다.

신규 또는 대체 endpoint:

```text
GET /api/v1/stocks/{ticker}/product-predictions/history?timeframe=1D&window=1y
GET /api/v1/stocks/{ticker}/predictions/latest?run_id={product_run_id}
```

응답 분리:

```json
{
  "history": [
    {
      "ticker": "NVDA",
      "timeframe": "1D",
      "role": "line",
      "run_id": "patchtst-1D-efad3c29d803",
      "asof_date": "2026-03-26",
      "display_horizon": 5,
      "display_date": "2026-03-26",
      "line_value": 180.12,
      "lower_value": null,
      "upper_value": null,
      "source": "rolling_replay",
      "model_feature_hash": "..."
    }
  ],
  "latest_forecast": {
    "line": "...latest PredictionResult...",
    "band": "...latest PredictionResult..."
  }
}
```

프론트 계약:
- `history`는 asof_date/display_date 기준으로 과거 구간에만 표시
- `latest_forecast`는 forecast_dates 기준으로 future 구간에만 표시
- 두 series는 연결하지 않음

## 10. 비용 정책

가정:
- 1년 거래일: 252
- role: line, band 2개
- ticker당 1년 row 수: 504

| 저장 방식 | 100티커 | 500티커 | 판단 |
|---|---:|---:|---|
| local parquet | 50,400 rows | 252,000 rows | 기본 권장 |
| Supabase 선택 5티커 | 2,520 rows | - | 데모/검증용 가능 |
| Supabase top 50 또는 선택 50티커 | 25,200 rows | - | 선택적 허용 |
| Supabase 전체 500티커 | - | 252,000 rows | 기본 비권장 |

Supabase 전체 500티커 1년 history는 scalar row만 저장해도 index와 JSON overhead 때문에 Free DB slim 정책과 충돌할 수 있다. 제품 화면은 한 번에 한 ticker history만 필요하므로 local parquet에서 ticker 단위로 읽는 쪽이 비용과 egress 면에서 맞다.

## 11. 최종 판단

기존 `predictions` row를 제품 history로 계속 쓰는 구조는 FAIL이다.

채택 계약:
- 제품 과거 history는 `product_prediction_history_1D` local parquet로 분리
- latest future forecast는 기존 product latest-only prediction 1건 사용
- 평가용 prediction은 제품 chart history에서 제외
- Supabase에는 필요 시 선택 ticker 또는 최근 N개만 thin upload
