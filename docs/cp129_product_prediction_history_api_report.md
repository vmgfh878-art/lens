# CP129-DG Product Prediction History API Report

생성일: 2026-05-06

## 1. Executive Summary

최종 판정: PASS

CP128에서 생성한 `data/parquet/product_prediction_history_1D.parquet`를 백엔드 API로 노출했다. 새 endpoint는 local parquet와 manifest만 읽으며, 기존 `predictions` 테이블의 test split bulk row나 latest-only product row를 섞지 않는다.

신규 endpoint:

```text
GET /api/v1/stocks/{ticker}/predictions/product-history
```

## 2. 구현 범위

변경 파일:
- `backend/app/services/product_prediction_history_svc.py`
- `backend/app/routers/v1/stocks.py`
- `backend/app/schemas/stocks.py`
- `backend/tests/test_product_prediction_history_api.py`

산출물:
- `docs/cp129_product_prediction_history_api_report.md`
- `docs/cp129_product_prediction_history_api_metrics.json`

프론트는 수정하지 않았다.

## 3. API 계약

Query:

| 항목 | 기본값 | 설명 |
|---|---:|---|
| `timeframe` | `1D` | 현재 product replay parquet는 1D만 지원 |
| `roles` | `all` | `all`, `line`, `band`, `line,band` |
| `run_id` | null | 선택 run_id 필터 |
| `limit` | null | role별 최근 row 수 |
| `lookback_days` | null | latest asof_date 기준 최근 N일 |

Response 핵심:
- `ticker`
- `timeframe`
- `latest_asof_date`
- `source=product_rolling_replay`
- `line_history[]`
- `band_history[]`
- `manifest_summary`
- `empty_reason`

`source`는 응답 계약에서 `product_rolling_replay`로 고정했다. parquet 내부 source 값인 `rolling_replay`는 생성 provenance이고, API 응답에서는 제품 전용 history임을 더 명확히 드러낸다.

## 4. 기존 Endpoint와의 분리

기존 endpoint:

```text
GET /api/v1/stocks/{ticker}/predictions/history
```

이 endpoint는 계속 legacy/evaluation prediction row 조회 경로로 남겨둔다. 제품 차트 history는 이 endpoint를 사용하면 안 된다.

새 endpoint는 아래를 읽지 않는다:
- Supabase `price_data`
- Supabase `indicators`
- Supabase `predictions`
- product latest-only row
- test split bulk prediction row

## 5. 실제 API 확인

CP128 실 parquet 기준 TestClient 확인 결과:

| ticker | status | latest_asof_date | line rows | band rows | duplicate | lower > upper |
|---|---:|---|---:|---:|---:|---:|
| AAPL | 200 | 2026-05-04 | 251 | 251 | 0 | 0 |
| MSFT | 200 | 2026-05-04 | 251 | 251 | 0 | 0 |
| NVDA | 200 | 2026-05-04 | 251 | 251 | 0 | 0 |
| T | 200 | null | 0 | 0 | 0 | 0 |

`T`는 CP128 parquet 대상 5티커에 없으므로 empty state를 반환한다.

## 6. 검증

실행한 검증:

```powershell
..\.venv\Scripts\python.exe -m py_compile app\services\product_prediction_history_svc.py app\routers\v1\stocks.py app\schemas\stocks.py tests\test_product_prediction_history_api.py
.\.venv\Scripts\python.exe -c "import sys, unittest; sys.path[:0]=['backend','.']; unittest.main(module='backend.tests.test_product_prediction_history_api', exit=True)"
```

결과:
- py_compile PASS
- backend unittest 5개 PASS
- AAPL/MSFT/NVDA 실제 API 응답 PASS
- T empty state PASS
- metrics JSON parse PASS

## 7. 금지 작업 확인

| 금지 작업 | 발생 |
|---|---:|
| DB write | false |
| Supabase upload | false |
| Supabase price/indicator/prediction 대량 read | false |
| 모델 학습 | false |
| inference 저장 | false |
| composite 사용 | false |
| 프론트 수정 | false |
| fake data | false |

## 8. 다음 단계

프론트 연결 CP에서는 주식 보기 1D 차트의 과거 history source를 기존 `/predictions/history`에서 새 `/predictions/product-history`로 바꾸면 된다. 미래 forecast는 기존 latest-only prediction endpoint를 계속 사용하고, 두 series를 같은 선으로 이어 붙이면 안 된다.
