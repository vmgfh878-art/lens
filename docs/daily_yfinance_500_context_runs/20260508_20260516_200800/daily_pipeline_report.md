# CP189 Daily YFinance 500 + Context Feature Automation Result

보낼 곳: D/G
모델 학습 없음
DB write 없음
EODHD fallback 없음
Supabase bulk 0
Codex automation 대체 여부: Windows Task Scheduler 전환 준비 완료, Codex automation은 2~3회 Windows 성공 후 pause/delete 권장
등록 방식: 기본 AtLogon, fallback Daily 10:00 KST
CP186/187/187B feature refresh 포함 여부: 포함

## Summary

- status: FAIL_AUTOMATION_UNSAFE
- run_date: 2026-05-08
- mode: dry_run
- lock: lock_acquired
- price_status: DRY_RUN_FETCH_BLOCKED
- price_gate_status: DRY_RUN_FETCH_BLOCKED
- context_status: validated_existing

## Context Validation

```json
{
  "price_yfinance_500": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\price_data_yfinance_500.parquet",
    "exists": true,
    "row_count": 1383438,
    "ticker_count": 501,
    "date_min": "2015-01-02",
    "date_max": "2026-05-08",
    "column_count": 15,
    "duplicate_count": 0,
    "nonfinite_count": 2766876
  },
  "indicators_yfinance_1D_500": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\indicators_yfinance_1D_500.parquet",
    "exists": true,
    "row_count": 1351677,
    "ticker_count": 501,
    "date_min": "2015-03-30",
    "date_max": "2026-05-08",
    "column_count": 38,
    "duplicate_count": 0,
    "nonfinite_count": 0
  },
  "indicators_yfinance_1W_500": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\indicators_yfinance_1W_500.parquet",
    "exists": true,
    "row_count": 257802,
    "ticker_count": 500,
    "date_min": "2016-02-19",
    "date_max": "2026-05-08",
    "column_count": 38,
    "duplicate_count": 0,
    "nonfinite_count": 0
  },
  "cp186_future_event": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\cp186_future_event\\future_event_context_features_1D.parquet",
    "exists": true,
    "row_count": 1351677,
    "ticker_count": 501,
    "date_min": "2015-03-30",
    "date_max": "2026-05-08",
    "column_count": 25,
    "duplicate_count": 0,
    "nonfinite_count": 0
  },
  "cp187_future_event_v2": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\cp187_future_event\\future_event_context_features_1D_v2.parquet",
    "exists": true,
    "row_count": 1351677,
    "ticker_count": 501,
    "date_min": "2015-03-30",
    "date_max": "2026-05-08",
    "column_count": 89,
    "duplicate_count": 0,
    "nonfinite_count": 0
  },
  "cp187b_vix_short_context": {
    "path": "C:\\Users\\user\\lens\\data\\parquet\\cp187b_vix_short_context\\vix_short_context_features_1D.parquet",
    "exists": true,
    "row_count": 1383438,
    "ticker_count": 501,
    "date_min": "2015-01-02",
    "date_max": "2026-05-08",
    "column_count": 19,
    "duplicate_count": 0,
    "nonfinite_count": 0
  }
}
```

## Forbidden Action Check

```json
{
  "model_training": false,
  "product_save_run": false,
  "db_write": false,
  "inference_save": false,
  "eodhd_fallback": false,
  "supabase_bulk_read_write": false,
  "checkpoint_modify": false,
  "frontend_backend_runtime_modify": false
}
```
