# yfinance daily append gate automation result

- run_at: 2026-05-10 22:43:24 +09:00ST
- run_date: 2026-05-10
- mode: DryRun
- exit_code: 0
- status: PASS_WITH_NO_NEW_DAY
- eodhd_cancel_judgement: NO_NEW_COMPLETED_DAY_NOT_ENOUGH_FOR_CANCEL

## Command

~~~powershell
.\scripts\run_yfinance_daily_append_gate_automation.ps1 -DryRun -RunDate 2026-05-10 -LookbackDays 7 -Tickers AAPL,MSFT,NVDA,TSLA,NFLX
~~~

## Summary

신규 완료 거래일 row는 없었지만 daily update 순서, context refresh 경계, partial period gate, thin upload 경계가 문서와 dry-run으로 고정됐다.

## Snapshot

- yfinance_price_latest_date: 2026-05-08
- yfinance_indicator_1d_latest_date: 2026-05-08
- yfinance_indicator_1w_latest_date: 2026-05-08
- product_history_latest_asof_date: 2026-05-04
- fetch_status: PASS_WITH_NO_NEW_DAY
- append_candidate_rows: 0
- append_candidate_tickers:
- successful_fetch_tickers: AAPL, MSFT, NFLX, NVDA, TSLA
- empty fetch tickers:
- fetch failed tickers:
- failed tickers:
- fallback used count: 0

## Forbidden Action Check

- eodhd_call: False
- Supabase bulk read: False
- Supabase bulk write: False
- model_training: False
- full_inference_save: False

## Warnings

- no new completed market day rows available
- no new SEC filing rows for rehearsal tickers
- product inference/thin upload left as dry-run boundary

## Failures

- none

## Artifacts

- metrics: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_metrics_20260510_20260510_224204.json
- stdout: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_stdout_20260510_20260510_224204.log
- stderr: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_stderr_20260510_20260510_224204.log
