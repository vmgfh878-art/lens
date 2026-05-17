# yfinance daily append gate automation result

- run_at: 2026-05-15 01:55:48 +09:00ST
- run_date: 2026-05-15
- mode: Apply
- exit_code: 0
- status: PASS_APPEND_DONE
- eodhd_cancel_judgement: EODHD_CANCEL_CANDIDATE_NEEDS_2_TO_3_CONSECUTIVE_PASSES

## Command

~~~powershell
.\scripts\run_yfinance_daily_append_gate_automation.ps1 -RunDate 2026-05-15 -LookbackDays 7 -Tickers AAPL,MSFT,NVDA,TSLA,NFLX
~~~

## Summary

yfinance 완료 거래일 row를 local parquet에 append했고 1D/1W indicator refresh gate까지 통과했다.

## Snapshot

- yfinance_price_latest_date: 2026-05-14
- yfinance_indicator_1d_latest_date: 2026-05-14
- yfinance_indicator_1w_latest_date: 2026-05-08
- product_history_latest_asof_date: 2026-05-04
- fetch_status: PASS_APPEND_DONE
- append_candidate_rows: 10
- append_candidate_tickers: AAPL, MSFT, NFLX, NVDA, TSLA
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

- actual local parquet append executed
- no new SEC filing rows for rehearsal tickers
- product inference/thin upload left as dry-run boundary

## Failures

- none

## Artifacts

- metrics: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_metrics_20260515_20260515_015441.json
- stdout: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_stdout_20260515_20260515_015441.log
- stderr: C:\Users\user\lens\docs\daily_yfinance_append_gate_runs\daily_yfinance_append_gate_stderr_20260515_20260515_015441.log
