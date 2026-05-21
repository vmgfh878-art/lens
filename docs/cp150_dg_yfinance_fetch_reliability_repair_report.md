# CP150-DG yfinance fetch reliability repair + backfill ETA 보고서

## 1. 원인 한 줄 결론

yfinance 0.2.40의 `yf.download()` crumb/cookie 경로는 현재 429/JSONDecodeError로 막히지만, `trust_env=False`와 browser-like header를 쓰는 Yahoo chart 직접 fetch는 정상 JSON을 반환한다.

## 2. 진단 결과

환경:

- Python: 3.10.0
- yfinance: 0.2.40
- yfinance path: `C:\Users\user\lens\.venv\lib\site-packages\yfinance\__init__.py`
- requests: 2.32.3

분리 진단:

| 조건 | 결과 |
|---|---|
| `yf.download(AAPL)` | 0 rows, 내부 crumb 경로에서 429 후 JSONDecodeError |
| Yahoo chart `trust_env=True` | 로컬 차단 프록시 `127.0.0.1:9`로 빠져 ProxyError |
| Yahoo chart `trust_env=False`, browser header 없음 | 429 또는 JSON parse 실패 |
| Yahoo chart `trust_env=False`, browser-like header 있음 | 200 JSON, timestamp 정상 |

따라서 이번 장애는 단순히 “최신 거래일 없음”이 아니라 yfinance 라이브러리의 crumb/cookie 경로와 로컬 proxy/env 조합 문제다.

## 3. 고친 내용

- `YFinancePriceProvider`가 기본적으로 Yahoo chart direct fetch를 먼저 사용하도록 변경했다.
- 저장 source/provider는 계속 `yfinance`로 유지한다.
- fetch provenance는 `fetch_method=yahoo_chart`, `yahoo_chart_host=query1.finance.yahoo.com`로 metrics에 남긴다.
- yfinance 라이브러리 fallback은 남겼지만 기본 direct chart 우선이다.
- EODHD fallback은 사용하지 않는다.
- 일부 티커 실패 시 성공 티커만 append 가능한 상태를 추가했다.
  - dry-run 후보: `APPEND_READY_PARTIAL`
  - 실제 append 완료: `PARTIAL_APPEND_DONE`
- backfill state 파일을 추가했다.
  - `data/parquet/yfinance_backfill_state.json`
- 자동화 wrapper가 partial 상태와 성공/실패 티커를 보고서에 남기도록 보강했다.

## 4. 실제 append 결과

실행:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_local_daily_update.ps1 -Apply -MetricsPath docs\cp150_dg_yfinance_daily_append_apply_metrics.json -CurrentDate 2026-05-10 -LookbackDays 7
```

결과:

- 상태: `PASS_APPEND_DONE`
- 대상: AAPL, MSFT, NVDA, TSLA, NFLX
- append rows: 20
- append dates: 2026-05-05, 2026-05-06, 2026-05-07, 2026-05-08
- price rows: 284905 → 284925
- yfinance price latest: 2026-05-08
- yfinance 1D indicator latest: 2026-05-08
- yfinance 1W indicator latest: 2026-05-08
- 1D indicator refresh rows: 20
- 1W indicator refresh rows: 10
- duplicate 0
- indicator non-finite 0
- partial 1W row 0
- EODHD 500 parquet unchanged: true
- EODHD fallback 0
- Supabase bulk read/write 0

append 전 백업:

- `data/parquet/backups/price_data_yfinance_before_cp149_0_dg_20260510_133921.parquet`
- `data/parquet/backups/indicators_yfinance_1D_before_cp149_0_dg_20260510_133921.parquet`
- `data/parquet/backups/indicators_yfinance_1W_before_cp149_0_dg_20260510_133921.parquet`

## 5. 5/20/50 티커 측정

최근 10일 fetch, direct chart 기준:

| 샘플 | 성공 | 실패 | rows | 총 시간 | 평균 초/ticker | ticker/hour | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 0 | 35 | 1.789s | 0.358 | 10063.54 | 0 |
| 20 | 20 | 0 | 140 | 7.273s | 0.364 | 9899.98 | 0 |
| 50 | 50 | 0 | 350 | 18.663s | 0.373 | 9644.58 | 0 |

장기 fetch 샘플:

| 샘플 | 기간 | 성공 | 실패 | rows | 총 시간 | 평균 초/ticker |
|---:|---|---:|---:|---:|---:|---:|
| 5 | 2015년 이후 | 5 | 0 | 14275 | 3.064s | 0.613 |

## 6. Backfill ETA

상세 CSV:

- `docs/cp150_dg_yfinance_backfill_eta.csv`

요약:

| 시나리오 | best | base | worst |
|---|---:|---:|---:|
| 기존 100 ticker를 2026-05-08까지 daily append 복구 | 2분 | 4~6분 | 15~30분 |
| 100 ticker 전체 parquet를 2015년부터 재백필 | 3~5분 | 8~15분 | 30~60분 |
| 500 ticker 전체 yfinance parquet를 2015년부터 재백필 | 25~45분 | 90~180분 | 360~720분 |

500 ticker는 반드시 별도 parquet로 만들어야 한다.

- 기존 100 ticker 운영: `price_data_yfinance.parquet`
- 500 확장 후보: `price_data_yfinance_500.parquet`
- EODHD 500: `price_data_eodhd_500.parquet`

## 7. 자동화 확인

실제 append 후 자동화 dry-run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_yfinance_daily_append_gate_automation.ps1 -DryRun -RunDate 2026-05-10 -LookbackDays 7
```

결과:

- status: `PASS_WITH_NO_NEW_DAY`
- latest report: `docs/daily_yfinance_append_gate_latest.md`
- latest metrics: `docs/daily_yfinance_append_gate_metrics.json`
- history csv 갱신됨

append가 이미 2026-05-08까지 끝났기 때문에 no-new-day가 정상이다.

## 8. EODHD 해지 가능 여부

판정: **조건부 가능 후보로 상승**

이번 CP로 5티커 actual append와 1D/1W indicator refresh는 EODHD 없이 통과했다. 다만 해지 최종 판정 전에는 아래가 필요하다.

- 100 ticker 운영 parquet 전체를 2026-05-08까지 최신화.
- daily automation에서 최소 2~3회 연속 `PASS_APPEND_DONE` 또는 정상적인 `PASS_WITH_NO_NEW_DAY` 확인.
- 500 ticker yfinance parallel parquet는 EODHD 500과 섞지 않고 별도 생성.
- 500 ticker symbol 변환 실패 목록 확보.
- Yahoo direct chart rate limit이 장시간 backfill에서 재현되는지 확인.

## 9. 다음 자동화에서 기대할 상태

- 신규 완료 거래일이 없으면 `PASS_WITH_NO_NEW_DAY`
- 신규 완료 거래일이 있고 `-Apply`가 없으면 `BLOCKED_ACTUAL_APPEND_NOT_ENABLED`
- 신규 완료 거래일이 있고 `-Apply`가 있으면 `PASS_APPEND_DONE`
- 일부 티커만 성공하면 `PARTIAL_APPEND_DONE`
- Yahoo rate limit이면 `BLOCKED_YAHOO_429`
- 전체 fetch 실패면 `WARN_EXTERNAL_FETCH_BLOCKED`

## 10. 아직 돈 나가는 리스크

- EODHD 500 bootstrap 데이터는 아직 yfinance 500 parallel로 대체되지 않았다.
- 1D/1W 500 ticker full training 기준 데이터는 현재 EODHD 500이 더 완성돼 있다.
- Supabase raw price/indicator를 다시 대량 read/write하면 egress 비용 리스크가 재발한다.
- yfinance/Yahoo direct chart는 무료지만 SLA가 없고, 장기 500 ticker backfill에서 rate limit이 생길 수 있다.
- product latest upload와 scanner top-k 저장은 thin DB 원칙을 계속 지켜야 한다.

## 11. 검증

- `py_compile` 대상 Python 파일: PASS
- PowerShell parse check: PASS
- `backend.tests.test_market_data_providers`: PASS
- `backend.tests.test_collector_jobs`: PASS
- metrics JSON parse: PASS
- ETA CSV parse: PASS
- 모델 학습 없음
- inference 저장 없음
- product save-run 없음
- Supabase bulk read/write 없음
- EODHD fallback 없음
- 프론트 수정 없음

## 12. 산출물

- `docs/cp150_dg_yfinance_fetch_reliability_repair_report.md`
- `docs/cp150_dg_yfinance_fetch_reliability_repair_metrics.json`
- `docs/cp150_dg_yfinance_backfill_eta.csv`
- `scripts/cp150_yfinance_fetch_probe.py`
- `data/parquet/yfinance_backfill_state.json`
