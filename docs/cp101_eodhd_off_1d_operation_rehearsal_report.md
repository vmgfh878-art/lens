# CP101-D EODHD off 1D 운영 리허설

생성일: 2026-05-04

## 1. Executive Summary

판정: **WARN**

EODHD API key와 fallback을 비운 상태에서 `yfinance + local parquet + Supabase thin DB` 기반 1D 제품 루프를 리허설했다. 핵심 운영 루프는 통과했다.

WARN 사유는 실패가 아니라 최신성 조건이다. yfinance 5티커 최신 조회는 성공했지만, 조회된 최신 거래일이 local snapshot 최신일과 같은 `2026-05-01`이어서 local parquet append는 0건이었다. 즉 오늘 기준 신규 거래일 데이터가 아직 없어서 local update는 실질 dry-run으로 끝났다.

## 2. EODHD off 환경 확인

| 항목 | 값 | 결과 |
|---|---|---|
| `MARKET_DATA_PROVIDER` | `yfinance` | PASS |
| `MARKET_DATA_FALLBACK_PROVIDER` | 빈 값 | PASS |
| `EODHD_API_KEY` | 없음 | PASS |
| `LENS_DATA_BACKEND` | `local` | PASS |
| `LENS_REQUIRE_LOCAL_SNAPSHOTS` | `1` | PASS |
| `LENS_LOCAL_SNAPSHOT_DIR` | `C:\Users\user\lens\data\parquet` | PASS |
| collector settings fallback | `null` | PASS |

추가 보강:

`backend/collector/config.py`에서 `MARKET_DATA_FALLBACK_PROVIDER=`처럼 env가 명시적으로 빈 값이면 fallback을 `None`으로 해석하게 수정했다. env 자체가 없을 때만 기존 호환성대로 yfinance -> eodhd fallback default가 유지된다.

## 3. Local Snapshot Freshness

| snapshot | row 수 | ticker 수 | 날짜 범위 | source |
|---|---:|---:|---|---|
| `price_data_yfinance.parquet` | 284,900 | 100 | 2015-01-02 ~ 2026-05-01 | yfinance |
| `indicators_yfinance_1D.parquet` | 279,000 | 100 | 2015-03-30 ~ 2026-05-01 | yfinance |

`source_data_hash`: `3e4ee198`

Feature/split gate:

| 항목 | 결과 |
|---|---|
| `MODEL_N_FEATURES` | 36 |
| `FEATURE_CONTRACT_VERSION` | `v3_adjusted_ohlc` |
| `atr_ratio` 모델 feature 포함 여부 | false |
| feature NaN/Inf | 0 |
| target NaN/Inf | 0 |
| split samples | train 9,345 / val 2,000 / test 2,010 |

## 4. yfinance Latest Update

대상: AAPL, MSFT, NVDA, TSLA, NFLX

요청:

| 항목 | 값 |
|---|---|
| provider | yfinance |
| fallback provider | 없음 |
| start_date | 2026-04-24 |
| end_date | 2026-05-04 |

결과:

| 항목 | 값 |
|---|---:|
| successful fetch count | 5 |
| failed/empty fetch count | 0 |
| fallback_used | false |
| adjusted OHLC violations | 0 |
| new rows after snapshot latest | 0 |
| local price rows written | 0 |

각 ticker는 2026-04-24부터 2026-05-01까지 6개 row를 반환했고, local snapshot 최신일 `2026-05-01` 이후 신규 row는 없었다.

## 5. 1D Product Inference

사용 run:

| layer | run_id | checkpoint | 결과 |
|---|---|---|---|
| line | `patchtst-1D-efad3c29d803` | PatchTST 1D | PASS |
| band | `cnn_lstm-1D-d0c780dee5e8` | CNN-LSTM 1D | PASS |

Dry-run 결과:

| 항목 | line | band |
|---|---:|---:|
| prediction rows | 5 | 5 |
| evaluation rows | 5 | 5 |
| horizon | 5 | 5 |
| forecast date length | 5 | 5 |
| line/conservative/lower/upper length | 5 | 5 |
| lower <= upper | true | true |
| asof_date | 2026-05-01 | 2026-05-01 |

Composite 저장은 하지 않았다.

## 6. Supabase Thin Upload

저장 대상은 5티커 latest-only line/band 결과로 제한했다.

| 항목 | 값 |
|---|---:|
| target prediction rows | 10 |
| target evaluation rows | 10 |
| composite rows | 0 |
| prediction count delta | 0 |
| evaluation count delta | 0 |

이미 같은 `asof_date=2026-05-01` latest row가 있었기 때문에 upsert 후 row count 증가는 0이었다. 즉 prediction history 대량 증가는 발생하지 않았다.

## 7. API 확인

| 조회 | 결과 |
|---|---|
| AAPL 1D line latest | PASS |
| AAPL 1D band latest | PASS |
| line run_id | `patchtst-1D-efad3c29d803` |
| band run_id | `cnn_lstm-1D-d0c780dee5e8` |
| 1W/1M 정책 | 기존 준비 중/price-only 정책 유지 |

## 8. 금지 작업 확인

| 금지 항목 | 발생 여부 |
|---|---|
| EODHD API 호출 | false |
| EODHD fallback | false |
| Supabase `price_data`/`indicators` 대량 read | false |
| Supabase `price_data`/`indicators` 전체 write | false |
| indicators DB recompute | false |
| full model training | false |
| 1W/1M inference | false |
| composite 저장 | false |
| EODHD row 삭제 | false |
| 프론트 UI 수정 | false |

`fetch_all_rows("price_data", limit=None)`와 `fetch_all_rows("indicators", limit=None)`는 local mode bulk read guard로 차단됨을 확인했다.

## 9. 최종 판단

1D 제품 루프는 EODHD 없이도 동작한다.

다만 이번 실행 시점에는 yfinance가 local snapshot보다 새로운 거래일 row를 주지 않았으므로, “신규 장 종료 후 append + indicator refresh”까지는 실제 append로 검증되지 않았다. EODHD 해지 후보로 볼 수 있지만, 다음 거래일 장 종료 이후 한 번 더 `new_rows_after_snapshot_latest > 0`인 상태에서 local parquet append를 확인하면 더 깔끔하다.

## 10. 실행 명령

```powershell
python scripts\cp101_eodhd_off_1d_operation_rehearsal.py
```

첫 실행에서는 샌드박스 네트워크 프록시 문제로 yfinance가 빈 응답을 반환해, 외부 네트워크 권한으로 동일 명령을 재실행했다. 최종 metrics는 재실행 결과를 기준으로 저장했다.

## 11. 산출물

```text
docs/cp101_eodhd_off_1d_operation_rehearsal_report.md
docs/cp101_eodhd_off_1d_operation_rehearsal_metrics.json
logs/cp101_eodhd_off_1d_operation_rehearsal/cp101_eodhd_off_1d_operation_rehearsal_metrics.json
logs/cp101_eodhd_off_1d_operation_rehearsal/dry_run_predictions.json
```
