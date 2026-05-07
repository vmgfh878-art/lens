# CP128-DG 1D Product Rolling Prediction History Replay

생성일: 2026-05-06

## 1. Executive Summary

최종 판정: PASS

제품용 1D rolling prediction history parquet를 생성했고 CP127 계약을 통과했다.

이번 CP는 기존 `predictions/history`나 test split bulk row를 사용하지 않고, 제품 checkpoint와 local yfinance parquet만으로 1D 제품용 rolling prediction history를 생성했다. 생성 결과는 h5 scalar display point만 포함하며, latest-only product row나 미래 forecast row를 섞지 않았다.

## 2. 범위

| 항목 | 값 |
|---|---|
| timeframe | 1D |
| line run | patchtst-1D-efad3c29d803 |
| band run | cnn_lstm-1D-d0c780dee5e8 |
| tickers | AAPL, MSFT, NVDA, TSLA, NFLX |
| source/provider | yfinance local parquet |
| display_horizon | 5 |
| asof_start | 2025-05-05 |
| asof_end | 2026-05-04 |

## 3. 생성 파일

| 파일 | rows/bytes |
|---|---:|
| `data\parquet\product_prediction_history_1D.parquet` | 2510 rows / 40394 bytes |
| `data\parquet\product_prediction_history_1D.manifest.json` | 1100 bytes |

## 4. Role별 Inference 진단

| role | run_id | model | seq_len | samples | output rows | ticker count | date min | date max | non-finite |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| line | patchtst-1D-efad3c29d803 | patchtst | 252 | 1255 | 1255 | 5 | 2025-05-05 | 2026-05-04 | 0 |
| band | cnn_lstm-1D-d0c780dee5e8 | cnn_lstm | 60 | 1255 | 1255 | 5 | 2025-05-05 | 2026-05-04 | 0 |

## 5. Contract 검증

| 검증 | 결과 |
|---|---|
| required columns 누락 | [] |
| duplicate ticker/role/asof/source | 0 |
| asof_date 오름차순 | True |
| display_horizon unique | [5] |
| source values | ['rolling_replay'] |
| line run ids | ['patchtst-1D-efad3c29d803'] |
| band run ids | ['cnn_lstm-1D-d0c780dee5e8'] |
| line value non-finite | 0 |
| band lower non-finite | 0 |
| band upper non-finite | 0 |
| lower > upper violations | 0 |
| forecast array columns 포함 | False |
| latest_live source 포함 | False |
| test split source 포함 | False |

## 6. AAPL/MSFT/NVDA 연속성

| ticker | line rows | band rows | latest asof | max gap days |
|---|---:|---:|---|---:|
| AAPL | 251 | 251 | 2026-05-04 | 4 |
| MSFT | 251 | 251 | 2026-05-04 | 4 |
| NVDA | 251 | 251 | 2026-05-04 | 4 |

## 7. Manifest

주요 manifest:
- feature_version: `v3_adjusted_ohlc`
- source_data_hash: `f1a00036`
- price_snapshot_hash: `6f6527c791a2db96...`
- indicator_snapshot_hash: `ffce024591d57c5a...`
- row_count: `2510`
- ticker_count: `5`

## 8. 금지 작업 확인

| 항목 | 발생 |
|---|---:|
| DB write | False |
| Supabase upload | False |
| 기존 predictions 수정/삭제 | False |
| 모델 학습 | False |
| W&B | False |
| composite | False |
| 프론트 수정 | False |
| fake data | False |

## 9. 다음 단계

이 parquet는 프론트 연결 CP로 넘길 수 있다. 다음 CP에서는 backend API가 `product_prediction_history_1D.parquet`를 ticker 단위로 읽어 `/product-predictions/history` 형태로 제공하고, 프론트는 기존 `/predictions/history` 대신 신규 제품 history endpoint를 사용하면 된다.

## 10. 실행 명령

```powershell
.\.venv\Scripts\python.exe scripts\cp128_product_rolling_prediction_history_replay.py
```
