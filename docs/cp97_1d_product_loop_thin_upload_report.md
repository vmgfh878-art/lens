# CP97-D: local parquet + yfinance 기반 1D 제품 루프 thin upload 검증

작성일: 2026-05-04  
판정: WARN_BLOCKED  
범위: 1D, horizon 5, provider/source yfinance, line `patchtst-1D-efad3c29d803`, band `cnn_lstm-1D-d0c780dee5e8`

## 1. Executive Summary

CP97의 제품 루프는 아직 PASS로 닫을 수 없다.

이유는 모델이나 checkpoint가 아니라, 사전 조건인 local parquet snapshot이 지정 경로에 없기 때문이다. 지시된 `LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet`가 존재하지 않았고, 필수 파일인 `stock_info.parquet`, `price_data_yfinance.parquet`, `indicators_yfinance_1D.parquet`도 없었다.

따라서 최신 asof 기준 inference, latest-only thin upload, API 표시 확인은 실행하지 않았다. 이 중단은 의도된 안전 중단이다. Supabase `price_data`/`indicators` 대량 read 없이 이 단계를 우회하면 CP97의 핵심 계약을 깨게 된다.

좋은 점은 다음이다.

| 항목 | 결과 |
|---|---|
| line checkpoint | 존재, CPU load 가능 |
| band checkpoint | 존재, CPU load 가능 |
| `MODEL_N_FEATURES` | 36 유지 |
| feature contract | `v3_adjusted_ohlc` |
| `atr_ratio` 모델 feature 포함 여부 | 미포함 |
| band feature set | `price_volatility_volume`, 11개 |
| Supabase 대량 read guard | `price_data`, `indicators` 모두 차단 확인 |
| 금지 작업 위반 | 없음 |

## 2. 사전 조건 확인

요구 환경:

```text
LENS_DATA_BACKEND=local
LENS_REQUIRE_LOCAL_SNAPSHOTS=1
LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet
```

snapshot 확인 결과:

| 파일 | 상태 |
|---|---|
| `C:\Users\user\lens\data\parquet` | 없음 |
| `stock_info.parquet` | 없음 |
| `price_data_yfinance.parquet` | 없음 |
| `indicators_yfinance_1D.parquet` | 없음 |

참고로 `backend\data\parquet`에는 placeholder parquet가 있었지만 `stock_info.parquet`와 `price_data.parquet` 모두 row 0, column 0이었다. 제품 inference용 snapshot으로 사용할 수 없다.

`logs\cp87_yfinance_validation`에는 shadow parquet가 있지만, CP97의 필수 source-aware snapshot 세트가 아니다. 특히 `indicators_yfinance_1D.parquet`를 대체하지 못한다.

## 3. Render cron 상태

`render.yaml`에는 다음 cron이 남아 있다.

```text
lens-daily-market-sync
python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60
```

`docs/supabase_egress_incident_report.md`와 `docs/supabase_cost_guard_checklist.md`에는 Render Dashboard에서 이 cron을 Suspend/Pause해야 한다는 절차가 기록되어 있다.

다만 실제 Render Dashboard 상태는 로컬 파일만으로 확인할 수 없다. 따라서 이번 보고서에서는 “문서상 suspend 절차 확인, 실제 suspend 상태 미확인”으로 기록한다.

## 4. 데이터 gate 결과

| 항목 | 결과 | 비고 |
|---|---|---|
| provider/source | 미실행 | snapshot 부재 |
| source_data_hash | 미확인 | snapshot meta 필요 |
| feature/target NaN/Inf | 미확인 | feature build 미실행 |
| partial period | 1D라 구조상 해당 없음 | 데이터 확인은 미실행 |
| `MODEL_N_FEATURES` | 36 | 코드 기준 확인 |
| `FEATURE_CONTRACT_VERSION` | `v3_adjusted_ohlc` | 코드 기준 확인 |
| `atr_ratio` 모델 feature 미포함 | 확인 | 코드 기준 확인 |
| band feature set | 11개 | `price_volatility_volume` |
| local mode 대량 read guard | 확인 | unlimited `price_data`/`indicators` read 차단 |

대량 read guard는 local required 모드에서 직접 확인했다.

```text
price_data: RuntimeError로 Supabase REST 대량 조회 차단
indicators: RuntimeError로 Supabase REST 대량 조회 차단
```

## 5. checkpoint 확인

| layer | run_id | checkpoint | 결과 |
|---|---|---|---|
| line | `patchtst-1D-efad3c29d803` | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-efad3c29d803.pt` | 존재, CPU load 가능 |
| band | `cnn_lstm-1D-d0c780dee5e8` | `ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt` | 존재, CPU load 가능 |

checkpoint config 요약:

| layer | model | timeframe | horizon | seq_len | feature_set | feature columns |
|---|---|---|---|---|---|---|
| line | patchtst | 1D | 5 | 252 | full_features | 36 |
| band | cnn_lstm | 1D | 5 | 60 | price_volatility_volume | 11 |

주의할 점: checkpoint config 자체에는 `source_data_hash`와 `market_data_provider`가 null이었다. CP75 보고서에는 line 후보 source hash `3ac43945`와 feature version `v3_adjusted_ohlc`가 기록되어 있으나, CP97 실행 시에는 local snapshot fingerprint로 다시 확인해야 한다.

## 6. inference dry-run 결과

실행하지 않았다.

중단 이유:

| 항목 | 판단 |
|---|---|
| local snapshot 없음 | 사전 조건 실패 |
| Supabase 대량 read 우회 가능성 | 우회 금지 |
| source_data_hash 확인 | 불가 |
| 최신 asof_date 샘플 구성 | 불가 |
| forecast_dates h1~h5 확인 | 불가 |
| line/conservative/lower/upper 길이 확인 | 불가 |
| lower <= upper 확인 | 불가 |

이 단계에서 inference를 강행하면 local parquet 기반 제품 루프 검증이 아니라 Supabase 또는 stale cache 의존 검증이 될 수 있다.

## 7. thin upload 계약 판단

이번 CP에서는 저장하지 않았다.

다만 계약상 판단은 다음과 같다.

| 항목 | 판단 |
|---|---|
| 기존 `predictions` upsert key | latest-only record list를 넣는다면 사용 가능 |
| `ai.inference --save` | CP97 저장에는 부적합 |
| 이유 | 현재 CLI save는 split 전체 prediction/evaluation을 저장할 수 있음 |
| CP97 저장 방식 | line 5행 + band 5행 latest-only, evaluation도 각각 5행 |
| composite | 저장 금지 |
| history bulk save | 금지 |

예상 latest-only 저장량:

| 테이블 | line | band | 합계 |
|---|---:|---:|---:|
| `predictions` | 5 | 5 | 10 |
| `prediction_evaluations` | 5 | 5 | 10 |

CP72 보고서에는 과거 band 후보 저장에서 `predictions`와 `prediction_evaluations`가 각각 185,683건 저장됐다고 기록되어 있다. CP97에서는 그 방식이 아니라 latest-only thin upload로 제한해야 한다.

## 8. API와 제품 표시 확인

실행하지 않았다.

이유:

| 항목 | 결과 |
|---|---|
| latest-only upload | 미실행 |
| AAPL 1D latest line 조회 | 미실행 |
| AAPL 1D latest band 조회 | 미실행 |
| 프론트 UI 수정 | 없음 |
| 1W/1M 정책 | 이번 CP에서 건드리지 않음 |

API 확인은 snapshot 준비 후 5티커 저장이 끝난 다음에 실행해야 의미가 있다.

## 9. 검증

| 검증 | 결과 |
|---|---|
| `py_compile` | PASS |
| 관련 unittest | NOT_RUN |
| unittest 미실행 사유 | 시스템 Python과 `.venv` 모두 `pytest` 모듈 없음 |
| metrics JSON parse | 별도 확인 필요 |
| Supabase price_data/indicators 대량 read | 발생하지 않음 |
| Supabase write | 발생하지 않음 |

실행한 컴파일:

```text
python -m py_compile ai\inference.py ai\preprocessing.py ai\storage.py backend\collector\repositories\local_snapshots.py backend\collector\repositories\base.py backend\app\repositories\market_repo.py backend\app\repositories\prediction_repo.py
```

## 10. 판정

최종 판정: WARN_BLOCKED

PASS가 아닌 이유:

| 조건 | 상태 |
|---|---|
| local yfinance snapshot 기반 inference 성공 | 미실행 |
| 5티커 latest-only thin upload 성공 | 미실행 |
| API에서 1D line/band 조회 성공 | 미실행 |
| 대량 DB read/write 없음 | 충족 |
| composite 미사용 | 충족 |

이건 모델 실패가 아니라 입력 snapshot 사전 조건 실패다. CP97을 다시 실행하려면 먼저 아래 파일이 실제 row를 가진 상태로 준비돼야 한다.

```text
C:\Users\user\lens\data\parquet\stock_info.parquet
C:\Users\user\lens\data\parquet\price_data_yfinance.parquet
C:\Users\user\lens\data\parquet\indicators_yfinance_1D.parquet
```

## 11. 다음 실행 조건

1. 지정 snapshot 경로를 만든다.
2. 세 parquet 파일의 row count, ticker count, date_min/date_max, source coverage를 확인한다.
3. `LENS_REQUIRE_LOCAL_SNAPSHOTS=1` 상태에서 5티커 inference dry-run을 먼저 실행한다.
4. line과 band 각각 latest asof_date만 추출한다.
5. `predictions`와 `prediction_evaluations`에 총 20행 수준으로 제한 저장한다.
6. AAPL 1D API 조회로 line/band layer가 composite 없이 내려오는지 확인한다.

## 12. 실행한 명령 목록

```text
Get-ChildItem data\parquet
Get-ChildItem backend\data\parquet
Select-String render.yaml, docs\supabase_egress_incident_report.md, docs\supabase_cost_guard_checklist.md
Get-ChildItem ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-efad3c29d803.pt, ai\artifacts\checkpoints\cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt
python -c "read backend\data\parquet placeholder parquet row counts"
Select-String ai\preprocessing.py
python -c "MODEL_N_FEATURES, FEATURE_CONTRACT_VERSION, atr_ratio membership"
python -c "load product checkpoints on CPU"
python -c "local mode unlimited price_data read guard"
python -c "local mode unlimited indicators read guard"
python -m py_compile ...
python -m pytest ...  # pytest 모듈 없음으로 미실행
```

## 13. 금지 작업 확인

| 금지 항목 | 위반 여부 |
|---|---|
| 전체 yfinance write | 없음 |
| 전체 indicators recompute | 없음 |
| full model training | 없음 |
| 1W/1M 처리 | 없음 |
| prediction history 대량 저장 | 없음 |
| composite 저장 | 없음 |
| EODHD row 삭제/복구 | 없음 |
| Supabase price_data/indicators 대량 read | 없음 |
| 프론트 UI 수정 | 없음 |
