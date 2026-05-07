# CP94-D yfinance 50티커 장기 history write 및 모델 smoke 재검증 보고서

## 1. Executive Summary

판정: **PASS_WITH_NOTES**

CP93의 최근 2년 50티커 yfinance write를 `2015-01-01` 이후 장기 history로 확장했다. 최신 거래일 기준은 `2026-05-01`로 고정했다. 전체 universe write, EODHD row 삭제/복구, save-run, product run 교체, live inference 연결, 1W/1M 재계산은 실행하지 않았다.

핵심 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 대상 티커 | 50 |
| yfinance price_data rows | 142,448 |
| price date range | 2015-01-02 ~ 2026-05-01 |
| duplicate ticker/date/source | 0 |
| adjusted OHLC violation, tolerance 1e-10 | 0 |
| raw OHLC violation | 0 |
| source/provider/policy 누락 | 0 |
| yfinance 1D indicators rows | 139,505 |
| atr_ratio coverage | 100.0% |
| feature MODEL_N_FEATURES | 36 |
| atr_ratio in MODEL_FEATURE_COLUMNS | false |
| feature/target nonfinite | 0 |
| seq_len 60 split gate | PASS |
| seq_len 252 split gate | PASS |
| CNN-LSTM band smoke | PASS, exit code 0 |
| PatchTST line smoke | PASS, exit code 0 |

결론: 50티커 장기 yfinance 데이터는 source-aware price/indicator/feature/split/smoke 경로를 통과했다. 다음 단계는 100티커 제한 write 또는 50티커 local daily sync 리허설까지는 가능하다. 전체 universe write와 live inference 연결은 아직 금지 상태로 두는 것이 맞다.

## 2. 대상과 사전 상태

CP93 50티커 목록을 유지했다. `CMCSA`는 제외했고 `HON`은 유지했다. `SPY`와 `QQQ` 포함 50티커의 `stock_info` 누락은 0이었다.

대상 티커:

```text
AAPL MSFT NVDA TSLA NFLX AMZN GOOGL META AMD AVGO SPY QQQ GOOG BRK-B JPM V MA UNH HD PG COST XOM JNJ WMT LLY ORCL CRM ADBE CSCO BAC PEP KO MRK ABBV TMO ACN MCD LIN DIS INTC QCOM TXN AMAT MU IBM GE CAT BA NKE HON
```

사전 상태:

| 점검 | 결과 |
|---|---:|
| stock_info 누락 | 0 |
| CP93 recent yfinance row 존재 | 50/50 |
| 최신 yfinance date | 2026-05-01 |
| 사전 duplicate ticker/date/source | 0 |

## 3. 장기 write 결과

첫 번째 CLI dry-run gate는 exit code 1로 중단됐다. 원인은 adjusted OHLC contract 실패가 아니라 장기 EODHD/Yahoo provider 정책 차이와 기존 비교 fetch의 1000 row cap이 섞인 비교 gate 한계였다. 실패 티커도 yfinance contract 자체는 통과했고, 이 상태에서 무리하게 dry-run 판정을 완화하지 않았다.

그 다음 공식 collector write 경로인 `backend.collector.jobs.sync_prices.run()`을 직접 호출했다. 이 경로는 provider abstraction, adjusted OHLC validation, `source='yfinance'`, `provider='yfinance'`, `on_conflict=(ticker,date,source)` 계약을 그대로 사용한다.

| 항목 | 결과 |
|---|---:|
| processed | 50 |
| failed | 0 |
| skipped | 0 |
| stored_rows | 142,448 |
| quota_hit | false |
| fallback 사용 | 없음 |

품질 gate에서 `AMD` 2개 row가 제외됐다.

| 티커 | 제외일 | 사유 |
|---|---|---|
| AMD | 2015-01-02 | invalid_volume |
| AMD | 2016-04-22 | extreme_jump |

AMD의 adjusted OHLC contract violation은 0이었다. 따라서 이는 provider 전환 실패가 아니라 price quality gate가 이상 row를 차단한 정상 동작으로 분류한다.

## 4. 데이터 계약 검증

| 항목 | 결과 |
|---|---:|
| rows | 142,448 |
| tickers_with_rows | 50 |
| date_min | 2015-01-02 |
| date_max | 2026-05-01 |
| duplicate_ticker_date_source | 0 |
| raw_ohlc_violation_count | 0 |
| adjusted_factor_violation_count | 0 |
| nonfinite_core_count | 0 |
| source_missing_count | 0 |
| provider_missing_count | 0 |
| provider_adjustment_policy_missing_count | 0 |
| updated_at_missing_count | 0 |
| volume_null_count | 0 |
| volume_negative_count | 0 |

Strict adjusted OHLC 비교에서는 16건이 잡혔지만, 최대 양수 gap이 `1.42e-14`인 double precision 반올림 수준이었다. tolerance `1e-10` 기준 violation은 0이다. 이 항목은 실제 high/low 정합성 위반이 아니라 부동소수점 artifact로 판단한다.

## 5. 1D indicators 재계산

`provider/source=yfinance`, `timeframe=1D`, 50티커만 대상으로 source-aware indicators를 재계산했다. EODHD indicator row는 삭제하지 않았다. 1W/1M은 이번 CP에서 재계산하지 않았다.

| 항목 | 결과 |
|---|---:|
| rows | 139,505 |
| tickers_with_rows | 50 |
| date_min | 2015-03-26 |
| date_max | 2026-05-01 |
| duplicate_ticker/timeframe/date/source | 0 |
| provider_missing_count | 0 |
| core_nonfinite_count | 0 |
| atr_ratio_non_null | 139,505 |
| atr_ratio_coverage | 100.0% |

Ratio sanity:

| 컬럼 | abs p99 | abs max |
|---|---:|---:|
| open_ratio | 0.050072 | 0.454199 |
| high_ratio | 0.073239 | 0.458015 |
| low_ratio | 0.070979 | 0.390408 |
| atr_ratio | 0.071331 | 0.274956 |

p99는 안정적이다. max 값은 장기 기간의 개별 급변 구간 영향으로 보이며, NaN/Inf나 adjusted OHLC 혼용 폭주 징후는 없었다.

## 6. CP93 2년 데이터 대비 차이

CP93 recent window는 각 티커 500 trading day 수준으로 존재했다. CP94는 같은 50티커를 2015년 이후 장기 history로 확장했다.

| 항목 | CP93 recent | CP94 long |
|---|---:|---:|
| price 기간 | 2024-05-03 ~ 2026-05-01 | 2015-01-02 ~ 2026-05-01 |
| 대상 티커 | 50 | 50 |
| source_data_hash | 2c075526 | bef59538 |
| feature index rows | 최근 구간 기준 | 139,505 |
| feature rows | 최근 구간 기준 | 139,483 |
| seq_len 252 가능성 | 기간 부족 가능 | 기본 split gate PASS |

`source_data_hash`가 `2c075526`에서 `bef59538`로 바뀌어, 장기 yfinance cache가 CP93 2년 cache와 분리됨을 확인했다. EODHD hash `3e90764c`와도 다르며, feature/index cache path도 분리되어 있다.

## 7. Feature Cache 검증

| 항목 | 결과 |
|---|---:|
| provider/source | yfinance/yfinance |
| feature_contract_version | v3_adjusted_ohlc |
| source_data_hash | bef59538 |
| feature_df_rows | 139,483 |
| index_rows | 139,505 |
| feature_ticker_count | 50 |
| model_n_features | 36 |
| atr_ratio_in_model_features | false |
| source_feature_nonfinite_count | 0 |
| model_feature_nonfinite_count_after_calendar | 0 |
| cache_paths_differ from EODHD | true |
| hash_differs_from_cp93_2y | true |

horizon 5 target sanity:

| 항목 | 값 |
|---|---:|
| nonfinite | 0 |
| p01 | -0.093058 |
| p50 | 0.002165 |
| p99 | 0.102115 |
| abs > 50% rate | 0.00001969 |

희소한 50% 초과 future return은 남아 있지만 비율이 매우 낮다. 다음 확대 단계에서는 예외 리스트와 split/dividend 주변 원인 분류를 계속 유지해야 한다.

## 8. Split Gate 결과

기본 `min_fold_samples=50` 기준으로 seq_len 60과 seq_len 252가 모두 통과했다.

| 설정 | train | val | test | usable rows | estimated usable samples | excluded |
|---|---:|---:|---:|---:|---:|---:|
| seq_len 60, h5 | 93,458 | 20,000 | 20,097 | 139,505 | 135,555 | 0 |
| seq_len 252, h5 | 86,752 | 18,566 | 18,637 | 139,505 | 125,955 | 0 |

CP93의 핵심 메모였던 “PatchTST seq_len 252는 최근 2년이면 부족할 수 있음”은 장기 history write 후 해소됐다.

## 9. 모델 Smoke 결과

학습 조건:

- W&B: `--no-wandb`, `disabled_by_cli`
- save-run: 사용하지 않음
- device: cpu
- compile: off
- epochs: 1
- product run 교체: 없음
- live inference 연결: 없음

### 9.1 Band Smoke

명령 요약:

```powershell
python -m ai.train --model cnn_lstm --timeframe 1D --horizon 5 --seq-len 60 --feature-set price_volatility_volume --q-low 0.15 --q-high 0.85 --lambda-band 2.0 --band-mode direct --checkpoint-selection band_gate --market-data-provider yfinance --tickers <CP94 50 tickers> --epochs 1 --batch-size 64 --no-wandb --no-compile --device cpu --num-workers 0 --amp-dtype off --fp32-modules lstm,heads --local-log-dir logs\cp94_yfinance_validation
```

| 항목 | 값 |
|---|---:|
| run_id | cnn_lstm-1D-33bb52f2c584 |
| exit code | 0 |
| status | completed |
| seq_len | 60 |
| feature_set | price_volatility_volume |
| n_features | 11 |
| source_data_hash | bef59538 |
| val_total_loss | 0.028770 |
| test_total_loss | 0.014294 |
| best band_gate_pass | true |
| test empirical_coverage | 0.727123 |
| test coverage_abs_error | 0.027123 |
| test lower_breach_rate | 0.142001 |
| test upper_breach_rate | 0.130875 |
| test band_width_ic | 0.306316 |
| test downside_width_ic | 0.019523 |
| elapsed seconds | 111.76 |

### 9.2 Line Smoke

명령 요약:

```powershell
python -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --patch-len 32 --patch-stride 16 --feature-set full_features --checkpoint-selection line_gate --market-data-provider yfinance --tickers <CP94 50 tickers> --epochs 1 --batch-size 64 --no-wandb --no-compile --device cpu --num-workers 0 --amp-dtype off --local-log-dir logs\cp94_yfinance_validation
```

| 항목 | 값 |
|---|---:|
| run_id | patchtst-1D-1a962f10c2da |
| exit code | 0 |
| status | completed |
| seq_len | 252 |
| feature_set | full_features |
| n_features | 36 |
| source_data_hash | bef59538 |
| val_total_loss | 0.054487 |
| test_total_loss | 0.062562 |
| best line_gate_pass | true |
| test ic_mean | 0.019153 |
| test long_short_spread | 0.002094 |
| test false_safe_tail_rate | 0.266942 |
| test severe_downside_recall | 0.737220 |
| elapsed seconds | 1114.53 |

성능 수치는 1 epoch smoke이므로 제품 run 교체 판단에 사용하지 않는다. 이번 CP의 판단 기준은 source-aware pipeline이 비지 않고, NaN/Inf 없이, 기본 split gate와 최소 모델 경로를 통과하는지였다.

## 10. 100티커 또는 전체 universe write 가능 여부

100티커 제한 write: **진행 가능, 단 PASS_WITH_NOTES**

조건:

- CP94와 같은 source-aware write, 1D indicator 재계산, feature cache 검증, band/line smoke를 같은 CP 안에서 반복한다.
- 장기 dry-run 비교 gate는 장기 provider 정책 차이와 REST 1000 row cap을 분리해서 개선한다.
- AMD처럼 quality gate에서 제외된 row는 ticker/date/사유를 예외 목록으로 남긴다.
- split/dividend 이벤트 구간은 provider policy 차이인지 오류인지 계속 분류한다.

전체 universe write: **아직 금지**

이유:

- 50티커 장기 데이터는 통과했지만 전체 universe의 ticker mapping, delisted/illiquid ticker, long history 누락, Yahoo coverage 실패 가능성이 아직 검증되지 않았다.
- full universe write는 데이터 row 수와 indicator upsert 비용이 커서, 실패 시 migration artifact가 크게 남는다.
- 먼저 100티커 제한 write와 local daily sync 리허설을 통과해야 한다.

## 11. Live Inference 연결 전 남은 조건

1. 100티커 장기 write + source-aware 1D indicators + feature cache + smoke 재검증.
2. dry-run 비교 gate 개선: baseline 1000 row cap 제거 또는 ticker/date 범위별 비교로 분리.
3. yfinance 실패/부분 실패 시 EODHD fallback 사용 count와 ticker/date 범위 기록.
4. local daily sync 리허설: yfinance price sync, 1D indicator recompute, data quality check까지 자동 순서 검증.
5. product run 선택 로직이 provider/source를 명확히 드러내는지 확인.
6. EODHD legacy row와 yfinance row가 API/feature/read 경로에서 섞이지 않는지 재확인.

## 12. 실행한 명령 목록

사전/후속 검증은 `logs/cp94_yfinance_validation/*.json`에 기록했다.

장기 write dry-run gate:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --start-date 2015-01-01 --end-date 2026-05-01 --compare-tickers <CP94 50 tickers> --tickers <CP94 50 tickers> --write --batch-limit 50 --sleep-seconds 0.1 --metrics-path logs\cp94_yfinance_validation\yfinance_50_long_write_metrics.json
```

장기 write 공식 경로:

```python
from backend.collector.jobs.sync_prices import run
run(
    TICKERS,
    default_start="2015-01-01",
    lookback_days=7,
    repair_mode=False,
    provider="yfinance",
    fallback_provider="eodhd",
    batch_limit=50,
    sleep_seconds=0.1,
    allow_yahoo_fallback=False,
    require_fundamentals=False,
    force_start_date=True,
)
```

1D indicators 재계산:

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --provider yfinance --timeframes 1D --tickers <CP94 50 tickers> --lookback-days 14 --force-full-backfill
```

Feature/split 검증:

```text
fetch_feature_index_frame, fetch_training_frames, prepare_dataset_splits 기반 read 검증
```

Band smoke와 line smoke 명령은 9장에 기록했다.

## 13. 금지 사항 준수 확인

| 금지 항목 | 준수 |
|---|---|
| 전체 universe write 금지 | 준수 |
| EODHD row 삭제/복구 금지 | 준수 |
| full retraining 금지 | 준수 |
| save-run 금지 | 준수 |
| product run 교체 금지 | 준수 |
| live inference 금지 | 준수 |
| 1W/1M 전체 재계산 금지 | 준수 |

## 14. 산출물

- `docs/cp94_yfinance_50ticker_long_history_validation_report.md`
- `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json`
- `logs/cp94_yfinance_validation/`
