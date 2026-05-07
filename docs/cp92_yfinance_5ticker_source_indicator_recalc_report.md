# CP92-D yfinance 5티커 source-aware indicator 재계산 및 EODHD 덮임 처리 보고서

## 1. Executive Summary

CP92에서는 CP89 old unique 계약 시점에 발생한 5티커 최근 EODHD row 덮임 상태를 진단하고, 50티커 write 확대 전 yfinance source 기준 indicator/feature/smoke 경로가 실제로 비지 않고 도는지 확인했다.

결론은 **조건부 PASS**다. AAPL, MSFT, NVDA, TSLA, NFLX는 `price_data`에서 2015-01-02부터 2024-05-02까지 legacy null source, 2024-05-03부터 2026-05-01까지 `source='yfinance'`로 끊겨 있다. 같은 ticker/date에 EODHD와 yfinance가 병렬 존재하는 날짜는 0개였고, 이는 CP89 당시 `(ticker, date)` old unique로 최근 EODHD row가 yfinance row로 대체된 migration artifact로 보는 것이 맞다.

이번 CP에서는 EODHD 복구를 실행하지 않았다. 대신 이 상태를 known migration artifact로 기록하고, 필요 시 제한 복구 절차만 문서화했다. yfinance 5티커 1D indicator는 source-aware 경로로 재계산했고, `indicators.source='yfinance'` 2,205 rows가 생성됐다. duplicate는 0건, `atr_ratio` coverage는 100%, `open_ratio/high_ratio/low_ratio` p99도 정상 범위였다.

feature cache는 provider/source/hash/manifest 기준으로 EODHD 후보 path와 분리됐다. `MODEL_N_FEATURES=36`과 `atr_ratio` 모델 feature 미포함 계약도 유지됐다. CNN-LSTM band smoke는 1 epoch, W&B off, `save_run=False`로 실행했고 DB 저장 없이 학습 루프와 metrics 생성까지 통과했다. 단 `band_gate` 품질 게이트는 실패했으므로 성능 개선 근거가 아니라 source-aware pipeline 검증으로만 해석한다.

## 2. 현재 source 상태 진단

대상 티커는 AAPL, MSFT, NVDA, TSLA, NFLX다.

| ticker | source | rows | date_min | date_max | provider 기록 | policy 기록 | updated_at 기록 |
|---|---|---:|---|---|---:|---:|---:|
| AAPL | legacy null | 2,349 | 2015-01-02 | 2024-05-02 | 0 | 0 | 2,349 |
| AAPL | yfinance | 500 | 2024-05-03 | 2026-05-01 | 500 | 500 | 500 |
| MSFT | legacy null | 2,349 | 2015-01-02 | 2024-05-02 | 0 | 0 | 2,349 |
| MSFT | yfinance | 500 | 2024-05-03 | 2026-05-01 | 500 | 500 | 500 |
| NVDA | legacy null | 2,349 | 2015-01-02 | 2024-05-02 | 0 | 0 | 2,349 |
| NVDA | yfinance | 500 | 2024-05-03 | 2026-05-01 | 500 | 500 | 500 |
| TSLA | legacy null | 2,349 | 2015-01-02 | 2024-05-02 | 0 | 0 | 2,349 |
| TSLA | yfinance | 500 | 2024-05-03 | 2026-05-01 | 500 | 500 | 500 |
| NFLX | legacy null | 2,349 | 2015-01-02 | 2024-05-02 | 0 | 0 | 2,349 |
| NFLX | yfinance | 500 | 2024-05-03 | 2026-05-01 | 500 | 500 | 500 |

CP89 기간 overlap 진단:

| 항목 | 값 |
|---|---:|
| ticker별 yfinance 날짜 수 | 500 |
| ticker별 yfinance window 내 병렬 EODHD 날짜 수 | 0 |
| ticker별 yfinance window 내 EODHD 누락 날짜 수 | 500 |
| yfinance date_min | 2024-05-03 |
| yfinance date_max | 2026-05-01 |
| 병렬 price sample count | 0 |

해석: CP91 이후에는 병렬 저장이 가능하지만, CP89 write 당시에는 old unique 때문에 2024-05-03 이후 500거래일의 EODHD price row가 yfinance row로 대체된 상태다. 이것은 현재 yfinance 검증에는 치명적이지 않지만, EODHD 대비 최근 baseline 비교에는 공백을 만든다.

## 3. EODHD 덮임 처리 결정

결정: **즉시 복구하지 않는다.**

이유:

- 사용자 지시에서 이번 CP의 실제 EODHD 재수집은 금지됐다.
- yfinance source-aware indicator/feature/smoke 검증은 EODHD 최근 row 복구 없이도 진행 가능하다.
- 현재 목적은 50티커 write 확대 전 source-aware pipeline이 섞이지 않는지 확인하는 것이다.
- EODHD 복구는 API 비용과 운영 DB write를 동반하므로 별도 승인 단위로 분리하는 것이 맞다.

따라서 2024-05-03부터 2026-05-01까지 5티커 EODHD 공백은 **known migration artifact**로 기록한다.

필요 시 복구 절차:

1. 사용자 승인 후 5티커만 대상으로 EODHD 가격을 재수집한다.
2. 기간은 2024-05-03부터 2026-05-01까지로 제한한다.
3. 저장 source는 `eodhd`로 명시한다.
4. fallback이 yfinance로 섞이지 않게 EODHD 전용 provider로 실행한다.
5. 복구 후 `(ticker, date, source)` 병렬 row 수를 확인한다.
6. 필요하면 `provider=eodhd`, `source=eodhd` 기준으로 5티커 1D indicators를 별도 재계산한다.
7. yfinance/EODHD adjusted close와 ratio diff를 다시 비교한다.

이번 CP에서는 위 절차를 실행하지 않았다.

## 4. yfinance 5티커 1D indicator 재계산

실행 명령:

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --provider yfinance --timeframes 1D --tickers AAPL MSFT NVDA TSLA NFLX --lookback-days 14 --force-full-backfill
```

결과:

| 항목 | 값 |
|---|---:|
| 저장 row 수 | 2,205 |
| provider | yfinance |
| source | yfinance |
| timeframe | 1D |
| ticker 수 | 5 |
| duplicate `(ticker,timeframe,date,source)` | 0 |
| EODHD indicator 삭제 | 0 |
| `atr_ratio` non-null rows | 2,205 |
| `atr_ratio` coverage | 100% |

티커별 yfinance indicator 결과:

| ticker | rows | date_min | date_max | atr_ratio non-null |
|---|---:|---|---|---:|
| AAPL | 441 | 2024-07-30 | 2026-05-01 | 441 |
| MSFT | 441 | 2024-07-30 | 2026-05-01 | 441 |
| NVDA | 441 | 2024-07-30 | 2026-05-01 | 441 |
| TSLA | 441 | 2024-07-30 | 2026-05-01 | 441 |
| NFLX | 441 | 2024-07-30 | 2026-05-01 | 441 |

`date_min`이 2024-05-03이 아니라 2024-07-30인 이유는 indicator 계산이 이동평균, RSI, MACD, Bollinger, ATR 등 lookback을 요구하기 때문이다. price row는 ticker별 500개지만, feature-ready indicator row는 ticker별 441개로 줄어드는 것이 정상이다.

## 5. Indicator sanity

전체 yfinance 1D indicator sanity:

| 항목 | 값 |
|---|---:|
| core null rows | 0 |
| open_ratio abs p99 | 0.070883197962363 |
| high_ratio abs p99 | 0.0867406225931958 |
| low_ratio abs p99 | 0.0961268783561834 |
| open_ratio abs max | 0.147583004576747 |
| high_ratio abs max | 0.238123148316402 |
| low_ratio abs max | 0.181741726308069 |
| atr_ratio p99 | 0.0854327317385009 |
| atr_ratio max | 0.109748823563649 |

티커별 ratio max:

| ticker | open max | high max | low max |
|---|---:|---:|---:|
| AAPL | 0.0944691871275496 | 0.163496278557497 | 0.108523569301832 |
| MSFT | 0.0906995984891306 | 0.109064786403388 | 0.125843492981292 |
| NVDA | 0.14179159728928 | 0.195223271880904 | 0.181741726308069 |
| TSLA | 0.145237536464364 | 0.238123148316402 | 0.177202224065131 |
| NFLX | 0.147583004576747 | 0.148698358724046 | 0.117728938987885 |

판정: CP29 이전처럼 raw/adjusted 혼용으로 ratio가 폭주하는 패턴은 보이지 않았다. TSLA/NVDA의 intraday ratio max는 상대적으로 크지만, 단일 종목 변동성 범위로 해석 가능하며 `NaN/Inf` 또는 극단 폭주로 보이지 않는다.

## 6. Feature cache 검증

검증 결과:

| 항목 | 값 |
|---|---|
| market_data_provider | yfinance |
| source | yfinance |
| feature_version | v3_adjusted_ohlc |
| MODEL_N_FEATURES | 36 |
| atr_ratio in MODEL_FEATURE_COLUMNS | false |
| feature rows | 2,205 |
| feature index rows | 2,205 |
| price rows | 2,500 |
| seq_len 60, h5 dataset samples | 1,885 |
| dataset shape | `[1885, 60, 36]` |
| feature nonfinite count | 0 |
| line target nonfinite count | 0 |
| band target nonfinite count | 0 |
| raw target nonfinite count | 0 |

cache/hash 분리:

| 항목 | 값 |
|---|---|
| yfinance source_data_hash | `0b4f550e` |
| EODHD source_data_hash 후보 | `d5635005` |
| hash 분리 | true |
| cache path 분리 | true |
| yfinance feature cache | `ai/cache/features_1D_5e37acb6e673_0b4f550e.pt` |
| yfinance index cache | `ai/cache/feature_index_1D_e384547f03a3_0b4f550e.pt` |
| EODHD feature cache 후보 | `ai/cache/features_1D_2da8e89c0559_d5635005.pt` |
| EODHD index cache 후보 | `ai/cache/feature_index_1D_43d9114100db_d5635005.pt` |

manifest 확인:

| 항목 | 값 |
|---|---|
| manifest provider | yfinance |
| manifest source | yfinance |
| provider_adjustment_policy | `yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc` |
| manifest date_min | 2024-05-03 |
| manifest date_max | 2026-05-01 |
| manifest ticker_count | 5 |

판정: yfinance와 EODHD는 provider/source/hash/path/manifest 기준으로 분리된다. `atr_ratio`는 indicator에는 존재하지만 모델 feature 36개에는 아직 포함되지 않았다.

## 7. Split 진단

CNN-LSTM band smoke용 `seq_len=60`, `horizon=5` 진단:

| 항목 | 값 |
|---|---:|
| eligible ticker count | 5 |
| excluded reasons | 0 |
| usable row count | 2,205 |
| estimated usable sample count | 1,810 |
| train samples | 1,125 |
| val samples | 240 |
| test samples | 245 |
| min_fold_samples override | 1 |

5티커 smoke 목적에서는 split이 비지 않는다. 다만 CLI 기본 `min_fold_samples=50`은 5티커 제한 검증에서는 과하게 보수적이므로, 이번 smoke는 사전 생성 split을 넘겨 실행했다.

PatchTST line용 `seq_len=252`, `horizon=5` 진단:

| 항목 | 값 |
|---|---:|
| eligible ticker count | 5 |
| excluded reasons | 0 |
| usable row count | 2,205 |
| estimated usable sample count | 850 |
| ticker별 sample count | 170 |
| gap | 20 |

PatchTST는 split 자체는 가능하지만 5티커/최근 500거래일 조건에서 성능 smoke로 해석하기에는 표본이 작다. 따라서 이번 CP에서는 PatchTST 학습을 실행하지 않고 split 진단만 남겼다.

## 8. Band smoke 결과

실행 조건:

| 항목 | 값 |
|---|---|
| model | cnn_lstm |
| timeframe | 1D |
| horizon | 5 |
| seq_len | 60 |
| feature_set | price_volatility_volume |
| q_low/q_high | 0.15 / 0.85 |
| lambda_band | 2.0 |
| band_mode | direct |
| checkpoint_selection | band_gate |
| epochs | 1 |
| device | CPU |
| W&B | off |
| save-run | false |

결과:

| 항목 | 값 |
|---|---|
| run_id | `cnn_lstm-1D-1f4e31aa2c78` |
| pipeline status | PASS |
| quality gate | failed |
| selected_reason | `band_gate_failed_fallback_val_total` |
| model_runs DB rows | 0 |
| predictions DB rows | 0 |
| local result | `logs/cp92_yfinance_validation/band_smoke_result.json` |
| checkpoint | `ai/artifacts/checkpoints/cnn_lstm_1D_cnn_lstm-1D-1f4e31aa2c78.pt` |

주요 test metrics:

| metric | value |
|---|---:|
| empirical_coverage | 0.8897958993911743 |
| coverage_abs_error | 0.18979589939117436 |
| lower_breach_rate | 0.03591836616396904 |
| upper_breach_rate | 0.07428571581840515 |
| band_width_ic | -0.09487612134414772 |
| downside_width_ic | 0.06717494224776584 |
| asymmetric_interval_score | 0.36390411853790283 |
| mae | 0.08554311096668243 |
| smape | 1.5864005088806152 |

판정: 성능은 판단하지 않는다. 이번 smoke의 목적은 yfinance source-aware indicator, feature cache, split, train loop가 비지 않고 연결되는지 확인하는 것이다. 그 목적은 통과했다. `band_gate` 실패는 5티커 1 epoch smoke 조건에서는 expected risk로 본다.

## 9. 50티커 write 가능 여부

판정: **조건부 가능**.

가능한 이유:

- `price_data` source-aware 병렬 구조가 닫혔다.
- yfinance 5티커 1D indicator를 `source='yfinance'`로 재계산했다.
- EODHD indicator row를 삭제하지 않았다.
- yfinance feature/index cache가 EODHD 후보 cache와 path/hash/manifest 기준으로 분리됐다.
- 5티커 band smoke가 DB 저장 없이 source-aware pipeline을 통과했다.

조건:

- 50티커 write도 `source='yfinance'` 제한 write만 허용한다.
- write 직후 50티커에 대해 `provider=yfinance`, `timeframe=1D` indicator 재계산을 같은 CP 안에서 수행한다.
- full/장기 write는 아직 금지한다.
- EODHD 최근 baseline 공백은 별도 승인으로 복구하거나, migration artifact로 수용한다는 결정을 먼저 명확히 해야 한다.
- live inference 연결과 product run 교체는 다음 단계로 미룬다.

## 10. 금지 사항 준수 확인

수행하지 않은 작업:

- 50티커 write
- 장기 write
- 실제 EODHD 복구
- EODHD row 삭제
- full retraining
- save-run
- live inference
- product run 교체

수행한 write:

- `indicators`에 5티커 `source='yfinance'`, `timeframe='1D'` row 2,205건 upsert
- yfinance source indicator 재계산 과정에서 yfinance source row만 재작성
- feature/index cache와 manifest 생성
- 1 epoch smoke local log 및 checkpoint 생성

DB 저장 확인:

- band smoke run_id의 `model_runs` row 수: 0
- band smoke run_id의 `predictions` row 수: 0

## 11. 실행한 주요 명령

DB/source 진단은 읽기 전용 Python 쿼리로 수행했다. 핵심 실행 명령은 아래와 같다.

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --provider yfinance --timeframes 1D --tickers AAPL MSFT NVDA TSLA NFLX --lookback-days 14 --force-full-backfill
```

```powershell
python -m json.tool logs\cp92_yfinance_validation\band_smoke_result.json
```

band smoke는 CLI 기본 split gate를 우회하기 위해 `prepare_dataset_splits(..., min_fold_samples=1)`로 5티커 smoke용 precomputed bundle을 만든 뒤 `ai.train.run_training(..., save_run=False, use_wandb=False)`를 직접 호출했다. 이는 full training이나 save-run이 아니라 CP92의 제한 smoke 검증이다.

## 12. 다음 단계

추천 다음 CP:

1. 50티커 yfinance 제한 write를 실행하되, 같은 CP 안에서 50티커 1D indicator source-aware 재계산과 feature cache sanity를 함께 수행한다.
2. CP89로 덮인 5티커 EODHD 최근 구간을 복구할지, 아니면 migration artifact로 계속 두고 EODHD 비교 baseline에서 제외할지 결정한다.
3. 50티커 write 후에도 `source_data_hash`, manifest provider/source, duplicate, ratio p99, `NaN/Inf`를 같은 기준으로 재검증한다.

