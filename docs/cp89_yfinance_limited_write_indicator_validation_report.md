# CP89-D yfinance 제한 write 및 indicator 재계산 검증 보고서

작성일: 2026-05-03

대상: AAPL, MSFT, NVDA, TSLA, NFLX

범위: 운영 DB runtime schema 적용, yfinance 5티커 1D 제한 write, 1D indicators 재계산, yfinance feature cache/source hash 격리 검증, 최소 smoke 실행.

금지 준수:

- 전체 ticker write 실행하지 않음
- 기존 EODHD row 삭제하지 않음
- full retraining 실행하지 않음
- `--save-run` 사용하지 않음
- live inference 연결하지 않음
- product run 교체하지 않음
- fake data 생성하지 않음
- 1W/1M 전체 재계산하지 않음

최종 판정: **WARN**

schema migration, 5티커 yfinance write, `price_data` provenance 확인, 1D indicators 재계산, feature cache isolation은 통과했다. 다만 line/band model smoke는 둘 다 `prepare_dataset_splits` 단계에서 `ValueError: split 결과가 비어 있습니다.`로 실패했다.

이번 실패는 yfinance 가격 row 자체의 NaN/Inf 문제가 아니라, 제한 write 상태에서 `price_data`는 2024-05-03 이후 5티커만 `source='yfinance'`로 존재하고 `indicators`/feature index는 source/provider 없이 2015년부터의 전체 history를 기준으로 split plan을 잡는 불일치 때문이다. 따라서 전체 yfinance 전환은 아직 금지하고, 다음 CP에서 yfinance source-aware split/index 계약을 먼저 고쳐야 한다.

## 1. schema migration 결과

실행 명령:

```powershell
python -m backend.db.scripts.ensure_runtime_schema
```

결과:

- 실행 성공
- `price_data` provenance 컬럼 확인 완료
- 기존 EODHD legacy null row 대량 backfill은 실행하지 않음

확인된 `price_data` 컬럼:

| 컬럼 | 상태 |
|---|---|
| `source` | 존재 |
| `provider` | 존재 |
| `provider_adjustment_policy` | 존재 |
| `updated_at` | 존재 |

주의: `ensure_runtime_schema`는 기존 runtime schema 보강 스크립트라 `price_data` 외에도 기존 AI runtime table/column 보강 statement를 포함한다. 이번 CP에서 별도 수동 DDL이나 row 삭제는 하지 않았다.

## 2. 5티커 yfinance 제한 write 결과

실행 명령:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --tickers AAPL MSFT NVDA TSLA NFLX --compare-tickers AAPL MSFT NVDA TSLA NFLX --start-date 2024-05-03 --end-date 2026-05-03 --write --lookback-days 730 --batch-limit 5 --sleep-seconds 0.2 --metrics-path docs/cp89_yfinance_limited_write_indicator_validation_metrics.json
```

요약:

| 항목 | 결과 |
|---|---:|
| dry-run compare overall_pass | true |
| write attempted | true |
| stored_rows | 2,500 |
| processed tickers | AAPL, MSFT, NVDA, TSLA, NFLX |
| failed tickers | 0 |
| fallback_used | 0 |
| quota_hit | false |
| DB write 범위 | 5티커, 1D, 2024-05-03~2026-05-01 |

`price_data` provenance 확인:

| 항목 | 결과 |
|---|---:|
| `source='yfinance'` row 수 | 2,500 |
| `provider='yfinance'` row 수 | 2,500 |
| `provider_adjustment_policy` 기록 row 수 | 2,500 |
| `updated_at` null count | 0 |
| duplicate `ticker/date/source` count | 0 |

provider adjustment policy:

```text
yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc
```

가격 sanity:

| 항목 | 결과 |
|---|---:|
| adjusted_factor bad count | 0 |
| raw OHLC bad count | 0 |
| adjusted OHLC bad count | 0 |
| volume null/negative count | 0 |

ratio sanity:

| metric | p99 abs | max abs |
|---|---:|---:|
| open_ratio | 0.0688 | 0.1476 |
| high_ratio | 0.0860 | 0.2381 |
| low_ratio | 0.0933 | 0.1817 |

판정: **PASS**

단, 현재 `sync_prices` upsert key는 `ticker,date`다. 따라서 같은 ticker/date에 yfinance write를 수행하면 해당 key의 row가 yfinance provenance로 갱신된다. 기존 row를 삭제하지는 않았지만, provider별 parallel row를 같은 table에 동시에 보관하는 구조는 아니다. 이 한계는 전체 전환 전 정책으로 확정해야 한다.

## 3. indicators 재계산 결과

실행 명령:

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --tickers AAPL MSFT NVDA TSLA NFLX --timeframes 1D --lookback-days 730
```

결과:

| 항목 | 결과 |
|---|---:|
| timeframe | 1D |
| source_start_date | 2025-08-24 |
| stored | 570 |
| ticker_count | 5 |
| force_full_backfill | false |
| 1W/1M 재계산 | 실행하지 않음 |

재계산 후 1D indicators 검증:

| 항목 | 결과 |
|---|---:|
| 검증 row 수 | 865 |
| ticker별 row 수 | 각 173 |
| date_min | 2025-08-25 |
| date_max | 2026-05-01 |
| atr_ratio non-null | 865 |

indicator ratio sanity:

| metric | p99 abs | max abs |
|---|---:|---:|
| open_ratio | 0.0438 | 0.1148 |
| high_ratio | 0.0659 | 0.1438 |
| low_ratio | 0.0594 | 0.1258 |
| atr_ratio | 0.0535 | 0.0596 |

최신 2026-05-01 샘플에서는 AAPL, MSFT, NFLX, NVDA, TSLA 모두 `atr_ratio`가 non-null이었다.

판정: **PASS with limitation**

중요 한계:

- `indicators`에는 `source`/`provider` 컬럼이 없다.
- `compute_indicators`는 `ticker,timeframe,date` 기준으로 upsert한다.
- 따라서 5티커 yfinance 기반 재계산 결과는 같은 key의 indicator 값을 갱신하지만, indicator row 자체가 yfinance 기반인지 EODHD 기반인지 DB row에서 직접 구분할 수 없다.
- 이번 CP에서는 기존 indicator row를 무작정 삭제하지 않았고, 1D 5티커 증분 재계산만 수행했다.

## 4. yfinance feature cache 검증

`MARKET_DATA_PROVIDER=yfinance` 기준으로 5티커 1D feature/index cache를 생성하고 manifest를 확인했다.

요약:

| 항목 | 결과 |
|---|---:|
| provider | yfinance |
| feature_version | v3_adjusted_ohlc |
| MODEL_N_FEATURES | 36 |
| `atr_ratio in MODEL_FEATURE_COLUMNS` | false |
| feature_rows | 2,205 |
| price_rows | 2,500 |
| feature ticker count | 5 |
| price ticker count | 5 |
| feature finite | true |
| price finite | true |
| lazy dataset sample count | 925 |
| lazy dataset probe finite | true |

hash/cache isolation:

| 항목 | yfinance | EODHD |
|---|---|---|
| source_data_hash | `7b883d3c` | `d5635005` |
| feature cache path | `ai/cache/features_1D_7a8241e91ef6_7b883d3c.pt` | `ai/cache/features_1D_b2598b39b5e2_d5635005.pt` |
| feature index cache path | `ai/cache/feature_index_1D_75abbf1d1993_7b883d3c.pt` | `ai/cache/feature_index_1D_774971c27d4a_d5635005.pt` |

manifest 확인:

- feature manifest `provider=yfinance`
- feature index manifest `provider=yfinance`
- `provider_adjustment_policy=yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc`
- `feature_version=v3_adjusted_ohlc`
- `source_data_hash=7b883d3c`

판정: **PASS**

주의: smoke 시도 과정에서 feature_set별 cache digest가 추가로 생성되었다. 같은 `source_data_hash=7b883d3c`라도 feature_set 또는 feature columns digest가 다르면 cache 파일명이 달라지는 것은 정상이다.

## 5. 모델 smoke 결과

### A. line smoke

실행 명령:

```powershell
$env:MARKET_DATA_PROVIDER='yfinance'; $env:WANDB_MODE='disabled'; python -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --patch-len 32 --patch-stride 16 --feature-set full_features --checkpoint-selection line_gate --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp89_yfinance_smoke
```

결과:

- exit code: 1
- save-run: 사용하지 않음
- W&B: disabled
- 실패 위치: `prepare_dataset_splits`
- 오류: `ValueError: split 결과가 비어 있습니다.`

### B. band smoke

실행 명령:

```powershell
$env:MARKET_DATA_PROVIDER='yfinance'; $env:WANDB_MODE='disabled'; python -m ai.train --model cnn_lstm --timeframe 1D --horizon 5 --seq-len 60 --feature-set price_volatility_volume --q-low 0.15 --q-high 0.85 --lambda-band 2.0 --band-mode direct --checkpoint-selection band_gate --fp32-modules lstm,heads --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp89_yfinance_smoke
```

결과:

- exit code: 1
- save-run: 사용하지 않음
- W&B: disabled
- 실패 위치: `prepare_dataset_splits`
- 오류: `ValueError: split 결과가 비어 있습니다.`

해석:

이 실패는 feature tensor NaN/Inf가 아니라 split/index 계약 문제다. yfinance price source는 최근 2년 5티커로 제한되어 있는데, `fetch_feature_index_frame()`은 `indicators` 전체 history를 기준으로 split plan을 만든다. 이후 실제 yfinance price와 merge된 sample index는 최근 2년 범위로 다시 시작하므로, plan의 val/test sample index와 dataset sample index가 맞지 않아 split 결과가 비게 된다.

판정: **FAIL for smoke, migration gate WARN**

## 6. 전체 yfinance 전환 가능 여부

현재 판정은 **WARN**이다.

제한 write 자체는 가능하다. `price_data` provenance, adjusted OHLC sanity, indicator 1D 재계산, feature cache isolation은 모두 통과했다.

하지만 전체 yfinance 전환 또는 live inference 연결은 아직 금지한다. 이유는 다음과 같다.

1. `indicators`에 source/provider provenance가 없어 yfinance 기반 indicator row를 DB에서 직접 구분할 수 없다.
2. `prepare_dataset_splits`가 yfinance price availability가 아니라 전체 indicator index 기준으로 split plan을 만든다.
3. 그 결과 실제 train smoke가 line/band 모두 split-empty로 실패했다.
4. provider별 parallel row 저장 정책이 `price_data`에서 완전히 분리된 구조는 아니다. 현재는 `ticker,date` upsert라 제한 write가 같은 key row를 yfinance provenance로 갱신한다.

따라서 다음 단계는 50/100티커 write가 아니라, 먼저 yfinance source-aware feature index/split 계약을 수리하는 것이다.

## 7. 다음 단계

우선순위:

1. `prepare_dataset_splits`가 split plan을 만들 때 provider-filtered price availability 또는 merged feature/price frame 기준으로 index를 잡도록 수정한다.
2. `indicators`에 source/provider provenance를 둘지, 아니면 indicator는 price source에서 파생되는 cache-level artifact로만 관리할지 정책을 결정한다.
3. 제한 write가 기존 `ticker,date` row를 갱신하는 정책을 유지할지, `ticker,date,source` parallel row 구조로 바꿀지 결정한다.
4. 위 수정 후 같은 5티커로 line/band 1epoch smoke를 재실행한다.
5. smoke PASS 후 50티커 제한 write와 local daily sync rehearsal로 확장한다.
6. Render cron 또는 live inference 연결은 그 이후 단계로 둔다.

## 8. 실행한 명령 목록

```powershell
python -m backend.db.scripts.ensure_runtime_schema
```

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --tickers AAPL MSFT NVDA TSLA NFLX --compare-tickers AAPL MSFT NVDA TSLA NFLX --start-date 2024-05-03 --end-date 2026-05-03 --write --lookback-days 730 --batch-limit 5 --sleep-seconds 0.2 --metrics-path docs/cp89_yfinance_limited_write_indicator_validation_metrics.json
```

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --tickers AAPL MSFT NVDA TSLA NFLX --timeframes 1D --lookback-days 730
```

```powershell
$env:MARKET_DATA_PROVIDER='yfinance'; $env:WANDB_MODE='disabled'; python -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --patch-len 32 --patch-stride 16 --feature-set full_features --checkpoint-selection line_gate --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp89_yfinance_smoke
```

```powershell
$env:MARKET_DATA_PROVIDER='yfinance'; $env:WANDB_MODE='disabled'; python -m ai.train --model cnn_lstm --timeframe 1D --horizon 5 --seq-len 60 --feature-set price_volatility_volume --q-low 0.15 --q-high 0.85 --lambda-band 2.0 --band-mode direct --checkpoint-selection band_gate --fp32-modules lstm,heads --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp89_yfinance_smoke
```

추가로 schema/price/indicator/cache 검증용 read query와 metrics JSON 생성 스크립트를 실행했다. secret 원문은 출력하지 않았다.

