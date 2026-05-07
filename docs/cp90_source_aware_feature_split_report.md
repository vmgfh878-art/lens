# CP90-D source-aware feature index and split planning 보고서

작성일: 2026-05-03

대상 provider/source: `yfinance`

대상 ticker: AAPL, MSFT, NVDA, TSLA, NFLX

금지 준수:

- 전체 yfinance write 실행하지 않음
- EODHD row 삭제하지 않음
- full retraining 실행하지 않음
- `--save-run` 사용하지 않음
- live inference 연결하지 않음
- product run 교체하지 않음
- 1W/1M 실험 실행하지 않음

최종 판정: **PASS_SOURCE_AWARE_SPLIT_WITH_LINE_PERIOD_LIMIT**

source-aware feature index와 split planning 수리는 성공했다. CP89에서 `prepare_dataset_splits`가 source/provider 없는 전체 `indicators` history를 보고 2015년부터 split plan을 잡던 문제가 해결됐다. 이제 yfinance 모드에서는 `price_data.source='yfinance'`로 실제 존재하는 ticker/date만 feature index와 split plan에 반영된다.

다만 PatchTST line smoke의 `seq_len=252`는 5티커 최근 2년 yfinance 제한 데이터로는 val/test fold가 충분하지 않아 Gate C로 제외된다. 이것은 더 이상 모호한 empty split 실패가 아니라, provider/source/date/seq_len/gap/sample count가 포함된 기간 부족 진단이다.

## 1. CP89 split 실패 원인

CP89에서는 yfinance 5티커 제한 write와 1D indicator 재계산이 성공했지만, line/band smoke가 모두 다음 오류로 실패했다.

```text
ValueError: split 결과가 비어 있습니다.
```

원인은 다음과 같다.

- `price_data`는 최근 2년 5티커만 `source='yfinance'`로 존재했다.
- `indicators`에는 source/provider provenance가 없고, 같은 ticker의 과거 전체 history가 남아 있었다.
- 기존 `fetch_feature_index_frame()`은 `indicators`만 보고 feature index를 만들었다.
- 그래서 split plan은 2015년부터의 긴 history 기준으로 만들어졌다.
- 이후 실제 학습 frame은 yfinance source-filtered `price_data`와 merge되며 2024-05 이후로 줄어들었다.
- split spec의 sample index와 실제 dataset sample index가 어긋나 train/val/test 중 일부가 비었다.

즉 CP89 실패는 가격 품질 문제가 아니라 **feature index가 provider/source-aware하지 않은 split planning 문제**였다.

## 2. 구현 변경 내용

수정 파일:

- `ai/preprocessing.py`
- `ai/train.py`
- `ai/tests/test_preprocessing_cache_isolation.py`

핵심 변경:

| 영역 | 변경 |
|---|---|
| provider 명시 | `fetch_feature_index_frame`, `fetch_training_frames`, `prepare_dataset_splits`에 `market_data_provider` 인자를 추가했다. |
| CLI 명시 | `ai.train`에 `--market-data-provider` 옵션을 추가했다. |
| source-aware index | Postgres 경로에서 `indicators`를 provider-filtered `price_data`와 `ticker/date`로 join한다. |
| EODHD legacy 계약 | EODHD mode에서는 `source='eodhd' OR source IS NULL`을 유지한다. |
| REST fallback | REST fallback에서도 price frame을 provider/source 기준으로 필터링한 뒤 feature index를 만든다. |
| cache/manifest | feature/index manifest에 `source`를 추가하고 provider/source mismatch cache 재사용을 막는다. |
| split diagnostics | split 실패 시 provider, source, ticker_count, date_min/max, seq_len, horizon, gap, usable sample count, excluded reason을 JSON으로 출력한다. |
| dataset plan | `DatasetPlan`에 provider/source/source_data_hash/date range/usable row/sample count를 추가했다. |

## 3. source-aware feature index 결과

CP89와 같은 5티커 yfinance 제한 데이터 기준:

| 항목 | 결과 |
|---|---:|
| feature index rows | 2,500 |
| ticker count | 5 |
| date_min | 2024-05-03 |
| date_max | 2026-05-01 |
| source_data_hash | `7b883d3c` |
| EODHD source_data_hash | `d5635005` |
| hash differs | true |
| feature index cache path differs | true |
| feature cache path differs | true |

manifest 확인:

- `provider=yfinance`
- `source=yfinance`
- `provider_adjustment_policy=yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc`
- `feature_version=v3_adjusted_ohlc`
- `source_data_hash=7b883d3c`

feature contract 확인:

| 항목 | 결과 |
|---|---:|
| feature rows | 2,205 |
| price rows | 2,500 |
| feature date_min | 2024-07-30 |
| feature date_max | 2026-05-01 |
| MODEL_N_FEATURES | 36 |
| atr_ratio in MODEL_FEATURE_COLUMNS | false |
| feature finite | true |
| price finite | true |

`feature index rows`가 2,500이고 `feature rows`가 2,205인 차이는 정상이다. index는 yfinance price row availability를 기준으로 잡고, 실제 feature frame은 MA/RSI/MACD 등 lookback 이후부터 생성되기 때문이다.

## 4. yfinance 5티커 split 가능 여부

### band smoke plan

조건:

- model: CNN-LSTM
- seq_len: 60
- horizon: 5
- h_max/gap: 20
- min_fold_samples: 50
- provider/source: yfinance

결과:

| 항목 | 결과 |
|---|---:|
| eligible tickers | 5 |
| excluded tickers | 0 |
| usable row count | 2,500 |
| estimated usable sample count | 2,105 |
| planned train samples | 1,330 |
| planned val samples | 285 |
| planned test samples | 290 |

판정: **split 가능**

### line smoke plan

조건:

- model: PatchTST
- seq_len: 252
- horizon: 5
- h_max/gap: 20
- min_fold_samples: 50
- provider/source: yfinance

결과:

| 항목 | 결과 |
|---|---:|
| eligible tickers | 0 |
| usable row count | 2,500 |
| estimated usable sample count | 1,145 |
| excluded reason | AAPL/MSFT/NFLX/NVDA/TSLA 모두 Gate C |

line smoke 진단 payload:

```json
{
  "error": "no_eligible_tickers_for_provider_source",
  "provider": "yfinance",
  "source": "yfinance",
  "timeframe": "1D",
  "date_min": "2024-05-03",
  "date_max": "2026-05-01",
  "ticker_count": 5,
  "eligible_ticker_count": 0,
  "seq_len": 252,
  "horizon": 5,
  "gap": 20,
  "min_fold_samples": 50,
  "usable_row_count": 2500,
  "estimated_usable_sample_count": 1145,
  "excluded_reasons": {
    "AAPL": "Gate C",
    "MSFT": "Gate C",
    "NFLX": "Gate C",
    "NVDA": "Gate C",
    "TSLA": "Gate C"
  }
}
```

판정: **기간 부족**

이제 실패는 모호한 empty split이 아니라, seq_len/h_max/min_fold_samples 기준에서 최근 2년 5티커로는 PatchTST 252-window fold가 부족하다는 명확한 진단이다.

## 5. smoke 결과

### A. band smoke

실행 명령:

```powershell
$env:WANDB_MODE='disabled'; python -m ai.train --market-data-provider yfinance --model cnn_lstm --timeframe 1D --horizon 5 --seq-len 60 --feature-set price_volatility_volume --q-low 0.15 --q-high 0.85 --lambda-band 2.0 --band-mode direct --checkpoint-selection band_gate --fp32-modules lstm,heads --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp90_yfinance_smoke
```

결과:

| 항목 | 결과 |
|---|---|
| exit code | 0 |
| run_id | `cnn_lstm-1D-cfa1a15ca41e` |
| W&B | disabled_by_cli |
| save-run | false |
| status | failed_quality_gate |
| empirical_coverage | 0.8743 |
| coverage_abs_error | 0.1743 |
| lower_breach_rate | 0.0400 |
| upper_breach_rate | 0.0857 |
| asymmetric_interval_score | 0.3970 |
| band_width_ic | -0.0970 |
| downside_width_ic | -0.0239 |

해석:

- smoke 목적의 exit code, finite, metrics 생성 기준은 통과했다.
- `failed_quality_gate`는 1epoch/5티커 smoke 성능 gate 문제이며, source-aware split 실패가 아니다.
- 이 성능 수치는 전환 판정에 과도하게 쓰지 않는다.

### B. line smoke

실행 명령:

```powershell
$env:WANDB_MODE='disabled'; python -m ai.train --market-data-provider yfinance --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --patch-len 32 --patch-stride 16 --feature-set full_features --checkpoint-selection line_gate --tickers AAPL MSFT NVDA TSLA NFLX --epochs 1 --batch-size 64 --device cpu --no-wandb --no-compile --local-log-dir logs/cp90_yfinance_smoke
```

결과:

| 항목 | 결과 |
|---|---|
| exit code | 1 |
| 분류 | insufficient_period_gate_c |
| provider/source | yfinance/yfinance |
| date range | 2024-05-03~2026-05-01 |
| usable rows | 2,500 |
| excluded reason | 5티커 모두 Gate C |

해석:

- CP89의 모호한 split empty 실패는 해결됐다.
- PatchTST seq_len=252는 현재 5티커/최근 2년 제한 데이터로는 split gate를 충족하지 못한다.
- 이번 CP에서는 seq_len=120 변경이나 yfinance 기간 확장 write를 실행하지 않았다. 다음 CP에서 별도 판단한다.

## 6. 50티커 확대 전 필요한 조건

50티커 write 또는 local daily sync 전환 전 필요한 조건:

1. 이번 source-aware index/split 변경을 유지한다.
2. yfinance source 기준 feature index rows와 price rows가 ticker별로 기대 범위인지 확인한다.
3. indicators에 source/provider 컬럼을 둘지, 또는 source provenance를 price/cache/manifest에서만 관리할지 결정한다.
4. 50티커 제한 write 전 `ticker,date` upsert 정책을 유지할지, `ticker,date,source` 병렬 row 저장 구조로 바꿀지 결정한다.
5. PatchTST line smoke는 seq_len=120으로 별도 smoke를 하거나, yfinance write 기간을 5년 이상으로 늘린 뒤 재시도한다.
6. 50티커 확대 후에도 `MODEL_N_FEATURES=36`, `atr_ratio` 미포함, feature/target NaN/Inf 0을 다시 확인한다.

## 7. 다음 단계

추천 순서:

1. 5티커 yfinance 기준 CNN-LSTM band smoke는 통과했으므로, 다음에는 50티커 제한 write를 검토할 수 있다.
2. PatchTST line은 현재 2년/5티커로는 부족하므로, `seq_len=120` smoke 또는 yfinance 기간 확장 중 하나를 별도 CP로 결정한다.
3. 50티커 write 전 `price_data` upsert key 정책을 한 번 더 확정한다.
4. 50티커 write 후 1D indicators 재계산, source-aware feature cache, band smoke를 같은 순서로 반복한다.
5. live inference 연결은 50티커 제한 write와 smoke가 통과한 뒤로 둔다.

## 8. 검증

실행한 검증:

```powershell
python -m py_compile ai\preprocessing.py ai\train.py ai\tests\test_preprocessing_cache_isolation.py
```

```powershell
python -m unittest ai.tests.test_preprocessing_cache_isolation ai.tests.test_feature_set_selection backend.tests.test_market_data_providers
```

결과:

- py_compile PASS
- unittest 16개 PASS
- CP90 metrics JSON parse PASS

산출물:

- `docs/cp90_source_aware_feature_split_report.md`
- `docs/cp90_source_aware_feature_split_metrics.json`

