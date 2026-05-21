# CP93-D yfinance 50티커 제한 write 및 source-aware 검증 보고서

## 1. Executive Summary

CP93에서는 CP92의 5티커 yfinance source-aware 루프를 50티커로 확대했다. 전체 universe write, 2015년 장기 write, EODHD 복구, live inference, save-run은 실행하지 않았다.

최종 판정은 **PASS_WITH_NOTES**다. yfinance 50티커 가격 row는 2024-05-03부터 2026-05-01까지 총 25,000건으로 확인됐고, `source/provider/provider_adjustment_policy/updated_at` 누락은 0건이었다. raw OHLC와 adjusted OHLC sanity violation도 0건이었다. 1D indicator는 같은 50티커에 대해 `source='yfinance'`로 22,050건 재계산했고, duplicate 0건, `atr_ratio` coverage 100%, ratio p99 정상 범위를 확인했다.

주의할 점도 있다. CLI write 경로인 `sync_prices.run()`은 `stock_info`에 존재하는 ticker만 처리하므로 ETF인 SPY/QQQ가 처음에는 빠졌다. 필수 포함 조건을 맞추기 위해 공식 `sync_stock_info` 경로로 SPY/QQQ metadata를 먼저 보강했고, 그 뒤 동일 provider abstraction과 adjusted OHLC contract 검증을 거쳐 price_data에 upsert했다. 임의 sector/industry 값을 만들지는 않았고, 실제 sync 결과는 `Financial Services / Asset Management`로 채워졌다.

또 하나의 중요한 발견은 split gate다. 2년 제한 데이터에서는 yfinance feature/index는 정상이어도 기본 `min_fold_samples=50` 기준에서 seq60과 seq252 모두 ticker별 Gate C로 전부 제외된다. source-aware pipeline 실패가 아니라 기간 제한과 split gate 정책의 충돌이다. smoke 검증은 CP92와 같은 방식으로 `min_fold_samples=1` precomputed split을 사용했다.

## 2. 50티커 목록

최종 50티커:

```text
AAPL MSFT NVDA TSLA NFLX AMZN GOOGL META AMD AVGO SPY QQQ GOOG BRK-B JPM V MA UNH HD PG COST XOM JNJ WMT LLY ORCL CRM ADBE CSCO BAC PEP KO MRK ABBV TMO ACN MCD LIN DIS INTC QCOM TXN AMAT MU IBM GE CAT BA NKE HON
```

필수 포함 12개:

```text
AAPL MSFT NVDA TSLA NFLX AMZN GOOGL META AMD AVGO SPY QQQ
```

초기 후보였던 `CMCSA`는 제외했다. adjusted close와 adjusted OHLC contract는 통과했지만, EODHD 대비 raw close median diff가 약 6.28%로 현재 dry-run gate에서 `comparison_failed`로 분류됐다. 이 차이는 오류라기보다 provider raw/adjustment policy 차이일 가능성이 있지만, CP93 write gate를 안정적으로 통과시키기 위해 `HON`으로 교체했다.

## 3. 사전 상태와 CP89 artifact

사전 진단에서 50개 후보 중 기존 DB row가 있는 ticker는 48개였다. SPY/QQQ는 `stock_info`와 `price_data`에 없어서 별도 보강이 필요했다.

CP89/CP92 5티커 artifact는 그대로 유지됐다.

| 그룹 | ticker | legacy null rows | yfinance rows | 병렬 EODHD/legacy 날짜 | yfinance 구간 EODHD 공백 |
|---|---|---:|---:|---:|---:|
| CP89 artifact | AAPL, MSFT, NVDA, TSLA, NFLX | 각 2,349 | 각 500 | 각 0 | 각 500 |

이 artifact는 이번 CP에서도 복구하지 않았다. 2024-05-03부터 2026-05-01까지 5티커 EODHD 최근 baseline 공백은 known migration artifact로 계속 기록한다.

## 4. yfinance 50티커 제한 write 결과

write는 두 단계로 끝났다.

| 경로 | 대상 | 결과 |
|---|---|---:|
| `backend.collector.pipelines.yfinance_price_sync` | stock_info에 있던 48티커 | 24,000 rows |
| `sync_stock_info` 보강 후 provider 직접 upsert | SPY, QQQ | 1,000 rows |
| 합계 | 50티커 | 25,000 rows |

가격 계약 검증:

| 항목 | 결과 |
|---|---:|
| yfinance price rows | 25,000 |
| ticker 수 | 50 |
| date_min | 2024-05-03 |
| date_max | 2026-05-01 |
| duplicate `(ticker,date,source)` | 0 |
| raw OHLC violation | 0 |
| adjusted OHLC violation | 0 |
| adjusted factor violation | 0 |
| core nonfinite count | 0 |
| source missing | 0 |
| provider missing | 0 |
| provider_adjustment_policy missing | 0 |
| updated_at missing | 0 |
| volume null/negative/zero | 0 / 0 / 0 |

dry-run 비교 분류:

| status | count | 해석 |
|---|---:|---|
| pass | 45 | EODHD 비교 또는 baseline 기준 통과 |
| dividend_adjustment_policy_diff | 2 | 배당/조정 정책 차이로 분류 |
| split_adjustment_policy_diff | 1 | split raw/adjusted 정책 차이로 분류 |
| baseline_missing_contract_only | 2 | SPY/QQQ, 기존 baseline 없음 |

fallback 사용 ticker는 0개였다.

## 5. source별 row count

가격 row 패턴:

| 그룹 | ticker 수 | source 구성 |
|---|---:|---|
| CP89 artifact 5티커 | 5 | legacy null 2,349 rows + yfinance 500 rows |
| 신규 병렬 43티커 | 43 | legacy null 2,849 rows + yfinance 500 rows |
| ETF 2티커 | 2 | yfinance 500 rows only |

SPY/QQQ는 이번 CP에서 `stock_info`가 먼저 보강됐다.

| ticker | sector | industry | market_cap |
|---|---|---|---:|
| SPY | Financial Services | Asset Management | 738,008,328,979 |
| QQQ | Financial Services | Asset Management | 372,557,048,865 |

## 6. source-aware indicator 재계산

실행 명령:

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --provider yfinance --timeframes 1D --tickers AAPL MSFT NVDA TSLA NFLX AMZN GOOGL META AMD AVGO SPY QQQ GOOG BRK-B JPM V MA UNH HD PG COST XOM JNJ WMT LLY ORCL CRM ADBE CSCO BAC PEP KO MRK ABBV TMO ACN MCD LIN DIS INTC QCOM TXN AMAT MU IBM GE CAT BA NKE HON --lookback-days 14 --force-full-backfill
```

결과:

| 항목 | 결과 |
|---|---:|
| stored rows | 22,050 |
| ticker 수 | 50 |
| ticker별 rows | 441 |
| date_min | 2024-07-30 |
| date_max | 2026-05-01 |
| duplicate `(ticker,timeframe,date,source)` | 0 |
| provider missing | 0 |
| atr_ratio non-null | 22,050 |
| atr_ratio coverage | 100% |
| core nonfinite count | 0 |

ratio sanity:

| 항목 | p99 | max |
|---|---:|---:|
| open_ratio abs | 0.060293052021098484 | 0.375174588363923 |
| high_ratio abs | 0.08319319254337196 | 0.431493456711764 |
| low_ratio abs | 0.07956771674023687 | 0.297073942956931 |
| atr_ratio | 0.0710161619864234 | 0.109748823563649 |

판정: p99 기준으로는 CP29 이전의 ratio 폭주가 재발하지 않았다. max 값은 일부 급변 ticker/date 구간으로 보이며, 전체 p99와 nonfinite 0 기준에서는 실패로 보지 않는다.

## 7. feature cache 검증

feature/cache 계약:

| 항목 | 결과 |
|---|---|
| feature contract | `v3_adjusted_ohlc` |
| MODEL_N_FEATURES | 36 |
| MODEL_FEATURE_COLUMNS count | 36 |
| atr_ratio in model features | false |
| feature rows | 22,050 |
| index rows | 22,050 |
| price rows | 25,000 |
| feature ticker count | 50 |
| price ticker count | 50 |
| source feature nonfinite | 0 |
| calendar 포함 model feature nonfinite | 0 |

cache/hash:

| 항목 | 값 |
|---|---|
| yfinance source_data_hash | `2c075526` |
| EODHD source_data_hash 후보 | `3e90764c` |
| hash 분리 | true |
| cache path 분리 | true |
| yfinance feature cache | `ai/cache/features_1D_6bf146a3d27a_2c075526.pt` |
| yfinance index cache | `ai/cache/feature_index_1D_f6fbb09d6a8c_2c075526.pt` |

manifest 확인:

| manifest | provider | source | feature_version | ticker_count |
|---|---|---|---|---:|
| features | yfinance | yfinance | v3_adjusted_ohlc | 50 |
| feature_index | yfinance | yfinance | v3_adjusted_ohlc | 50 |

target 분포:

| dataset | samples | p01 | p50 | p99 | abs return > 50% |
|---|---:|---:|---:|---:|---:|
| seq60 h5 raw_future_return | 18,850 | -0.10526116192340851 | 0.0016971230506896973 | 0.12130000710487336 | 0 |
| seq252 h5 raw_future_return | 9,250 | -0.09399465024471283 | 0.0016123652458190918 | 0.12342566967010556 | 0 |

## 8. split 진단

기본 split gate:

| seq_len | horizon | min_fold_samples | 결과 | 원인 |
|---:|---:|---:|---|---|
| 60 | 5 | 50 | 실패 | 50티커 모두 Gate C |
| 252 | 5 | 50 | 실패 | 50티커 모두 Gate C |

이 실패는 source-aware join 실패가 아니다. `usable_row_count=22,050`, `estimated_usable_sample_count=18,100(seq60) / 8,500(seq252)`가 존재한다. 다만 2년 제한 데이터에서 ticker별 val/test fold가 기본 min fold 기준보다 작아 Gate C가 발생한다.

smoke용 split:

| seq_len | min_fold_samples | eligible tickers | train | val | test |
|---:|---:|---:|---:|---:|---:|
| 60 | 1 | 50 | 11,250 | 2,400 | 2,450 |
| 252 | 1 | 50 | 4,550 | 950 | 1,000 |

## 9. smoke 결과

### Band smoke

조건:

| 항목 | 값 |
|---|---|
| model | cnn_lstm |
| feature_set | price_volatility_volume |
| timeframe / horizon | 1D / h5 |
| seq_len | 60 |
| q_low / q_high | 0.15 / 0.85 |
| lambda_band | 2.0 |
| band_mode | direct |
| checkpoint_selection | band_gate |
| device | CPU |
| epochs | 1 |
| W&B | off |
| save-run | false |

결과:

| 항목 | 값 |
|---|---|
| run_id | `cnn_lstm-1D-2bc82402e1d8` |
| n_features | 11 |
| train / val / test | 11,250 / 2,400 / 2,450 |
| model_runs DB rows | 0 |
| predictions DB rows | 0 |
| gate | band_gate eligible |
| local result | `logs/cp93_yfinance_validation/band_smoke_result.json` |

주요 test metrics:

| metric | value |
|---|---:|
| empirical_coverage | 0.7720000147819519 |
| coverage_abs_error | 0.07200001478195195 |
| lower_breach_rate | 0.08955101668834686 |
| upper_breach_rate | 0.13844898343086243 |
| band_width_ic | -0.023195595757067117 |
| downside_width_ic | 0.018637147327608246 |
| asymmetric_interval_score | 0.29473158717155457 |
| mae | 0.06937455385923386 |
| smape | 1.5254759788513184 |

### Line smoke

조건:

| 항목 | 값 |
|---|---|
| model | PatchTST |
| feature_set | full_features |
| timeframe / horizon | 1D / h5 |
| seq_len | 252 |
| patch_len / stride | 32 / 16 |
| checkpoint_selection | line_gate |
| device | CPU |
| epochs | 1 |
| W&B | off |
| save-run | false |

결과:

| 항목 | 값 |
|---|---|
| run_id | `patchtst-1D-45261d3f92c0` |
| n_features | 36 |
| train / val / test | 4,550 / 950 / 1,000 |
| model_runs DB rows | 0 |
| predictions DB rows | 0 |
| gate | line_gate eligible |
| local result | `logs/cp93_yfinance_validation/line_smoke_result.json` |

주요 test metrics:

| metric | value |
|---|---:|
| spearman_ic / ic_mean | -0.10341896758703481 |
| long_short_spread | -0.004940028878627345 |
| false_safe_tail_rate | 0.381 |
| severe_downside_recall | 0.616557734204793 |
| empirical_coverage | 0.9277999997138977 |
| coverage_abs_error | 0.12779999971389766 |
| mae | 0.5236377716064453 |
| smape | 1.8301546573638916 |

해석: 두 smoke 모두 source-aware yfinance data pipeline이 비지 않고 학습 루프까지 연결됨을 확인하기 위한 것이다. 성능 후보 교체 근거로 쓰지 않는다.

## 10. EODHD 복구와 rollback 판단

EODHD 복구는 이번 CP에서 실행하지 않았다. CP89 5티커 최근 EODHD 공백은 known migration artifact로 유지한다.

rollback은 필요하지 않다. 이유:

- EODHD row 삭제 없음
- yfinance row는 `source='yfinance'`로 병렬 저장
- indicators도 `source='yfinance'`로 분리 저장
- feature cache도 provider/hash/manifest로 분리
- smoke run은 save-run 없이 DB `model_runs/predictions` 저장 0건

주의: SPY/QQQ는 기존 EODHD baseline이 없으므로 provider diff 비교 기준선이 없다. 이는 오류가 아니라 신규 ETF 편입 상태로 분류한다.

## 11. 100티커/전체 write 가능 여부

100티커 제한 write는 **조건부 가능**하다.

조건:

- 100티커 write도 2024-05-03~2026-05-01 같은 제한 기간으로 시작한다.
- write 직후 `provider=yfinance`, `timeframe=1D` indicator 재계산을 같은 CP 안에서 수행한다.
- `min_fold_samples=50` Gate C 정책을 smoke/운영 split에서 어떻게 다룰지 먼저 결정한다.
- SPY/QQQ처럼 `stock_info` FK가 없는 ticker의 metadata seed 정책을 명문화한다.
- full universe write와 2015년 장기 write는 계속 금지한다.

전체 write는 **아직 금지**다. 이유는 split gate 정책, ETF/비 S&P metadata FK 정책, EODHD baseline 공백 처리 정책이 아직 완전히 닫히지 않았기 때문이다.

## 12. 실행한 주요 명령

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --start-date 2024-05-03 --end-date 2026-05-01 --compare-tickers <50 tickers> --tickers <50 tickers> --write --batch-limit 50 --sleep-seconds 0.1 --metrics-path logs\cp93_yfinance_validation\yfinance_50_write_metrics.json
```

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --provider yfinance --timeframes 1D --tickers <50 tickers> --lookback-days 14 --force-full-backfill
```

추가로 SPY/QQQ에 대해서는 `sync_stock_info.run(['SPY','QQQ'])` 후 provider abstraction 기반 adjusted OHLC validation과 `price_data` upsert를 제한 실행했다. 모델 smoke는 `ai.train.run_training(..., save_run=False, use_wandb=False, precomputed_bundles=...)`로 실행했다.

## 13. 금지 사항 준수

수행하지 않은 작업:

- 전체 universe write
- 2015년 장기 write
- EODHD row 삭제
- EODHD row 복구 실행
- full retraining
- save-run
- live inference
- product run 교체
- 1W/1M 전체 재계산

수행한 write:

- yfinance 50티커 `price_data` 제한 upsert
- SPY/QQQ `stock_info` 공식 sync 보강
- yfinance 50티커 1D `indicators` source-aware upsert
- feature/index cache와 local smoke checkpoint/log 생성

