# CP95-D yfinance 100티커 장기 write 및 전체 전환 전 게이트 보고서

## 1. Executive Summary

판정: **PASS_WITH_NOTES**

CP94의 50티커 장기 yfinance 검증을 100티커로 확대했다. 전체 universe write는 하지 않았고, 최종 100티커에 대해서만 `2015-01-01` 이후 1D price, source-aware indicators, feature cache, split gate, model smoke를 확인했다. 최신 거래일은 `2026-05-01`로 고정했다.

핵심 결과:

| 항목 | 결과 |
|---|---:|
| 최종 yfinance ticker | 100 |
| yfinance price_data rows | 284,898 |
| price date range | 2015-01-02 ~ 2026-05-01 |
| duplicate ticker/date/source | 0 |
| adjusted OHLC violation, tolerance 1e-10 | 0 |
| source/provider/policy 누락 | 0 |
| yfinance 1D indicators rows | 278,994 |
| atr_ratio coverage | 100.0% |
| feature rows | 278,956 |
| MODEL_N_FEATURES | 36 |
| atr_ratio in MODEL_FEATURE_COLUMNS | false |
| feature/target nonfinite | 0 |
| seq_len 60 split gate | PASS, 97 eligible |
| seq_len 252 split gate | PASS, 97 eligible |
| CNN-LSTM band smoke | PASS, exit code 0 |
| PatchTST line smoke | PASS, exit code 0 |

결론: 100티커 yfinance 장기 데이터는 전체 전환 전 게이트를 통과했다. 다만 `FI`, `MMC`는 yfinance primary 실패로 보류하고 `REGN`, `SPGI`로 대체했다. 모델 split에서는 `LMT`, `MS`, `T`가 기존 fundamentals gate로 제외되어 97 eligible이 됐다. 따라서 전체 1D yfinance write 후보로 올릴 근거는 생겼지만, EODHD를 바로 삭제하거나 live inference를 연결하기에는 아직 이르다.

## 2. 최종 100티커

CP94 50티커는 모두 유지했다. 추가 후보 중 `FI`, `MMC`는 yfinance primary 실패로 최종 목록에서 제외했고, `REGN`, `SPGI`를 대체로 추가했다. `CMCSA`는 포함하지 않았다.

```text
AAPL MSFT NVDA TSLA NFLX AMZN GOOGL META AMD AVGO SPY QQQ GOOG BRK-B JPM V MA UNH HD PG COST XOM JNJ WMT LLY ORCL CRM ADBE CSCO BAC PEP KO MRK ABBV TMO ACN MCD LIN DIS INTC QCOM TXN AMAT MU IBM GE CAT BA NKE HON ABT AMGN AMT AXP ADI APD ADP ANET BKNG BLK BX BSX BMY C CB CDNS CMG CME CL COP CVX DE DHR ELV ETN GILD GS ISRG LOW MDT MDLZ MS NOW PANW PFE PLD PM RTX SBUX SCHW SYK TJX UPS VRTX ZTS T UNP LMT REGN SPGI
```

사전 진단:

| 항목 | 결과 |
|---|---:|
| 초기 ticker count | 100 |
| CP94 ticker 포함 | 50/50 |
| 사전 duplicate ticker/date/source | 0 |
| 사전 yfinance row 보유 ticker | 50 |
| stock_info 누락 | FI, MMC |

`sync_stock_info.run()` 공식 경로로 `FI`, `MMC`를 보강했지만 실제 metadata fetch는 실패했고 placeholder row만 생성됐다. 이 둘은 이후 yfinance price primary에서도 실패했으므로 최종 판정용 yfinance 100티커에서 제외했다.

## 3. Price Write 결과

처음 100티커 write는 공식 `sync_prices.run()` 경로로 실행했다.

| 항목 | 결과 |
|---|---:|
| processed | 100 |
| failed | 0 |
| skipped | 0 |
| stored_rows | 284,704 |
| quota_hit | false |
| validation issue ticker | AMD, FI |

하지만 사후 DB 검증에서 `FI`, `MMC`는 `source='yfinance'`가 아니라 EODHD fallback row로 남은 것이 확인됐다.

| 티커 | 상태 | 처리 |
|---|---|---|
| FI | yfinance download 실패 후 EODHD row 생성 | 최종 100에서 제외 |
| MMC | yfinance download 실패 후 EODHD row 생성 | 최종 100에서 제외 |
| REGN | yfinance 장기 write 성공 | 대체 편입 |
| SPGI | yfinance 장기 write 성공 | 대체 편입 |

대체 write 결과:

| 항목 | 결과 |
|---|---:|
| replacement tickers | REGN, SPGI |
| processed | 2 |
| failed | 0 |
| stored_rows | 5,698 |
| validation issue | 0 |

`FI`, `MMC`의 fallback artifact row는 삭제하지 않았다. 이는 CP95 금지 조건에 맞춰 known migration artifact로만 기록한다.

## 4. 데이터 품질 검증

최종 100티커의 `source='yfinance'` price_data 기준:

| 항목 | 결과 |
|---|---:|
| rows | 284,898 |
| tickers_with_rows | 100 |
| missing_tickers | 0 |
| date_min | 2015-01-02 |
| date_max | 2026-05-01 |
| duplicate_ticker_date_source | 0 |
| raw_ohlc_violation_count | 0 |
| adjusted_factor_violation_count | 0 |
| adjusted_ohlc_violation_count_strict | 41 |
| adjusted_ohlc_max_positive_gap | 5.68e-14 |
| adjusted_ohlc_violation_count_tolerance_1e_10 | 0 |
| nonfinite_core_count | 0 |
| source_missing_count | 0 |
| provider_missing_count | 0 |
| provider_adjustment_policy_missing_count | 0 |
| updated_at_missing_count | 0 |
| volume_null_count | 0 |
| volume_negative_count | 0 |

Strict adjusted OHLC 41건은 최대 gap이 `5.68e-14`인 부동소수점 반올림 수준이다. tolerance `1e-10` 기준 violation은 0이므로 contract PASS로 판정한다.

품질 gate 제외 row:

| 티커 | 제외일 | 사유 |
|---|---|---|
| AMD | 2015-01-02 | invalid_volume |
| AMD | 2016-04-22 | extreme_jump |
| FI | 2025-10-29 | extreme_jump, 최종 목록 제외 |

## 5. 1D Indicator 재계산

처음 100티커 단일 batch CLI 실행은 Supabase statement timeout으로 실패했다. 계산식 문제가 아니라 100개 장기 price row를 한 번에 읽는 fetch batch 문제였다.

대응은 기존 공식 `compute_indicators.run()` 경로를 유지하고 40/40/20 ticker batch로 나눠 실행했다.

| batch | ticker count | stored |
|---|---:|---:|
| 1 | 40 | 111,596 |
| 2 | 40 | 111,599 |
| 3 | 20 | 55,799 |
| 합계 | 100 | 278,994 |

최종 indicators 검증:

| 항목 | 결과 |
|---|---:|
| rows | 278,994 |
| tickers_with_rows | 100 |
| date_min | 2015-03-24 |
| date_max | 2026-05-01 |
| duplicate ticker/timeframe/date/source | 0 |
| provider_missing_count | 0 |
| core_nonfinite_count | 0 |
| atr_ratio_non_null | 278,994 |
| atr_ratio_coverage | 100.0% |

Ratio sanity:

| 컬럼 | abs p99 | abs max |
|---|---:|---:|
| open_ratio | 0.046799 | 0.552429 |
| high_ratio | 0.068455 | 0.612184 |
| low_ratio | 0.068121 | 0.550825 |
| atr_ratio | 0.068507 | 0.274956 |

p99는 안정적이다. max는 장기 history의 개별 이벤트 영향으로 보이며, NaN/Inf나 CP29 이전식 가격 피처 폭주 징후는 없다.

## 6. Feature Cache 검증

| 항목 | 결과 |
|---|---:|
| source_data_hash | 5be36437 |
| CP94 50티커 hash와 다름 | true |
| EODHD hash와 다름 | true |
| cache path 분리 | true |
| feature_contract_version | v3_adjusted_ohlc |
| index_rows | 278,994 |
| feature_df_rows | 278,956 |
| price_df_rows | 284,898 |
| feature_ticker_count | 100 |
| price_ticker_count | 100 |
| MODEL_N_FEATURES | 36 |
| atr_ratio_in_model_features | false |
| source_feature_nonfinite_count | 0 |
| model_feature_nonfinite_count_after_calendar | 0 |

horizon 5 raw future return 분포:

| 항목 | 값 |
|---|---:|
| nonfinite | 0 |
| p01 | -0.089637 |
| p50 | 0.002014 |
| p99 | 0.095140 |
| abs > 50% rate | 0.00002038 |

희소한 50% 초과 return은 남아 있지만 비율이 매우 낮다. 전체 universe 확대 전에는 같은 기준의 exception list를 계속 유지해야 한다.

## 7. Split Gate 결과

기본 `min_fold_samples=50` 기준:

| 설정 | eligible | train | val | test | excluded |
|---|---:|---:|---:|---:|---|
| seq_len 60, h5 | 97 | 181,297 | 38,801 | 38,983 | LMT, MS, T |
| seq_len 252, h5 | 97 | 168,285 | 36,014 | 36,158 | LMT, MS, T |

`LMT`, `MS`, `T`는 yfinance data 문제가 아니라 기존 `Gate fundamentals`로 제외됐다. 이번 CP는 가격 provider 전환 게이트이므로 FAIL로 보지는 않지만, 제품 학습 100 eligible을 원하면 fundamentals 보강 또는 해당 gate 정책 재검토가 필요하다.

## 8. Model Smoke 결과

공통 조건:

- W&B: `--no-wandb`
- save-run: 사용하지 않음
- device: cpu
- compile: off
- epochs: 1
- market_data_provider: yfinance
- product run 교체: 없음
- live inference 연결: 없음

### 8.1 CNN-LSTM Band Smoke

```powershell
python -m ai.train --model cnn_lstm --timeframe 1D --horizon 5 --seq-len 60 --feature-set price_volatility_volume --q-low 0.15 --q-high 0.85 --lambda-band 2.0 --band-mode direct --checkpoint-selection band_gate --market-data-provider yfinance --tickers <CP95 final 100 tickers> --epochs 1 --batch-size 64 --no-wandb --no-compile --device cpu --num-workers 0 --amp-dtype off --fp32-modules lstm,heads --local-log-dir logs\cp95_yfinance_validation
```

| 항목 | 값 |
|---|---:|
| run_id | cnn_lstm-1D-302d8632586d |
| exit code | 0 |
| status | completed |
| eligible_ticker_count | 97 |
| source_data_hash | 5be36437 |
| best band_gate_pass | true |
| val_total_loss | 0.025148 |
| test_total_loss | 0.012709 |
| test empirical_coverage | 0.689552 |
| test coverage_abs_error | 0.010448 |
| test lower_breach_rate | 0.162173 |
| test upper_breach_rate | 0.148275 |
| test band_width_ic | 0.372640 |
| test downside_width_ic | 0.065398 |
| elapsed seconds | 235.53 |

### 8.2 PatchTST Line Smoke

```powershell
python -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --patch-len 32 --patch-stride 16 --feature-set full_features --checkpoint-selection line_gate --market-data-provider yfinance --tickers <CP95 final 100 tickers> --epochs 1 --batch-size 64 --no-wandb --no-compile --device cpu --num-workers 0 --amp-dtype off --local-log-dir logs\cp95_yfinance_validation
```

| 항목 | 값 |
|---|---:|
| run_id | patchtst-1D-d6847d353e2a |
| exit code | 0 |
| status | completed |
| eligible_ticker_count | 97 |
| source_data_hash | 5be36437 |
| best line_gate_pass | true |
| val_total_loss | 0.022245 |
| test_total_loss | 0.025134 |
| test ic_mean | 0.032067 |
| test long_short_spread | 0.006177 |
| test false_safe_tail_rate | 0.346313 |
| test severe_downside_recall | 0.659790 |
| elapsed seconds | 2022.66 |

성능 수치는 1 epoch smoke이므로 product run 교체 근거로 사용하지 않는다. 이번 판정 기준은 yfinance source-aware 데이터가 모델 경로를 비우거나 NaN/Inf로 깨뜨리지 않는지였다.

## 9. 전환 판정

100티커 결과는 **PASS_WITH_NOTES**다.

통과 근거:

- 최종 100티커 모두 `source='yfinance'` price row 존재.
- adjusted OHLC sanity, duplicate, source/provider/policy, volume contract 통과.
- yfinance 1D indicators source-aware 재계산 통과.
- feature cache hash/path가 CP94 50티커 및 EODHD와 분리됨.
- MODEL_N_FEATURES=36 유지, atr_ratio 모델 feature 미포함.
- seq_len 60/252 split gate 통과.
- CNN-LSTM band와 PatchTST line smoke 모두 exit code 0.

주의 근거:

- `FI`, `MMC`는 yfinance primary 실패로 최종 목록에서 제외했다.
- 처음 100티커 indicator 단일 batch는 statement timeout이 발생해 40/40/20 배치가 필요했다.
- 모델 eligible은 100이 아니라 97이다. `LMT`, `MS`, `T`는 fundamentals gate에서 제외됐다.
- 전체 universe의 ticker mapping, delisted/short-history ticker, yfinance coverage 실패율은 아직 미검증이다.

## 10. EODHD 해지 가능성 판단

5월 18일 전 의사결정 관점에서, **1D 가격 primary를 yfinance로 전환할 가능성은 충분히 살아 있다.** 다만 CP95만으로 EODHD를 즉시 삭제하거나 완전히 해지하는 판단은 이르다.

권장 판단:

1. EODHD 코드는 유지한다.
2. yfinance 1D local primary 확대는 계속 진행한다.
3. 다음 CP에서 150~200티커 제한 write 또는 전체 universe dry-run coverage audit를 수행한다.
4. yfinance 실패 ticker 목록과 fallback artifact 정책을 확정한다.
5. 최소 local daily sync 리허설까지 통과한 뒤 EODHD 비용 축소 또는 해지 여부를 결정한다.

## 11. 다음 단계

권장 우선순위:

1. `FI`, `MMC` 같은 yfinance 실패 ticker의 대체/보류 정책을 공식화한다.
2. indicator 재계산 batch size를 운영 명령에 명시한다. 100티커 장기 full batch는 timeout 위험이 있다.
3. 150~200티커 제한 write로 coverage failure rate를 측정한다.
4. `LMT`, `MS`, `T` fundamentals gate 제외를 보강할지, 가격 provider 검증에서는 별도 분리할지 결정한다.
5. local daily sync dry-run: yfinance price sync, 1D indicator recompute, data quality check 순서 검증.

## 12. 금지 사항 준수

| 금지 항목 | 준수 |
|---|---|
| 전체 universe write 금지 | 준수 |
| EODHD row 삭제 금지 | 준수 |
| EODHD 복구 실행 금지 | 준수 |
| full retraining 금지 | 준수 |
| save-run 금지 | 준수 |
| live inference 금지 | 준수 |
| product run 교체 금지 | 준수 |
| 1W/1M 전체 재계산 금지 | 준수 |

## 13. 산출물

- `docs/cp95_yfinance_100ticker_long_history_validation_report.md`
- `docs/cp95_yfinance_100ticker_long_history_validation_metrics.json`
- `logs/cp95_yfinance_validation/`
