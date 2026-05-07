# CP66-LM post-backfill cache gate + h20 / 1W line 재진입

## 1. 원칙 확인
- 이번 CP는 line_model만 다뤘고 band 모델 실험, composite/overlay, line_inside_band 평가는 사용하지 않았다.
- DB 쓰기는 하지 않았고, cache stale 확인 후 필요한 학습 cache만 DB 읽기로 생성했다.
- save-run=false, W&B off, no-compile 조건으로 실행했다.
- full 473티커 학습은 실행하지 않았다. 1D h20 학습은 limit_tickers=50 범위다.

## 2. post-backfill cache gate
| 항목 | 값 | 판정 |
| --- | --- | --- |
| feature_version | v3_adjusted_ohlc | PASS |
| source data hash | 0c1d7f52 -> f7c7b101 | stale |
| new cache | True | created |
| cache refresh note | CP66 초회 gate에서 새 post-backfill cache를 생성했고, 최종 GPU 실행에서는 재사용했다. |  |
| MODEL_FEATURE_COLUMNS | 36 | PASS |
| atr_ratio model input | False | PASS |
| feature tensor NaN/Inf | 0 | PASS |
| target NaN/Inf | 0 | PASS |
| ratio p99 sanity | True | PASS |

feature_version은 `v3_adjusted_ohlc`로 유지됐고, 바뀐 것은 source data hash/cache hash다. `atr_ratio`는 indicators에는 존재할 수 있으나 MODEL_FEATURE_COLUMNS 36개에는 포함되지 않는다.

## 3. 1D h20 line 결과
| name | feature_set | status | ic | ic_ir | spread | spread_ir | false_safe_tail | false_safe_severe | severe_recall | dir | mae | smape | fee_ret | fee_sharpe | d_recall | d_fstail | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h20_full_features_post_backfill_seq252_p32_s16 | full_features | completed | 0.0230 | 0.0960 | 0.0067 | 0.0816 | 0.2895 | 0.2880 | 0.7120 | 0.4769 | 0.0762 | 1.4876 | 1.9491 | 0.0763 | 0.0253 | -0.0254 | visual_risk_survive |
| h20_no_fundamentals_post_backfill_seq252_p32_s16 | no_fundamentals | completed | 0.0214 | 0.0859 | 0.0136 | 0.1797 | 0.3361 | 0.3353 | 0.6647 | 0.4862 | 0.0709 | 1.4781 | 45.5232 | 0.1743 | -0.0221 | 0.0211 | watch_default_display_forbidden |
| h20_technical_only_post_backfill_seq252_p32_s16 | technical_only | completed | 0.0222 | 0.0965 | 0.0063 | 0.0785 | 0.3237 | 0.3217 | 0.6783 | 0.4768 | 0.0725 | 1.4796 | 1.7153 | 0.0736 | -0.0085 | 0.0088 | watch_default_display_forbidden |

## 4. 비교 기준
### CP65 h20
| feature_set | name | ic | spread | false_safe_tail | severe_recall |
| --- | --- | --- | --- | --- | --- |
| technical_only | h20_technical_only_seq252_p32_s16 | 0.0137 | 0.0062 | 0.3375 | 0.6645 |
| no_fundamentals | h20_no_fundamentals_seq252_p32_s16 | 0.0230 | 0.0147 | 0.3015 | 0.6993 |
| price_volatility_volume | h20_price_volatility_volume_seq252_p32_s16 | 0.0137 | 0.0062 | 0.3375 | 0.6645 |
| full_features | h20_longer_context_seq252_p32_s16 | 0.0199 | 0.0072 | 0.3149 | 0.6868 |

### statistical baseline
| name | ic | spread | false_safe_tail | severe_recall |
| --- | --- | --- | --- | --- |
| historical_mean_line_w60 | 0.0211 | 0.0031 | 0.5541 | 0.4495 |
| reversal_line_horizon | 0.0120 | 0.0046 | 0.4860 | 0.5085 |
| random_or_shuffled_score | 0.0213 | 0.0019 | 0.5113 | 0.4989 |

## 5. feature_set별 개선 여부
- full_features: severe_recall delta=0.0253, false_safe_tail delta=-0.0254, verdict=visual_risk_survive
- no_fundamentals: severe_recall delta=-0.0221, false_safe_tail delta=0.0211, verdict=watch_default_display_forbidden
- technical_only: severe_recall delta=-0.0085, false_safe_tail delta=0.0088, verdict=watch_default_display_forbidden

## 6. h20 제품 표시 제안
- h20 최우선 관찰 후보는 `h20_full_features_post_backfill_seq252_p32_s16`이다.
- h5는 단기 line으로 유지하고, h20은 중기 line / 위험 맥락 branch로 분리한다.
- h20은 제품 기본 굵은 선이 아니라 점선 또는 낮은 신뢰도 표시를 권장한다. false_safe_tail_rate가 0.30 이상인 후보는 기본 표시 금지다.

## 7. 1W readiness
| 항목 | 값 |
| --- | --- |
| status | PASS |
| eligible_ticker_count | 447 |
| feature NaN/Inf | 0 |
| target NaN/Inf | 0 |
| split gap h_max=12 | True |
| future label leakage blocker | False |
| stale ticker registry full/smoke | True / False |

누수 blocker 없음: 1W 날짜는 W-FRI period label이고 forecast_dates는 asof_date 이후 예측 대상 기간이다.

## 8. 1W smoke
- 실행 여부: 실행, status=completed, log=docs\cp66_lm_post_backfill_h20_1w_logs\1w_h12_full_features_readiness_smoke_seq104_p16_s8.log
- line metrics: ic_mean=-0.0144, spread=0.0067, false_safe_tail=0.3424, severe_recall=0.6521

## 9. 다음 CP 추천
- 1D h20은 이번 post-backfill hash에서 false_safe_tail과 severe_downside_recall이 같이 개선되는 feature_set만 100티커로 한 단계 확장한다.
- h20 제품 노출은 기본선이 아니라 h5 보조 중기 점선으로 분리하고, false_safe_tail_rate 0.30 이상이면 UI 기본 표시는 막는다.
- 1W는 readiness PASS와 1epoch smoke 결과를 기준으로 seq_len=104 유지 여부와 h12 line risk-only 지표를 다음 CP에서 50->100티커로 확인한다.

## 10. 검증
- `.venv\Scripts\python.exe -m py_compile ai\cp66_lm_post_backfill_h20_1w.py ai\train.py ai\evaluation.py ai\tests\test_feature_set_selection.py ai\tests\test_metric_definition_contract.py`: 통과
- `python -m json.tool docs\cp66_lm_post_backfill_h20_1w_metrics.json`: 통과
- `.venv\Scripts\python.exe -m unittest ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 27개 통과
- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음
