# CP75-LM 1D h5 line full product candidate training

## 요약

- 후보 run_id: `patchtst-1D-efad3c29d803`
- 저장 상태: `completed` / 판단: `completed_line_watch`
- 범위: PatchTST 1D h5 line_model 전용. band/composite/overlay 실험은 실행하지 않았다.
- 제품 산출물은 `line_series`이며, line_model의 lower/upper 출력은 제품 band 후보로 쓰지 않는다.

## 사전 게이트

- source_data_hash: `3ac43945`
- CP72 BM source_data_hash 동일 여부: `True`
- feature_version: `v3_adjusted_ohlc`
- feature_set: `full_features`
- MODEL_FEATURE_COLUMNS: `36`
- full_features resolved columns: `36`
- atr_ratio 모델 입력 포함: `False`
- feature NaN/Inf: `0`
- target NaN/Inf: `0`
- open/high/low_ratio sanity: `True`
- eligible ticker count: `473`
- 학습 샘플 train/val/test: `798950` / `170960` / `171781`

## 실행

- python: `C:\Users\user\lens\.venv\Scripts\python.exe`
- device: `cuda` / amp_dtype: `bf16` / compile_model: `False`
- epochs: `5` / batch_size: `256` / num_workers: `0`
- 실행 전 예상: `70-140`분, `3-6`GB VRAM
- local log dir: `C:\Users\user\lens\docs\cp75_lm_1d_h5_full_line_product_candidate_logs\patchtst-1D-efad3c29d803`
- epoch_count: `5` / peak VRAM MB: `3030.66`
- W&B status: `disabled_by_cli`

## 저장 확인

- model_runs row 존재: `True`
- role/config: `config.role=line_model`
- checkpoint 존재: `True`
- predictions 저장 수: `171781` / 확인 방식: `inference_stdout_count_after_sample_check`
- prediction_evaluations 저장 수: `171781` / 확인 방식: `inference_stdout_count`
- predictions 샘플 line_series 확인: `True`

## h5 line 결과

| metric | value |
|---|---:|
| `ic_mean` | 0.040583 |
| `ic_ir` | 0.208004 |
| `ic_t_stat` | 4.038697 |
| `long_short_spread` | 0.006300 |
| `spread_ir` | 0.191630 |
| `spread_t_stat` | 3.730643 |
| `direction_accuracy` | 0.502710 |
| `false_safe_tail_rate` | 0.393146 |
| `false_safe_severe_rate` | 0.394211 |
| `severe_downside_recall` | 0.605789 |
| `conservative_bias` | -0.011932 |
| `upside_sacrifice` | 0.058494 |
| `mae` | 0.048700 |
| `smape` | 1.485538 |
| `fee_adjusted_return` | 6.127765 |
| `fee_adjusted_sharpe` | 0.174862 |

## 비교 기준

- CP49/CP53 h5 longer_context 참조 후보 수: `8`
- 통계 baseline 참조 수: `0`
- 참조 후보는 기존 문서의 line_metrics만 요약했다. band/composite 지표는 ranking에 쓰지 않았다.

## 제품 판단

- h5 line 제품 기본 후보 판단: `completed_line_watch`
- CP72 BM과 결합 저장하지 않았고, 이번 CP의 저장 run은 line_model 단독 후보로 남겼다.
- 다음 LM 추천: 저장 run의 실서비스 노출 전, 동일 해시에서 추론 latency와 최신 asof_date 샘플 line_series sanity만 별도 smoke로 확인한다.
