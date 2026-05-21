# CP112-LM 1W 보수적 예측선 스모크

## 요약

- 상태: `PASS` / 실패 분류: `None`
- run_id: `patchtst-1W-584cc2586b4a`
- 범위: PatchTST 1W h4 `line_model` smoke validation. 성능 채택 판단이 아니라 학습 파이프라인 작동 확인이다.
- `save-run`, DB write, inference 저장, W&B, band/composite/overlay 실험은 실행하지 않았다.

## 로컬 스냅샷 확인

- provider/source: `yfinance` / `yfinance`
- source_data_hash: `0f185b48`
- BM smoke hash와 동일: `True`
- feature_version: `v3_adjusted_ohlc`
- snapshot price 존재: `True`
- snapshot indicators 존재: `True`

## 데이터 게이트

- feature_set: `full_features`
- MODEL_FEATURE_COLUMNS: `36`
- resolved feature columns: `36`
- atr_ratio 모델 입력 포함: `False`
- feature NaN/Inf: `0`
- target NaN/Inf: `0`
- eligible ticker count: `97`

## split row 수

| split | rows | min | max |
| --- | --- | --- | --- |
| train | 26675 | 2018-02-09 | 2023-05-12 |
| val | 5723 | 2023-08-11 | 2024-09-20 |
| test | 5820 | 2024-12-20 | 2026-02-06 |

## 학습 실행

- exit code: `0`
- elapsed seconds: `62.5816`
- device/amp: `cuda` / `bf16`
- checkpoint_selection: `line_gate`
- line_gate pass: `True`
- gate_failed: `False`
- role: `line_model`
- h4 forecast shape pass: `True` / shapes: `{'line': [2, 4], 'lower_band': [2, 4], 'upper_band': [2, 4]}`

## line_metrics

| metric | value |
|---|---:|
| `spearman_ic` | -0.014605 |
| `ic_mean` | -0.014605 |
| `ic_std` | 0.205012 |
| `ic_ir` | -0.071242 |
| `ic_t_stat` | -0.551837 |
| `long_short_spread` | -0.007054 |
| `spread_mean` | -0.007054 |
| `spread_std` | 0.058405 |
| `spread_ir` | -0.120777 |
| `spread_t_stat` | -0.935536 |
| `direction_accuracy` | 0.450515 |
| `mae` | 0.171447 |
| `smape` | 1.606140 |
| `false_safe_negative_rate` | 0.228643 |
| `false_safe_tail_rate` | 0.239261 |
| `false_safe_severe_rate` | 0.239659 |
| `severe_downside_recall` | 0.760341 |
| `downside_capture_rate` | 0.224227 |
| `conservative_bias` | -0.124406 |
| `upside_sacrifice` | 0.231413 |
| `fee_adjusted_return` | -0.484434 |
| `fee_adjusted_sharpe` | -0.160782 |

## bucket line_metrics

| bucket | ic_mean | long_short_spread | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- |
| h1_h5 | -0.015356 | -0.003523 | 0.155000 | 0.833013 |

## 실패 분리

- split/data/shape 실패 여부: `None`
- line_metrics가 약하더라도 이번 CP에서는 성능 실패로 보지 않는다. 성능 평가는 후속 LM CP에서 별도로 해야 한다.
