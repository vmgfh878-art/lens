# CP139-LM 1W 보수적 예측선 v1 save-run 재현

## 요약

- 상태: `PASS`
- run_id: `patchtst-1W-fe7f05a84c93`
- display_name: `1W 보수적 예측선 v1`
- 1W line_model만 제품 후보로 저장했다.
- 1W band 저장, composite, inference 저장, 프론트 수정은 하지 않았다.

## 데이터 기준

- source_data_hash: `13a7f83d` / expected `13a7f83d`
- context_checksum: `ecb532122fca5eee` / expected `ecb532122fca5eee`
- feature_set: `price_volatility_volume`
- feature/target NaN/Inf: `0` / `0`
- eligible_ticker_count: `97`
- test_exposure_count: `23280`

## 저장 검증

- model_runs.status: `completed`
- role: `line_model`
- timeframe/horizon: `1W` / `4`
- feature_set: `price_volatility_volume`
- beta: `2` / config `2.0`
- checkpoint 존재: `True`
- line_gate_pass: `True`
- predictions delta: `0`
- prediction_evaluations delta: `0`
- band model row delta: `0`
- composite model row delta: `0`

## 주요 line_metrics

| split | ic_mean | long_short_spread | fee_adjusted_return | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- | --- |
| validation | 0.029400 | 0.005852 | 0.183052 | 0.079057 | 0.901293 |
| test | 0.011420 | 0.008156 | 0.320460 | 0.166667 | 0.828062 |

## 제품 메타데이터

- product_candidate: `True`
- product_layer: `line`
- line_only: `True`
- band_candidate: `False`
- composite: `False`
- UI note: 1W 화면은 line만 표시하고 AI 밴드는 준비 중/검증 중으로 둔다.

## 판정

- PASS: 1W line_model save-run 완료, checkpoint 존재, model_runs completed, 제품 후보 metadata 명확.
- inference 저장과 프론트 연결은 다음 CP로 넘긴다.
