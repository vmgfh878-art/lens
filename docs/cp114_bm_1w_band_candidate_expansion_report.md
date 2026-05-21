# CP114-BM 1W band 후보 확장 보고서

## 1. 결론

- 상태: PASS
- 실행 범위: CP113 기준 후보 재사용 1개, 신규 실험 4개 실행
- save-run / DB write / inference 저장 / W&B / composite: 모두 사용 안 함
- 제품 후보 제한 검증 통과:
  - `cnn_h4_q15_pvv_direct` 재사용 CP113 기준 후보
  - `cnn_h4_q10_pvv_direct`
  - `tide_h4_q15_pvv_param`
- watch:
  - `cnn_h4_q15_no_fundamentals_direct`: test는 좋지만 val band_gate 실패
- feasibility only:
  - `cnn_h8_q15_pvv_direct_feasibility`: 실행은 가능하지만 h8 하방 breach와 interval score가 약함

CP114 기준으로는 1W band를 제품 AI 밴드 후보군으로 올릴 근거가 생겼다. 다만 이번 CP에서는 저장 후보로 확정하지 않고, 다음 단계에서 seed 재현성과 save-run 후보 1개 선정이 필요하다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| python | `C:\Users\user\lens\.venv\Scripts\python.exe` |
| torch/CUDA | `2.11.0+cu128`, CUDA 사용 |
| GPU | `NVIDIA GeForce RTX 5060 Ti` |
| provider/source | `yfinance` |
| local snapshot | `LENS_USE_LOCAL_SNAPSHOTS=1`, `LENS_REQUIRE_LOCAL_SNAPSHOTS=1` |
| snapshot dir | `C:\Users\user\lens\data\parquet` |
| W&B | disabled, `--no-wandb` |
| save-run | false |

`cnn_h4_q15_pvv_direct`는 CP113과 동일 조건이라 재실행하지 않고 기준 후보로 재사용했다. 나머지 4개는 CP114에서 순차 실행했다.

## 3. 후보 결과표

| 후보 | 실행 | gate | 판정 | cov_abs | lower | upper | avg_w | p90_w | interval | bw_ic | down_ic | squeeze |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn_h4_q15_pvv_direct | CP113 재사용 | PASS | product_candidate_limited | 0.015765 | 0.135653 | 0.148582 | 0.196124 | 0.333164 | 0.317112 | 0.208742 | 0.020930 | 0.107388 |
| cnn_h4_q10_pvv_direct | 신규 | PASS | product_candidate_limited | 0.006014 | 0.080541 | 0.113445 | 0.196302 | 0.319798 | 0.301977 | 0.282361 | 0.024778 | 0.074527 |
| cnn_h4_q15_no_fundamentals_direct | 신규 | FAIL | watch_gate_failed | 0.040077 | 0.129768 | 0.130155 | 0.160372 | 0.272114 | 0.269365 | 0.225532 | 0.024248 | 0.096864 |
| tide_h4_q15_pvv_param | 신규 | PASS | product_candidate_limited | 0.042784 | 0.157045 | 0.185739 | 0.135569 | 0.174758 | 0.291406 | 0.326849 | 0.005363 | 0.057118 |
| cnn_h8_q15_pvv_direct_feasibility | 신규 | PASS | feasibility_only | 0.016774 | 0.197058 | 0.119716 | 0.219959 | 0.359916 | 0.419904 | 0.252135 | 0.012150 | 0.106207 |

## 4. CP113 대비 개선/악화

| 후보 | coverage_abs_error | interval_score | avg_width | band_width_ic | downside_width_ic | squeeze |
|---|---:|---:|---:|---:|---:|---:|
| cnn_h4_q10_pvv_direct | -0.009751 | -0.015134 | +0.000178 | +0.073618 | +0.003847 | -0.032861 |
| cnn_h4_q15_no_fundamentals_direct | +0.024313 | -0.047747 | -0.035752 | +0.016789 | +0.003318 | -0.010524 |
| tide_h4_q15_pvv_param | +0.027019 | -0.025706 | -0.060555 | +0.118106 | -0.015568 | -0.050270 |
| cnn_h8_q15_pvv_direct_feasibility | +0.001009 | +0.102793 | +0.023836 | +0.043393 | -0.008780 | -0.001181 |

해석:

- `cnn_h4_q10_pvv_direct`는 coverage, interval, dynamic width, squeeze가 모두 CP113보다 좋다.
- `tide_h4_q15_pvv_param`은 coverage_abs_error는 CP113보다 나빠졌지만 제품 후보 기준 0.05 이내이며, 폭이 크게 줄고 dynamic width가 가장 좋다.
- `no_fundamentals`는 test metric은 매력적이지만 val coverage_abs_error 0.105827로 band_gate fail이라 제품 후보로 올리지 않는다.
- h8은 coverage만 보면 가능하지만 interval score와 lower breach가 악화되어 장기 horizon 제품 후보로 보기 어렵다.

## 5. q10/q90 과보수 여부

`cnn_h4_q10_pvv_direct`는 nominal coverage가 0.8이다.

| 항목 | 값 |
|---|---:|
| test empirical - nominal | +0.006014 |
| test avg_width_delta vs CP113 | +0.000178 |
| test p90_width_delta vs CP113 | -0.013366 |
| val empirical - nominal | +0.074279 |
| val band_gate | PASS |

test 기준으로는 과보수라고 보기 어렵다. 폭은 CP113 q15/q85와 거의 같고 p90 폭은 더 낮다. 다만 val에서는 empirical coverage가 0.874279로 nominal 0.8보다 높아 과보수 성향이 남아 있으므로, q10/q90은 제품 저장 전에 seed 재현성 확인이 필요하다.

## 6. TiDE 대안성

`tide_h4_q15_pvv_param`은 1W band 대안으로 확인할 가치가 있다.

- band_gate PASS
- test coverage_abs_error 0.042784
- avg width 0.135569로 가장 좁음
- asymmetric_interval_score 0.291406으로 CP113보다 개선
- band_width_ic 0.326849로 후보 중 최고
- downside_width_ic 0.005363으로 양수지만 약함
- upper breach 0.185739가 다소 높아 tail 쏠림 모니터링 필요

TiDE는 CNN-LSTM q10보다 coverage calibration은 약하지만 폭과 dynamic width 면에서 강하다. 최소 1개 1W TiDE band 후보로는 살아남았다.

## 7. h1_h4 / h1_h8 bucket

- h4 실험은 evaluator가 `h1_h5_band_*`로 기록한다. 이번 horizon=4에서는 이를 h1_h4 bucket으로 해석했다.
- h8 실험은 all_horizon이 h1_h8이다. 추가로 evaluator가 `h1_h5_band_*`, `h6_h10_band_*`를 분리 기록했다.

h8 주요 bucket:

| bucket | cov_abs | interval | band_width_ic | downside_width_ic | squeeze |
|---|---:|---:|---:|---:|---:|
| h1_h8 all | 0.016774 | 0.419904 | 0.252135 | 0.012150 | 0.106207 |
| h1_h5 | 0.013918 | 0.348744 | 0.194179 | 0.027463 | 0.079210 |
| h6_h10 | 0.021535 | 0.538505 | 0.119715 | -0.012188 | 0.260882 |

h8은 후반 bucket에서 downside_width_ic가 음수이고 squeeze_breakout_rate가 높다. feasibility는 통과했지만 제품 후보로는 보류한다.

## 8. 판정

제품 band 후보 제한 검증 통과:

1. `cnn_h4_q10_pvv_direct`
   - 가장 균형이 좋다.
   - q10/q90인데 test 폭이 CP113 대비 거의 늘지 않았고 coverage 안정성이 좋아졌다.

2. `tide_h4_q15_pvv_param`
   - TiDE 대안 후보로 생존.
   - 폭과 dynamic width가 좋지만 upper breach와 낮은 downside_width_ic는 주의.

3. `cnn_h4_q15_pvv_direct`
   - CP113 기준 후보로 유지.
   - 새 후보 대비 1순위는 q10 또는 TiDE로 이동 가능하다.

Watch:

- `cnn_h4_q15_no_fundamentals_direct`
  - test는 매우 좋지만 val band_gate fail이라 저장 후보 제외.
  - feature removal 자체는 가능성이 있으나 gate 안정성이 먼저다.

Fail/product 보류:

- `cnn_h8_q15_pvv_direct_feasibility`
  - 실행 가능성은 확인.
  - h6~h8 쪽 하방 위험과 interval cost가 커서 제품 후보 제외.

## 9. 다음 제안

- CP115-BM으로 `cnn_h4_q10_pvv_direct`와 `tide_h4_q15_pvv_param`만 seed 재현성 검증을 권장한다.
- save-run 후보는 아직 바로 실행하지 말고, seed 2개 또는 3개에서 band_gate와 test tail imbalance가 유지되는지 먼저 본다.
- q10/q90은 test에서는 과보수 아님. 다만 val 과보수 경향이 있으니 seed 재현성에서 val empirical coverage를 반드시 본다.
- TiDE는 upper breach가 높아지는지 확인해야 한다.

## 10. 산출물

- `docs/cp114_bm_1w_band_candidate_expansion_report.md`
- `docs/cp114_bm_1w_band_candidate_expansion_metrics.json`
- `docs/cp114_bm_1w_band_candidate_expansion_logs/`
