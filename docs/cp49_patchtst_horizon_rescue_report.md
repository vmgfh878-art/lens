CP49-M은 PatchTST를 line_model 주력 후보로 유지하되, h5/h10/h20을 같은 후보로 섞지 않고 horizon별 line 가능성과 보수적 risk line 가능성을 분리 평가하는 CP다.

## 구현 상태

평가 함수에 보수적 line 지표와 horizon 구간별 지표를 추가했다. 모델 구조, loss, target, CNN-LSTM/TiDE는 변경하지 않았다.

새로 기록되는 일반 구간 지표:

| 구간 | 지표 |
|---|---|
| all_horizon | spearman_ic, long_short_spread, MAE, SMAPE |
| h1_h5 | spearman_ic, long_short_spread, MAE, SMAPE |
| h6_h10 | spearman_ic, long_short_spread, MAE, SMAPE |
| h11_h20 | spearman_ic, long_short_spread, MAE, SMAPE |

새로 기록되는 보수적 line 지표:

| 지표 | 해석 |
|---|---|
| overprediction_rate | `line > realized return` 비율. 낮을수록 하방 보수적 |
| mean_overprediction | 과대예측 구간의 평균 과대폭 |
| underprediction_rate | `line < realized return` 비율 |
| mean_underprediction | 과소예측 구간의 평균 signed error. 음수 |
| downside_capture_rate | 실제 하위 20% 구간에서 line도 하위 20%로 잡은 비율 |
| severe_downside_recall | severe downside를 line < 0 위험 신호로 잡은 비율 |
| false_safe_rate | 실제 하위 20% 또는 음수인데 line >= 0으로 본 비율 |
| conservative_bias | `mean(line - realized_return)`. 음수면 보수 편향 |
| upside_sacrifice | 실제 상위 20%에서 `actual - line` 평균. 높으면 기회비용 큼 |

severe downside 기준은 1D 기준으로 h1~h5는 5%, h6~h10은 8%, h11~h20은 12%로 고정했다.

## 실행 매트릭스

50티커 3epoch부터 시작한다. W&B는 metric only로 사용하고, save-run은 사용하지 않는다.

| horizon | geometry | seq_len | patch_len | stride |
|---:|---|---:|---:|---:|
| 5 | baseline | 252 | 16 | 8 |
| 5 | longer_context | 252 | 32 | 16 |
| 5 | dense_overlap | 252 | 16 | 4 |
| 10 | baseline | 252 | 16 | 8 |
| 10 | longer_context | 252 | 32 | 16 |
| 10 | dense_overlap | 252 | 16 | 4 |
| 20 | baseline | 252 | 16 | 8 |
| 20 | longer_context | 252 | 32 | 16 |
| 20 | dense_overlap | 252 | 16 | 4 |
| 20 | baseline_seq504 | 504 | 16 | 8 |

## VSCode 실행 명령

아래 명령은 사용자가 로컬 VSCode PowerShell에서 실행한다.

```powershell
cd C:\Users\user\lens
.\.venv\Scripts\Activate.ps1
$env:WANDB_MODE="online"

C:\Users\user\lens\.venv\Scripts\python.exe -m ai.cp49_patchtst_horizon_rescue `
  --phase matrix `
  --limit-tickers 50 `
  --epochs 3 `
  --batch-size 256 `
  --device cuda `
  --amp-dtype bf16 `
  --wandb-train `
  --wandb-project lens-cp49 `
  --output-json docs\cp49_patchtst_horizon_rescue_metrics.json `
  --log-dir logs\cp49
```

짧은 1개 smoke만 먼저 확인하려면 다음을 사용한다.

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.cp49_patchtst_horizon_rescue `
  --phase smoke `
  --limit-tickers 50 `
  --epochs 1 `
  --batch-size 256 `
  --device cuda `
  --amp-dtype bf16 `
  --output-json logs\cp49\cp49_smoke_metrics.json `
  --log-dir logs\cp49
```

## 판정 원칙

IC/spread가 좋으면 일반 line 후보로 본다. IC/spread가 약해도 `false_safe_rate`가 낮고 `downside_capture_rate` 또는 `severe_downside_recall`이 좋으면 보수적 risk line 후보로 보류한다.

IC가 음수이고 `false_safe_rate`도 높으면 탈락이다. h20 전체가 약해도 h1~h10이 살아 있으면 h10 후보로 전환한다. h11~h20만 망가지면 multi-horizon 표시를 분리한다.

## 실제 실행 결과

50티커 3epoch matrix는 완료됐다. 모든 후보 exit code 0이고, full 473티커와 save-run은 실행하지 않았다.

| 후보 | test IC | spread | all IC | h1~5 IC | h6~10 IC | h11~20 IC | false safe | severe recall | bias | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h5_baseline_seq252_p16_s8 | 0.0129 | 0.0017 | 0.0207 | 0.0207 |  |  | 0.2170 | 0.7710 | -0.0600 | 보류 |
| h5_longer_context_seq252_p32_s16 | 0.0244 | 0.0038 | 0.0190 | 0.0190 |  |  | 0.1490 | 0.8453 | -0.1092 | 생존 |
| h5_dense_overlap_seq252_p16_s4 | -0.0060 | -0.0024 | 0.0023 | 0.0023 |  |  | 0.0069 | 0.9895 | -0.1911 | risk-only 보류 |
| h10_baseline_seq252_p16_s8 | 0.0084 | 0.0010 | 0.0050 | 0.0119 | 0.0083 |  | 0.1879 | 0.8142 | -0.0577 | 약한 보류 |
| h10_longer_context_seq252_p32_s16 | -0.0085 | -0.0038 | 0.0134 | 0.0235 | 0.0104 |  | 0.2789 | 0.7108 | -0.0360 | 탈락 |
| h10_dense_overlap_seq252_p16_s4 | 0.0127 | -0.0003 | 0.0063 | 0.0216 | 0.0006 |  | 0.0991 | 0.9155 | -0.1077 | risk-only 보류 |
| h20_baseline_seq252_p16_s8 | -0.0063 | 0.0021 | 0.0212 | 0.0261 | 0.0215 | 0.0169 | 0.2665 | 0.7260 | -0.0444 | h20 구간 보류 |
| h20_longer_context_seq252_p32_s16 | 0.0190 | 0.0069 | 0.0219 | 0.0280 | 0.0219 | 0.0151 | 0.3156 | 0.6840 | -0.0334 | line 후보 보류 |
| h20_dense_overlap_seq252_p16_s4 | 0.0137 | 0.0016 | -0.0096 | 0.0059 | -0.0214 | -0.0065 | 0.1075 | 0.8913 | -0.0898 | risk-only 보류 |
| h20_baseline_seq504_p16_s8 | -0.0518 | -0.0258 | -0.0401 | -0.0041 | -0.0186 | -0.0429 | 0.0398 | 0.9516 | -0.1130 | 탈락 |

## W&B run mapping

| 후보 | W&B run |
|---|---|
| h5_baseline_seq252_p16_s8 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/n5nfvxy7 |
| h5_longer_context_seq252_p32_s16 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/9ujyrba4 |
| h5_dense_overlap_seq252_p16_s4 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/utyvn6ft |
| h10_baseline_seq252_p16_s8 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/vd2xdfsi |
| h10_longer_context_seq252_p32_s16 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/lbcn6ghl |
| h10_dense_overlap_seq252_p16_s4 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/xl91sqsq |
| h20_baseline_seq252_p16_s8 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/m12ft2mv |
| h20_longer_context_seq252_p32_s16 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/6nanbc8l |
| h20_dense_overlap_seq252_p16_s4 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/enayyqm0 |
| h20_baseline_seq504_p16_s8 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-cp49/runs/ik8i6omb |

## 해석

h5 longer_context 32/16이 일반 line 기준 최선이다. test IC 0.0244, long_short_spread 0.0038로 h5 baseline보다 좋고 false_safe_rate도 0.1490으로 낮아졌다. 단 conservative_bias -0.1092와 upside_sacrifice 0.1669로 상승 기회비용이 크다.

h20 longer_context 32/16은 final horizon 기준 IC 0.0190, spread 0.0069로 좋지만 false_safe_rate 0.3156이 너무 높다. h20을 제품 기본 line으로 바로 올리기는 위험하다. 다만 h1~5, h6~10, h11~20 구간 IC가 모두 양수라 h20 branch를 폐기할 근거는 없다.

dense overlap 계열은 보수적 risk line으로는 흥미롭지만 일반 line으로는 약하다. h5 dense는 false_safe_rate 0.0069, severe_downside_recall 0.9895지만 IC와 spread가 음수다. h20 seq504도 비슷하게 매우 보수적이지만 IC/spread가 크게 음수라 탈락이다.

## 결론

제품 기본 line 후보는 h5 longer_context 32/16으로 올린다. h10은 baseline만 약한 보류이고, h20은 Phase 1.5 branch로 유지한다. h20에서 h11~h20만 완전히 무너진 것은 아니지만 false safe가 높아 리스크 보조 없이 바로 쓰면 안 된다.

다음 CP에서는 h5 longer_context 32/16을 100티커로 재확인하거나, h20 longer_context를 false_safe_rate 낮추는 checkpoint selector로 재평가한다.
