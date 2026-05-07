# WidthPenaltyLoss 제거 위치 + LossBreakdown 변경
`C:\Users\user\lens\ai\loss.py`에서 `WidthPenaltyLoss`를 제거했고 `ForecastCompositeLoss` total에서 width 항을 완전히 뺐다. `LossBreakdown`은 이제 `total`, `line`, `band`, `cross`만 가지며 `width_loss` 로그 키도 더 이상 나오지 않는다.

# NaN 감지 추가 위치
`C:\Users\user\lens\ai\train.py`의 `run_epoch()`에서 batch별 loss 계산 직후 `torch.isfinite(losses.total)`를 검사한다. NaN/Inf가 나오면 epoch, batch, trial/run 정보를 포함한 `RuntimeError`를 바로 던져서 sweep에서도 fail로 남게 했다.

# MAPE → MAE 교체 위치
`C:\Users\user\lens\ai\train.py`의 `_summarize_predictions()`에서 `MAPE`를 제거하고 `MAE`, `SMAPE`를 계산하도록 바꿨다. 평가 요약과 `evaluate_bundle()` 반환값도 `mae`, `smape` 기준으로 바뀌었다.

# 신규 테스트 3건 결과
추가한 테스트는 `test_width_loss_removed_from_composite`, `test_nan_loss_raises_runtime_error`, `test_mae_metric_finite_for_zero_targets` 3건이며 모두 통과했다. 기존 width loss 테스트는 cross loss 테스트로 바꿨다.

# CPU smoke 결과 + 출력 키 변화
`python -m ai.sweep --study-name smoke_cpu_tiny_cp95 --n-trials 1 --max-epoch 1 --model patchtst --timeframe 1D --limit-tickers 5 --seq-len 60 --device cpu --no-wandb`는 통과했다. 로그에서 `width_loss`는 사라졌고 `val/mae`, `val/smape`가 새로 들어왔으며 `dataset_build_seconds=15.3655`, 1 epoch 학습 시간은 `74.8258초`였다.

# 메모/잔여
이번 단계는 손실 함수 안정화와 디버깅성 회복이 목적이었고, 그 목표는 달성했다. 다만 `ai/inference.py`와 `ai/baselines.py` 쪽에는 아직 `mape`가 남아 있으므로, 지표 체계를 완전히 통일하려면 다음 단계에서 같이 정리하는 편이 맞다.
