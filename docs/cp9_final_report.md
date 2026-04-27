# DataLoader num_workers Windows 폴백
`C:\Users\user\lens\ai\train.py`의 `make_loader()`에서 worker 로더를 먼저 시도하고, `PermissionError` 또는 `OSError`가 나면 `num_workers=0`으로 자동 폴백하도록 바꿨다. `test_make_loader_windows_falls_back_to_zero_workers`로 실제 폴백과 경고 출력까지 검증했다.

# SequenceDataset 설계 + memory 비교
`C:\Users\user\lens\ai\preprocessing.py`에 `SequenceDataset`을 추가해 티커별 원본 배열 1회 보관 + `(ticker, end_idx)` 참조만 들고 가는 lazy 구조로 바꿨다. 기존 eager 구조는 503티커 1D에서 추정 메모리 약 25 GiB까지 커질 수 있었고, 현재 측정값은 5티커 `617.99 MB`, 50티커 `777.48 MB`였다.

# evaluate_bundle batch loop 변경
`C:\Users\user\lens\ai\train.py`의 `evaluate_bundle()`는 이제 split 전체를 한 번에 GPU로 올리지 않고 `make_loader()` 기반 batch loop로 평가한다. `evaluate_loader()`를 분리해 validation/test 공통 경로를 한 번만 타게 정리했다.

# infer_bundle batch loop 변경
`C:\Users\user\lens\ai\inference.py`의 `infer_bundle()`도 batch loop 기반으로 바꿨다. 예측/평가 레코드는 batch별로 누적 후 마지막에 합치도록 바꿔서 큰 split에서도 OOM 없이 돌 수 있게 했다.

# PatchTST --ci-target-fast 옵션
`C:\Users\user\lens\ai\models\patchtst.py`에 `ci_target_fast`를 추가했고, `ci_aggregate="target"`일 때는 target 채널 1개만 backbone에 넣는다. 속도는 좋아지지만 channel-independent weight-share 이점은 줄어드는 트레이드오프가 있어 기본값은 `False`로 유지했다.

# Future cov 조건부 변경
`C:\Users\user\lens\ai\preprocessing.py`의 dataset 빌드 경로에 `include_future_covariate`를 넣어 TiDE가 아닐 때는 미래 캘린더 공변량 자체를 만들지 않게 했다. PatchTST/CNN-LSTM sweep에서 불필요한 pandas 작업을 제거한 상태다.

# Feature cache fingerprint 알고리즘
캐시 키는 이제 `price_data MAX(date)`, `indicators MAX(date)`, `timeframe별 indicators COUNT(*)`를 해시한 `data_hash`를 포함한다. 파일명은 `features_{timeframe}_{feature_hash}_{data_hash}.pt`, `feature_index_{timeframe}_{feature_hash}_{data_hash}.pt` 형식이며, 기존 형식 캐시는 1회 경고 후 자동 무효화된다.

# sweep enable_compile=False 변경
`C:\Users\user\lens\ai\train.py`에 `enable_compile` 인자를 추가했고, `C:\Users\user\lens\ai\sweep.py`의 `objective_lr_sweep()`는 trial마다 `enable_compile=False`로 호출한다. 짧은 Optuna trial에서 `torch.compile` 오버헤드가 학습 시간보다 커지는 문제를 피하기 위한 변경이다.

# 신규 테스트 9건 결과
추가한 테스트는 9건이고 모두 통과했다. 범위는 Windows worker 폴백, lazy dataset 메모리/정합성, evaluation/inference batch loop, `ci_target_fast`, 미래 공변량 건너뛰기, 캐시 무효화, sweep compile 비활성 확인이다.

# CPU 작은 smoke 결과 (timing + RAM)
강제 CPU smoke 명령 `python -m ai.sweep --study-name smoke_cpu_tiny_device_cpu --n-trials 1 --max-epoch 1 --model patchtst --timeframe 1D --limit-tickers 5 --seq-len 60 --device cpu --no-wandb`는 통과했다. `dataset_build_seconds=7.0268`, 1 epoch 학습 시간 `73.7475초`였고, 별도 lazy build 측정은 5티커 `15.0469초 / 617.99 MB`, 50티커 `35.1872초 / 777.48 MB`였다. 50티커는 목표 30초에 약간 못 미친다.

# 사용자 GPU smoke 명령 (실행 안 함, 안내만)
사용자 GPU 환경에서는 아래 명령으로 확인하면 된다. `C:\Users\user\lens\.venv\Scripts\python.exe -m ai.sweep --study-name smoke_v4 --n-trials 3 --max-epoch 5 --model patchtst --timeframe 1D --limit-tickers 50 --no-wandb`

# 기존 회귀 통과 건수 (AI / backend)
전체 AI 테스트는 `83건` 통과했고, backend 회귀 테스트는 `23건` 통과했다. `.venv` 기준으로 다시 돌려 확인한 결과다.

# 메모/잔여
이번 단계에서 가장 큰 병목은 `forecast_dates` 문자열을 sample마다 반복 생성하던 비용이었고, 이를 미리 문자열 리스트로 바꿔 재사용하도록 줄였다. 다만 50티커 cold build 시간은 아직 30초를 약간 넘으므로, 다음 단계에서는 metadata 직렬화 비용이나 split 계획 생성 비용을 추가로 줄일 여지가 있다.
