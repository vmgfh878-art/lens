# CP75-G local training progress logger 보고서

작성일: 2026-05-01

## 목표

W&B를 끈 full training에서도 epoch별 진행률과 주요 지표를 로컬 파일로 추적할 수 있도록 `ai/train.py`에 로컬 JSON logging 경로를 추가했다. 이번 CP에서는 모델 구조, 학습 수식, loss, checkpoint selector, DB schema, UI는 변경하지 않았다. 실제 full training도 실행하지 않았다.

## 변경 요약

- `ai/local_logging.py` 추가
  - `LocalTrainingProgressLogger` helper를 추가했다.
  - 기본 파일은 `metrics.jsonl`, `summary.json`, `config.json`이다.
  - numpy scalar, torch scalar/tensor, non-finite float를 JSON-safe 값으로 정리한다.
  - 파일 쓰기 실패는 warning만 출력하고 logger를 비활성화한다.

- `ai/train.py` 연결
  - 기본 local logging은 on이다.
  - 기본 경로는 `logs/runs/{run_id}/`이다.
  - CLI 옵션을 추가했다.
    - `--no-local-log`
    - `--local-log-dir logs/runs`
  - epoch마다 `metrics.jsonl`에 한 줄을 기록한다.
  - 종료 시 `summary.json`을 기록한다.
  - run 시작 시 `config.json`을 만들고, dataset plan이 준비된 뒤 한 번 더 최신 config를 저장한다.

## metrics.jsonl 필드

epoch별 local log에는 다음 필드를 기록한다.

| 필드 | 설명 |
|---|---|
| `run_id` | 학습 run id |
| `epoch`, `epochs` | 현재 epoch와 전체 epoch |
| `elapsed_seconds` | train loop 기준 누적 시간 |
| `epoch_seconds` | epoch 소요 시간 |
| `estimated_remaining_seconds` | 단순 epoch 기반 ETA |
| `learning_rate` | 현재 optimizer learning rate |
| `train_total_loss` | train total loss |
| `val_total_loss` | validation total loss |
| `val_forecast_loss` | validation forecast loss |
| `best_so_far` | early stopping 기준 best value |
| `epochs_since_improve` | 개선 없는 epoch 수 |
| `checkpoint_selection` | checkpoint selection 모드 |
| `selected_reason` | epoch 중에는 아직 `null` |
| `gate_status` | line/band/combined gate 통과 상태 |
| `vram_peak_allocated_mb` | CUDA peak allocated memory |
| `ic_mean` | line metric, 없으면 `null` |
| `long_short_spread` | line metric, 없으면 `null` |
| `false_safe_tail_rate` | line risk metric, 없으면 `null` |
| `severe_downside_recall` | line risk metric, 없으면 `null` |
| `nominal_coverage` | band metric, 없으면 `null` |
| `empirical_coverage` | band metric, 없으면 `null` |
| `coverage_abs_error` | band metric, 없으면 `null` |
| `asymmetric_interval_score` | band metric, 없으면 `null` |

## summary.json 필드

학습 종료 시 summary에는 다음을 기록한다.

- `run_id`
- `status`
- `model`
- `timeframe`
- `horizon`
- `feature_set`
- `source_data_hash`
- `checkpoint_path`
- `best_metrics`
- `test_metrics`
- `wandb_status`
- `total_elapsed_seconds`
- `error`

현재 `DatasetPlan`에는 source data hash가 직접 들어 있지 않아서 `source_data_hash`는 가능한 경우에만 채워지는 nullable 필드로 두었다. CP73에서 확인한 cache/hash manifest 개선과 같이 묶어 후속 정리가 필요하다.

## 비변경 확인

- W&B init/fallback 정책은 변경하지 않았다.
- loss 계산과 checkpoint selector 로직은 변경하지 않았다.
- DB schema와 save-run storage schema는 변경하지 않았다.
- UI/frontend는 변경하지 않았다.
- 모델 학습/추론은 실행하지 않았다.

## 검증

실행한 검증:

```powershell
python -m py_compile ai/local_logging.py ai/train.py ai/tests/test_local_logging.py
python -m unittest ai.tests.test_local_logging
```

검증 결과:

- py_compile 통과
- local logging helper 단위 테스트 4건 통과
- 임시 디렉터리에 `metrics.jsonl` 한 줄 쓰기 확인
- `summary.json` 쓰기 확인
- numpy/torch scalar 직렬화 확인
- local logging 실패 시 예외를 던지지 않고 warning만 남기는지 확인
- 실제 학습 실행 없음

## 후속 권장

1. `DatasetPlan` 또는 cache manifest에 `source_data_hash`를 명시적으로 싣는다.
2. full run wrapper에서 종료 후 `logs/runs/{run_id}/summary.json` 경로를 사용자에게 출력한다.
3. 장기 run 관측성 강화를 위해 DataLoader wait time, cache load time, split build time을 별도 local metric으로 추가한다.
