# CP74-G W&B optional fallback 수리

## 1. 목적

CP72-BM full candidate training이 W&B API key 확인 단계에서 멈추는 문제를 막기 위해 `ai.train`의 W&B 초기화를 optional fallback 구조로 바꿨다. 이번 CP에서는 학습, save-run, DB 쓰기, 모델 구조 변경, 데이터/feature contract 변경, UI/backend 수정은 하지 않았다.

## 2. 변경 요약

`ai.train`에 `WandbInitOutcome`과 `init_wandb_run`, `init_wandb_for_training`을 추가했다. full training에서는 `config.use_wandb=True`라도 W&B 초기화가 실패하면 warning만 출력하고 `wandb_run=None`으로 계속 진행한다.

기록 상태는 다음 값 중 하나다.

| status | 의미 |
|---|---|
| `online_ok` | W&B online init 성공 |
| `package_missing` | `wandb` 패키지가 없음 |
| `disabled_by_env` | `WANDB_MODE=disabled` |
| `disabled_missing_key` | `WANDB_API_KEY`를 현재 환경과 `.env`에서 찾지 못함 |
| `disabled_init_failed` | key는 있으나 `wandb.init()`이 예외를 냄 |
| `disabled_by_cli` | `--no-wandb` 명시 |

API key 원문은 출력하지 않는다. `.env`에서 `WANDB_API_KEY`를 자동 로드할 수 있지만 key 값은 로그나 status에 남기지 않는다. `.env` 로드 실패도 학습 중단 사유가 아니다.

## 3. full training 정책

`ai.train`의 기본 `--wandb`는 유지했다. 단, full training에서는 W&B online init 실패가 학습 실패로 이어지지 않는다. `--no-wandb`는 기존처럼 완전 비활성이다.

train 결과 dict에는 `wandb_status`, `wandb_run_id`, `wandb_run_url`이 들어간다. `wandb_run_id`와 URL은 `online_ok`일 때만 기록한다. save-run 시에는 model_runs 컬럼을 추가하지 않고 config/meta JSON 내부에 `wandb_status`를 저장한다.

## 4. sweep 정책

`ai.sweep`은 full training fallback을 그대로 물려받지 않도록 분리했다. sweep에서 `--wandb`가 켜진 trial은 `run_training(..., wandb_required=True)`로 호출된다. 따라서 W&B가 필요한 sweep에서 init이 실패하면 trial 학습을 조용히 진행하지 않고 `W&B required for sweep, set WANDB_API_KEY or use --no-wandb` 메시지로 중단한다.

`WANDB_MODE=disabled`도 sweep의 `--wandb`를 조용히 꺼 버리지 않는다. sweep에서 W&B 없이 돌리려면 `--no-wandb`를 명시해야 한다.

## 5. 테스트

추가한 테스트는 실제 W&B online 호출 없이 mock으로만 검증한다.

- `WANDB_API_KEY` 없음 + `use_wandb=True` -> `disabled_missing_key`, 예외 없음, init 미호출
- `wandb` 패키지 없음 mock -> `package_missing`
- `WANDB_MODE=disabled` -> `disabled_by_env`
- `wandb.init` 예외 mock -> `disabled_init_failed`, secret 미노출
- `wandb.init` 성공 mock -> `online_ok`, run 반환
- required sweep mode -> 명확한 RuntimeError
- sweep objective -> `wandb_required=True` 전달

## 6. 검증 결과

- `py_compile ai/train.py ai/sweep.py ai/tests/test_wandb_optional_fallback.py` 통과
- `python -m unittest ai.tests.test_wandb_optional_fallback ai.tests.test_sweep` 통과

이번 CP에서는 실제 full training, save-run, DB 쓰기를 실행하지 않았다.
