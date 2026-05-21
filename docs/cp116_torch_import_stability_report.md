# CP116-S Windows Torch DLL import 안정화 보고서

## 1. 원인

Windows에서 `pandas`/`numpy`가 먼저 로드된 뒤 CUDA 포함 `torch`가 뒤늦게 로드되면 native DLL 초기화 순서가 꼬여 `c10.dll` 초기화 실패가 발생할 수 있다. 특히 CP runner가 데이터 확인과 모델 inference를 한 프로세스에 섞을 때 재현 가능성이 높았다.

## 2. 재현 조건

위험 조건은 Windows venv 프로세스에서 `pandas` 또는 `numpy`를 먼저 import하고, 이후 `ai.train`, `ai.inference`, `ai.preprocessing`, 모델 모듈, checkpoint inference helper를 통해 `torch`를 간접 import하는 흐름이다. pandas-first 실패 재현은 환경 의존 native crash라 강제 unittest로 넣지 않고, 보고서에 조건만 남겼다.

## 3. 적용한 import 순서 계약

`ai/torch_bootstrap.py`를 추가하고 torch 사용 entrypoint는 `bootstrap_torch()`를 통해 `pandas`/`numpy`보다 먼저 torch를 로드하도록 통일했다. 공통 환경변수는 `PYTHONUTF8=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `TORCHDYNAMO_DISABLE=1`, `PYTORCH_NVML_BASED_CUDA_CHECK=1`이다.

CUDA 학습 스크립트에는 `CUDA_VISIBLE_DEVICES=-1`을 강제하지 않는다. CP99/CP116처럼 제품 리허설에서 CPU-only inference가 명확한 경우에만 `bootstrap_torch(cpu_only=True)`를 사용한다.

## 4. 수정 범위

torch 사용 경로는 `ai.train`, `ai.inference`, `ai.evaluation`, `ai.band_calibration`, `ai.composite_inference`, `ai.baselines`, `ai.benchmark_gpu`, 주요 CP runner에서 bootstrap을 먼저 호출하도록 정리했다. `ai.preprocessing`도 model dataset 경로에서 torch를 쓰므로 bootstrap 계약에 편입했다.

CP101은 data-only import 경로가 CP99 inference helper를 top-level에서 끌고 오지 않도록 lazy wrapper로 바꿨다. 그 결과 CP103이 CP101을 import해도 `torch`와 `scripts.cp99_1d_product_loop_thin_upload`는 import되지 않는다.

## 5. data-only/model script 분리 원칙

CP103/CP115 같은 data-only runner는 import 시점에 torch를 로드하지 않는다. 데이터 확인과 모델 inference가 모두 필요한 제품 리허설은 장기적으로 한 프로세스에서 `pandas -> torch`로 이어지지 않게 data process와 model process를 subprocess로 분리하는 것을 권장한다.

CP99/CP116은 현재 CPU-only model helper를 쓰는 mixed runner라 bootstrap을 먼저 호출하도록 방어했다. 이 경로는 안전장치이지, data-only script로 분류하지 않는다.

## 6. 검증 결과

| 검증 | 결과 |
|---|---|
| `py_compile` 수정 파일 | PASS |
| torch-first 최소 import | PASS, `torch 2.11.0+cu128` |
| `ai.train` / `ai.inference` import | PASS |
| CP103 import 후 torch 미로드 | PASS, `cp103_torch_loaded False` |
| CP115 import 후 torch 미로드 | PASS, `cp115_torch_loaded False` |
| `ai.tests.test_torch_import_contract` | PASS, 4 tests |
| `ai.tests.test_evaluation_targets` | PASS |
| `ai.tests.test_metric_definition_contract` | PASS |
| `ai.tests.test_checkpoint_selection` | PASS |

## 7. 추가 테스트

`ai/tests/test_torch_import_contract.py`를 추가했다. 테스트는 bootstrap 환경변수, `ai.train`/`ai.inference` import, CP103/CP115 data-only import 경로의 torch 미로드, CP103 import 시 CP99 helper 미로드를 검증한다.

## 8. 남은 주의점

`ai.models`, `ai.loss`, `ai.postprocess`, `ai.finite` 같은 하위 torch 모듈은 library module이라 직접 `import torch`를 유지한다. 제품/학습 entrypoint에서 bootstrap을 먼저 통과한다는 전제로 안전하다. 임의 스크립트에서 `pandas`를 먼저 import한 뒤 하위 모델 모듈을 직접 import하는 패턴은 계속 금지한다.

CP98 같은 오래된 snapshot/model dataset 혼합 runner는 모델 dataset helper를 쓰므로 data-only로 보기 어렵다. 이후 제품 리허설을 더 안정화하려면 CP101/CP103 runtime inference 단계와 CP98 계열도 subprocess split 후보로 분리하는 것이 좋다.
