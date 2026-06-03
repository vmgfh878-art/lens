# ADR 0010 — 품질 도구: ruff + pytest + coverage + mypy + pre-commit

- **상태**: 채택 (2026-06-03, CP222 Step 1~7)
- **컨텍스트**: CP222 직전 백엔드 품질 도구 0 상태. requirements.txt(3줄)/backend/requirements.txt(14줄)에 pytest/ruff/mypy 부재, pyproject.toml/conftest.py/pytest.ini 부재. 테스트 51 파일 / 397 함수가 unittest 스타일이라 pytest 호환은 가능하나 자동 회귀가 안 돈다.

## 결정 요지

| 항목 | 선택 | 근거 |
|---|---|---|
| Lint+Format | **ruff** (0.6.9) 단일 | flake8 + isort + pyupgrade + bugbear를 Rust 속도로 하나에. 설정 단순. |
| 룰셋 | `E, F, I, B, UP, SIM` | E/F=기본 오류, I=import 정렬, B=bugbear, UP=pyupgrade(3.10+), SIM=단순화. D/ANN 등은 초기 노이즈 회피. |
| line-length | `100` | 레거시 코드가 100 근처. 88(black)은 과도 변경 유발. |
| 테스트 러너 | **pytest** (8.3.3) | 기존 `unittest.TestCase` 397개를 변환 없이 흡수. plugin 생태계(cov, pandera 등). |
| pythonpath | `[".", "backend"]` | `app.*` (root=backend) + `backend.*`/`ai.*` (root=repo) 혼합 import 동시 해결. 한쪽만 넣으면 절반 ImportError. |
| 커버리지 | `pytest-cov` (5.0.0), branch+source=[backend/app, ai] | 표준 |
| 타입체크 | **mypy** (1.11.2) | 초기 느슨: `ignore_missing_imports=true`, `follow_imports=silent`, `namespace_packages=true`, `explicit_package_bases=true`. `mypy_path="."`. 후속 CP에서 모듈별 strict. |
| 자동수정 | **CP222에서 보류** → CP224b | `ruff check --fix` / `ruff format`은 동작 변경 위험. 분리해서 별도 CP에서 안전한 룰만 적용. |
| pre-commit | 3.8.0, hook install만 | `run --all-files` 비실행. 일괄 자동수정 금지 범위 보호. 다음 커밋부터 변경분에만 hook 적용. |

## 주요 결정 근거 — namespace package

- `ai/__init__.py` 부재(implicit namespace package). mypy 기본설정은 `blocks` / `ai.models.blocks` 두 이름으로 같은 파일 해석해 크래시 → `namespace_packages=true + explicit_package_bases=true` 조합으로 해결.
- 지시서는 "ai/__init__.py 임의 신설 금지"였고 이 설정 변경으로 **소스 0 변경**을 유지한 채 mypy를 통과.

## 대안과 거부 이유

- **flake8/black/isort 분리** → ruff 하나가 동일 기능 + 더 빠름. 거부.
- **Strict mypy 즉시 도입** → 1391 에러 발견됨. 초기 strict는 작업 마비. 모듈별 점진 도입으로.
- **`--strict` 룰셋 ruff** → D/ANN/ERA 등 추가 시 검출 폭증. baseline 단계엔 시그널 묻힘.
- **CI 즉시 도입** → CP237로 미룸. 안전망 도구가 먼저 안정화돼야 CI가 의미 있음.

## 결과 (CP222 baseline)

- pytest collected: **382** (passed 361 / failed 21 / collection error 1 / 17~20s)
- ruff 검출: **약 2155건** (E501 1718, E402 165, I001 117, F401 28, ... — 전부 baseline 기록만)
- mypy 에러: **1391 in 130 files (checked 251 sources)**
- 소스 코드 변경: **0** (인터페이스 보존)
- frontend tsc --noEmit: **PASS**

## 후속

- **CP223**: 백엔드 characterization snapshot (syrupy / snaptol 택1) — 채택: syrupy.
- **CP224a**: requirements 버전 핀 강화.
- **CP224b**: ruff 안전 자동수정 적용 (UP, I001, F401 위주).

## 보강 — v1 parquet 영구 가드 (2026-06-03 후행)

CP222 Step 5 직후 운영 v1 parquet 2개가 modified로 노출된 사건(런북 §0.8 위반 잠재) 1차 색출이 의심 모듈을 특정하지 못했다(재현 0/4 in 사후 검증). 사용자 결정으로 `backend/tests/conftest.py`에 `_guard_v1_parquet_integrity` session-scoped autouse fixture 추가:

- 세션 시작 SHA256 박제 → 세션 종료 재비교 → 변경 시 `git checkout -- <파일>` 즉시 복원 + `sys.__stderr__` 경고.
- 운영 코드 0 변경. conftest만.
- 가드 적용 후 검증: `pytest backend/tests` 87 passed (CP223 baseline 동일), git status backend/data/v1/ clean, 오염 0건.

이 가드는 "원인 모름 상태에서도 안전망 신뢰도 회복" 목적. CP223 9 snapshot baseline 입력 데이터 무결성 영구 보호.
