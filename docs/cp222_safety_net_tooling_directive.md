# CP222 BE/FE 공통 안전망 도구 구축 (Directive)

> 이 문서는 런북 `docs/cp221_237_refactoring_runbook.md` 가 자동으로 꺼내 실행하는 단일 지시서다.
> 실행 세션은 이 문서만 읽고 정확히 도구를 설치·설정하고, 기존 테스트 수집/실행을 복구하고, 차단 트리거를 판단한다.
> CP222 는 리팩토링 전체의 **0번 안전망**이다. 여기서 pytest/ruff/mypy 가 동작해야 CP223(백엔드 스냅샷) 이후가 성립한다.

---

## 역할 고정

- **모드**: `code` (구현 모드). 설계 토론·계획 금지, 지시대로 구현하고 자가 점검만 보고.
- **권한**: 코드/설정 파일 수정, 로컬 검증(설치·pytest·ruff·mypy 실행)만.
- **금지**:
  - 새 학습(training) 실행 금지.
  - 새 calibration 실행 금지.
  - DB write 금지 (Supabase upsert/insert/delete 일절 금지).
  - Supabase 호출 금지 (네트워크로 실제 Supabase 에 붙는 동작 금지. 테스트는 전부 mock 기반이어야 함).
  - 사용자가 직접 수정한 파일 revert 금지.
  - **이 CP에서 `ruff check --fix` 일괄 적용 금지** (동작 변경 위험 → 안전한 자동수정은 CP224b 에서만).
  - 기존 테스트 코드(`backend/tests/`, `ai/tests/`)의 **로직 수정 금지**. CP222 는 도구만 깐다. 테스트가 깨지면 고치지 말고 **기록 후 보고**.
- **자가점검**: 작업 끝에 [Plan v3 정합] / [구조 결함] / [모델 영향] 3축 PASS·WARN·FAIL 보고 (양식 하단).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python **3.10.0**, torch **2.11.0+cu128** — 사용자 환경 핀). venv 활성화: `.\.venv\Scripts\Activate.ps1`
  - 주의: 셸 기본 인터프리터는 시스템 `Python310`(`C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe`)일 수 있다. **반드시 `.venv` 안에 설치**해야 한다. 설치/실행 시 `.\.venv\Scripts\python.exe -m ...` 형태로 인터프리터를 명시하라.
- **백엔드 기동**(검증에 필요할 때만): `scripts\start_demo.ps1` 은 **존재하지 않는다**(확인됨 — `scripts/` 에는 `wandb_status.py` 와 `diagnostics/` 뿐). 실제 기동은 README 기준:
  ```powershell
  # backend/ 디렉토리에서
  .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
  ```
  CP222 는 **기동 불필요**(도구 설치/테스트 수집만). 기동 명령은 참고용.
- **프론트**: `cd frontend; npm run dev`. CP222 는 프론트 도구 변경 없음(Sub-step 6 참고) → 기동 불필요.
- **포트 충돌 회피**: 만약 기동이 필요하면 8000 대신 8010 등 비표준 포트 사용. CP222 에선 해당 없음.

---

## 진단 (근거)

조사 출처: 2026-06-02 직접 Read/Grep/Glob + `git ls-files` + 인터프리터 probe.

**1) 품질 도구가 0 상태다.**
- `requirements.txt`(루트, 3줄): `optuna`, `optuna-dashboard`, `matplotlib` 뿐. pytest/ruff/mypy 없음.
- `backend/requirements.txt`(14줄): fastapi·uvicorn·supabase·pandas·httpx·wandb 등. pytest/ruff/mypy 없음.
- 설정 파일 부재 확인(Glob 결과 `No files found`): `pyproject.toml` 없음 / `pytest.ini` 없음 / `.pre-commit-config.yaml` 없음 / `ruff.toml` 없음 / `conftest.py`(루트·서브) 없음.
- 인터프리터 probe: 셸 `python` → 시스템 3.10.0. `.venv` 는 존재(`/.venv/Scripts/Activate.ps1`, `python.exe` 확인). 즉 venv 는 있으나 품질 도구가 안 깔림.

**2) 기존 테스트는 unittest 스타일이라 pytest 가 그대로 수집 가능하다.**
- 모든 테스트가 `import unittest` + `class XxxTestCase(unittest.TestCase)` + `if __name__ == "__main__": unittest.main()` 패턴.
  - 예: `backend/tests/test_api.py:1-12,527-528` (`import unittest` / `class ApiTestCase(unittest.TestCase)` / `unittest.main()`), 현재 **528줄**.
  - 예: `ai/tests/test_loss.py:1-8,29-30` (`from ai.loss import AsymmetricBCELoss, PinballLoss` / `class PinballAndBCELossTestCase(unittest.TestCase)`), 현재 **30줄**.
- pytest 는 `unittest.TestCase` 를 네이티브로 수집·실행한다 → 테스트 코드 변환 불필요.

**3) 실제 테스트 규모는 스펙의 "31개" 가 아니라 훨씬 크다(수집 baseline 을 새로 박아야 함).**
- 파일: `backend/tests/` **9개** + `ai/tests/` **42개** = **51개**(각 `__init__.py` 제외).
- `def test_` 함수 합계: backend **104** + ai **293** = **397개**(Grep 카운트).
- ⇒ 런북/스펙에 적힌 "기존 31개 테스트" 는 오래된 수치. **이 CP의 산출물에 실제 수집/통과 수를 baseline 으로 기록**하라(아래 Sub-step 3).

**4) import 루트가 섞여 있어 pythonpath 설정이 핵심이다(가장 큰 함정).**
- `app.*` top-level import(루트가 `backend/` 여야 함):
  `test_api.py:7-9`(`from app.core.exceptions ...` / `from app.db ...` / `from app.main import app`), 그리고 `test_services.py`, `test_cp209_admin_rebuild_contracts.py`, `test_product_prediction_history_api.py`.
  - 패키지 실체: `app` 은 `backend/app/`(`backend/app/main.py`, `backend/app/__init__.py` 존재).
- `backend.*` prefix import(루트가 repo 루트여야 함):
  `test_collector_jobs.py:20`(`from backend.collector.jobs...`), `test_feature_svc.py`, `test_market_data_providers.py`, `test_db_bootstrap.py`, `test_cp151_yfinance_500_backfill.py` 등. 이들 다수는 파일 안에서 **직접 `sys.path.insert(0, ROOT_DIR)`** 한다(예: `test_collector_jobs.py:9-12`).
- `ai.*` import(루트가 repo 루트여야 함):
  `test_loss.py:5`(`from ai.loss import ...`) 등 ai 테스트 대부분. **단 `ai/__init__.py` 는 없다**(Glob `No files found`) → `ai` 는 implicit namespace package 로 잡힌다. repo 루트가 path 에 있으면 import 된다.
- ⇒ **pytest `pythonpath` 에 루트(`.`)와 `backend` 둘 다** 넣어야 `app.*` 와 `backend.*`/`ai.*` 가 동시에 풀린다. 한쪽만 넣으면 절반이 ImportError.

**5) 일부 테스트는 gitignore 된 파일을 import 한다 → 수집 단계에서 깨질 수 있다(예상된 실패).**
- `.gitignore:20-23` 가 `ai/cp*.py`, `ai/tests/test_cp*.py`, `backend/tests/test_cp*.py`, `scripts/cp*.py` 를 무시.
- `git ls-files` 기준 추적 테스트 **44개** vs 디스크 **51개**. 즉 `test_cp*` **12개**가 untracked(로컬 디스크엔 있음). pytest 는 디스크에서 수집하므로 **51개 전부 수집 시도**.
- 결정적: `test_collector_jobs.py:20` 이 `from scripts.cp134_local_daily_update_rehearsal import (...)` 하는데 **`scripts/cp134_local_daily_update_rehearsal.py` 는 존재하지 않는다**(Glob `No files found`, 게다가 `scripts/cp*.py` 는 ignore). ⇒ 이 파일은 **collection 단계 `ModuleNotFoundError` 로 깨진다**. 이건 코드 자체 결함이 아니라 "로컬에만 있던 보조 스크립트 부재" 이므로, **수집 에러로 기록**하고 차단 트리거(코드 결함)와 **구분**하라.

**결론**: CP222 는 도구만 추가하는 additive 작업이지만, (4) pythonpath 와 (5) 수집 에러를 정확히 다루지 않으면 "테스트가 다 깨진 것처럼" 보인다. baseline 을 정직하게 박는 것이 이 CP의 본질이다.

---

## 선행 의존

**없음.** CP222 는 CP221(라이브 버그픽스, 완료) 직후의 첫 리팩토링 CP이며 안전망의 0번이다. 단, 런북 § "리팩토링 시작 전 커밋" 지침대로 **작업 트리에 섞여 있는 미커밋 변경분이 있으면 먼저 분리 커밋**되어 있어야 깔끔하다. CP222 자체는 다른 CP의 그린을 요구하지 않는다.

---

## 범위

**포함**
- 설정 파일 신규 작성: `pyproject.toml`(ruff + pytest + coverage + mypy 섹션), `.pre-commit-config.yaml`, `requirements-dev.txt`.
- `.venv` 에 dev 도구 설치(pytest, pytest-cov, ruff, mypy, httpx, pandera, 스냅샷 후보).
- 기존 51개 테스트의 pytest 수집/실행 결과 측정 및 **baseline 기록**.
- ruff **검출만**(`ruff check`, `--fix` 없음) 1회 실행해 풍경 파악.
- mypy 1회 실행해 동작 확인(에러 수는 baseline 기록만, 수정 안 함).
- pre-commit 훅 설치(`pre-commit install`).
- ADR 1장 작성(`docs/adr/0010-...`), 보고서 1장(`docs/cp222_report.md`).

**제외(이 CP에서 절대 안 함)**
- `ruff check --fix` / `ruff format` 로 **코드 일괄 변경**(→ CP224b).
- mypy 에러 **수정**(→ 후속, 모듈별 strict 도입 시).
- 테스트 **로직 수정**, 깨진 수집 에러 **소스 수정**(기록만).
- 프론트엔드 도구(ESLint/Prettier/vitest 등) **도입**(CP230 트랙. CP222 의 FE 몫은 "현 상태 확인 + tsc 동작 확인" 까지만, Sub-step 6).
- Supabase 관련 일체(보류). 테스트는 mock 으로만 돌린다.
- CI(GitHub Actions) 워크플로 추가(후속).
- `mypy --strict`, 추가 ruff 룰(D, ANN 등) 도입(초기엔 느슨).

---

## Sub-step (Strangler Fig, 작은 단위)

> 원칙: 각 Step = 한 revert 단위. additive 라 "옛 코드 제거" 단계는 없지만, **각 Step 끝에 commit + 검증**을 둔다. 설정 파일이라 순수→I/O→상태 추출 순서는 N/A.

### Step 1 — 설정 파일 작성 (코드 변경 0, 설정만)
1. `pyproject.toml`(루트) 신규 작성. 아래 4블록 포함:

   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py310"
   # 초기엔 디렉토리만 본다. cp* 실험/캐시/아티팩트 제외.
   extend-exclude = [".venv", "ai/cache", "ai/artifacts", "data", "logs", "wandb", "frontend"]

   [tool.ruff.lint]
   select = ["E", "F", "I", "B", "UP", "SIM"]
   # 초기 baseline 파악 단계 — 여기선 어떤 룰도 끄지 않는다(검출만 할 거라 안전).

   [tool.pytest.ini_options]
   testpaths = ["backend/tests", "ai/tests"]
   # (진단 4) app.* 와 backend.*/ai.* 동시 해결을 위해 루트와 backend 둘 다.
   pythonpath = [".", "backend"]
   addopts = "-ra"

   [tool.coverage.run]
   branch = true
   source = ["backend/app", "ai"]
   omit = ["*/tests/*", "*/__init__.py"]

   [tool.mypy]
   python_version = "3.10"
   ignore_missing_imports = true
   follow_imports = "silent"
   warn_unused_ignores = false
   # 초기 느슨. 모듈별 strict 는 후속 CP에서 [[tool.mypy.overrides]] 로.
   ```
   - 작성 후 **(진단 4)** 검증: pythonpath 가 `[".", "backend"]` 인지 재확인. 한쪽만 넣지 말 것.

2. `.pre-commit-config.yaml`(루트) 신규 작성. **실행 순서 엄수: ruff check --fix → ruff format → mypy.**

   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.6.9   # 설치된 ruff 버전과 맞춘다(Step 2 후 실제 버전으로 정정 가능)
       hooks:
         - id: ruff
           args: ["--fix"]
         - id: ruff-format
     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.11.2
       hooks:
         - id: mypy
           additional_dependencies: []
           args: ["--config-file=pyproject.toml"]
   ```
   - 주의: pre-commit 훅은 **이 CP에서 실제로 코드를 고치는 게 아니다**(Step 5 에서 `install` 만 하고, 전체 파일에 `pre-commit run --all-files` 강제 실행은 하지 않는다 — 그건 일괄 --fix 가 되어 금지 범위). 훅은 "다음 커밋부터 변경분에만" 작동하게 두는 게 목적.

3. `requirements-dev.txt`(루트) 신규 작성:

   ```text
   # CP222 안전망 도구. 운영 의존성(requirements.txt)과 분리.
   pytest==8.3.3
   pytest-cov==5.0.0
   ruff==0.6.9
   mypy==1.11.2
   httpx==0.27.0          # FastAPI TestClient 의존 (test_api.py 가 fastapi.testclient 사용)
   pandera==0.20.4        # CP223 데이터프레임 스키마 스냅샷 후보
   # ML 수치 스냅샷 라이브러리 — CP223 에서 택1 확정. 둘 다 후보로 명시:
   #   - syrupy (pytest 플러그인, 직렬화 스냅샷)
   #   - snaptol (수치 허용오차 스냅샷)
   # CP222 에서는 설치하지 않는다(CP223 가 고른 뒤 추가). 여기엔 주석으로만 남긴다.
   ```
   - 버전 핀은 설치 시점에 해석 실패하면 마이너 하향 가능(예: ruff 0.6.x 범위). 단 **메이저는 고정**.

4. **commit**: `git add pyproject.toml .pre-commit-config.yaml requirements-dev.txt` → `CP222: add ruff/pytest/coverage/mypy config + dev requirements`
5. **검증(Step 1)**: 파일 3개 생성 확인(`git status`). 이 단계는 설치 전이라 도구 실행은 아직.

### Step 2 — .venv 에 dev 도구 설치
1. venv 인터프리터로 설치:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   ```
2. 설치 검증(버전 출력):
   ```powershell
   .\.venv\Scripts\python.exe -m pytest --version
   .\.venv\Scripts\python.exe -m ruff --version
   .\.venv\Scripts\python.exe -m mypy --version
   ```
   - `ruff --version` 결과가 `.pre-commit-config.yaml` 의 `rev` 와 다르면 **rev 를 실제 버전으로 정정**(예: 출력이 `0.6.9` 면 `v0.6.9`).
3. **commit**: 설치는 venv 상태 변경(파일 변경 없음)이라 커밋 대상 없음. `.pre-commit-config.yaml` rev 를 정정했다면 그 변경만 `CP222: pin pre-commit ruff rev to installed version` 로 커밋.
4. **검증(Step 2)**: 세 도구 모두 버전 출력되면 통과.

### Step 3 — 기존 테스트 pytest 수집/실행 결과 기록 (baseline)
1. **수집만**(실행 없이 어떤 게 깨지는지 먼저):
   ```powershell
   .\.venv\Scripts\python.exe -m pytest --collect-only -q
   ```
   - **(진단 5)** `test_collector_jobs.py` 가 `scripts.cp134_local_daily_update_rehearsal` 부재로 collection error 날 것으로 예상. 에러 나는 파일 목록을 그대로 기록.
2. **전체 실행**(수집 에러 파일은 자동 제외하고 나머지 측정). 수집 에러가 전체를 막으면 `--continue-on-collection-errors` 사용:
   ```powershell
   .\.venv\Scripts\python.exe -m pytest --continue-on-collection-errors -q
   ```
3. 결과에서 **passed / failed / errors / skipped / collected 총수**를 그대로 보고서에 baseline 으로 박는다. (참고 상한: 디스크 51파일 / 함수 397개. 실제 수집수는 환경/스킵에 따라 다를 수 있으니 **실측값**을 적는다.)
4. 실패·에러가 났을 때 **(진단 5) 분류**:
   - "import 부재(gitignore 된 보조 스크립트/cp 파일 없음)" → **환경/수집 에러**로 분류, 소스 수정 금지, 기록만.
   - "Supabase 실접속 시도" → mock 누락 가능성. 소스 수정 금지, 기록 후 보고.
   - "코드 로직 실패(assert 깨짐)" → **차단 트리거 후보**(아래 참조).
5. **commit**: 코드/설정 변경 없음 → 커밋 없음. baseline 수치는 Step 마지막의 보고서 작성 단계에서 기록.
6. **검증(Step 3)**: pytest 가 "에러 없이 수집 프로세스를 끝냄"(=프로세스가 크래시하지 않고 수집 리포트를 냄) + passed 수가 0이 아님 → 통과. (수집 에러 일부는 진단대로 허용.)

### Step 4 — ruff 첫 실행 (검출만, --fix 절대 금지)
1. 검출:
   ```powershell
   .\.venv\Scripts\python.exe -m ruff check . --statistics
   ```
   - `--fix` / `--fix-only` / `format` **사용 금지**(동작 변경 위험, 범위 제외). 통계만 본다.
2. 검출 건수와 상위 룰 코드(E501, F401, I001, B008, SIM 등) Top 항목을 보고서에 기록.
3. **수천 건이 나와도 정상**(레거시 코드). 이 CP에선 **0 으로 만들 의무 없음**. CP224b 가 안전한 것만 고친다.
4. **commit**: 없음(코드 변경 0).
5. **검증(Step 4)**: `ruff check` 가 에러 없이 통계 출력하면 통과.

### Step 5 — mypy 동작 확인 + pre-commit install
1. mypy 1회:
   ```powershell
   .\.venv\Scripts\python.exe -m mypy backend/app ai --config-file=pyproject.toml
   ```
   - 에러 수 기록만. **수정 금지**. ImportError 류는 `ignore_missing_imports=true` 로 대부분 흡수됨.
2. pre-commit 설치:
   ```powershell
   .\.venv\Scripts\python.exe -m pre_commit install
   ```
   - `pre-commit` 이 requirements-dev 에 없으면 추가 후 재설치(`pip install pre-commit`). (스펙상 pre-commit 사용이 요구되므로 requirements-dev 에 `pre-commit` 줄을 포함시켜라 — Step 1.3 에서 누락 시 보완.)
   - **`pre-commit run --all-files` 는 실행하지 않는다**(일괄 --fix 금지 범위). 설치만.
3. **commit**: requirements-dev 에 `pre-commit` 보완했다면 그 변경만 커밋(`CP222: add pre-commit to dev requirements`). `.pre-commit install` 자체는 `.git/hooks` 변경이라 커밋 대상 아님.
4. **검증(Step 5)**: mypy 가 리포트 출력(크래시 X) + `pre-commit install` 이 "installed at .git/hooks/pre-commit" 출력 → 통과.

### Step 6 — 프론트 현 상태 확인 (FE 도구는 도입 안 함, 동작 확인만)
1. `frontend/package.json` 기준 도구 부재 확인(현재 devDeps: typescript/tailwind/postcss/autoprefixer 만, lint/test 없음 — 확인됨).
2. tsc 컴파일 동작만 확인(설정 변경 없이):
   ```powershell
   cd frontend; npm run build
   ```
   또는 타입체크만: `npx tsc --noEmit` (node_modules 설치되어 있어야 함; 없으면 `npm install` 후).
   - **결과는 baseline 기록만.** ESLint/Prettier/vitest 도입은 CP230 트랙 → **여기서 하지 않는다**.
3. **commit**: 없음(FE 파일 변경 0).
4. **검증(Step 6)**: tsc/build 가 돌아가는지(혹은 기존 에러가 있는지)만 기록.

### Step 7 — 보고서 + ADR 작성
1. `docs/cp222_report.md` 작성(요구/한일/결정/후속 + Step 3·4·5·6 의 baseline 수치).
2. `docs/adr/` 디렉토리가 **없으므로 생성** 후 `docs/adr/0010-quality-tooling-ruff-pytest.md` 작성(아래 ADR 섹션).
3. **commit**: `CP222: report + ADR-0010 quality tooling`.

---

## 인터페이스 보존

- CP222 는 **소스 코드(함수 signature / API 응답 schema / props)를 한 줄도 바꾸지 않는다.** 추가하는 것은 설정·의존성 파일뿐.
- 테스트 코드도 **수정하지 않는다**(수집 결과만 측정).
- pre-commit 훅은 이 CP에서 코드를 고치지 않도록 `install` 만 하고 `run --all-files` 를 돌리지 않는다 → 기존 코드 무변경 보장.
- 만약 어떤 도구 설치가 기존 import 동작을 바꿔야만 통과한다면(예: `ai/__init__.py` 를 만들어야 namespace 가 풀린다 등) → **임의로 만들지 말고 차단 보고**. 호출자 영향(어떤 테스트가 `from ai...` 하는지)을 정리해 올린다. (현재 분석상 pythonpath `[".", "backend"]` 로 충분할 것으로 예상 — `ai/__init__.py` 신설 불필요.)

---

## 성공 기준 (측정 가능)

| 항목 | 기대 | 비고 |
|---|---|---|
| pytest 수집 | 프로세스 크래시 없이 수집 리포트 출력 | `--collect-only` 가 끝까지 돈다 |
| pytest 통과(baseline) | passed > 0, 회귀 0 (CP222 가 만든 새 실패 0) | 실측 passed/failed/errors 를 보고서에 박음 |
| 수집 에러 허용 한도 | gitignore 부재 import 류만(예: `test_collector_jobs.py`) | 코드 로직 실패는 차단 |
| ruff 실행 | `ruff check . --statistics` 정상 출력 | 검출 건수 기록만, 0 만들 의무 없음 |
| mypy 실행 | `mypy backend/app ai` 리포트 출력(크래시 X) | 에러 수 기록만, 수정 0 |
| pre-commit | `pre-commit install` 성공 | 훅 등록 확인, `run --all-files` 안 함 |
| 설정 파일 | pyproject.toml / .pre-commit-config.yaml / requirements-dev.txt 3종 생성 | — |
| 소스 변경 | 0 라인 | 인터페이스 보존 |
| 예상 시간 | 약 1.5~2.5시간 | 설치 네트워크/수집 디버깅 포함 |

---

## 검증

각 명령은 워킹 디렉토리 `C:\Users\user\lens` 기준(프론트 단계만 `frontend`).

```powershell
# 0) venv 도구 존재
.\.venv\Scripts\python.exe -m pytest --version      # 기대: pytest 8.x
.\.venv\Scripts\python.exe -m ruff --version        # 기대: ruff 0.6.x
.\.venv\Scripts\python.exe -m mypy --version        # 기대: mypy 1.11.x

# 1) 수집 (진단 5 확인용)
.\.venv\Scripts\python.exe -m pytest --collect-only -q
#   기대: 대부분 수집됨. test_collector_jobs.py 등 일부 ModuleNotFoundError(scripts.cp134...) 가능 → 정상(기록).

# 2) 전체 실행 baseline
.\.venv\Scripts\python.exe -m pytest --continue-on-collection-errors -q
#   기대: "N passed, M errors" 형태 요약. passed > 0. CP222 가 만든 새 실패 0.

# 3) ruff 검출만
.\.venv\Scripts\python.exe -m ruff check . --statistics
#   기대: 룰별 건수 표. (--fix 안 씀)

# 4) mypy
.\.venv\Scripts\python.exe -m mypy backend/app ai --config-file=pyproject.toml
#   기대: 에러 리스트 + "Found X errors" (또는 Success). 크래시 없음.

# 5) pre-commit
.\.venv\Scripts\python.exe -m pre_commit install
#   기대: "pre-commit installed at .git/hooks/pre-commit"

# 6) 프론트(선택, baseline)
cd frontend; npm run build
#   기대: 빌드 성공 또는 기존 타입에러 목록(기록만). 설정 변경 0.
```

---

## 차단 트리거 (중요)

> 다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **기존 테스트가 다수(수집 가능분의 30% 이상) 실패** → 코드 자체 문제 가능. 진행 멈추고 실패 테스트 **목록 + 대표 traceback** 정리해 보고. (단, 진단 5 의 import 부재 collection error 는 "수집 에러" 로 따로 분류 — 이건 30% 계산에서 제외하되 별도 목록으로 보고.)
2. **수집 에러가 진단 5 범위를 넘어 확산**(예: `app.*` 또는 `ai.*` 코어 import 까지 ImportError) → pythonpath 설정 오류 신호. `[".", "backend"]` 인지 재확인하고, 그래도 안 풀리면 **`ai/__init__.py` 신설 등 소스/패키지 변경을 임의로 하지 말고** 영향 분석과 함께 보고.
3. **테스트가 Supabase 에 실접속을 시도**(network timeout / 인증 에러로 실패) → mock 누락. 금지(Supabase 호출)에 저촉되므로 소스 수정 없이 멈추고 보고.
4. **ruff 가 수천 건을 검출** → 이건 차단이 아님(정상). **단 `--fix` 를 돌리고 싶은 유혹이 생기면 멈춰라.** 이 CP에서 일괄 --fix 금지(CP224b 몫). 실수로 --fix 를 돌렸다면 즉시 `git checkout -- .` 로 복원하고 보고.
5. **도구 설치가 venv 가 아니라 시스템 Python 에 들어감**(`which`/`Source` 가 `AppData\...\Python310` 가리킴) → 격리 깨짐. 재설치(`.\.venv\Scripts\python.exe -m pip ...`)하고 보고.
6. **mypy/ruff 설치 또는 실행이 크래시**(설정 파싱 에러, 플러그인 충돌) → 설정 파일 문법 점검. 해결 안 되면 멈추고 보고.
7. **pre-commit 훅 설치가 기존 코드를 일괄 변경**(실수로 `run --all-files` 실행 등) → 즉시 복원하고 보고. CP222 산출물에 소스 diff 가 있으면 안 된다.
8. **버전 핀 충돌로 설치 실패**(예: torch 2.11 환경과 의존 충돌) → 어떤 패키지가 충돌인지 로그 첨부해 보고. 임의로 운영 의존성(requirements.txt) 을 건드려 해결 시도 금지.

---

## ADR

작성: `docs/adr/0010-quality-tooling-ruff-pytest.md` (디렉토리 없으면 `docs/adr/` 생성). 200~300단어.
기록 내용 한 줄: **왜 ruff(올인원 lint+format) + pytest + pytest-cov + mypy + pre-commit 조합인지, 그리고 ruff 룰셋(E,F,I,B,UP,SIM)·line-length 100·mypy 초기 느슨(ignore_missing_imports) 선택 근거.**
포함할 결정 요지:
- ruff 단일화 이유(flake8+isort+pyupgrade+일부 bugbear 를 하나로, Rust 속도, 설정 단순).
- 룰셋 선택: E/F(기본 오류), I(import 정렬), B(bugbear 실수), UP(pyupgrade 3.10+), SIM(단순화). D/ANN 등은 초기 제외(노이즈).
- line-length 100(레거시 코드가 100 근처, 88 은 과도 변경 유발).
- pytest 채택 이유: 기존 unittest.TestCase 를 변환 없이 수집(진단 2), 397개 함수 그대로 흡수.
- pythonpath `[".", "backend"]` 결정 근거(진단 4 의 혼합 import).
- mypy 초기 느슨 + 모듈별 strict 는 후속(점진 도입) 결정.
- 일괄 --fix 를 CP222 가 아닌 CP224b 로 미룬 이유(동작 변경/리뷰 분리).

---

## 자가 점검 결과 양식

작업 완료 후 아래를 채워 보고:

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ____ (CP222 는 모델/밴드 로직 무관. 정합 영향 없어야 PASS)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ____ (pythonpath·수집 에러를 정직히 baseline 화했는지)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ____ (소스 0 변경이므로 모델 영향 없음 = PASS 예상)

---

## 산출물

- **변경/생성 파일**:
  - `pyproject.toml` (신규)
  - `.pre-commit-config.yaml` (신규)
  - `requirements-dev.txt` (신규)
  - `docs/adr/0010-quality-tooling-ruff-pytest.md` (신규, 디렉토리 포함)
  - `docs/cp222_report.md` (신규)
- **보고서** `docs/cp222_report.md` 에 담을 것(필요한 만큼만):
  - **요구**: 품질 도구 0 → 안전망 1차(ruff/pytest/cov/mypy/pre-commit) 구축.
  - **한일**: 설정 3종 작성, venv 설치, 수집/실행 baseline 측정, ruff·mypy 1회 검출, pre-commit install.
  - **결정**: ruff 올인원·룰셋·pythonpath `[".", "backend"]`·일괄 --fix 보류(ADR-0010 참조).
  - **baseline 수치(실측 박기)**: pytest collected N / passed / failed / errors(+ 수집 에러 파일 목록, 특히 `test_collector_jobs.py` 의 `scripts.cp134...` 부재) / ruff 검출 건수·상위 룰 / mypy 에러 수 / 프론트 build 결과.
  - **후속**: CP223(스냅샷, 스냅샷 라이브러리 syrupy↔snaptol 확정) / CP224a(requirements 핀) / CP224b(안전한 ruff --fix 적용).
