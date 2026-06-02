# CP237 CI (GitHub Actions) (Directive)

> 이 문서는 단독 실행 가능한 지시서다. 새 Claude Code 세션이 런북(`docs/cp221_237_refactoring_runbook.md`)에서 이 CP를 꺼내 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지. 막히면 멈추고 보고.
>
> **이 CP는 런북상 마지막이다.** CP222(도구) + CP223(BE 스냅샷) + CP230(FE 테스트)가 만든 모든 안전망을 GitHub Actions push/PR 게이트로 박제한다. 새 검증 로직을 발명하지 않는다 — 이미 로컬에서 도는 명령을 CI 워크플로로 옮길 뿐이다.

---

## 역할 고정
- **모드**: `code` (구현 + 자가 점검만 보고. 기획/설계 토론 아님).
- **권한**: 코드/설정 파일 수정, 로컬 검증(워크플로 YAML 문법 검사 + 각 잡이 실행할 명령을 로컬에서 동일하게 재현).
- **금지**:
  - 새 모델 학습 금지.
  - 새 calibration 금지.
  - DB write 금지 (Postgres / parquet 스냅샷 쓰기 금지).
  - Supabase 호출 금지. CI 잡에 Supabase 시크릿/실연결을 넣지 않는다 (테스트는 전부 mock — §진단 참조).
  - 사용자가 직접 수정한 파일 revert 금지.
  - **기존 테스트 코드/소스 로직 수정 금지.** 이 CP는 CI 설정 추가(additive)다. 테스트가 CI(CPU)에서 깨지면 "고쳐서 통과시키지 말고" 차단 트리거대로 멈추고 보고한다. baseline은 "지금 통과하는 것"을 박제하는 것이지 새로 통과하게 만드는 게 아니다.
- **자가 점검**: 종료 시 [Plan v3 정합] / [구조 결함] / [모델 영향] 3축 PASS·WARN·FAIL + 사유 보고 (아래 양식).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경
- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0 로컬; torch 2.11.0+cu128 로컬 GPU 휠). 활성화: `.\.venv\Scripts\Activate.ps1`
  - ⚠ **CI는 로컬과 다르다.** CI(GitHub Actions ubuntu-latest, CPU)는 `backend/requirements.txt`를 그대로 설치하며 거기에는 **`torch==2.3.1`**(CPU 휠)이 핀되어 있다(확인: `backend/requirements.txt:9`). 즉 CI에는 cu128/GPU가 없고 CPU torch가 깔린다. 아래 §진단에서 "현재 테스트가 CPU에서 도는가"를 이미 검증해 두었다.
  - CI Python 버전은 **3.11**로 고정한다(render 운영과 일치: `render.yaml:13` `PYTHON_VERSION=3.11.11`). 로컬 3.10.0과의 미세 차이는 §차단 트리거에서 다룬다.
- **백엔드 기동**(이 CP에선 불필요. CI도 서버를 띄우지 않는다 — 단위/통합 테스트만): 참고용 `scripts\start_demo.ps1` 또는 `python -m uvicorn app.main:app --port <빈포트>` (`app` 객체: `backend/app/main.py:45`).
- **프론트**(CI에서 빌드/타입체크/Vitest, Playwright만): `npm run dev`는 CI 불필요. CI는 `npm ci` → `tsc`/`vitest`/`playwright`.
- **검증용 포트 충돌 주의**: 이 CP는 로컬에서 서버를 띄울 일이 없다. 혹 act 등으로 컨테이너를 돌릴 경우에도 데모가 점유한 8000/3000을 건드리지 않는다.

---

## 진단 (근거)
**무엇이 문제인가**: 리포지토리에 **CI가 전혀 없다.** `.github/` 디렉토리 자체가 없고(확인: 워크트리/메인 모두 `.github` 부재, `git ls-files | grep .github` → 0건), push/PR 시 자동으로 도는 게이트가 없다. CP222~CP236에서 쌓은 안전망(ruff/mypy/pytest/스냅샷/Vitest/Playwright)이 **로컬에서 사람이 수동 실행해야만** 의미를 갖는다. main 머지 전에 회귀를 자동 차단하는 마지막 고리가 비어 있다.

**리포 사실(확인된 것, CI 설계에 직접 영향)**:

1. **원격/브랜치**: `origin = https://github.com/vmgfh878-art/lens.git`. 기본 브랜치 `main`(확인: `git symbolic-ref refs/remotes/origin/HEAD` → `refs/remotes/origin/main`). 작업 브랜치로 `develop`(main 대비 35 커밋 ahead)와 `claude/*` 다수 존재. → **게이트 트리거**: `push`(모든 브랜치) + `pull_request`(target `main`, `develop`).

2. **백엔드 테스트 = 전부 mock, 네트워크/DB/Supabase 미접촉** (CI에서 시크릿 불필요):
   - `backend/tests/test_api.py` (529줄): `fastapi.testclient.TestClient`로 인메모리. 모든 데이터 fetch는 `unittest.mock.patch`로 대체(`app.routers.v1.stocks.get_price_response_data` 등). `app.db.reset_supabase_client()`만 호출(실연결 아님). 시크릿 없음.
   - `backend/tests/test_services.py` (486줄): 전부 mock. local-parquet 분기는 `tempfile.TemporaryDirectory`에 parquet 써서 검증하고, 엔진 없으면 `self.skipTest(...)`로 graceful skip(`test_services.py:90-91`). Supabase는 `patch("app.repositories.market_repo.get_supabase", ...)`로 차단.
   - `backend/tests/test_collector_jobs.py` (641줄): 전부 mock. `fetch_frame`/`upsert_records`/`get_supabase` 등 외부 I/O를 patch. 네트워크 0.
   - `backend/tests/test_feature_svc.py`: 순수 pandas 계산, mock 불필요.
   - → **결론**: 백엔드 테스트는 CI(CPU, 시크릿 0)에서 그대로 돈다.

3. **임포트 루트가 2종으로 갈린다 (CI 최대 함정 — 반드시 처리)**:
   - `backend/tests/test_api.py:7-9`, `test_services.py:10-12` → **`from app.X import ...`** (예: `from app.main import app`). 이건 **`backend/`가 `sys.path`에 있어야** import된다 (render도 `rootDir: backend`, `render.yaml:5`).
   - `backend/tests/test_feature_svc.py:7-11`, `test_collector_jobs.py:9-24` → **`from backend.X import ...`** + **`from scripts.X import ...`**. 이건 **리포 루트가 `sys.path`에 있어야** import된다. 두 파일 모두 상단에서 `ROOT_DIR = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT_DIR))`로 루트를 자가 주입한다(`test_feature_svc.py:7-9`, `test_collector_jobs.py:9-11`).
   - `ai/tests/*` → **`from ai.X import ...`** (리포 루트 필요). `ai/tests/__init__.py` 존재. `ai/__init__.py`·`scripts/__init__.py`는 **없음**(implicit namespace package로 동작, 리포 루트가 path에 있으면 OK).
   - → **CI는 pytest 한 번에 리포 루트 + `backend/` 둘 다 `sys.path`에 올려야 collection이 안 깨진다.** 권장: 리포 루트에서 `PYTHONPATH=.:backend pytest ...` (Linux 구분자 `:`). 또는 백엔드 잡과 ai 잡을 분리하되, 백엔드 잡 안에서도 `app.`/`backend.` 둘 다 필요하므로 **단일 pytest 실행 + PYTHONPATH 둘 다**가 가장 안전하다. Step 1에서 로컬로 먼저 collection을 확인하고 박는다.

4. **GPU(cu128) 의존 테스트는 없다 — CPU CI 안전 (검증 완료)**:
   - `ai/tests/*`의 torch 진입은 `from ai.torch_bootstrap import bootstrap_torch` (`ai/tests/test_preprocessing.py:3-5` 등). `ai/torch_bootstrap.py:22-32`의 `bootstrap_torch()`는 단순 `import torch` + 환경변수 설정일 뿐 GPU를 강제하지 않는다(`cpu_only=True`면 `CUDA_VISIBLE_DEVICES=-1`). `ai/tests/test_architecture_postprocess.py:3`은 raw `import torch`지만 모든 텐서가 기본(CPU) 디바이스다 — `.cuda(`/`device='cuda'` 호출 0건(확인: `ai/tests`에서 cuda grep → 매치 없음).
   - 리포 전체에서 `"cuda" if torch.cuda.is_available() else "cpu"` 패턴은 **`ai/cp153_*` 학습 스크립트에만** 있고(`ai/cp153_bm_1d_band_500_stage0_1_baseline.py:1125` 등 6곳), 이들은 `.gitignore`로 git 추적에서 제외되며 **테스트가 아니다**. 게다가 전부 `is_available()` 가드라 CPU에서도 죽지 않는다.
   - → **현 시점 GPU 강제 테스트 0. CPU CI에서 ai 테스트가 그대로 통과해야 정상.** 만약 CP223/CP230 산출물이나 향후 변경으로 `@pytest.mark.gpu`나 무가드 `.cuda()`가 테스트에 새로 생겼다면 = §차단 트리거.

5. **프론트(Next 14.2.3, `frontend/package.json`)**: `scripts`에 현재 `dev/build/start`만 있고 **`tsc`/`test`/`vitest`/`playwright` 스크립트가 없다**(`frontend/package.json:5-9`). `tsconfig.json`은 `noEmit:true`, `strict:true`(`frontend/tsconfig.json:7-8`). `frontend/src` 소스 6개(`api/client.ts`, `app/layout.tsx`, `app/page.tsx`, `components/{Chart,DashboardPage,ModelSelector}.tsx`). **Vitest/Playwright 테스트 파일과 devDependency는 CP230 산출물이다** — 이 CP는 그것을 호출만 한다(§선행 의존).

**조사 출처**: `.github` 부재 확인(워크트리/메인 `ls`, `git ls-files`). `git remote -v` / `git symbolic-ref` / `git branch -a`. `backend/requirements.txt`(13줄, torch==2.3.1). `render.yaml`(3.11.11, rootDir backend, start `uvicorn app.main:app`). `backend/tests/*` 4파일 Read(import/ mock 패턴). `ai/torch_bootstrap.py` 전체 Read. `ai/tests/*` torch/cuda grep. `frontend/package.json`·`tsconfig.json` Read. 런북 `docs/cp221_237_refactoring_runbook.md`(CP237 = 마지막, GPU skip 주의).

---

## 선행 의존
- **CP222 (안전망 도구: ruff + mypy + pytest + coverage + pre-commit 설정) 존재해야 한다.** CI backend 잡이 `ruff check` / `ruff format --check` / `mypy`를 호출하려면 그 설정 파일(`pyproject.toml` 또는 `ruff.toml`/`setup.cfg`/`mypy.ini` — CP222가 정한 형식)이 리포에 있어야 한다. **시작 전 확인**:
  ```powershell
  # CP222 도구 설정이 있는지
  Test-Path pyproject.toml; Test-Path ruff.toml; Test-Path setup.cfg; Test-Path mypy.ini
  .\.venv\Scripts\Activate.ps1
  ruff --version; mypy --version   # 설치돼 있어야
  ```
  설정/도구가 없으면 = CP222 미완 → **즉시 중단, "CP222 미충족"으로 보고.**
- **CP223 (백엔드 characterization 스냅샷) 존재해야 한다.** CI backend 잡의 "스냅샷" 스텝이 호출할 테스트가 있어야 한다. **확인**:
  ```powershell
  python -m pytest backend/tests -k "snapshot or characterization or golden" -q --collect-only
  ```
  `0 collected`면 CP223 산출물 부재 → **중단, "CP223 미충족" 보고.** (스냅샷 테스트 이름/경로는 CP223 산출물 기준. 보통 `backend/tests/` 하위.)
- **CP230 (프론트 characterization: Vitest + Playwright) 존재해야 한다.** CI frontend 잡이 호출할 `vitest`/`playwright` 스크립트와 테스트가 있어야 한다. **확인**:
  ```powershell
  Get-Content frontend/package.json   # scripts에 test / test:e2e (또는 vitest / playwright) 있나
  Test-Path frontend/playwright.config.ts; Test-Path frontend/vitest.config.ts
  ```
  없으면 = CP230 미완. → frontend 잡 중 Vitest/Playwright 스텝은 **만들되**, 스크립트가 없으면 그 잡은 실패한다 → **차단 트리거(아래)대로 보고.** (임의로 빈 테스트를 만들어 통과시키지 말 것.)
- **선행이 하나라도 빨강/부재면 이 CP를 끝까지 진행하지 않는다.** CI는 "이미 있는 안전망"을 박제하는 것이지 안전망 자체를 만드는 게 아니다.

> 런북 권장 직렬 순서상 CP237은 CP222·CP223·CP230·(CP224~CP236) 뒤의 **맨 끝**이다(`runbook §2`). 따라서 정상 실행 시점엔 위 선행이 모두 그린이어야 한다.

---

## 범위
**포함**
- 신규 `.github/workflows/ci.yml` 1개 작성. 잡 2개:
  - **backend**: `ruff check` + `ruff format --check` + `mypy` + `pytest`(전체) + characterization snapshot(`pytest -k "snapshot or characterization or golden"`).
  - **frontend**: `npx tsc --noEmit` + Vitest + Playwright.
- **캐싱(보강6, 필수)**: pip 캐시(`actions/setup-python@v5`의 `cache: pip`) / node_modules 또는 npm 캐시(`actions/setup-node@v4`의 `cache: npm`) / Playwright 브라우저 캐시(`~/.cache/ms-playwright`, ~300MB — 캐싱 안 하면 매 PR 브라우저 다운로드 ~3분 추가) / pytest 입력(스냅샷·fixture는 리포에 커밋된 파일이므로 별도 캐시 불필요하나, pip 캐시로 충분).
- main(+develop) 머지 게이트: `pull_request` 트리거로 PR에서 필수 체크가 되게 한다. (branch protection 활성화는 GitHub UI 작업 — 코드로 못 함 → §검증에서 안내만.)
- 로컬에서 가능한 범위의 dry 점검(YAML 문법 + 각 잡 명령 로컬 재현). `act` 사용 가능하면 사용, 불가하면 명령 단위 재현으로 대체.
- ADR 1장(`docs/adr/0028-ci-github-actions.md`).
- 리포트 `docs/cp237_report.md`.

**제외**
- 테스트/소스 로직 수정 (additive만). CI에서 깨지면 고치지 말고 보고.
- 배포(CD) 파이프라인, Render 자동배포 변경(`render.yaml` 손대지 않음. autoDeploy는 Render가 관리).
- Supabase 시크릿/실연결을 CI에 추가하는 것 (보류, 테스트는 mock).
- GPU 러너 도입 (CPU 러너만. ai 테스트는 CPU로 통과 — §진단4).
- branch protection rule을 강제로 켜는 것(UI 권한 작업, 안내만).
- `.gitignore` 정책 변경 (단, ADR/report가 `docs/cp*` ignore에 걸리는 문제는 §ADR 참조).

---

## Sub-step (Strangler Fig, 작은 단위)
> CI에는 "옛 코드"가 없다(신규 추가). 그래서 Strangler 패턴은 **"잡을 한 번에 다 켜지 말고, 한 잡씩 추가→로컬 재현으로 그린 확인→커밋, 마지막에 캐싱 보강"** 으로 적용한다. 한 Step = 한 커밋 = 한 revert 단위. 각 Step 끝에 **로컬에서 그 잡이 실행할 명령을 동일하게 재현**해 그린을 확인한 뒤 커밋한다(CI는 push 후에야 돌므로, 로컬 재현이 1차 게이트).
>
> **공통 로컬 재현 원칙**: CI YAML에 적는 명령과 **토씨까지 같은 명령**을 로컬에서 돌려 통과를 확인한다. CI 전용 추정 금지.

### Step 0 — 선행 확인 + 베이스라인 기록 (커밋 없음)
- §선행 의존의 3개 확인 블록(CP222 도구 / CP223 스냅샷 / CP230 프론트) 실행. 하나라도 부재/빨강이면 **중단·보고**.
- 현재 테스트가 로컬에서 그린인지 박제 전 확인(= CI에서 통과해야 할 baseline):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  # 백엔드: 두 임포트 루트가 한 번에 잡히는지 + 통과
  $env:PYTHONPATH = ".;backend"
  python -m pytest backend/tests ai/tests -q
  ```
  - **기대**: 백엔드 4파일 + ai 6파일 전부 collected, `0 failed`(local-parquet 1건은 환경에 따라 `skipped` 가능 — 허용). 빨강이면 = 박제할 baseline이 빨강 → **중단·보고**(CI 탓 아님, 사전 상태 문제).
  - 통과한 `PYTHONPATH` 조합(여기선 루트+backend)을 기록. CI YAML에 Linux 구분자(`.:backend`)로 그대로 옮긴다.

### Step 1 — `.github/workflows/ci.yml` backend 잡 (캐싱 전, 최소 동작)
- 신규 `.github/workflows/ci.yml` 생성. 트리거 + backend 잡만:
  - `name: CI`
  - `on: { push: { branches: ['**'] }, pull_request: { branches: [main, develop] } }`
  - `jobs.backend`:
    - `runs-on: ubuntu-latest`
    - `steps`:
      1. `actions/checkout@v4`
      2. `actions/setup-python@v5` with `python-version: '3.11'` (캐싱은 Step 3에서 추가)
      3. 의존 설치: `pip install -r backend/requirements.txt` + ruff/mypy/pytest 도구(설치 출처는 CP222 산출물 — `pyproject.toml`의 dev extra가 있으면 `pip install -e .[dev]` 류, 없으면 명시 `pip install ruff mypy pytest`. **CP222가 정한 방식을 따른다**).
      4. `ruff check .`
      5. `ruff format --check .`
      6. `mypy backend ai`  (대상 경로는 CP222 mypy 설정의 `files`/제외와 일치시킨다. 충돌 시 CP222 설정 우선.)
      7. pytest(스냅샷 포함 전체): 작업 디렉토리 리포 루트, `env: PYTHONPATH: .:backend` 후 `python -m pytest backend/tests ai/tests -q`
      8. 스냅샷 전용 재실행(명시 게이트): `python -m pytest backend/tests -k "snapshot or characterization or golden" -q` (7에 포함되지만, 실패 시 원인을 분명히 드러내기 위한 별도 스텝)
- **로컬 재현**(YAML과 동일 명령): Step 0의 pytest + `ruff check .` + `ruff format --check .` + `mypy backend ai` 를 로컬에서 돌려 전부 그린 확인.
- **YAML 문법 검증**:
  ```powershell
  # 둘 중 가능한 것
  python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8')); print('yaml-ok')"
  # (선택) actionlint 있으면: actionlint .github/workflows/ci.yml
  ```
- 그린 → 커밋: `ci: add GitHub Actions backend job (ruff/mypy/pytest/snapshot)`.

### Step 2 — frontend 잡 추가
- 같은 `ci.yml`에 `jobs.frontend` 추가:
  - `runs-on: ubuntu-latest`
  - `defaults.run.working-directory: frontend` (또는 각 step에서 `cd frontend` 대신 working-directory 지정)
  - `steps`:
    1. `actions/checkout@v4`
    2. `actions/setup-node@v4` with `node-version: '20'` (캐싱은 Step 3)
    3. `npm ci` (lockfile 있음: `frontend/package-lock.json`)
    4. 타입체크: `npx tsc --noEmit`  (스크립트 없어도 직접 호출 가능. `tsconfig.json`은 noEmit:true.)
    5. Vitest: CP230이 정의한 스크립트 호출(예: `npm run test -- --run` 또는 `npx vitest run`). **CP230 산출물의 정확한 스크립트명을 `frontend/package.json`에서 확인해 그대로 적는다.**
    6. Playwright 브라우저 설치: `npx playwright install --with-deps chromium` (CI 최초 1회, 캐싱은 Step 3)
    7. Playwright 실행: CP230이 정의한 스크립트(예: `npm run test:e2e` 또는 `npx playwright test`). config: `frontend/playwright.config.ts`(CP230 산출).
- **로컬 재현**: `cd frontend; npm ci; npx tsc --noEmit; <vitest 스크립트>; npx playwright install chromium; <playwright 스크립트>` 를 로컬에서 돌려 그린 확인. (Playwright는 백엔드 mock/정적 모드로 도는지 CP230 설정을 따른다 — 이 CP에서 e2e 동작을 새로 설계하지 않는다.)
- ⚠ **CP230 스크립트가 없으면**: 빈 테스트로 통과를 위장하지 말고 §차단 트리거대로 보고.
- YAML 문법 재검증 → 그린 → 커밋: `ci: add GitHub Actions frontend job (tsc/vitest/playwright)`.

### Step 3 — 캐싱 보강 (pip / npm / Playwright 브라우저)
- backend 잡:
  - `setup-python@v5`에 `cache: 'pip'` + `cache-dependency-path: backend/requirements.txt` 추가.
- frontend 잡:
  - `setup-node@v4`에 `cache: 'npm'` + `cache-dependency-path: frontend/package-lock.json` 추가.
  - **Playwright 브라우저 캐시**: `actions/cache@v4` 스텝 추가.
    - `path: ~/.cache/ms-playwright`
    - `key: ${{ runner.os }}-playwright-${{ hashFiles('frontend/package-lock.json') }}`
    - cache hit 시 `npx playwright install`을 건너뛰거나(조건 step) 그냥 다시 호출(이미 캐시돼 빠름). 안전하게: cache 복원 후에도 `npx playwright install --with-deps chromium`을 호출하되, 캐시가 있으면 다운로드를 건너뛰어 ~3분을 절약한다(브라우저가 캐시에 있으면 install이 즉시 끝남).
- pytest 입력 캐시: 스냅샷/fixture는 **리포에 커밋된 파일**(checkout으로 항상 존재)이라 별도 캐시 불필요. pip 캐시로 torch 등 대형 휠 재다운로드만 막으면 충분.
- **로컬 재현**: 캐싱은 GitHub 인프라 기능이라 로컬에서 효과 재현은 제한적. 대신 (a) YAML 문법 재검증, (b) 캐시 추가가 **명령 자체를 바꾸지 않았는지**(설치/테스트 스텝은 동일) 확인.
- → 커밋: `ci: add pip/npm/playwright caching`.

### Step 4 — 로컬 dry 점검 (act 또는 명령 재현) + 최종 검수
- **act 사용 가능하면**(Docker 필요):
  ```powershell
  act --version   # 설치 확인
  act -W .github/workflows/ci.yml -j backend --container-architecture linux/amd64 -n   # dry-run(-n)부터
  ```
  - act가 없거나 Docker가 없으면 **건너뛰고**, Step 0~2의 로컬 명령 재현 결과로 갈음한다(이 환경은 Windows라 act 미설치가 정상 — 강요 금지).
- 최종 YAML 전체 문법 검증(`python -c yaml.safe_load`) + 두 잡의 모든 명령을 로컬에서 한 번 더 통과 확인.
- 워크플로 파일이 `git add` 되는지 확인(`.gitignore`는 `.github`를 막지 않음 — 확인: gitignore에 `.github` 항목 없음).
- → 이미 Step 1~3에서 커밋됨. 추가 변경이 있으면 `ci: finalize ci.yml after local dry-check` 로 커밋.

> Step 1~3은 각각 독립 revert 단위. CI가 push 후 빨강이면, 빨간 잡을 추가한 Step 커밋만 되돌려 직전 그린으로 복귀.

---

## 인터페이스 보존
- **소스/테스트/응답 schema/props 인터페이스를 일절 바꾸지 않는다.** 이 CP는 `.github/workflows/ci.yml`(신규) + ADR + report만 추가하는 additive 변경이다.
- `backend/requirements.txt`, `frontend/package.json`의 **기존 의존/스크립트를 수정하지 않는다.** CI는 이들을 *읽어* 설치/호출할 뿐이다. (CP230이 추가한 frontend 스크립트가 이미 있으면 그대로 호출. 없으면 만들지 말고 보고 — 스크립트 추가는 CP230 책임.)
- `render.yaml` 불변(배포 경로 보존).
- 만약 CI를 통과시키기 위해 테스트/소스/설정의 동작을 바꿔야 하는 상황이 오면: **그냥 바꾸지 말고**, 무엇을 왜 바꿔야 하는지(어느 테스트가 CPU/3.11/Linux에서 깨지는지)와 영향을 적어 **차단 보고** 후 사용자 판단을 기다린다.

---

## 성공 기준 (측정 가능)
| 항목 | 시작 | 목표 |
|---|---|---|
| `.github/workflows/ci.yml` | 없음 | 존재, YAML 문법 유효(`yaml.safe_load` 통과) |
| 잡 개수 | 0 | 2 (backend, frontend) |
| backend 잡 스텝 | — | ruff check / ruff format --check / mypy / pytest(전체) / snapshot pytest 포함 |
| frontend 잡 스텝 | — | tsc --noEmit / Vitest / Playwright 포함 |
| 로컬 재현: `pytest backend/tests ai/tests` (PYTHONPATH=.;backend) | green | green, `0 failed` (local-parquet 1건 skip 허용) |
| 로컬 재현: `ruff check .` / `ruff format --check .` | clean | clean (신규 위반 0) |
| 로컬 재현: `mypy backend ai` | baseline | 신규 error 0 추가 |
| 로컬 재현: `npx tsc --noEmit` (frontend) | 0 | 0 errors |
| 캐싱 | 없음 | pip + npm + Playwright 브라우저 캐시 3종 모두 YAML에 존재 |
| 트리거 | 없음 | push(all) + pull_request(main, develop) |
| 예상 시간 | — | 2~3시간 |

> 측정 불가 항목(실제 CI 잡 그린)은 push 후 GitHub Actions UI에서 확인. 이 CP의 로컬 성공 기준은 "YAML 유효 + 모든 잡 명령을 로컬에서 동일 재현해 그린". push 후 CI가 빨강이면 §차단 트리거.

---

## 검증
**로컬 (이 CP의 1차 게이트, 매 Step + 최종)**:
```powershell
.\.venv\Scripts\Activate.ps1

# 1) YAML 문법
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8')); print('yaml-ok')"

# 2) backend 잡 명령 재현 (CI와 동일 명령)
$env:PYTHONPATH = ".;backend"
ruff check .
ruff format --check .
mypy backend ai
python -m pytest backend/tests ai/tests -q
python -m pytest backend/tests -k "snapshot or characterization or golden" -q

# 3) frontend 잡 명령 재현
cd frontend
npm ci
npx tsc --noEmit
# CP230 스크립트명 확인 후 실행 (예시 — 실제 package.json 스크립트로 대체)
# npm run test -- --run         # Vitest
# npx playwright install chromium
# npm run test:e2e              # Playwright
cd ..
```
**기대 결과**:
- `yaml-ok` 출력.
- `ruff check .` → `All checks passed!`; `ruff format --check .` → 재포맷 대상 0.
- `mypy backend ai` → 신규 error 0(기존 baseline 유지).
- `pytest backend/tests ai/tests` → 백엔드 4 + ai 6 파일 collected, `0 failed`(local-parquet skip 허용).
- 스냅샷 pytest → `0 failed`.
- `npx tsc --noEmit` → 0 errors. Vitest/Playwright → CP230 정의대로 통과.

**push 후 (GitHub)**:
- Actions 탭에서 backend/frontend 두 잡 모두 ✅.
- PR 생성 시 두 잡이 체크로 표시.
- **branch protection(머지 게이트 강제)은 GitHub UI 작업**: Settings → Branches → `main`(+`develop`) → "Require status checks to pass before merging" → backend/frontend 선택. 이건 코드로 못 켠다 → report에 "사용자가 UI에서 켜야 함" 1줄로 안내(이 CP가 직접 켜지 않음).

---

## 차단 트리거 (중요)
다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**
1. **CP222 도구(ruff/mypy 설정·설치) 부재** (Step 0). → CI가 호출할 lint/type 대상이 없음. 시작 금지, "CP222 미충족" 보고.
2. **CP223 스냅샷 테스트가 `0 collected`이거나 빨강** (Step 0). → 박제할 BE 안전망 없음. 중단, "CP223 미충족" 보고.
3. **CP230 프론트 스크립트(Vitest/Playwright) 부재** (Step 0/2). → frontend 잡이 호출할 대상 없음. 빈 테스트로 위장 금지. 중단, "CP230 미충족" 보고.
4. **Step 0 로컬 baseline pytest가 빨강** (CI 박제 전부터 빨강). → CI 탓이 아니라 사전 상태 문제. 무엇이 실패하는지(어느 파일/테스트) 정리해 중단·보고. **CI로 통과시키려고 테스트를 고치지 말 것.**
5. **CI(CPU)에서 기존 테스트가 실패해 baseline을 못 잡음** (push 후). 특히:
   - **임포트 루트 문제**: `app` 또는 `backend.`/`scripts.`/`ai.` import가 CI에서 `ModuleNotFoundError`. → PYTHONPATH/working-directory 조합 문제. YAML의 path 설정을 보고와 함께 점검(임의 sys.path 해킹 코드 주입 금지).
   - **GPU 의존**: 어떤 테스트가 CPU에서 `torch.cuda` 관련으로 실패하거나 `@pytest.mark.gpu` 류로 표시돼야 하는데 안 돼 있음. → 해당 테스트를 CI(CPU)에서 skip 마킹해야 한다(예: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")`). **이건 테스트 동작 변경이므로 임의로 하지 말고**, 어느 테스트인지 + 제안 마킹을 적어 **보고** 후 진행.
   - **Python 3.11 vs 로컬 3.10 차이**: 3.11에서만 깨지는 테스트(문법/표준라이브러리 차이). → 버전 차이임을 명시해 보고.
6. **torch 설치 실패/타임아웃**(CI에서 `backend/requirements.txt`의 torch==2.3.1 CPU 휠 설치 실패). → requirements 핀/인덱스 문제. 임의로 torch 버전 바꾸지 말고(그건 CP224a 영역) 보고.
7. **YAML 문법 무효**(`yaml.safe_load` 예외, 또는 actionlint 에러). → 고치되, 구조가 모호하면 보고.
8. **캐싱이 동작/명령을 바꿔 테스트 결과가 달라짐**(캐시 추가 후 설치/실행 스텝이 달라져 그린→빨강). → 캐시는 순수 성능 최적화여야 한다. 명령이 바뀌었으면 되돌리고 보고.
9. **Supabase/시크릿이 필요해지는 테스트가 보임**(mock이 아닌 실연결을 요구). → 보류 정책 위반 위험. CI에 시크릿 넣지 말고 보고.
10. **사용자가 직접 수정한 흔적이 있는 파일(`render.yaml`, `package.json`, `requirements.txt`, 테스트)을 바꿔야 함**. → revert/동작변경 위험. 멈추고 확인.

---

## ADR
- 완료 후 `docs/adr/0028-ci-github-actions.md` 1장(200~300단어) 작성. (`docs/adr/` 디렉토리가 없으면 생성. ⚠ `.gitignore`가 `docs/cp*`를 무시하지만 `docs/adr/`는 무시하지 않으므로 일반 `git add`로 추적된다 — 확인 후 커밋.)
- 기록할 것 (표준 ADR: Context / Decision / Consequences):
  - **Context**: CI 부재, CP222/223/230으로 쌓은 안전망이 수동 실행에만 의존.
  - **Decision**: GitHub Actions 2-잡(backend/frontend) push+PR 게이트. 러너 = ubuntu CPU(GPU 테스트 0이므로). Python 3.11(render 일치). 임포트 루트 2종 처리(`PYTHONPATH=.:backend`). 캐싱 3종(pip/npm/Playwright). 테스트는 전부 mock이라 시크릿/Supabase 미연결.
  - **대안 기각 이유**: GPU 러너(불필요·고비용, 테스트가 CPU로 충분), backend 잡 2분할(불필요, 단일 pytest+이중 path가 단순), Supabase 통합테스트(보류 정책), CD 포함(범위 밖).
  - **Consequences**: main/develop 머지 전 회귀 자동 차단(단 branch protection은 UI로 켜야 발효). 캐싱으로 PR당 ~3분(Playwright)+torch 재설치 시간 절약. 향후 GPU 테스트 추가 시 skip 마킹 또는 별도 self-hosted 러너 필요.

---

## 자가 점검 결과 양식
종료 보고에 아래를 채운다.
- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: (CI는 read-path/검증만 추가. 학습·calibration·DB write·Supabase 실연결 없음. fidelity 보장 안전망(CP223 스냅샷)을 게이트로 강제하는지.)
- **[구조 결함]** PASS / WARN / FAIL — 사유: (임포트 루트 2종을 단일 pytest에서 올바로 처리했는지, 캐싱이 명령을 바꾸지 않는지, YAML 유효, 잡 분리 적절성.)
- **[모델 영향]** PASS / WARN / FAIL — 사유: (테스트/소스/모델 산출물 무변경. CPU CI에서 ai 테스트가 GPU 강제 없이 통과. 모델 학습·추론 파이프라인 무영향.)

---

## 산출물
- 변경/신규 파일:
  - 신규: `.github/workflows/ci.yml`
  - 신규: `docs/adr/0028-ci-github-actions.md`
- 리포트: `docs/cp237_report.md` (요구 / 한 일 / 결정 / 후속 — 필요한 만큼만). 후속에 반드시 포함:
  1. "GitHub Settings → Branches에서 `main`(+`develop`) branch protection의 required status checks(backend/frontend)를 **사용자가 UI에서 켜야** 머지 게이트가 실제 발효된다" (이 CP는 코드로 못 켬).
  2. "CP222/223/230 산출물이 변하면 CI 스텝(특히 mypy 대상 경로, frontend 스크립트명)을 동기화해야 함."
  3. (해당 시) "향후 GPU 의존 테스트가 추가되면 CPU CI에서 skip 마킹 필요."
