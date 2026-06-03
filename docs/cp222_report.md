# CP222 보고서 — BE/FE 공통 안전망 도구 구축

- **상태**: ✅ 완료
- **기간**: 2026-06-03 (단일 세션)
- **모드**: code (구현 + 자가점검)
- **선행 의존**: CP221 (라이브 500 fix) ✅
- **위치**: 마지막 그린 커밋 = (Step 7 커밋, 본 보고서/ADR 포함)

---

## 요구

품질 도구 0 → 안전망 1차(ruff/pytest/coverage/mypy/pre-commit) 구축. 기존 51 파일 / 397 테스트 함수를 pytest로 수집해 자동 회귀 가능 상태로. **소스 코드 변경 없이** 도구만 설치하고 baseline 박기.

---

## 한 일 (Sub-step 별)

### Step 1 — 설정 파일 3종 (커밋 `1a0d870`)
- `pyproject.toml` 신규: ruff(E,F,I,B,UP,SIM, line=100, py310), pytest(pythonpath=[".","backend"]), coverage(branch+source), mypy(loose initial + namespace_packages 추후 보강).
- `.pre-commit-config.yaml` 신규: ruff(check --fix + format) + mypy hooks. **install only, run --all-files 비실행.**
- `requirements-dev.txt` 신규: pytest 8.3.3 / pytest-cov 5.0.0 / ruff 0.6.9 / mypy 1.11.2 / pre-commit 3.8.0 / httpx 0.27.0 / pandera 0.20.4.

### Step 2 — venv 설치 (커밋 `a780519`)
- `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` 성공.
- 인코딩 정정 1건: pip 21.2.3 (.venv)가 한국 Windows에서 requirements를 cp949로 읽어 한글 주석에 `UnicodeDecodeError`. 영문 주석으로 정정 → `a780519`.
- 설치 버전: pytest 8.3.3 / ruff 0.6.9 / mypy 1.11.2 / pre-commit 3.8.0.
- ruff 0.6.9 == `.pre-commit-config.yaml` rev `v0.6.9` 일치 → 정정 불필요.

### Step 3 — pytest baseline (커밋 없음, 측정만)

| 지표 | 값 |
|---|---|
| collected | **382** (디스크 51 파일 / 함수 ≈397) |
| passed | **361** (94.5%) |
| failed | **21** (5.5%) |
| collection error | **1** (`test_services.py`) |
| 시간 | ~17~20초 |
| 차단 트리거 1 (30% 실패) | **미달** |

**실패 21개는 모두 pre-existing, CP222가 만든 신규 실패 0:**
- `test_api.py` x7: `app.routers.v1.stocks`에 `get_latest_prediction_data` / `get_prediction_history_data` 함수 부재 (CP214/216 라우터 진화 시 함수 이동·통폐합).
- `test_collector_jobs.py` x3: assertion count drift (compute_indicators 동작 변경).
- `test_market_data_providers.py` x1: 기본 provider `'yahoo' != 'eodhd'` (yfinance fallback 정책 변경).
- `test_product_prediction_history_api.py` x3: 라우터 함수 부재 + fixture 날짜 stale (`'2026-06-01' != '2026-05-04'`) + mock signature drift.
- `test_inference_backtest.py` x1: anchor close fixture 부재.
- `test_overfit_tiny_batch.py` x1 + `test_sweep_caching.py` x3: `DatasetPlan.__init__` 옛 시그니처 호출 (`absolute_min_rows`/`required_history_rows` 누락).
- `test_preprocessing_cache_isolation.py` x2: snapshot 모드 source_data_hash 계산용 parquet 부재.

**Collection error 1**: `test_services.py:12` 가 `app.services.api_service.get_latest_prediction_data` import 시도 → 함수 부재.

**지시서 진단 5 정정**: 지시서는 `scripts/cp134_local_daily_update_rehearsal.py` 부재 가정했으나 **실제 디스크에 존재** (gitignore가 `git ls-files`에서만 가렸을 뿐). `test_collector_jobs.py:20` 의 collection error는 **발생하지 않음**.

**테스트 코드 무수정**: 21 실패 + 1 error 모두 지시서 룰("기존 테스트 로직 수정 금지")대로 **소스 변경 없이 기록만**.

### Step 4 — ruff 검출 (`--fix` 절대 금지)
- 총 검출 **약 2155건** (--statistics 통계):
  - E501 line-too-long: **1718** (line=100)
  - E402 module-import-not-at-top: 165
  - I001 unsorted-imports: 117
  - F401 unused-import: 28
  - UP038 non-pep604-isinstance: 25
  - SIM117 multiple-with: 17
  - UP006/UP007 non-pep585/non-pep604-annotation: 12/12
  - UP035 deprecated-import: 11
  - F541 f-string-missing-placeholders: 9
  - F841 unused-variable: 8
  - B905 zip-without-strict: 5, E712: 5
  - 기타 SIM/UP/B/E 잡것: ~30
- **잠재 실제 버그 신호** (CP224b 우선 처리 후보):
  - `F821` undefined-name: **2건**
  - `B023` function-uses-loop-variable: **1건**
- 차단 트리거 4("수천 건 검출") **미달**(정상 레거시 풍경).
- `--fix` 절대 미실행.

### Step 5 — mypy + pre-commit install
- mypy 1차 실행 시 크래시: `Source file found twice under different module names: "blocks" and "ai.models.blocks"` (ai/__init__.py 부재 + 기본설정).
- **소스 임의 변경 대신 mypy 설정만 보강**: pyproject.toml에 `namespace_packages = true / explicit_package_bases = true / mypy_path = "."` 추가. 지시서 "ai/__init__.py 임의 신설 금지" 룰 준수.
- 재실행 결과: **Found 1391 errors in 130 files (checked 251 source files)**. baseline 기록만, 수정 0.
- `pre-commit install` 성공: `.git\hooks\pre-commit` 등록 확인. `run --all-files` 비실행.

### Step 6 — frontend baseline
- `cd frontend; npx tsc --noEmit`: **PASS** (출력 0줄 = 에러 0).
- FE 도구(ESLint/Prettier/vitest) 도입은 CP230 트랙 → CP222에서 미실행.

### Step 7 — 보고서 + ADR
- `docs/adr/0010-quality-tooling-ruff-pytest.md` 신규.
- 본 보고서 `docs/cp222_report.md` 신규.

---

## 운영 parquet 덮어쓰기 의심 조사 (CP222.5 후속)

**경위**: Step 5 마지막에 `git status`에서 `backend/data/v1/market_indicators_1d.parquet` / `market_prices_1d.parquet` 2개가 modified로 등장 (+6KB, +10KB binary diff 확인). 런북 §0.8 "운영 parquet 덮어쓰기 금지" 잠재 위반.

**즉시 대응**: 두 parquet `git checkout`으로 복원 (커밋 `e810711` daily refresh 시점). 1차 색출 진행.

**색출 결과 (4회 재현 시도)**:
| 시나리오 | parquet 변경 |
|---|---|
| backend/tests 단독 | ✗ |
| ai/tests 단독 | ✗ |
| 첫 전체 실행 (rollback 전) | ✓ |
| 두 번째 전체 (rollback 후) | ✗ |
| Cold start 전체 (__pycache__/`.pytest_cache` 정리 후) | ✗ |

**해석**: 어제(2026-06-02) `build_v1_market_local` 같은 빌드 스크립트가 작업 트리에 modified 잔여 남겼고, 첫 pytest read access가 git stat 캐시를 무효화하면서 그제서야 modified로 노출됐을 가능성 가장 높음. **pytest 자체 부수효과 가능성 낮음** (재현 0/4). 차단 트리거 미달.

**원인 추정의 한계**: 1차 색출은 의심 모듈을 특정하지 못한 채 끝났다. 재현 안 됨 = "원인 모름 + 재폭발 가능성 잔존". → **사용자 결정으로 영구 가드 도입**.

### CP222 보강 — v1 parquet 영구 가드 (커밋 후행, 2026-06-03)

**도입 동기**: 차단 트리거(런북 §0.8 "운영 parquet 덮어쓰기 금지")가 한 번 발동한 이상, 원인 모르고 진행하면 매 pytest 실행이 잠재 폭탄. CP223 baseline + CP230 screenshot이 그 위에 박혀 있어 오염 1회면 안전망 전체 신뢰도 손상.

**구현 위치**: `backend/tests/conftest.py` (운영 코드 0 변경 원칙 유지).

**설계**:
- `@pytest.fixture(scope="session", autouse=True)` `_guard_v1_parquet_integrity`.
- **세션 시작**: `backend/data/v1/*.parquet` 각 파일의 SHA256 박제.
- **세션 종료 (yield 후)**: SHA256 재비교.
- **변경 감지 시**: `git checkout -- <파일>` 즉시 복원 + `sys.__stderr__`로 경고 출력(pytest stdout 캡처 우회). 어떤 테스트가 어떤 경로로 만져도 자동 복원.
- **복원 실패 시**: 차단 트리거 — `git status backend/data/v1/`가 dirty로 남아 즉시 발견.

**검증 (가드 적용 후 실측)**:
- `pytest backend/tests --continue-on-collection-errors -q`: **87 passed / 11 failed / 1 error** (CP223 baseline 동일, 회귀 0).
- 가드 경고 출력: **0 건** (이번 실행에서 오염 없음).
- `git status --porcelain backend/data/v1/`: **빈 출력** (clean).
- 가드는 박혀 있어 다음 어떤 pytest 실행에서 오염 발생하면 자동 복원 + 경고.

**효과**:
- 런북 §0.8 영구 보장. 어떤 테스트가 운영 v1 parquet를 만져도 즉시 원복.
- "원인 모름" 상태에서 분리 리팩토링(CP225+) 시작 가능. 안전망 신뢰도 회복.
- CP223 9 snapshot baseline의 입력 데이터 무결성 영구 보호.

---

## 인터페이스 보존 (성공 기준)

- 소스 코드 **0 라인** 변경. `backend/app`, `ai/`, `frontend/src` 무수정.
- 테스트 코드 **0 라인** 변경 (21 실패 + 1 error 그대로 기록).
- API 응답 schema 무변경. 함수 signature 무변경. props 무변경.
- 운영 parquet 무수정 (rollback 완료).

---

## 성공 기준 충족표

| 항목 | 기대 | 실측 | 결과 |
|---|---|---|---|
| pytest 수집 | 크래시 없이 리포트 | 382 collected | ✅ |
| pytest 통과 baseline | passed > 0, 신규 실패 0 | passed 361, 신규 0 | ✅ |
| 수집 에러 허용 | gitignore 부재 import 류만 | 1건 (test_services stale import) | ✅ |
| ruff 실행 | 통계 정상 | 약 2155건 통계 출력 | ✅ |
| mypy 실행 | 리포트 정상 (크래시 X) | 1391 errors / 130 files | ✅ |
| pre-commit | install 성공 | `.git/hooks/pre-commit` 등록 | ✅ |
| 설정 3종 | pyproject + pre-commit + req-dev | 3개 생성 | ✅ |
| 소스 변경 | 0 라인 | 0 라인 | ✅ |

---

## 후속

- **CP223 (다음)**: 백엔드 characterization snapshot (syrupy ↔ snaptol 택1).
- **CP224a**: requirements 버전 핀.
- **CP224b**: ruff 안전 자동수정 (UP, I001, F401 위주 + F821/B023 잠재 버그 우선 검토).
- **별도 작은 작업** (CP223 전): `conftest.py` v1 parquet hash 가드 신설.
- **테스트 stale reference 정리** (별도 cleanup CP): test_api.py 7건 + test_services.py / test_product_prediction_history_api.py / DatasetPlan signature 4건. CP222 범위 밖.

---

## 자가 점검

- **[Plan v3 정합]** **PASS** — CP222는 도구 설치/설정만. 모델/밴드 로직, postprocess, RevIN, loss 무관. Plan v3 의사결정(α=1/β=2, fidelity 우선, EODHD 유지 등)에 영향 0.
- **[구조 결함]** **PASS** — pythonpath `[".", "backend"]` 결정, mypy namespace_packages 보강, 21 pre-existing 실패의 정직한 baseline 기록, 운영 parquet 위험 신호의 정직한 조사+격리. 진단 5 정정도 보고서에 명시.
- **[모델 영향]** **PASS** — 소스 0 변경. ForecastOutput, channel layout, ticker registry, sufficiency gate, dropout 위치 모두 무변경.
