# CP224a BE 재현성 — requirements 핀 보강 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 단독 실행하는 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측하지 말고, 적힌 줄번호/파일경로를 직접 확인한 뒤 작업한다.

## 역할 고정

- **모드**: code (구현 + 자가 점검)
- **권한**: 코드(=requirements/문서 파일) 수정, 로컬 검증(가상환경 freeze, pip dry-run, render 빌드 영향 추론)
- **금지**:
  - 새 학습(training) 실행 금지
  - 새 calibration 실행 금지
  - DB write 금지
  - Supabase 호출 금지
  - 사용자가 직접 수정한 파일 revert 금지 (특히 `backend/requirements.txt` 의 torch 무력화 주석 — 의도된 것이므로 절대 되살리지 마라)
- **자가 점검**: 작업 종료 시 [Plan v3 정합] / [구조 결함] / [모델 영향] 3축 PASS/WARN/FAIL 보고 (양식은 맨 아래)
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **가상환경**: `.venv` (Python **3.10.0**, `torch==2.11.0+cu128`)
  - venv python 절대경로: `C:\Users\user\lens\.venv\Scripts\python.exe`
  - freeze 명령: `.venv\Scripts\python.exe -m pip freeze`
- **백엔드 기동(검증용, 필요 시만)**: `scripts\start_demo.ps1` 또는 `uvicorn app.main:app` (이 CP는 코드 동작을 안 바꾸므로 기동 검증 불필요. 포트 점유 피하려면 기동하지 마라.)
- **프론트**: 이 CP 범위 아님 (`npm run dev` 불필요)
- **포트 충돌 주의**: 8000/3000 이미 떠 있을 수 있음. 이 CP는 서버 기동이 필요 없으므로 새로 띄우지 마라.
- **주의 — Python 버전 불일치 사실**:
  - 로컬 venv = **3.10.0**
  - `render.yaml` `PYTHON_VERSION` = **3.11.11** (line 13)
  - `README.md` badge = **3.11**
  - 이 CP에서 버전 정합을 강제로 맞추지 마라. 핀은 "실제 설치된 버전"(=venv freeze)을 진리로 삼되, render 가 3.11 로 빌드한다는 사실을 ADR/리포트에 기록만 한다. (3.10↔3.11 차이로 일부 핀이 render 에서 해석 실패하면 차단 트리거.)

## 진단 (근거)

조사 출처: 아래 모든 인용은 이 CP 작성 시 `Read`/`Grep`/`pip freeze` 로 직접 확인한 실측이다.

### 문제 1 — 루트 `requirements.txt` 가 핵심을 안 핀한다 (재현성 구멍 본체)

`C:\Users\user\lens\requirements.txt` (현재 **3줄**):

```
1  optuna==4.8.0
2  optuna-dashboard==0.20.0
3  matplotlib==3.10.9
```

이 파일은 **dev/training/HPO 환경** 매니페스트다. 근거: optuna/matplotlib 를 실제로 쓰는 코드는 `ai/` 디렉토리의 학습·튜닝 스크립트뿐이다 (`grep` 실측).

`ai/` 디렉토리 top-level import 집계 (`grep -rhoE "^(import|from) ..." ai/` 실측, 외부 라이브러리만 발췌):

| 모듈 | 등장 파일 수 | 루트 requirements 핀 여부 |
|---|---|---|
| pandas | 125 | ❌ 안 핀 |
| numpy | 119 | ❌ 안 핀 |
| torch | 77 | ❌ 안 핀 |
| sklearn (scikit-learn) | 8 | ❌ 안 핀 |
| optuna | 6 | ✅ 핀됨 |
| scipy | 5 | ❌ 안 핀 |
| matplotlib | 2 | ✅ 핀됨 |

→ 학습/재현에 결정적인 **torch / pandas / numpy / scikit-learn / scipy** 가 전부 unpin. 이 루트 파일로 환경을 재구성하면 optuna/matplotlib 만 고정되고 나머지는 그때그때 최신으로 끌려와 학습 결과 재현이 깨질 수 있다. **이것이 본 CP가 메우는 구멍이다.**

루트 `requirements.txt` 최종 변경: commit `2bbdff2` (2026-04-27). 그 뒤 방치.

### 문제 2 — render 배포는 루트가 아니라 `backend/requirements.txt` 를 쓴다 (핀 위치 함정)

`C:\Users\user\lens\render.yaml`:

```
5      rootDir: backend
7      buildCommand: pip install -r requirements.txt
```

`rootDir: backend` 이므로 line 7 의 `requirements.txt` 는 **`backend/requirements.txt`** 로 해석된다. 루트 파일이 아니다.

`C:\Users\user\lens\backend\requirements.txt` (현재 **14줄**, 실측):

```
1   fastapi==0.111.0
2   uvicorn==0.30.1
3   supabase==2.5.0
4   pandas==2.2.2
5   pyarrow==16.1.0
6   python-dotenv==1.0.1
7   numpy==1.26.4
8   scikit-learn==1.5.0
9   # torch는 CUDA 빌드라 별도 설치 (https://pytorch.org/get-started/locally/)
10  # 5060 Ti (sm_120)는 cu128 빌드 + torch>=2.7 권장
11  # torch==2.3.1  # 옛날 핀 — CPU 빌드로 덮어쓰는 문제 → 무력화
12  statsmodels==0.14.2
13  httpx==0.27.0
14  wandb==0.19.11
```

즉 **배포용 파일은 이미 핵심을 핀하고 있고 torch 는 의도적으로 제외**돼 있다(line 9~11, 사용자가 직접 무력화한 주석 — revert 금지 대상). 따라서:
- render 배포 빌드는 이 CP 범위에서 **건드리지 않는다** (이미 핀됨, torch 없음 → CPU free tier 정상).
- 재현성 구멍은 **루트 파일(dev/training)** 에만 있다.

### 문제 3 — torch cu128 은 PyPI 핀 불가

`pip freeze` 실측:

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
```

`+cu128` 은 **로컬 버전 라벨**이라 일반 PyPI 인덱스에 없다. 루트 `requirements.txt` 에 `torch==2.11.0+cu128` 을 그대로 적으면 `--index-url https://download.pytorch.org/whl/cu128` 같은 별도 인덱스 없이는 `pip install` 이 즉시 실패한다. → torch 는 **핀 본문에 박지 말고 주석 + 별도 설치 안내**로 처리 (backend/requirements.txt 와 동일 전략).

### 확보된 실제 버전 (루트 dev requirements 에 반영할 진리값, `pip freeze` 실측)

| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.11.0+cu128 | **핀 본문 금지**, 주석 처리 |
| torchvision | 0.26.0+cu128 | 〃 (필요 시) |
| numpy | 1.26.4 | |
| pandas | 2.2.2 | |
| scipy | 1.15.3 | 어느 requirements 에도 현재 없음 |
| scikit-learn | 1.5.0 | |
| statsmodels | 0.14.2 | arch 동반 |
| arch | 8.0.0 | GARCH 베이스라인 (cp203_baselines 등) |
| optuna | 4.8.0 | 이미 핀됨 |
| optuna-dashboard | 0.20.0 | 이미 핀됨 |
| matplotlib | 3.10.9 | 이미 핀됨 |
| pyarrow | 16.1.0 | parquet I/O |
| (참고) fastapi | 0.111.0 | backend 전용, 루트 불필요 |
| (참고) httpx | 0.27.0 | backend 전용 |
| (참고) pydantic | 2.13.3 | fastapi 동반 |

> 주: scipy/statsmodels/arch 는 `ai/` 학습·통계 검정에서 쓰이지만 루트 핀에 빠져 있다. statsmodels/arch 포함 여부는 Step 2 에서 `ai/` import 실측으로 최종 판단 (현재 `import statsmodels`/`import arch` 가 `ai/` 에 있는지 grep 확인 후 추가).

## 선행 의존

- **없음.** 이 CP 는 텍스트 의존성 매니페스트만 손대며 코드 동작/스냅샷에 영향이 없다. CP223(백엔드 characterization 스냅샷) 그린을 기다릴 필요 없다.
  - 단, **backend 코드 구조 분리(CP225~)는 여전히 CP223 그린 전 금지** — 본 CP는 거기에 해당하지 않는다.

## 범위

### 포함

1. `.venv` 전체 freeze 를 docs 에 스냅샷으로 보존.
2. 루트 `C:\Users\user\lens\requirements.txt` 에 dev/training 핵심 런타임 의존성 핀 추가 (torch 제외, 주석 처리).
3. torch cu128 의 render 배포 빌드 영향 확인 → 영향 없음을 근거와 함께 리포트 (배포는 `backend/requirements.txt` 사용, torch 미포함).

### 제외

- **`backend/requirements.txt` 수정 금지.** 이미 핀돼 있고 torch 무력화는 사용자 의도(revert 금지). 손대지 마라.
- **render.yaml 수정 금지.** 빌드 경로/버전 변경 없음.
- **Supabase 관련 일체 보류** (호출·키·sync). 매니페스트에 supabase 핀을 루트에 추가하지 마라 (루트는 training 용, supabase 미사용).
- **torch 를 PyPI 핀으로 박는 행위 금지** (문제 3).
- **Python 버전 강제 정합 금지** (3.10 vs 3.11 불일치는 기록만).
- `backend/collector/requirements.txt`, `backend/db/requirements-*.txt` 수정 금지 (별도 파이프라인, 범위 밖).

## Sub-step (Strangler Fig, 작은 단위)

> 이 CP의 "옛 코드"는 `requirements.txt` 의 3줄짜리 부분 핀이다. 새 핀 블록을 추가(공존)한 뒤, 루트 매니페스트의 단일 진리로 승격하고, 옛 부분 핀이 새 블록에 흡수됐는지 확인하는 흐름이다. 각 Step = 한 commit = 한 revert 단위.

### Step 0 — 사실 재확인 (코드 변경 없음, commit 없음)

실행자는 작업 전 아래를 직접 재확인한다 (이 지시서의 줄번호가 드리프트했을 수 있음):

```powershell
# 루트 vs 백엔드 requirements 현재 내용
Get-Content C:\Users\user\lens\requirements.txt
Get-Content C:\Users\user\lens\backend\requirements.txt
# render 가 어떤 requirements 를 쓰는지 (rootDir + buildCommand)
Select-String -Path C:\Users\user\lens\render.yaml -Pattern 'rootDir|buildCommand'
```

기대: 루트 3줄(optuna/optuna-dashboard/matplotlib) / backend 14줄(torch 주석 무력화 포함) / render line5 `rootDir: backend` + line7 `pip install -r requirements.txt`.

만약 위와 다르면(특히 backend torch 주석이 사라졌거나 render 경로가 바뀌었으면) **차단 트리거** → 멈추고 보고.

### Step 1 — venv freeze 스냅샷 보존 (commit 1)

1. 전체 freeze 를 docs 에 파일로 저장:

```powershell
.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 C:\Users\user\lens\docs\cp224a_venv_freeze.txt
```

2. `ai/` 에서 statsmodels/arch 사용 여부 실측(Step 2 핀 결정 근거):

```powershell
Select-String -Path C:\Users\user\lens\ai\*.py -Pattern '^(import|from) (statsmodels|arch|scipy|sklearn|scikit)' | Select-Object -First 20
```

3. **검증**: `docs\cp224a_venv_freeze.txt` 가 생성되고 `torch==2.11.0+cu128` 라인을 포함하는지 확인.

```powershell
Select-String -Path C:\Users\user\lens\docs\cp224a_venv_freeze.txt -Pattern 'torch==2.11.0\+cu128'
```

4. **commit 1**: `chore(cp224a): snapshot .venv freeze for reproducibility baseline`
   - 변경: `docs/cp224a_venv_freeze.txt` (신규)
   - lint/tsc/pytest 해당 없음 (docs 전용). 코드 미변경.

### Step 2 — 루트 requirements.txt 에 핀 블록 추가 (옛 3줄 옆 공존) (commit 2)

> 기존 3줄을 지우지 말고, 위에 핵심 핀 블록을 **추가**하고 기존 3줄은 "HPO/plotting" 그룹으로 남긴다(공존). 한 commit.

추출 순서 원칙(순수→I/O→상태)에 맞춰, 부작용 없는 수치 라이브러리부터 배치한다. 아래를 루트 `C:\Users\user\lens\requirements.txt` 전체로 작성한다 (Step 1 의 statsmodels/arch 실측 결과에 따라 해당 두 줄 포함/제외):

```
# ── Lens dev / training (ai/) 재현 환경 ──
# 배포(render)는 backend/requirements.txt 를 쓴다. 이 파일은 학습·튜닝·통계 검정용.
# 버전 진리값: docs/cp224a_venv_freeze.txt (.venv freeze, Python 3.10.0)

# torch 는 CUDA 빌드(+cu128 로컬 라벨)라 PyPI 핀 불가.
# 별도 설치: https://pytorch.org/get-started/locally/  (5060 Ti sm_120 → cu128, torch>=2.7)
# 현재 검증된 빌드: torch==2.11.0+cu128 / torchvision==0.26.0+cu128

# 수치 코어
numpy==1.26.4
pandas==2.2.2
scipy==1.15.3
pyarrow==16.1.0

# ML / 통계
scikit-learn==1.5.0
statsmodels==0.14.2   # ai/ 통계 검정·베이스라인에서 사용 시 유지 (Step1 실측으로 확정)
arch==8.0.0           # GARCH 베이스라인 (cp203_baselines 등) 사용 시 유지

# HPO / plotting (기존 핀 유지)
optuna==4.8.0
optuna-dashboard==0.20.0
matplotlib==3.10.9
```

- statsmodels/arch 가 `ai/` 에 import 되지 않으면 두 줄 제거하고 그 사실을 리포트에 적는다.
- 버전은 **반드시 `docs/cp224a_venv_freeze.txt` 의 실측값과 일치**시킨다. 임의 버전 금지.

**검증 (핀 해석 가능성 dry-run, 실제 설치 금지)**:

```powershell
# 새 venv 오염 방지를 위해 --dry-run + --ignore-installed 로 resolver 만 돌린다.
.venv\Scripts\python.exe -m pip install --dry-run --ignore-installed -r C:\Users\user\lens\requirements.txt
```

기대: resolver 가 충돌/PyPI 미존재 없이 끝난다. torch 라인이 본문에 없으므로 `+cu128` 해석 실패가 발생하면 안 된다. (오류 나면 차단 트리거.)

**commit 2**: `chore(cp224a): pin core training deps in root requirements`
- 변경: `requirements.txt`
- pytest 해당 없음(코드 미변경). dry-run resolver 통과가 검증.

### Step 3 — render 배포 빌드 영향 확인 (코드 변경 없음, 분석 + 리포트) (commit 3, 리포트만)

> 옛(우려) 가설 = "torch cu128 핀이 render 빌드를 깬다"를 제거하는 단계. 실제 배포 경로를 근거로 영향 없음을 확정한다.

1. render 가 쓰는 파일이 `backend/requirements.txt` 임을 재확인(Step 0 결과 인용).
2. 그 파일에 torch 핀이 **없음**을 재확인 (line 9~11 주석).
3. 루트 `requirements.txt` 변경은 render 빌드 입력이 **아님**을 명시.
4. 결론: **render 배포 빌드 영향 = 없음** (torch 미포함 + 루트 파일 미사용).

이 단계는 코드/설정 변경이 없다. 결과를 `docs/cp224a_report.md` 에 기록(아래 산출물 참조).

**commit 3**: `docs(cp224a): record render build impact analysis + report`
- 변경: `docs/cp224a_report.md` (+ 필요 시 `docs/adr/0014-deploy-vs-dev-requirements-split.md`)

## 인터페이스 보존

- 이 CP 는 함수 signature / API 응답 schema / 프론트 props 를 **일절 바꾸지 않는다.** 코드 파일을 수정하지 않으므로 자동 보존.
- 변경 대상은 의존성 매니페스트(`requirements.txt`)와 docs 뿐.
- 만약 핀 추가가 어떤 패키지의 **메이저 다운그레이드**를 강제하게 되면(예: 누군가 이미 더 높은 버전을 설치한 상태) 그것은 환경 동작 변경이므로 호출자(=학습 스크립트) 영향이 생김 → 차단 보고. (단, 핀 값은 freeze 실측과 동일하므로 정상 경로에선 발생하지 않아야 함.)

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 | 측정 방법 |
|---|---|---|---|
| 루트 requirements.txt 핀된 핵심 런타임 수 | 3 (optuna/optuna-dashboard/matplotlib) | ≥ 9 (numpy/pandas/scipy/pyarrow/scikit-learn + 기존 3 + 필요 시 statsmodels/arch) | `Get-Content requirements.txt` 라인 카운트 |
| torch 본문 핀 | 해당 없음 | **0개** (주석으로만 안내) | requirements.txt 에 `torch==` 활성 라인 없음 |
| 핀 버전 ↔ venv freeze 일치 | — | 100% 일치 | freeze 와 대조 |
| pip resolver dry-run | — | 충돌/미존재 0 | Step 2 dry-run exit 0 |
| render 배포 빌드 깨짐 | — | **0 (영향 없음)** | Step 3 분석: backend/requirements.txt 사용 + torch 미포함 |
| 코드 동작 변경 | — | 0 (코드 미수정) | git diff 가 .py 미포함 |
| 예상 시간 | — | ~0.5~1시간 | — |

> pytest/snapshot/tsc/mypy 항목은 **해당 없음** (코드 미변경, 프론트 미변경) → 생략.

## 검증

전체를 한 번에 재확인하는 명령 묶음:

```powershell
# 1) freeze 스냅샷 존재 + torch cu128 포함
Test-Path C:\Users\user\lens\docs\cp224a_venv_freeze.txt
Select-String -Path C:\Users\user\lens\docs\cp224a_venv_freeze.txt -Pattern 'torch==2.11.0\+cu128'

# 2) 루트 핀: 핵심 추가됨 + torch 본문 핀 없음
Get-Content C:\Users\user\lens\requirements.txt
Select-String -Path C:\Users\user\lens\requirements.txt -Pattern '^numpy==|^pandas==|^scipy==|^scikit-learn=='   # 매치 4
Select-String -Path C:\Users\user\lens\requirements.txt -Pattern '^torch=='                                     # 매치 0 (주석은 ^# 라 미매치)

# 3) 핀 값이 freeze 와 일치 (수동 대조)
Select-String -Path C:\Users\user\lens\docs\cp224a_venv_freeze.txt -Pattern '^(numpy|pandas|scipy|scikit-learn|statsmodels|arch|pyarrow)=='

# 4) resolver dry-run (설치 금지)
.venv\Scripts\python.exe -m pip install --dry-run --ignore-installed -r C:\Users\user\lens\requirements.txt

# 5) backend/render 불변 확인 (수정 안 했는지)
git diff --name-only
# → backend/requirements.txt, render.yaml 이 목록에 있으면 안 됨
```

기대 결과:
- (1) `True` + cu128 매치 1건
- (2) numpy/pandas/scipy/scikit-learn 각 1매치, `^torch==` 0매치
- (4) resolver "Would install ..." 로 정상 종료, 에러 없음
- (5) diff 에 `requirements.txt`, `docs/cp224a_*` 만. backend/render 없음.

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **torch 핀이 render 배포 빌드를 깬다고 판단되는 정황** — 즉, 누군가 루트가 아니라 `backend/requirements.txt` 에 torch 를 박았거나, render `buildCommand`/`rootDir` 가 루트 파일을 쓰도록 바뀌어 있으면 → 멈추고 보고. **배포 requirements 분리 전략은 사용자 판단 사항.** (스펙: "torch cu128 핀이 render 배포 빌드를 깨면 즉시 멈추고 보고.")
2. **`pip install --dry-run` resolver 가 충돌/미존재로 실패** — 특히 `+cu128` 미해석, 또는 핀 버전이 PyPI 에 없음. 임의로 버전을 낮추거나 핀을 빼지 말고 보고.
3. **핀 버전과 `docs/cp224a_venv_freeze.txt` 실측이 불일치** — freeze 가 진리. 추측 버전을 박지 말고 보고.
4. **`backend/requirements.txt` 의 torch 무력화 주석(line 9~11)이 이미 사라졌거나 torch 활성 핀이 들어가 있음** — 사용자 의도 훼손 신호. revert 시도 금지, 보고.
5. **Step 0 재확인 결과가 이 지시서의 인용(루트 3줄 / backend 14줄 / render rootDir=backend)과 다름** — 코드가 드리프트함. 무리하게 진행하지 말고 현재 상태를 보고.
6. **3.10↔3.11 차이로 특정 핀이 render(3.11) 에서 해석 불가** 가능성이 보이면(예: 3.10 전용 빌드만 존재하는 패키지) → 보고. (루트는 training 전용이라 render 와 무관하지만, 만약 분리 전략이 필요하다는 판단이 서면 사용자에게 올린다.)

## ADR

`docs/adr/0014-deploy-vs-dev-requirements-split.md` (200~300단어) 작성.
- `docs/adr/` 디렉토리가 **없으면 생성**한다(현재 미존재).
- 기록할 결정: "배포용(backend/requirements.txt, CPU, torch 제외)과 dev/training용(루트 requirements.txt, torch 는 별도 cu128 설치) 의존성 매니페스트를 의도적으로 분리한다." 이유(render free tier CPU 빌드 vs 로컬 GPU 학습), torch cu128 을 PyPI 핀하지 않는 근거, Python 3.10(local)/3.11(render) 불일치를 인지하되 본 CP 범위에서 통일하지 않는 결정, 후속(필요 시 dev requirements 를 `requirements-dev.txt` 로 개명/분리 검토)을 적는다.

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다 (빈칸 금지, 사유 1줄):

- **[Plan v3 정합]**: PASS / WARN / FAIL — 사유: ____ (Plan v3 의 EODHD 유지·fidelity 우선 등과 충돌 없는지. 의존성 핀은 모델 정책 불변 → PASS 예상)
- **[구조 결함]**: PASS / WARN / FAIL — 사유: ____ (dev/배포 매니페스트 분리가 구조적으로 일관적인지, 루트 파일 역할이 명확해졌는지)
- **[모델 영향]**: PASS / WARN / FAIL — 사유: ____ (핀 값=freeze 실측이라 현재 설치 환경 무변 → 학습/추론 동작 불변. 다운그레이드 강제 시 WARN/FAIL)

## 산출물

- **변경 파일 목록**:
  - `requirements.txt` (핀 블록 추가)
  - `docs/cp224a_venv_freeze.txt` (신규, freeze 스냅샷)
  - `docs/adr/0014-deploy-vs-dev-requirements-split.md` (신규)
  - `docs/cp224a_report.md` (신규)
- **`docs/cp224a_report.md`** (필요한 만큼만, 과하지 않게):
  - **요구**: 루트 requirements 재현성 구멍 메우기.
  - **한 일**: freeze 스냅샷 / 루트 핀 추가(목록) / torch 주석 처리 / render 영향 분석.
  - **결정**: 배포 vs dev 분리, torch PyPI 핀 제외, Python 버전 불일치 기록.
  - **후속**: (선택) `requirements-dev.txt` 로 명시적 분리, render 3.11 ↔ local 3.10 통일 검토, collector/db requirements 핀 점검은 별 CP.
