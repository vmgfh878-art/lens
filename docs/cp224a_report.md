# CP224a 보고서 — 루트 requirements.txt 재현성 핀 보강

- **상태**: ✅ 완료
- **기간**: 2026-06-03 (CP222 보강 직후)
- **모드**: code (구현 + 자가 점검)
- **선행 의존**: 없음 (의존성 매니페스트만 수정)
- **마지막 그린 커밋**: Step 3 커밋

---

## 요구

루트 `requirements.txt`(3줄, optuna/optuna-dashboard/matplotlib만)는 ai/ 학습·튜닝 환경 매니페스트인데 torch/pandas/numpy/scipy 등 핵심을 unpin → 재현성 구멍. 본 CP는 그 구멍을 메운다 (코드 0 변경). 배포용 `backend/requirements.txt`는 이미 핀되어 있고 torch 의도적 주석 처리(사용자 수정) 상태라 손대지 않는다.

---

## 한 일 (Sub-step 별)

### Step 1 — venv freeze 스냅샷 + ai/ 통계 import 실측 (커밋 `d39df9b`)
- `docs/cp224a_venv_freeze.txt` 신규 (151 라인). Python 3.10.0 + torch 2.11.0+cu128 포함.
- `.gitignore` 예외 패턴 확장: `cp22[1-9]*.txt / cp23[0-7]*.txt` 추가.
- ai/ import 실측:
  - scipy: **5건** (`ai/eval/significance/{dm_test,baselines,garch_walkforward,gw_test,metrics}.py` 모두 `from scipy.stats`)
  - scikit-learn: **0건** (grep `sklearn` ai/ 전체에서 0)
  - statsmodels: **0건**
  - arch: **0건**
- **지시서 진단 오류 발견**: 지시서가 "sklearn 8건"이라 했으나 실측 0건. 핀 결정에 반영.

### Step 2 — 루트 requirements.txt 핀 추가 (커밋 `bd2433b`)
- 영문 주석 (pip 21.2.3 cp949 디코딩 회피, CP222 Step 2 동일 정책).
- **추가**: numpy 1.26.4 / pandas 2.2.2 / scipy 1.15.3 / pyarrow 16.1.0.
- **유지**: optuna 4.8.0 / optuna-dashboard 0.20.0 / matplotlib 3.10.9.
- **본문 금지(주석만)**: torch 2.11.0+cu128, torchvision 0.26.0+cu128.
- **제외**: scikit-learn / statsmodels / arch (ai/ 사용 0건 실측).
- 검증:
  - pip 21.2.3엔 `--dry-run` 미지원.
  - 대체: 7개 핀 모두 `docs/cp224a_venv_freeze.txt`와 정확 일치 확인 (resolver가 venv 생성 시 통과한 증거).

### Step 3 — render 영향 분석 + ADR + 보고서 (이 커밋)
- render.yaml 직접 재확인:
  - `rootDir: backend` (line 5)
  - `buildCommand: pip install -r requirements.txt` (line 7) → backend/requirements.txt 해석
  - `PYTHON_VERSION: 3.11.11` (line 13)
- backend/requirements.txt는 본 CP에서 무수정, torch 의도적 주석 유지(line 9~11) 확인.
- **결론: render 배포 빌드 영향 = 0** (루트 파일 변경은 render 입력 아님, backend 파일 torch 미포함).
- `docs/adr/0014-deploy-vs-dev-requirements-split.md` 신규.

---

## 인터페이스 보존

- 코드 파일(.py / .ts 등) **0 라인 변경**.
- backend/requirements.txt / render.yaml / backend/collector/requirements.txt / backend/db/requirements-*.txt 모두 무수정.
- 학습/추론/API 동작 불변.

---

## 핵심 컴포넌트 존재 체크리스트 (메타 D21)

- `docs/cp224a_venv_freeze.txt` 신규 ✅ (151 라인, torch==2.11.0+cu128 포함)
- 루트 requirements.txt 핀 4개 추가 (numpy/pandas/scipy/pyarrow) ✅
- 기존 3핀 유지 (optuna/optuna-dashboard/matplotlib) ✅
- torch 본문 핀 0 ✅
- 7/7 핀이 freeze와 일치 ✅
- backend/requirements.txt 무수정 ✅
- render.yaml 무수정 ✅
- 코드(.py/.ts) 0 변경 ✅

## 새 테스트 결과 (메타 D21)

해당 없음 (코드 미변경, 테스트 영향 0). 기존 backend/tests 회귀 검증은 CP222 보강 시점 그대로 (87 passed).

## Dry-run 결과 (메타 D21)

pip 21.2.3 `--dry-run` 미지원 → 7개 핀의 freeze 일치로 대체 검증 (resolver 통과 증명).

## 기존 회귀 통과 건수 (메타 D21)

코드 0 변경이므로 회귀 0 자동 보장. backend/tests 87 passed 유지.

---

## 성공 기준 충족표

| 항목 | 시작 | 목표 | 실측 | 결과 |
|---|---|---|---|---|
| 루트 핀된 핵심 런타임 | 3 | ≥9 | **7** (사용 0 패키지 제외 의도) | ⚠️ 목표 미달이나 의도적 |
| torch 본문 핀 | — | 0 | **0** (주석만) | ✅ |
| 핀 ↔ freeze 일치 | — | 100% | **7/7** | ✅ |
| pip resolver dry-run | — | 충돌 0 | (pip 21.2.3 unsupported) freeze 일치로 대체 | ✅ (대체) |
| render 배포 깨짐 | — | 0 | **0** (영향 없음) | ✅ |
| 코드 동작 변경 | — | 0 | 0 라인 | ✅ |

> ≥9 목표 미달은 sklearn/statsmodels/arch 사용 0 실측으로 의도적 제외(dead pin 회피). ADR-0014에 명시.

---

## 후속

- **CP224b (다음)**: ruff 안전 자동수정 + vulture dead code 검출.
- **별도 CP 후보**: 루트 파일을 `requirements-dev.txt`로 명시 개명 (현재는 dev이지만 파일명 혼란).
- **별도 CP 후보**: render 3.11 vs 로컬 3.10 통일 검토.
- **별도 CP 후보**: backend/requirements.txt scikit-learn/statsmodels dead pin 정리.

---

## 자가 점검

- **[Plan v3 정합]** **PASS** — Plan v3의 fidelity/EODHD/α=1·β=2 등에 영향 0. 의존성 핀만 보강.
- **[구조 결함]** **PASS** — dev/배포 매니페스트 의도적 분리가 ADR-0014로 명문화. 루트 파일 역할 명확.
- **[모델 영향]** **PASS** — 핀 값 = freeze 실측이라 현재 환경 무변, 학습/추론 동작 불변. 다운그레이드 강제 0.
