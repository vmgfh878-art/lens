# ADR 0014 — Deploy vs Dev/Training Requirements 분리

- **상태**: 채택 (2026-06-03, CP224a Step 1~3)
- **컨텍스트**: 루트 `requirements.txt`(3줄, optuna/optuna-dashboard/matplotlib만)와 `backend/requirements.txt`(14줄, fastapi/pandas/supabase 등) 두 매니페스트가 공존. render.yaml의 `rootDir: backend` + `buildCommand: pip install -r requirements.txt` 조합으로 배포는 backend 파일만 사용. 루트는 학습/HPO 환경. 그러나 루트의 torch/pandas/numpy/scipy unpin으로 학습 재현성 구멍.

## 결정 요지

| 항목 | 결정 | 근거 |
|---|---|---|
| 매니페스트 분리 | **루트 = dev/training, backend/ = 배포** | render free tier는 CPU 전용·torch 미포함. 로컬 학습은 GPU(torch cu128). 분리하지 않으면 render 빌드가 GPU torch를 끌어와 깨짐. |
| torch cu128 PyPI 핀 | **본문 금지, 주석만** | `+cu128`은 로컬 버전 라벨이라 표준 PyPI 인덱스에 없음. `--index-url https://download.pytorch.org/whl/cu128` 없이는 즉시 실패. |
| Python 버전 | **로컬 3.10.0 / render 3.11.11 (의도적 불일치 인정)** | 본 CP 범위에서 통일하지 않음. 통일은 별도 정책 CP. render는 backend 파일만 사용하므로 루트의 Python 의존성은 render와 무관. |
| 핀 결정 | **freeze 진리값 우선** | 추측 버전 금지. `docs/cp224a_venv_freeze.txt`의 실측만 박음. |
| 핀 대상 | **ai/ 실측 import만** | scikit-learn / statsmodels / arch는 ai/ 사용 0건(grep 확인) → 루트 핀 제외. 지시서 진단의 "sklearn 8건"은 실측 시 0건 → 진단 오류 발견. |

## 대안과 거부 이유

- **단일 requirements.txt 통합** — render가 GPU torch를 끌어와 빌드 실패. 불가.
- **torch를 별도 인덱스로 핀** — render 빌드 명령에 `--index-url` 주입 필요, 빌드 환경 복잡. 거부.
- **scikit-learn/statsmodels/arch 함께 핀** — ai/ 사용 0건이라 dead pin. 거부 (보고서에 기록).
- **Python 3.10→3.11 통일** — torch cu128 빌드의 Python 3.10 의존을 풀어야 함. 본 CP 범위 외.

## 결과 (CP224a baseline)

| 항목 | 시작 | 종료 |
|---|---|---|
| 루트 핀된 핵심 런타임 수 | 3 | **7** (목표 ≥9 미달, sklearn/statsmodels/arch 제외 의도) |
| torch 본문 핀 | 0 | **0** (주석만) |
| 핀 ↔ freeze 일치 | — | **7/7** 정확 일치 |
| pip resolver dry-run | — | (pip 21.2.3엔 `--dry-run` 미지원) freeze 일치로 대체 검증 |
| render 배포 영향 | — | **0** (backend/requirements.txt 사용, 루트 변경 무관) |
| 코드 동작 변경 | — | 0 라인 |

## 후속

- **별도 CP 후보**: `requirements-dev.txt`로 루트 dev 파일 개명/분리 검토 (현 루트는 명시적으로 dev이지만 파일명이 헷갈림).
- **별도 CP 후보**: render Python 3.11 vs 로컬 3.10 통일 검토 (torch cu128 영향 평가 필요).
- **별도 CP 후보**: backend/requirements.txt의 scikit-learn/statsmodels dead pin 정리.
- **별도 CP 후보**: backend/collector/requirements.txt, backend/db/requirements-*.txt 핀 점검.
