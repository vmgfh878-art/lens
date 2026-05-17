<div align="center">

# Lens

**AI 보조지표 기반 미국 주식 분석 및 연구 플랫폼**

예측을 맹신하지 않고, **검증 가능한 밴드, 보수적 신호, 백테스트, 실패 기록**으로 투자 판단을 보조하는 연구 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white)
![Status](https://img.shields.io/badge/status-Phase%201%20research-orange)

</div>

---

## 개요

Lens는 주식 가격을 단순히 맞히려는 앱이 아닙니다.

목표는 다음에 가깝습니다.

- AI 모델이 만든 line, band, risk 신호를 검증한다.
- 데이터 스냅샷, baseline, seed stability, walk-forward를 남긴다.
- Supabase에는 제품 표시에 필요한 얇은 신호만 올린다.
- 연구용 Lens와 배포용 Lens Signal을 분리한다.
- 사용자는 가격 예측값이 아니라, 오늘 어떤 종목을 볼지 판단할 근거를 얻는다.

현재는 **Phase 1 연구 및 제품 후보 검증 단계**입니다. 공개 서비스, 투자 자문, 자동 매매는 범위 밖입니다.

---

## 아키텍처

```text
Collector
  - yfinance / EODHD / FRED 등 데이터 수집
  - parquet snapshot
  - indicator/context 계산

AI
  - PatchTST / CNN-LSTM / TiDE / TCN
  - line, band, warning, distributional 후보 실험
  - baseline, seed stability, walk-forward 평가

Backend
  - FastAPI
  - Supabase thin table 연동
  - product slot 기반 모델 결과 조회

Frontend
  - Next.js
  - 종목별 신호, 밴드, 백테스트 요약 표시
  - 연구용 화면과 배포용 Lens Signal 분리 예정
```

```text
lens/
├─ ai/              모델 학습, 추론, 평가, 백테스트
├─ backend/         FastAPI, collector, DB schema, 테스트
├─ frontend/        Next.js 프론트엔드
├─ docs/            계획서, 설계 문서, 운영 기록
├─ data/            로컬 parquet/cache 산출물
├─ logs/            로컬 실행 로그
└─ scripts/         데이터, 운영, 검증 자동화 스크립트
```

---

## 빠른 시작

### 1. 환경 변수

프로젝트 루트에 `.env`를 만듭니다.

필수:

| 이름 | 설명 |
|---|---|
| `SUPABASE_URL` | Supabase 프로젝트 URL |
| `SUPABASE_KEY` | Supabase service role key |
| `EODHD_API_KEY` | EODHD API key |

선택:

| 이름 | 설명 |
|---|---|
| `FRED_API_KEY` | 거시지표 수집 |
| `FMP_API_KEY` | 보조 시장 데이터 |
| `WANDB_API_KEY` | 실험 추적 |
| `BACKEND_CORS_ORIGINS` | 프론트엔드 허용 도메인 |

### 2. 백엔드

```powershell
cd backend
uvicorn app.main:app --reload
```

기본 주소:

```text
http://127.0.0.1:8000
```

### 3. 프론트엔드

```powershell
cd frontend
npm install
npm run dev
```

기본 주소:

```text
http://localhost:3000
```

### 4. 데이터 상태 확인

```powershell
python -m backend.collector.pipelines.backfill_status
```

### 5. 일일 동기화

```powershell
python -m backend.collector.pipelines.daily_sync
```

---

## 데이터 파이프라인

| 단계 | 저장 위치 | 역할 |
|---|---|---|
| raw price | local parquet / provider cache | 가격 원천 데이터 |
| indicators | local parquet | 보조지표와 context feature |
| model snapshot | local artifact | 학습용 고정 스냅샷 |
| product signal | Supabase thin table | 사용자 화면에 필요한 요약 신호 |
| logs/reports | local docs/logs | 실험 기록과 실패 분석 |

원칙:

- 학습 원천 데이터는 로컬 parquet 중심으로 둡니다.
- Supabase는 전체 가격 저장소가 아니라 제품 표시용 얇은 DB로 씁니다.
- raw 가격, 대량 prediction history, CP 로그는 배포 DB에 넣지 않습니다.
- provider/source/hash/asof_date를 기록해 재현성을 확보합니다.

---

## 모델 연구

현재 다루는 주요 모델군:

| 모델 | 역할 |
|---|---|
| PatchTST | line, distributional, 일부 band 후보 |
| CNN-LSTM | 비교군, risk ensemble 후보 |
| TiDE | band와 warning 후보 |
| TCN | 1D band 후보 |

주요 평가 축:

- baseline 대비 개선
- seed stability
- true walk-forward
- worst seed/fold
- false-safe 감소
- severe downside recall
- quantile calibration
- product slot 승격 가능성

Phase 2에서는 자동 후보 탐색을 붙이되, product slot 교체는 자동화하지 않습니다.

```text
자동 허용:
snapshot 생성, 후보 학습, 평가, 리포트, 승격 추천

수동 승인:
product slot 교체, gate 기준 변경, 사용자 노출 신호 변경
```

---

## GitHub 정리 원칙

GitHub에는 제품 코드, 공통 모델 코드, API, 프론트, 핵심 설계 문서만 올립니다.

올리지 않는 것:

- CP 단위 실험 스크립트
- CP 보고서
- CP 로그
- 임시 metrics, summary, run output
- 로컬 parquet/cache 산출물
- provider 원천 데이터

`.gitignore`는 새로 생기는 CP 파일을 기본적으로 막습니다.

```text
docs/cp*
ai/cp*.py
ai/tests/test_cp*.py
backend/tests/test_cp*.py
scripts/cp*.py
scripts/cp*.ps1
logs/cp*/
data/**/cp*/
```

CP 파일은 정리 전까지 로컬 산출물로 둡니다. 필요할 때 사용자가 정리 명령을 내리면 `docs/cp_archive/` 같은 로컬 보관 구조로 묶습니다.

이미 git에 추적된 과거 CP 파일은 `.gitignore`만으로 빠지지 않습니다. 그런 파일은 별도 정리 커밋 또는 로컬 `skip-worktree` 처리가 필요합니다.

---

## 문서

| 문서 | 역할 |
|---|---|
| `docs/current/phase2_lens_signal_plan.md` | Phase 2 제품, 모델, MLOps, 배포 계획 |
| `docs/model_architecture.md` | 모델 구조와 설계 메모 |
| `docs/training_hyperparameters.md` | 학습 파라미터와 실험 기준 |
| `docs/project_journal.md` | 프로젝트 진행 기록 |

CP 보고서와 실행 로그는 GitHub 공개 문서가 아니라 로컬 연구 기록으로 취급합니다.

---

## 현재 방향

Phase 1:

- US equity 1D band 후보 정리
- 1W 실험 마감
- line 계열 실패 유형 정리
- product slot에 올릴 수 있는 후보 선별

Phase 2:

- Lens 연구용과 Lens Signal 배포용 분리
- ETF 전용 band 모델 후보
- Line V3 distributional 후보
- 자동 후보 탐색과 자동 추천
- Vercel frontend와 Docker backend 배포 리허설

---

## 원칙

- AI는 판단 주체가 아니라 보조지표입니다.
- 모델 성능보다 재현성과 검증 가능성을 먼저 봅니다.
- 좋은 결과보다 실패 원인을 더 잘 남깁니다.
- 가격 예측보다 사용자 의사결정 루프를 중요하게 봅니다.
- 투자 자문, 자동 매매, 수익 보장은 하지 않습니다.

---

## 라이선스

비공개 연구 및 개발용 프로젝트입니다. 외부 배포와 수익화는 데이터 라이선스, 운영 비용, 법적 고지를 별도로 검토한 뒤 결정합니다.

<div align="center">
<sub>Lens 2026</sub>
</div>
