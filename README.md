<div align="center">

# Lens

**미국 주식 보수 예측선과 위험 범위 신호 연구 플랫폼**

예측값을 맹신하지 않고, 보수적 line, calibrated band, 백테스트, 실패 기록을 통해 투자 판단을 보조하는 학술 및 포트폴리오 프로젝트입니다.

<h2>Live Demo</h2>

배포 URL 자리입니다. 실제 데모 연결 후 링크를 추가합니다.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white)
![Status](https://img.shields.io/badge/status-v1%20partial%20ready-orange)

</div>

---

## 한 페이지 요약

### 목적

Lens는 미국 주식의 미래 가격을 단정적으로 맞히려는 도구가 아닙니다. 사용자가 오늘 어떤 종목을 더 볼지, 예측선과 범위 신호를 참고해 판단할 수 있도록 돕는 연구용 의사결정 보조 도구입니다.

핵심 출력은 세 가지입니다.

| 출력 | 의미 | 현재 판단 |
|---|---|---|
| 보수 예측선 | h5 수익률 기준 보수적 기준선 | CP203까지 재검증했으나 신규 v2는 불안정. v1 계열 CP175 beta5 frozen line을 우선 사용 |
| 위험 범위 band | 예상 수익률 범위와 변동성 해석 | CP202 기준 GARCH baseline 동급 영역이 있어, 예측 우위보다 범위 신호로 한정 |
| warning | 별도 위험 경고 신호 | CP196~198 dvol 계열은 제품 신호로 채택하지 않음. v2에서 band와 uncertainty로 흡수 검토 |

### 핵심 결정과 결과

- Line: CP203 Conservative Stop-Loss Line v2까지 진행했지만 seed stability와 source hash drift 이슈로 신규 v2 후보를 확정하지 않았습니다.
- Line v1: CP164 계열은 기준선 역할, CP175 beta5 frozen line은 현재 v1 package의 1D line 우선 후보입니다.
- Band: CP202와 CP202.1에서 1D/1W band를 baseline과 비교했습니다. 정상 구간에서는 의미가 있지만 stress regime에서는 GARCH(1,1) baseline과 동급에 가까워, 과장된 예측 모델이 아니라 calibrated risk interval로 설명합니다.
- Warning: CP196~198의 dvol/residual volatility warning은 standalone 제품 신호로 쓰기 어렵다고 판단했습니다.
- 데이터: yfinance local parquet를 중심으로 사용하며, 원천 데이터와 대량 실험 산출물은 Supabase에 올리지 않습니다.
- 외부 데이터: SEC 8-K, Form 4, earnings, macro event proximity는 v2에서 risk 설명력 보강 후보로 검토합니다.

### 현재 v1 상태

CP204 save-run 기준 현재 상태는 완성 배포가 아니라 `partial ready`입니다.

| 슬롯 | 후보 | 상태 | 비고 |
|---|---|---|---|
| 1D Line | CP175 beta5 frozen line | 준비됨 | local package 기준. 새 학습, 새 calibration, DB write 없음 |
| 1D Band | CP153 Tide q15 lower_focused | 준비됨 | CP202 target lock의 1D primary band |
| 1W Band | CP178/CP202 1W band | 차단 | row-level prediction payload가 없어 추정 생성 금지 원칙상 save-run 중단 |
| 1W Line | 없음 | 보류 | 이번 v1 save-run 대상 아님 |

---

## 이 저장소를 읽는 방법

이 저장소는 직접 손으로 모든 코드를 작성했다는 증명이 아니라, AI 코딩 도구를 활용해 금융 시계열 ML 연구를 설계하고 검증한 기록입니다.

제가 소유한 부분은 문제 정의, 실험 방향, 평가 기준, 실패 분석, 모델 승격 판단, 제품 분리 방향입니다. 구현 코드는 AI 코딩 도구의 도움을 받아 작성되었고, 저는 그 구현이 연구 목적과 판단 기준에 맞는지 검토하며 프로젝트를 진행했습니다.

따라서 이 프로젝트를 볼 때는 단순 코드량보다 다음 흐름을 중심으로 읽는 것이 좋습니다.

- 어떤 투자 판단 보조 문제를 풀려고 했는가
- 예측값을 제품 신호로 바꾸기 위해 어떤 기준을 세웠는가
- 어떤 모델 결과를 실패로 판단했는가
- 어떤 결과를 제품 후보로 남겼는가
- 데이터, 평가, 모델, 제품 노출을 어떻게 분리했는가
- AI 도구를 사용하면서도 어떤 판단은 사람이 소유했는가

### 설명 가능한 판단 범위

- line, band, warning, product slot의 역할 분리
- baseline, seed stability, walk-forward, calendar-aligned split 기준
- false-safe, severe downside, break positivity, coverage, ECE, calibration 해석
- Supabase를 전체 가격 저장소가 아니라 얇은 제품 DB로 쓰는 설계
- 연구용 Lens와 배포용 Lens Signal을 분리하려는 방향
- CP203 v2를 무리하게 채택하지 않고 CP175 frozen line을 우선 사용한 판단
- CP202 band를 미래 변동성 예측기가 아니라 범위 신호로 제한한 판단

### 공개 문서 묶음

핵심 CP 원자료 중 공개 첫 화면에서 읽을 만한 문서는 [docs/public](docs/public/README.md)에 따로 묶었습니다.

이 묶음에는 line v2 실패 종합, CP175 beta5 후보, stop-loss line v2 계획, 1D band save-run, band baseline 비교, v1 save-run 상태, band v2 계획만 포함합니다. 단순 기록용 CP 로그와 원천 데이터 산출물은 포함하지 않습니다.

---

## 데모 사용법

데모가 연결되면 다음 순서로 확인합니다.

1. 메인 페이지에서 종목을 검색합니다.
2. Stock View에서 보수 예측선과 band layer를 켜고 끕니다.
3. Backtest View에서 단일 티커 전략 결과를 확인합니다.
4. AI Model 또는 모델 설명 화면에서 현재 표시 중인 model slot과 한계를 확인합니다.

현재 데모는 투자 판단 도구가 아니라 연구 결과를 시연하는 화면입니다. 실제 투자 의사결정에 사용해서는 안 됩니다.

---

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS, lightweight-charts |
| Backend | FastAPI, Supabase client, pandas, pyarrow |
| Models | PyTorch, PatchTST, TiDE, CNN-LSTM |
| Evaluation | coverage, break positivity, ECE, calibration, IC, walk-forward, seed stability |
| Baseline | Bollinger Bands, Historical Quantile, GARCH(1,1) closed-form fallback, linear regression |
| Data | yfinance local parquet 중심, EODHD/FRED/FMP는 보조 또는 과거 호환 경로 |
| Deploy | Vercel frontend 예정, Render backend cron 검토 |

---

## 아키텍처

```text
Collector
  - yfinance 중심 가격 데이터 수집
  - local parquet snapshot
  - indicator/context feature 계산

AI
  - PatchTST / CNN-LSTM / TiDE
  - line, band, warning, distributional 후보 실험
  - baseline, seed stability, walk-forward 평가

Backend
  - FastAPI
  - Supabase thin table 연동
  - product slot 기반 모델 결과 조회

Frontend
  - Next.js
  - 종목별 가격, 보수 예측선, band, 백테스트 요약 표시
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

## 모델 연구 요약

### Line

Line은 h5 보수 예측선입니다. 사용자가 가격 흐름을 볼 때 기준선처럼 참고하는 역할을 목표로 했습니다.

CP203에서는 stop-loss line v2를 새로 검증했지만, seed에 따라 coverage와 break composition이 크게 흔들렸고 일부 track은 source hash drift로 비교 계약이 깨졌습니다. 따라서 신규 v2를 무리하게 ship하지 않고, CP175 beta5 frozen line을 v1 package의 우선 후보로 둡니다.

### Band

Band는 예측 범위와 변동성 해석을 위한 신호입니다. CP202/CP202.1 결과상 deep band가 모든 상황에서 통계 baseline을 명확히 압도한다고 말하면 안 됩니다. 특히 stress regime에서는 GARCH baseline과 동급에 가까운 영역이 있습니다.

따라서 band는 “미래를 미리 맞히는 강한 예측기”가 아니라 “calibrated risk interval과 범위 신호”로 설명합니다.

### Warning

별도 warning 신호는 아직 제품화하지 않습니다. CP196~198의 dvol 계열은 recall을 높이면 warning share가 커지고, share를 줄이면 위험 분리력이 약해졌습니다. 현재는 warning을 독립 신호로 노출하기보다 v2 band의 uncertainty, concept, selective output에 흡수하는 방향을 검토합니다.

---

## Phase 2 방향

CP204 Band v2의 방향은 다음과 같습니다.

| 축 | 목표 |
|---|---|
| Conformal coverage | 분포 가정이 약한 coverage 보장 |
| Deep ensemble | aleatoric uncertainty와 epistemic uncertainty 분리 |
| Selective output | 모델이 자신 없을 때 표시하지 않거나 낮은 신뢰도로 표시 |
| Concept Bottleneck | band가 넓어진 이유를 사람이 이해 가능한 concept으로 설명 |
| 외부 데이터 | Form 4, 8-K, earnings, macro event proximity를 위험 설명 feature로 검토 |

v2의 목표는 단순 성능 향상이 아니라, “왜 오늘 이 종목의 band가 넓어졌는지”와 “언제 모델을 믿지 말아야 하는지”를 사용자에게 설명하는 것입니다.

---

## 데이터와 운영 원칙

| 단계 | 저장 위치 | 역할 |
|---|---|---|
| raw price | local parquet / provider cache | 가격 원천 데이터 |
| indicators | local parquet | 보조지표와 context feature |
| model snapshot | local artifact | 학습용 고정 스냅샷 |
| product signal | Supabase thin table | 사용자 화면에 필요한 최신 요약 신호 |
| logs/reports | local docs/logs | 실험 기록과 실패 분석 |

원칙:

- 학습 원천 데이터는 로컬 parquet 중심으로 둡니다.
- Supabase는 전체 가격 저장소가 아니라 제품 표시용 얇은 DB로 씁니다.
- raw 가격, 대량 prediction history, CP 로그는 배포 DB에 넣지 않습니다.
- provider/source/hash/asof_date를 기록해 재현성을 확보합니다.
- 모델 결과는 product slot에 수동 승인 후 연결합니다.

---

## 로컬 실행

### 1. 환경 변수

프로젝트 루트에 `.env`를 만듭니다.

필수:

| 이름 | 설명 |
|---|---|
| `SUPABASE_URL` | Supabase 프로젝트 URL |
| `SUPABASE_KEY` | Supabase service role key |

선택:

| 이름 | 설명 |
|---|---|
| `EODHD_API_KEY` | EODHD fallback 또는 과거 호환 경로 |
| `FRED_API_KEY` | 거시지표 수집 |
| `FMP_API_KEY` | 보조 시장 데이터 |
| `WANDB_API_KEY` | 실험 추적 |
| `BACKEND_CORS_ORIGINS` | 프론트엔드 허용 도메인 |

### 2. 백엔드

```powershell
cd backend
pip install -r requirements.txt
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

## GitHub 정리 원칙

GitHub에는 제품 코드, 공통 모델 코드, API, 프론트, 핵심 설명 문서만 올립니다.

올리지 않는 것:

- CP 단위 원본 실험 스크립트
- CP 원본 보고서
- CP 로그
- 임시 metrics, summary, run output
- 로컬 parquet/cache 산출물
- provider 원천 데이터

CP 파일은 정리 전까지 로컬 연구 산출물로 둡니다. README에는 공개 첫 화면에서 이해해야 하는 결론만 요약합니다.

---

## 면책

Lens는 학술 및 포트폴리오 목적의 연구 프로젝트입니다. 모든 예측, 신호, 백테스트 결과는 정보 제공 목적이며 투자 자문, 매매 권유, 수익 보장을 의미하지 않습니다.

자세한 내용은 [DISCLAIMER.md](DISCLAIMER.md)를 확인해 주세요.

---

## 라이선스

코드는 [MIT License](LICENSE)를 따릅니다.

단, yfinance/Yahoo Finance 등 제3자 데이터의 이용, 데모 운영, 원천 데이터 재배포 가능 여부는 각 데이터 제공자의 약관을 따릅니다. 이 프로젝트는 데이터와 데모를 비상업적 학술 및 포트폴리오 목적에 한정해 사용합니다.

<div align="center">
<sub>Lens 2026</sub>
</div>
