<div align="center">

# Lens

미국 주식 보수적 기준선과 위험 범위 신호 연구 플랫폼

예측을 단언하지 않고, 보수적 기준선 · calibrated band · 백테스트 · 실패 기록을 통해
사용자 판단을 보조하는 학술 및 포트폴리오 프로젝트.

Live Demo: 배포 직후 URL 추가 예정

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/v1-ready-success)

</div>

---

## 1. 한 페이지 요약

### 문제 정의

일반적인 주가 예측 모델은 점추정만 제공하며, 사용자는 그 값을 얼마나 신뢰해야 할지 알기 어렵습니다.
Lens 는 약간 보수적인 예측선을 의도적으로 제시하여, 가격이 그 선 아래로 떨어지는 시점을
stop-loss 신호로 해석할 수 있게 합니다.

### 세 가지 출력

| 출력 | 의미 | 현재 후보 |
|---|---|---|
| Line | h5 보수적 기준선 (5 거래일 기준) | F4 β=4 ensemble |
| Band | 예측 범위 (q15/q85 calibrated) | CP153 TiDE lower_focused (1D) · CP178 walk-forward lower (1W) |
| Warning | 위험 경고 | v1 미노출. v2 에서 band uncertainty 로 흡수 검토 |

### 진행 / 결과 요약

- Line: CP212에서 F4 β=4 ensemble을 v1 1D line serving 후보로 승격. raw output은 `safe_line_score`이며 화면에서 가격으로 환산.
- Band: CP202 / CP202.1 결과 정상 regime 에서 GARCH 우위, stress regime 에서 동등. 예측기가 아닌 calibrated risk interval 로 한정.
- Warning: CP196 ~ CP198 의 dvol 계열 warning detector 폐기. Phase 2 의 v2 band concept 으로 흡수.
- 데이터: yfinance local parquet 중심. Supabase / 외부 API 의존 최소화.

---

## 2. 이 저장소를 읽는 방법

이 저장소는 손으로 모든 코드를 쓴 증명이 아니라, AI 코딩 도구를 활용해 금융 시계열 ML 연구를 설계 / 검증한 기록입니다.

제가 소유한 부분은 문제 정의 · 실험 방향 · 평가 기준 · 실패 분석 · 모델 승격 판단 · 제품 분리 방향입니다.
구현 코드는 AI 도구의 도움을 받았고, 저는 그 결과가 연구 의도에 맞는지 검토하며 진행했습니다.

따라서 다음 관점에서 읽어주시면 좋습니다.

- 어떤 투자 판단 보조 문제를 풀려고 했는가
- 예측값을 제품 신호로 바꾸기 위해 어떤 기준을 세웠는가
- 어떤 결과를 실패로 판단했는가
- 어떤 결과를 제품 후보로 남겼는가
- 데이터 · 평가 · 모델 · 노출을 어떻게 분리했는가

### 핵심 문서

| 문서 | 내용 |
|---|---|
| [docs/current/product_development_log_2026_05.md](docs/current/product_development_log_2026_05.md) | 최근 9 일 (CP194 ~ CP204) 의 결정 / 실패 / 배운점 정리 |
| [docs/current/research_governance_log.md](docs/current/research_governance_log.md) | 누적 의사결정 / 실패 로그 (2026-04 ~ 현재) |
| [docs/current/phase1_project_status.md](docs/current/phase1_project_status.md) | Phase 1 스냅샷 (2026-05-11 기준) |
| [docs/cp204_band_v2_plan.md](docs/cp204_band_v2_plan.md) | Band v2 설계: Conformal · CBM · Selective output · 13 metrics |
| [docs/current/phase2_lens_signal_plan.md](docs/current/phase2_lens_signal_plan.md) | Phase 2 (signal track 분리) 계획 |
| [docs/cp_archive/](docs/cp_archive/) | 전체 CP 실험 기록 (50+ 개) |

---

## 3. v1 모델 슬롯 (현재 배포 상태)

| 슬롯 | 후보 | 상태 | Row count |
|---|---|---|---|
| 1D Line | F4 β=4 ensemble | READY | serving parquet 기준 |
| 1D Band | CP153 TiDE q15 lower_focused (historical) | READY | 929,385 (full) / 543,615 (past 1y) |
| 1W Band | CP178 walk-forward lower calibration (9 ensemble) | READY | 416,724 (full) / 138,908 (ensemble avg, past 2y) |
| 1W Line | — | Deferred | v1.1 이상에서 검토 |

상세 평가 / 컷 / drift 처리는 [docs/current/product_development_log_2026_05.md](docs/current/product_development_log_2026_05.md) Section 4 ~ 6 참조.

---

## 4. 기술 스택

| 영역 | 사용 기술 |
|---|---|
| Frontend | Next.js 14 · React 18 · TypeScript · Tailwind · lightweight-charts |
| Backend | FastAPI · pandas · pyarrow |
| Models | PyTorch · PatchTST · TiDE · CNN-LSTM |
| Evaluation | coverage · break positivity · ECE · pinball · calibration · IC · walk-forward · seed stability |
| Baselines | Bollinger · Historical Quantile · GARCH(1,1) · Linear Regression |
| Data | yfinance local parquet (501 ticker × 11 년) |
| Deploy | Vercel (frontend) · Render (backend) |
| DB | 없음 (v1 은 로컬 parquet 서빙). Supabase 는 optional. |

---

## 5. 아키텍처

```text
[로컬]
  └─ parquet 생성 / 평가 / 검증
       ↓ git push
[GitHub Repo]
  └─ backend/data/v1/*.parquet (~18 MB)
       ↓
[Render — Backend (FastAPI)]
  └─ startup 시 parquet 메모리 로드
       └─ /api/v1/stocks/{ticker}/predictions/product-history
       └─ /api/v1/predictions/{slot}/{ticker}
            ↓ axios
[Vercel — Frontend (Next.js)]
  └─ 종목 검색 / 차트 / 백테스트
```

```text
lens/
├─ ai/                        모델 학습 / 추론 / 평가 / 백테스트
├─ backend/
│  ├─ app/                    FastAPI (routers · services · repositories)
│  ├─ data/v1/                v1 frontend serving parquets
│  ├─ collector/              yfinance 데이터 수집 (학교 데모는 로컬에서만 실행)
│  └─ scripts/                build / import utility scripts
├─ frontend/                  Next.js 프론트엔드
├─ docs/                      계획서 / 설계 / 실험 기록
└─ data/                      로컬 parquet / 학습 캐시 (git 제외)
```

---

## 6. API Endpoints

### 제품 표시용 (현재 frontend 가 사용)

```
GET  /api/v1/stocks/{ticker}/prices?timeframe=1D&limit=300
GET  /api/v1/stocks/{ticker}/indicators?timeframe=1D
GET  /api/v1/stocks/{ticker}/predictions/product-history?timeframe=1D&roles=all
GET  /api/v1/stocks/{ticker}/predictions/latest?model=patchtst&timeframe=1D
GET  /api/v1/stocks?search=AA&limit=50
```

### v1 신규 — Local parquet 직접 서빙

```
GET  /api/v1/predictions/health
GET  /api/v1/predictions/tickers?search=AA
GET  /api/v1/predictions/line/{ticker}?days=365
GET  /api/v1/predictions/band/1d/{ticker}?days=365&horizon=5
GET  /api/v1/predictions/band/1w/{ticker}?days=730&horizon=4
```

### 운영

```
GET  /api/v1/health/live      # liveness
GET  /api/v1/health/ready     # readiness (Supabase optional)
```

---

## 7. 로컬 실행

### 1) 환경 변수

`backend/.env` (없으면 root `.env` 도 가능):

```
# Optional — v1 local-parquet 모드는 Supabase 없어도 동작
SUPABASE_URL=
SUPABASE_KEY=

# Frontend CORS
BACKEND_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Optional — yfinance 보조
FRED_API_KEY=
FMP_API_KEY=
```

`frontend/.env.local`:

```
# Frontend 가 호출할 backend URL.
# 배포 환경은 반드시 Render URL을 명시한다.
NEXT_PUBLIC_BACKEND_URL=https://lens-backend-7stj.onrender.com

# 로컬 개발 시에는 아래처럼 바꿔서 사용한다.
# NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
```

### 2) 백엔드

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

→ http://127.0.0.1:8000/docs (FastAPI 자동 문서)

### 3) 프론트엔드

```powershell
cd frontend
npm install
npm run dev
```

→ http://localhost:3000

### 4) (선택) v1 parquet 재빌드

```powershell
# CP204 import package → frontend serving 형태로 변환
python backend/scripts/build_v1_predictions_local.py
python backend/scripts/rebuild_product_history_parquet.py
```

---

## 8. 데이터 / 운영 원칙

| 단계 | 저장 위치 | 역할 |
|---|---|---|
| raw price | local parquet | 가격 원천 |
| indicators | local parquet | 보조지표 / context |
| model artifacts | local | 학습 checkpoint / metrics |
| v1 product signal | `backend/data/v1/*.parquet` (git 포함) | frontend 서빙용 (~18 MB) |
| logs / reports | local `docs/` `logs/` | 실험 기록 |
| Supabase | optional | v2 1번. 사용량 제한 해제 후 thin product DB 재연결 |

원칙

- 학습 원천 데이터는 로컬 parquet 중심
- v1 product signal 만 git 에 포함 (frontend deploy 와 함께 배포)
- 대량 prediction history · CP 로그 · 원천 가격은 git / DB 제외
- provider / source / hash / asof_date 기록으로 재현성 확보
- 모델 결과는 product slot 에 수동 승인 후 연결 (`build_v1_predictions_local.py` → `git push`)

---

## 9. Phase 2 방향 (Band v2)

| 축 | 목표 |
|---|---|
| Supabase reconnect | 사용량 제한 해제 후 thin product DB 재연결, egress guard 확인 |
| Conformal coverage | 분포 가정 약한 coverage 수학적 보장 (Vovk 2005, Romano 2019) |
| Deep ensemble | aleatoric / epistemic uncertainty 분리 (Lakshminarayanan 2017) |
| Selective output | 모델이 자신 없을 때 표시 안 함 또는 낮은 신뢰도 표시 (Geifman 2017) |
| Concept Bottleneck | band 가 넓어진 이유를 사람이 이해 가능한 12 concept 으로 설명 (Koh 2020) |
| Leading volatility | forward vol mse loss + event-aware widening regularizer (Lens-specific) |
| 외부 데이터 | Form 4 (insider) · 8-K (item code) · cross-asset stress |
| Band MLOps | 기존 1D/1W band 자동 재평가 · 재학습 · 승격 추천 루프 |

목표는 단순 성능 향상이 아니라
왜 오늘 이 종목의 band 가 넓어졌는지와 언제 모델을 믿지 말아야 하는지를 사용자에게 설명하는 것입니다.

자세한 설계는 [docs/cp204_band_v2_plan.md](docs/cp204_band_v2_plan.md) 참조.
기존 band 모델의 지속 개선 계획은 [docs/public/band_mlops_plan.md](docs/public/band_mlops_plan.md) 참조.

---

## 10. Roadmap

| Version | 상태 | 내용 |
|---|---|---|
| v1.0 | 진행 중 | 3 slot 모델 + frontend deploy (1D Line · 1D Band · 1W Band) |
| v1.1 | 계획 | 1W Line 검토 (필요 시 신규 학습) |
| v2.0 | 설계 완료 | 1. Supabase thin product DB 재연결 2. Conformal Band + CBM + Band MLOps ([docs/cp204_band_v2_plan.md](docs/cp204_band_v2_plan.md), [docs/public/band_mlops_plan.md](docs/public/band_mlops_plan.md)) |
| v3.0 | 후보 | Form 4 / 8-K NLP 외부 데이터 통합 |

---

## 11. 면책

Lens 는 학술 및 포트폴리오 목적의 연구 프로젝트입니다. 모든 예측 / 신호 / 백테스트 결과는
정보 제공 목적이며 투자 자문, 매매 권유, 수익 보장을 의미하지 않습니다.

자세한 내용은 [DISCLAIMER.md](DISCLAIMER.md) 참조.

---

## 12. 라이선스

코드는 [MIT License](LICENSE) 를 따릅니다.

단, yfinance · Yahoo Finance 등 제 3 자 데이터의 이용, 데모 운영, 원천 데이터 재배포 가능 여부는
각 데이터 제공자의 약관을 따릅니다. 본 프로젝트는 데이터 / 데모를 비상업적 학술 · 포트폴리오
목적에 한정해 사용합니다.

---

## 저자

김지형 (Kim Jihyeong)
세종대학교 데이터사이언스학과
GitHub [@vmgfh878-art](https://github.com/vmgfh878-art)

<div align="center">
<sub>Lens · 2026</sub>
</div>
