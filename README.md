<div align="center">

# Lens

미국 주식 보수적 기준선과 위험 범위 신호 연구 플랫폼

예측을 단언하지 않고, 보수적 기준선 · calibrated band · 백테스트 · 실패 기록을 통해
사용자 판단을 보조하는 학술 및 포트폴리오 프로젝트.

**Live Demo**: [lens-ten-delta.vercel.app](https://lens-ten-delta.vercel.app) (Frontend) · [lens-backend-7stj.onrender.com](https://lens-backend-7stj.onrender.com) (Backend, free tier — cold start ~10s)

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

| 출력 | 의미 | 운영 모델 |
|---|---|---|
| Line | h5 보수적 기준선 (5 거래일 후 도착가) | CP210 PatchTST F4 β=4 5-seed ensemble |
| Band | 예측 범위 (분위수 + conformal 보정) | CP153 TiDE q15/q85 lower_focused (1D) · CP178 TiDE q10/q90 WFLOCK (1W) |
| Warning | 위험 경고 | v1 미노출. v2 에서 selective output trigger 로 통합 (CP216.2 GW regime 정량 근거 확보) |

### 진행 / 결과 요약 (2026-05 기준)

- **Line (CP210)**: F4 β=4 5-seed ensemble 운영 진입. IC 0.0325 / severe recall 0.7727 / false-safe 0.2048. WF IC range 0.0457 으로 ship 기준 0.040 초과 → NO_SHIP 사유 정직 노출 후 채택.
- **Band 1D (CP153)**: TiDE q15/q85 + lower_focused conformal 보정. coverage_abs_error 0.0099 (학계 standard 통과) · band_width_ic 0.376.
- **Band 1W (CP178)**: WFLOCK walk-forward + lower calibration. coverage_abs_error 0.039 · band_width_ic 0.34. 1D 대비 약하지만 주봉 난이도 감안 채택.
- **통계 검정 (CP216.2)**: DM + Bootstrap CI (cluster + block) + GW conditional 로 4 베이스라인 (bollinger / historical_quantile / GARCH 정적 · walk-forward) 과 비교.
  - 라인: 통계 베이스라인과 IC 통계 동등 (Bonferroni p=1.00) — ensemble 추가 가치 통계 미확인
  - 1D 밴드: 단순 분위수·walk-forward GARCH 대비 pinball 통계 열위 (d 0.45~0.93). 볼린저 대비는 우위
  - 1W 밴드: 단순 분위수 열위 (d=1.17), walk-forward GARCH 우위 (d=-0.98). **GW 3 regime 모두 wald p<0.001 — regime-conditional 차이 systematic**
- **정체성 전환**: 학계 통설 (Welch & Goyal 2008 / Makridakis M-competitions) 과 일관된 결과를 정직 재현. v1 은 정확도 경쟁이 아닌 **selective output + 위험 시각화** 로 가치 정의.
- 데이터: yfinance local parquet (S&P 500, 501 ticker × 11 년). Supabase / 외부 API 의존 최소화 (v2 로 이관 검토).

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

| 슬롯 | 운영 모델 | run_id | 상태 | 핵심 metric |
|---|---|---|---|---|
| 1D Line | CP210 F4 β=4 5-seed ensemble (PatchTST) | `cp210_F4_b4_ensemble_mean` | READY (WARN) | IC 0.0325 · severe recall 0.7727 · WF range 0.0457 (NO_SHIP 사유) |
| 1D Band | CP153 TiDE q15/q85 lower_focused | `tide-1D-ea54dcae654d` | READY | coverage_abs_err 0.0099 · band_width_ic 0.376 |
| 1W Band | CP178 TiDE q10/q90 WFLOCK | `tide_s60_q10_q90_param` | READY | coverage_abs_err 0.039 · band_width_ic 0.34 |
| 1W Line | — | — | Deferred | 주봉 horizon 정의 자체 재검토 (v2) |

목표 근거 / 통계 검정 결과 / 재현 매니페스트는 화면 (AI 모델 상세) 또는 [`docs/v1_operating_models_reproducibility.md`](docs/v1_operating_models_reproducibility.md) 참조.

---

## 3.5 통계 검정 결과 (CP216.2)

**왜 했나**: "운영 모델이 단순 통계 기준선보다 진짜로 통계 우위인가" 정직 평가. Selective output 의 정량 근거 확보.

**방법**: DM (Diebold-Mariano HAC) + Bootstrap CI (cluster 종목별 / block 시간 √T·22d 둘 다) + GW conditional (Giacomini-White). 4 베이스라인 (`bollinger`, `historical_quantile`, `garch_p_q_1_1` 정적, `garch_walkforward`). Bonferroni n_tests=11.

**핵심 결론**

| 모델 | DM (Bonf p) | Cohen's d | 95% CI | 결론 |
|---|---|---|---|---|
| Line vs historical_mean | 1.00 | +0.12 (negligible) | block [−0.0090, +0.0264] | 통계 동등 |
| Line vs CP175 frozen | 1.00 | −0.07 (negligible) | block [−0.0095, +0.0048] | 통계 동등 |
| 1D Band vs bollinger | <0.001 | −1.97 (large) | cluster [−0.00521, −0.00486] | 통계 우위 |
| 1D Band vs historical_quantile | <0.001 | +0.93 (large) | cluster [+0.000528, +0.000791] | **통계 열위** |
| 1D Band vs garch_walkforward | <0.001 | +0.45 (medium) | cluster [+0.000188, +0.000437] | **통계 열위** |
| 1W Band vs garch_walkforward | <0.001 | −0.98 (large) | cluster [−0.01082, −0.00224] | **통계 우위** |
| 1W Band GW (vix>30) | β=+0.00648, p<0.001 | — | — | **regime-conditional 차이 systematic** |
| 1W Band GW (dd≤-10%) | β=+0.01037, p<0.001 | — | — | systematic |
| 1W Band GW (combined) | β=+0.00797, p<0.001 | — | — | systematic |

**해석**

- ML 라인이 단순 통계 베이스라인을 통계적으로 못 이김 — **학계 통설** (Welch & Goyal 2008 / Makridakis M4·M5) **정직 재현**.
- 1D 밴드는 단순 분위수 / walk-forward GARCH 대비 pinball 열위. 다만 coverage_abs_err 0.0099 / band_width_ic 0.376 의 **trade-off** 가치 유지.
- 1W 밴드는 평상시 walk-forward GARCH 대비 우위 (d̄ = −0.0069). 위기 구간 (VIX>30, drawdown ≤−10%, 결합) 에서 우위 사라지거나 역전 — Bonferroni p<0.001 로 systematic. **Selective output trigger 의 정량 근거** 확보.

산출물: [`docs/cp216_2_significance/`](docs/cp216_2_significance/) (summary.csv + metrics.json) · 화면 AI 모델 상세 패널 안 "통계 검정" 섹션.

---

## 3.6 초기 계획서 (2026-04-19) vs 현재 (2026-05-30)

| 초기 계획 | 결과 | 차이/사유 |
|---|---|---|
| 밴드 포함률 > 90% | q15/q85 → 70% (1D), q10/q90 → 80% (1W) | 5거래일 90% CI 는 너무 넓어 실용성 떨어짐 → q-pair 좁히고 conformal 보정 |
| 방향 정확도 > 55% | IC 0.0325 (라인) — 대체 metric | rank correlation 으로 대체. IC > 0 으로 약하지만 통계 베이스라인과 동등 |
| MAPE < 5% | 미사용 | 라인이 가격 절대값이 아니라 수익률 score |
| 모델 비교 | PatchTST (라인) / TiDE (밴드) 채택, CNN-LSTM/TCN 비교 reserve | CP153 model zoo + CP208Z baseline lock |
| API 응답 < 3초 | 추정 PASS (lru_cache + columns filter) | 정량 측정 미수록 — v2 관측성 트랙에서 박을 것 |
| Database = Supabase (PostgreSQL) | v1 = local parquet 서빙. Supabase 는 v2 첫 트랙으로 이관 검토 | 운영 비용 / 응답 속도 / 학교 데모 운영 단순성 절충 |
| Docker 배포 표준화 | v2 트랙 | v1 = git push → Render/Vercel 자동 배포 |

**신규 발견 (계획 외)**

- 정확도 ML 가 단순 통계 베이스라인 못 이김이 학계 통설임을 통계 검정으로 정직 입증 → **"정확도 경쟁 X, 위험 인식 도구"** 정체성 전환
- DM + Bootstrap CI + GW 통계 검정 적용으로 selective output trigger 정량 정당화
- regime-conditional (vix_high / drawdown_low / combined) 차이 systematic 확인 → v2 selective output 트리거 정의 가능

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

### 5) 운영 모델 학습 재현 — 한 줄 wrapper

운영 진입한 3 모델 (CP210 라인 / CP153 1D 밴드 / CP178 1W 밴드) 학습을 외부 재현자가 한 줄로 실행할 수 있는 wrapper 를 제공합니다. GPU 학습이라 sm_120 (RTX 5060 Ti) 권장 — 다른 GPU 도 동작합니다 (statistical 재현, 아래 GPU 가이드 참조).

**준비**

1. 드롭박스에서 외부 패키지 다운로드 (제출자에게 URL 문의):
   - `external_package_line.zip` — CP209 5-seed checkpoint + 학습 시점 parquet
   - `external_package_band_1d.zip` — CP153 3-seed checkpoint + parquet
   - `external_package_band_1w.zip` — CP178 3-seed checkpoint + 1W parquet
2. 압축 풀어 `lens/external_package/` 폴더에 배치.

**실행**

```powershell
# 환경 / 외부 패키지 점검만 (학습 X, 5초)
python scripts\reproduce_line_cp210.py --external .\external_package --verify-only

# 학습 + 재현 결과 비교 (~12h on RTX 5060 Ti)
python scripts\reproduce_line_cp210.py --external .\external_package
python scripts\reproduce_band_1d_cp153.py --external .\external_package
python scripts\reproduce_band_1w_cp178.py --external .\external_package
```

각 wrapper 는 환경 검증 → 외부 패키지 점검 → checkpoint/parquet 배치 → 학습 entry 실행 (cascade 의존 자동 import) → 학습 후 운영 parquet 과 statistical 비교 (row count + 핵심 metric 평균/표준편차 허용 오차 안이면 PASS) 까지 자동 수행합니다.

**다른 GPU 사용 가이드**

bit-exact 일치는 GPU / CUDA / kernel 구현 차이로 불가능하지만 statistical 재현은 모든 CUDA GPU 에서 가능합니다.

| GPU | torch 권장 | 비고 |
|---|---|---|
| RTX 5060 Ti (sm_120) | `torch==2.11.0+cu128` (학습 시점) | nightly. 학습 시점 매니페스트 정확 일치 |
| RTX 4090 (sm_89) / RTX 4070 / A100 (sm_80) / V100 (sm_70) | `torch==2.2.x+cu121` 또는 `torch==2.0.1+cu118` | 표준 안정 버전. 학습 시간 / 메모리 다름. metric 거의 동일 |
| 메모리 부족 시 | batch size 줄임 | wrapper 의 `--batch-size-train`, `--batch-size-eval` 인자 활용 |

학계 재현 관행상 IC ±0.005 / coverage ±0.005 수준 차이는 정상이며, wrapper 의 결과 비교 허용 오차 (평균 ±5%, 표준편차 ±10%) 안이면 통계적 재현 성공입니다.

---

## 8. 데이터 / 운영 원칙

### 깃 / 외부 패키지 분리

| 단계 | 위치 | 역할 |
|---|---|---|
| 소스코드 (`backend / frontend / ai / scripts`) | **깃** | 운영 + 추론 + 재현 wrapper |
| v1 운영 매니페스트 (`docs/v1_operating_models_reproducibility.md`) | **깃** | 환경 / fold / seeds / GPU 단일 진리 |
| 운영 모델 관련 핵심 보고서 (CP210 / CP153 / CP178 / CP216.2) | **깃** | 결과 narrative + 통계 검정 |
| v1 product signal parquet (`backend/data/v1/*.parquet`) | **깃** (~18 MB) | frontend 서빙용 (Render 배포 동봉) |
| 학습 entry + cascade 의존 (`ai/cp{209,210,153,178}*.py` + 9 cascade) | **깃** (텍스트 작음) | 학습 재현 wrapper 가 호출 |
| 모델 checkpoint (`data/artifacts/cp{210,153,178}/*.pt`) | **드롭박스** (~수 GB) | 학습 재현 시 필요 |
| 학습 시점 parquet (`data/parquet/*.parquet`) | **드롭박스** (~1 GB) | 학습 재현 시 필요 |
| 전체 CP 60+ 보고서 (운영 외) | **드롭박스** | 참고 자료 |
| W&B logs / 학습 로그 | git/DB 제외 | 로컬만 |

### 원칙

- **운영 모델 3개만** 깃에 핵심 보고서 + 학습 entry + cascade 의존 포함 (`.gitignore` 예외)
- 대량 binary (checkpoint, parquet, W&B) 는 깃 제외 → 드롭박스 외부 패키지
- v1 product signal parquet 만 깃 포함 (Render 배포 동봉)
- provider / source / hash / asof_date 기록으로 재현성 확보
- 모델 결과는 product slot 에 수동 승인 후 연결 (`build_v1_predictions_local.py` → `git push`)
- **v1 운영 매니페스트** (`docs/v1_operating_models_reproducibility.md`) 가 단일 진리. 화면 (AI 모델 상세 → 재현 매니페스트) 이 같은 출처

---

## 9. v2 방향

v2 의 핵심은 **단순 정확도 향상이 아닙니다** (CP216.2 통계 검정으로 ML 라인이 단순 통계 베이스라인 못 이김이 학계 통설임을 확인). 대신 다음 축에 집중합니다:

| 트랙 | 우선순위 | 내용 |
|---|---|---|
| Stage 1 — Data Layer (Supabase Pro) | 🔴 v1 막바지 | thin product DB 재연결, egress guard, 자동 갱신 cron |
| Stage 2 — Inference 클라우드화 | 🟡 v2 초반 | Modal / GitHub Actions CPU 추론. 운영/MLOps 가치 |
| Stage 3 — 제품 기능 | 🟡 v2 초반 | 전략 백테스트 확장, 알림, 개인화 |
| Stage 4 — 관측 / 신뢰성 | 🟢 운영 시작 후 | 응답 시간 SLO · 에러 모니터링 |
| Band v2 — Conformal / CBM | ⚠️ 야망 축소 | 정확도 향상 기대 낮음. **calibration 정직성 + selective output** 으로 정체성 강화 |
| Line v2 paradigm | 🔴 **포기 권고** | IC 통계 베이스라인과 동등 — 학계 통설상 추가 ROI 낮음 |
| LLM 통합 (XAI) | 🟢 차별화 핵심 | 자연어 설명 + 정직성 narrative + 침묵 trigger |
| 자동 후보 탐색 | 🟡 band 한정 | champion / challenger MLOps |
| Docker 트랙 | 🟡 v2 첫 CP | dev (CPU) + train (GPU). 재현성 환경 잠금 |

목표는 **"왜 오늘 이 종목 밴드가 넓어졌는지 + 언제 모델을 믿지 말아야 하는지"** 를 사용자에게 설명하는 것입니다.

자세한 설계는 [`docs/lens_v2_master_plan.md`](docs/lens_v2_master_plan.md) 참조. 본문에 XAI 트랙 / 도커 / 재현성 lock 확장 / Band v2 야망 갱신 결정 항목이 박혀있습니다.

---

## 10. Roadmap

| Version | 마감 | 내용 |
|---|---|---|
| v1.0 | 2026-06-21 (딥러닝실습 제출) | 3 slot 운영 (CP210 / CP153 / CP178) + CP216.2 통계 검정 + Vercel/Render 배포 + 학습 재현 wrapper 3개 |
| v1.5 | v2 진입 직전 | Supabase Pro 데이터 레이어 재연결 + 자동 갱신 cron |
| v2.0 | 설계 단계 | XAI 트랙 · Docker · MLOps champion/challenger · Band v2 calibration 정직성 + selective output · LLM 설명 |
| v3.0 | 후보 | Form 4 / 8-K NLP 외부 데이터 통합, 자산군 확장 (ETF · 한국) |

---

## 11. 면책

Lens 는 학술 및 포트폴리오 목적의 연구 프로젝트입니다. 모든 예측 / 신호 / 백테스트 결과는
정보 제공 목적이며 투자 자문, 매매 권유, 수익 보장을 의미하지 않습니다.

자세한 내용은 [DISCLAIMER.md](DISCLAIMER.md) 참조.

---

## 12. 운영 모델 매니페스트

운영 3 모델 (CP210 라인 / CP153 1D 밴드 / CP178 1W 밴드) 의 환경 / fold / seeds / 핵심 metric / 재현 절차는 화면 **AI 모델 → 모델 슬롯 → 재현 매니페스트** 에서 확인할 수 있습니다. 단일 진리 출처는 다음 문서:

- [`docs/v1_operating_models_reproducibility.md`](docs/v1_operating_models_reproducibility.md) — §0 공통 환경 + §1~3 모델별 매니페스트 + §4.5 PPT vs v1 매핑

학습 재현은 본 README §7.5 wrapper 또는 화면의 "학습 재현 — 한 줄 wrapper" 섹션 참조.

---

## 13. References

**ML for finance 학계 통설**

- Welch, I., & Goyal, A. (2008). *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*. Review of Financial Studies, 21(4), 1455–1508. — 50년 검증된 historical mean 베이스라인 우위 결과
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 / M5 Competitions: 100,000 time series and 61 forecasting methods*. International Journal of Forecasting. — ML 모델이 단순 statistical baseline 못 이기는 결과

**통계 검정 (CP216.2)**

- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics, 13(3), 253–263.
- Giacomini, R., & White, H. (2006). *Tests of Conditional Predictive Ability*. Econometrica, 74(6), 1545–1578.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. — cluster / block bootstrap 원전

**시계열 모델**

- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*. ICLR 2023.
- Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. Transactions on Machine Learning Research.
- Lu, W., Li, J., Li, Y., Sun, A., & Wang, J. (2020). *A CNN-LSTM-Based Model to Forecast Stock Prices*. IEEE Access.

**Quantile / Conformal**

- Koenker, R., & Bassett, G. (1978). *Regression Quantiles*. Econometrica, 46(1), 33–50.
- Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile Regression*. NeurIPS 2019.

---

## 14. 라이선스

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
