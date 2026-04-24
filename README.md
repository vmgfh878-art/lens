<div align="center">

# Lens

**AI 보조지표 기반 미국 주식 분석·연구 플랫폼**

예측이 아닌, **검증 가능한 밴드와 규칙**으로 의사결정을 돕습니다.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white)
![Status](https://img.shields.io/badge/status-Phase%201%20research-orange)

</div>

---

## 개요

Lens는 "AI가 미래를 맞춘다"가 아니라, **calibrated 예측 밴드 + 규칙 엔진 + 투명한 백테스트**로 개인 투자자의 중기 스윙 의사결정을 돕는 연구·시연 플랫폼입니다.

- **밴드 중심 학습** — 조건부 quantile(q10·q50·q90)을 예측하고 reliability diagram으로 검증
- **규칙 엔진** — AI 밴드 + 전통 기술지표 + 시장 regime 조합
- **투명한 평가** — walk-forward 백테스트, regime-split 성과, conformal 후처리

현재는 **Phase 1 연구 프로토타입** 단계이며, 공개 배포·상용화는 범위 밖입니다.

---

## 아키텍처

```text
  ┌───────────────┐      ┌─────────────────┐      ┌────────────────┐      ┌──────────────┐
  │   Collector   │ ───▶ │    Supabase     │ ◀─── │    Backend     │ ◀─── │   Frontend   │
  │   (Python)    │      │    Postgres     │      │    FastAPI     │      │   Next.js 14 │
  └───────────────┘      └─────────────────┘      └────────────────┘      └──────────────┘
          ▲                       ▲                        ▲
          │                       │                        │
   EODHD · FRED             배치 예측 저장            Local GPU 학습
   FMP · SEC EDGAR        (predictions 테이블)    PatchTST·CNN-LSTM·TiDE
```

```text
lens/
├── ai/              모델 학습·추론·백테스트 (PatchTST · CNN-LSTM · TiDE)
├── backend/
│   ├── app/         FastAPI API 서비스
│   ├── collector/   수집·백필·파생지표 계산
│   ├── db/          Supabase 스키마 및 런타임 마이그레이션
│   └── tests/       백엔드 테스트
├── frontend/        Next.js 14 프론트엔드
└── docs/            설계·진행 기록
```

---

## 빠른 시작

### 1. 환경 변수

프로젝트 루트에 `.env` 생성.

**필수**

| 키 | 설명 |
|---|---|
| `SUPABASE_URL` | Supabase 프로젝트 URL |
| `SUPABASE_KEY` | Supabase service role key |
| `EODHD_API_KEY` | EODHD Personal 플랜 키 |

**권장**

| 키 | 설명 |
|---|---|
| `FRED_API_KEY` | 거시지표 (FRED) |
| `FMP_API_KEY` | 재무제표 (FMP) |
| `WANDB_API_KEY` | 실험 추적 |
| `BACKEND_CORS_ORIGINS` | 프론트 허용 도메인 |

### 2. 초기 백필

```powershell
# 진행률 확인
python -m backend.collector.pipelines.backfill_status

# 역사 구간 일괄 백필 (중단·재개 지원)
python -m backend.collector.pipelines.bootstrap_backfill --indicator-batch-size 25
```

### 3. 일일 증분 동기화

```powershell
python -m backend.collector.pipelines.daily_sync
```

### 4. 백엔드·프론트엔드 실행

```powershell
# 터미널 1
cd backend
uvicorn app.main:app --reload

# 터미널 2
cd frontend
npm run dev
```

### 5. 모델 학습

`indicators` 데이터가 충분히 쌓인 뒤 실행합니다.

```powershell
python ai/train.py `
  --model patchtst `
  --timeframe 1D `
  --epochs 50 `
  --batch-size 128 `
  --use-wandb --save-run
```

---

## 데이터 파이프라인

| 층 | 테이블 | 소스 | 갱신 주기 |
|---|---|---|---|
| 원천 | `price_data` | EODHD | 일일 |
|  | `macroeconomic_indicators` | FRED | 일일 |
|  | `market_breadth` | FMP | 일일 |
|  | `company_fundamentals` | SEC EDGAR (XBRL) | 분기 |
| 파생 | `indicators` | 자체 계산 | 일일 (incremental) |
| 예측 | `predictions` | AI 배치 | 일일 |
| 평가 | `prediction_evaluations` · `backtest_results` | walk-forward | 온디맨드 |

### 피처 구성 (총 29종)

- **Base (17)** — OHLCV 파생, 로그 수익률, RSI · MACD · Bollinger, rolling volatility, volume z-score
- **Regime (3)** — VIX 기반 3단계 one-hot (`low` / `normal` / `high`)
- **Fundamentals (6)** — revenue·net_income·EPS YoY, ROE, debt_ratio, operating_margin
- **Presence flags (3)** — `has_macro` · `has_breadth` · `has_fundamentals`
  - 컨텍스트 결측 구간은 0 imputation + 이진 플래그 패턴으로 모델에 명시

### Leakage 방지 정책

- FRED는 **publication date** 기준 as-of join
- SEC EDGAR 재무는 **filing_date** 기준 `merge_asof(direction="backward")`
- 당일 신호는 t-1 종가 기반 지표만 사용
- Normalization 파라미터는 **training set에서만** 산출 (PatchTST는 RevIN per-sample)
- Train/Val/Test 경계에 **gap = h_max** (1D=20, 1W=12)

---

## 모델 학습

### 유니버스

- **S&P 500** 중 fundamentals 8-quarter gate + `seq_len` sufficiency 통과 **477 ticker**
- 기간: 2015-01-02 ~ 현재 (약 10년, 2,840 거래일)
- Timeframe: **1D** (seq=252, h∈{1, 5, 20}) · **1W** (seq=104, h∈{1, 4, 12})
- **1M은 표시 전용** — AI 예측·밴드·시그널 비활성 (차트·가격·거래량·기술지표만 노출)

### 아키텍처

| 모델 | 특징 | 역할 |
|---|---|---|
| **PatchTST** | Patch-based Transformer · RevIN | 메인 |
| **CNN-LSTM** | Convolutional + LSTM hybrid | 비교군 |
| **TiDE** | MLP dense encoder-decoder | 경량 선택지 |

### Loss 설계

```
total = λ_line · line + λ_band · (q_low + q_high) + λ_width · width + λ_cross · cross
```

| 컴포넌트 | 구현 | 역할 |
|---|---|---|
| **Line head** | `AsymmetricHuberLoss (α=1, β=2)` | 점 예측선. 과대예측 페널티 |
| **Band head** | `PinballLoss (q=0.1, 0.5, 0.9)` | 조건부 quantile 밴드 |
| **Width** | 평균 밴드 폭 | 과도한 불확실성 억제 |
| **Cross** | `ReLU(lower − upper)` | quantile crossing 방지 |

하이퍼파라미터·결정 근거는 [`docs/training_hyperparameters.md`](docs/training_hyperparameters.md).

---

## 프로젝트 상태

**Phase 1 (수업 6주)** — 연구 완성 + 시연 가능한 제품 확보가 목표.

| 배치 | 내용 | 상태 |
|---|---|:---:|
| 1 | 기반 스키마 · regime 피처 · AI 저장 테이블 | 완료 |
| 2 | EDGAR 다분기 · 8-quarter sufficiency gate | 완료 |
| 3 | Loss 구현 · baseline(ARIMA·Naive·Drift) · E2E 검증 | 완료 |
| **4** | **모델 학습 — 속도 최적화 · sufficiency · sweep · 본 학습** | **진행 중** |
| 5 | Walk-forward 백테스트 · calibration 진단 | 대기 |
| 6 | 연구 실험 — ablation · conformal · regime-conditional | 대기 |
| 7 | 연구 결과 대시보드 프론트 | 대기 |
| 8 | 보안 · 통합 테스트 · 발표 | 대기 |

---

## 문서

| 파일 | 역할 |
|---|---|
| [`docs/project_journal.md`](docs/project_journal.md) | 기획 변천 · 배치·CP 로그 · 결정 이력 · 블로커 (append-only) |
| [`docs/training_hyperparameters.md`](docs/training_hyperparameters.md) | 학습 파라미터 단일 진실 소스 (결정 · 대안 · 탈락 근거) |

---

## 철학

Lens는 다음 원칙을 유지합니다.

- **AI는 보조지표**, 의사결정 주체는 사용자다.
- **밴드 품질이 연구의 본체**, 점 예측은 참고지표다.
- **Calibration으로 증명**할 수 없는 confidence는 제품에 넣지 않는다.
- Survivorship bias, release lag, normalization leakage 같은 함정을 **명시적으로** 처리한다.
- 금융 예측의 구조적 한계(낮은 SNR · non-stationarity)를 인정하고, 승부는 **밴드·규칙의 투명성**에서 본다.

---

## 라이선스

연구·학습 목적의 비공개 프로토타입입니다. 외부 배포 계획은 Phase 1 범위 밖입니다.

<div align="center">
<sub>Built as a 6-week research project — Lens 2026</sub>
</div>
