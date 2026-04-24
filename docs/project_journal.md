# Lens 개발·기획 저널

> 기획 변천·의사결정·배치 진행·발견 이슈를 **연대기 순으로 누적**하는 단일 로그.
> 업데이트 규칙: Edit 전용 append. Rewrite 금지. 섹션별 하단에 신규 항목 추가.
>
> 마지막 업데이트: 2026-04-25
> 관련 문서:
> - 기획안: `.claude/plans/c-users-user-onedrive-desktop-2026-1-22-quiet-gray.md` (v3 확정)
> - 학습 하이퍼파라미터: `docs/training_hyperparameters.md`

---

## 0. 문서 지도

| 문서 | 역할 | 업데이트 성격 |
|---|---|---|
| 기획안 v3 | 프로젝트 전체 범위·방법론·일정 | 큰 방향 바뀔 때만 (v4 등) |
| `training_hyperparameters.md` | 학습 파라미터 SoT | 새 sweep 결과·결정 추가될 때 |
| **`project_journal.md` (이 파일)** | **연대기 기록 전부** | **매 배치·CP·결정·발견마다** |

---

## 1. 기획 변천

### v1 (2026-04-20) — 초안
- PPTX 공백 전반 채움
- Loss α/β 방향 pushback
- 경쟁사 분석 (TradingView 예측 밴드 4종)
- EODHD 약관 이슈 부상
- Phase 2 로드맵 초안

### v2 (2026-04-20) — 사용자 확정 3건
- Loss α=1, β=2 (보수적) 확정
- 월봉 표시 전용 확정
- Phase 2 데이터 소스 FMP Pro 우선

### v3 (2026-04-23) — 근본 방향 전환
Phase 1 목표: "제품 완성·수익화 대비" → **"연구 완성 + 시연 가능 수준"**

확정 7건:
1. 데이터 소스 Phase 1~2 전 기간 EODHD 유지 (공개 배포 결정 시점에만 FMP Pro 검토)
2. 타깃 우선순위 재정렬: 밴드(quantile) 본체, 점 예측 보조 (Makridakis·EMH·Gneiting 근거)
3. 평가 메트릭 재순위: pinball·CRPS·coverage ★★★, MAPE·방향정확도 ★
4. 분봉·10분봉 raw 학습 **명시 반려** (SNR·저장·horizon·플랜 근거)
5. 단타 방향 제품 전환 **명시 반려** (Barber·Chague·Odean 실증 근거)
6. 유니버스 확장: 코인 Phase 2 무조건, 한국주식 후순위, intraday aggregate Phase 2
7. Regime-conditional band를 Phase 1 후반으로 당김 (원래 Phase 2)

신규 섹션: §3.8 연구 실험 계획, §6.2·§10 확장 로드맵.

---

## 2. 배치 로드맵 (8 배치)

| # | 이름 | 범위 | 상태 |
|---|---|---|---|
| 1 | 기반 스키마·regime·AI 저장 테이블 | `indicators.regime_*`, `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results` | ✅ 완료 |
| 2 | EDGAR 다분기 + fundamentals 8-quarter gate | multi-quarter fetch, as-of join, sufficiency gate | ✅ 완료 |
| 3 | Loss 실장 + Baselines + E2E | `PinballLoss`, `AsymmetricHuberLoss`, ARIMA/Naive baseline, predictions writer 검증 | ✅ 완료 |
| 4 | **모델 학습 (Phase A/B/C)** | A: 속도 최적화·데이터 준비 · B: HP sweep · C: 본 학습 | 🔄 CP2.5 진행 |
| 5 | Walk-forward backtest + calibration | regime split, reliability diagram, PIT, empirical coverage | ⏳ |
| 6 | 연구 실험 집중 | feature ablation, target design, conformal 후처리, regime-conditional | ⏳ |
| 7 | 프론트엔드 연구 대시보드 | 핵심 5 화면 + 연구 결과 대시보드 (6번 화면) | ⏳ |
| 8 | 보안·통합 테스트·발표 | RLS/CORS, integration, 발표자료 | ⏳ |

---

## 3. 완료 배치

### 배치 1 — 기반 스키마 + regime + AI 저장

**기간**: ~2026-04-23 초중반

**산출물**:
- `indicators` 테이블에 `regime_label`, `regime_low`, `regime_normal`, `regime_high` 컬럼.
- VIX 기반 3단계 regime 탐지 (low<15 / normal 15~25 / high>25).
- AI 저장 테이블 신설: `model_runs`, `predictions`(line_series·band_quantile_*), `prediction_evaluations`, `backtest_results`.
- 1M 표시 전용 정책 적용 — `/predict?timeframe=1M`은 `timeframe_disabled` 응답, predictions 1M 행 생성 금지.

**핵심 수치**:
- regime 백필 **472,844행**.
- 1M sufficiency threshold 63 → 24로 조정 (실측 분포 반영).

**발견**:
- predictions 테이블 0행 이유는 writer 문제 아님 → 학습 미실행이 원인. 배치 3 E2E 검증으로 writer 정상 확인.

---

### 배치 2 — EDGAR 다분기 + fundamentals 8-quarter gate

**기간**: ~2026-04-23 후반

**산출물**:
- `backend/collector/sources/edgar.py` multi-quarter fetch: `fp∈{Q1,Q2,Q3,FY}`, `MAX_QUARTERS=12`.
- `REVENUE_TAGS` fallback 리스트 (업종·시기별 XBRL 태그 변이 흡수).
- `_dedupe_quarter_rows` filed-latest 선택, `_match_value` as-of 매칭.
- `feature_svc._apply_fundamental_features` — `pd.merge_asof(direction="backward")`, 8-quarter sufficiency gate via `np.searchsorted`.

**핵심 수치**:
- 503/503 ticker 재수집 완료.
- 총 **5,807 row**, 평균 **11.54 분기/ticker**.
- 8-quarter gate 통과 477 ticker, **26 ticker 제외**.

**발견**:
- `compute_indicators.py`에서 fundamentals를 `gte date` 필터로 가져오면 8-quarter gate 판정이 증분 모드에서 깨짐 → CP2에서 "전체 기간 로드"로 수정.

---

### 배치 3 — Loss + Baselines + E2E

**기간**: ~2026-04-23 말

**산출물**:
- `ai/loss.py`: `PinballLoss`, `AsymmetricHuberLoss(α=1,β=2)`, `WidthPenaltyLoss`, `BandCrossPenaltyLoss`, `ForecastCompositeLoss`.
- `ai/baselines.py`: `_naive_predictions`, `_drift_predictions`, `_arima_predictions` + `save_model_run` 연결.
- `ai/inference.py`:
  - `sort_prediction_triplet` (quantile crossing 후처리 정렬)
  - `PinballLoss(sort_quantiles=True)` 사용
  - pinball_loss DB save
- AAPL 5-epoch smoke 학습 완주, `baseline_arima` run 1건 저장, 9 예측 행 작성 확인.

**핵심 발견**:
- coverage 13%(5 epoch) → under-trained. seq_len=60 × 50 epoch 수동 돌리면 coverage 0.7425 도달 확인 → loss weight가 올바른 coverage region 찾아갈 수 있음.

---

## 4. 배치 4 (진행 중) — 모델 학습

### 구성

**Phase A (pre-training 준비)** — sweep 전 환경·데이터 확정
- A-0: 속도 최적화 + 기초 검증
- A-0.5: sufficiency 재확인 (477 ticker breakdown)
- A-1: Loss weight sweep
- A-2: Transfer validation

**Phase B (hyperparameter sweep)**
- B-1: LR sweep
- B-2: Architectural sweep
- B-3: 1W transfer

**Phase C (본 학습)**
- 6 결정 축 확정 후 최종 스펙 잠그고 Wave=seed × 3.

### CP 로그

#### CP1 — 속도 최적화 (코드 완료 / 환경 검증 대기)
**적용**:
- `ai/train.py`·`ai/inference.py`: CUDA-only `autocast(bf16)`, `torch.compile(mode="reduce-overhead")`, `set_float32_matmul_precision("high")`, DataLoader `num_workers=6 / pin_memory=True / persistent_workers=True`.
- `ai/preprocessing.py`: `ai/cache/features_{tf}_{hash}.pt` 캐시.
- wandb `lens-batch4` 프로젝트 기본 설정.

**수치**:
- CPU baseline (AAPL 1D PatchTST batch=128 1epoch): **4.337s → 2.074s**.
- 기존 17 테스트 green.

**상태**: 코드 레벨 완료. GPU bf16+compile 실측 타이밍 + W&B 프로젝트 링크 검증은 **CP7(A-0 최종)** 로 이관. 환경(CUDA 미보유, wandb 미설치) 의존이라 코드 블로커 아님.

#### CP2 — Fundamental NaN 처리 (코드 완료 / 데이터 블로커 노출)
**적용**:
- `feature_svc.py`: 재무 6컬럼 `fillna(0.0)`, `has_fundamentals` BOOLEAN 플래그, `dropna` 범위를 `_BASE + _REGIME` + 플래그로 축소.
- `schema.sql` / `ensure_runtime_schema.py`: `indicators.has_fundamentals BOOLEAN NOT NULL DEFAULT FALSE`.
- `compute_indicators.py`: fundamentals 전체 기간 로드로 증분 모드 이슈 해결.
- `preprocessing.py`: 학습 dropna를 `REQUIRED_FEATURE_COLUMNS` 기준으로 변경.
- 신규/수정 테스트 5건.

**수치**:
- feature columns: 26 → **27** (has_fundamentals 추가)
- AAPL 1D rows after dropna: **119 → 753** (재무 NaN 블로커 제거 확인)
- AAPL 1W rows after dropna: **26 → 157**
- has_fundamentals=1 비율: 1D **19.22%**, 1W **19.73%**
- indicators `has_fundamentals` 컬럼 존재: Y
- 기존 17 + 신규 5 테스트 green.

**블로커 (신규, OPEN)**:
- AAPL 1D 753행은 목표 ≥2200에 못 미침. 원인은 재무 NaN이 아니라 **가격/지표 실데이터 절대 길이 부족** — AAPL 기준 대략 2023-04 이후 데이터만 존재.
- has_fundamentals 19%는 기대치 대비 낮음. 재무 데이터 커버리지 재확인 필요.
- **다음 단계**: CP3(26 ticker 제외) 아님. **CP2.5 데이터 길이 감사** 선행.

#### CP2.5 — 데이터 길이 감사 (조사 전용, 완료)
**산출물**:
- `scripts/diagnostics/data_length_audit.py`
- `docs/cp2.5_report.md`

**확정된 수치**:
- `price_data`: 503 ticker, row_count p50=**2843** (2015-01-02~2026-04-23). **10년 완전 보유**.
- `company_fundamentals`: 503 ticker, 분기 수 p50=**12**, 8분기 이상 **477** ticker 일치.
- `indicators` 1D: min_date p10=**2023-04-18**, row_count p50=**757**. **3년치만 저장됨**.
- `indicators` 1W: min_date p10=2023-04-21, row_count p50=158.
- `indicators` 1M: row_count p50=37.
- has_fundamentals=1 time-weighted 비율 **0.03%** (거의 0. 이전 CP2 보고 19%는 단순 행 수 비율, 실제 학습 기여는 거의 없음).

**AAPL 파이프라인 단계별**:
```
N0 (price raw)      2843
N1 (resample)       2843
N2 (base dropna)     753   ← 2090행 증발
N3 (regime merge)    753
N4 (fund merge+8Q)  2843   (측정 아티팩트로 판단)
N5 (final dropna)    753
```

**가설 판정**:
- H1 (price 자체 짧다): REJECTED
- H2 (price OK, indicators 짧다): HOLDS (저장된 indicators가 3년뿐)
- H3 (피처 파이프라인 손실): HOLDS (N2 단계에서 2090행 손실)
- **결정: H2 + H3 복합**

**근본 원인 추정** (CP2.6에서 확정):
- indicators `min_date = 2023-04-18`과 base dropna 생존선 일치.
- 컨텍스트 테이블(`macroeconomic_indicators` 또는 `market_breadth`) 중 하나가 **2023-04-18 이후만 존재**하여 merge 후 NaN → base dropna에서 과거 구간 일괄 삭제 가능성 유력.
- 또는 `price_data.per/pbr` 등 개별 컬럼이 최근에만 채워졌을 가능성.

**다음 단계**: CP2.6 — N1→N2 손실 컬럼 단위 특정 + 컨텍스트 백필 또는 플래그 패턴 확장.

#### CP2.6 — 컨텍스트 플래그 확장 + 과거 구간 복구 (완료)
**적용**:
- `feature_svc.py`: `has_macro`·`has_breadth` BOOLEAN 플래그 신설, 각 컨텍스트 컬럼 NaN→0.0 imputation. fundamentals와 동일 패턴 복제.
- `schema.sql` / `ensure_runtime_schema.py`: `indicators.has_macro`·`has_breadth` 컬럼 추가.
- `compute_indicators.py`: `drop_duplicates(["ticker", "timeframe", "date"])` 중복 upsert 가드 추가.
- `FEATURE_COLUMNS`: 27 → **29** (has_macro + has_breadth).

**진단 결과 (원인 확정)**:
- `macroeconomic_indicators`: 테이블 자체는 2015-01-01부터 있으나 **`credit_spread_hy` 첫 실값 2023-04-24** → merge 후 과거 NaN.
- `market_breadth`: 시작일 **2016-04-22** → 그 이전 NaN.
- `price_data.per/pbr`: 첫 실값 2026-03-03. `_BASE_FEATURE_COLUMNS`에 포함 안 되어 이번 병목 무관 (기록만).
- **원인 분기: 복합(A+B)**. 해결 전략은 백필 대신 **플래그 + 0 imputation**으로 우선 복구.

**복구 수치**:
- AAPL 1D: 753 → **2784** (목표 2200 초과)
- AAPL 1W: 157 → **532** (목표 450 초과)
- 전체 1D row_count p10/p50/p90: **1730.2 / 2783 / 2784.0**
- 전체 1W row_count p10/p50/p90: **314.4 / 532.0 / 532.0**
- indicators 1D min_date p10: **2015-03-27** (2023-04-18 → 2015로 복구)
- has_macro time-weighted: 1D **98.4%**, 1W **98.2%**
- has_breadth time-weighted: 1D **89.3%**, 1W **96.6%**
- has_fundamentals time-weighted: 1D **6.24%**, 1W **6.93%** (과거 구간 복구로 분모 확장에 따른 자연 감소. 설계적 수용)
- 기존 17+5 + 신규 21 테스트 green.

**미완료 (분리 관리)**:
- `1M` full-range recompute: `RemoteProtocolError`로 끝까지 닫히지 않음. 1M은 표시 전용이라 학습 경로 무영향. **CP2.7(별도 안정화)** 로 분리.

**CP3 준비 확정 수치**:
- 26 ticker sufficiency 제외 대상 확정: Y
- seq_len=252 (1D) 가능: 476/477 (1 ticker 추가 탈락)
- seq_len=104 (1W) 가능: 472/477 (5 ticker 추가 탈락)
- **timeframe별 제외 리스트가 다름** — CP3에서 분리 필터 처리 필요.

---

## 5. 주요 결정 이력

| # | 결정 | 채택값 | 근거 섹션 |
|---|---|---|---|
| D1 | Loss α/β | α=1, β=2 (보수적 과대예측 페널티) | 기획안 §3.4.2 |
| D2 | 월봉 정책 | 표시 전용, AI 비활성 | §1.3, §2.2.1 |
| D3 | 타깃 우선순위 | 밴드 본체, 점 예측 보조 | §3.3 |
| D4 | 분봉 raw 학습 | 명시 반려 | §2.2.2 |
| D5 | 단타 제품 전환 | 명시 반려 | §2.2.3 |
| D6 | 데이터 소스 | Phase 1~2 EODHD 유지 | §2.1, §2.6 |
| D7 | 백테스트 cost | 10 bps 기본 + [5·10·15·20·30] 민감도 | §4.2 |
| D8 | Regime-conditional | Phase 1 후반으로 당김 | §2.7, §3.8 |
| D9 | Baseline | ARIMA + Naive + Drift | §3.2 |
| D10 | Quantile | (10%, 50%, 90%) | §3.4.1 |
| D11 | Quantile crossing | 후처리 정렬 + cross_loss 병용 | §3.4.1 |
| D12 | Gradient clipping | `max_norm=1.0` 고정 | `training_hyperparameters.md §4` |
| D13 | LR scheduler | Cosine + warmup 5 고정 | 同 §5 |
| D14 | Augment / Ensemble / Bagging | Phase 2 이관 | 同 §6 |
| D15 | Train/Val gap | `h_max` (1D=20, 1W=12) | 同 §7 |
| D16 | Run orchestration | Wave=seed × 3 | 同 §8 |
| D17 | Val 주기 / primary | every epoch / `val_pinball_loss` | 同 §9 |
| D18 | 컨텍스트 결측 처리 | `has_macro`·`has_breadth` BOOLEAN 플래그 + NaN→0.0 imputation. `fundamentals` 패턴 복제. 과거 백필보다 플래그 우선 | CP2.6 |
| D19 | `1M` timeframe | 표시 전용이므로 학습 파이프라인과 분리. recompute 실패는 학습 블로커 아님 | 기획안 §2.2.1 + CP2.6 판단 |
| D20 | 재무 시계열 범위 (Phase 1) | 2021~2025 12분기만 신호. 2015~2020 무정보 허용 (has_fundamentals=0). 10년 백필은 Phase 2 | CP2.6 수용 판단 |

---

## 6. 현재 블로커

| 상태 | 항목 | 조사/해결 |
|---|---|---|
| ✅ RESOLVED (CP2.6) | indicators 과거 구간 부재 | `has_macro`·`has_breadth` 플래그 + 0 imputation 패턴 적용. 전체 유니버스 p50 2783(1D)/532(1W) 복구 |
| 🟢 ACCEPTED | `has_fundamentals` time-weighted 6~7% — 재무 데이터가 2021~2025년 12분기만 있음. 2015~2020 구간은 무정보 학습 | Phase 2에서 FMP full history 백필 검토 |
| 🟡 OPEN | `1M` full-range recompute 미완료 (`RemoteProtocolError`). 학습 경로 무영향, 표시 전용 피처만 부분 결손 | CP2.7 (저우선순위 분리 트랙) |
| 🟡 PENDING | CP1 GPU bf16+compile 실측 / W&B 프로젝트 링크 | CP7(A-0 최종)에서 해결 |

---

## 7. Phase 2 적치 (Phase 1 범위 밖)

- Data augmentation (Cutout/Masking 우선 후보)
- Ensemble / Bagging / Stacking
- Walk-forward backtest + calibration 다각화 (원래 배치 5에서 Phase 1 범위는 커버)
- 공개 배포 + FMP Pro/Tiingo Commercial 소스 전환
- 논문화 ("AI Band + Rule-based Hybrid Decision System")
- 코인 universe (Binance 무료 API, 별도 모델)
- 한국 주식 (후순위 옵션)
- Intraday aggregate features (VWAP·realized vol·volume profile, 일봉 머지)
- HMM regime state 추론
- Purged k-fold + embargo CV (Lopez de Prado Ch.7)

---

## 8. 지시서 운영 룰 (CP3부터 적용)

모든 CP 지시서에 3개 블록 필수:
1. **예상 시간** — 코드·테스트·스모크·총 체감 (기준: 5060 Ti 16GB, bf16+compile)
2. **권한 번들** — 파일 쓰기 범위, 허용 명령, 금지, 휴먼 개입 트리거
3. **종료 보고 포맷** — 숫자 명시 CP 보고 (각 CP별 템플릿)

사용자의 "권한 승인 매번 눌러야 하는 피로" 완화 목적.

---

## CHANGELOG

- **2026-04-24**: 최초 작성. 기획 v1~v3, 배치 1~3 완료 내역, 배치 4 CP1·CP2 진행 상태, 결정 D1~D17, 블로커 3건, Phase 2 적치 반영.
- **2026-04-25**: CP2.5 (데이터 길이 감사) 완료 반영.
  - price_data·fundamentals는 10년·477 ticker 정상 확정.
  - 병목 특정: indicators 테이블 min_date=2023-04-18, base feature dropna에서 AAPL 2843→753 손실.
  - 가설 H2+H3 복합으로 결정. 원인은 컨텍스트 merge NaN 유력.
  - 블로커 재분류 (재무 NaN 19%는 측정 관점 혼동이었음, 실제 time-weighted 0.03%로 교체).
  - CP2.6 지시서 발주.
- **2026-04-25**: CP2.6 (컨텍스트 플래그 확장) 완료 반영.
  - 원인 복합(A+B) 확정: `credit_spread_hy` 2023-04-24·`market_breadth` 2016-04-22 시작.
  - 해결 전략: 백필 대신 `has_macro`·`has_breadth` 플래그 + 0 imputation. fundamentals 패턴 복제.
  - AAPL 복구: 1D 753→2784, 1W 157→532. 전체 p50 1D 2783 / 1W 532 달성.
  - FEATURE_COLUMNS 27→29.
  - 결정 D18(컨텍스트 결측 플래그), D19(1M 분리), D20(재무 5년 허용) 추가.
  - `1M` recompute 미완료는 CP2.7로 분리 관리.
  - CP3 지시서 발주.
