# Lens 개발·기획 저널

> 기획 변천·의사결정·배치 진행·발견 이슈를 **연대기 순으로 누적**하는 단일 로그.
> 업데이트 규칙: Edit 전용 append. Rewrite 금지. 섹션별 하단에 신규 항목 추가.
>
> 마지막 업데이트: 2026-04-27
> 관련 문서:
> - 기획안: `.claude/plans/c-users-user-onedrive-desktop-2026-1-22-quiet-gray.md` (v3 확정)
> - 학습 하이퍼파라미터: `docs/training_hyperparameters.md`

---

## 0. 문서 지도

| 문서 | 역할 | 업데이트 성격 |
|---|---|---|
| 기획안 v3 | 프로젝트 전체 범위·방법론·일정 | 큰 방향 바뀔 때만 (v4 등) |
| `training_hyperparameters.md` | 학습 파라미터 SoT | 새 sweep 결과·결정 추가될 때 |
| `model_architecture.md` | 모델 구조 갭 분석·개선 옵션 | CP3.5 결정 시 / 구조 변경 시 |
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
- 본인 계좌 자동매매 (Alpaca 우선, 미국주식 paper → live; 본인 계좌만이라 규제 자유로움)
- (조건부 Phase 3) Phase 1·2 트랙레코드가 calibration·백테스트로 1년+ 쌓이면 TradingView Pine Script Marketplace 진입안 검토

---

## 8. 지시서 운영 룰 (CP3부터 적용)

모든 CP 지시서에 3개 블록 필수:
1. **예상 시간** — 코드·테스트·스모크·총 체감 (기준: 5060 Ti 16GB, bf16+compile)
2. **권한 번들** — 파일 쓰기 범위, 허용 명령, 금지, 휴먼 개입 트리거
3. **종료 보고 포맷** — 숫자 명시 CP 보고 (각 CP별 템플릿)

사용자의 "권한 승인 매번 눌러야 하는 피로" 완화 목적.

---

## 9. PPT 대비 변경 이력 (중간발표 → 기말발표 비교용)

> PPT 초안(2026-04 초)이 중간발표 기준 자료. 그 이후 누적된 변경을 카테고리별로 정리.
> 새 결정·변경 발생할 때마다 해당 카테고리 표 하단에 한 줄 추가 (Edit append). 기말 발표 자료 정리 시 그대로 인용.

### 9.1 범위 축소 (PPT에 있었으나 제외/축소)

| 항목 | PPT | 현재 | 근거 / 결정 시점 |
|---|---|---|---|
| 1M 월봉 AI 학습·예측 | 1D/1W/1M 모두 학습·예측 | **표시 전용** (캔들·기술지표만) | 기획 v2, 데이터·신뢰성 부족 |
| 점 예측 정확도(MAPE/방향) 핵심 KPI | 메인 평가축 | ★ 보조 메트릭으로 강등 | 기획 v3, Makridakis 경쟁 결과·EMH 근거 |
| 학습 universe 477 ticker 전부 | S&P 500 ~503 | **1D 473 / 1W 421** (sufficiency gate 통과분만) | CP3, 데이터 부족 ticker 명시 제외 정책 |

### 9.2 범위 추가 (PPT에 없었으나 추가)

| 항목 | 내용 | 결정 시점 |
|---|---|---|
| Sufficiency gate 정책 | 1D ≥ 450 거래일 / 1W ≥ 78주 / 재무 ≥ 8분기. 부족 시 학습·추론 제외 (NaN 끌고 가지 말 것) | 기획 v3, CP2 |
| 출력 정합성 단일 함수 | `apply_band_postprocess` 한 곳으로 val/test/inference 통일 | CP3.5 (P1-2) |
| 출력 헤드 두 방식 | `band_mode="direct"` (q10/q90 직접) vs `"param"` (center + log_half_width로 cross 구조 차단) | CP3.5 (A-3) |
| RevIN denormalize_target | norm-only → norm + target-channel denorm 정합 (PatchTST 출력 raw scale 복원) | CP4 (F-1) |
| Channel Independence aggregate | 29채널 mean 평균화 → `target / mean / attention` 3종 선택 (기본 target) | CP4 (F-2) |
| Ticker embedding | global model + concat-after-backbone, 1D 473 / 1W 421, OOV id = num_tickers, emb_dim=32 | CP4 |
| TiDE per-step head | `decoded.mean(dim=1)`로 horizon 표현 소실 → per-step `Linear(_, 1)` / `Linear(_, 2)` head로 복원 | CP5 |
| CNN-LSTM dilated TCN | kernel=3 단일 dilation RF=5 → `[1,2,4,8]` exponential dilation, RF=31 | CP5 |
| Calendar future covariate | TiDE 논문 spec 보강. 7개 결정론적 캘린더 채널 (`day_of_week_sin/cos`, `month_sin/cos`, `is_month_end`, `is_quarter_end`, `is_opex_friday`)을 dataloader에서 즉석 파생, 3 모델 모두 입력 (n_features 29→36, target_channel_idx=0 보존), TiDE는 추가로 horizon decoder 단에 future_covariate per-step 주입 (`forward(x, ticker_id, future_covariate)`). DB·collector·cron 무관 | CP6.5 ✅ |
| Init 정책 통합 | trunc_normal(std=0.02), bias=0, LayerNorm gamma=1 — 3 모델 모두 BERT 스타일 | CP3.5 (E-1) |
| Dropout 위치 통일 | rate는 sweep / 위치는 고정 (PatchTST input·output·encoder, CNN-LSTM conv·attn·LSTM, TiDE ResBlock) | CP3.5 (E-2) |
| Early stopping + best 복원 | val_total monitor, patience=10, min_delta=1e-4, best epoch state_dict 복원 후 저장. 메타에 `best_epoch / best_val_total / early_stopped` | CP6 |
| Regime-conditional band | Phase 2 → **Phase 1 후반(5주차)으로 당김** | 기획 v3 |
| Conformal 후처리 (Split / CQR) | Phase 2 논문화 → Phase 1 5주차 실험으로 당김 | 기획 v3 |
| 연구 결과 대시보드 페이지 | 기존 5화면에 calibration·reliability·백테스트 종합 6번 화면 추가 | 기획 v3 |
| 본인 계좌 자동매매 (Phase 2) | Alpaca 기반, 본인 계좌 한정 (규제 자유) | 2026-04-25 토론 |
| TradingView Marketplace (Phase 3 조건부) | 1년+ trackrecord 쌓이면 진입 검토 | 2026-04-25 토론 |

### 9.3 우선순위·메서드 변경 (PPT에 있던 것을 다르게)

| 항목 | PPT | 현재 | 근거 |
|---|---|---|---|
| Phase 1 목표 | 제품 완성·수익화 대비 | **연구 완성 + 시연 가능 수준** | 기획 v3 |
| 타깃 우선순위 | 점 예측선 본체 | **밴드(quantile) 본체**, 점 예측 보조 | Makridakis·EMH·Gneiting 근거 |
| 평가 메트릭 순위 | MAPE·방향정확도 중심 | pinball·CRPS·empirical coverage ★★★ | 기획 v3 |
| Loss α/β | 미정 | α=1, β=2 (overprediction 페널티 큰 보수적 예측, 철학 결정) | 기획 v2 |
| 데이터 소스 | "EODHD" 단순 명기 | **Phase 1~2 EODHD 유지**, 공개 배포 결정 시점에만 FMP Pro 검토 (소스 swap 난이도 낮음 근거) | 기획 v3 |
| Cross-validation | 70/15/15 time-ordered | 70/15/15 + **Phase 1 후반 walk-forward 추가** (백테스트와 split 일치) | 기획 v3 |
| 방향 정확도 목표 | 55% | **52% (완화)** | 기획 v3, validation 성적 보며 재조정 |

### 9.4 명시적 반려 (검토 후 채택 안 함)

| 제안 | 반려 사유 | 결정 시점 |
|---|---|---|
| 분봉·10분봉 raw 학습 | (a) Signal-to-noise 악화 (b) EODHD Personal 미포함 (c) 저장 비용 폭발 (d) horizon 미스매치 | 기획 v3 §2.2.2 |
| 단타 방향 제품 전환 | Barber·Chague·Odean 실증: 개인 데이트레이더 0.5%만 순이익. HFT 인프라 격차 | 기획 v3 §2.2.3 |
| 데이터 무작정 늘리기 (분봉·과거 2015 이전) | 효과 작음 + 구조적 break/생존편향 + leakage 위험 | 2026-04 토론 |
| 토스/카카오페이 등 슈퍼앱 "등록" | 한국 핀테크 마켓플레이스 시스템 부재. BD 계약만 가능 | 2026-04-25 토론 |
| 거시 release **값** future covariate | 누수 위험. release **date**만 합법 | 2026-04 토론 |
| TiDE에만 future covariate (다른 모델 미주입) | 비교 평가 오염. 캘린더 7종은 3 모델 모두 입력으로 받기 | CP6.5 |

### 9.5 메타 운영 룰 추가 (PPT에 없던 프로세스 원칙)

| 룰 | 내용 | 결정 시점 |
|---|---|---|
| 리뷰어 / 구현 분담 | 사용자 = 의사결정 + 리뷰. 오케스트레이터 = 지시 + 검증. 구현 에이전트 = 코드 + 테스트 + 보고서 | D1 |
| CP 단위 분할 | 큰 배치를 CP1·CP2·CP3.5 등으로 쪼개 단위 검증 | 배치 4부터 |
| 핵심 컴포넌트 존재 체크리스트 (D21) | 모든 CP 보고서에 "RevIN denorm 호출", "ticker emb concat" 등 핵심 컴포넌트 존재 검증 헤딩 필수 | CP3.5 후속 |
| 비용보다 fidelity 우선 (D22) | 시간 들어도 논문대로 정확하게 (TiDE rename 거부, RevIN 반쪽 구현 거부 등) | 2026-04-24 |
| 토큰 효율 룰 (D24) | 별도 `cpN_instruction.md` 금지. 지시는 직접 프롬프트. 반복 컨텍스트는 `CLAUDE.md`에 박기. 보고서만 `.md` | 2026-04-25 |

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
- **2026-04-25**: CP3 (sufficiency gate + 70/15/15 split) 완료 반영.
  - 학습 대상 ticker: 1D 473 / 1W 421 (입력 477 기준).
  - Sample 합계: 1D train 759k / val 162k / test 163k. 1W train 116k / val 24k / test 25k.
  - `min_fold_samples=50`, `gap=h_max` (1D=20, 1W=12) 확정.
  - 신규 4 테스트 추가, 전체 25개 green.
  - `ai/splits.py`, `ai/preprocessing.py` (`build_dataset_plan`, `prepare_dataset_splits`), `ai/train.py` 연결 완료. Dry-run으로 split 구성 확인.
- **2026-04-25**: 리뷰어 라운드 — 모델 아키텍처 갭 5건 도출.
  - **P1-1**: 추론 정렬이 line head를 q50 자리에 끼워 의미 파괴 (`ai/inference.py:87~94`).
  - **P1-2**: val·inference 후처리 불일치로 모델 선택·calibration 신뢰성 손상 (`ai/train.py:213~220`).
  - **P2-1**: PatchTST에 RevIN 없음. 문서 명시(README, hyperparameters)와 코드 불일치. **지시서·검수 양식 빈틈** 인정. 다음 CP부터 "구조 fidelity 체크 항목" 보고 양식에 추가.
  - **P2-2**: TiDE가 사실상 일반 MLP. 사용자 지시: 리네이밍 옵션 폐기, **TD-1 (논문 fidelity 70%) 골격 구현**.
  - **P2-3**: WidthPenaltyLoss 음수 가능 (`ai/loss.py:73~84`).
  - 사용자 결정 (확정):
    - P1-1: line head 보존 / `torch.sort` 대상에서 line 제외 / sort는 q10·q90만.
    - P1-2: 단일 후처리 함수 `apply_band_postprocess`로 val·test·inference 통일.
    - P2-2: TiDE 제대로 구현 (TD-1 이상).
  - 사용자 결정 (보류 — `docs/model_architecture.md`에서 옵션 비교 후 선택):
    - **A**: 출력 헤드 구조 (직접 q10/q90 vs center+log_half_width 파라미터화 vs ablation).
    - **B**: PatchTST 보강 범위 (RevIN만 vs RevIN+channel independence vs 단계 진행).
    - **C**: CNN-LSTM 안정화 + attention pooling 여부.
    - **D**: TiDE 골격 범위 (TD-1 권고 / TD-2 풀구현은 비용 대비 이득 작음).
    - **E**: Init·Dropout 보강 적용 여부.
  - 산출물: 신규 `docs/model_architecture.md` (옵션·장단점·비용·결정표).
  - **메타 룰 추가 (D21)**: CP 지시서·검수 양식에 "이 모듈에 명시된 핵심 구성요소가 코드에 실제 존재하는가? 누락 항목 명시" 섹션 필수 포함. RevIN 누락 같은 fidelity gap 재발 방지.
- **2026-04-25**: CP3.5 결정 확정 (사용자 직접 선택, "비용보다 fidelity 우선" 방침).
  - **A = A-3**: 출력 헤드 두 방식 (직접 q10/q90 + center/log_half_width 파라미터화) **둘 다 구현, ablation 비교**.
  - **B = B-2**: PatchTST 풀 구현 — **RevIN + Channel Independence** 둘 다 적용 (논문 fidelity 80%).
  - **C = C-2**: CNN-LSTM **안정화 (LayerNorm + residual + grad clip) + attention pooling** 적용.
  - **D = D-1**: TiDE 최소 골격 (논문 fidelity 70%) — encoder/decoder + ResidualBlock + temporal decoder + lookback skip. **TD-2 (풀구현, 미래 covariate 처리 포함) 채택 안 함**: 사용자 명시 합의 — Lens 데이터에 미래 covariate가 사실상 없음 (시점 t에 t+h의 거시·재무·breadth는 미지). 풀 구현 추가 가치 없음.
  - **E1 = 적용**: BERT 식 truncated normal init (std=0.02) 전 모델 통일. 사유: init은 sweep 대상 아님 → 고정으로 통일해야 모델 비교 노이즈 제거. 비용 0 (apply 한 줄).
  - **E2 = 적용**: Dropout 위치 표준 보강 (PatchTST 3곳, CNN-LSTM 2곳, TiDE residual block 통합). 사유: 위치는 sweep 대상 아님 (조합 폭발), 비율만 sweep. 위치를 표준화해야 sweep 결과가 재현 가능.
  - **결정 D22 메타**: 다음 CP들에서 비용보다 fidelity 우선 원칙 명시. 사용자가 시간 투자해서 진도 벌었으니 "제대로 하는" 방향으로 가는 것이 합의.
  - **다음 단계**: CP3.5 지시서 작성 (P1-1·P1-2·P2-3 fix + 모델 구조 보강 통합 묶음).
- **2026-04-25**: CP3.5 지시서 발주 (`docs/cp3.5_instruction.md`).
  - 범위: P1·P2 fix 3건 + A-3·B-2·C-2·D-1·E-1·E-2 결정 6건 한 묶음.
  - 예상 시간: 11~16시간 (CPU 위주, 풀 학습 미실행).
  - 권한 번들 사전 승인 + 종료 보고 12섹션 (모델별 핵심 구성요소 체크리스트 필수).
  - 신규 파일: `ai/models/revin.py`, `ai/models/blocks.py`, `ai/postprocess.py`.
  - 수정 파일: `ai/models/{patchtst,cnn_lstm,tide,common}.py`, `ai/loss.py`, `ai/train.py`, `ai/inference.py`.
  - 다음 단계: CP3.5 보고 통과 → CP4 (ticker embedding) 지시서.
- **2026-04-25**: CP3.5 완료 (`docs/cp3.5_report.md`).
  - **출력 정합성 fix 3건 모두 코드 통과**.
    - `apply_band_postprocess` 단일 함수 (line 보존, band sort) — 14줄.
    - `WidthPenaltyLoss` `F.relu` 적용 — 음수 폭 보상 차단.
    - `ForecastCompositeLoss`에 `band_mode` 분기 — `param` 모드일 때 cross loss 0 처리.
  - **PatchTST B-2**: RevIN + Channel Independence 둘 다 `forward` 경로에 반영. CI 분기에서 [B*C, L] reshape → patch_proj → encoder → 채널별 출력 → channel-wise 평균.
  - **CNN-LSTM C-2**: Conv 2층 + LayerNorm + 1x1 residual + LSTM 2층 + LSTM LayerNorm + AttentionPooling1D 정착.
  - **TiDE D-1**: feature_proj → enc(ResidualBlock × 4) → dec(ResidualBlock × 2) → temporal_decoder → lookback_skip(baseline_idx=0) 정착. 미래 covariate 의도적 생략.
  - **E-1**: 전 모델 `apply(init_weights)` 호출. trunc_normal std=0.02.
  - **E-2**: dropout 위치 표준화 적용.
  - **테스트 25 → 41 (16건 신규)**, 전부 green.
  - **검수에서 발견한 후속 검증 항목 (블로킹 아님, CP4·sweep 단계에서 ablation으로 확인)**:
    - **F-1 (RevIN denormalize 미사용)**: `forward`에서 `revin(norm)`만 호출, 출력 시 `denorm` 호출 없음. 사용자 보고서 메모: "출력 타깃이 수익률이라 forward에서는 normalize 중심". 의미: RevIN의 "reversible" 효과는 **input distribution stabilization**으로만 사용, output denorm은 안 함. 표준 PatchTST 사용법과는 다른 적용 방식. CP4 sweep에서 (a) denorm 추가 vs (b) 현재 방식 비교 ablation 권고.
    - **F-2 (Channel Independence 출력 channel-wise 평균)**: `[B*C, H]` 채널별 예측을 `view(B, C, H).mean(dim=1)`로 평균. 표준 PatchTST는 채널별로 그 채널 자신의 미래를 예측하는데, Lens는 단일 target(log_return at h)을 모든 채널이 예측 시도 → 평균. 합리적 ensemble 적용이지만 비표준. 대안: target 채널 (log_return idx=0) 하나만 사용, 또는 attention 기반 channel weighting. CP4 sweep에서 비교 가능.
    - **F-3 (init std 측정 허용 범위 15-25%)**: 보고서에 "15% 이상, 25% 이하 범위 통과". trunc_normal_ 표준은 std≈0.02에 1~2% 편차가 정상. Linear weight tensor가 작아서 발생한 sample noise일 가능성 — 검증할 가치 있지만 학습엔 무관.
  - **결정 D23**: F-1·F-2·F-3은 CP4 종료 후 sweep(A-1) 직전에 ablation 1회 권고. CP4 자체는 ticker embedding에 집중.
  - **CP4 준비 상태**: ticker embedding 인터페이스 준비 안 됨(`N`). CP4 지시서에서 처리.
  - **부수 작업 (보고서 12섹션)**: `backend/collector/jobs/compute_indicators.py` 시작일 규칙 정합성 복구. partial state에서 불필요한 전체 백필 방지.
  - **다음 단계**: CP4 (ticker embedding) 지시서 발주 가능 상태.
- **2026-04-25 — 모델별 논문 fidelity 의도적 갭 정리**:
  본 프로젝트가 어느 부분에서 논문 표준과 다르게 가는지·왜 그런지 명시. 발표·논문화 단계에서 "왜 이 선택?" 질문에 대비.

  ### PatchTST (목표 fidelity 80%)
  - **포함**: RevIN, Channel Independence, Patching, Transformer encoder, learned positional embedding, flatten head, line+band 분리 head.
  - **생략**: 논문은 단일 forecast head. Lens는 line head + band head 분리 (제품 철학상 보수적 점예측 + quantile 밴드 별도 학습).
  - **이유**: 제품 요구가 점 예측 정확도가 아닌 "보수적 진입선 + 밴드"라서 head 구조가 다른 게 맞음. 이 갭은 의도된 차이이지 부족함이 아님.

  ### CNN-LSTM (논문 표준 자체가 자유로운 패턴)
  - **포함**: Conv1d 2층 + LSTM 2층 + LayerNorm + residual + attention pooling + dropout 표준 위치 + line/band head.
  - **생략**: 단일 표준 논문이 없으므로 "갭"이라는 개념이 약함.
  - **이유**: 비교군 (baseline) 역할이라 표준 안정화 장치만 갖추면 충분.

  ### TiDE (목표 fidelity 70% — D-1 선택)
  - **포함**: Feature projection, Dense Encoder (ResNet-style ResidualBlock), Dense Decoder, Temporal Decoder, Lookback skip.
  - **생략 (의도)**:
    1. **미래 covariate 처리**: 논문은 미래 시점의 known covariate (예: 캘린더 변수, 알려진 이벤트)를 디코더에 추가 입력. Lens는 t+h 시점의 거시·재무·breadth가 모두 미지 → 미래 covariate가 사실상 없음. 이 부분 구현해도 입력이 비어 있어서 효과 없음.
    2. **Dropout schedule (점진적 감소)**: 논문에서 학습 단계별 dropout rate 변경 옵션 있음. Lens는 단일 rate로 단순화 (sweep 비용 절약).
    3. **Layer count tuning**: 논문은 데이터셋별 enc/dec layer 수 grid search. Lens는 enc=4, dec=2로 고정 후 sweep에서 1~2 step만 시도.
  - **이유**: TiDE의 핵심은 "MLP에 residual + encoder/decoder 분리". 미래 covariate 처리는 도메인 의존이고 우리 케이스엔 불필요. 골격만 갖추면 이름값 충분.

  ### 출력 헤드 (A-3 결정)
  - **포함**: 두 방식 다 구현 → ablation으로 비교.
    - 방식 1 (현재): 직접 q10/q90 출력 + 정렬·relu 후처리.
    - 방식 2 (신규): center + log_half_width 파라미터화 → 구조적 crossing 방지.
  - **이유**: 어느 쪽이 우월한지 데이터로 판단할 수 있게. 논문화 시 ablation 기여점.

  ### 후처리 정합성 (P1-1, P1-2 fix)
  - **단일 함수 `apply_band_postprocess`**로 val·test·inference 통일.
  - **line head는 sort 대상에서 제외** (line ≠ q50, β=2 비대칭으로 학습된 별도 head).
  - **이유**: 모델 선택 신호와 calibration 보고 신뢰성 확보. 학습 grad는 raw에 유지 (sort 통과하면 grad 끊김).

  ### Loss (P2-3 fix + 출력 파라미터화 채택 시 단순화)
  - 출력 방식 2 (center + log_half_width) 채택 시 cross penalty 불필요 → 손실 항 4개 → 3개로 단순화.
  - 방식 1 유지 시 width loss는 `mean(F.relu(upper - lower))` 로 음수 방지.
  - A-3 (둘 다 구현)이므로 두 loss 구성 모두 코드에 포함.
- **2026-04-25**: §9 PPT 대비 변경 이력 섹션 신설 (중간발표 → 기말발표 비교용). 5개 카테고리 표 (축소·추가·우선순위 변경·반려·메타 룰)로 누적 정리. 신규 결정 시 해당 표에 한 줄 append.

---

## 10. 2026-04-27 CP4~CP12 통합관리 업데이트

이 섹션은 CP3.5 이후 누락된 진행분을 기획자 관점에서 다시 묶은 기록이다. 세부 구현 보고서는 각 CP 보고서를 원본으로 두고, 본 문서는 다음 발주와 발표 대비를 위한 상위 판단을 남긴다.

### 10.1 CP 진행 요약

| CP | 핵심 내용 | 판단 |
|---|---|---|
| CP4 | PatchTST RevIN denorm, Channel Independence aggregate, ticker embedding, CI aggregate 비교 기반 | PatchTST fidelity 보강 완료 |
| CP5 | TiDE per-step head, CNN-LSTM 4-layer dilated `[1,2,4,8]`, RF=31 | horizon별 표현 손실과 CNN-LSTM receptive field 한계 보강 |
| CP6/CP6.5 | calendar future covariate 7채널, TiDE horizon decoder 추가 | CP3.5의 미래 covariate 생략 판단을 캘린더 변수 기준으로 정정 |
| CP7 | 학습 최적화 A~F, Optuna/W&B 도입 | sweep 기반 운영 준비 |
| CP8 | 첫 sweep에서 매 trial 데이터셋 재빌드와 Trial 0 장시간 hang 발견 | 긴 sweep 전에 데이터 경로 최적화 필요 확인 |
| CP9 | precomputed bundles, lazy SequenceDataset, batch evaluate/infer, Windows num_workers fallback, cache fingerprint, sweep `enable_compile=False` | sweep 병목 대부분 제거 |
| CP9.5 | WidthPenaltyLoss 부호 버그 제거, NaN/Inf check, MAPE에서 MAE/SMAPE로 전환 | coverage를 누르던 손실 버그 제거 |
| CP10 | 공통 평가판과 target plumbing 추가. raw, volatility-normalized, direction_label 지원. market_excess/rank는 자리만 확보 | 모델보다 평가 목적을 먼저 고정 |
| CP10.1 | 투자 지표는 항상 raw_future_returns 기준으로 고정. 비 raw target inference는 score_only로 제한 | target 변경이 투자 평가를 오염시키지 않도록 분리 |
| CP11 | CNN-LSTM direction auxiliary head 추가. forecast loss 유지, direction loss는 보조로만 추가 | 5티커는 개선, 50티커는 불확실. 채택 보류 |
| CP12 | finite gate, failed_nan status, amp dtype 분기, NaN diagnostics, inference/backtest status guard | CUDA/bf16 NaN이 결과 테이블과 체크포인트를 오염시키지 못하게 차단 |

### 10.2 새 결정 이력

| # | 결정 | 채택값 | 근거 |
|---|---|---|---|
| D25 | Lens 정체성 잠금 | 하방 보수적 line + AI band는 불변 | 프로젝트 차별점. 성능 개선보다 우선하는 제품 철학 |
| D26 | 투자 지표 기준 | 어떤 target으로 학습해도 투자 지표는 raw realized return 기준 | direction_label, volatility_normalized_return 실험 시 평가 의미 보존 |
| D27 | direction head 지위 | 본체가 아닌 보조 head. `lambda_direction=0.1` 기본 | direction_accuracy 단독 최적화 금지. band/line guardrail 유지 |
| D28 | NaN run 저장 정책 | `model_runs.status="failed_nan"` 메타만 저장, 결과 테이블과 checkpoint 차단 | 디버그 재현성과 결과 신뢰성 동시 확보 |
| D29 | mixed precision 정책 | 기본 bf16 유지, `--amp-dtype off/fp16` 분기 제공 | 정상 경로 속도 유지, NaN 원인 분리 가능 |
| D30 | CP13 진입 조건 | CP12 50티커 bf16 baseline PASS 전 direction 확대 금지 | precision 문제가 남은 상태에서 모델 성능 판단 금지 |

### 10.3 현재 블로커와 보류 항목

| 상태 | 항목 | 처리 |
|---|---|---|
| PENDING | CP12 매트릭스 6개 실행 | 사용자 GPU에서 실행 후 `docs/cp12_report.md` 10번 표에 PASS/NAN 기록 |
| PENDING | DB 마이그레이션 | `C:\Users\user\lens\.venv\Scripts\python.exe -m backend.db.scripts.ensure_runtime_schema` 1회 실행 |
| OPEN | CNN-LSTM + CUDA + bf16 NaN 원인 | CP12 매트릭스 결과로 amp, device, direction head, ticker 규모 분리 |
| OPEN | `market_excess_return` 계산 경로 | 다음 target CP에서 SPY/시장 수익률 조인 경로 추가 |
| OPEN | `rank_target` 계산 경로 | 날짜 단면별 랭킹 라벨 생성 필요 |
| LEGACY | `lambda_width` | CLI 호환 인자로만 유지. 현재 손실 계산에는 사용하지 않음 |
| RESOLVED | W&B mape 키 잔여 | CP12 grep 기준 코드 경로 0건. 과거 sweep artifact의 `test/mape`는 레거시 |

### 10.4 다음 발주 분기

CP12 매트릭스 결과를 받은 뒤 아래 중 하나로 발주한다.

| 조건 | 다음 CP |
|---|---|
| `[3]` PASS, `[5]` 또는 `[6]` NaN | CP12.5: LSTM/head 부분 fp32 강제 옵션 도입 |
| 5티커와 50티커 bf16 모두 PASS | CP13: `lambda_direction` sweep + PatchTST/TiDE direction head 미러 |
| direction head만 NaN | CP11 direction loss/logit 경로 분리 CP |
| baseline도 NaN | direction/rank 실험 중단. 데이터, precision, CNN-LSTM forward 경로부터 분리 |

### 10.5 PPT 대비 변경 이력 추가

| 구분 | 변경 | 발표 시 메시지 |
|---|---|---|
| 범위 추가 | 공통 투자 평가판 추가: IC, top-k spread, fee-adjusted return/sharpe/turnover | 단순 예측 오차가 아니라 투자 목적 지표로 모델을 검증 |
| 우선순위 변경 | 긴 LR sweep보다 target/loss/evaluation 재설계 우선 | 성능 정체 원인을 모델 크기보다 목적 함수 불일치로 판단 |
| 정체성 고정 | line은 하방 보수적, band는 AI 리스크 지표 | Lens의 차별점은 방향 맞히기가 아니라 보수적 판단선과 calibrated band |
| 안전성 보강 | NaN run을 failed_nan으로 추적하고 결과 저장 차단 | 잘못된 실험 결과가 DB, W&B, checkpoint에 섞이지 않도록 통제 |

### 10.6 CHANGELOG 추가

- **2026-04-27**: CP4~CP12 누락분 통합관리 업데이트. 결정 D25~D30 추가. CP12 매트릭스 전까지 direction head 확장 보류를 명시. 세 문서(`model_architecture.md`, `project_journal.md`, `training_hyperparameters.md`)를 이후 CP 발주 전후로 함께 갱신하는 운영 원칙 확정.

### 10.7 CP12 매트릭스 실행 결과와 CP12.5 확정

사용자가 DB 마이그레이션을 1회 실행했고 `ensure_runtime_schema` 성공을 확인했다. 이후 CP12 보고서 8번 섹션의 6개 매트릭스를 순서대로 실행했다.

| # | 조건 | 결과 | first NaN |
|---|---|---|---|
| 1 | 5티커 CPU fp32 baseline | PASS | - |
| 2 | 5티커 CUDA amp off baseline | PASS | - |
| 3 | 5티커 CUDA bf16 baseline | NAN | val / `avg_band_width` |
| 4 | 5티커 CUDA amp off direction_head | PASS | - |
| 5 | 5티커 CUDA bf16 direction_head | NAN | val / `avg_band_width` |
| 6 | 50티커 CUDA bf16 baseline | NAN | val / `avg_band_width` |

결론:

- direction head만 NaN이 아니다.
- CNN-LSTM baseline도 CUDA bf16에서 NaN이 난다.
- CPU fp32와 CUDA amp off는 통과하므로 데이터 자체보다는 bf16 autocast 경로가 유력하다.
- `[2]`, `[4]`는 결과 JSON은 정상 출력됐지만 종료코드가 1이었다. PASS로 보되, CP12.5에서 별도 확인이 필요하다.

확정 발주:

- **CP12.5**: CNN-LSTM CUDA bf16 안정화.
- CP13 direction/rank 성능 실험은 보류.
- 목표는 `--fp32-modules` 같은 부분 fp32 강제 옵션과 val metric NaN 진단 강화다.

### 10.8 CHANGELOG 추가

- **2026-04-27**: CP12 매트릭스 결과 반영. `[3]`, `[5]`, `[6]`이 val `avg_band_width` NaN으로 실패해 CP13 진입을 차단하고 CP12.5 발주로 확정. `[2]`, `[4]` 종료코드 1 현상을 별도 리스크로 기록.

### 10.9 CP12.5 결과와 CP12.6 확정

CP12.5에서 `--fp32-modules` 옵션과 aggregate tensor finite diagnostics를 적용한 뒤 사용자 GPU에서 매트릭스를 실행했다.

| 케이스 | 조건 | 결과 | 메모 |
|---|---|---|---|
| A | bf16 + `fp32_modules=none` | NAN | val aggregate `tensor:raw.line`; raw/post 예측 94.0% finite, target 100% finite |
| B | bf16 + `fp32_modules=lstm` | PASS | 지표 정상, 종료코드 `-1073740791` |
| C | bf16 + `fp32_modules=heads` | NAN | heads만 fp32로는 해결 불가 |
| D | bf16 + `fp32_modules=lstm,heads` | PASS | baseline 지표 정상, 종료코드 `-1073740791` |
| E | bf16 direction head + `fp32_modules=lstm,heads` | PASS | direction 경로 지표 정상, 종료코드 `-1073740791` |
| F | 50티커 bf16 baseline + `fp32_modules=lstm,heads` | PASS | 지표 정상, 종료코드 `-1073740791` |
| G | 5티커 CUDA amp off baseline 종료코드 재확인 | 실패 | 출력 정상, 종료코드 `-1073740791` |

판단:

- NaN 원인은 LSTM bf16 autocast 경로로 보는 것이 맞다.
- `lstm,heads` fp32 강제는 지표상 D/E/F를 살렸다.
- 그러나 성공 run의 CUDA 프로세스 종료코드가 계속 비정상이므로 CP12.5 closure는 불가하다.
- 이 상태에서 sweep을 돌리면 출력은 좋아 보여도 orchestrator가 실패 trial로 처리하거나 결과 수집이 흔들릴 수 있다.

확정 발주:

- **CP12.6**: 성공 run 종료 시점의 Windows/CUDA 비정상 종료 `-1073740791` 분리.
- NaN 안정화 후보값은 임시로 `--fp32-modules lstm,heads`.
- CP13은 CP12.6에서 CUDA 성공 run 종료코드 0을 확인한 뒤 발주한다.

### 10.10 CHANGELOG 추가

- **2026-04-27**: CP12.5 결과 반영. `lstm` fp32 강제로 CNN-LSTM bf16 NaN 원인을 LSTM 구간으로 좁혔지만, CUDA 성공 run 종료코드 `-1073740791` 때문에 closure를 보류하고 CP12.6으로 분리.

### 10.11 CP12.6 closure와 범위 재정의

CP12.6 처리 결과, Windows CUDA 환경에서 cuDNN LSTM 출력이 Linear head로 이어진 뒤 프로세스 종료 시 네이티브 크래시가 나는 문제로 분리됐다. `ai/models/cnn_lstm.py`의 LSTM 실행 구간에서 CUDA일 때 cuDNN을 비활성화했고, hard exit 없이도 CUDA 성공 run이 exit code 0으로 종료됐다.

검증 결과:

| 케이스 | 결과 |
|---|---|
| 5티커 CPU amp off baseline | PASS, exit code 0 |
| 5티커 CUDA amp off baseline | PASS, exit code 0 |
| 5티커 CUDA amp off + explicit cleanup | PASS, exit code 0 |
| 5티커 CUDA bf16 + `--fp32-modules lstm,heads` | PASS, exit code 0 |
| 50티커 CUDA bf16 + `--fp32-modules lstm,heads` | PASS, exit code 0 |

CP12.6은 closure한다. 다만 프로젝트 운영 판단은 바뀐다.

새 범위:

- **Phase 1**: PatchTST 단일 주력 모델로 제품 루프와 데모 완성.
- **Phase 1.5**: TiDE, CNN-LSTM 재도입. 현재 수정은 이때를 위한 안정성 확보로 보관.
- **Phase 2**: ranker, ensemble, 확장 universe, 고급 research.

새 결정:

| # | 결정 | 채택값 | 근거 |
|---|---|---|---|
| D31 | Phase 1 모델 범위 | PatchTST 단일 주력 | 세 모델 동시 개선이 checkpoint-수정-버그 루프를 만들었음 |
| D32 | TiDE/CNN-LSTM 지위 | Phase 1.5 backlog | 구조 폐기 아님. 제품 루프 완성 뒤 재도입 |
| D33 | CP 유형 분리 | Product CP와 Research CP 분리 | 연구 버그와 제품 연결 버그가 한 CP에서 섞이는 문제 차단 |
| D34 | PatchTST 개선 순서 | target/eval/loss 먼저, architecture sweep 나중 | 지금 성능 정체는 모델 크기보다 목적 함수 불일치 가능성이 큼 |

### 10.12 PatchTST Solo Track 로드맵

| CP | 성격 | 목표 |
|---|---|---|
| CP13 | Scope reset | 문서와 CLI 기본값을 PatchTST Solo Track에 맞춤. TiDE/CNN-LSTM은 freeze |
| CP14 | Research | PatchTST supervised baseline 재확정. 현재 best checkpoint와 평가 리포트 고정 |
| CP15 | Research | `raw_future_return`, `volatility_normalized_return`, `market_excess_return` 비교 |
| CP16 | Product | 예측/평가/백테스트 조회 API 정리 |
| CP17 | Product | 프론트 데모 화면 구축: 예측선, 밴드, coverage, backtest, run status |
| CP18 | Research | PatchTST patch_len/stride/seq_len/ci_aggregate ablation |
| CP19 | Product | 밴드 calibration/백테스트 리포트 화면 |
| CP20 | Release freeze | 발표용 end-to-end demo 고정 |

### 10.13 CHANGELOG 추가

- **2026-04-27**: CP12.6 closure. cuDNN LSTM 종료 크래시를 CUDA LSTM 구간 cuDNN off로 해결. 동시에 Phase 1을 PatchTST 단일 주력 모델로 재정의하고 TiDE/CNN-LSTM은 Phase 1.5로 이동.

### 10.14 CP13 closure

CP13 산출물 `docs/cp13_patchtst_solo_plan.md` 작성 완료. 이 문서를 PatchTST Solo Track의 시작점으로 승인한다.

핵심 정리:

- TiDE/CNN-LSTM은 Phase 1에서 freeze.
- Phase 1.5 backlog: CNN-LSTM direction head 재개, TiDE 미래 공변량 실험, rank/direction target 확장, multi-model 비교표 확대.
- PatchTST 감사 결과:
  - RevIN, Channel Independence, learned positional embedding, dropout, line/band head는 현재 구현에 존재.
  - `ci_aggregate=target/mean/attention`, `ci_target_fast`, ticker embedding, line+band head는 Lens 목적에 따른 의도적 차이.
  - `ci_aggregate=target` 기본값은 full channel 정보 활용보다 target 채널 편향이 강할 수 있으므로 추후 ablation 필요.
- 바로잡은 점:
  - `patch_len`, `stride`, `d_model`, `n_layers`, `n_heads`는 PatchTST 모델 init에는 있으나 `ai.train` CLI와 `ai.sweep` 축에는 아직 직접 노출되어 있지 않다.
  - 따라서 CP14 Stage B/E 전에 CLI 또는 전용 sweep entry 정리가 필요하다.

CP14 진입 조건:

- 긴 LR sweep 금지.
- direction/rank head 금지.
- raw target baseline과 volatility-normalized target baseline을 먼저 비교.
- `market_excess_return`은 아직 계산 경로가 없으므로 CP14 실행 대상에서 제외.

### 10.15 CHANGELOG 추가

- **2026-04-27**: CP13 closure. PatchTST Solo Track 계획 승인. CP14는 target baseline 2종 비교와 PatchTST 전용 실험 인자 노출 결정으로 발주 예정.

### 10.16 Product Demo Track 분리

모델 연구 난이도가 높아지고 checkpoint-수정-버그 루프가 길어진 문제를 줄이기 위해 Product CP를 Research CP와 분리한다.

신규 문서:

- `docs/cp_product_demo_plan.md`

결정:

| # | 결정 | 채택값 | 근거 |
|---|---|---|---|
| D35 | CP 트랙 분리 | Research CP / Product CP 분리 | 모델 실험과 화면/API 버그를 한 CP에서 섞지 않기 위함 |
| D36 | Phase 1 프론트 모델 범위 | PatchTST만 노출 | TiDE/CNN-LSTM은 Phase 1.5 backlog |
| D37 | 프론트 성격 | 운영형 리서치 콘솔 | 현재 hero 중심 화면은 데모에는 가능하지만 모델 검수 효율이 낮음 |
| D38 | 학습 실행 UI | Phase 1에서는 만들지 않음 | read-only 조회와 해석 화면부터 완성 |

Product Track 목표:

- 첫 화면은 주식 보기 완성형 화면으로 구성.
- 가격 + 보수적 예측선 + AI band overlay.
- 1M은 AI 밴드가 없어도 가격 화면으로 자연스럽게 표시.
- 캔들/라인 차트 전환과 보조지표 layer toggle 제공.
- coverage, avg_band_width, IC, top-k, fee-adjusted 지표 표시.
- backtest return/sharpe/turnover는 별도 백테스트 화면에서 표시.
- run status와 config는 별도 모델 학습 화면에서 표시.
- failed_nan run은 결과에 섞지 않고 상태로만 표시.

CP 분리:

| CP | 트랙 | 목표 |
|---|---|---|
| CP14-R | Research | PatchTST target baseline 2종 비교 |
| CP14-P | Product | AI run/evaluation/backtest 조회 API 최소 구현 |
| CP15-R | Research | PatchTST patch geometry 실험 |
| CP15-P | Product | 주식 보기 화면 레이아웃 전환 |
| CP16-P | Product | line/band chart overlay |
| CP17-P | Product | 백테스트/평가 지표 패널 연결 |

### 10.17 CHANGELOG 추가

- **2026-04-27**: Product Demo Track 분리. 모델 연구와 백/프론트 완성 CP를 별도로 발주하기로 결정. 현재 프론트는 hero 중심 대시보드에서 PatchTST 리서치 콘솔로 전환 예정.
- **2026-04-27**: Product Demo Track 방향 수정. 첫 화면은 리서치 콘솔이 아니라 TradingView식 차트 조작력과 토스증권식 간결함을 섞은 주식 보기 화면으로 정의. 백테스트와 모델 학습은 왼쪽 내비게이션의 별도 화면으로 분리.

### 10.18 CP14-P 검수

CP14-P는 read-only API 범위를 지켰다. 모델, 학습 코드, DB schema 변경 없이 AI run/evaluation/backtest 조회 API를 추가했고 backend unittest가 통과했다.

추가 API:

- `GET /api/v1/ai/runs`
- `GET /api/v1/ai/runs/{run_id}`
- `GET /api/v1/ai/runs/{run_id}/evaluations`
- `GET /api/v1/ai/runs/{run_id}/backtests`
- `GET /api/v1/stocks/{ticker}/predictions/latest?run_id=...`

검수 결과:

- `/api/v1/ai` 라우터 등록 확인.
- `failed_nan` run은 기본 목록에서 제외되고, 명시 요청 시 조회 가능.
- `predictions/latest?run_id=...`는 `status != completed` run을 409로 거부.
- `include_config=true`를 제외하면 원본 config를 직접 노출하지 않음.
- `NaN`/무한대 값은 JSON 응답에서 `null`로 변환.

주의할 제한:

1. `prediction_evaluations` 테이블에는 `spearman_ic`, `top_k_*`, `fee_adjusted_*` 컬럼이 없다. 해당 aggregate 지표는 현재 run detail의 `val_metrics`/`test_metrics`에서 보는 것이 맞다. `/evaluations`는 ticker/asof 단위 지표 중심으로 해석한다.
2. `predictions` 테이블의 unique key가 `(ticker, model_name, timeframe, horizon, asof_date)`라서 과거 run별 prediction 전체 이력을 보장하지 않는다. `run_id` 조회는 해당 run의 prediction row가 아직 남아 있는 경우에만 안정적이다. 향후 run별 prediction 비교 화면이 필요하면 DB schema 재설계를 별도 CP로 분리한다.

판정:

- CP14-P 최소 read-only API는 조건부 closure 가능.
- 위 2개 제한은 CP15-P/CP16-P 프론트 연결 전에 사용자에게 보이는 의미를 정리해야 한다.

### 10.19 학습 실행 ETA 운영 원칙

모델 학습은 사용자가 직접 GPU에서 실행하므로, 이후 Research CP 지시서에는 반드시 예상 소요 시간을 함께 적는다. 하염없이 기다리는 상태를 줄이기 위해 다음 항목을 기본 포함한다.

| 항목 | 원칙 |
|---|---|
| 사전 ETA | 실행 전 `예상 1 epoch 시간`, `총 예상 시간`, `중간 확인 시점`을 적는다 |
| 1 epoch 재추정 | 첫 epoch가 끝나면 실제 시간을 기준으로 남은 시간을 다시 계산한다 |
| 중단 기준 | NaN, OOM, GPU 사용률 저하, 지표 악화가 보이면 몇 epoch에서 멈출지 명시한다 |
| 자원 기록 | batch size, VRAM 사용량, GPU util, epoch time을 보고서에 남긴다 |
| batch 증량 | 입력 NaN 정합을 먼저 닫은 뒤 `64 → 128 → 256` 순서로 처리량을 확인한다 |

초기 ETA는 보수적으로 잡는다. 실제 epoch 시간이 나오면 그 값이 더 중요하다.

### 10.20 CHANGELOG 추가

- **2026-04-27**: Research CP 실행 지시서에 학습 시간 ETA, 1 epoch 재추정, 중단 기준, VRAM/batch 기록을 포함하기로 결정.

### 10.21 CP15-P closure와 CP16-P 진입 기준

CP15-P에서 첫 화면을 hero 소개가 아니라 주식 보기 화면으로 전환했고, 왼쪽 내비게이션으로 `주식 보기 / 백테스트 / 모델 학습` 3화면을 분리했다.

CP15-P 검수 기준:

- 1M은 가격 조회만 수행하고 AI layer를 비활성 처리한다.
- 모델 선택과 target type은 주식 보기 화면에서 숨긴다.
- 백테스트와 모델 학습 화면은 read-only 구조를 유지한다.
- 차트 overlay는 props 자리만 열렸고 실제 렌더링은 CP16-P로 넘긴다.

CP16-P 진입 기준:

- 주식 보기와 백테스트 화면의 사용자 노출 문구는 한국어로 정리한다.
- 모델 학습 화면은 리서치 콘솔이므로 `completed`, `failed_nan`, `run_id` 같은 원문 상태값은 괄호 병기까지 허용한다.
- AI 밴드와 보수적 예측선 overlay를 실제 차트에 렌더링한다.
- overlay 데이터가 없거나 1M인 경우 차트가 깨지지 않아야 한다.

### 10.22 학습 진행 확인 대안

Codex 내부 터미널에서 긴 학습을 실행하면 사용자가 진행 상황을 바로 보기 어렵다. 큰 학습은 W&B를 켜면 되지만, 짧은 검증 run에도 최소 진행률 확인 수단이 필요하다.

다음 Research CP 후보에 `run progress sidecar`를 추가한다.

| 방식 | 용도 |
|---|---|
| stdout | 지금처럼 즉시 확인 가능한 환경용 |
| W&B | 긴 sweep과 큰 학습용 |
| `ai/runs/progress/{run_id}.jsonl` | W&B를 끈 짧은 학습에서도 epoch 진행률 확인 |
| DB status | `completed`/`failed_nan`/향후 `running` 상태 추적 |

1차 구현은 파일 기반 jsonl을 우선한다. DB에 `running` 상태를 추가하는 것은 schema 변경이므로 별도 CP로 판단한다.

### 10.23 CP16-P 검수

CP16-P에서 주식 보기 차트에 실제 AI overlay를 연결했다. 프론트는 최신 완료 PatchTST run을 먼저 고른 뒤 `run_id` 기준으로 prediction을 조회한다.

확인 결과:

- `forecast_dates`, `line_series`, `conservative_series`, `upper_band_series`, `lower_band_series`, `band_quantile_low`, `band_quantile_high`는 백엔드 응답 스키마와 일치한다.
- 1D/1W는 completed PatchTST run 기준으로 prediction을 조회한다.
- 1M은 가격 전용으로 유지하고 prediction/evaluation API를 호출하지 않는다.
- AI 밴드는 상단/하단 점선으로 표시하고, 보수적 예측선은 실선으로 표시한다.
- band fill은 안정 구현 부담 때문에 보류했다. 현재 단계에서는 가격 스케일 안정성을 우선한다.
- 주식 보기와 백테스트 주요 라벨은 한국어로 정리됐다.
- 승인된 환경에서 `frontend`의 `npm run build`가 통과했다.

남은 제품 cleanup:

- 백테스트 화면은 latest completed run을 고를 때 현재 선택한 timeframe을 함께 넘겨야 한다. 지금은 전체 latest run을 고른 뒤 backtest timeframe만 필터링하므로, 최신 run과 선택 timeframe이 다르면 빈 결과가 나올 수 있다.
- 모델 학습 화면의 일부 리서치 라벨은 원문 병기 정책에 맞춰 천천히 정리한다.

판정:

- CP16-P는 조건부 closure 가능.
- 다음 제품 CP는 백테스트 timeframe run 선택 cleanup과 평가/백테스트 지표 패널 고도화를 우선한다.

### 10.24 CP17-P 검수

CP17-P에서 백테스트와 모델 학습 화면의 평가 지표 표시가 강화됐다. 핵심 cleanup이었던 latest completed run 조회의 timeframe 불일치 문제도 수정됐다.

확인 결과:

- `BacktestView`의 `fetchAiRuns` 호출에 `timeframe: nextTimeframe`이 추가됐다.
- 백테스트 조회도 같은 `nextTimeframe`으로 이어진다.
- 수수료 반영 지표와 수수료 전 gross 지표가 분리 표시된다.
- 백테스트 화면에 latest completed PatchTST run의 `val_metrics`/`test_metrics` 품질 패널이 추가됐다.
- 모델 학습 화면의 상태 라벨은 `완료(completed)`, `NaN 실패(failed_nan)`로 정리됐다.
- 모델 학습 화면에도 `val_metrics`/`test_metrics` 품질 패널이 추가됐다.
- 주식 보기 화면에는 target type 관련 용어를 새로 노출하지 않았다.
- 승인된 환경에서 `frontend`의 `npm run build`가 통과했다.

남은 제품 TODO:

- 백테스트 `equity curve`와 `drawdown` series가 저장되면 placeholder를 실제 차트로 교체한다.
- metric별 단위가 더 명확해지면 퍼센트/소수 표시 규칙을 지표별로 세분화한다.
- 모델 학습 화면의 연구 라벨은 제품 polish 단계에서 원문 병기/한글명을 더 정리한다.

판정:

- CP17-P는 closure 가능.
- 제품 트랙은 이제 `주식 보기 overlay → 백테스트 지표 → 모델 학습 run 품질`의 기본 조회 루프를 갖췄다.

### 10.25 CP18-P 검수

CP18-P에서 데모 실행 절차와 발표 동선을 고정했다. `scripts/start_demo.ps1`이 추가됐고, 수동 실행 절차도 `docs/cp18_product_demo_report.md`에 정리됐다.

확인 결과:

- `scripts/start_demo.ps1`은 별도 PowerShell 창 2개로 backend/frontend dev server를 실행한다.
- 수동 실행 절차가 백엔드/프론트 각각 문서화됐다.
- `GET http://127.0.0.1:8000/api/v1/health/live` 응답 200을 확인했다.
- `GET http://127.0.0.1:3000` 응답 200을 확인했다.
- 보고서에 `주식 보기 → 1M 전환 → 백테스트 → 모델 학습` 발표 동선이 정리됐다.
- 백엔드 연결 실패 문구, metric card 줄바꿈, chart legend 간격, 모바일 padding polish가 반영됐다.
- 주식 보기 화면에는 target type 관련 용어를 새로 노출하지 않았다.

현재 데모 리스크:

- 로컬 DB 기준으로 AAPL prediction/backtest/evaluation이 비어 있어 실제 AI overlay는 보이지 않을 수 있다.
- 1W/1M completed PatchTST run이 없어 백테스트/평가 화면은 empty state 중심이다.
- 인앱 브라우저 자동 navigation은 타임아웃이 발생했으나, 셸 HTTP 응답은 3000/8000 모두 200이다. 최종 시각 QA는 사용자의 일반 브라우저에서 수행한다.

판정:

- CP18-P는 closure 가능.
- 다음 제품 단계는 fake data 없이 실제 inference/backtest 산출물을 생성해 AI overlay가 보이는 데모 run을 확보하는 것이다.

#### CP18-P 추가 안정화

CP18 추가 지시로 데모 산출물 부족 상태를 숨기지 않고 화면이 깨지지 않게 보강했다.

- `scripts/check_demo_readiness.ps1`이 추가됐다.
- `/api/v1/stocks?search=AAPL` 503은 전체 화면 오류가 아니라 작은 안내 문구로 처리한다.
- prediction 404와 일반 AI prediction API 오류를 구분한다.
- prediction row가 없어도 가격 차트는 유지한다.
- readiness 결과는 `health`, `1D prices`, `1M prices`, `stock search`, `completed run`, `prediction`, `backtest`, `evaluation`을 확인한다.
- 현재 결과는 health/price/completed run은 OK, stock search는 FAIL, prediction/backtest/evaluation은 WARN이다.

이로써 CP18은 데모 실행 안정화와 산출물 부족의 명시적 표시까지 포함해 closure 처리한다.

### 10.26 PatchTST 입력 정합 복구와 학습 운영 재설계

CP14-R 이후 막혔던 PatchTST 입력 feature NaN 문제는 복구됐다.

확인 결과:

- 재무 6개 컬럼 `revenue`, `net_income`, `equity`, `eps`, `roe`, `debt_ratio`의 NaN은 학습 tensor 진입 전 0.0으로 impute된다.
- `has_fundamentals`는 원본 재무 데이터 존재 여부 기준으로 0.0/1.0이 들어간다.
- `raw_future_return` PatchTST 1D 전체 학습은 NaN 없이 완료됐다.
- 다만 실행 조건이 사실상 `473 ticker × seq_len 252 × batch 64 × 5 epoch` 전체 학습이어서 epoch당 약 800~930초, 전체 약 80분이 걸렸다.
- `volatility_normalized_return` 전체 학습은 시간 문제로 중단됐고 비교 판단은 아직 불가하다.

운영 판단:

- 이제 병목은 입력 정합이 아니라 학습 운영이다.
- 다음 target/geometry 실험은 전체 473 ticker 5 epoch로 바로 들어가지 않는다.
- 모든 새 조건은 `--limit-tickers 50`, `--epochs 1` smoke를 먼저 통과해야 한다.
- smoke 통과 후 batch size ladder `64 → 128 → 256`으로 처리량과 VRAM을 확인한다.
- 그 다음에만 473 ticker 또는 5 epoch 확대를 허용한다.

다음 즉시 실행 후보:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model patchtst --timeframe 1D --seq-len 252 --epochs 1 --batch-size 64 --device cuda --no-wandb --no-compile --ci-aggregate target --line-target-type volatility_normalized_return --band-target-type volatility_normalized_return --limit-tickers 50
```

이 명령의 목적은 성능 판단이 아니라 volatility target 경로가 finite하게 도는지 확인하는 것이다.

### 10.27 CP19 데모용 실제 AI 산출물 확보

CP19에서 fake data 없이 실제 저장 산출물이 있는 데모 후보를 확보했다.

확보 산출물:

| 항목 | 값 |
|---|---|
| demo run | `patchtst-1D-fc096a026a1e` |
| demo ticker | `AAPL` |
| checkpoint | `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-fc096a026a1e.pt` |
| prediction row | 존재 |
| `forecast_dates` | 길이 5 |
| `line_series` | 길이 5 |
| `upper_band_series` | 길이 5 |
| `lower_band_series` | 길이 5 |
| evaluation row | 존재 |
| backtest row | `ai.backtest --save`로 실제 생성 완료 |

프론트 동작 변경:

- 주식 보기 화면은 최신 completed run만 고집하지 않는다.
- 최근 completed PatchTST run들을 순서대로 확인해, 실제 prediction row가 있는 run을 사용한다.
- 현재 로컬 DB에서는 최신 run `patchtst-1D-94d61c4e84d3`에 AAPL prediction이 없으므로, 다음 후보 `patchtst-1D-fc096a026a1e`를 데모 run으로 사용한다.
- 이 변경으로 주식 보기 화면에서 실제 AI 밴드와 보수적 예측선 overlay 표시가 가능해졌다.

남은 리스크:

- `patchtst-1D-94d61c4e84d3` checkpoint는 현재 PatchTST 코드와 state_dict shape가 맞지 않아 inference 저장에 실패했다.
- 모델 구조 변경 금지 원칙에 따라 호환 로더를 만들지 않았다.
- `stock_info` 기반 티커 검색 503은 여전히 남아 있으나, 직접 티커 입력과 가격/AI overlay 데모에는 영향을 주지 않도록 fallback 처리되어 있다.

판정:

- CP19는 closure 가능.
- 제품 트랙은 이제 empty state 데모가 아니라 실제 AI overlay가 보이는 데모 run을 갖췄다.
### 10.28 CP21-R PatchTST geometry smoke

PatchTST 전용 실험 인자를 `ai.train` CLI에 노출했다. 추가 인자는 `--patch-len`, `--patch-stride`, `--patchtst-d-model`, `--patchtst-n-heads`, `--patchtst-n-layers`이며, PatchTST가 아닐 때는 모델 생성에 전달하지 않는다.

50티커, 3epoch, raw target, batch 256 조건에서 patch geometry 5개를 비교했다.

| 조건 | patch_len | stride | n_patches | coverage | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 8 | 30 | 0.999954 | 0.661335 | 0.065957 | 0.069970 | 0.005070 | 6.002602 |
| short | 8 | 4 | 62 | 1.000000 | 1.631322 | 0.162684 | 0.047372 | 0.002970 | 0.488734 |
| dense | 16 | 4 | 60 | 1.000000 | 1.838474 | 0.183312 | -0.015071 | -0.003404 | -0.972538 |
| long | 32 | 16 | 14 | 0.988010 | 0.352508 | 0.035526 | 0.030214 | 0.003971 | 2.140137 |
| overlap | 32 | 8 | 28 | 1.000000 | 0.943122 | 0.094038 | 0.062391 | 0.004038 | 2.314449 |

결론:

- 모든 조건은 NaN/OOM 없이 통과했다.
- coverage는 전부 0.98 이상으로 경고 구간이다.
- baseline `patch_len=16`, `stride=8`이 투자 지표 기준으로 가장 낫다.
- `patch_len=32`, `stride=16`은 가장 빠르고 밴드 폭이 좁지만 IC와 spread가 baseline보다 낮다.
- Patch geometry만으로는 밴드 과보수와 투자 지표를 동시에 개선하지 못했다.

다음 방향:

- full run으로 바로 가지 않는다.
- baseline geometry를 유지하고, 다음 CP에서는 band calibration을 먼저 본다.
### 10.29 CP22-R PatchTST band calibration smoke

CP21-R에서 geometry 변경만으로는 밴드 과보수와 투자 지표를 동시에 개선하지 못했다. 따라서 baseline geometry `patch_len=16`, `stride=8`을 유지하고 q 범위, `lambda_band`, `band_mode`만 짧게 흔들었다.

결과:

| 조건 | q_low | q_high | lambda_band | band_mode | coverage | avg_band_width | band_loss | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.10 | 0.90 | 1.0 | direct | 0.999954 | 0.661335 | 0.065957 | 0.069970 | 0.005070 | 6.002602 |
| q15 | 0.15 | 0.85 | 1.0 | direct | 0.998004 | 0.465148 | 0.069646 | 0.069701 | 0.004849 | 5.183134 |
| q20 | 0.20 | 0.80 | 1.0 | direct | 0.987256 | 0.349639 | 0.070110 | 0.070359 | 0.005125 | 6.504665 |
| q15-b2 | 0.15 | 0.85 | 2.0 | direct | 0.995148 | 0.401907 | 0.060262 | 0.069561 | 0.004955 | 5.584883 |
| q20-b2 | 0.20 | 0.80 | 2.0 | direct | 0.972607 | 0.303344 | 0.061271 | 0.069890 | 0.005254 | 6.956047 |
| param | 0.15 | 0.85 | 2.0 | param | 0.993976 | 0.384076 | 0.057617 | 0.051032 | 0.002202 | 0.201084 |

판단:

- `q20-b2`가 밴드 폭을 가장 유의미하게 줄이면서 IC, spread, fee-adjusted return을 유지하거나 개선했다.
- 다만 coverage 0.9726은 아직 목표 구간 0.75~0.90에 닿지 못했다.
- `param`은 band_loss와 overprediction은 좋아졌지만 투자 지표가 크게 약해져 탈락이다.
- 다음 smoke는 `q20-b2`를 기준점으로 두고 q 범위를 더 좁히는 쪽이 맞다.
### 10.30 CP23-R PatchTST narrow band calibration

CP22-R의 `q20-b2`를 기준으로 q 범위를 더 좁혀 coverage와 band width를 낮추는 smoke를 수행했다.

결과:

| 조건 | q_low | q_high | lambda_band | coverage | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q20-b2 | 0.20 | 0.80 | 2.0 | 0.972607 | 0.303344 | 0.069890 | 0.005254 | 6.956047 |
| q25-b2 | 0.25 | 0.75 | 2.0 | 0.936846 | 0.250114 | 0.069396 | 0.005063 | 6.115722 |
| q30-b2 | 0.30 | 0.70 | 2.0 | 0.890662 | 0.214104 | 0.069014 | 0.004678 | 4.653222 |
| q25-b3 | 0.25 | 0.75 | 3.0 | 0.948210 | 0.263623 | 0.066624 | 0.004731 | 4.723813 |
| q30-b3 | 0.30 | 0.70 | 3.0 | 0.921479 | 0.237363 | 0.064092 | 0.003857 | 2.291304 |
| q35-b2 | 0.35 | 0.65 | 2.0 | 0.837282 | 0.186324 | 0.068284 | 0.004552 | 4.168714 |

판단:

- `q30-b2`는 coverage 0.8907로 목표 구간 상단에 들어왔고, IC/spread/return도 양수로 유지됐다.
- `q25-b2`는 coverage가 0.9368로 아직 높지만 투자 지표 보존이 가장 좋다.
- `q35-b2`는 coverage는 좋지만 upper breach가 11.3%로 커져 보수적 예측 정체성에 리스크가 있다.
- lambda_band 3.0 계열은 overprediction 지표는 좋아졌지만 투자 지표가 약해져 우선순위를 낮춘다.

다음 후보:

- 1차 후보: `q30-b2`
- 보수 후보: `q25-b2`
- 다음은 full run이 아니라 100티커 3epoch 또는 1W smoke로 안정성을 확인한다.

### 10.31 CP24-R Bollinger baseline 비교

CP24-R에서는 CP23-R 체크포인트를 새로 학습하지 않고 재사용했다. Bollinger 기준선은 가격 밴드가 아니라 PatchTST와 같은 `raw_future_return` 공간에서 계산했다. 각 horizon별로 as-of 시점 이전에 완료된 h-step 누적수익률의 rolling 평균과 표준편차만 사용했으므로 target 누수는 없다.

Validation 기준 주요 비교:

| 후보 | coverage | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|
| q25-b2 | 0.936846 | 0.046347 | 0.250114 | 0.069396 | 0.005063 | 6.115722 |
| q30-b2 | 0.890662 | 0.078626 | 0.214104 | 0.069014 | 0.004678 | 4.653222 |
| q35-b2 | 0.837282 | 0.112994 | 0.186324 | 0.068284 | 0.004552 | 4.168714 |
| BB20-2.0s | 0.879566 | 0.058766 | 0.102315 | - | - | - |
| BB20-1.5s | 0.788660 | 0.105403 | 0.076736 | - | - | - |
| BB20-1.0s | 0.622274 | 0.189240 | 0.051158 | - | - | - |

판단:

- `q25-b2`는 보수형 후보로 유지한다. coverage는 BB60-2.0s와 비슷하지만 절대 폭은 Bollinger보다 훨씬 넓다.
- `q30-b2`는 기본형 후보로 확정한다. coverage와 upper breach가 BB20-2.0s와 BB20-1.5s 사이에 있다.
- `q35-b2`는 공격형 후보로 유지한다. upper breach 11.30%는 BB20-1.5s 10.54%와 비슷하고 BB20-1.0s 18.92%보다 낮다.

다음은 full 473티커가 아니라 `q30-b2`, `q35-b2` 100티커 안정성 확인으로 간다.

### 10.32 CP25-R q30/q35 100티커 안정성 확인

CP25-R에서는 CP24-R의 기본형 `q30-b2`와 공격형 `q35-b2`를 100티커 3epoch로 확장해 확인했다. 입력 100개 중 재무 gate로 7개가 제외되어 실제 eligible ticker는 93개였다.

Validation 결과:

| 후보 | coverage | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return |
|---|---:|---:|---:|---:|---:|---:|
| q30-b2 50티커 | 0.890662 | 0.078626 | 0.214104 | 0.069014 | 0.004678 | 4.653222 |
| q30-b2 100티커 | 0.463702 | 0.283601 | 0.069402 | 0.026855 | 0.001382 | -0.184498 |
| q35-b2 50티커 | 0.837282 | 0.112994 | 0.186324 | 0.068284 | 0.004552 | 4.168714 |
| q35-b2 100티커 | 0.384560 | 0.323423 | 0.056351 | 0.028621 | 0.001519 | -0.108444 |

판단:

- q30-b2는 기본 후보 유지 기준에 실패했다. coverage가 0.85~0.93 기준에서 크게 벗어났다.
- q35-b2는 공격 후보 유지 기준에 실패했다. upper breach가 0.15 경고선을 크게 넘었다.
- 두 후보 모두 test 지표는 양호했지만 validation/test 격차가 커서 full run 근거로 쓰면 안 된다.
- full 473티커 실행은 금지한다.

선택 항목인 1W q30 smoke는 학습 데이터가 비어 있어 준비 단계에서 실패했다. q35 1W는 진행하지 않았다.

다음은 `q25-b2` reserve를 100티커로 재검증한다. q25도 실패하면 `q20-b2` 쪽으로 되돌려 100티커 기준 calibration을 다시 잡는다.

### 10.33 CP26-R 100티커 보수 calibration 재확인

CP26-R에서는 q30/q35를 폐기하고 100티커 기준에서 q25/q20/q15 계열로 되돌려 밴드 calibration을 다시 확인했다. 1W 실패로 비워진 `ai/cache/ticker_id_map_1w.json`은 기존 매핑으로 복원했고, 이번 CP에서는 1W 학습을 재시도하지 않았다.

최종 checkpoint 기준 validation 결과:

| 후보 | coverage | upper_breach_rate | avg_band_width | spearman_ic | long_short_spread | fee_adjusted_return | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| q25-b2 | 0.544565 | 0.242948 | 0.083824 | 0.026765 | 0.001303 | -0.206959 | 탈락 |
| q20-b2 | 0.650025 | 0.188960 | 0.104741 | 0.012186 | 0.001467 | -0.118567 | 탈락 |
| q15-b2 | 0.749335 | 0.137470 | 0.128752 | 0.015283 | 0.002072 | 0.314993 | 공격 후보 |
| q20-b1 | 0.652703 | 0.186760 | 0.105156 | 0.015393 | 0.001486 | -0.094951 | 탈락 |

중요 진단:

- validation total loss가 좋아질수록 밴드가 계속 좁아져 coverage가 무너지는 경향이 확인됐다.
- q25-b2 epoch1은 coverage 0.8897, q20-b2 epoch1은 coverage 0.9244로 후보권이지만 현재 저장 checkpoint가 아니다.
- q15-b2 epoch2는 coverage 0.8026, upper breach 0.0953으로 가장 균형이 좋지만, 현재 checkpoint는 epoch3이다.
- 따라서 단순 val_total 기준 checkpoint selection은 band calibration 목적과 충돌한다.

결론:

- 기본 후보는 아직 없다.
- q15-b2만 공격 후보로 생존했다.
- full run은 계속 금지한다.
- 다음 CP는 coverage-aware checkpoint selection 또는 q15-b2 `epochs=2` 재현 smoke가 우선이다.

### 10.34 CP28-R PatchTST 200티커 안정성 확인

CP27-R에서 coverage-aware checkpoint selection으로 q20/q25/q15 후보가 100티커 기준 복구됐지만, CP28-R에서 같은 정책을 200티커로 확장하자 세 후보 모두 coverage gate를 통과하지 못했다. 입력 200티커 중 재무 gate로 12개가 제외되어 실제 학습 대상은 188티커였다.

Validation 선택 checkpoint 기준:

| 후보 | selected_epoch | coverage | upper_breach_rate | spearman_ic | long_short_spread | fee_adjusted_return | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| q20-b2 | 3 | 0.598411 | 0.224333 | 0.055914 | 0.010699 | 356.585229 | 탈락 |
| q25-b2 | 3 | 0.499277 | 0.276971 | 0.055661 | 0.010664 | 347.089471 | 탈락 |
| q15-b2 | 3 | 0.702103 | 0.166290 | 0.056516 | 0.010826 | 385.782933 | 탈락 |

Test 기준으로는 세 후보 모두 fee_adjusted_return이 약한 음수였고, coverage도 0.49~0.69 수준으로 낮았다. q15-b2 epoch1은 coverage 0.757660으로 하한은 넘겼지만 upper breach 0.168904가 공격형 한계 0.15를 넘어서 통과 후보가 아니었다.

결론:

- full 473티커 실행은 계속 금지한다.
- CP27의 전환점은 유효했지만, 200티커 확장 안정성은 실패했다.
- 다음은 PatchTST full run이 아니라 DLinear/NLinear 같은 단순 baseline 준비 또는 ticker universe 확장에 강한 calibration 재설계가 우선이다.
### 10.35 CP29-D 가격/피처 계약 수리

CP28-D 감사에서 발견한 `open_ratio/high_ratio/low_ratio` 폭주 원인을 수리했다. 원인은 raw EODHD OHLC와 `adjusted_close`를 섞어 가격 ratio를 계산하던 계약 오류였다. CP29-D부터 모델 입력 가격 피처는 adjusted OHLC 기준으로 통일한다.

적용 계약:

- `adj_factor = adjusted_close / close`
- `adj_open = open * adj_factor`
- `adj_high = high * adj_factor`
- `adj_low = low * adj_factor`
- `adj_close = adjusted_close`

수리 결과 200 universe 기준 `open_ratio` p99는 23.470229에서 0.033812로, `high_ratio` p99는 23.797772에서 0.074500으로, `low_ratio` p99는 22.646632에서 0.018251로 정상화됐다. feature contract version은 `v3_adjusted_ohlc`로 올렸고, 기존 v2 캐시는 삭제하지 않은 채 v3 feature/index 캐시를 새로 만들었다.

추가로 `vol_change`에서 이전 거래량 0 때문에 생기던 `Inf`를 제거했다. 50/100/200 universe 모두 feature contract 이후 non-finite count는 0이다.

50티커 PatchTST 1epoch smoke는 `--save-run` 없이 실행했다. 학습/평가 결과 JSON과 `after_stdio_flush` marker까지 출력되어 finite contract smoke는 통과했지만, 프로세스가 출력 후 종료되지 않아 셸 timeout 코드 124가 남았다. full 473티커 학습은 실행하지 않았다.

### 10.36 CP30-G 모델 실험 게이트 수리

CP29-D clean feature 이후 바로 성능 실험을 키우지 않고, 저장/재현/백테스트 신뢰성 P1을 먼저 수리했다. full 473티커 실행, W&B sweep, 대형 성능 비교, UI 수정은 하지 않았다.

수리한 계약:

- `model_runs.band_mode`를 schema와 runtime migration에 추가했다.
- `prediction_evaluations.lower_breach_rate`, `upper_breach_rate`를 schema와 runtime migration에 추가했다.
- `model_runs.feature_version`은 실제 `FEATURE_CONTRACT_VERSION=v3_adjusted_ohlc` 기준으로 저장한다.
- `predictions` unique/upsert key는 `run_id,ticker,model_name,timeframe,horizon,asof_date`로 바꿔 run별 prediction 이력을 보존한다.
- inference는 checkpoint config의 `ticker_registry_path`를 로드해 ticker id를 만든다. registry mismatch는 실패시킨다.
- backtest anchor는 `adjusted_close` 우선으로 통일했다.
- backtest position/turnover는 날짜별 equal absolute exposure 포트폴리오 단위로 재정의했다.
- coverage gate 실패 fallback run은 `completed`가 아니라 `failed_quality_gate`로 저장한다.

실제 `python -m backend.db.scripts.ensure_runtime_schema` 마이그레이션은 성공했다. `model_run_contract_columns=['band_mode', 'feature_version', 'status']`, `prediction_evaluation_contract_columns=['lower_breach_rate', 'upper_breach_rate']`가 확인됐다.

검증은 `ai.tests.test_inference_backtest`, `ai.tests.test_storage_contracts`, `ai.tests.test_checkpoint_selection` 12건이 통과했다. RevIN은 이번 CP에서 수정하지 않았고, 별도 ablation 계획만 `docs/cp30_model_experiment_gate_repair_report.md`에 남겼다.
