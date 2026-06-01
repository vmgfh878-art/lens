# Lens v2 Master Plan

> 통합 문서. 기존 `docs/current/phase2_lens_signal_plan.md` + `docs/public/band_mlops_plan.md`
> + 2026-05-28 제품 구조 논의를 하나로 모았다. v2 작업의 단일 출처(single source).
> 기존 두 문서는 상세 이력 참조용으로 남기되, v2 진행은 이 문서 기준.

작성일: 2026-05-28

---

## 0. 정체성 & 포지셔닝 (가장 먼저)

### Lens 는 무엇인가
**daily batch 기반 "리스크 인식 보조 도구".** 미국 주식의 보수적 기준선 + 위험 범위(band) + 보조지표 + 전략 백테스트를 한 화면에서 본다.

### 핵심 포지셔닝 전환: "예측" → "리스크 인식"
- line IC ~0.04 는 학술적으로 약하다. **이건 한계가 아니라 시장 효율성의 본질** — 누구도 IC 0.2 못 낸다 (TradingView/Bloomberg 도 "예측 정확도"로 안 판다).
- 따라서 **"맞추는 도구"가 아니라 "위험을 먼저 보는 도구"** 로 포지셔닝한다.
- 약한 IC 가 약점이 아니라 "우리는 과신하지 않는다"는 정직함의 강점으로 전환된다.
- 제품 성공 = 모델 정확도(IC) 가 아니라 **신뢰도 + UX + 차별화된 인사이트.**

### 차별화 축 (v2 가 만드는 것)
1. **Conformal coverage** — "이 범위에 X% 확률" 수학적 보장
2. **CBM(설명가능)** — "왜 위험한지" concept 단위 설명
3. **Selective output** — "확신 없으면 침묵" (과대예측 안 함)
4. **LLM 리스크 브리핑** — 가격+뉴스+8-K → 자연어 요약 (기존 제품에 거의 없음)

→ 정체성: **"가장 정직하고 설명 잘하는 리스크 도구".** 학술(KAIRI) + 제품 + 포트폴리오 동시 충족.

---

## 1. 제품 구조 발전 로드맵 (Stage 0~5)

"구조적으로 필요한 만큼" = daily batch 리스크 도구 정체성에 맞는 만큼.

### Stage 0 — 현재 (v1)
```
윈도우 PC (inference + parquet 생성) → git push → Render(FastAPI, parquet serve) → Vercel(Next.js)
```
약점: PC 의존 / git 비대(.git 이미 1.4GB) / 캐싱 없음 / 계정 없음 / 정합성 취약

### Stage 1 — 데이터 레이어 🔴 (v1 막바지, 최우선)
- **Supabase Pro (Postgres + Auth)** 로 parquet 탈피
- **캐싱 레이어 필수** (egress 20GB 터진 교훈 — Pro 라도 캐싱 없으면 또 터짐)
- 증분 업데이트 (매일 row append, parquet 통째 rebuild 불필요)
- Auth → 계정/개인화 문 열림
- 학습 포인트: DB schema, ORM(SQLAlchemy), 캐싱 전략, Auth flow
- **동시에 .git history 1회 정리(filter-repo)** 로 1.4GB 청소 + parquet git 탈피

### Stage 2 — Inference 클라우드화 🟡
- 윈도우 PC → Docker 이미지 → Modal/GitHub Actions cron
- PC 안 켜도 매일 자동 inference
- 학습 포인트: Docker(환경 재현), serverless/cron, 모델 아티팩트 버전 관리

### Stage 3 — 제품 기능 🟡 (도구 → 제품)
- 워치리스트 / 알림(위험 구간 진입 시) / 포트폴리오
- LLM 리스크 브리핑
- 학습 포인트: 상태 관리, webhook/email, 백그라운드 큐

### Stage 4 — 관측/신뢰성 🟢 (운영 시작하면)
- Sentry(에러) + drift 모니터링(rolling IC) + uptime
- 학습 포인트: 모니터링, ML 관측(evidently)

### Stage 5 — 스케일 ⚪ (당분간 불필요)
- CDN / 다중 인스턴스 / 실시간(필요 시)
- daily batch 라 실시간 inference 는 거의 끝까지 불필요

### 우선순위
```
1. Stage 1 (DB+캐싱+Auth)        ← v1 막바지, 가장 큰 ROI
2. Stage 2 (inference 클라우드화)  ← Docker 학습
3. Stage 3 (제품 기능)            ← v2 모델과 함께
4. Stage 4 (관측)                ← 운영 시작 시
```
실시간 serving·K8s·Triton 은 daily batch 리스크 도구엔 **과투자**. 손대지 않는다.

---

## 2. 데이터 레이어 (Stage 1 상세)

### Supabase Pro 선택 이유
- 다른 무료 DB(Neon/CockroachDB) 대비 **올인원**: Postgres + **Auth** + Storage + Realtime + Edge Functions
- "기업 제품 급" 가려면 사용자 계정/개인화 필수 → Supabase Auth 가 공짜로 줌
- $25/월 비용은 실제 출시 목표면 투자 가치

### 터진 원인 (재발 방지)
- Database size 2.06GB / 1.1GB (full data 과적재)
- Egress 20GB / 5.5GB (캐싱 부재, 매 요청 큰 쿼리)
→ **교훈: (1) serving 1년치만 적재 (2) 응답 캐싱 레이어 필수.** Pro 사도 캐싱 없으면 재발.

### 배포 DB 에 넣을 것 (thin table)
ticker, asof_date, timeframe, signal_group/label, reason_code/text, risk_summary,
backtest_summary, model_version, data_freshness, line_score/band_lower/upper(serving 1년치)

### 넣지 않을 것
전체 price history, indicator matrix, 학습 feature tensor, 대량 prediction history,
CP 로그, raw provider payload

### parquet → DB 전환 작업 (며칠)
- backend parquet read 4곳(local_market_svc / predictions / strategy_backtest_svc / product_history) → SQL/ORM
- 적재 파이프라인 (inference 결과 직접 DB insert)
- Render free tier 외부 DB 연결 + connection pool + 쿼리 캐싱
- **v2 Band(Conformal/CBM) 작업과 함께 데이터 레이어 재설계가 자연스러움**

---

## 3. 모델 서빙 (batch 답게 가볍게)

### batch vs realtime
**Lens 는 batch.** 5거래일 예측을 매일 1번 계산 → 저장 → serve. 사용자 요청마다 모델 안 돌림.
→ TorchServe/Triton 같은 실시간 serving 인프라 **과투자.**

### "서빙 개선" 의 진짜 의미 = inference 클라우드화
현재 inference 가 윈도우 PC 에 묶임(5/26 멈춤이 증상). PC 탈피가 핵심.

| 방법 | GPU | 비용 | 적합 |
|---|---|---|---|
| 윈도우 PC + cron (현재) | 5060Ti | 무료 | PC 켜야 함(약점) |
| GitHub Actions cron | CPU | 무료 | daily 면 CPU 충분 |
| Modal/Replicate | serverless GPU | 무료크레딧+사용량 | inference 만 GPU, 끝나면 끔 ⭐ |
| Docker + 클라우드 VM | 선택 | VM 비용 | 무거움 |

### Docker 역할
서빙이 아니라 **환경 재현(PyTorch/CUDA/의존성 패키징)**. Docker 이미지로 환경 박고 → Modal/Actions 에서 매일 1번 실행.

---

## 4. v2 모델 후보

### 4.1 Band v2 (1순위 — MLOps 첫 대상)

#### v1 의 정직한 한계 (CP204 plan §0 인용)
- **Lagging forecaster**: 과거 변동성 clustering 따라가는 수준. "변동성 커지기 전 미리 넓어짐" 없음
- CP202.2 진단: width 의 forward / backward 변동성 상관 = **0.34 (1D), 0.38 (1W)** → 명확한 lagging. 미래 변동성 예측 거의 못 함
- CP216.2 통계 검정도 같은 결론: 1D 는 단순 분위수 / GARCH walk-forward 못 이김. 1W 는 walk-forward GARCH 보다는 강하지만 historical_quantile 못 이김
- → v1 평가 narrative 가 "현재 상태 표시" 위주로 굳어진 이유 (= §0 "리스크 인식 도구" 정체성). v1 한계 인정한 정직 narrative

#### v2 1번 목표: 미래 예측성 강화
- v1 의 lagging 한계 자체를 푸는 게 v2 1번 목표 — "변동성 커지기 전에 미리 넓어지는" 밴드
- 그 위에 정직성 + 설명 가능성 (Conformal / CBM / Selective / XAI) 을 같이 얹음 — 정확도만 추구하다 over-trust 위험 없게
- 라인의 정확도 야망은 ROI 낮음 (CP216.2 결론: 통계 베이스라인과 동등). 정확도 야망은 라인이 아닌 밴드에서 다시 살림

#### v2 기둥
- **미래 예측성 강화** (CP204 plan §0 의 lagging 진단을 직접 풀기) — 1번 목표
- **Conformal Prediction**: coverage 보장 (Vovk/Romano)
- **CBM (Concept Bottleneck)**: 위험 concept 설명 (Koh 2020)
- **Selective output**: 확신 없으면 침묵
- **XAI 통합**: 모델 판단을 사용자가 이해할 수 있는 방향
- band 는 coverage/breach/width 지표가 명확 → 자동 검증 (MLOps) 붙이기 쉬움

### 4.2 Line v2 (별도 paradigm)
- 현재 Asymmetric MSE + deep backbone = multi-modal collapse 구조적 (CP203 결론)
- 후보: pinball loss(quantile), expectile, mean-variance, ensemble
- **주의: paradigm 바꿔도 IC 극적 개선 어려움(시장 효율성).** "안정성/재현성/설명" 목적이면 OK, "정확도 극대화" 목적이면 함정
- `docs/current/line_v2_brainstorm.md` 참조

### 4.3 Line V3 distributional (reserve)
- 단일 line → q05/q10/q35/q50/q90 분포
- q50 중심 / q35 표시 보수선 / q10 downside warning
- gate: alpha 유지(q50 IC 80-85%), 보수선 false-safe 개선, downside recall 개선, calibration, crossing, seed 안정성

### 4.4 LLM 통합 (차별화 핵심)
- Option C: LLM primary forecaster (도전적)
- Option D: CBM + LLM 자연어 설명 (XAI 정합)
- Option A: 8-K/뉴스 text → line feature 보강
- **LLM 리스크 브리핑** = 진짜 차별화 ("AAPL: 변동성 급등 + 8-K 소송 + 추세 약화 → 단기 주의")

### 4.5 모델 후보 표 (기존 phase2 유지)
| 후보 | 목적 | 통과 기준 |
|---|---|---|
| PatchTST direction head | 방향성 별도 head | direction acc/IC/spread 개선 |
| PatchTST severe downside head | 큰 하락 위험 감지 | severe recall 개선, false-safe 감소 |
| risk-aware multi-task loss | 수익/방향/위험 동시 | seed stability + risk metric 동시 통과 |
| CNN-LSTM risk ensemble | 구조 다양성 보조 | stress 구간 risk metric 보완 |

실험 순서: ① CP153 band 완료 → ② severe downside head → ③ direction head → ④ multi-task → ⑤ ensemble → ⑥ distributional(reserve)

---

## 5. Band MLOps (champion/challenger)

### 왜 band 부터
line 은 false-safe/break/stop-loss 해석이 복잡. band 는 coverage/breach/width/stress 지표가 명확 → 자동 검증 쉬움.

### 핵심 원칙
자동 데이터검증/champion재평가/challenger학습/calibration갱신/승격추천 = 허용.
**product slot 자동 교체 = 금지. 최종 promotion = 사람 승인.**

### 두 트랙
**Recalibration Track** (저비용, 먼저): daily coverage/breach monitor → weekly conformal Q 후보 → monthly calibration 교체 추천
**Retraining Track**: monthly 제한 retrain → quarterly 넓은 sweep → event-driven(provider 변경/drift/coverage 붕괴)

### 자동 루프
```
새 데이터 → snapshot/manifest → data validation → feature build
→ champion 재평가 → baseline 재평가 → challenger 학습 → calibration fit
→ seed stability → walk-forward → stress regime → 비용/속도 확인
→ recommendation report → 사람 승인 → product slot promotion
```

### Gate
| gate | 기준 |
|---|---|
| data | provider/source/hash 고정, duplicate 0, non-finite 0, split overlap 0 |
| metric | coverage_abs_error, lower/upper breach, asymmetric interval score, band_width_ic |
| stress | VIX high/stress regime coverage 붕괴 없음 |
| stability | seed 3개(42/7/123), walk-forward 5-8 window, worst fold 붕괴 없음 |
| product | payload schema, inference latency, rollback 가능성 |

### 추천 등급
PROMOTE_RECOMMENDED / HOLD_MORE_VALIDATION / REJECT_REGRESSION / REJECT_INSTABILITY / REJECT_DATA_RISK

### baseline (비교 기준)
GARCH / Historical Quantile / Bollinger (사용자가 이해하기 쉬운 chart 기준)

### Registry 기록
snapshot id, provider/source/hash, universe, timeframe, horizon, model role/family, seed,
calibration method, metric summary, gate result, recommendation, promotion decision, rollback target

---

## 6. 자동 후보 탐색 루프 (반자동)

자동 탐색/평가/추천 = 허용. 자동 product slot 교체 = 보류. gate 자동 완화 = 금지.

자동 탐색 범위(허용): lr, weight decay, dropout, seq_len, patch_len/stride, q_low/q_high,
loss weight, feature pack, backbone(TCN/TiDE/PatchTST/CNN-LSTM) 비교
보류: raw source 자동 변경, feature contract 전역 변경, universe 자동 변경, gate 완화, slot 자동 교체

seed stability: 3개(42/7/123) — median/worst/std/pass비율 모두 확인
walk-forward: 5-8 window (상승/하락/횡보/고변동/저변동/금리스트레스/최근)

---

## 7. 제품 기능 & 개인화 (Stage 3, 도구→제품 갭)

v2 plan 에 없던 "제품화" 핵심:

1. **포지셔닝 정직화** 🔴 — "예측" → "보수적 기준선 + 리스크 범위"
2. **개인화/계정** 🔴 — Supabase Auth → 워치리스트/포트폴리오/저장
3. **알림** 🟡 — "보유 종목 위험 구간 진입 시 email/push" (리스크 도구의 핵심 가치)
4. **LLM 리스크 브리핑** 🟡 — 자연어 통합 요약 (차별화)
5. 데이터 확장 🟢 — 실시간/ETF/글로벌 (나중)
6. 전략 마켓플레이스 🟢 — 사용자 전략 공유 (먼 미래)

### 전략 관리 (모델 급)
- 전략 정의 = `backend/app/strategies/strategy_rules.py` 단일 출처 (진입/청산/위험 명세)
- 전략 cache 는 admin/reload 에 연결 (데이터 갱신 시 같이 비움)
- 전략 변경해도 데이터 갱신 파이프라인 불변

---

## 7.5 전략 중심 제품 방향 & 저작권 (2026-05-28 시장 조사)

### 배경
가격 raw 차트 표시는 데이터 라이센스 위험(yfinance/EODHD 재배포 제약)이 있어, 상업화 시
"전략을 직접 짜고 그 결과(MDD/해석)를 보는 앱" 방향을 검토했다.

### 시장 현실 (냉정)
전략 빌더 + 백테스트는 **이미 성숙한 레드오션.** 빈 시장이 아니다.
- **ChartingLens** — 자연어 입력("Buy when RSI crosses above 30 & MACD bullish") → AI 백테스트. 거의 동일 컨셉이 #1.
- **TradingView** — Pine Script + Strategy Tester (equity curve/metrics), 거대 커뮤니티
- **Composer** — drag-drop 포트폴리오 전략 (무료 tier 강력)
- **TrendSpider / NinjaTrader / AlgoAlpha / LuxAlgo** — no-code 빌더 다수

### 평가
- "그런 앱이 없다" = 사실 아님 (경쟁자 다수)
- "차트 없이" = 강점 아니라 약점. 저작권 회피는 제약이지 가치가 아님
- 저작권상 결과 중심이 안전 = **타당.** raw 가격 차트는 위험, 백테스트 결과(equity/drawdown/metrics)는 derived data 라 안전
- 단순 백테스트 빌더로는 ChartingLens 에 밀린다

### Lens 차별화 경로 (경쟁자에 없는 결합)
1. **AI 리스크 신호 + 사용자 전략 결합** ⭐ — "내 RSI 전략 + Lens AI 밴드 하단 이탈 시 청산". 사용자 전략에 보수적 line/band 를 필터/안전장치로 끼움. 단순 백테스트가 아니라 "AI 리스크 인식이 결합된 전략"
2. **LLM 교육적 결과 해석** ⭐ — "MDD -30%, 2022 약세장 6개월 미회복. 변동성 필터 권장". 숫자만 주는 경쟁자와 차별. "가장 정직하고 설명 잘하는" 정체성과 일치
3. **저작권 안전 = 결과 중심** — raw 캔들 대신 equity curve/drawdown/전략 신호(derived) 시각화

### 더 develop
- 전략 템플릿(초보) + 커스텀 + Lens AI 필터 오버레이
- 전략 비교 (내 전략 vs buy&hold vs Lens AI 밴드 방어)
- 리스크 교육 레이어 (MDD/샤프/승률 스토리텔링)
- 전략 공유 커뮤니티 (먼 미래)

### 결론
방향 자체(전략+백테스트)는 레드오션이라 단독으론 약함. **"AI 리스크 신호 결합 + LLM 교육적 해석 + 저작권 안전한 결과 중심"** 으로 좁히면 차별화 가능. 저작권 인식이 "결과 중심" 설계로 자연스럽게 이어짐. (사용자가 나중에 이 방향 직접 훑고 결정 예정.)

---

## 8. 자산군 확장 (ETF / 한국)

프론트 화면은 하나, 내부 모델은 asset_class/market/universe/timeframe/model_role 로 분리.

### 라우팅
```
request: ticker=SPY, timeframe=1D, model_role=band
registry: SPY → asset_class=ETF, market=US, universe=us_etf_core
router: 1d_band_us_etf_v1 선택
```

### ETF 전용 band (TCN/TiDE 중심, idiosyncratic 보다 macro/regime)
초기 30-80개: Market(SPY/QQQ/DIA/IWM), Sector(XLK/XLF/XLE...), Bonds(TLT/IEF/SHY),
Credit(HYG/LQD), Commodities(GLD/SLV/USO), Intl(EFA/EEM)
주의: VIXY/UVXY 는 reference/regime feature 로, leveraged/inverse 제외

### 한국시장 (Phase 후반)
별도 universe: 데이터 소스/거래일 캘린더/환율/상하한가/호가단위 다름. universe contract 먼저.

---

## 9. 배포 구조

```
Vercel             : Lens(Signal) frontend, 공개 데모, 클릭 이벤트
Render/Cloud Run   : FastAPI backend, signal/inference read API
Supabase           : thin signal table + Auth + feedback + analytics
Modal/Actions      : inference batch job (PC 탈피)
Local(또는 R2/HF)  : model checkpoint, raw 데이터, 학습 로그
```
학습은 운영 서버에서 안 돌림. 승격된 결과만 serve.

### git 비대 해결 (Stage 1 동시)
- 현재 serving parquet(53MB) 가 git tracked → .git 1.4GB
- DB 전환 시 parquet git 탈피 → .git 안 커짐 + history 1회 정리

---

## 10. 마일스톤

```
v1 막바지 (지금):
  A. 자동 push(임시) → 프론트 정리 → README        ← 이번 주말
  B. Supabase Pro DB + 캐싱 + Auth (Stage 1)        ← v1 마감
  C. 크론(자동화 일원화) + .git history 정리

v2:
  Phase 2-A: Band v2(Conformal/CBM/Selective) + 모델 후보 검증
  Phase 2-B: 제품 신호 계약(thin schema) + DB 정합
  Phase 2-C: 제품 기능(워치리스트/알림/LLM 브리핑) + 사용자 검증
  Phase 2-D: inference 클라우드화(Docker+Modal) + 배포 리허설
  Phase 2-E: Band MLOps 자동 추천 루프(champion/challenger)
  Phase 2-F: 자산군 확장(ETF), 한국시장(후반)
```

---

## 11. 결정 필요 항목 (열린 질문)

- DB 전환을 v2 Band 작업과 묶을지, 독립 CP 로 먼저 할지
- inference 클라우드화: Modal vs GitHub Actions(CPU)
- 모델 checkpoint 저장소: Local vs R2 vs HF
- Line v2 paradigm: pinball / ensemble / distributional 중 우선
- LLM 통합 시점: v2 초반 vs 후반
- 공개 범위: 완전 공개 데모 vs 링크 공유 반공개
- 자동 탐색 주기: weekly vs 월 2회
- 모델 버전 토글 UI 위치 (§11.5 참조): 카드 내 아코디언 vs "모델 진화" 별도 섹션 vs 탭
- **XAI 트랙 신설 여부** (사용자 제기, 2026-05-30) — 정확도 야망 낮추는 결정과 짝. SHAP / Integrated Gradients / Attention 시각화 / LLM 자연어 설명. v1 마무리 후 v2 재설계 시점에 박기. 후보 위치: §0 차별화 축, §3 모델 서빙, §7 제품 기능 중 한 곳
- **Band v2 — 전통 통계 (GARCH walk-forward, historical_quantile) 이기는 게 명시 목표** (사용자 제기, 2026-05-30). CP216.2 결과 = 현 1D/1W 밴드는 두 베이스라인 못 이김. Conformal/CBM 야망을 "calibration 정직성"에서 "pinball/coverage 둘 다 통계 베이스라인 이김"으로 격상. v1 마무리 후 v2 재설계 시점에 §4.1 Band v2 야망 갱신

## 11.5 v2 프론트 표현 — 모델 버전 토글 (후속안)

v1 끝까지는 정적이고 운영 모델은 (CP210 라인 / CP153 1D 밴드 / CP178 1W 밴드) 한 세트뿐이다. v2에서 line v2 / band v2 가 붙는 시점부터 "이전 모델 / 현재 모델" 비교가 화면에 필요하다.

**원칙**
- 1D / 1W × 라인 / 밴드의 4개 슬롯은 고정. 슬롯 자체는 안 바꾼다.
- 각 슬롯 안에서 "모델 버전"만 토글로 갈아낀다 (v1 / v2 / v3 ...). 선택 즉시 해당 모델 설명·평가·매니페스트가 화면에 갱신.
- "이전 모델을 영구히 볼 수 있다"는 점을 명시 — v2 붙는다고 v1 모델 설명을 지우지 않는다.

**적용 화면 (2단계)**
1. AI 모델 화면 — 모델 설명 패널 상단에 버전 셀렉터. 운영 모델만이 아니라 이전 v1 운영 모델도 선택 가능.
2. 주식 보기 화면 — 차트 오버레이가 버전별 예측선/밴드를 보여주도록 (v2 붙은 뒤 진행). v1까지는 운영 버전 한 가지만 그린다.

**구조 제안 (frontend)**
- `frontend/src/lib/training/registry.ts` (v2 도입 시 신설)
  - `MODEL_REGISTRY: Record<SlotId, ModelVersion[]>` — 슬롯당 버전 배열
  - `ModelVersion = { id, label, cp, status: 'current'|'archived', manifest_path, evaluation_targets }`
  - v2가 붙을 때 객체 한 줄 추가만으로 화면이 늘어남 (= 갈아엎지 않음)

**열린 질문**
- UI 위치: 카드 내 아코디언 vs 별도 "모델 진화" 섹션 vs 탭 (§11에 등록)
- 주식 보기에서 v1/v2 동시 비교 모드를 줄지, 단일 버전만 선택할지

## 11.6 도커 트랙 (v2 첫 CP)

**왜 v1이 아니라 v2 첫 CP인가**
- 재현성의 5축 (코드 / 데이터 / config / 환경 / 시드) 중 도커는 환경만 잠근다. 라인 v1 재현 사고는 데이터 윈도우/시드/config가 원인이었지 환경이 아니었음.
- v1 막바지엔 데이터·config·시드 lock (Stage 1 DB 적재 + manifest)이 더 효율적.
- 도커는 v2 mlops (champion/challenger 자동 루프 + GitHub Actions CI)의 전제조건이라 그 직전에 박는 게 자연스럽다.

**2단 구성**
1. **도커 dev (CPU)** — 백엔드 + 프론트 dev 환경. CI 회귀 테스트용. 학습 X.
2. **도커 train (GPU)** — 학습 파이프라인 재현용. `nvidia-container-toolkit` + CUDA 12.8 base + torch nightly (사용자 GPU = RTX 5060 Ti `sm_120` `cu128`).

**예상 난점**
- `sm_120` 은 일반 PyTorch 빌드에 없을 수 있어 nightly 또는 자체 빌드 필요. base image 잠그기 까다로움.
- 학습 한 번 재현하려면 데이터 parquet 마운트 + checkpoint 경로 mount 정책 필요.

**ROI 우선순위**
1. 데이터/config/seed lock — 재현성 +80%, 비용 낮음 → v1
2. requirements pin — +10%, 무료 → v1
3. 도커 dev — +5% → v2 첫 CP
4. 도커 train (GPU) — +5% → v2 mlops 트랙 안

## 11.7 재현성 lock 확장 (v2)

v1에서 운영 모델 3개 한정 매니페스트로 시작 (`docs/v1_operating_models_reproducibility.md`). v2에서는 다음 확장이 필요:

- 모든 학습 스크립트가 표준 manifest JSON을 자동 생성 (run_id / cp / git_sha / dataset_hash / feature_pack / hyperparams / seeds / GPU / wall_clock)
- 학습 시점 코드 git tag 자동 박기 (`v2/<cp>-<run_id>`)
- 모델 카드 (Google Model Cards 표준) 자동 생성: intended use / data / metrics / limitations / ethical considerations
- 회귀 스냅샷 테스트: AAPL 등 고정 ticker의 예측 vs 저장된 기댓값 (허용 오차 안에서) → PR CI 게이트
- GitHub Actions CI: ruff + mypy + pytest + 스냅샷 → main 머지 게이트
- 실험 메타데이터 자동 적재 (DB의 `experiment_runs` 테이블) — v1.5 trgtraffic만 정적이고, v2부터는 새 실험 추가 시 프론트가 자동으로 읽어 표시

---

## 12. 한 줄 요약

**v1 막바지 = Stage 1(Supabase+캐싱+Auth+크론), v2 = Stage 2~3(inference 클라우드화 + Band v2 Conformal/CBM/Selective + 제품기능 + LLM 브리핑).**
모델 정확도(IC) 경쟁은 포기하고, **"가장 정직하고 설명 잘하는 리스크 도구"** 정체성으로 신뢰도+UX+차별화에 집중. 실시간 serving·K8s 는 batch 도구엔 과투자.

---

## 참조 (상세 이력)
- `docs/current/phase2_lens_signal_plan.md` — Phase 2 원본 상세 (사용자 검증 이벤트, 배포 문서 등)
- `docs/public/band_mlops_plan.md` — Band MLOps 원본 상세
- `docs/current/line_v2_brainstorm.md` — Line v2 paradigm 후보
- `docs/cp204_band_v2_plan.md` — Band v2 5 pillars / 12 concepts / 13 metrics
