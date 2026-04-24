# CP2.6 지시서 — N1→N2 손실 원인 특정 + 과거 구간 복구

## 배경

CP2.5에서 확정된 사실:
- `price_data` 10년치 완전 보유 (503 ticker × p50=2843행, 2015-01-02 시작)
- `company_fundamentals` 477 ticker가 8분기 이상 보유 (목표 달성)
- `indicators` 테이블 min_date p10 = **2023-04-18**, row_count p50 = **757**
- AAPL 파이프라인: **N1(2843) → N2(base dropna, 753)** 에서 2090행 증발
- N3·N5 동일 753 유지 (regime·fundamentals 쪽은 무고)

**추정 원인**: base feature 계산 중 **컨텍스트 merge(`macroeconomic_indicators`, `market_breadth`)** 또는 **`price_data` 개별 컬럼(`per`, `pbr`)** 이 2023-04-18 이전에 NaN → `dropna(REQUIRED_FEATURE_COLUMNS)` 에서 일괄 삭제.

indicators 저장 min_date와 dropna 생존선이 **정확히 일치**한다는 게 결정적 단서.

이 CP의 미션:
1. 2090행 손실의 **컬럼 단위 원인 특정**
2. 컨텍스트 테이블 부족분 **백필 실행**
3. 복구 불가능한 구간은 **`has_macro`/`has_breadth` 플래그 + 0 imputation** 패턴으로 살림
4. recompute 후 AAPL 1D ≥2200, 1W ≥450 달성 검증

---

## 예상 시간

| 단계 | 시간 |
|---|---|
| 진단 (컨텍스트 테이블 길이 + 컬럼별 NaN 분포) | 30~60분 |
| 분기 판단 | 10분 |
| 분기 A: macro 백필 (FRED API) | 30분~2시간 |
| 분기 B: breadth 백필 | 30분~1시간 |
| 분기 C: feature_svc 플래그 확장 + 테스트 | 60~90분 |
| indicators recompute (full range) | 30~60분 |
| 최종 검증 + 테스트 | 15~30분 |
| **총 체감** | **2~5시간** (분기별 편차 큼) |

기준: 로컬 Postgres 연결, FRED API rate limit 고려, 503 ticker 재계산.

---

## 권한 번들 (사전 승인)

**파일 쓰기 범위**:
- `scripts/diagnostics/**` (신규 진단 스크립트)
- `backend/app/services/feature_svc.py` (플래그 확장 필요 시)
- `backend/db/schema.sql` (플래그 컬럼 추가 시)
- `backend/db/scripts/ensure_runtime_schema.py` (플래그 컬럼 추가 시)
- `backend/collector/jobs/**` (macro/breadth 백필 관련 수정)
- `backend/collector/sources/**` (FRED/breadth 소스 어댑터)
- `tests/**` (회귀·신규 테스트)
- `docs/cp2.6_report.md` (리포트)

**허용 명령**:
- `uv run python scripts/diagnostics/*.py`
- `uv run pytest tests/`
- `uv run python -m backend.collector.jobs.sync_macroeconomic_indicators` (또는 프로젝트 실제 job 이름)
- `uv run python -m backend.collector.jobs.sync_market_breadth`
- `uv run python -m backend.collector.jobs.compute_indicators` (full range 모드)
- `uv run python -m backend.db.scripts.ensure_runtime_schema`

**금지**:
- `price_data` 테이블 **수정 금지** (CP2.5에서 정상 확인됨)
- `company_fundamentals` 테이블 **수정 금지** (정상 확인됨)
- `DROP` / `TRUNCATE` 금지
- `git push` 금지 (commit만)

**휴먼 개입 트리거**:
- FRED API key 없음 또는 rate limit으로 백필 불가
- 진단 결과가 A/B/C 분기 중 어디에도 해당 안 되는 이상 패턴
- 예상 시간 3배 초과 (15시간+)
- `indicators` 테이블 recompute 후에도 row_count p50 < 1500

---

## 실행 상세

### 1단계 — 컨텍스트 테이블 길이 확인 (진단 전용)

`scripts/diagnostics/context_coverage.py` 신규 생성.

확인 항목:
```python
# macroeconomic_indicators
macro = query("""
    SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS row_count,
           SUM(CASE WHEN us10y IS NULL THEN 1 ELSE 0 END) AS us10y_null,
           SUM(CASE WHEN yield_spread IS NULL THEN 1 ELSE 0 END) AS yield_spread_null,
           SUM(CASE WHEN vix_close IS NULL THEN 1 ELSE 0 END) AS vix_close_null,
           SUM(CASE WHEN credit_spread_hy IS NULL THEN 1 ELSE 0 END) AS credit_spread_hy_null
    FROM macroeconomic_indicators
""")

# market_breadth
breadth = query("""
    SELECT MIN(date), MAX(date), COUNT(*),
           SUM(CASE WHEN nh_nl_index IS NULL THEN 1 ELSE 0 END),
           SUM(CASE WHEN ma200_pct IS NULL THEN 1 ELSE 0 END)
    FROM market_breadth
""")

# price_data per/pbr 컬럼 NaN 분포
pricecols = query("""
    SELECT MIN(date), MAX(date),
           SUM(CASE WHEN per IS NULL THEN 1 ELSE 0 END) AS per_null,
           SUM(CASE WHEN pbr IS NULL THEN 1 ELSE 0 END) AS pbr_null,
           COUNT(*) AS total
    FROM price_data
    WHERE ticker = 'AAPL'
""")
```

### 2단계 — AAPL base feature 컬럼별 NaN 시작점 추적

`scripts/diagnostics/base_feature_nan_tracer.py` 신규 생성.

AAPL로 `build_features` 호출하되, **dropna 직전** 의 DataFrame을 뽑아 각 base feature 컬럼별로 "언제부터 non-null"인지 확인.

```python
# feature_svc.build_features 를 rubber-stamp 호출
# dropna 직전 frame 추출하려면 feature_svc 를 약간 리팩터해서 debug hook을 열거나
# 동일 로직을 스크립트에서 재현

for column in _BASE_FEATURE_COLUMNS:
    first_non_null = frame.loc[frame[column].notna(), "date"].min()
    null_ratio = frame[column].isna().mean()
    print(f"{column}: first_non_null={first_non_null}, null_ratio={null_ratio:.3f}")
```

이게 **결정적 증거**. 어떤 컬럼이 2023-04-18부터 시작하는지 이 표로 확정.

### 3단계 — 분기 판단

진단 결과 기반:

| 분기 | 조건 | 조치 |
|---|---|---|
| **A** | `macroeconomic_indicators` min_date가 2023-04-18 (또는 그 근방)부터 | FRED 전체 기간(2015-01-01~) 재백필 |
| **B** | `market_breadth` min_date가 2023-04-18 근방부터 | breadth 전체 기간 재백필 |
| **C** | 컨텍스트 테이블은 길지만 `price_data.per/pbr` 같은 특정 컬럼이 2023-04-18부터만 non-null | `per`/`pbr` 등을 REQUIRED에서 제외하거나 `has_price_metrics` 플래그 + 0 imputation |
| **복합** | A+B, A+C 등 동시 발생 가능 | 각각 조치 병행 |

**어떤 분기든 적용해야 하는 공통 조치 (안전망)**:
- 컨텍스트 merge 후 과거 구간에 NaN이 남는 경우를 대비해 **`has_macro`·`has_breadth` BOOLEAN 플래그** 추가 + NaN→0 imputation 패턴 도입.
- fundamentals와 **완전히 동일한 패턴**. 이미 §10의 `has_fundamentals`가 있으니 복제.

이 안전망이 있으면, 백필이 부분적으로만 성공해도(예: FRED VIX는 1990년부터, NH/NL index는 2010년부터 등 시리즈별 시작점 차이) 남는 NaN을 학습 중단 원인으로 만들지 않음.

### 4단계 — 실행

1. 백필이 필요하면 백필 job 실행 (macro·breadth).
2. `feature_svc.py`에 `has_macro`·`has_breadth` 플래그 추가:
   - `_MACRO_FEATURE_COLUMNS = ["us10y", "yield_spread", "vix_close", "credit_spread_hy"]`
   - `_BREADTH_FEATURE_COLUMNS = ["nh_nl_index", "ma200_pct"]`
   - merge 후: `merged["has_macro"] = merged[_MACRO_FEATURE_COLUMNS].notna().any(axis=1)` → `fillna(0.0)`
   - breadth 동일 패턴
   - `FEATURE_COLUMNS`에 2개 플래그 추가 (총 27 + 2 = **29**)
   - `REQUIRED_FEATURE_COLUMNS`에도 2개 플래그 추가 (항상 non-null이니 dropna 거르지 않음)
3. `schema.sql` / `ensure_runtime_schema.py`에 `indicators.has_macro`, `indicators.has_breadth` BOOLEAN 추가.
4. `compute_indicators.py` 실행 — `force_full_backfill=True`로 2015-01-01부터 전체 재계산.
5. `price_data.per`/`pbr`이 원인이었다면:
   - 임시 옵션 1: `_BASE_FEATURE_COLUMNS`에서 per/pbr 파생 피처 제거
   - 옵션 2: fundamentals 패턴 한 번 더 복제 (`_PRICE_METRIC_COLUMNS` + `has_price_metrics`)
   - 옵션 선택은 진단 결과 보고 결정

### 5단계 — 검증

- AAPL 1D row_count ≥ **2200** (원래 price_data 2843 기준, rolling 200일 손실 감안)
- AAPL 1W row_count ≥ **450**
- 전체 유니버스 row_count p50:
  - 1D: ≥2000
  - 1W: ≥400
- `has_macro` / `has_breadth` time-weighted 비율: 이상적 **90%+**, 최소 **50%+** (백필 성공도에 따라)
- 기존 17+5 테스트 green
- 플래그 관련 신규 테스트 추가 (최소 3건)

---

## 종료 보고 포맷

```
[CP2.6] 완료

## 1. 컨텍스트 테이블 분포
- macroeconomic_indicators: min_date=_, max_date=_, row_count=_, null 비율(us10y/yield_spread/vix_close/credit_spread_hy)=_
- market_breadth: min_date=_, max_date=_, row_count=_, null 비율(nh_nl_index/ma200_pct)=_
- price_data.per/pbr null 비율 (AAPL): _

## 2. AAPL base feature 컬럼별 first_non_null 날짜
(리스트)
- open: _
- close: _
- log_return_1/5/20: _
- rsi_14: _
- macd_*: _
- bb_*: _
- realized_vol_20: _
- volume_zscore: _
- us10y: _
- yield_spread: _
- vix_close: _
- credit_spread_hy: _
- nh_nl_index: _
- ma200_pct: _
- per/pbr (만약 포함): _

## 3. 원인 결정
분기 A / B / C / 복합 중: _
근거: _

## 4. 적용한 조치
- [ ] macro 백필 실행: Y/N, 백필 범위 _ ~ _
- [ ] breadth 백필 실행: Y/N, 백필 범위 _ ~ _
- [ ] feature_svc has_macro 플래그 추가: Y/N
- [ ] feature_svc has_breadth 플래그 추가: Y/N
- [ ] per/pbr 등 price metrics 플래그 추가: Y/N (해당 시)
- [ ] indicators recompute full range 완료: Y/N

## 5. 재계산 후 수치
- AAPL 1D row_count: _ (이전 753 → 목표 ≥2200, 실제 _)
- AAPL 1W row_count: _ (이전 157 → 목표 ≥450, 실제 _)
- 전체 유니버스 1D row_count p10/p50/p90: _
- 전체 유니버스 1W row_count p10/p50/p90: _
- indicators 1D min_date p10: _ (목표 2015년대)
- has_macro=1 time-weighted 비율: _
- has_breadth=1 time-weighted 비율: _
- has_fundamentals=1 time-weighted 비율: _ (참고, 여전히 낮을 예상)
- FEATURE_COLUMNS count: _ (27 → 29 예상)
- 기존 17+5 테스트: all green Y/N
- 신규 테스트: _건

## 6. CP3 준비 상태
- 26 ticker 제외 대상 확정 Y/N
- seq_len=252 (1D) / 104 (1W) 가능한 ticker 수: _ / _
```

---

## 체크포인트 준수

- 이 CP는 **진단 + 조치 + 검증** 3단 복합.
- 진단 결과로 분기가 갈리지만, 중간 보고는 분기 결정 시점 1회만 허용.
- 분기 결정 후엔 단일 종료 보고로 closure.
- 진단에서 "애매"면 중단하고 휴먼 개입 요청.
- 위 종료 보고 섹션 6개 전부 숫자로 채워야 CP2.6 closure.
