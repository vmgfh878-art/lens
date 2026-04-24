# CP2.5 지시서 — 데이터 길이 감사

## 배경

CP2 종료 시점 발견된 블로커:
- AAPL 1D 학습 가능 행 **753** (목표 ≥2200 미달)
- AAPL 1W 학습 가능 행 **157** (목표 ≥450 미달)
- has_fundamentals=1 비율 **19.22% / 19.73%** (기대 대비 낮음)
- 실행 에이전트 추정: AAPL 가격/지표가 **~2023-04 이후만** 존재

재무 NaN 블로커는 CP2에서 제거됨. 남은 원인은 **가격/지표/재무의 절대 길이**.
어디서 짧은지 확정 전엔 CP3(26 ticker 제외)도 B-0(sufficiency 재확인)도 의미 없음.

**이 CP는 조사 전용. 수정·백필·recompute는 이 CP에서 하지 않는다.** 원인 규명 후 CP2.6에서 분기.

---

## 목표

다음 3개 축에 대해 **전체 유니버스 커버리지 분포**를 숫자로 확정한다.

1. **price_data** — ticker별 실제 보유 기간·행 수
2. **indicators** — timeframe별 (1D·1W·1M) ticker별 실제 저장 기간·행 수
3. **company_fundamentals** — ticker별 분기 커버리지

그리고 **근본 원인 3개 가설 중 어느 것인지** 확정한다:

- (H1) **price_data 자체가 짧다** — EODHD backfill이 완결 안 됨. 해결: full 10년 재백필.
- (H2) **price_data는 충분한데 indicators가 짧다** — compute_indicators가 과거를 컴파일 안 함. 해결: recompute full range.
- (H3) **양쪽 다 있는데 피처 생성 중 떨어진다** — rolling window, regime join, fundamentals merge에서 row가 삭제됨. 해결: feature_svc 로직 점검.

---

## 예상 시간

| 단계 | 시간 |
|---|---|
| 진단 스크립트 작성 | 30~45분 |
| DB 쿼리 실행 + 집계 | 10~20분 |
| 원인 규명 리포트 | 10분 |
| **총 체감** | **약 1~1.5시간** |

기준: Supabase Postgres 읽기 전용 쿼리, CPU 작업 위주.

---

## 권한 번들 (사전 승인)

**파일 쓰기 범위**:
- `scripts/diagnostics/**` (신규 진단 스크립트 생성 허용)
- `docs/cp2.5_report.md` (리포트 저장)

**허용 명령**:
- `uv run python scripts/diagnostics/*.py` (읽기 전용)
- `uv run pytest tests/` (회귀 확인)
- DB 읽기: `fetch_frame(...)`, `SELECT ...`

**금지**:
- `price_data`·`indicators`·`company_fundamentals` 테이블 **수정 금지**
- `DROP`/`TRUNCATE`/`DELETE` 금지
- 백필·recompute 실행 금지 (CP2.6 영역)
- 스키마 변경 금지
- `git push` 금지 (commit만 OK)

**휴먼 개입 트리거**:
- 쿼리가 3회 이상 timeout
- DB 연결 실패
- 예상 시간 3배 초과 (4시간+)
- 가설 3개 중 어디에도 해당 안 되는 이상한 패턴

---

## 실행 상세

### 1단계: `scripts/diagnostics/data_length_audit.py` 생성

다음을 한 스크립트로 뽑아 stdout + `docs/cp2.5_report.md` 양쪽에 출력.

```python
# 개념 스펙 (실제 구현은 프로젝트 convention 맞춰서)

def audit():
    # 1. price_data 분포
    pd_stats = query("""
        SELECT ticker,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*)  AS row_count
        FROM price_data
        GROUP BY ticker
    """)
    # 전체 N ticker, row_count 분위수 p10/p25/p50/p75/p90/max
    # min_date 분위수 (언제부터 시작하는가)

    # 2. indicators 분포 (timeframe별)
    for tf in ("1D", "1W", "1M"):
        ind_stats = query(f"""
            SELECT ticker,
                   MIN(date) AS min_date,
                   MAX(date) AS max_date,
                   COUNT(*)  AS row_count,
                   SUM(CASE WHEN has_fundamentals THEN 1 ELSE 0 END) AS with_fund_count
            FROM indicators
            WHERE timeframe = '{tf}'
            GROUP BY ticker
        """)
        # 같은 통계 + has_fundamentals 비율

    # 3. company_fundamentals 분포
    fund_stats = query("""
        SELECT ticker,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*) AS quarter_count
        FROM company_fundamentals
        GROUP BY ticker
    """)
    # 몇 ticker가 8분기 이상인가

    # 4. sync_state 확인
    sync = query("""
        SELECT job_name, target_key, last_cursor_date, message
        FROM sync_state
        WHERE job_name LIKE 'sync_prices%' OR job_name LIKE 'compute_indicators%'
        ORDER BY job_name, target_key
        LIMIT 50
    """)

    # 5. AAPL 집중 진단
    aapl = {
        "price_data":  query("... WHERE ticker='AAPL' ORDER BY date LIMIT 5 / OFFSET N-5"),
        "indicators_1D": query("... WHERE ticker='AAPL' AND timeframe='1D' ..."),
        "indicators_1W": query("... AND timeframe='1W' ..."),
        "fundamentals":  query("... company_fundamentals WHERE ticker='AAPL' ..."),
    }

    # 6. 가설 판정
    # H1 price 짧다:    p50(price_data row_count) < 2000
    # H2 price OK indicators 짧다:   p50(price) >= 2000 AND p50(indicators_1D) < 2000
    # H3 양쪽 OK 피처에서 떨어진다:  양쪽 >= 2000인데 dropna 후 짧음
```

### 2단계: AAPL 샘플 데이터 눈으로 확인

- `price_data` AAPL 가장 오래된 날짜 5행 + 가장 최근 5행
- `indicators` AAPL 1D 같은 조회
- `indicators` AAPL 1W 같은 조회

### 3단계: feature_svc.build_features에 실제 넣었을 때 AAPL 기준 어느 단계에서 row 손실되는지 추적

다음 체크포인트별 row 수를 찍는다:

```
AAPL price_data raw            : N0
→ resample to 1D                : N1
→ base feature 계산 후 dropna (rolling window 탓 초반 손실): N2
→ regime merge 후                : N3
→ fundamentals merge 후 (8Q gate 판정 포함): N4
→ 최종 REQUIRED_FEATURE_COLUMNS dropna : N5
```

이 6개 수가 나와야 H3 여부 확정 가능.

### 4단계: 리포트 생성

`docs/cp2.5_report.md`에 아래 순서로 작성.

---

## 종료 보고 포맷 (필수)

```
[CP2.5] 완료

## 1. price_data 분포
- 전체 ticker 수: _
- row_count 분포: p10=_, p25=_, p50=_, p75=_, p90=_, max=_
- min_date 분포: 가장 이른 p10=_, 가장 늦은 p90=_
- 2015-01-01 이전부터 있는 ticker 수: _
- 2020-01-01 이후 시작하는 ticker 수: _

## 2. indicators 분포 (1D / 1W / 1M)
각 timeframe별:
- 전체 ticker 수: _
- row_count p10/p50/p90: _
- has_fundamentals=1 비율 (time-weighted): _
- min_date p10/p90: _

## 3. company_fundamentals
- 전체 ticker 수: _
- 분기 수 분포 p10/p50/p90: _
- 8분기 이상 보유 ticker 수: _
- (재확인: 기대치 477, 실제 _)

## 4. AAPL 집중 진단
- price_data: min_date=_, max_date=_, row_count=_
- indicators 1D: min_date=_, max_date=_, row_count=_
- indicators 1W: min_date=_, max_date=_, row_count=_
- fundamentals: min_date=_, max_date=_, quarter_count=_

- 피처 파이프라인 row 손실 단계별:
  N0 (price raw): _
  N1 (resample): _
  N2 (base dropna): _
  N3 (regime merge): _
  N4 (fundamentals merge + 8Q gate): _
  N5 (final dropna): _

## 5. sync_state 스냅샷
- sync_prices:1D earliest last_cursor: _
- sync_prices:1D latest last_cursor: _
- compute_indicators:1D earliest last_cursor: _
- (필요 시 더)

## 6. 가설 판정

가설 H1 (price_data 자체가 짧다): [HOLDS / REJECTED]
  근거: _
가설 H2 (price OK, indicators 짧음): [HOLDS / REJECTED]
  근거: _
가설 H3 (양쪽 OK, 피처에서 손실): [HOLDS / REJECTED]
  근거: _

**원인 결정**: (H1 | H2 | H3 | 복합)

## 7. CP2.6 제안 (다음 CP 예고)

결정된 원인에 따라:
- H1 → 전 유니버스 EODHD 10년 백필 재실행
- H2 → compute_indicators full-range recompute
- H3 → feature_svc 피처 단계 수정
```

---

## 체크포인트 준수

- 이 CP는 **단일 CP**. 중간 진행 보고 없음, 종료 보고 1회.
- 위 8개 섹션 전부 채워야 CP2.5 closure.
- 가설 판정 섹션에 "모르겠다"는 없음. 숫자로 결정.
- 제안된 CP2.6 내용 보고만, 실행하지 않는다.

---

## 현재 가설 방향성 힌트 (편향 주의, 단 참고용)

사전 의심:
- **H1이 가장 유력**. 이유:
  - AAPL처럼 10년 존재 확실한 블루칩이 3년 수준 → 가격 수집 자체가 최근만 된 것
  - CP1에서 `sync_prices.py` overlap을 45→5~7일로 줄였는데, 이게 full backfill 로직을 건드렸을 가능성
- **H2는 부분적으로 가능**. compute_indicators가 TIMEFRAME_CONFIG `source_history_days` 기준으로만 돌면 과거 전체 재계산 안 함.
- **H3는 가장 가능성 낮음**. rolling 200일 window로 약 200행 정도만 잃는 게 정상. 2500행이 753으로 떨어질 수준 아님.

확정은 숫자로. 위 힌트에 끌려가지 말 것.
