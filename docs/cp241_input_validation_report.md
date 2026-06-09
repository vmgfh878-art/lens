# CP241 — Input Validation 보고서

작성: 2026-06-09. OWASP A03 대응. ADR-0032 동반.

## 0. 한 줄 요약

`backend/app/core/validators.py` 신규 (`TickerStr` + `SearchStr` + 보조
`is_valid_ticker`). predictions.py 3 + stocks.py 3 = 총 6 endpoint 의 ticker
path param 을 `TickerStr` 로 교체. `timeframe` Literal["1D","1W"] +
`search` SearchStr. RequestValidationError handler 의 details minimal
보강. backend pytest 22 test 영구 안전망 박음. 운영 ticker 1d 188 +
1w 421 = 0 invalid 검증. **CP241 마감**.

## 1. 핵심 컴포넌트 존재 체크리스트

- [x] `backend/app/core/validators.py` (`TickerStr`, `SearchStr`,
  `is_valid_ticker`)
- [x] ticker 패턴 `^([A-Z]{1,5}(?:[.-][A-Z]{1,2})?|\d{6}\.K[SQ])$` — 점/하이픈
  둘 다 허용 (yfinance `BRK-B` 호환 보강)
- [x] `predictions.py` 3 endpoint: `/line/{ticker}` / `/band/1d/{ticker}` /
  `/band/1w/{ticker}` — TickerStr 적용
- [x] `stocks.py` 3 endpoint: `/{ticker}/prices` / `/{ticker}/indicators` /
  `/{ticker}/predictions/product-history` — TickerStr 적용
- [x] `stocks.py` timeframe — `Literal["1D", "1W"]`
- [x] `stocks.py` list_stocks search — `SearchStr | None`
- [x] `main.py` RequestValidationError handler 의 details minimal
  ({loc, type} 만)
- [x] `backend/tests/test_input_validation.py` 22 test 영구 안전망
- [x] `docs/adr/0032_input_validation_pattern.md`
- [x] 운영 코드 / ML 모델 / 응답 schema 0 수정 (검증만 추가)

## 2. 새 테스트 결과

```
pytest backend/tests/test_input_validation.py -q
22 passed in 1.15s
```

분류:

| 분류 | 수 | 설명 |
|---|---|---|
| invalid ticker parametrize | 9 | path traversal / 슬래시 변형 / SQL fragment / XSS payload / lowercase / 길이 초과 / 공백 / 특수문자 / 언더스코어 → 차단 (400/404/422) |
| null byte (별도) | 1 | httpx client-side InvalidURL 즉시 차단 (서버 도달 전, 가장 강한 layer) |
| valid ticker parametrize | 7 | AAPL / MSFT / GOOGL / BRK-B (yfinance) / BF-B / 005930.KS / 035720.KQ → 200/404 통과 |
| query 한계 | 4 | limit 9999 / search 100자 / search SQL fragment / timeframe 1M (Literal 외) → 422 |
| 응답 schema | 1 | 통일 (code "VALIDATION_ERROR" + meta.request_id) + details minimal (ctx/msg/input 노출 0) |
| **합계** | **22** | **22/22 PASS** |

## 3. 회귀 안전망 (CP241 변경 후 재확인)

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot | ✅ 9 passed (응답 schema 영향 0) |
| drift sim | ✅ 11 passed |
| CP240 security_headers | ✅ 8 passed |
| CP241 input_validation | ✅ 22 passed |
| 기타 cp223 | ✅ 1 passed |
| frontend Vitest | ✅ 166 passed (CP240 commit 시점 확인) |
| frontend tsc | ✅ 0 error |

## 4. 운영 ticker dry 검증 (Step 1 / 7)

```python
PYTHONUTF8=1 python -c "
import json, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'backend')
from pathlib import Path
from app.core.validators import is_valid_ticker
for tf in ['1d', '1w']:
    m = json.loads(Path(f'ai/cache/ticker_id_map_{tf}.json').read_text())
    bad = [t for t in m['mapping'] if not is_valid_ticker(t)]
    print(f'{tf}: total {len(m[\"mapping\"])}, invalid {len(bad)}: {bad[:10]}')
"
```

**결과**:
```
1d: total 188, invalid 0: []
1w: total 421, invalid 0: []
```

운영 ticker 회귀 0 확보 (daily refresh + 운영 모델 추론 영향 0).

## 5. 진행 중 발견 + 보강

### F1. 운영 ticker 의 yfinance 하이픈 표기 (Step 1)

초기 패턴 `[A-Z]{1,5}(\.[A-Z]{1,2})?` 으로 dry 검증 → **`BRK-B`, `BF-B` 2개
invalid** 발견. yfinance 의 표준 표기는 점 (`BRK.B`) 이 아니라 하이픈
(`BRK-B`) 임.

→ 패턴 `[.-]` 으로 보강 → 둘 다 허용 → 0 invalid.

### F2. null byte URL 의 httpx client-side 차단

`AAPL\x00` test 시 422 가 아니라 **httpx `InvalidURL` exception**. 서버
도달 전 차단 — RFC 3986 URL 표준 위반. production proxy (Render / Vercel)
도 동일 차단.

→ 별도 test (`pytest.raises(InvalidURL)`).

### F3. path traversal 의 routing 단계 unmatch

`../../etc/passwd` test 시 422 가 아니라 **404** (FastAPI routing 이 `/`
구분자 보고 unmatch). Pydantic constraint 도달 전 차단.

→ test 의 status assertion 을 `(400, 404, 422)` 모두 허용.

### F4. RequestValidationError handler 의 details 정보 노출

기존 handler 의 `details=exc.errors()` 가 raw pydantic 메시지 (`ctx.pattern`,
`input` 등) 노출 → 공격자 힌트.

→ `details=[{"loc": ..., "type": ...}]` 만 minimal 으로 변환. status/code
보존 (frontend `apiErrors.ts` 영향 0).

## 6. 산출물

### 신규
- `backend/app/core/validators.py` (TickerStr / SearchStr / is_valid_ticker)
- `backend/tests/test_input_validation.py` (22 test 영구 안전망)
- `docs/adr/0032_input_validation_pattern.md`
- `docs/cp241_input_validation_report.md` (본 보고서)

### 수정
- `backend/app/routers/v1/predictions.py` (3 endpoint TickerStr)
- `backend/app/routers/v1/stocks.py` (3 endpoint TickerStr + timeframe Literal + search SearchStr)
- `backend/app/main.py` (validation handler details minimal)

### 운영 코드 / ML 모델 / 응답 schema 0 수정

## 7. commit 이력 (CP241, 5 commit + closure)

```
5113742 CP241 Step 1: add core/validators.py (TickerStr + SearchStr) + 패턴 보강
76dd184 CP241 Step 2: apply TickerStr to predictions.py routes
0d091d1 CP241 Step 3-4: apply TickerStr + SearchStr + Literal timeframe to stocks.py
b9... CP241 Step 5: RequestValidationError handler 보강 (details minimal)
<step 6 commit hash> CP241 Step 6: input validation pytest (22 case 영구 안전망)
<본 commit> CP241 Step 8 closure: report + ADR-0032
```

(정확한 hash 는 `git log --oneline` 으로 확인.)

## 8. v2 재검토 (ADR-0032 §v2)

- 새 거래소 ticker 형식 (.HK, .TO, .DE 등) 도입 시 패턴 확장
- Auth 도입 시 사용자 input 다양화 → SearchStr 도 한국어 허용 검토
- `roles` (product-history endpoint) 등 enum 강제 보강
- 1M timeframe 도입 시 Literal 갱신

## 9. 다음 CP

CP242 (CORS + rate limit + 보안 트랙 종료). 진입 조건 (CP235 Pydantic
Settings 존재) 충족.
