# CP227 BE 안정성 — 에러 응답 표준화 + pandera 경계 검증 (Directive)

> 이 문서는 런북(`docs/refactoring_master_plan.md` 로드맵의 BE 안정성 트랙, 마스터플랜 표의 `CP-RF-4`)이
> 자동으로 꺼내 실행할 단일 지시서다. 실행자(새 Claude Code 세션)는 이 문서만 읽고도 코드를 고치고,
> 검증하고, 중단 판단을 내릴 수 있어야 한다. 추측 금지 — 아래 인용된 파일/줄번호는 작성 시점
> (2026-06-02, 브랜치 `main` 기준 HEAD `6ef47dd`) 실측이다. 실행 전 줄번호가 어긋나면 재확인 후 진행.

---

## 역할 고정

- **모드: code.** 지시받은 코드 작업을 직접 수행하고, 같은 턴에 자가 점검만 보고한다.
- **권한:** 코드 수정, 로컬 검증(pytest / mypy / ruff / 백엔드 기동 후 curl).
- **금지(하드):**
  - 새 모델 학습 금지. 새 calibration 금지.
  - DB write 금지. Supabase 호출 금지(이 CP는 로컬 parquet 경계만 다룬다 — `market_repo.py` 의 Supabase 분기는 **구조만** 좁히고 실제 호출/연결은 하지 않는다).
  - 사용자가 직접 수정한 파일을 revert 금지. 운영 parquet(`backend/data/v1/*.parquet`) write/재생성 금지(읽기 검증만).
- **자가 점검(필수, 보고 양식 하단):** Plan v3 정합 / 구조 결함 / 모델 영향 각각 PASS/WARN/FAIL + 사유.
- **커밋 메시지:** 간결. 한 줄 요약 + 필요한 만큼만. 끝에 반드시
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- 워킹 디렉토리: `C:\Users\user\lens`
- venv: `.venv` (Python **3.10.0**, pandas **2.2.2**, pyarrow 16.1.0, fastapi 0.111.0, torch는 cu128 별도). 실행 인터프리터: `.venv\Scripts\python.exe`.
- 백엔드 기동: `scripts\start_demo.ps1` 또는 직접 `uvicorn`:
  ```powershell
  cd C:\Users\user\lens\backend
  ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8123
  ```
  검증용 포트는 **8123** 등 비표준으로 띄워 기존 데모(8000/8080)와 충돌 피한다.
- 프론트: 이 CP는 프론트 변경 없음. `npm run dev` 불필요.
- 테스트 실행:
  ```powershell
  cd C:\Users\user\lens\backend
  ..\.venv\Scripts\python.exe -m pytest -q
  ```
- **pandera 미설치 확인됨** (`.venv` 에 `pandera` 없음, `backend/requirements.txt` 에도 없음). Step 2에서 설치 + 핀 추가가 선행 작업이다. 아래 Sub-step 참조.

---

## 진단 (근거)

조사 출처: 아래 파일을 직접 Read/Grep(2026-06-02). 줄번호는 실측.

### 진단 1 — admin 에러 응답에 민감정보(예외 타입·메시지·traceback) 직접 노출

`backend/app/routers/v1/admin.py` (현재 **171줄**), `debug_state()` 의 market probe 루프:

```python
# admin.py:149-155
        except Exception as exc:  # noqa: BLE001
            market_probes[slot] = {
                "status": "error",
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc)[:500],
                "traceback_tail": traceback.format_exc()[-1000:],
            }
```

- 이 엔드포인트는 `_require_reload_allowed` 같은 인증 게이트가 **없다**(docstring 79행: "인증 없이 호출 가능").
  즉 누구나 `GET /api/v1/predictions/health` 가 아닌 `GET /api/v1/admin/debug-state` 로 **traceback 꼬리 1000자 + 예외 메시지**를 받을 수 있다.
- traceback 꼬리는 파일 절대경로(`C:\Users\user\lens\...`), 내부 모듈 구조, 때로는 데이터 값까지 흘린다 → 정보 유출.
- 같은 파일 92행, 128행에도 `error`/`exc` 문자열을 응답에 싣는 약한 노출이 있다(파일명/메모리 probe). 이 두 곳은 경로/상태 진단 목적이라 위험도는 낮지만 표준 포맷과 일관성을 맞춘다.

### 진단 2 — 광범위 `except Exception` 9곳 (실측 위치)

Grep `except Exception` (backend/app):

| # | 파일:줄 | 맥락 | 처리 |
|---|---|---|---|
| 1 | `repositories/market_repo.py:142` | Supabase price 조회 | `raise UpstreamUnavailableError(...) from exc` (재발생 O, 로깅 X) |
| 2 | `repositories/market_repo.py:193` | indicator 컬럼 누락 재시도 루프 내부 | `_extract_missing_indicator_column` 판단 후 `raise` |
| 3 | `repositories/market_repo.py:211` | indicator 조회 바깥 | `raise UpstreamUnavailableError(...) from exc` (로깅 X) |
| 4 | `repositories/market_repo.py:245` | `_merge_indicator_volume` | `return`(삼킴 — volume 머지는 best-effort) |
| 5 | `repositories/market_repo.py:359` | stock_info 1차 조회 | `rows = []` (삼킴, 폴백으로 진행) |
| 6 | `repositories/market_repo.py:366` | fetch_stocks 바깥 | `raise UpstreamUnavailableError(...) from exc` (로깅 X) |
| 7 | `services/local_market_svc.py:117` | `_jsonable` numpy `.item()` | `return str(v)` (삼킴 — 직렬화 폴백, 정당) |
| 8 | `services/strategy_backtest_svc.py:230` | `_sector_map` parquet 읽기 | `return {}` (삼킴 — sector 없으면 Unknown) |
| 9 | `routers/v1/predictions.py:39` | `load_caches` 슬롯별 로드 | `summary[slot] = {"status":"error","error":str(exc)}` |

추가(노출 약함, 같이 정리 권장): `db.py:64`(check_supabase_ready, 재발생 O), `main.py:82`(startup eager load, 로깅 O), `predictions.py:81`/`local_market_svc.py:118`(numpy item 폴백), `product_prediction_history_svc.py:109`(parquet 엔진 오류, `# pragma: no cover` + 재발생 O).

문제: 위 중 **"파일없음 / 권한 / 메모리부족 / dtype / Supabase 네트워크"** 를 구분하지 못하고 전부 한 덩어리로 잡는다. 특히 #1/#3/#6은 **로깅 없이** UpstreamUnavailable 로 변환 → 운영 중 진짜 원인(예: pyarrow 깨진 parquet vs Supabase 5xx)이 로그에 안 남는다.

### 진단 3 — main.py 글로벌 핸들러는 표준 포맷이 있으나 일부 라우터가 우회

`backend/app/main.py:97-146` 에 `AppError` / `RequestValidationError` / `ValueError` / `Exception` 4개 핸들러가 있고, 표준 포맷은 `app/core/http.py:error_response` =
```json
{"error": {"code": "...", "message": "...", "details": ...}, "meta": {"request_id": "..."}}
```
그러나 `predictions.py`, `strategy_backtest_svc.py` 는 **`fastapi.HTTPException(status_code=..., detail="문자열")`** 를 직접 던진다(예: `predictions.py:60,134,165,199`; `strategy_backtest_svc.py:280,461,469,486,581`). FastAPI 기본 HTTPException 핸들러는 `{"detail": "..."}` 형태라 **표준 포맷과 다른 스키마**로 응답한다. 이는 이번 CP의 **차단 트리거 대상**(아래 "인터페이스 보존" 참조): 정상 동작 중인 404/503 응답 스키마를 바꾸면 프론트가 깨진다. **이번 CP에서는 HTTPException → AppError 전환을 하지 않는다**(별도 CP). 진단으로만 기록.

### 진단 4 — CP214 dtype 사고 재발 방지 장치 부재

`parquet_store._compress_strings` (`services/parquet_store.py:73-95`) 는 CP214 회귀를 막기 위해 `asof_date`/`forecast_date` 를 **categorical 에서 제외**(object 유지)한다. `strategy_backtest_svc._align_date_dtype` (`:27-44`) 가 머지 직전 방어선이다. 하지만 **"읽는 순간 dtype 계약을 강제로 검증하는" 장치는 없다.** 새 source/컬럼이 추가되거나 parquet 재생성 시 dtype이 흔들리면 머지가 깨지는 silent 버그가 다시 들어올 수 있다. pandera `DataFrameModel` 을 **읽기 직후 + 머지 직전/직후 경계**에 두면 "datetime64 vs category" 같은 사고를 읽는 순간 잡는다.

**실측 dtype(`.venv` 로 parquet 직접 로드, 2026-06-02):**

| frame | 파일 | 핵심 컬럼 dtype (on disk) |
|---|---|---|
| line_1d | `predictions_line_1d.parquet` (182,276행) | `ticker` object, `asof_date` **object(str)**, `line_score`/`safe_line_score`/`line_rank_by_date`/`safe_line_rank_by_date`/`actual_h5_return` float64, `line_top_decile_flag`/`safe_line_top_decile_flag` bool, `model_id`/`source_cp` object |
| band_1d | `predictions_band_1d.parquet` (594,865행) | `ticker` object, `asof_date` **object**, `forecast_date` **object**, `horizon_step` int64, `band_lower`/`band_upper`/`actual_return` float64, `actual_return_available` bool, `model_id`/`source_cp` object |
| band_1w | `predictions_band_1w.parquet` (186,900행) | `ticker` object, `asof_date` **object**, `horizon_step` int64, `band_lower`/`band_upper`/`actual_return` float64, `model_id`/`source_cp` object |
| market_prices_1d | `market_prices_1d.parquet` | `ticker` object, `date` **object**, `open`/`high`/`low`/`close`/`adjusted_close` float64, `volume` int64 |
| market_indicators_1d | `market_indicators_1d.parquet` | `ticker` object, `date` **object**, `timeframe`/`regime_label` object, `rsi`/`macd_ratio`/`ma_5_ratio`/`ma_20_ratio`/`ma_60_ratio`/`bb_position`/`vol_change`/`log_return` float64 |

> ★ **결정적 제약:** disk 상 `asof_date`/`forecast_date`/`date` 는 **str(object)** 이고, `parquet_store` 가 의도적으로 그렇게 유지한다(CP214 fix). 따라서 **pandera 스키마는 이 컬럼들을 datetime이 아니라 `str`(또는 object)로 모델링하고 `coerce=True` 로 둬야 한다.** 스키마를 datetime64로 잡으면 CP214 fix와 정면충돌해 검증이 실패하거나 dtype을 datetime으로 강제로 바꿔 머지 회귀를 다시 일으킨다. **이 점을 어기면 안 된다.**

---

## 선행 의존

- **CP223(백엔드 characterization 스냅샷) 그린 필수.** 이 CP는 정상 경로 API 응답이 무변경임을 "snapshot diff 0" 으로 증명해야 하는데, 그 스냅샷 안전망은 CP223이 만든다. CP223이 그린이 아니면 **이 CP를 시작하지 말고 보고**한다.
- pytest/ruff/mypy 안전망(마스터플랜 `CP-RF-1`)이 설치되어 동작해야 한다(`.venv\Scripts\python.exe -m pytest -q` 가 수집·실행됨). 미설치면 보고 후 중단.
- (FE 관련 선행 없음 — 이 CP는 BE 단독.)

---

## 범위

### 포함
- admin `debug_state` 응답에서 `exc_type` / `exc_msg` / `traceback_tail` 제거. `request_id` 만 상관 키로 노출. 내부 예외 상세는 **서버 로그로만**.
- pandera 도입: `requirements.txt` 핀 추가 + 설치. `app/schemas/frames.py` 에 5개 `DataFrameModel`(line_1d / band_1d / band_1w / market_prices_1d / market_indicators_1d) 정의 + `coerce`.
- 읽기 경계(`parquet_store._load`, `local_market_svc._load`) 와 머지 경계(`strategy_backtest_svc._load_frame` 머지 직전/직후)에 검증 적용.
- 광범위 `except Exception` 중 **재발생 경로(#1,#3,#6)** 를 구체 예외로 좁히고 좁히기 전에 **최소 로깅(logger.warning/exception) 후 재발생**. best-effort 삼킴(#4,#5,#7,#8)은 **삼킴 사유를 주석 + debug 로깅**만 추가(동작 유지).

### 제외(이번 CP 금지/보류)
- **Supabase 보류:** `market_repo.py` 의 Supabase 호출 경로는 구조(except 좁히기)만. 실제 연결/쿼리 변경·실행 금지(Plan v3 — EODHD 로컬 유지, Supabase 추후).
- **HTTPException → AppError 전환 금지**(진단 3). 응답 스키마 변경은 별도 CP.
- pydantic Settings / structlog / async 리팩토링(다른 CP).
- 사용자 직접수정 파일 revert. 운영 parquet 재생성.
- `feature_svc.py` 내부 `_validate_*` 계약(이미 ValueError 던짐)은 그대로 둔다 — 이번 pandera 는 **parquet I/O 경계**용이고 feature 계산 내부 계약과 중복 적용하지 않는다.

---

## Sub-step (Strangler Fig, 작은 단위)

각 Step = 한 revert 단위. 각 Step 끝에 commit + 명시된 검증을 통과해야 다음 Step으로 간다.
추출 순서 원칙(순수 → I/O 경계 → 상태 의존)을 따른다: Step 1(순수 응답 정리) → Step 2~3(스키마=순수 정의 + I/O 경계) → Step 4(상태/예외 경계).

### Step 1 — admin 민감정보 제거 (request_id 상관키만 노출, 표준화)

목표: `debug_state` 가 traceback/예외타입/예외메시지를 응답 본문에 싣지 않게 한다. 대신 서버 로그에 `request_id` 와 함께 남기고, 응답에는 `{"status":"error"}` (+ 필요 시 안전한 비민감 라벨만).

1. `admin.py` 상단에 로깅 준비: `import logging` + `logger = logging.getLogger("lens.admin")`. (현재 `traceback` import 는 149행 블록 외 사용처 없음 → 제거 가능.)
2. `:149-155` market probe except 블록을 아래로 교체(옛 코드 제거):
   ```python
   except Exception as exc:  # noqa: BLE001 — debug 엔드포인트, 원인은 로그로만
       rid = getattr(request.state, "request_id", "-")
       logger.warning("[%s] debug-state market probe '%s' 실패", rid, slot, exc_info=exc)
       market_probes[slot] = {"status": "error"}
   ```
3. `:92` 파일 stat except, `:128` 메모리 except 의 `"error": str(exc)` 도 동일 원칙으로: 응답엔 `{"exists": False}` / `{}` 만, 상세는 `logger.warning(..., exc_info=exc)`. (request 객체가 해당 스코프에 있으므로 rid 동일 패턴.)
4. `traceback` import 제거(쓰는 곳이 사라지면). ruff `F401` 로 확인.
5. **검증:**
   - `..\.venv\Scripts\python.exe -m pytest -q` (회귀 0).
   - `..\.venv\Scripts\python.exe -m ruff check app/routers/v1/admin.py` (F401/BLE001 잔여 확인).
   - 백엔드 8123 기동 후 `curl http://127.0.0.1:8123/api/v1/admin/debug-state` → 응답 JSON에 `exc_type` / `exc_msg` / `traceback_tail` 문자열이 **없어야** 한다(정상 경로면 애초에 error 분기 안 탐 — 그래도 grep로 키 부재 확인).
   - **CP223 snapshot 재실행 → diff 0** (debug-state 가 스냅샷 대상이면, error 분기 미발생 시 정상 경로 동일).
6. commit: `refactor(admin): debug-state 응답에서 traceback/예외 상세 제거, request_id 로깅으로 이전`.

### Step 2 — pandera 도입 + 스키마 정의 (순수, I/O 미연결)

목표: 의존성 설치 + 스키마 모듈만 추가. 아직 어떤 read 경로에도 연결하지 않는다(공존 단계).

1. `backend/requirements.txt` 에 핀 추가. **Python 3.10 + pandas 2.2.2 호환 버전**을 설치하고, 실제로 설치된 정확한 버전으로 핀한다:
   ```powershell
   ..\.venv\Scripts\python.exe -m pip install "pandera[pandas]"
   ..\.venv\Scripts\python.exe -c "import pandera; print(pandera.__version__)"
   ```
   설치된 버전을 확인해 `requirements.txt` 에 `pandera==<설치버전>` 한 줄 추가(pandas 위치 근처).
   > pandera 최신 계열은 import 경로가 `import pandera.pandas as pa` (구버전은 `import pandera as pa`)일 수 있다. 설치 후 `python -c "import pandera.pandas"` 가 되는지 확인하고, 되면 그 경로를, 안 되면 `import pandera as pa` 를 쓴다. **어느 쪽을 썼는지 report 에 기록.**
2. `backend/app/schemas/__init__.py` (빈 파일) + `backend/app/schemas/frames.py` 신설. 진단 4의 실측 dtype 그대로 5개 모델 정의. **날짜 컬럼은 str, coerce=True.** 예시(line_1d):
   ```python
   from __future__ import annotations
   import pandera.pandas as pa            # 설치 결과에 맞춰 조정
   from pandera.typing import Series

   class LineDailyFrame(pa.DataFrameModel):
       ticker: Series[str] = pa.Field(coerce=True)
       asof_date: Series[str] = pa.Field(coerce=True)        # ★ datetime 아님 (CP214)
       line_score: Series[float] = pa.Field(coerce=True, nullable=True)
       safe_line_score: Series[float] = pa.Field(coerce=True, nullable=True)
       line_rank_by_date: Series[float] = pa.Field(coerce=True, nullable=True)
       safe_line_rank_by_date: Series[float] = pa.Field(coerce=True, nullable=True)
       model_id: Series[str] = pa.Field(coerce=True, nullable=True)
       source_cp: Series[str] = pa.Field(coerce=True, nullable=True)
       class Config:
           strict = False     # 추가 컬럼(actual_h5_return, *_flag 등) 허용
           coerce = True
   ```
   나머지 4개 모델도 동일 패턴: band_1d(`horizon_step: Series[int]`, `band_lower/upper` float nullable, `forecast_date: Series[str]`), band_1w, MarketPrices1d(`volume: Series[int]`, `date: Series[str]`), MarketIndicators1d(`date: Series[str]`, `regime_label: Series[str] nullable`).
   - `nullable=True` 는 실측에서 NaN 가능한 float/flag 컬럼에 둔다(예: line_score 는 안전망 join 후 NaN 존재). band_lower/upper 는 nullable=True(데이터에 결측 가능).
   - `strict=False` 필수(스키마에 안 적은 컬럼이 disk 에 더 있다 — 위 진단 4 표가 핵심 컬럼만 나열).
3. 검증 헬퍼 한 곳: `frames.py` 에 `def validate(model, df, *, name): return model.validate(df, lazy=True)` 형태로 lazy 검증(모든 위반을 한 번에 수집) + 실패 시 `pandera.errors.SchemaErrors` 그대로 전파(다음 Step에서 경계가 잡는다). 아직 호출자는 없다.
4. **검증:**
   - `..\.venv\Scripts\python.exe -c "from app.schemas import frames"` import 성공(`cd backend`, `PYTHONPATH=.`).
   - 스키마가 **현재 운영 parquet 에서 실제로 통과하는지** 일회성 스크립트로 확인(아래 "검증" 섹션의 스모크 스크립트). **여기서 실패하면 차단 트리거** — 스키마를 데이터에 맞출지, 데이터가 문제인지 판단 필요하므로 멈추고 보고.
   - `..\.venv\Scripts\python.exe -m mypy app/schemas/frames.py` (신규 파일 mypy error 0 추가).
5. commit: `feat(schemas): parquet 경계용 pandera DataFrameModel 5종 추가 (미연결)`.

### Step 3 — 읽기/머지 경계에 검증 연결 (I/O 경계)

목표: Step 2 스키마를 실제 read 직후·머지 직전/직후에 끼운다. 옛 코드(read) 옆에 새 코드(validate) 추가 → 검증 통과 확인 → 안정화.

1. `parquet_store._load` (`:60-70`): `df = pd.read_parquet(path)` 직후, `_compress_strings` **이전**에 슬롯명→모델 매핑으로 검증 추가. 매핑: `line_1d→LineDailyFrame`, `band_1d→Band1dFrame`, `band_1w→Band1wFrame`. (검증은 dtype coerce 결과를 반영하되, `_compress_strings` 가 categorical 변환을 하므로 **검증을 compress 앞**에 둬서 categorical 비교 충돌을 피한다.)
   ```python
   df = pd.read_parquet(path)
   df = _validate_slot(name, df)   # 신규: 모르는 슬롯이면 그대로 통과
   df = _compress_strings(df)
   ```
   `_validate_slot` 은 매핑에 없으면 df 그대로 반환(예: 신규 슬롯). 검증 실패 시 `SchemaErrors` 를 그대로 올린다 — `_load` 호출자(`get_raw`→`require`)는 기존에도 예외를 전파하므로 동작 일관.
2. `local_market_svc._load` (`:25-33`): `df = pd.read_parquet(path)` 직후 파일명→모델(market_prices_1d / market_indicators_1d) 검증. 단 이 함수는 직후 `date` 를 `strftime("%Y-%m-%d")` 문자열로 만들므로(`:31-32`), **검증은 read 직후·strftime 이전**, 그리고 스키마의 `date` 는 str(coerce) 이므로 충돌 없음.
3. `strategy_backtest_svc._load_frame` (`:126-223`): 이미 `parquet_store.get_raw`(=Step3-1에서 검증됨)와 `pd.read_parquet(market_*)`(=Step3-2 경유 아님, 직접 읽음 `:128,154`) 를 섞어 쓴다. 머지(`:196-206`) **직전** 4개 프레임(price/indicators/line/band)이 `_align_date_dtype` 후 `date` 가 `datetime64[ns]` 임을 **assert 한 줄**로 박는다(여기서는 pandera 가 아니라 가벼운 dtype assert — 머지 키는 datetime이어야 하므로). 머지 **직후** frame 의 `ticker`/`date` 가 의도 dtype인지 동일 assert.
   ```python
   # 머지 직전 (:191-194 _align_date_dtype 다음)
   for _nm, _f in (("price", price), ("indicators", indicators), ("line", line), ("band", band)):
       assert pd.api.types.is_datetime64_any_dtype(_f["date"]), f"{_nm}.date not datetime before merge"
   ```
   > 주의: `market_prices_1d` 를 직접 읽는 `:128` 경로에는 pandera 를 끼우지 않는다(`_load_frame` 은 자체 정규화 로직이 많아 read 직후 dtype 이 disk 와 다름). 대신 위 머지 직전 assert 가 dtype 계약을 지킨다. local_market_svc 경유가 아닌 이 직접 읽기에 스키마를 강제하려면 별도 판단 필요 — 이번엔 assert 로만.
4. **검증:**
   - 백엔드 8123 기동 → 다음 정상 경로가 200 이고 응답 동일:
     - `GET /api/v1/predictions/line/AAPL`
     - `GET /api/v1/predictions/band/1d/AAPL`
     - `GET /api/v1/strategies/scan/ai_balance_v2` (실제 라우트는 `strategies.router` 확인 후 사용)
   - `..\.venv\Scripts\python.exe -m pytest -q` (회귀 0).
   - **CP223 snapshot 재실행 → diff 0.** 검증을 끼웠는데 정상 경로 출력이 1바이트라도 바뀌면 차단(아래 트리거).
5. commit: `feat(boundary): parquet 읽기/머지 경계에 pandera+dtype assert 검증 적용`.

### Step 4 — `except Exception` 좁히기 (상태/예외 경계)

목표: 재발생 경로를 구체 예외 + 최소 로깅으로. 동작(응답 코드/메시지)은 보존.

1. `market_repo.py:142`, `:211`, `:366` (3개 재발생 경로): `except Exception as exc:` 직전에 좁힐 수 있으면 좁히고, Supabase 클라이언트 예외 타입이 불명확하면 **`except Exception` 유지하되 `logger.warning("[...] ... 실패", exc_info=exc)` 한 줄 추가 후 `raise UpstreamUnavailableError(...) from exc`**. (Supabase 호출 자체는 건드리지 않음 — 로깅만 추가.) 모듈 상단에 `logger = logging.getLogger(__name__)` 는 이미 있음(`:14`).
2. `market_repo.py:245`(volume best-effort), `:359`(stock 1차 폴백): 삼킴 유지하되 `logger.debug("... best-effort 실패, 폴백 진행", exc_info=exc)` + 삼킴 사유 주석. 동작 변경 없음.
3. `strategy_backtest_svc.py:230`(_sector_map): `except Exception:` → `except (FileNotFoundError, OSError, pyarrow.lib.ArrowInvalid):` 로 좁히기 가능하면 좁히고, 어려우면 `except Exception` 유지 + `logger.debug` 추가. (sector 없으면 Unknown 으로 동작 — 유지.) `strategy_backtest_svc` 에 logger 없으면 추가.
4. `predictions.py:39`(load_caches): 삼킴 유지(슬롯별 status 반환이 목적). `logger.warning` 추가.
5. numpy `.item()` 폴백(`local_market_svc:117`, `predictions.py:81`)은 **그대로 둔다**(정당한 직렬화 폴백, 좁히면 오히려 취약). report 에 "의도적 유지" 명시.
6. **검증:**
   - `..\.venv\Scripts\python.exe -m pytest -q` (회귀 0).
   - `..\.venv\Scripts\python.exe -m ruff check app/repositories/market_repo.py app/services/strategy_backtest_svc.py` (BLE001 개수 **감소** 확인 — 좁힌 만큼).
   - **CP223 snapshot diff 0** (예외 경로는 정상 응답을 안 바꾼다).
7. commit: `refactor(errors): 재발생 except 경로에 로깅 추가 + 일부 구체 예외로 좁힘`.

---

## 인터페이스 보존

- **함수 signature 불변:** `parquet_store.get_raw/require/_load`, `local_market_svc.get_*`, `strategy_backtest_svc._load_frame`, `admin.debug_state` 의 시그니처/반환 타입을 바꾸지 않는다. (`_validate_slot`/`validate` 는 내부 신규 helper.)
- **API 응답 schema 불변:** 정상(2xx) 응답 본문은 CP223 스냅샷과 byte-identical 이어야 한다. `debug-state` 는 error 분기에서만 키가 줄어든다(`exc_type`/`exc_msg`/`traceback_tail` 삭제) — 이는 **의도된 보안 변경**이며, 정상 분기 응답은 불변. CP223 스냅샷이 debug-state 의 error 분기를 캡처하지 않는다면 영향 없음.
- **HTTPException 응답 불변:** 진단 3의 비표준 `{"detail":...}` 응답은 **이번 CP에서 건드리지 않는다.** 만약 작업 중 이걸 바꿔야만 진행되는 상황이 오면 → 호출자(프론트 fetch) 영향 분석을 첨부해 **차단 보고**.
- pandera coerce 가 정상 경로에서 dtype 을 바꾸면 안 된다: 스키마는 disk dtype을 그대로 모델링(특히 날짜=str). coerce 결과가 기존 `_compress_strings`/`_align_date_dtype` 입력과 동일해야 한다.

---

## 성공 기준 (측정 가능)

| 항목 | 기준 |
|---|---|
| admin 응답 민감정보 | `debug-state` 응답 본문에 `exc_type` / `exc_msg` / `traceback_tail` 키 **0개** |
| pandera 검증 | 5개 스키마가 현재 운영 parquet 5개에서 **전부 통과**(스모크 스크립트 exit 0) |
| snapshot diff | CP223 스냅샷 재실행 **diff 0** (정상 경로 응답 무변경) |
| pytest | 기존 수집 테스트 **전부 통과, 회귀 0** (CP223 포함) |
| ruff BLE001 | `except Exception`(BLE001) 경고 수가 작업 전보다 **감소**(최소 재발생 경로 3개에 로깅·일부 좁힘) |
| mypy | 신규 파일(`schemas/frames.py`) 기준 **error 0 추가** |
| 예상 시간 | 약 **3~4시간** (스키마 설계·dtype 대조에 시간 배분) |

> tsc / screenshot 항목은 이 CP에 해당 없음(FE 무변경) → 생략.

---

## 검증

작업 디렉토리 가정: `C:\Users\user\lens\backend` (PowerShell). 인터프리터 `..\.venv\Scripts\python.exe`.

### A. 단위/회귀
```powershell
cd C:\Users\user\lens\backend
..\.venv\Scripts\python.exe -m pytest -q          # 기대: 전부 pass, 회귀 0
..\.venv\Scripts\python.exe -m ruff check app     # 기대: BLE001 감소, 신규 F401 없음
..\.venv\Scripts\python.exe -m mypy app\schemas\frames.py   # 기대: error 0
```

### B. pandera 스모크 (Step 2 직후 필수 — 여기서 실패하면 차단)
일회성 스크립트로 운영 parquet 5개를 스키마로 검증한다(파일 만들지 말고 `-c` 또는 임시 스크립트로 실행, 커밋하지 않음):
```powershell
cd C:\Users\user\lens\backend
$env:PYTHONPATH="."
..\.venv\Scripts\python.exe -c "import pandas as pd, pathlib; import app.schemas.frames as F; base=pathlib.Path('data/v1'); [print(n,'OK', F.validate(m, pd.read_parquet(base/f), name=n).shape) for n,f,m in [('line_1d','predictions_line_1d.parquet',F.LineDailyFrame),('band_1d','predictions_band_1d.parquet',F.Band1dFrame),('band_1w','predictions_band_1w.parquet',F.Band1wFrame),('prices','market_prices_1d.parquet',F.MarketPrices1d),('ind','market_indicators_1d.parquet',F.MarketIndicators1d)]]"
```
기대: 5줄 모두 `OK (행,열)`. 어느 한 줄이라도 `SchemaError(s)` → **즉시 중단, 위반 컬럼/사유를 그대로 복사해 보고.**

### C. 정상 경로 응답 (Step 1·3 후)
```powershell
cd C:\Users\user\lens\backend
Start-Process -NoNewWindow ..\.venv\Scripts\python.exe "-m uvicorn app.main:app --host 127.0.0.1 --port 8123"
# 기동 후:
curl.exe http://127.0.0.1:8123/api/v1/admin/debug-state          # exc_type/traceback_tail 키 없어야
curl.exe http://127.0.0.1:8123/api/v1/predictions/line/AAPL      # 200, 데이터 동일
curl.exe http://127.0.0.1:8123/api/v1/predictions/band/1d/AAPL   # 200
```
(기동 프로세스는 검증 후 종료. 포트 8123 점유 시 다른 비표준 포트.)

### D. CP223 스냅샷 (각 Step 커밋 전)
CP223 이 정의한 스냅샷 테스트를 재실행한다(정확한 경로/명령은 CP223 산출물 참조, 예: `pytest -q backend/tests/test_cp223_*snapshot*`). 기대: **diff 0**. diff 발생 시 아래 차단.

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **pandera 스키마가 현재 운영 parquet 에서 검증 실패**(검증 B 스모크 또는 Step3 read 경계에서 `SchemaError(s)`). → 실제 데이터가 스키마와 불일치. **스키마를 데이터에 맞출지 / 데이터(parquet)가 잘못됐는지** 판단이 필요하므로 위반 컬럼·dtype·샘플 값을 정리해 보고. (특히 날짜 컬럼이 str 가 아니거나, float 컬럼에 예상 못한 object 가 섞인 경우 = CP214 류 사고 징후.)
2. **CP223 snapshot diff ≠ 0** (정상 경로 응답이 바뀜). → pandera coerce 가 dtype/값을 바꿨거나 admin 정리가 정상 응답을 건드림 = 동작 변경. 어느 Step에서 깨졌는지와 diff 를 첨부해 중단.
3. **CP223 가 그린이 아님**(선행 미충족) 또는 pytest 가 수집/실행 불가(안전망 부재). → 시작 전 보고.
4. **pandera 설치 실패 / pandas 2.2.2·Python 3.10 비호환**(import 에러, `pandera.pandas` 경로 부재 + `pandera` 단독도 실패). → 버전 충돌. 설치 로그 첨부해 중단(임의로 pandas 업그레이드 금지).
5. **기존 테스트 다수 실패**(작업 전 그린이던 테스트가 깨짐). → 회귀. 중단.
6. **HTTPException → AppError 를 바꿔야만 진행 가능한 상황**(진단 3). 응답 스키마 변경은 범위 밖 → 호출자 영향 분석 첨부해 중단.
7. **백엔드 기동 실패**(환경변수 누락 등으로 `/api/v1/admin/debug-state` 조차 안 뜸). → 원인(누락 env/포트 충돌) 정리해 보고.
8. **except 좁히기가 동작을 바꿈**(좁힌 예외 타입이 실제로는 더 넓게 잡히던 케이스를 놓쳐 응답 코드가 달라짐). 확신 없으면 좁히지 말고 `except Exception` + 로깅 유지하고 그 사실을 보고.

---

## ADR

완료 후 `docs/adr/0017-pandera-boundary-validation.md` 1장(200~300단어) 작성.
- `docs/adr/` 디렉토리는 **현재 없음** → 생성한다.
- 기록할 것: (맥락) CP214 datetime64 vs category 머지 사고와 안전망 부재. (결정) parquet I/O 경계(읽기 직후 + 머지 직전/직후)에 pandera `DataFrameModel` + dtype assert 를 둔다. (핵심 제약) 날짜 컬럼을 str(coerce)로 모델링해 `_compress_strings` 의 CP214 fix 와 충돌하지 않게 한 이유. (대안) feature 계산 내부 ValueError 계약/snaptol 스냅샷과의 역할 분담(경계=구조 dtype, 스냅샷=수치 출력). (결과) dtype 사고를 읽는 순간 차단, 정상 경로 무변경.

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다.

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ____ (EODHD 로컬 유지·Supabase 보류·fidelity 우선 위배 없는지)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ____ (레이어 경계: 라우터가 서비스 공개 인터페이스만 쓰는지, 스키마 모듈 위치·의존 방향)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ____ (예측/밴드/전략 수치 출력 무변경 — snapshot diff 0 로 증명했는지)

---

## 산출물

- 변경 파일(예상):
  - `backend/app/routers/v1/admin.py` (민감정보 제거)
  - `backend/app/schemas/__init__.py`, `backend/app/schemas/frames.py` (신규)
  - `backend/app/services/parquet_store.py` (read 경계 검증)
  - `backend/app/services/local_market_svc.py` (read 경계 검증)
  - `backend/app/services/strategy_backtest_svc.py` (머지 경계 assert + except 로깅)
  - `backend/app/repositories/market_repo.py` (except 로깅/좁히기)
  - `backend/app/routers/v1/predictions.py` (load_caches except 로깅)
  - `backend/requirements.txt` (pandera 핀)
  - `docs/adr/0017-pandera-boundary-validation.md` (신규)
- `docs/cp227_report.md` — 요구 / 한 일(Step별) / 결정(pandera import 경로·설치 버전, 좁힌 except 목록, 유지한 삼킴) / 후속(HTTPException 표준화 별도 CP, Supabase except 정리는 Supabase 도입 CP에서). 필요한 만큼만.
