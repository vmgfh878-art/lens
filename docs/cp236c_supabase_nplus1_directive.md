# CP236c DB (Supabase 재연결 준비 — N+1 회피 & from_attributes 함정 가이드) (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 CP 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 내릴 수 있어야 한다.
> **이 CP는 "준비(가이드/주석)" 전용이다. 실 ORM 도입·실 관계 로딩·실연결 N+1 측정은 사용자 Supabase Pro 결제 후 + ORM 전환(CP236d 계열) 이후로 명시 보류한다.**
> **현재 백엔드에는 SQLAlchemy ORM이 없다(`supabase-py` REST 클라이언트만). 따라서 `selectinload`/`joinedload`/`from_attributes`는 "지금 고칠 코드"가 아니라 "장차 repository 작성 시 지킬 규약"이다. 이 CP는 그 규약을 문서·주석으로 박는다.**

---

## 역할 고정

- **모드**: `code` (구현 모드). 코드를 직접 고치고 자가 점검만 보고한다.
- **권한**: 코드 수정 · 로컬 검증(lint/pytest/백엔드 기동/health probe)만.
- **금지**:
  - 새 학습(training) 실행 금지.
  - 새 calibration 실행 금지.
  - DB write 금지 (Supabase·Postgres 어디로도 INSERT/UPDATE/UPSERT 금지).
  - **Supabase 호출 금지** (REST `.execute()` / psycopg2 connect / SQLAlchemy `engine.connect()` 실행 금지). 이 CP는 가이드 문서 + 주석만 박는다.
  - 사용자가 직접 수정한 파일 revert 금지.
  - **새 ORM/SQLAlchemy 의존 추가 금지** (`import sqlalchemy`/`from sqlalchemy.orm import selectinload` 같은 실코드 도입 금지 — 예시는 전부 문서/주석 안 fenced 코드로만). ORM 실도입은 CP236d 영역.
- **자가 점검(필수, 같은 턴에)**: [Plan v3 정합] · [구조 결함] · [모델 영향] 세 축으로 점검해 본문 끝 양식에 PASS/WARN/FAIL + 사유 기재.
- **커밋 메시지**: 간결. 끝에 한 줄:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens` (메인 체크아웃. 워크트리가 아니라 **실제 코드가 사는 main 작업트리**에서 작업한다 — 워크트리 사본의 repositories는 stale 골격일 수 있으니 신뢰하지 말 것).
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). Python 호출은 `.venv\Scripts\python.exe`.
- **백엔드 기동**: `scripts\start_demo.ps1` 또는 직접
  `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (cwd=`backend`, `PYTHONPATH=backend`).
  - `start_demo.ps1`은 `LENS_USE_LOCAL_SNAPSHOTS=1`, `LENS_LOCAL_SNAPSHOT_DIR=...\data\parquet`, `MARKET_DATA_PROVIDER=yfinance`로 띄운다 → **Supabase 없이 로컬 parquet 경로로 동작**. 이 CP 검증은 이 상태를 그대로 쓴다.
  - health: `GET http://127.0.0.1:8000/api/v1/health/live` → 200.
- **프론트**: `npm run dev` (cwd=`frontend`, 포트 3000). **이 CP는 프론트 무관 — 띄울 필요 없음.**
- **검증용 포트 충돌 피하기**: 이미 8000/3000을 점유한 dev 서버가 있으면 새로 띄우지 말고 기존 프로세스의 health를 그대로 probe. 새로 띄울 거면 8000이 비었는지 먼저 확인.
- **이 CP는 본질적으로 문서/주석 작업**이라 백엔드 기동 검증은 "회귀 없음(import smoke + health 200 유지)" 확인 용도로만 쓴다.

---

## 진단 (근거)

조사 출처: 아래 파일들을 본 CP 작성 시 main 작업트리에서 직접 Read/Grep으로 확인. 라인 번호는 확인 시점(2026-06-02) 기준.

### (A) 현재 데이터 접근은 전부 PostgREST(supabase-py REST), ORM 아님
- `backend/app/db.py` (현재 **66줄**): `from supabase import Client, create_client` (L4). `get_supabase()` (L25–35)가 전역 `_client` 캐시 + `create_client(url, key)` (L34). **HTTP 경유 PostgREST 클라이언트**다.
- Grep 결과: `backend/` 전체에 `selectinload` / `joinedload` / `declarative_base` / `from_attributes` **0건**. SQLAlchemy ORM 세션·relationship·lazy load 개념이 **현재 코드에 존재하지 않는다.**
- 결론: 스펙이 말하는 `selectinload`(2차 SELECT) / `joinedload`(JOIN) / `from_attributes` lazy-load 함정은 **전부 SQLAlchemy ORM 전제**다. 지금은 적용 대상 코드가 없다 → **"향후 repository 작성 가이드"**로 문서화하는 게 이 CP의 실체.

### (B) Pydantic 응답 모델은 v2이지만 ORM 객체가 아니라 dict로 채운다 → `from_attributes` 현재 미사용
- `backend/app/schemas/common.py` (L5, L10–28): `ApiResponse[T]`(`data` + `meta`), `MetaResponse`(`model_config = ConfigDict(extra="allow")`, L12). Pydantic v2.
- 라우터: `backend/app/routers/v1/stocks.py` L36/52/75/92, `health.py` L15/30, `ai.py` L197/225/241/257 에서 `response_model=ApiResponse[...]`.
- 그러나 데이터 소스는 **plain dict**다: `market_repo.fetch_price_rows` 등이 `query.execute().data`(list[dict])를 그대로 반환(예: `market_repo.py` L139, L191/208, L282/331). 즉 Pydantic은 **dict에서 validate**하지 ORM 인스턴스에서 attribute를 끌어오지 않는다 → `from_attributes=True`(구 `orm_mode`) 미설정·미사용.
- 결론: **함정은 "from_attributes를 켠 채 unloaded 관계를 응답에 직렬화"할 때만** 터진다. 장차 ORM repository가 생기고 그 ORM 객체를 `model_validate(obj)`로 직렬화하기 시작하면 비로소 위험 구간 진입. **지금은 그 패턴을 도입하지 말라는 가드(주석)를 미리 박는다.**

### (C) "N+1의 PostgREST 아날로그"는 이미 코드에 실재한다 — 가이드의 현실 근거
- `market_repo.py` `_merge_indicator_volume()` (L227–250): 지표 행을 먼저 가져온 뒤(L191) **두 번째 쿼리**로 `price_data`에서 `volume`을 `in_("date", dates)`로 끌어와 병합(L237–244). 이건 **의도된 단일 추가 쿼리(배치 IN)**라 N+1이 아니다 — 오히려 "행마다 쿼리 금지, IN으로 한 방에"의 **모범 사례**다. 가이드에서 이걸 "PostgREST에서 관계 데이터를 한 번 더 가져올 때의 안전 패턴(루프 내 쿼리 금지·배치 IN)"의 실예로 인용한다.
- `_fetch_stock_info_rows` (L280–283), `_fetch_price_ticker_fallback` (L312–332): 단일 select + 후처리 필터(`_filter_stock_rows`). 루프 안에서 행마다 쿼리하는 곳은 **현재 없음** → 현재 코드에 N+1 결함은 없다(가이드는 "이 무결성을 유지하라"는 규약).
- PostgREST는 관계를 `select("*, related(*)")` **embedded resource**로 한 번에 가져온다(= ORM의 eager load에 대응). 가이드는 "REST에서는 embedded select, ORM에서는 selectinload/joinedload"를 **같은 원칙의 두 표현**으로 정리한다.

### (D) repositories 디렉토리 현황 (main 작업트리 기준)
- `backend/app/repositories/__init__.py` — **빈 파일**(1바이트). 가이드 주석을 모듈 docstring으로 박기 좋은 자리.
- `backend/app/repositories/market_repo.py` — **368줄**, REST.
- `backend/app/repositories/ai_repo.py` — **mock 전용**. 파일 상단 docstring에 "v1 에서는 Supabase 가 비활성이라 mock parquet/JSON 만 사용한다."라고 명시 → **현 시점 Supabase 데이터 경로가 사실상 비활성**임을 코드가 증언. 이 CP의 "결제 후 적용" 보류 근거.
- `backend/collector/repositories/base.py` — REST 헬퍼(`fetch_all_rows` 등). 역시 ORM 아님.

### 함정 요약 (가이드가 박을 핵심)
1. **N+1**: 관계 데이터를 "행마다 한 번씩" 가져오면 1+N 쿼리. → ORM은 `selectinload`(관계당 IN으로 2차 SELECT) 또는 `joinedload`(LEFT JOIN). REST는 `select("parent, child(*)")` embedded. **루프 안 쿼리 금지, 미리 eager/embedded.**
2. **from_attributes lazy-load 함정**: Pydantic `model_config = ConfigDict(from_attributes=True)`로 ORM 객체를 직렬화할 때, **unloaded 관계 속성에 접근하면 SQLAlchemy가 lazy load를 트리거**한다. async 세션에서는 이게 **금지된 동기 I/O**(`MissingGreenlet`/`greenlet_spawn has not been called`)로 터진다. → **응답에 필요한 모든 관계는 쿼리 시점에 eager load**해 두고, Pydantic은 이미 로딩된 속성만 읽게 한다. (관계를 응답에 안 넣을 거면 스키마에서 빼라.)

---

## 선행 의존

- **없음**(이 CP는 가이드 문서 + repositories 모듈/스키마에 **주석만** 더하는 additive 작업이며, 백엔드 구조 분리가 아니다).
- 참고 정합:
  - **CP236d(Alembic/async + ORM 골격)** 와 한 쌍이다. CP236c가 "규약(가이드)", CP236d가 "골격(엔진/세션/모델)". 런북 권장 직렬 순서는 `… → CP236a → CP236b → CP236c → CP236d → …`. **CP236c는 CP236d보다 먼저** 실행되어 규약을 먼저 박는 것이 정상.
  - 본 CP의 `snapshot diff 0` 검증은 **CP223(백엔드 characterization 스냅샷)이 그린일 때만** 의미가 있다. CP223 스냅샷이 있으면 그걸로 회귀를 본다. **없으면** 스냅샷 비교는 "해당 없음" 처리하고 §검증 (V1)(V3)(V4) 폴백(import smoke + health + pytest)으로 무회귀를 입증하고 그 사실을 리포트에 명시한다(없는 스냅샷을 새로 만들지 말 것 — CP223 범위 침범).
  - `docs/adr/0013-...md` 는 **CP236a가 생성**한다(ADR-0013: 직접연결 5432). 이 CP는 0013에 섹션을 **append**한다. 0013이 아직 없으면 §ADR 지침대로 생성 후 append.

---

## 범위

### 포함
- **가이드 문서 신규**: `docs/db_repository_guide.md` — (1) N+1 회피(eager/embedded load), (2) `from_attributes` lazy-load 함정 방어. REST 현재형 + ORM 미래형 둘 다, 예시 골격 포함.
- **repositories 모듈 주석**: `backend/app/repositories/__init__.py`(현재 빈 파일)에 **모듈 docstring**으로 "이 패키지에 ORM repository를 추가할 때의 규약" 한 단락 + 가이드 문서 링크.
- **from_attributes 주의 주석**: `backend/app/schemas/common.py` `MetaResponse`/`ApiResponse` 근처에 **주석 한 블록** — "from_attributes를 켜고 ORM 객체를 직렬화한다면 관계는 반드시 eager load. async에서 lazy load = 동기 I/O 폭발." (코드 동작은 불변, 주석만.)
- **ADR append**: `docs/adr/0013-supabase-port-5432-not-6543.md` 하단에 "## N+1 & from_attributes (CP236c)" 섹션 추가(없으면 §ADR대로 생성).
- `docs/cp236c_report.md` 작성.

### 제외 (절대 건드리지 않음)
- **실 ORM 도입 / SQLAlchemy 세션 / relationship 정의 / `selectinload` 실코드** → CP236d + 결제 후. 이 CP에서 실코드 금지(전부 문서·주석 fenced 예시).
- **`market_repo.py` 등 기존 REST 쿼리 로직 변경** → Supabase 보류 원칙(런북 전역 규칙). `_merge_indicator_volume` 등 현 동작 1바이트도 바꾸지 않는다(인용만).
- **Pydantic 모델 동작 변경**: `ConfigDict`에 `from_attributes=True`를 **실제로 켜지 않는다**(주석으로 "켤 때 규약"만). `extra="allow"`(common.py L12) 등 기존 설정 유지.
- `ai_repo.py` mock 경로, `LENS_FORCE_LOCAL`/`LENS_USE_LOCAL_SNAPSHOTS` 로컬 폴백 로직 변경.
- 사용자가 직접 수정한 파일의 revert.

---

## Sub-step (Strangler Fig, 작은 단위)

각 Step = 한 revert 단위. 각 Step 끝에 commit + 검증.
패턴: 이 CP는 실코드 교체가 없으므로 Strangler의 **"규약·가이드 씨앗 심기"** 단계만 수행한다(옛 코드 옆 새 코드 공존·caller 이전·옛 제거 **없음** — 실 repository는 결제 후 작성). 즉 "장차 새 코드가 따를 규약을 옛 코드 옆에 문서/주석으로 먼저 둔다".

### Step 1 — eager load / N+1 회피 가이드 문서 작성
1. `docs/db_repository_guide.md` 신규 작성. 최소 아래 섹션:
   - **머리말**: "현재 Lens는 supabase-py REST만 쓴다(ORM 없음). 이 가이드는 (a) 지금 REST 쿼리를 작성할 때, (b) 결제 후 ORM repository를 도입할 때 **둘 다** 적용되는 데이터 로딩 규약이다."
   - **§1 N+1이란**: 부모 N행 + 행마다 자식 1쿼리 = 1+N. 예시(나쁜 패턴 fenced):
     ```python
     # BAD (개념 예시 — 실제 도입 금지): 행마다 추가 쿼리
     for row in indicator_rows:
         vol = client.table("price_data").select("volume").eq("date", row["date"]).execute().data
     ```
   - **§2 회피 — REST(현재형)**:
     - 배치 IN: 자식을 **한 번에** 가져와 dict로 병합. **실예 인용**: `backend/app/repositories/market_repo.py` `_merge_indicator_volume()` (L227–250) — `in_("date", dates)`로 1쿼리(L242), `volume_by_date` 병합(L248–250). "루프 안 쿼리 금지, IN으로 한 방"의 모범.
     - embedded resource: 관계는 `select("parent_cols, child_table(child_cols)")` 로 한 번에. (PostgREST FK 관계 전제. 실스키마 확정 후 적용.)
   - **§3 회피 — ORM(미래형, 결제 후 CP236d 이후)**:
     ```python
     # selectinload: 관계당 별도 SELECT ... WHERE fk IN (...). 자식 다(多)·중복 적을 때 유리.
     stmt = select(Parent).options(selectinload(Parent.children))
     # joinedload: LEFT OUTER JOIN 한 방. 자식 1:1·소수일 때 유리. 1:N에 쓰면 부모 행 중복.
     stmt = select(Parent).options(joinedload(Parent.child))
     ```
     - 선택 기준 표: 1:N many → `selectinload`, to-one/소수 → `joinedload`. lazy(기본)·`subquery`는 피한다.
   - **§4 골든 룰**: "응답 스키마에 들어가는 관계는 **쿼리 시점에 전부 eager/embedded로 로딩**한다. 직렬화 단계에서 추가 I/O가 일어나면 안 된다." + §2/§3로 백링크.
   - **상태 배너**: 문서 상단에 "상태: 가이드(준비). 실 ORM·실 N+1 측정은 Supabase Pro 결제 + CP236d 이후. — CP236c"
2. `backend/app/repositories/__init__.py`(빈 파일)에 **모듈 docstring** 추가:
   ```python
   """Lens repositories.

   현재: supabase-py REST 전용(ORM 없음). 장차 ORM repository를 추가할 때는
   응답에 필요한 관계를 반드시 eager load(selectinload/joinedload)하고,
   루프 내 행별 쿼리(N+1)를 금지한다. 상세 규약: docs/db_repository_guide.md (CP236c).
   """
   ```
   - 기존 `import`/공개 심볼이 없으므로 docstring 추가는 동작 무영향(빈 파일 → docstring만).
3. **검증**: `ruff check backend/app/repositories/__init__.py`(설정 있으면) + import smoke(§검증 V1) + Markdown 깨짐 없는지 육안. (가이드 문서는 lint 무관.)
4. **commit**: `docs(db): add repository eager-load / N+1 avoidance guide (CP236c)`

### Step 2 — from_attributes lazy-load 주의 주석
1. `backend/app/schemas/common.py` 의 Pydantic 설정 근처(`MetaResponse` 위 또는 `ApiResponse` 위, L10 부근)에 **주석 블록** 추가(코드/동작 불변):
   ```python
   # CP236c: from_attributes(=구 orm_mode) 함정 주의.
   #   현재 응답 data는 plain dict에서 validate되며 from_attributes는 꺼져 있다(유지).
   #   장차 ORM 객체를 직렬화하려고 from_attributes=True 를 켜는 모델을 만들면,
   #   Pydantic이 unloaded 관계 속성에 접근하는 순간 SQLAlchemy lazy load가 트리거된다.
   #   async 세션에서는 이게 금지된 동기 I/O(MissingGreenlet)로 터진다.
   #   => 그런 모델을 쓰려면 쿼리에서 해당 관계를 반드시 eager load 할 것. docs/db_repository_guide.md.
   ```
   - **`ConfigDict(from_attributes=True)`를 실제로 추가하지 말 것.** 주석만. `extra="allow"`(L12) 그대로.
2. (선택, 빈 파일이거나 명백히 안전할 때만) repository `__init__.py` docstring과 가이드 문서에서 이 주석으로 상호 참조 1줄. ORM 스키마 파일이 없으므로 추가 대상 파일은 만들지 않는다.
3. **검증**: `ruff check backend/app/schemas/common.py` + import smoke로 스키마 모듈 정상 로드(§검증 V2). **백엔드 health probe(§검증 V3)로 응답 직렬화 무영향 확인.**
4. **commit**: `docs(schemas): warn on from_attributes lazy-load in async (CP236c)`

### Step 3 — ADR-0013 append + 리포트
1. `docs/adr/0013-supabase-port-5432-not-6543.md` 하단에 `## N+1 & from_attributes (CP236c)` 섹션 추가(§ADR). 0013이 없으면(=CP236a 미실행 순서) §ADR 지침대로 0013을 생성하고 그 안에 본 섹션 포함.
2. `docs/cp236c_report.md` 작성(§산출물).
3. **검증**: `git diff --stat`로 의도한 파일만 변경됐는지(§검증 V0). Markdown 링크 경로 유효성 육안.
4. **commit**: `docs(adr): note N+1 & from_attributes guard in ADR-0013 (CP236c)`

---

## 인터페이스 보존

- **함수 signature 불변**: `market_repo.py`(`fetch_price_rows`, `fetch_indicator_rows`, `fetch_stocks`, `_merge_indicator_volume`, …), `db.py`(`get_supabase` 등), `base.py`(`fetch_all_rows` 등) — **인용만 하고 한 글자도 수정하지 않는다.**
- **API 응답 schema 불변**: `ApiResponse[T]` / `MetaResponse` / `ErrorResponse`(common.py)의 필드·`model_config` 변경 금지. `from_attributes`를 켜지 않음 → 직렬화 입력(dict) 동일 → 응답 바이트 동일.
- **repositories `__init__.py`**: 현재 공개 심볼 0개. docstring 추가는 `import` 시 부작용 없음(re-export 추가 금지).
- **props 인터페이스**: 해당 없음(프론트 무관).
- 만약 가이드/주석을 박으려다 **실제로 스키마 `model_config`를 바꾸거나 repository 코드를 고쳐야만** 하는 상황이 생기면 → **호출자 영향 분석 후 §차단 트리거에 따라 멈추고 보고.** (이 CP에서 그럴 일은 없어야 정상.)

---

## 성공 기준 (측정 가능)

| 항목 | 기준 |
|---|---|
| `docs/db_repository_guide.md` | 신규 1개. §1~§4 + 상태 배너 포함. REST 실예(`_merge_indicator_volume` L227–250) 인용 1건 이상 |
| `backend/app/repositories/__init__.py` 줄 수 | 1바이트(빈) → 약 6~10줄(docstring만). 공개 심볼 추가 0 |
| `backend/app/schemas/common.py` 변경 | 주석 블록 1개만 추가. 코드 라인(필드/`ConfigDict`) 변경 0 |
| 기존 pytest | `backend/tests/` 전부 통과, 회귀 0 (실패 0 추가) |
| snapshot diff | 0 (CP223 스냅샷 존재 시). 미존재 시 "해당 없음" + V1/V3/V4 폴백 통과로 대체 |
| mypy error 추가 | 0 (설정 존재 시. 미존재면 ruff 0 추가) |
| tsc | 해당 없음(프론트 무변경) |
| 로컬 health | `GET /api/v1/health/live` → 200 (변경 전후 동일) |
| 실 N+1 측정 / ORM 검증 | **해당 없음 — Supabase Pro 결제 + CP236d 이후로 보류(리포트에 명시)** |
| 예상 시간 | 약 1~1.5시간 |

---

## 검증

> 모든 명령은 cwd=`C:\Users\user\lens`, venv 파이썬 `.\.venv\Scripts\python.exe` 기준. **Supabase/Postgres 실연결 호출 금지** — 아래는 전부 오프라인 검증.

**(V0) 변경 범위 확인**
```powershell
git diff --stat
# 기대: docs/db_repository_guide.md, backend/app/repositories/__init__.py,
#       backend/app/schemas/common.py, docs/adr/0013-supabase-port-5432-not-6543.md,
#       docs/cp236c_report.md 만. (market_repo.py/db.py/base.py 변경 0)
```

**(V1) repositories 패키지 import smoke (docstring이 패키지를 깨지 않았는가)**
```powershell
$env:PYTHONPATH = "$PWD\backend"
.\.venv\Scripts\python.exe -c "import app.repositories as r; print('repos ok', (r.__doc__ or '')[:24])"
# 기대: repos ok Lens repositories.   (모듈 docstring 앞부분)
```

**(V2) schemas 모듈 import + from_attributes 미설정 확인 (주석이 동작을 안 바꿨는가)**
```powershell
$env:PYTHONPATH = "$PWD\backend"
.\.venv\Scripts\python.exe -c "from app.schemas.common import MetaResponse, ApiResponse; mc=MetaResponse.model_config; print('from_attributes=', mc.get('from_attributes', False), 'extra=', mc.get('extra'))"
# 기대: from_attributes= False extra= allow   (켜지지 않음 + 기존 extra 유지)
```

**(V3) 로컬 서빙 경로 무영향 (Supabase 없이 기존 모드)**
```powershell
# 이미 8000에 dev 서버가 떠 있으면 그걸 probe. 없으면 start_demo.ps1로 띄움.
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health/live" -UseBasicParsing -TimeoutSec 5 | Select-Object -ExpandProperty StatusCode
# 기대: 200 (변경 전후 동일)
# (선택) 종목 검색이 그대로 dict 직렬화되는지 한 건:
# Invoke-WebRequest "http://127.0.0.1:8000/api/v1/stocks?search=AAPL&limit=1" -UseBasicParsing | Select-Object -ExpandProperty StatusCode  # 기대 200
```

**(V4) 기존 테스트 회귀**
```powershell
$env:PYTHONPATH = "$PWD\backend"
.\.venv\Scripts\python.exe -m pytest backend/tests -q
# 기대: 기존과 동일 pass, 실패 0 추가. (이 CP는 문서/주석이라 테스트 영향 0이어야 정상.)
```

**(V5) lint (설정 존재 시)**
```powershell
.\.venv\Scripts\python.exe -m ruff check backend/app/repositories/__init__.py backend/app/schemas/common.py
# 기대: 신규 에러 0. docstring/주석은 코드가 아니므로 미사용 import 경고 없음(import 추가 금지 준수).
```

**(V6) 가이드 문서 자기검증 (REST 실예 인용이 실재하는가)**
```powershell
Select-String -Path backend/app/repositories/market_repo.py -Pattern "_merge_indicator_volume" | Select-Object -First 1
# 기대: 해당 함수가 실제로 존재(가이드가 죽은 줄번호를 인용하지 않았는지 확인). 줄번호 어긋나면 가이드 본문 갱신.
```

---

## 차단 트리거 (중요)

> **다음 상황이면 즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **(V3) 로컬 health가 200이 아니거나 백엔드가 변경 후 기동 실패** = 로컬 서빙 경로 깨짐. 문서/주석 작업이 이걸 깰 리 없으므로, 깨졌다면 무언가 코드를 건드린 것. → 즉시 멈춤.
2. **(V2)에서 `from_attributes`가 True로 나오거나 `extra`가 `allow`가 아니게 됨** = 주석만 박으랬는데 `ConfigDict`를 실제로 건드림(인터페이스 위반). → 멈춤.
3. **(V0) `git diff --stat`에 `market_repo.py`/`db.py`/`base.py` 등 REST 로직 파일이 끼어 있음** = Supabase 보류 원칙 위반(런북 전역 규칙). → 멈추고 되돌림 후 보고.
4. **(V4) 기존 pytest가 변경 전 대비 1개라도 추가 실패** = 문서/주석이 동작 경로로 샜다는 신호. → 멈춤.
5. **snapshot diff 발생**(CP223 스냅샷 존재 시 어느 항목이든 diff ≠ 0) = 동작이 바뀜. → 멈춤.
6. 작업 중 **`import sqlalchemy` / `from sqlalchemy.orm import selectinload` 같은 실코드를 추가해야만** 가이드가 성립하는 상황 = 범위 위반(ORM 실도입은 CP236d). 예시는 **반드시 fenced 코드(문서/주석)** 로만. → 실코드를 넣으려는 순간 멈춤.
7. **실스키마(테이블 FK·관계)가 없어 embedded select / selectinload 예시를 "검증"할 수 없음** = 이건 **멈춤이 아니라 명시 보류**다. 가이드에 "예시 골격, 실스키마 확정·결제 후 적용"이라 명시하고 진행. 단, 검증을 핑계로 **실연결을 시도하면 #1/#3 계열로 멈춤**.
8. `from_attributes` 주의를 박을 **ORM 스키마 파일이 없다고 새로 만들어** 거기에 `from_attributes=True` 모델을 추가하려는 충동 = 범위 위반. 주의는 **기존 `common.py` 주석 + 가이드 문서**로만. → 멈춤.

---

## ADR

완료 후 **`docs/adr/0013-supabase-port-5432-not-6543.md`** 에 **섹션 append**(별도 새 ADR 파일을 만들지 않는다 — 스펙: "0013에 통합").
- 추가 섹션 제목: `## N+1 & from_attributes lazy-load (CP236c)`
- 기록 내용(200~300단어): **컨텍스트**(현재 supabase-py REST만, ORM 없음 → N+1/from_attributes 함정은 미래 ORM 도입 시점에 발현). **결정**(응답에 필요한 관계는 쿼리 시점에 eager/embedded load한다 = 골든 룰; ORM은 `selectinload`(1:N many)·`joinedload`(to-one), REST는 embedded `select("parent, child(*)")`; 루프 내 행별 쿼리 금지, 배치 IN 사용). **from_attributes 가드**(ORM 객체 직렬화용 `from_attributes=True` 모델은 unloaded 관계 접근 시 lazy load → async에서 동기 I/O(MissingGreenlet) → 그런 관계는 반드시 eager load하거나 응답 스키마에서 제외). **대안**(lazy 기본/`subquery` 로딩 — 거부: N+1·async 위험). **결과/상태**(Proposed — 규약·가이드만 확정; 실 ORM repository·실 N+1 측정·embedded select 검증은 Supabase Pro 결제 + CP236d(ORM/async/Alembic) 이후 Accepted 전환). 가이드 본문 링크(`docs/db_repository_guide.md`) 포함.
- `docs/adr/`가 아직 없으면(=CP236a 미실행 순서) 디렉토리 생성 후 0013을 만들고 그 안에 이 섹션을 포함한다.

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다.

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 밴드 본체·fidelity 우선·EODHD 유지·backtest cost 등 Plan v3 핵심을 이 변경이 건드리지 않는가. 본 CP는 데이터 로딩 **규약 문서/주석**이라 모델 수치·데이터 소스 무관해야 PASS.)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 가이드가 현재 코드와 모순되지 않는가(REST 실예 인용이 실재·줄번호 일치), 주석이 동작을 바꾸지 않는가(`from_attributes` 미설정 유지), 규약이 repositories 한 곳 + 가이드 문서로 정본화됐는가.)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 학습/추론/calibration·응답 직렬화 수치에 0 영향. 문서/주석만 → 입력·출력 바이트 불변이어야 PASS.)

---

## 산출물

1. **변경 파일 목록**:
   - `docs/db_repository_guide.md` (신규 — N+1 회피 & from_attributes 가이드)
   - `backend/app/repositories/__init__.py` (모듈 docstring 규약)
   - `backend/app/schemas/common.py` (from_attributes lazy-load 주의 주석)
   - `docs/adr/0013-supabase-port-5432-not-6543.md` (섹션 append)
2. **`docs/cp236c_report.md`** — 요구 / 한일 / 결정 / 후속(필요한 만큼만).
   - **후속에 반드시 명시**: "실 ORM repository 작성·`selectinload`/`joinedload` 실적용·embedded select 검증·실 N+1 측정·`from_attributes=True` 모델 도입은 Supabase Pro 결제 + CP236d(ORM/async/Alembic) 이후 별 CP에서 수행. 본 CP는 규약(가이드)+주석까지."
