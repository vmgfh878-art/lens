# CP228 보고서 — structlog + request_id ContextVar 전파

## 요구

`docs/cp228_structlog_correlation_id_directive.md`. (1) structlog 도입 + dev/prod 렌더러 분기, (2) request_id 가 service / repo 로그 라인에 ContextVar 기반으로 자동 머지, (3) backend/app 의 3 stdlib 로거 + `local_market_svc.py` 신규 로거를 structlog 로 전환, (4) 응답 schema 1bit 변경 없음 (CP223 snapshot diff 0).

## 한 일

4 step, 4 commit (중간 1 회 차단 트리거 발동 + 자체 해결):

| Step | 커밋 | 작업 | 검증 |
|---|---|---|---|
| 1 | 4021592 | structlog 25.5.0 + asgi-correlation-id 5.0.0 핀 추가 (requirements) | import 가능 |
| 2 | 695f15c (amended) | core/logging.py 신설 — configure_logging() 멱등 함수 | Console/JSON 두 모드 로컬 검증 |
| 3 | d60fd6b | main.py 에 configure_logging 호출 + request_id_middleware 에 structlog bind | snapshot diff 0, 87 passed |
| 4 | 04725d2 | parquet_store / market_repo / local_market_svc 로거 structlog 전환 | snapshot diff 0, 87 passed |

### 차단 트리거 발동 + 해결

Step 3 초안에서 `asgi-correlation-id.CorrelationIdMiddleware` 를 등록하자 CP223 snapshot 9 개 전부 fail (`AssertionError: [stocks_list] request_id leak: '-' assert '-' == 'test-fixed'`).

원인: 라이브러리 기본 validator 가 UUID 만 허용 → 테스트의 `X-Request-Id: test-fixed` 헤더가 거부됨. `validator=None` 으로 우회해도 자체 `request_id_middleware` 가 `correlation_id.get()` 으로 None 을 받아 `'-'` 폴백 (ContextVar 가 자체 미들웨어 안쪽 task 에서 set 되어 자체 미들웨어가 못 읽음).

해결: `asgi-correlation-id` 도입 자체를 포기. 자체 `request_id_middleware` 유지하고 `structlog.contextvars.bind_contextvars(request_id=...)` 만 추가. ContextVar 전파는 structlog 단독으로 충분. 변경 즉시 9 snapshot 다시 통과.

요구사항의 본질 ("request_id 가 service / repo 로그에 자동 머지") 은 structlog ContextVar 만으로 달성. asgi-correlation-id 는 dep 에 남아있으나 import 0, 사용 0 — Step 4 시점에 requirements 에서 제거할지 검토했지만 후속 CP (HTTPException 표준화 / Sentry 도입) 에서 재사용 가능성이 있어 일단 유지.

## 핵심 컴포넌트 존재 체크리스트

- **configure_logging()**: `backend/app/core/logging.py`. 멱등 (여러 번 호출 안전). dev (ConsoleRenderer) / prod (JSONRenderer) 분기. `LENS_LOG_JSON`, `LENS_ENV`, `LENS_LOG_LEVEL` env. shared_processors 에 `structlog.contextvars.merge_contextvars` 필수 (request_id 전파의 핵심).
- **stdlib bridge**: `ProcessorFormatter` 의 `foreign_pre_chain=shared_processors`. 기존 `logging.getLogger` 호출도 같은 포맷 + 같은 ContextVar 머지.
- **request_id_middleware** 동작 보존: 헤더 X-Request-Id echo, uuid4 폴백, `request.state.request_id` set. 추가로 `structlog.contextvars.bind_contextvars(request_id=...)` 호출.
- **structlog 로거 보유 backend/app 소스**: 4 개 (`main.py`, `parquet_store.py`, `market_repo.py`, `local_market_svc.py`). 목표 4 달성.
- **CP223 snapshot 9 endpoint** 통과 (diff 0). `meta.request_id` 가 헤더 값 그대로 echo 됨 (테스트 `'test-fixed'`).
- **응답 헤더 X-Request-Id** 유지 (자체 미들웨어 echo, 키 대소문자 그대로).

## 새 테스트 결과

신규 테스트 0. 로컬 검증:
- `configure_logging()` 호출 후 `structlog.get_logger('t').info('hello', k=1)` → Console / JSON 두 모드 모두 정상 출력 (timestamp + level + event + k=1).
- stdlib `logging.getLogger('legacy').info('legacy line')` → 같은 포맷터로 통과 (foreign_pre_chain 효과).

## dry-run 결과

매 Step 직후:
- `pytest backend/tests/test_characterization_api.py` → 9 passed, diff 0.
- `pytest backend/tests/test_feature_svc.py` → 11 passed.
- `pytest backend/tests --ignore=test_services.py` → 87 passed, 11 failed (전부 pre-existing).

응답 body / schema 1bit 변경 없음. service / repo 로그 라인에 request_id 가 자동 머지될 준비 완료 (실제 stdout 확인은 통합 환경 검증 항목).

## 기존 회귀 통과 건수

- `pytest backend/tests` (test_services.py 제외): **87 passed** — CP227 baseline 그대로.
- `ruff check`: pre-existing E501/I001 baseline 그대로, 신규 위반 0.
- `mypy backend/app/core/logging.py`: error 0.
- import sanity: `import app.main` OK.

## 결정

- **asgi-correlation-id 사용 안 함** — 자체 미들웨어 헤더 정책 (임의 문자열 echo) 과 충돌. structlog ContextVar 만으로 동일 효과. ADR-0018 에 차단 트리거 + 해결 기록.
- 자체 `request_id_middleware` 유지 + structlog bind 추가. 응답 헤더 / `request.state.request_id` / uuid4 폴백 동작 모두 보존.
- 모든 호출 형식 (`logger.info("msg %s", x)`) 무변경 — stdlib bridge 로 호환. kwargs 형식 (`logger.info("event_name", key=val)`) 으로의 점진 전환은 후속 권고.
- `local_market_svc._load` 의 parquet 미존재 분기에만 최소 `logger.warning` 추가. 과한 로그 금지.
- Sentry 미설치 → 자동 연동 skip. ADR 에 후속 연동 안 메모.

## 후속

1. **kwargs 호출 형식 점진 전환**: `logger.info("v1_predictions_cache", slot=slot, info=info)` 등. JSON 모드에서 structured key=value 활용도 극대화.
2. **ai/eval/significance / collector / db 로거 전환** (별도 CP): backend/app 외 4+ 모듈도 structlog 로.
3. **Sentry 설치 + 연동** (별도 CP): structlog → sentry_sdk 처리기 추가, request_id → transaction_id 매핑.
4. **asgi-correlation-id dep 제거 검토**: 본 CP 에서 import 0. HTTPException 표준화 / Sentry 연동 CP 에서 재사용 가능성 검토 후 결정.
