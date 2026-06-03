# ADR-0018: structlog + ContextVar based request_id propagation

Status: Accepted
Date: 2026-06-03
Context: CP228 (refactoring runbook CP221~237)

## Context

CP228 진단 전 backend/app 로깅 현황:
- 표준 `logging` + `%`-format. JSON 출력 아님 → 로그 집계 / 검색 어려움.
- backend/app 34 소스 중 로거 보유 파일 3 개 (`main.py`, `repositories/market_repo.py`, `services/parquet_store.py`).
- `services/local_market_svc.py` 등 폴백 경로 로그 침묵.
- `request_id` 는 `request.state.request_id` 에만 살아 service / repo 로그 라인에 박히지 않음 → 어느 요청이 어떤 로드를 유발했는지 추적 불가.

ASGI 단일 이벤트루프 / 코루틴 동시성 때문에 `threading.local()` 은 요청 격리를 보장하지 못한다. ContextVar 기반 전파가 필수.

## Decision

`structlog 25.5.0` 단독 도입. **`asgi-correlation-id` 는 도입하지 않는다** — 자체 `request_id_middleware` 의 헤더 정책 (임의 문자열 허용) 과 충돌 (asgi-correlation-id 의 기본 validator 가 UUID 만 허용 → CP223 snapshot 9 개가 `request_id='test-fixed'` 단언 실패).

설계:
1. `backend/app/core/logging.py` — `configure_logging()` 멱등 함수. shared_processors: `structlog.contextvars.merge_contextvars`, `add_log_level`, `TimeStamper(iso)`, `format_exc_info`. 환경 분기 — `LENS_LOG_JSON=1` 또는 `LENS_ENV=production` 이면 `JSONRenderer`, 그 외 `ConsoleRenderer`. `LENS_LOG_LEVEL` 기본 INFO.
2. stdlib `logging.getLogger` 호출도 `structlog.stdlib.ProcessorFormatter` 의 `foreign_pre_chain=shared_processors` 로 통과 → 같은 포맷 + 같은 `merge_contextvars` (즉 stdlib 로거 라인에도 request_id 자동 머지).
3. `app/middleware/request_id.py` — 자체 미들웨어 유지. 응답 헤더 echo / uuid4 폴백 동작 보존. 추가로 `structlog.contextvars.bind_contextvars(request_id=request_id)` 호출 → ContextVar 에 set 되어 이후 모든 service / repo structlog 로거 라인에 자동 머지.
4. `main.py` / `parquet_store.py` / `market_repo.py` 의 stdlib `logging.getLogger` 를 `structlog.get_logger` 로 전환. `local_market_svc.py` 에 structlog 로거 신설 + 폴백 parquet 미존재 분기에 최소 `logger.warning`.

## Consequences

긍정:
- request_id 가 모든 service / repo 로그 라인에 자동 머지. 운영 중 "어느 요청이 어떤 로드 / 어떤 예외를 유발했나" 가 ContextVar 기반으로 동시 요청 안전하게 추적 가능.
- dev 는 ConsoleRenderer 로 가독성, prod 는 `LENS_LOG_JSON=1` 로 JSON. 같은 코드 같은 포맷터.
- 자체 미들웨어 동작 보존 → CP223 snapshot 9 개 diff 0. test_api 의 `meta.request_id == 'test-fixed'` 단언 통과.
- stdlib logger 호출 형식 (`logger.info("msg %s", x)`) 무변경 → 호출 코드 churn 최소화.

부정 / 미해결:
- structlog `BoundLogger.info(msg, *args)` 는 stdlib 호환 모드로 `*args` 를 `event` 본문에 그대로 출력. `%`-format 결과가 풀리지 않을 수 있음. 향후 구조화된 kwargs 형식 (`logger.info("event_name", key=val)`) 으로 점진 전환 검토.
- `ai/eval/significance/*` (4 로거), `backend/collector/*`, `backend/db/*` 의 로거 전환은 본 CP 범위 밖. 별도 CP.
- Sentry 미설치 → 자동 연동 skip. 설치 시 후속에서 `structlog → sentry_sdk` 처리기 추가 + `transaction_id` 매핑.

대안 및 기각 이유:
- (a) **asgi-correlation-id 도입** — 기본 validator 가 UUID 만 허용 → 자체 미들웨어가 uuid4() 외에도 임의 문자열을 echo 하는 동작 (CP223 테스트의 `'test-fixed'`) 과 충돌. `validator=None` 으로 우회 가능하나 ContextVar 가 자체 미들웨어 안쪽 task 에서 set 되어 자체 미들웨어의 `correlation_id.get()` 이 None 폴백 → 결국 동일 fail. structlog ContextVar 만으로 동일 효과 달성.
- (b) **모든 호출을 structlog kwargs 형식으로 일괄 변환** — 호출 코드 변경량 큼. stdlib bridge 로 호환 유지하면서 점진 전환이 안전.

Fidelity 보장:
- 본 CP 는 로깅 / 관측성만. 응답 schema 1 비트 변경 없음.
- CP223 snapshot 9 endpoint diff 0 (4 step 전체).
- backend/tests 87 passed (CP227 baseline 유지).
