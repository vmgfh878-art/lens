# ADR-0028: GitHub Actions CI — push/PR 게이트

Status: Accepted
Date: 2026-06-04
CP: CP237

## 결정

`.github/workflows/ci.yml` 단일 워크플로에 backend / frontend 두 잡을 박는다. CP222 (ruff/mypy/pytest) + CP223 (스냅샷) + CP230 (Vitest/Playwright)이 만든 안전망을 모든 push와 main/develop 대상 PR에서 자동 게이트로 측정. **새 검증 로직 발명 0** — 이미 로컬에서 도는 명령을 CI 워크플로로 옮길 뿐.

## 잡 구조

### backend (ubuntu-latest, Python 3.11)

1. checkout
2. setup-python (pip 캐시)
3. `pip install -r backend/requirements.txt` + dev 도구 (requirements-dev.txt 우선)
4. `ruff check .`
5. `ruff format --check .`
6. `mypy backend ai` — `continue-on-error: true` (CP222 baseline 1391 errors pre-existing, 측정만)
7. `pytest backend/tests -q --ignore=test_services.py` — `PYTHONPATH=.:backend`, `LENS_FORCE_LOCAL=1`
8. snapshot 게이트 `pytest -k "snapshot or characterization or cp223"` (별도 출력)

### frontend (ubuntu-latest, Node 20)

1. checkout
2. setup-node (npm 캐시)
3. `npm ci`
4. `npx tsc --noEmit`
5. `npm run test:unit` (Vitest)
6. Playwright 브라우저 캐시 (`~/.cache/ms-playwright`)
7. `npx playwright install --with-deps chromium`
8. `npm run test:e2e` — `continue-on-error: true` (CP230 baseline은 dev server 필요, CI에서 webServer 미정의라 부분 결과 허용 → strict화는 별도 CP)

## GPU (torch cu128) 처리

사용자 명시 "GPU 의존 테스트는 CI에서 skip 마킹". `backend/requirements.txt`에 torch 핀 없음 (로컬은 cu128 별도 설치). `ai/tests/*`는 torch import 필요 → CI에서 collection 깨질 위험.

**해결**: `pytest backend/tests` 만 실행 (ai/tests 제외). 로컬 GPU 환경에서는 `pytest backend/tests ai/tests` 그대로 동작 — `ai/tests/*` 안의 CPU/GPU 분기는 `bootstrap_torch(cpu_only=...)`로 이미 안전.

## mypy pre-commit hook 복원

`.pre-commit-config.yaml`의 mypy hook을 CP224b에서 비활성화했던 것을 복원하되 `stages: [manual]`로 둔다. 사용자 commit 시 자동 실행 안 함 (1391 baseline errors가 모든 commit을 막는 사고 방지). 수동 / CI 경로:

```
pre-commit run --hook-stage manual mypy -a
```

CI의 mypy step은 `continue-on-error: true`로 측정만. strict 적용은 baseline 1391 errors 정리 후 별도 CP.

## 캐싱 (보강6)

- `setup-python@v5 cache: pip` + `cache-dependency-path: backend/requirements.txt`.
- `setup-node@v4 cache: npm` + `cache-dependency-path: frontend/package-lock.json`.
- `actions/cache@v4`로 Playwright 브라우저 디렉토리 (`~/.cache/ms-playwright`) — 매 PR ~3분 절약.

## 트리거

- `push: { branches: ['**'] }` — 모든 브랜치.
- `pull_request: { branches: [main, develop] }` — main/develop 대상 PR만.

branch protection rule (필수 체크 강제)은 GitHub UI 작업. ADR에 안내만.

## additive

테스트/소스/응답 schema/props 인터페이스 0 수정. `backend/requirements.txt` / `frontend/package.json` 기존 내용 무수정. CI는 *읽어* 설치/호출만. CI에서 깨지면 고치지 말고 보고 (사용자 명시).

## 후속 (별도 CP)

1. mypy strict 적용 (baseline 1391 errors 정리 후).
2. Playwright CI strict화 — webServer 정의 + backend mock + 4 screen baseline 재촬영.
3. branch protection rule UI 활성화 (사용자 직접 수행).
4. coverage upload (codecov 등).
