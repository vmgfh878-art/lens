# ADR-0029: Dependency Vulnerability Audit Policy

Status: Accepted
Date: 2026-06-08
CP: CP238 (OWASP A06)

## Context

v1 운영 진입 후 backend (FastAPI / pandas / pyarrow / starlette transitive) 및
frontend (Next.js / React / axios / postcss) 의존성에 알려진 CVE 점검을 한 번도
한 적 없음. OWASP Top 10 A06 무방비. CP238 에서 audit + fix + CI 게이트 + 정책
박는다.

## Decision

### 1. 도구 선정

| 영역 | 도구 | 이유 |
|---|---|---|
| backend | `pip-audit==2.7.3` | PyPA 공식, PyPI advisory DB 직접 조회. `--strict` + `--ignore-vuln <ID>` 로 정밀 게이트. |
| frontend | `audit-ci` (latest) | IBM 메인, JSON schema 기반 allowlist 표준. npm audit 직접보다 CI 게이트 정밀화 + allowlist 정직성. |

기각 대안:
- `safety` (backend): commercial DB, free tier 제약.
- `bandit` (backend): 소스 분석이라 의존성 CVE 아님 — 별도 트랙.
- `npm audit --audit-level=high` 단독 (frontend): allowlist 박을 곳 없음. PR 마다 같은
  CVE 가 noise. audit-ci 이 통합 정책 박기 좋음.
- `better-npm-audit` (frontend): 활동 적음, audit-ci 가 표준.

### 2. CI 게이트 임계값

- **backend**: `pip-audit -r backend/requirements.txt --strict --ignore-vuln <ID> ...`
  → 모든 vuln 차단, ignore 명시된 것만 통과.
- **frontend**: `npx audit-ci --config audit-ci.json` (`"high": true`) → high+ 만 차단,
  moderate/low 는 noise 라 통과. allowlist 의 GHSA ID 는 high 격상 시도 통과.

### 3. 예외 처리 정책 — 2 카테고리

#### Justified (영향 0)

- Lens 사용 패턴 grep 0 hit 확인 — `docs/cp238_grep_evidence.md` 박제.
- 영향 0 이라 ignore. 회귀 없이 운영 진입 가능.

#### Acknowledged (잠재 영향 인정)

- 사용 패턴 있음 + Lens v1 mitigating factor 명시.
- v2 mitigation 트랙 commitment 박힘.
- 재검토 주기 단축 (3개월).
- 사용자 직접 acknowledgement 서명 — 정직성의 lock.

모든 ignore / allowlist 박힌 CVE 는 `docs/security/cve_exceptions.md` 에 공격 벡터 +
grep 명령 + 영향 결론 + fix 가능 시점 + 재검토 주기 박제.

### 4. Backend policy 적용 (CP238 Step 1-3)

bump:
- `fastapi`: 0.111.0 → 0.115.6 (starlette transitive 0.37.2 → 0.41.3)
- `python-dotenv`: 1.0.1 → 1.2.2
- `pyarrow`: 16.1.0 → 17.0.0

ignore (모두 justified — `docs/security/cve_exceptions.md §Backend` + grep_evidence
§Backend):
- PYSEC-2026-113 (pyarrow Arrow C++ UAF, IPC pre-buffering) — 내부 데이터만 read
- PYSEC-2026-161 (starlette host header injection) — `backend/app` grep 0 hit
- GHSA-2c2j-9gv5-cj73 (starlette multipart sync block) — multipart 0 hit
- GHSA-7f5h-v6xp-fcq8 (starlette FileResponse Range DoS) — file serving 0 hit

CI step:
```yaml
- name: pip-audit (CP238 — backend dep CVE gate)
  env:
    PYTHONUTF8: '1'
  run: |
    pip install pip-audit==2.7.3
    pip-audit -r backend/requirements.txt --strict \
      --ignore-vuln PYSEC-2026-113 \
      --ignore-vuln PYSEC-2026-161 \
      --ignore-vuln GHSA-2c2j-9gv5-cj73 \
      --ignore-vuln GHSA-7f5h-v6xp-fcq8
```

재검토: 6개월 (다음 2026-12-07).

### 5. Frontend CVE allowlist policy (CP238 Step 4-6)

bump:
- `axios`: 1.7.x → 1.17.0 (transitive only, `npm audit fix` 자동, 21 advisories fix)
- `next`: 14.2.3 → 14.2.35 (수동 package.json 수정, 1 critical → high 격하)

allowlist (`frontend/audit-ci.json`): **15 GHSA** — 8 justified + 7 acknowledged.

#### 8 Justified
- Image Optimization 3개 (GHSA-9g9p-9gw9-jx7f, GHSA-3x4c-7xq6-9pq8, GHSA-h64f-5h5j-jqjh)
  — next/image grep 0 hit
- WebSocket SSRF (GHSA-c4j6-fc7j-m34r) — WebSocket grep 0 hit
- Pages Router i18n bypass (GHSA-36qx-fr4f-26g5) — Pages Router 미사용 + i18n 미설정
- CSP nonces XSS (GHSA-ffhc-5mcf-pf4q) — CSP nonces 미사용
- beforeInteractive Script XSS (GHSA-gx5p-jg67-6x7h) — next/script 미사용
- postcss XSS (GHSA-qx2v-qp2m-jg93) — devDep build-time only

#### 7 Acknowledged (rewrites + RSC)
- rewrites HTTP smuggling (GHSA-ggv3-7p47-pfv8) — same-origin backend proxy 사용
- Middleware/Proxy redirects cache poisoning (GHSA-3g8h-86w9-wvmq) — proxy 부분
- RSC DoS/cache poisoning 5개 (GHSA-h25m-26qc-wcjf, GHSA-q4gf-8mx6-v5v3, GHSA-8h8q-6873-q5fj,
  GHSA-vfv6-92ff-j949, GHSA-wfc6-r584-vfw7) — App Router default RSC

Mitigation factor (v1):
- Lens 는 인증 없음 → cross-user impact 0
- 공개 read-only 데이터만 → 유출 손실 0
- ISR/revalidate/generateStaticParams 0 hit (F8) → cache 거의 없음 → cache poisoning 실효 매우 낮음
- experimental.staleTimes 0 hit (F9) → cache 정책 default
- backend Pydantic input validation (CP241 예정) → invalid payload reject

v2 mitigation: **next 16 major bump 동시 진행** — Supabase Auth + production 본격화
트랙 진입 시점에 강제. RSC + React 19 호환 재검증 + Auth 도입으로 attack surface
증가하므로 acknowledged 7 정리 timing 일치.

CI step:
```yaml
- name: audit-ci (CP238 — high+ with documented allowlist)
  run: npx audit-ci --config audit-ci.json
```

재검토: 3개월 (단축, 다음 2026-09-08).

### 6. Acknowledged 진입 정직성 lock

`docs/security/cve_exceptions.md` 의 §B 끝에 사용자 acknowledgement 서명 placeholder
박힘. CP238 closure 직전 사용자 직접 채워서 commit. v2 mitigation commitment 도 같이
박음 (next 16 major bump 동시).

이게 "그냥 ignore 박고 잊어" 패턴 차단의 lock. 정책상 acknowledged advisory 는 항상
사용자 서명 + commitment 의 짝.

**CP238 closure 결과**: 사용자 김지형 (2026-06-08) 본인 서명 박힘 (commit
ccc82c2). agent 세 번 검수 후 박음. browser cache 영향 1건 보강 추가.
v2 mitigation commitment: Supabase Auth + production 본격화 트랙 진입 시
next 16 major bump 동시 진행 + RSC + React 19 호환 재검증.

### 7. Dependabot alerts

GitHub Settings → Security → Code security and analysis:
- **Dependabot alerts**: enabled
- **Dependabot security updates**: enabled (자동 PR)
- **Dependabot version updates**: 보류 (PR noise 우려)

CP238 Step 8 에서 사용자 직접 GitHub UI 작업 (R3).

**CP238 closure 결과**: 사용자 응답 `"238 8 done (or skip)"`. GitHub UI 외부라
agent 가 직접 상태 확인 불가. 사용자 본인 결정 박힘. 보고서 §Step 8 참조.
미활성 상태로 운영 진입 시 GitHub Advisory DB 의 push-time alert 못 받음 —
CI workflow 의 `pip-audit` + `audit-ci` 게이트 가 사실상 대체 (단 신규 CVE
출시 시 PR 깨질 때까지 모름, Dependabot 활성 시 즉시 알람).

### 8. 재발 차단

CI workflow 의 audit step (backend pip-audit + frontend audit-ci) 이 PR/push 마다
자동 검사. 새 의존성 추가 / 기존 의존성 bump 시 CVE 통과 못 하면 머지 차단.

신규 CVE 발견 시 흐름:
1. CI RED → PR 작성자가 CVE 확인
2. fix 가능: bump
3. fix 불가 (transitive / breaking): `docs/security/cve_exceptions.md` 에 grep 근거 + justification/acknowledgement 박은 후 `audit-ci.json` allowlist 갱신 (또는 `pip-audit --ignore-vuln`)
4. acknowledged 라면 사용자 서명 알람

## Consequences

장점:
- OWASP A06 무방비 → 대응. 자동 게이트 + 정책 명문화 + 정직성 lock
- audit-ci allowlist 가 JSON schema 기반 → 새 advisory ID 발견 시 즉시 검출

단점 / Trade-off:
- 신규 의존성 추가 시 audit 통과 필수 → PR 속도 약간 저하
- 의존성 bump 잦아짐 → 회귀 안전망 (CP223 snapshot + CP230 smoke) 부담 증가, 단 자동
- acknowledged 7 (frontend) 가 v2 까지 잠재 — Vercel quota / DoS risk 인정한 상태로 운영
- pip 21.2.3 (venv 의 pip) 가 오래됨 → `pip-audit --dry-run` 같은 기능 못 씀. pip
  upgrade 는 별도 cleanup CP (운영 영향 확인 필요)

## v2 진입 조건

다음 트랙에서 본 ADR 재검토:
- Supabase Auth 도입 = 인증 도입 = attack surface 증가
- production 본격화 시 → next 16 major bump 동시 진행
- 그 시점에 `docs/security/cve_exceptions.md §Frontend §B` 의 7 acknowledged 모두 정리
- backend acknowledged 4 도 fastapi 가 starlette 1.x 호환 minor 출시 시 정리

## References

- OWASP Top 10 A06 (https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/)
- pip-audit (https://github.com/pypa/pip-audit)
- audit-ci (https://github.com/IBM/audit-ci)
- npm audit docs (https://docs.npmjs.com/cli/v10/commands/npm-audit)
- OSV.dev (https://osv.dev)
- GitHub Advisory DB (https://github.com/advisories)
- 본 트랙 보고서: `docs/cp238_dependency_audit_report.md`
- 사용 패턴 grep 근거: `docs/cp238_grep_evidence.md`
- CVE 예외 박제: `docs/security/cve_exceptions.md`
