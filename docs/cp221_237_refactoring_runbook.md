# Lens 리팩토링 런북 (CP221~CP237)

> **이 문서 하나만 새 세션에 넘기면 된다.** 실행 세션은 이 런북을 읽고 CP221부터 순서대로 각 지시서를 꺼내 실행 → 검증 → 커밋 → 다음. 차단 트리거에 걸리면 즉시 멈추고 보고.
>
> 진단 근거: `docs/refactoring_master_plan.md` (코드 전수조사 4-에이전트 + 현업 베스트프랙티스 서칭 + 시니어 9-Layer 검수 통과).
> 각 CP 상세 지시서: `docs/cp<번호>_*_directive.md`.

---

## 0. 절대 규칙 (먼저 읽어라)

1. **문제 발견 시 그냥 넘어가기 절대 금지.** 각 지시서의 "차단 트리거"에 하나라도 걸리면 **즉시 중단하고 사용자에게 보고**한다. 진행보다 정직이 우선.
2. **안전망 우선.** `CP223`(백엔드 characterization 스냅샷) 그린 전에는 백엔드 구조 분리(CP225~) 금지. `CP230`(프론트 characterization) 그린 전에는 프론트 분리(CP231~) 금지.
3. **한 CP = 한 묶음 커밋.** 리팩토링 커밋과 기능 커밋을 섞지 않는다. CP 안에서도 Sub-step 단위로 커밋(revert 단위 작게).
4. **동작 보존 검증을 매 Step.** 백엔드는 snapshot diff 0, 프론트는 screenshot diff 허용오차 내 + Vitest 통과. diff가 나면 = 의도치 않은 동작/시각 변경 → 차단.
5. **인터페이스 보존.** 함수 signature / API 응답 schema / props 인터페이스를 바꾸지 않는다. 바꿔야 하면 호출자 영향 분석 + 보고.
6. 커밋 메시지 간결 + 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
7. **Supabase 코드는 보류.** 사용자가 다시 살릴 예정 → market_repo 등 Supabase 관련 제거/변경 금지. (CP236a~d는 "재연결 준비 골격"만, 실연결 검증은 사용자 Supabase Pro 결제 후.)
8. **새 학습 / 새 calibration / DB write / 운영 parquet 덮어쓰기 금지.** 리팩토링은 read-path 구조 변경이 전부.

---

## 1. 시작 전 0단계 — working tree 정리

리팩토링 시작 전, 현재 섞여있는 변경분(README + CP220 + § 제거 + v2 방향 + ai_runs_mock + CP221 버그픽스)을 **먼저 커밋**한다. 리팩토링 변경과 섞이면 revert/diff가 지옥이 된다.

```powershell
cd C:\Users\user\lens
git status                      # 섞인 변경분 확인
# 의미 단위로 나눠 커밋 (예: docs/README, frontend CP217~220, backend CP221 버그픽스)
git add backend/app/routers/v1/predictions.py
git commit -m "CP221: fix /predictions/health NameError (_CACHE -> parquet_store.stats)"
# 나머지(README/CP220/§/v2)는 묶어서 또는 분리해서 커밋
```

> **차단**: 사용자가 "섞인 채로 진행" vs "먼저 정리"를 아직 결정 안 했으면, 여기서 멈추고 물어본다.

---

## 2. CP 실행 순서 + 의존성 그래프

```
CP221 (P0 라이브 500 버그픽스) ✅ 완료
  │
  └─ CP222 (안전망 도구: ruff+pytest+coverage+pre-commit+mypy)
       │
       ├─ CP224a (requirements 핀)                  ← CP222 후 독립
       │
       ├─ CP223 (백엔드 characterization 스냅샷) ★안전망
       │     ├─ CP224b (dead code 자동 검출)
       │     ├─ CP225 (feature_svc 591 분리)
       │     ├─ CP226 (strategy_backtest_svc 590 분리)
       │     ├─ CP227 (에러 표준화 + admin 민감정보 + pandera)
       │     ├─ CP228 (structlog + correlation-id)
       │     ├─ CP229 (async blocking + 캐시 안전성)
       │     └─ CP235 (Pydantic Settings)
       │
       └─ CP230 (프론트 characterization: Playwright + Vitest) ★안전망
             ├─ CP231 (BacktestView 1643 분리)
             ├─ CP232 (TrainingView 1560 분리)
             ├─ CP233 (StockView 1097 분리)
             └─ CP234 (globals.css 5208 분리) ⚠️고위험

CP236a~d (Supabase 재연결 준비 골격, 독립, 실검증은 결제 후)
CP237 (CI GitHub Actions) ← 위 모든 안전망/테스트가 baseline. 마지막.
```

**병렬 가능**: CP223 트랙(백엔드)과 CP230 트랙(프론트)은 서로 독립 → 병렬 가능. CP236a~d는 언제든. 단 자동 실행은 **직렬 권장**(한 번에 하나씩 검증하며).

**권장 직렬 순서**: CP222 → CP223 → CP224a → CP224b → CP225 → CP226 → CP227 → CP228 → CP229 → CP230 → CP231 → CP232 → CP233 → CP234 → CP235 → CP236a → CP236b → CP236c → CP236d → CP237.

---

## 3. CP 인덱스 (지시서 경로 + 자동 실행 적합도)

| CP | 지시서 | 트랙 | 자동 적합도 | 비고 |
|---|---|---|---|---|
| CP221 | (완료) | BE 버그 | ✅ 완료 | `/predictions/health` 500 픽스. frontend 호출자 0 확인됨 |
| CP222 | `cp222_safety_net_tooling_directive.md` | 안전망 | ⭐ 매우 적합 | additive. 도구 설치+설정 |
| CP223 | `cp223_backend_characterization_snapshot_directive.md` | BE 안전망 | ⭐ 매우 적합 | additive. snapshot 박제 |
| CP224a | `cp224a_requirements_pin_directive.md` | BE 재현성 | ⭐ 적합 | render 빌드 영향만 주의 |
| CP224b | `cp224b_dead_code_detection_directive.md` | 청소 | ⭐ 적합 | vulture 오탐 사람 확인 |
| CP225 | `cp225_feature_svc_split_directive.md` | BE 분리 | △ 부분 | snapshot diff 차단 |
| CP226 | `cp226_strategy_backtest_svc_split_directive.md` | BE 분리 | △ 부분 | scan/backtest diff 차단 |
| CP227 | `cp227_error_standardization_pandera_directive.md` | BE 안정성 | △ 부분 | 동작 변경 가능, 신중 |
| CP228 | `cp228_structlog_correlation_id_directive.md` | BE 관측성 | △ 부분 | 로깅만, 응답 무변경 |
| CP229 | `cp229_async_blocking_cache_safety_directive.md` | BE 안정성 | △ 부분 | def 전환 schema 보존 |
| CP230 | `cp230_frontend_characterization_directive.md` | FE 안전망 | ⭐ 매우 적합 | additive. baseline 박제 |
| CP231 | `cp231_backtestview_split_directive.md` | FE 분리 | △ 부분 | screenshot diff 차단 |
| CP232 | `cp232_trainingview_split_directive.md` | FE 분리 | △ 부분 | CP216~220 산출물 보존 |
| CP233 | `cp233_stockview_split_directive.md` | FE 분리 | △ 부분 | 차트 오버레이 보존 |
| CP234 | `cp234_globals_css_split_directive.md` | CSS 분리 | ⚠️ **고위험** | cascade 깨짐. 한 영역씩, screenshot diff 0 필수. 의심되면 무조건 멈춤 |
| CP235 | `cp235_pydantic_settings_directive.md` | BE 설정 | △ 부분 | .env.example 동기화 |
| CP236a | `cp236a_supabase_pooling_directive.md` | DB 준비 | △ 부분 | 골격만, 실검증 결제 후 |
| CP236b | `cp236b_supabase_session_directive.md` | DB 준비 | △ 부분 | 골격만 |
| CP236c | `cp236c_supabase_nplus1_directive.md` | DB 준비 | △ 부분 | 가이드만 |
| CP236d | `cp236d_alembic_async_directive.md` | DB 준비 | △ 부분 | 골격만 |
| CP237 | `cp237_ci_github_actions_directive.md` | CI | ⭐ 적합 | 마지막. GPU 테스트 skip 주의 |

---

## 4. 각 CP 실행 프로토콜

각 CP마다 이 절차를 반복한다.

1. **지시서 읽기**: `docs/cp<번호>_*_directive.md` 전체를 읽는다.
2. **선행 의존 확인**: 지시서의 "선행 의존" CP가 그린(통과+커밋)인지 확인. 아니면 멈추고 보고.
3. **Sub-step 실행**: 지시서의 Sub-step을 Step 1, 2, 3... 순서로. 각 Step은 작은 단위(옛 코드 옆 새 코드 공존 → caller 이전 → 옛 제거).
4. **매 Step 검증**:
   - 백엔드: `pytest` 통과(회귀 0) + CP223 snapshot diff 0 + `ruff check` + `mypy`(error 0 추가)
   - 프론트: `npx tsc --noEmit` 0 + Vitest 통과 + Playwright screenshot diff 허용오차 내
5. **Step 커밋**: 검증 통과하면 Step 단위 커밋.
6. **차단 트리거 점검**: 지시서의 "차단 트리거"에 걸리면 즉시 멈추고 §5 양식으로 보고.
7. **CP 완료**: 지시서의 "성공 기준"(측정 가능) 충족 확인 → `docs/cp<번호>_report.md` 작성 → ADR 작성(해당 시) → 다음 CP.

---

## 5. 중간 점검 / 차단 보고 양식

차단 트리거에 걸리거나 판단이 애매하면 **즉시 멈추고** 이 양식으로 보고:

```
## 중단 보고 — CP<번호> Step<N>

[무엇] 어느 Step에서 무슨 검증이 실패했나
[증거] snapshot diff / screenshot diff / 테스트 실패 / 에러 메시지 (구체적으로)
[원인 추정] 왜 그런지 (동작 변경? cascade 깨짐? 데이터 불일치?)
[선택지] (a) ... (b) ... — 사용자 판단이 필요한 갈림길
[롤백 상태] 마지막 그린 커밋 해시 (여기로 되돌릴 수 있음)
```

---

## 6. 진행 상태 추적

각 CP 완료 시 이 표를 갱신(런북 하단 또는 별도 progress 파일):

| CP | 상태 | 커밋 | snapshot/screenshot | 비고 |
|---|---|---|---|---|
| CP221 | ✅ 완료 | (커밋 대기) | — | health 200 로컬 확인 |
| CP222 | ⬜ | | | |
| ... | ⬜ | | | |

---

## 7. 자동 실행 정책 요약

- ⭐ **매우 적합/적합** (CP222, CP223, CP224a, CP224b, CP230, CP237): 자동 진행. additive 또는 명확한 검증.
- △ **부분 적합** (CP225~229, CP231~233, CP235, CP236a~d): Sub-step 단위 자동 진행하되, 각 Step 검증 + 차단 트리거 엄수. 동작/schema 변경 의심되면 멈춤.
- ⚠️ **고위험** (CP234 CSS): 한 영역씩만. 매 영역 screenshot diff 0 확인. 조금이라도 의심되면 무조건 멈추고 보고. 자동 진행의 가장 신중한 지점.

**핵심**: 최대한 자동화하되, 문제 있는데 그냥 넘어가는 것은 절대 허용 안 됨. 의심 = 멈춤 = 보고.
