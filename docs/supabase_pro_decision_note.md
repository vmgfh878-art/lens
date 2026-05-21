# Supabase Pro 결제 판단 메모

생성일: 2026-05-04

## 결론

지금 즉시 Supabase Pro를 선결제하지 않는다.

권장 전략:
- Free 유지
- egress/read-only 방화벽 유지
- local parquet 기반 제품 루프 유지
- 402 또는 read-only가 실제로 발생하거나 발표/시연 복구 시간이 더 중요한 순간에 Pro 1개월 결제
- Pro 전환 시 Spend Cap은 켠 상태로 시작

## 현재 Lens 상황

CP98~CP101 결과:
- yfinance/local parquet 기반 1D product loop 동작
- Supabase `price_data`/`indicators` 대량 read 차단
- Supabase에는 latest-only product prediction만 얇게 저장
- 전체 yfinance write, indicators full recompute, full model training 없음

따라서 Supabase를 전체 데이터 창고로 쓰기 위해 Pro를 결제하는 방향이 아니라, Supabase를 제품 thin DB로 줄여 Free 복귀 가능한 구조를 목표로 한다.

## 공식 문서 기준 비용/제한

2026-05-04 확인한 Supabase 공식 문서 기준:

- Free Plan egress: 5GB
- Pro/Team egress: 250GB included, 이후 GB당 과금
- Free Plan Database size: 500MB per project
- Pro Plan disk: project당 8GB included, 초과분 과금
- Spend Cap은 Pro Plan에서 사용 가능하며, quota 초과 시 추가 과금 대신 사용 제한을 걸 수 있음

참고:
- Supabase Billing: https://supabase.com/docs/guides/platform/billing-on-supabase
- Supabase Cost Control / Spend Cap: https://supabase.com/docs/guides/platform/cost-control
- Supabase Database and Disk Size: https://supabase.com/docs/guides/platform/database-size
- Supabase Disk Size Usage: https://supabase.com/docs/guides/platform/manage-your-usage/disk-size

주의:
- 기존 문서나 기억의 "Free DB 2GB" 전제는 현재 공식 문서 기준과 다를 수 있다.
- 실제 프로젝트 dashboard Usage 수치를 기준으로 최종 판단해야 한다.

## 지금 결제하지 않는 이유

- CP98~CP101에서 대량 DB read 없이 제품 루프가 동작했다.
- local parquet 구조가 이미 egress를 줄이는 방향으로 전환됐다.
- 최신 제품 저장은 5티커 latest-only 수준이라 DB write payload가 작다.
- Pro를 먼저 켜면 DB slimming과 pruning 압력이 줄어들 수 있다.
- 최종 목표는 Free 운영 가능한 얇은 DB 구조다.

## 402/read-only 발생 시 결제 전략

402 또는 read-only가 실제로 발생하면:

1. Supabase dashboard에서 원인 확인
   - Database API egress
   - Storage egress
   - Realtime egress
   - Database size
2. 즉시 필요한 작업이 제품 thin upload/API 확인인지 판단
3. Free 상태에서 해결 가능하면 pruning/export로 해결
4. 당장 발표/검증이 막히면 Pro 1개월 결제
5. Pro 결제 후에도 full history DB upload 금지
6. 다음 billing cycle 전 Free 복귀 가능하도록 pruning/export 계획 실행

## 개발 단계 1개월 Pro 장점

- egress/read-only 리스크를 단기적으로 낮춤
- 발표 전 장애 복구 시간이 줄어듦
- DB slimming 전에 임시 여유 확보
- Usage 원인 분석 중에도 API 확인 가능

## 개발 단계 1개월 Pro 단점

- 비용이 고정비로 생김
- 문제 원인인 대량 read/write 습관이 다시 숨어버릴 수 있음
- Spend Cap을 끄면 예기치 않은 추가 비용이 생길 수 있음
- Pro에서 정상 동작한다고 Free 구조가 검증되는 것은 아님

## Spend Cap 권장

Pro로 전환한다면 Spend Cap은 켜는 것을 권장한다.

이유:
- 버그나 실수로 대량 read/write가 재발했을 때 비용 폭주를 막는다.
- 현재 Lens 목표는 대량 DB 사용이 아니라 thin DB 유지다.

예외:
- 발표 직전이고 quota 제한 때문에 서비스가 막혔으며, 초과 비용을 감수하겠다는 명확한 판단이 있을 때만 Spend Cap off를 검토한다.

## 최종 목표

최종 목표는 Pro 상시 운영이 아니다.

목표 구조:
- Supabase: `stock_info`, 최신 가격 일부, 최신 예측, 모델 요약, 제품 표시용 최소 지표
- local parquet: 전체 `price_data`, 전체 `indicators`, 학습 feature, 실험 prediction/evaluation history
- yfinance: 개인 로컬 primary
- EODHD: 해지 또는 필요 시 별도 검증용 fallback

## 결제 트리거

Pro 결제 후보:
- 402/read-only가 제품 thin upload 또는 API 확인을 막음
- 발표/시연 일정상 즉시 복구가 필요함
- Free DB 500MB 이하 감축이 당장 불가능함

결제 보류:
- local parquet loop가 유지됨
- 제품 latest-only upload가 정상
- dashboard Usage가 Free 제한 안쪽
- pruning/export를 먼저 실행할 시간 여유가 있음
