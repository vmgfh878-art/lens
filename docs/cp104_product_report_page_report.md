# CP104-P Lens 설명/리포트 화면 1차 구현 보고서

## 1. 목표

Lens를 처음 보는 사람이 프로젝트의 의도, 데이터 구조, AI 모델 역할, 실패에서 배운 점을 한 화면에서 이해할 수 있도록 `Lens 리포트` 화면을 추가했다.

이번 화면은 모델 학습 콘솔이 아니라 프로젝트 설명 화면이다. “주가를 맞히는 앱”이 아니라 “리스크를 먼저 보는 AI 투자 보조지표”라는 해석 프레임을 전면에 둔다.

## 2. 변경 파일

- `frontend/src/components/ReportView.tsx`
  - 신규 리포트 화면 추가
  - Lens 정체성, AI 지표, 데이터, 모델, 실험에서 배운 점, 현재 상태와 다음 단계 구성
- `frontend/src/components/AppShell.tsx`
  - 사이드바에 `Lens 리포트` 메뉴 추가
  - 깨진 한글 노출 문구를 정상 한국어로 복구
- `frontend/src/components/DashboardPage.tsx`
  - `report` view 라우팅 추가
- `frontend/src/app/globals.css`
  - 리포트 화면 전용 레이아웃과 반응형 스타일 추가
- `docs/cp104_product_report_page_report.md`
  - CP104 구현 보고서 추가

## 3. 화면 구조

1. 첫 화면 요약
   - `리스크를 먼저 보는 AI 투자 보조지표`
   - 투자 추천이나 매매 지시가 아니라 리스크 점검용 보조지표임을 설명
   - 1D 제품 후보, 1W/1M 준비 상태, local parquet와 Supabase thin DB 구조 요약

2. AI 지표 설명
   - 보수적 예측선: 하방 위험을 덜 낙관적으로 보려는 AI 예측선
   - AI 밴드: 향후 변동 가능 범위를 보는 리스크 참고 지표
   - line과 band를 합치지 않는 이유를 RSI/MACD처럼 역할이 다른 지표라는 방식으로 설명

3. 데이터 설명
   - yfinance local 후보
   - EODHD는 fallback/검증용에서 해지 후보로 이동 중
   - adjusted OHLC 기준 피처 계약
   - local parquet와 Supabase thin DB 저장 분리

4. 모델 설명
   - PatchTST: 보수적 예측선 후보
   - CNN-LSTM: AI 밴드 후보
   - TiDE: 현재 제품 후보가 아닌 보류/비교 후보

5. 실험에서 배운 점
   - 가격 피처 계약 재정리
   - 밴드 예측의 난이도
   - composite 제외
   - 역할별 평가 지표 분리
   - local-first 데이터 운영 전환

6. 현재 상태와 다음 단계
   - 1D line/band 제품 후보
   - latest-only thin upload 구조
   - yfinance append gate 검증
   - 1W 모델 실험, 백테스트 전략 재설계, 다크모드

## 4. AI 모델 화면과 역할 분리

- `AI 모델` 화면은 현재 제품 모델과 실험 결과를 보는 곳으로 유지했다.
- `Lens 리포트` 화면은 프로젝트 전체 의도, 데이터, 모델 구조, 실패에서 배운 점을 설명하는 화면으로 분리했다.
- run_id, feature_set, raw metric 같은 내부 용어는 리포트 기본 화면에 노출하지 않았다.

## 5. 문구 결정

- 핵심 문구는 `리스크를 먼저 보는 AI 투자 보조지표`로 정했다.
- “투자 추천”, “수익 보장”, “주가를 맞힌다”처럼 오해될 수 있는 표현은 쓰지 않았다.
- composite는 제품 기본에서 제외된 이전 접근으로만 조용히 설명했다.
- 초심자용 문장을 우선하고, 모델 상세는 `자세히` 접기 영역에만 짧게 남겼다.

## 6. 검증 결과

- `npm run build`
  - 샌드박스 내부 실행: `spawn EPERM` 실패
  - 승인된 샌드박스 외부 실행: 통과
- `scripts/check_demo_readiness.ps1`
  - backend health: OK
  - CORS: OK
  - frontend: OK
  - stock search: OK
  - LM/BM run, prediction, evaluation, history: OK
  - AAPL 1D/1M prices: 현재 DB 기준 404
  - 이 가격 데이터 404는 CP104 리포트 화면 구현과 무관한 기존 데이터 상태로 남긴다.
- 브라우저 확인
  - `Lens 리포트` 메뉴 진입 확인
  - 리포트 주요 섹션 렌더링 확인
  - 기존 `주식 보기`, `백테스트`, `AI 모델` 화면 진입 확인
  - `주식 보기` 1D/1W/1M 전환 확인
  - console error/warn 0건
- 서버 상태
  - backend `http://127.0.0.1:8000/api/v1/health/live`: 200
  - frontend `http://127.0.0.1:3000`: 200

## 7. 수정하지 않은 것

- 백엔드 API는 수정하지 않았다.
- DB schema는 수정하지 않았다.
- 모델 학습, inference 저장, Supabase write는 실행하지 않았다.
- fake data는 만들지 않았다.
- 백테스트 전략과 다크모드는 이번 CP에서 건드리지 않았다.

## 8. 남은 리스크와 TODO

- AAPL 가격 API가 현재 readiness에서 404로 남아 있어 주식 보기 기본 데모 데이터 상태는 별도 확인이 필요하다.
- 리포트 화면은 1차 정보 구조이며, 다음 단계에서 실제 데이터셋/모델 문서와 연결하는 상세 페이지를 만들 수 있다.
- 모바일 폭에서 큰 문제는 없도록 반응형을 넣었지만, 긴 설명의 읽기 리듬은 실제 사용자 검토 후 더 줄일 수 있다.
