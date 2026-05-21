# CP122-P 백테스트 전략 신호 화면 재구성 보고서

CP122-P는 백테스트 화면을 단순 티커 검색형 화면에서 `Lens Balance v1` 전략 신호 중심 화면으로 바꾼 작업이다. 사용자는 먼저 전략이 해석한 종목 후보를 보고, 그중 하나를 선택해 단일 티커 백테스트 상세를 확인한다.

## 1. 변경 목표 반영

- 화면 상단 주인공을 티커 검색 입력에서 `전략 신호`로 변경했다.
- `Lens Balance v1` 기준 최신 신호 요약을 먼저 보여준다.
- 티커 직접 입력은 `직접 종목 확인` 보조 영역으로 내렸다.
- 포트폴리오 추천, 로그인, 구독, 사용자 보유 종목 관리는 만들지 않았다.

## 2. 전략 신호 그룹

4개 그룹을 추가했다.

| 그룹 | 의미 |
| --- | --- |
| 매수 후보 | AI 지표 기준으로 신규 진입을 검토할 수 있는 종목 |
| 보유 유지 | 전략상 보유 상태를 이어가는 종목 |
| 위험 확대 | line 약화나 밴드 위험 확대로 방어 확인이 필요한 종목 |
| 관망 | 진입 기준이 아직 충분하지 않거나 신호가 부족한 종목 |

각 그룹은 최대 10개 카드만 표시한다. 전체 universe를 모두 펼쳐 보여주지 않는다.

## 3. 종목 카드 표시 정보

각 카드에는 아래 정보만 간결하게 표시했다.

- ticker
- sector, 종목명 데이터가 없으면 `종목명 없음`
- 전략 신호
- 보수적 예측선 값
- AI 밴드 하단 위험도
- AI 밴드 폭 상태
- 60일 추세
- 최신 신호 이유

RSI는 이번 카드 전면에는 올리지 않았다. 카드가 길어지지 않게 60일 추세를 우선 표시했다.

## 4. 신호 산정 방식

기존 `Lens Balance v1` 룰을 재사용했다.

- line run: `patchtst-1D-efad3c29d803`
- band run: `cnn_lstm-1D-d0c780dee5e8`
- composite 사용: 없음
- timeframe: 1D
- line은 방향과 진입 판단
- band는 하단 위험과 밴드 폭 확장 확인
- `targetPosition`, 최종 `position`, line 약화, lower risk, width expansion을 함께 보고 그룹을 나눴다.

## 5. 데이터 소스와 제한

새 DB 테이블이나 scanner API를 추가하지 않았다.

사용한 데이터:
- 가격 API
- indicator API
- line prediction history
- band prediction history
- stock search fallback

현재 scanner API가 없으므로 프론트에서 12개 대표 ticker subset만 read-only로 조회한다.

사용한 subset:

`AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, AMD, NFLX, JPM, XOM`

데이터가 없거나 prediction history가 부족하면 fake 값을 만들지 않고 `아직 신호 없음`으로 표시한다.

## 6. 상세 영역

카드를 클릭하면 기존 단일 티커 상세가 아래에 유지된다.

- 성과 비교
- 가격 차트
- 매수/매도 마커
- 전략 판단 요소
- 최근 매매 기록
- 매매 구간 보기
- read-only 데이터 출처

직접 입력한 티커도 신호 목록에 없어도 상세를 조회할 수 있다.

## 7. 문구 처리

투자 권유처럼 보이는 표현은 사용하지 않았다.

사용한 표현:
- 매수 후보
- 보유 유지
- 위험 확대
- 관망
- 전략상 보유 유지
- AI 지표 기준 신호

사용하지 않은 표현:
- 사세요
- 팔아야 합니다
- 수익 보장
- 최적 포트폴리오
- 확정 추천

## 8. 검증 결과

### 명령 검증

- `npm run build`: 통과
- `scripts/check_demo_readiness.ps1`: 통과
  - backend health OK
  - frontend OK
  - frontend-static OK
  - AAPL 가격 OK
  - product LM/BM run/history OK

### 브라우저 검증

- 백테스트 화면 진입 확인
- 전략 신호 그룹 4개 표시 확인
- 카드 클릭 시 MSFT 상세 백테스트 전환 확인
- 직접 티커 입력으로 `T` 조회 확인
- `T`는 가격은 있으나 product prediction history가 없어 저장 결과 없음 상태로 표시됨
- console error/warn 0건 확인

## 9. 남은 한계

- 아직 500티커 전체 scanner API가 없으므로 프론트 subset 기반이다.
- subset 조회는 데모용으로 충분하지만, 본 서비스에서는 backend read-only scanner API가 필요하다.
- 종목명 필드가 API에 없어서 카드에는 sector 또는 `종목명 없음`만 표시한다.
- 신호 그룹은 최신 history 기준 요약이며, 투자 추천이나 포트폴리오 구성 기능이 아니다.
