# yfinance fetch retry 정책

## 목적

local daily update에서 yfinance 응답 실패와 신규 거래일 없음 상태를 분리한다. 빈 응답, HTML 응답, `JSONDecodeError`, HTTP 429는 모두 `fetch_failed`이며 parquet append 대상이 아니다.

## 상태값

| 상태 | 의미 | append |
|---|---|---|
| `fetch_failed` | Yahoo/yfinance 응답이 비었거나 429/HTML/JSONDecodeError/네트워크 예외가 발생했다. | 금지 |
| `no_new_rows` | fetch는 성공했고 최신 fetched date가 snapshot latest 이하이다. | 없음 |
| `partial_day_filtered` | fetch는 성공했지만 신규 row가 모두 current date 이상이라 완료 거래일 gate에서 제외됐다. | 금지 |
| `append_ready` | adjusted OHLC 계약을 통과한 신규 완료 거래일 row가 있다. | 별도 append 단계 가능 |
| `append_done` | append 후 duplicate/source/adjusted OHLC 검증까지 통과했다. | 완료 |

## 재시도 순서

1. 5티커 batch `yf.download(period=10d, interval=1d, auto_adjust=False, threads=False, timeout=20)` 호출
2. 실패 티커만 단건 `yf.download(start/end)` 재시도
3. `JSONDecodeError` 또는 HTTP 429가 의심되면 direct Yahoo chart API 1건으로 rate limit 여부 확인
4. 429 `Too Many Requests`이면 같은 실행에서 호출을 늘리지 않고 쿨다운한다
5. `OperationalError` 또는 SQLite lock이면 yfinance cache reset 후보를 제시한다
6. 실패가 지속되면 `fetch_failed`로 종료하고 append하지 않는다

## 쿨다운

- 로컬 daily job은 장 마감 후 1회 실행한다.
- 실패하면 30분 뒤 1회만 재시도한다.
- 두 번 모두 실패하면 실패 상태 파일과 metrics를 남기고 다음 운영 판단까지 중단한다.

## cache reset 정책

`backend/data/cache/yfinance/*.db`는 git ignore 대상이며 재생성 가능하다. 다만 CP135 기준 직접 Yahoo chart API가 429를 반환했으므로 cache reset은 1차 해결책이 아니다. cache reset은 SQLite 오류, lock, 파일 손상 징후가 있을 때만 수동 승인 후 수행한다.

## EODHD fallback

EODHD fallback은 daily yfinance 운영 경로에서 사용하지 않는다. 보조 무료 source는 자동 append provider가 아니라 검증용 후보로만 둔다.
