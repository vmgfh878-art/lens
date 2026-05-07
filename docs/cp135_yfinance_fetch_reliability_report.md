# CP135-DG yfinance fetch 신뢰성 진단 보고서

## 1. 요약

판정은 **PASS**이다.

원인 분류는 **YAHOO_EDGE_RATE_LIMIT_429**이며 신뢰도는 **HIGH**이다.

Yahoo chart API가 HTML `Edge: Too Many Requests`를 반환했다. yfinance JSONDecodeError는 이 HTML 응답을 JSON으로 파싱하려다 발생한 2차 증상이다.

CP134의 빈 응답은 `no_new_rows`가 아니었다. Yahoo chart API 직접 호출도 `429 Edge: Too Many Requests` HTML을 반환했고, yfinance는 이 응답을 JSON으로 파싱하다 `JSONDecodeError`를 냈다. 따라서 local daily update는 append를 중단해야 한다.

## 2. 실행 범위

- 티커: AAPL, MSFT, NVDA, TSLA, NFLX
- start_date: 2026-04-22
- end_date: 2026-05-06
- EODHD fallback: 사용 안 함
- DB/parquet write: 없음
- 모델 학습/inference: 없음

## 3. Yahoo 직접 응답

| ticker | query1 status | query1 head | query2 status | query2 head |
|---|---:|---|---:|---|
| AAPL | 429 | `Edge: Too Many Requests` | 429 | `Edge: Too Many Requests` |
| MSFT | 429 | `Edge: Too Many Requests` | 429 | `Edge: Too Many Requests` |
| NVDA | 429 | `Edge: Too Many Requests` | 429 | `Edge: Too Many Requests` |
| TSLA | 429 | `Edge: Too Many Requests` | 429 | `Edge: Too Many Requests` |
| NFLX | 429 | `Edge: Too Many Requests` | 429 | `Edge: Too Many Requests` |

## 4. yfinance 호출 비교

| 호출 | 상태 | rows | stderr tail |
|---|---|---:|---|
| batch period=10d | EMPTY | 0 | 5 Failed downloads: ['AAPL', 'TSLA', 'MSFT', 'NFLX', 'NVDA']: JSONDecodeError('Expecting value: line 1 column 1 (char 0)') |
| batch start/end | EMPTY | 0 | 5 Failed downloads: ['AAPL', 'TSLA', 'MSFT', 'NFLX', 'NVDA']: JSONDecodeError('Expecting value: line 1 column 1 (char 0)') |

단건 호출도 `download_period_10d`, `download_start_end`, `Ticker.history` 모두 빈 프레임이었다. 상세 결과는 metrics JSON에 기록했다.

## 5. cache/cookie 상태

- cache dir: `backend\data\cache\yfinance`
- git tracked files: []
- git ignore: ['.gitignore:37:backend/data/cache/yfinance/*.db\tbackend/data/cache/yfinance/cookies.db', '.gitignore:37:backend/data/cache/yfinance/*.db\tbackend/data/cache/yfinance/tkr-tz.db']
- reset policy: 필요 시 실행 전 모든 Python 프로세스를 닫고 *.db를 백업/삭제하면 yfinance가 재생성할 수 있다. CP135에서는 삭제하지 않았다.

SQLite 파일은 읽기 가능했다. 이번 원인은 직접 chart API 429이므로 cache reset은 1차 해결책이 아니다. 다만 `OperationalError` 또는 sqlite lock이 재발할 때만 수동 reset 후보로 둔다.

## 6. retry/failure 정책

상태값:
- `fetch_failed`: fetch 실패. append 금지.
- `no_new_rows`: fetch 성공, 신규 row 없음. 정상 종료 가능.
- `partial_day_filtered`: fetch 성공, 신규 row가 current date 이상이라 제외. append 금지.
- `append_ready`: 완료 거래일 신규 row 존재. 별도 append 단계 가능.
- `append_done`: append 후 검증 통과.

재시도 순서:
1. batch 호출 1회
2. 실패 티커 단건 재시도
3. 429/JSONDecodeError면 60초 이상 대기 후 direct chart 1건 확인
4. cache/SQLite 오류일 때만 cache reset 후보 제시
5. 실패 지속 시 `fetch_failed`로 중단

## 7. 무료 보조 source 후보

| source | 역할 | 판단 | 이유 |
|---|---|---|---|
| Stooq | 보조 확인 후보 | WARN | pandas-datareader로 daily OHLCV 조회가 가능하지만 ticker suffix와 adjusted close/split/dividend 계약이 Lens v3_adjusted_ohlc와 바로 맞지 않는다. |
| Alpha Vantage | 제한적 보조 확인 후보 | WARN | TIME_SERIES_DAILY는 무료 compact 최신 100개에 맞지만, adjusted daily 함수와 full history는 문서상 premium 제약이 있어 100티커 daily primary로는 부적합하다. |
| Nasdaq Data Link | macro/일부 공개 데이터 후보 | WARN | 무료/프리미엄 데이터가 혼재하고 API rate limit은 명확하지만, Lens 100~500티커 US EOD adjusted OHLC primary 대체로 바로 쓸 수 있는 무료 계약은 별도 dataset 검증이 필요하다. |

## 8. 최종 판단

실패 원인을 Yahoo edge 429 rate limit으로 분류했고, no_new_rows와 fetch_failed를 분리하는 retry/abort 정책을 확정했다.

## 9. 금지 작업 미발생 확인

DB write, parquet append, 모델 학습/inference, Supabase 대량 read/write, EODHD fallback, 프론트 수정은 수행하지 않았다.

## 10. 참고 링크

- [yfinance PyPI](https://pypi.org/project/yfinance/)
- [Alpha Vantage API 문서](https://www.alphavantage.co/documentation/)
- [Nasdaq Data Link 시작 문서](https://docs.data.nasdaq.com/docs/getting-started)
- [Nasdaq Data Link rate limit](https://help.data.nasdaq.com/article/490-is-there-a-rate-limit-or-speed-limit-for-api-usage)
- [pandas-datareader Stooq 문서](https://pydata.github.io/pandas-datareader/readers/stooq.html)
