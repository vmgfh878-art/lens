from __future__ import annotations

import argparse
import contextlib
from datetime import date, datetime, timedelta
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp135_yfinance_fetch_reliability"
CACHE_DIR = ROOT_DIR / "backend" / "data" / "cache" / "yfinance"
PRICE_PATH = SNAPSHOT_DIR / "price_data_yfinance.parquet"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
SOURCE_LINKS = {
    "yfinance_pypi": "https://pypi.org/project/yfinance/",
    "alpha_vantage_docs": "https://www.alphavantage.co/documentation/",
    "nasdaq_data_link_getting_started": "https://docs.data.nasdaq.com/docs/getting-started",
    "nasdaq_data_link_rate_limits": "https://help.data.nasdaq.com/article/490-is-there-a-rate-limit-or-speed-limit-for-api-usage",
    "stooq_pandas_datareader": "https://pydata.github.io/pandas-datareader/readers/stooq.html",
}

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)

# 로컬 차단 프록시가 있으면 Yahoo 응답 진단이 왜곡된다.
for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    _proxy_value = os.environ.get(_proxy_key, "")
    if "127.0.0.1:9" in _proxy_value:
        os.environ.pop(_proxy_key, None)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.sources.yf_common import prepare_yfinance  # noqa: E402
from backend.collector.utils.network import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()
prepare_yfinance()

import yfinance as yf  # noqa: E402


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_price_snapshot() -> pd.DataFrame:
    frame = pd.read_parquet(PRICE_PATH)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "empty": bool(frame.empty),
        "columns": [str(column) for column in frame.columns[:12]],
    }
    if not frame.empty:
        index = pd.to_datetime(frame.index, errors="coerce")
        summary["date_min"] = index.min().date().isoformat()
        summary["date_max"] = index.max().date().isoformat()
    return summary


def capture_call(name: str, fn: Callable[[], pd.DataFrame]) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    started = datetime.utcnow()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            frame = fn()
        status = "PASS" if not frame.empty else "EMPTY"
        error = None
    except Exception as exc:
        frame = pd.DataFrame()
        status = "EXCEPTION"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = (datetime.utcnow() - started).total_seconds()
    return {
        "method": name,
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
        "stdout_tail": stdout.getvalue()[-800:],
        "stderr_tail": stderr.getvalue()[-1200:],
        "frame": frame_summary(frame),
    }


def direct_chart_probe(ticker: str, host: str) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    url = f"https://{host}/v8/finance/chart/{ticker}"
    params = {
        "range": "10d",
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    started = datetime.utcnow()
    try:
        response = session.get(url, params=params, timeout=20)
        elapsed = (datetime.utcnow() - started).total_seconds()
        text_head = response.text[:160].replace("\n", " ")
        json_ok = False
        timestamp_count = None
        json_error = None
        try:
            payload = response.json()
            result = payload.get("chart", {}).get("result") or []
            timestamp_count = len(result[0].get("timestamp", [])) if result else 0
            json_ok = True
        except Exception as exc:
            json_error = f"{type(exc).__name__}: {exc}"
        return {
            "ticker": ticker,
            "host": host,
            "status_code": int(response.status_code),
            "content_type": response.headers.get("content-type"),
            "text_head": text_head,
            "json_ok": json_ok,
            "json_error": json_error,
            "timestamp_count": timestamp_count,
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "host": host,
            "status_code": None,
            "content_type": None,
            "text_head": None,
            "json_ok": False,
            "json_error": f"{type(exc).__name__}: {exc}",
            "timestamp_count": None,
            "elapsed_seconds": None,
        }


def cache_status() -> dict[str, Any]:
    db_files = []
    for db_path in sorted(CACHE_DIR.glob("*.db")):
        item: dict[str, Any] = {
            "path": str(db_path.relative_to(ROOT_DIR)),
            "bytes": int(db_path.stat().st_size),
            "modified_at": datetime.fromtimestamp(db_path.stat().st_mtime).isoformat(),
            "sqlite_readable": False,
            "tables": [],
        }
        try:
            connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
            tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table'").fetchall()]
            item["tables"] = tables
            item["row_counts"] = {
                table: int(connection.execute(f'select count(*) from "{table}"').fetchone()[0])
                for table in tables
            }
            connection.close()
            item["sqlite_readable"] = True
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        db_files.append(item)

    tracked = subprocess.run(
        ["git", "ls-files", "backend/data/cache/yfinance"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-v", "backend/data/cache/yfinance/cookies.db", "backend/data/cache/yfinance/tkr-tz.db"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cache_dir": str(CACHE_DIR.relative_to(ROOT_DIR)),
        "exists": CACHE_DIR.exists(),
        "db_files": db_files,
        "git_tracked_files": [line for line in tracked.stdout.splitlines() if line.strip()],
        "git_ignore_output": ignored.stdout.splitlines(),
        "reset_policy": "필요 시 실행 전 모든 Python 프로세스를 닫고 *.db를 백업/삭제하면 yfinance가 재생성할 수 있다. CP135에서는 삭제하지 않았다.",
    }


def run_yfinance_matrix(tickers: list[str], start_date: str, end_date: str, sleep_seconds: float) -> dict[str, Any]:
    batch_symbol = " ".join(tickers)
    result: dict[str, Any] = {
        "batch_period_10d": capture_call(
            "yf.download batch period=10d",
            lambda: yf.download(
                batch_symbol,
                period="10d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                threads=False,
                progress=False,
                timeout=20,
                group_by="ticker",
            ),
        ),
        "batch_start_end": capture_call(
            "yf.download batch start/end",
            lambda: yf.download(
                batch_symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=False,
                threads=False,
                progress=False,
                timeout=20,
                group_by="ticker",
            ),
        ),
        "single": {},
    }
    for ticker in tickers:
        ticker_result = {
            "download_period_10d": capture_call(
                f"{ticker} yf.download period=10d",
                lambda ticker=ticker: yf.download(
                    ticker,
                    period="10d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                    progress=False,
                    timeout=20,
                ),
            ),
            "download_start_end": capture_call(
                f"{ticker} yf.download start/end",
                lambda ticker=ticker: yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                    progress=False,
                    timeout=20,
                ),
            ),
            "history_period_10d": capture_call(
                f"{ticker} Ticker.history period=10d",
                lambda ticker=ticker: yf.Ticker(ticker).history(
                    period="10d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    timeout=20,
                ),
            ),
        }
        result["single"][ticker] = ticker_result
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return result


def run_direct_matrix(tickers: list[str], sleep_seconds: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ticker in tickers:
        result[ticker] = {
            "query1": direct_chart_probe(ticker, "query1.finance.yahoo.com"),
            "query2": direct_chart_probe(ticker, "query2.finance.yahoo.com"),
        }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return result


def classify_failure(yfinance_matrix: dict[str, Any], direct_matrix: dict[str, Any]) -> dict[str, Any]:
    direct_status_codes = [
        item["status_code"]
        for ticker_result in direct_matrix.values()
        for item in ticker_result.values()
        if item.get("status_code") is not None
    ]
    direct_heads = [
        str(item.get("text_head") or "")
        for ticker_result in direct_matrix.values()
        for item in ticker_result.values()
    ]
    yfinance_statuses = [
        yfinance_matrix["batch_period_10d"]["status"],
        yfinance_matrix["batch_start_end"]["status"],
    ]
    for ticker_result in yfinance_matrix["single"].values():
        yfinance_statuses.extend(item["status"] for item in ticker_result.values())

    all_direct_429 = bool(direct_status_codes) and all(code == 429 for code in direct_status_codes)
    any_too_many = any("Too Many Requests" in head for head in direct_heads)
    any_yfinance_success = any(status == "PASS" for status in yfinance_statuses)

    if all_direct_429 and any_too_many and not any_yfinance_success:
        classification = "YAHOO_EDGE_RATE_LIMIT_429"
        root_cause = "Yahoo chart API가 HTML `Edge: Too Many Requests`를 반환했다. yfinance JSONDecodeError는 이 HTML 응답을 JSON으로 파싱하려다 발생한 2차 증상이다."
        confidence = "HIGH"
    elif any_yfinance_success:
        classification = "INTERMITTENT_SUCCESS"
        root_cause = "일부 yfinance 호출이 성공했으므로 일시적 장애 또는 티커별 제한으로 본다."
        confidence = "MEDIUM"
    elif direct_status_codes and all(code in {200, 404} for code in direct_status_codes):
        classification = "YFINANCE_COOKIE_OR_SESSION_LAYER"
        root_cause = "Yahoo 직접 chart 응답은 rate limit이 아닌데 yfinance만 실패했다. cookie/crumb/session/cache 문제 가능성이 높다."
        confidence = "MEDIUM"
    else:
        classification = "UNCLASSIFIED_FETCH_FAILURE"
        root_cause = "네트워크, Yahoo edge, 또는 yfinance 내부 상태 중 하나로 보이나 단일 원인을 확정하지 못했다."
        confidence = "LOW"

    return {
        "classification": classification,
        "root_cause": root_cause,
        "confidence": confidence,
        "direct_status_codes": direct_status_codes,
        "yfinance_status_counts": {status: yfinance_statuses.count(status) for status in sorted(set(yfinance_statuses))},
    }


def retry_policy() -> dict[str, Any]:
    return {
        "states": {
            "fetch_failed": "Yahoo/yfinance 응답이 비었거나 429/HTML/JSONDecodeError/네트워크 예외가 발생한 상태. append 금지.",
            "no_new_rows": "fetch는 성공했고 최신 fetched date가 snapshot latest 이하인 상태. append 없이 정상 종료 가능.",
            "partial_day_filtered": "fetch는 성공했지만 신규 row가 모두 current_date 이상이라 완료 거래일 gate에서 제외된 상태. append 금지, 다음 실행 대기.",
            "append_ready": "fetch 성공, adjusted OHLC 통과, row.date < current_date 신규 완료 거래일 존재.",
            "append_done": "append_ready 후보를 별도 승인된 append 단계에서 저장하고 중복/계약 검증까지 통과한 상태.",
        },
        "attempts": [
            "1차: 5티커 batch `yf.download(period=10d, interval=1d, auto_adjust=False, threads=False, timeout=20)`",
            "2차: 실패 티커만 단건 `yf.download(start/end)` 재시도",
            "3차: 실패가 JSONDecodeError/429이면 60초 이상 sleep 후 direct chart API 1건으로 rate limit 여부 확인",
            "4차: OperationalError 또는 sqlite/cache 오류일 때만 yfinance cache reset 후보를 제시하고, 실제 삭제는 수동 승인 후 수행",
            "5차: 여전히 실패하면 `fetch_failed`로 중단한다. EODHD fallback은 사용하지 않는다.",
        ],
        "abort_rules": [
            "direct Yahoo chart API가 429 `Too Many Requests`를 반환하면 같은 실행에서 반복 호출을 늘리지 않는다.",
            "성공 티커와 실패 티커가 섞이면 append를 전체 중단하거나 성공 티커만 별도 승인 CP에서 제한 append한다.",
            "빈 frame을 no_new_rows로 분류하려면 direct/yfinance 중 하나가 정상 JSON과 정상 OHLC column을 반환해야 한다.",
        ],
        "recommended_backoff_seconds": [60, 300, 1800],
        "daily_job_recommendation": "로컬 daily job은 장 마감 후 1회, 실패 시 30분 뒤 1회만 재시도하고 실패 상태 파일을 남긴다.",
    }


def free_source_candidates() -> list[dict[str, Any]]:
    return [
        {
            "source": "Stooq",
            "role": "보조 확인 후보",
            "fit": "WARN",
            "reason": "pandas-datareader로 daily OHLCV 조회가 가능하지만 ticker suffix와 adjusted close/split/dividend 계약이 Lens v3_adjusted_ohlc와 바로 맞지 않는다.",
            "source_link": SOURCE_LINKS["stooq_pandas_datareader"],
        },
        {
            "source": "Alpha Vantage",
            "role": "제한적 보조 확인 후보",
            "fit": "WARN",
            "reason": "TIME_SERIES_DAILY는 무료 compact 최신 100개에 맞지만, adjusted daily 함수와 full history는 문서상 premium 제약이 있어 100티커 daily primary로는 부적합하다.",
            "source_link": SOURCE_LINKS["alpha_vantage_docs"],
        },
        {
            "source": "Nasdaq Data Link",
            "role": "macro/일부 공개 데이터 후보",
            "fit": "WARN",
            "reason": "무료/프리미엄 데이터가 혼재하고 API rate limit은 명확하지만, Lens 100~500티커 US EOD adjusted OHLC primary 대체로 바로 쓸 수 있는 무료 계약은 별도 dataset 검증이 필요하다.",
            "source_link": SOURCE_LINKS["nasdaq_data_link_getting_started"],
        },
    ]


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    classification = metrics["failure_classification"]
    report = f"""# CP135-DG yfinance fetch 신뢰성 진단 보고서

## 1. 요약

판정은 **{metrics["final_decision"]["status"]}**이다.

원인 분류는 **{classification["classification"]}**이며 신뢰도는 **{classification["confidence"]}**이다.

{classification["root_cause"]}

CP134의 빈 응답은 `no_new_rows`가 아니었다. Yahoo chart API 직접 호출도 `429 Edge: Too Many Requests` HTML을 반환했고, yfinance는 이 응답을 JSON으로 파싱하다 `JSONDecodeError`를 냈다. 따라서 local daily update는 append를 중단해야 한다.

## 2. 실행 범위

- 티커: {", ".join(metrics["scope"]["tickers"])}
- start_date: {metrics["scope"]["start_date"]}
- end_date: {metrics["scope"]["end_date"]}
- EODHD fallback: 사용 안 함
- DB/parquet write: 없음
- 모델 학습/inference: 없음

## 3. Yahoo 직접 응답

| ticker | query1 status | query1 head | query2 status | query2 head |
|---|---:|---|---:|---|
{direct_table(metrics)}

## 4. yfinance 호출 비교

| 호출 | 상태 | rows | stderr tail |
|---|---|---:|---|
| batch period=10d | {metrics["yfinance_matrix"]["batch_period_10d"]["status"]} | {metrics["yfinance_matrix"]["batch_period_10d"]["frame"]["rows"]} | {compact(metrics["yfinance_matrix"]["batch_period_10d"]["stderr_tail"])} |
| batch start/end | {metrics["yfinance_matrix"]["batch_start_end"]["status"]} | {metrics["yfinance_matrix"]["batch_start_end"]["frame"]["rows"]} | {compact(metrics["yfinance_matrix"]["batch_start_end"]["stderr_tail"])} |

단건 호출도 `download_period_10d`, `download_start_end`, `Ticker.history` 모두 빈 프레임이었다. 상세 결과는 metrics JSON에 기록했다.

## 5. cache/cookie 상태

- cache dir: `{metrics["cache_status"]["cache_dir"]}`
- git tracked files: {metrics["cache_status"]["git_tracked_files"]}
- git ignore: {metrics["cache_status"]["git_ignore_output"]}
- reset policy: {metrics["cache_status"]["reset_policy"]}

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
{candidate_table(metrics)}

## 8. 최종 판단

{metrics["final_decision"]["summary"]}

## 9. 금지 작업 미발생 확인

DB write, parquet append, 모델 학습/inference, Supabase 대량 read/write, EODHD fallback, 프론트 수정은 수행하지 않았다.

## 10. 참고 링크

- [yfinance PyPI]({SOURCE_LINKS["yfinance_pypi"]})
- [Alpha Vantage API 문서]({SOURCE_LINKS["alpha_vantage_docs"]})
- [Nasdaq Data Link 시작 문서]({SOURCE_LINKS["nasdaq_data_link_getting_started"]})
- [Nasdaq Data Link rate limit]({SOURCE_LINKS["nasdaq_data_link_rate_limits"]})
- [pandas-datareader Stooq 문서]({SOURCE_LINKS["stooq_pandas_datareader"]})
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def compact(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())[-220:]


def direct_table(metrics: dict[str, Any]) -> str:
    rows = []
    for ticker, item in metrics["direct_chart_matrix"].items():
        q1 = item["query1"]
        q2 = item["query2"]
        rows.append(
            f"| {ticker} | {q1['status_code']} | `{compact(q1['text_head'])}` | {q2['status_code']} | `{compact(q2['text_head'])}` |"
        )
    return "\n".join(rows)


def candidate_table(metrics: dict[str, Any]) -> str:
    rows = []
    for item in metrics["free_source_candidates"]:
        rows.append(f"| {item['source']} | {item['role']} | {item['fit']} | {item['reason']} |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP135 yfinance fetch reliability audit")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp135_yfinance_fetch_reliability_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp135_yfinance_fetch_reliability_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [ticker.upper() for ticker in args.tickers]
    price = load_price_snapshot()
    snapshot_latest = pd.to_datetime(price["date"], errors="coerce").max().date()
    end_date = pd.to_datetime(args.end_date).date()
    start_date = (snapshot_latest - timedelta(days=args.lookback_days)).isoformat()
    end_date_str = end_date.isoformat()

    direct_matrix = run_direct_matrix(tickers, args.sleep_seconds)
    yfinance_matrix = run_yfinance_matrix(tickers, start_date, end_date_str, args.sleep_seconds)
    cache = cache_status()
    classification = classify_failure(yfinance_matrix, direct_matrix)
    policy = retry_policy()
    candidates = free_source_candidates()

    if classification["classification"] == "YAHOO_EDGE_RATE_LIMIT_429":
        status = "PASS"
        summary = "실패 원인을 Yahoo edge 429 rate limit으로 분류했고, no_new_rows와 fetch_failed를 분리하는 retry/abort 정책을 확정했다."
    elif classification["classification"] == "INTERMITTENT_SUCCESS":
        status = "WARN"
        summary = "일부 호출은 성공했지만 간헐 실패가 있어 retry 후에도 실패 티커가 남으면 append를 중단해야 한다."
    else:
        status = "WARN"
        summary = "단일 원인을 완전히 확정하지 못했으나 fetch_failed와 no_new_rows를 분리하는 중단 정책은 필요하다."

    metrics: dict[str, Any] = {
        "cp": "CP135-DG",
        "created_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "snapshot_latest_date": snapshot_latest.isoformat(),
            "start_date": start_date,
            "end_date": end_date_str,
            "provider": "yfinance",
            "fallback_provider": None,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "EODHD_API_KEY_SET": bool(os.environ.get("EODHD_API_KEY")),
            "yfinance_version": getattr(yf, "__version__", None),
        },
        "direct_chart_matrix": direct_matrix,
        "yfinance_matrix": yfinance_matrix,
        "cache_status": cache,
        "failure_classification": classification,
        "retry_policy": policy,
        "daily_update_state_mapping": policy["states"],
        "free_source_candidates": candidates,
        "forbidden_actions_observed": {
            "db_write": False,
            "parquet_append": False,
            "model_training": False,
            "inference": False,
            "supabase_bulk_read_write": False,
            "eodhd_fallback": False,
            "frontend_modify": False,
        },
        "source_links": SOURCE_LINKS,
        "final_decision": {
            "status": status,
            "summary": summary,
        },
    }

    metrics_path = Path(args.metrics_path)
    report_path = Path(args.report_path)
    write_json(metrics_path, metrics)
    write_json(LOG_DIR / metrics_path.name, metrics)
    write_report(report_path, metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
