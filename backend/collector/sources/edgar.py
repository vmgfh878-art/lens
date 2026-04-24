from __future__ import annotations

import os
from datetime import datetime

import requests

from backend.collector.utils.network import sanitize_proxy_env

SEC_BASE = "https://data.sec.gov"
SEC_WEB = "https://www.sec.gov"
_TICKER_CIK_CACHE: dict[str, str] | None = None


def _sec_headers(host: str = "data.sec.gov") -> dict[str, str]:
    user_agent = os.environ.get("SEC_USER_AGENT", "LensDataBot admin@example.com")
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


def _get_json(url: str) -> dict | list:
    sanitize_proxy_env()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, headers=_sec_headers("data.sec.gov"), timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def ticker_to_cik(ticker: str) -> str | None:
    global _TICKER_CIK_CACHE
    if _TICKER_CIK_CACHE is None:
        sanitize_proxy_env()
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            f"{SEC_WEB}/files/company_tickers_exchange.json",
            headers=_sec_headers("www.sec.gov"),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        mapping: dict[str, str] = {}
        for row in payload.get("data", []):
            cik, _name, sec_ticker, _exchange = row
            mapping[str(sec_ticker).upper()] = str(cik).zfill(10)
        _TICKER_CIK_CACHE = mapping
    return _TICKER_CIK_CACHE.get(ticker.upper())


def _extract_fact(companyfacts: dict, tag: str, unit: str = "USD") -> list[dict]:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    fact = facts.get(tag)
    if not fact:
        return []
    units = fact.get("units", {}).get(unit, [])
    rows: list[dict] = []
    for row in units:
        end = row.get("end")
        filed = row.get("filed")
        if not end or not filed:
            continue
        rows.append(
            {
                "tag": tag,
                "value": row.get("val"),
                "date": end,
                "filing_date": filed,
                "frame": row.get("frame"),
                "fy": row.get("fy"),
                "fp": row.get("fp"),
            }
        )
    return rows


def fetch_companyfacts(cik: str) -> dict:
    normalized = cik.zfill(10)
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{normalized}.json"
    return _get_json(url)  # type: ignore[return-value]


REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "InterestAndDividendIncomeOperating",
    "InterestIncomeOperating",
    "InterestIncome",
)
EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "PartnersCapital",
    "MembersEquity",
)
EPS_TAGS = (
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
)
_ALLOWED_FP = {"Q1", "Q2", "Q3", "FY"}
_MAX_QUARTERS = 12


def _extract_fact_any(
    companyfacts: dict, tags: tuple[str, ...], unit: str = "USD"
) -> list[dict]:
    """여러 후보 태그를 순서대로 시도하고 첫 매칭을 반환한다.

    SEC XBRL 태그가 업종·시기별로 다른 변이를 흡수한다(은행·파트너십 등).
    """
    for tag in tags:
        rows = _extract_fact(companyfacts, tag, unit=unit)
        if rows:
            return rows
    return []


def fetch_core_financials(ticker: str, cik: str) -> dict[str, list[dict]]:
    """SEC EDGAR companyfacts에서 핵심 재무 태그를 뽑는다."""
    payload = fetch_companyfacts(cik)
    return {
        "revenue": _extract_fact_any(payload, REVENUE_TAGS),
        "net_income": _extract_fact(payload, "NetIncomeLoss"),
        "assets": _extract_fact(payload, "Assets"),
        "liabilities": _extract_fact(payload, "Liabilities"),
        "equity": _extract_fact_any(payload, EQUITY_TAGS),
        "eps": _extract_fact_any(payload, EPS_TAGS, unit="USD/shares"),
    }


def fetch_company_submission(cik: str) -> dict:
    normalized = cik.zfill(10)
    url = f"{SEC_BASE}/submissions/CIK{normalized}.json"
    return _get_json(url)  # type: ignore[return-value]


def latest_filing_date(cik: str) -> str | None:
    payload = fetch_company_submission(cik)
    recent = payload.get("filings", {}).get("recent", {})
    filed = recent.get("filingDate") or []
    return filed[0] if filed else None


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.min
    return datetime.fromisoformat(value)


def _dedupe_quarter_rows(rows: list[dict]) -> list[dict]:
    filtered = [row for row in rows if str(row.get("fp") or "").upper() in _ALLOWED_FP]
    deduped: dict[str, dict] = {}
    for row in filtered:
        end = str(row.get("date") or "")
        current = deduped.get(end)
        if current is None or _parse_date(str(row.get("filing_date") or "")) >= _parse_date(str(current.get("filing_date") or "")):
            deduped[end] = row
    ordered = sorted(deduped.values(), key=lambda row: _parse_date(str(row.get("date") or "")), reverse=True)
    return ordered[:_MAX_QUARTERS]


def _match_value(rows_by_date: dict[str, dict], ordered_dates: list[str], target_date: str) -> tuple[object | None, str | None]:
    for current_date in ordered_dates:
        if current_date <= target_date:
            row = rows_by_date[current_date]
            return row.get("value"), str(row.get("filing_date") or "") or None
    return None, None


def fetch_edgar_fundamentals(ticker: str) -> list[dict]:
    try:
        cik = ticker_to_cik(ticker)
        if not cik:
            return []
        core = fetch_core_financials(ticker, cik)
        revenue_rows = core.get("revenue", [])
        if not revenue_rows:
            return []
    except Exception:
        return []

    normalized = {
        "revenue": _dedupe_quarter_rows(revenue_rows),
        "net_income": _dedupe_quarter_rows(core.get("net_income", [])),
        "assets": _dedupe_quarter_rows(core.get("assets", [])),
        "liabilities": _dedupe_quarter_rows(core.get("liabilities", [])),
        "equity": _dedupe_quarter_rows(core.get("equity", [])),
        "eps": _dedupe_quarter_rows(core.get("eps", [])),
    }
    if not normalized["revenue"]:
        return []

    metric_maps = {
        name: {str(row.get("date") or ""): row for row in rows}
        for name, rows in normalized.items()
    }
    metric_dates = {
        name: sorted(metric_map.keys(), reverse=True)
        for name, metric_map in metric_maps.items()
    }

    records: list[dict] = []
    for revenue_row in sorted(normalized["revenue"], key=lambda row: _parse_date(str(row.get("date") or ""))):
        end_date = str(revenue_row.get("date") or "")
        filing_date = str(revenue_row.get("filing_date") or "") or None
        revenue = revenue_row.get("value")
        net_income, _ = _match_value(metric_maps["net_income"], metric_dates["net_income"], end_date)
        total_assets, _ = _match_value(metric_maps["assets"], metric_dates["assets"], end_date)
        total_liabilities, _ = _match_value(metric_maps["liabilities"], metric_dates["liabilities"], end_date)
        equity, _ = _match_value(metric_maps["equity"], metric_dates["equity"], end_date)
        eps, _ = _match_value(metric_maps["eps"], metric_dates["eps"], end_date)

        valid_equity = equity if isinstance(equity, (int, float)) and equity > 0 else None
        roe = (net_income / valid_equity) if isinstance(net_income, (int, float)) and valid_equity else None
        debt_ratio = (
            total_liabilities / valid_equity
            if isinstance(total_liabilities, (int, float)) and valid_equity
            else None
        )

        records.append(
            {
                "ticker": ticker.upper(),
                "date": end_date,
                "filing_date": filing_date,
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity": equity,
                "shares_issued": None,
                "eps": eps,
                "roe": roe,
                "debt_ratio": debt_ratio,
                "interest_coverage": None,
                "operating_cash_flow": None,
            }
        )

    return records
