from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from collector.errors import SourceLimitReachedError
from collector.sources.yf_common import prepare_yfinance

prepare_yfinance()


def _get_safe_value(row: pd.Series, keys: list[str]) -> float | None:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return None


def fetch_yahoo_fundamentals(ticker: str) -> list[dict]:
    """야후에서 최근 몇 년 재무 데이터를 읽는다."""
    stock = yf.Ticker(ticker)
    try:
        fin_frame = stock.financials.T
        bal_frame = stock.balance_sheet.T
        cash_frame = stock.cashflow.T
    except Exception:
        return []

    if fin_frame.empty or bal_frame.empty:
        return []

    fin_frame.index = pd.to_datetime(fin_frame.index)
    bal_frame.index = pd.to_datetime(bal_frame.index)
    cash_frame.index = pd.to_datetime(cash_frame.index)

    merged = fin_frame.join(bal_frame, lsuffix="_fin", rsuffix="_bal", how="inner")
    merged = merged.join(cash_frame, rsuffix="_cash", how="left")

    rows: list[dict] = []
    for date_idx, row in merged.iterrows():
        revenue = _get_safe_value(row, ["Total Revenue", "Operating Revenue"])
        net_income = _get_safe_value(row, ["Net Income", "Net Income Common Stockholders"])
        total_assets = _get_safe_value(row, ["Total Assets"])
        total_liabilities = _get_safe_value(
            row,
            ["Total Liabilities Net Minority Interest", "Total Liabilities"],
        )
        equity = _get_safe_value(row, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        eps = _get_safe_value(row, ["Basic EPS", "Diluted EPS"])
        operating_cash_flow = _get_safe_value(
            row,
            ["Operating Cash Flow", "Total Cash From Operating Activities"],
        )
        shares_issued = _get_safe_value(row, ["Share Issued", "Ordinary Shares Number"])
        operating_income = _get_safe_value(row, ["Operating Income", "EBIT"])
        interest_expense = _get_safe_value(row, ["Interest Expense", "Interest Expense Non Operating"])

        roe = (net_income / equity) if net_income and equity else None
        debt_ratio = (total_liabilities / equity) if total_liabilities and equity else None
        interest_coverage = (
            operating_income / abs(interest_expense)
            if operating_income and interest_expense and abs(interest_expense) > 0
            else None
        )

        rows.append(
            {
                "ticker": ticker.upper(),
                "date": date_idx.date().isoformat(),
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity": equity,
                "shares_issued": shares_issued,
                "eps": eps,
                "roe": roe,
                "debt_ratio": debt_ratio,
                "interest_coverage": interest_coverage,
                "operating_cash_flow": operating_cash_flow,
            }
        )
    return rows


def fetch_fmp_fundamentals(ticker: str, api_key: str, limit: int = 5) -> list[dict]:
    """FMP에서 최근 재무 데이터를 읽는다."""
    if not api_key:
        return []

    safe_limit = max(1, min(int(limit), 5))
    base_url = "https://financialmodelingprep.com/stable"
    try:
        income_response = requests.get(
            f"{base_url}/income-statement?symbol={ticker}&limit={safe_limit}&apikey={api_key}",
            timeout=20,
        )
        balance_response = requests.get(
            f"{base_url}/balance-sheet-statement?symbol={ticker}&limit={safe_limit}&apikey={api_key}",
            timeout=20,
        )
        cash_response = requests.get(
            f"{base_url}/cash-flow-statement?symbol={ticker}&limit={safe_limit}&apikey={api_key}",
            timeout=20,
        )

        for response in (income_response, balance_response, cash_response):
            if response.status_code == 429:
                raise SourceLimitReachedError("FMP", response.text.strip())
            response.raise_for_status()

        income_data = income_response.json()
        balance_data = balance_response.json()
        cash_data = cash_response.json()
    except SourceLimitReachedError:
        raise
    except Exception:
        return []

    if isinstance(income_data, dict) and "Error Message" in income_data:
        message = str(income_data.get("Error Message", ""))
        if "Limit Reach" in message:
            raise SourceLimitReachedError("FMP", message)
        return []

    if not income_data:
        return []

    income_frame = pd.DataFrame(income_data)
    balance_frame = pd.DataFrame(balance_data)
    cash_frame = pd.DataFrame(cash_data)
    if income_frame.empty or balance_frame.empty or cash_frame.empty:
        return []

    merged = pd.merge(income_frame, balance_frame, on="date", how="inner", suffixes=("", "_bal"))
    merged = pd.merge(merged, cash_frame, on="date", how="inner", suffixes=("", "_cf"))

    rows: list[dict] = []
    for _, row in merged.iterrows():
        try:
            date_value = datetime.strptime(row["date"], "%Y-%m-%d").date().isoformat()
        except Exception:
            continue

        revenue = float(row.get("revenue")) if pd.notna(row.get("revenue")) else None
        net_income = float(row.get("netIncome")) if pd.notna(row.get("netIncome")) else None
        total_assets = float(row.get("totalAssets")) if pd.notna(row.get("totalAssets")) else None
        total_liabilities = float(row.get("totalLiabilities")) if pd.notna(row.get("totalLiabilities")) else None
        equity_source = row.get("totalStockholdersEquity", row.get("totalEquity"))
        equity = float(equity_source) if pd.notna(equity_source) else None
        eps = float(row.get("eps")) if pd.notna(row.get("eps")) else None
        shares_source = row.get("weightedAverageShsOutDil", row.get("weightedAverageShsOut"))
        shares_issued = float(shares_source) if pd.notna(shares_source) else None
        operating_cash_source = row.get("netCashProvidedByOperatingActivities", row.get("operatingCashFlow"))
        operating_cash_flow = float(operating_cash_source) if pd.notna(operating_cash_source) else None
        operating_income = row.get("operatingIncome")
        interest_expense = row.get("interestExpense")

        roe = (net_income / equity) if net_income and equity else None
        debt_ratio = (total_liabilities / equity) if total_liabilities and equity else None
        interest_coverage = (
            float(operating_income) / abs(float(interest_expense))
            if pd.notna(operating_income) and pd.notna(interest_expense) and abs(float(interest_expense)) > 0
            else None
        )

        rows.append(
            {
                "ticker": ticker.upper(),
                "date": date_value,
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity": equity,
                "shares_issued": shares_issued,
                "eps": eps,
                "roe": roe,
                "debt_ratio": debt_ratio,
                "interest_coverage": interest_coverage,
                "operating_cash_flow": operating_cash_flow,
            }
        )
    return rows
