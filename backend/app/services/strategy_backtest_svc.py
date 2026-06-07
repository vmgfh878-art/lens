"""
전략 스캔 / 백테스트 — 공개 API facade.

이 파일은 라우터 (strategies.py, admin.py) 와의 import 계약을 유지하기 위한
얇은 facade. 실제 구현은 CP226 분리 산출 3 모듈에 있다:
  - strategy_indicators       : 전략 규칙 / 지표 / 상태머신 (순수)
  - strategy_backtest_engine  : 수익률 / 드로우다운 / 샤프 / 시뮬레이션 지표 (순수)
  - strategy_scan             : parquet I/O / lru_cache / 티커별 집계 (상태)

기존 caller (라우터) 의 import 경로 (`app.services.strategy_backtest_svc.X`) 와
공개 + 사설 심볼 (`STRATEGIES`, `StrategyRule`, `get_strategy_scan`,
`get_strategy_backtest`, `clear_strategy_cache`, 그리고 일부 진단 / 호환 목적의
사설 심볼) 을 보존한다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from app.services.strategy_backtest_engine import (
    FEE_RATE,  # noqa: F401 (re-export)
    _average_holding_days,  # noqa: F401 (re-export)
    _large_loss_threshold,  # noqa: F401 (re-export)
    _max_drawdown,  # noqa: F401 (re-export)
    _points,
    _sharpe,  # noqa: F401 (re-export)
    _signal_row,
    _sortino,  # noqa: F401 (re-export)
    _ticker_metrics,
    _total_return,  # noqa: F401 (re-export)
    _trade_events,
)
from app.services.strategy_indicators import (
    _align_date_dtype,  # noqa: F401 (re-export)
    _compute_signal_frame,  # noqa: F401 (re-export)
    _jsonable,
    _normalize_rsi,  # noqa: F401 (re-export)
    _raw_target,  # noqa: F401 (re-export)
    _reason,  # noqa: F401 (re-export)
    _safe,  # noqa: F401 (re-export)
)
from app.services.strategy_scan import (
    MIN_EVAL_DAYS,  # noqa: F401 (re-export)
    _backtest_signal_frame,
    _data_dir,  # noqa: F401 (re-export)
    _load_base_frame,
    _load_frame,  # noqa: F401 (re-export)
    _scan_results,
    _sector_map,
)

# 전략 정의 (StrategyRule + STRATEGIES) 는 strategy_rules.py 에서 단일 관리한다 (모델 급 관리).
# 이 파일은 전략의 "실행" (pandas 조건 계산 + 백테스트) 만 담당한다.
from app.strategies.strategy_rules import (
    STRATEGIES,  # noqa: F401 (re-export for router/admin caller)
    StrategyRule,
)


def get_strategy_scan(strategy_id: str, limit: int = 500) -> dict[str, Any]:
    result = _scan_results(strategy_id)
    rule: StrategyRule = result["rule"]
    sectors = _sector_map()
    cards = []
    for ticker, row in result["latest_rows"].items():
        has_usable_signal = bool(pd.notna(row.get("close")))
        cards.append(
            {
                "ticker": ticker,
                "sector": sectors.get(ticker),
                "group": str(row["signal_group"]),
                "signalLabel": str(row["signal_label"]),
                "reason": str(row["reason"]),
                "asofDate": pd.Timestamp(row["date"]).date().isoformat(),
                "conservativeReturn": _jsonable(row.get("line_score")) if rule.uses_line else None,
                "lowerBandReturn": _jsonable(row.get("band_lower_return"))
                if rule.uses_band
                else None,
                "bandWidthReturn": _jsonable(row.get("band_width_return"))
                if rule.uses_band
                else None,
                "bandWidthExpansion": _jsonable(row.get("band_width_expansion"))
                if rule.uses_band
                else None,
                "bandWidthPercentile": _jsonable(row.get("band_width_percentile"))
                if rule.uses_band
                else None,
                "ma60Ratio": _jsonable(row.get("ma_60_ratio")),
                "ma20Ratio": _jsonable(row.get("ma_20_ratio")),
                "macdRatio": _jsonable(row.get("macd_ratio")),
                "rsi": _jsonable(row.get("rsi_norm")),
                "atrRatio": _jsonable(row.get("atr_ratio_calc")),
                "strategyScore": _jsonable(row.get("line_score"))
                if rule.uses_line
                else _jsonable(row.get("ma_60_ratio")),
                "hasUsableSignal": has_usable_signal,
            }
        )
    group_order = {"buy": 0, "hold": 1, "risk": 2, "watch": 3}
    cards.sort(
        key=lambda row: (
            group_order.get(row["group"], 9),
            -(row["strategyScore"] or -9999.0),
            row["ticker"],
        )
    )
    return {
        "strategyId": rule.id,
        "strategyLabel": rule.label,
        "timeframe": "1D",
        "asofDate": pd.Timestamp(result["end_date"]).date().isoformat(),
        "scopeTickerCount": len(result["latest_rows"]),
        "usableSignalCount": sum(1 for card in cards if card["hasUsableSignal"]),
        "latestValidTickerCount": sum(1 for card in cards if card["hasUsableSignal"]),
        "cards": cards[:limit],
        "portfolioMetrics": result["aggregate"],
        "aggregateMetrics": result["aggregate"],
        "contract": "single_ticker_long_cash_average",
    }


def clear_strategy_cache() -> None:
    """admin/reload 에서 호출. parquet 갱신 후 전략 frame/결과/sector cache 를 비운다.
    이게 안 되면 새 데이터를 받아도 전략은 옛 결과를 계속 노출한다."""
    _load_base_frame.cache_clear()
    _load_frame.cache_clear()
    _scan_results.cache_clear()
    _sector_map.cache_clear()


def get_strategy_backtest(strategy_id: str, ticker: str) -> dict[str, Any]:
    rule, signal_frame = _backtest_signal_frame(strategy_id, ticker)
    metrics = _ticker_metrics(signal_frame)
    return {
        "points": _points(signal_frame),
        "signals": [_signal_row(row, rule) for _, row in signal_frame.iterrows()],
        "tradeEvents": _trade_events(signal_frame),
        **metrics,
    }
