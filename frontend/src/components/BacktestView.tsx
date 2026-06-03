"use client";

import { FormEvent, Fragment, startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import {
  DisplayTimeframe,
  fetchIndicators,
  fetchStrategyBacktest,
  fetchStrategyScan,
  fetchTickers,
  IndicatorPoint,
  PredictionResult,
  PriceBar,
  StockSummary,
} from "@/api/client";
import MetricCard from "@/components/MetricCard";
import {
  DEFAULT_FEE_BPS,
  getStrategyDefinition,
  SIGNAL_GROUP_DEFAULT_LIMIT,
  SIGNAL_GROUP_MAX_LIMIT,
  SIGNAL_GROUPS,
  SIGNAL_SCAN_TICKERS,
  STRATEGIES,
  isBackendStrategy,
  strategyNeedsBand,
  strategyNeedsLine,
} from "@/lib/backtest/constants";
import {
  evaluateDrawdown,
  evaluateFollowReturn,
  evaluateLossAvoidance,
  evaluateTradeFrequency,
} from "@/lib/backtest/evaluators";
import { runStrategyBacktest } from "@/lib/backtest/simulationEngine";
import { MiniLineChart, PositionStrip } from "@/components/backtest/Charts";
import { getBandWidthState } from "@/lib/backtest/signalBuilder";
import { fetchPriceHistory, loadStrategySignalCard } from "@/lib/backtest/signalCardLoader";
import type {
  BacktestSimulationResult,
  DecisionFactorId,
  SignalGroupId,
  StrategyId,
  StrategySignalCard,
  TradeRecord,
} from "@/lib/backtest/types";
import { sortUniqueByDate } from "@/lib/dateUtils";
import {
  formatNumber,
  formatPercent,
  formatRatioAsPercent,
  formatUnsignedRatioAsPercent,
} from "@/lib/formatters";
import StatusInline from "@/components/StatusInline";
import {
  ApiError,
  classifyApiError,
  describeApiError,
} from "@/lib/apiErrors";

// BacktestView 전용 에러 메시지 (StockView 와 별도 톤).
function extractErrorMessage(error: unknown) {
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    return "백테스트 데이터를 불러오는 중 문제가 생겼습니다.";
  }
  return "백테스트 데이터를 불러오지 못했습니다.";
}

export default function BacktestView() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const timeframe: DisplayTimeframe = "1D";
  const [strategyId, setStrategyId] = useState<StrategyId>("indicator_balance_v2");
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorPoint[]>([]);
  const [lineHistory, setLineHistory] = useState<PredictionResult[]>([]);
  const [bandHistory, setBandHistory] = useState<PredictionResult[]>([]);
  const [result, setResult] = useState<BacktestSimulationResult | null>(null);
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchErrorMessage, setSearchErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [signalCards, setSignalCards] = useState<StrategySignalCard[]>([]);
  const [signalLoading, setSignalLoading] = useState(false);
  const [signalErrorMessage, setSignalErrorMessage] = useState<string | null>(null);
  const [signalApiError, setSignalApiError] = useState<ApiError | null>(null);
  const [backtestApiError, setBacktestApiError] = useState<ApiError | null>(null);
  const [expandedSignalGroups, setExpandedSignalGroups] = useState<Partial<Record<SignalGroupId, boolean>>>({});
  const detailRef = useRef<HTMLDivElement | null>(null);
  const deferredTicker = useDeferredValue(tickerInput);

  useEffect(() => {
    const keyword = deferredTicker.trim();
    if (keyword.length < 1) {
      setSuggestions([]);
      setSearchErrorMessage(null);
      return;
    }

    let active = true;
    setSearchLoading(true);
    setSearchErrorMessage(null);
    fetchTickers({ search: keyword, limit: 6 })
      .then((response) => {
        if (active) {
          setSuggestions(response.data);
        }
      })
      .catch(() => {
        if (active) {
          setSuggestions([]);
          setSearchErrorMessage("티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 백테스트 조회는 계속 가능합니다.");
        }
      })
      .finally(() => {
        if (active) {
          setSearchLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [deferredTicker]);

  useEffect(() => {
    let active = true;
    setSignalLoading(true);
    setSignalErrorMessage(null);
    const signalPromise = isBackendStrategy(strategyId)
      ? fetchStrategyScan(strategyId, { limit: 500 }).then((response) => response.data.cards)
      : Promise.all(SIGNAL_SCAN_TICKERS.map((ticker) => loadStrategySignalCard(ticker, strategyId)));

    signalPromise
      .then((cards) => {
        if (active) {
          setSignalCards(cards);
          setSignalApiError(null);
        }
      })
      .catch((err) => {
        if (active) {
          setSignalCards([]);
          const classified = classifyApiError(err, `/api/v1/strategies/${strategyId}/scan`);
          setSignalApiError(classified);
          setSignalErrorMessage(describeApiError(classified));
        }
      })
      .finally(() => {
        if (active) {
          setSignalLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [strategyId]);

  async function loadBacktest(nextTicker: string, nextTimeframe: DisplayTimeframe, nextStrategyId = strategyId) {
    setIsLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    setResult(null);
    setLineHistory([]);
    setBandHistory([]);
    setBacktestApiError(null);

    try {
      const normalizedTicker = nextTicker.trim().toUpperCase() || "AAPL";
      if (isBackendStrategy(nextStrategyId)) {
        const response = await fetchStrategyBacktest(nextStrategyId, normalizedTicker);
        const nextResult = response.data;
        setResult(nextResult);
        setPriceData(
          nextResult.points.map((point) => ({
            date: point.date,
            open: point.price,
            high: point.price,
            low: point.price,
            close: point.price,
            volume: null,
          }))
        );
        setIndicatorData([]);
        setLineHistory([]);
        setBandHistory([]);
        setStatusMessage(null);
        return;
      }
      const [prices, indicatorsResponse] = await Promise.all([
        fetchPriceHistory(normalizedTicker, nextTimeframe),
        fetchIndicators(normalizedTicker, { timeframe: nextTimeframe, limit: 300 }).catch(() => null),
      ]);
      const indicators = sortUniqueByDate(indicatorsResponse?.data.data ?? []);
      setPriceData(prices);
      setIndicatorData(indicators);

      if (prices.length === 0) {
        setStatusMessage("가격 데이터가 없습니다. 가격 데이터가 연결되면 전략 결과와 차트를 표시합니다.");
        return;
      }

      if (nextTimeframe !== "1D") {
        setStatusMessage(`${getStrategyDefinition(nextStrategyId).label}은 1D 제품 line/band layer로만 계산합니다. ${nextTimeframe}에서는 가격 데이터만 확인할 수 있습니다.`);
        return;
      }

      const needsLine = strategyNeedsLine(nextStrategyId);
      const needsBand = strategyNeedsBand(nextStrategyId);
      // legacy /predictions/history endpoint 제거됨. lens_balance_v1 류 strategy 는 v1 endpoint 로 옮겨야 한다.
      const [lineResponse, bandResponse] = await Promise.all([
        Promise.resolve(null as { data: PredictionResult[] } | null),
        Promise.resolve(null as { data: PredictionResult[] } | null),
      ]);
      const lineRows = lineResponse?.data ?? [];
      const bandRows = bandResponse?.data ?? [];
      setLineHistory(lineRows);
      setBandHistory(bandRows);

      if ((needsLine && lineRows.length === 0) || (needsBand && bandRows.length === 0)) {
        const missingParts = [
          needsLine && lineRows.length === 0 ? "보수적 기준선" : null,
          needsBand && bandRows.length === 0 ? "AI 밴드" : null,
        ].filter(Boolean);
        setStatusMessage(`이 티커에는 아직 ${missingParts.join(", ")} 저장 결과가 없습니다. 가격 데이터만 표시합니다.`);
        return;
      }

      const nextResult = runStrategyBacktest({
        strategyId: nextStrategyId,
        priceRows: prices,
        lineHistory: lineRows,
        bandHistory: bandRows,
        indicators,
        feeBps: DEFAULT_FEE_BPS,
      });
      setResult(nextResult);
      setStatusMessage(nextResult ? null : "전략 계산에 필요한 날짜가 맞지 않아 결과를 계산할 수 없습니다.");
    } catch (error) {
      setPriceData([]);
      setIndicatorData([]);
      setResult(null);
      const classified = classifyApiError(error, `/api/v1/strategies/${nextStrategyId}/backtest/${nextTicker}`);
      setBacktestApiError(classified);
      setErrorMessage(describeApiError(classified));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadBacktest("AAPL", "1D", "indicator_balance_v2");
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextTicker = tickerInput.trim().toUpperCase() || "AAPL";
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadBacktest(nextTicker, timeframe, strategyId);
    });
    window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
  }

  function handleTickerSelect(nextTicker: string, scrollToDetail = true) {
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadBacktest(nextTicker, timeframe, strategyId);
    });
    if (scrollToDetail) {
      window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    }
  }

  function handleStrategyChange(nextStrategyId: StrategyId) {
    setStrategyId(nextStrategyId);
    startTransition(() => {
      void loadBacktest(selectedTicker, timeframe, nextStrategyId);
    });
  }

  function toggleSignalGroup(groupId: SignalGroupId) {
    setExpandedSignalGroups((current) => ({
      ...current,
      [groupId]: !current[groupId],
    }));
  }

  const priceSeries = useMemo(
    () =>
      priceData.map((row) => ({
        date: row.date,
        value: row.close,
      })),
    [priceData]
  );
  const strategyEquitySeries = useMemo(
    () => result?.points.map((point) => ({ date: point.date, value: (point.strategyEquity - 1) * 100 })) ?? [],
    [result]
  );
  const buyHoldEquitySeries = useMemo(
    () => result?.points.map((point) => ({ date: point.date, value: (point.buyHoldEquity - 1) * 100 })) ?? [],
    [result]
  );
  const priceByDate = useMemo(() => new Map(priceData.map((row) => [row.date, row])), [priceData]);
  const tradeRecords = useMemo<TradeRecord[]>(() => {
    if (!result) {
      return [];
    }
    let previousPosition: 0 | 1 = 0;
    return result.signals.map((signal) => {
      const priceIndex = priceData.findIndex((row) => row.date === signal.date);
      const price = priceIndex >= 0 ? priceData[priceIndex] : priceByDate.get(signal.date);
      const nextPrice = priceIndex >= 0 ? priceData[priceIndex + 1] : undefined;
      const action =
        signal.position === 1 && previousPosition === 0
          ? "매수"
          : signal.position === 0 && previousPosition === 1
            ? "매도"
            : signal.position === 1
              ? "보유"
              : "대기";
      previousPosition = signal.position;
      return {
        date: signal.date,
        action,
        price: price?.close ?? null,
        reason: signal.reason,
        nextDayReturn: price && nextPrice ? nextPrice.close / price.close - 1 : null,
        conservativeReturn: signal.conservativeReturn,
        lowerBandReturn: signal.lowerBandReturn,
        bandWidthReturn: signal.bandWidthReturn,
        bandWidthExpansion: signal.bandWidthExpansion,
        bandWidthPercentile: signal.bandWidthPercentile,
        ma60Ratio: signal.ma60Ratio,
        ma20Ratio: signal.ma20Ratio,
        macdRatio: signal.macdRatio,
        rsi: signal.rsi,
        atrRatio: signal.atrRatio,
      };
    });
  }, [priceData, priceByDate, result]);
  const executionRecords = tradeRecords.filter((record) => record.action === "매수" || record.action === "매도");
  const latestTradeRecord = tradeRecords.length > 0 ? tradeRecords[tradeRecords.length - 1] : null;
  const strategyDefinition = getStrategyDefinition(strategyId);
  const groupedSignalCards = useMemo(
    () =>
      SIGNAL_GROUPS.map((group) => {
        const allCards = signalCards.filter((card) => card.group === group.id);
        const isExpanded = Boolean(expandedSignalGroups[group.id]);
        const visibleLimit = isExpanded ? SIGNAL_GROUP_MAX_LIMIT : SIGNAL_GROUP_DEFAULT_LIMIT[group.id];
        return {
          ...group,
          allCards,
          cards: allCards.slice(0, visibleLimit),
          isExpanded,
          canToggle: allCards.length > SIGNAL_GROUP_DEFAULT_LIMIT[group.id] || group.id === "watch",
        };
      }),
    [expandedSignalGroups, signalCards]
  );
  const usableSignalCount = signalCards.filter((card) => card.hasUsableSignal).length;
  const scanScopeCount = isBackendStrategy(strategyId) ? signalCards.length : SIGNAL_SCAN_TICKERS.length;
  const selectedSignalCard = signalCards.find((card) => card.ticker === selectedTicker);
  const latestSignalDate = signalCards
    .map((card) => card.asofDate)
    .filter((date): date is string => Boolean(date))
    .sort()
    .at(-1);
  const visibleDecisionFactors = latestTradeRecord
    ? [
        {
          id: "conservative",
          label: "보수적 기준선",
          value: formatRatioAsPercent(latestTradeRecord.conservativeReturn),
          description: "-0.2% 이상이면 진입 후보, -1.4% 이상이면 보유 후보로 봅니다.",
        },
        {
          id: "lowerBand",
          label: "AI 밴드 하단 위험",
          value: formatRatioAsPercent(latestTradeRecord.lowerBandReturn),
          description: "-5% 이하이면 하방 위험이 깊어진 상태로 봅니다.",
        },
        {
          id: "bandWidth",
          label: "AI 밴드 폭",
          value: formatUnsignedRatioAsPercent(latestTradeRecord.bandWidthReturn),
          description: "AI가 보는 예상 변동 범위의 넓이입니다.",
        },
        {
          id: "bandExpansion",
          label: "밴드 폭 확장",
          value: latestTradeRecord.bandWidthExpansion == null ? "-" : `${formatNumber(latestTradeRecord.bandWidthExpansion, 2)}배`,
          description: "최근 20개 신호의 밴드 폭 중앙값 대비 1.10배 이상이면 신규 진입을 막습니다.",
        },
        {
          id: "bandPercentile",
          label: "밴드 폭 위치",
          value: latestTradeRecord.bandWidthPercentile == null ? "-" : formatUnsignedRatioAsPercent(latestTradeRecord.bandWidthPercentile),
          description: "최근 history 안에서 밴드 폭이 어느 정도 넓은 편인지 보는 참고값입니다.",
        },
        {
          id: "ma60Trend",
          label: "60일 추세",
          value: formatRatioAsPercent(latestTradeRecord.ma60Ratio),
          description: "+2% 이상이면 상승 추세로 보고, -5% 이하이면 방어 후보로 봅니다.",
        },
        {
          id: "ma20Trend",
          label: "20일 추세",
          value: formatRatioAsPercent(latestTradeRecord.ma20Ratio),
          description: "단기 흐름이 크게 무너지면 청산 후보로 봅니다.",
        },
        {
          id: "macd",
          label: "MACD",
          value: formatRatioAsPercent(latestTradeRecord.macdRatio),
          description: "0보다 크면 추세 모멘텀이 살아 있는 쪽으로 봅니다.",
        },
        {
          id: "rsi",
          label: "RSI",
          value: formatNumber(latestTradeRecord.rsi, 1),
          description: "75 이상이면 신규 진입을 조심합니다.",
        },
        {
          id: "atr",
          label: "ATR",
          value: formatRatioAsPercent(latestTradeRecord.atrRatio),
          description: "변동성이 커질 때 포지션 위험을 확인합니다.",
        },
      ].filter((factor) => strategyDefinition.visibleFactors.includes(factor.id as DecisionFactorId))
    : [];

  return (
    <div className="view-stack">
      <header className="view-header">
        <div className="view-header__title">
          <div className="eyebrow">백테스트</div>
          <h1>전략 신호</h1>
          <p>전략과 티커를 먼저 선택하고, 아래쪽 500티커 신호판은 같은 룰을 각 티커에 독립 적용한 최신 상태로 확인합니다.</p>
        </div>
        <div className="status-badge status-badge--neutral">{strategyDefinition.label}</div>
      </header>

      <section className="panel strategy-signal-overview">
        <div>
          <div className="eyebrow">오늘의 신호 요약</div>
          <h2>{strategyDefinition.label} 기준 신호</h2>
          <p>
            {strategyDefinition.usesAi
              ? "저장된 1D 제품 예측선과 AI 밴드 history가 있는 종목만 읽어 최신 전략 상태를 요약합니다."
              : "AI 예측 없이 가격과 보조지표만 읽어 최신 전략 상태를 요약합니다."}
            {" "}포트폴리오 추천이 아니라 단일 티커 백테스트로 이어지는 신호 탐색 화면입니다.
          </p>
          <p className="strategy-signal-scope-note">
            {isBackendStrategy(strategyId)
              ? "현재 백엔드 로컬 parquet 기준 단일 티커 long/cash 신호입니다. 500개 내외 티커에 같은 룰을 각각 적용하고, 종목을 선택하면 그 티커 하나의 상세 백테스트를 봅니다."
              : "현재는 12개 주요 종목 기준 전략 신호입니다. 전체 scanner API가 붙기 전까지는 subset 신호로만 봅니다."}
          </p>
        </div>
        <div className="strategy-signal-summary">
          <div>
            <span>스캔 범위</span>
            <strong>{scanScopeCount}개</strong>
          </div>
          <div>
            <span>신호 확인</span>
            <strong>{signalLoading ? "조회 중" : `${usableSignalCount}개`}</strong>
          </div>
          <div>
            <span>기준일</span>
            <strong>{latestSignalDate ?? "-"}</strong>
          </div>
          <div>
            <span>전략 단위</span>
            <strong>1D</strong>
          </div>
        </div>
      </section>

      {signalApiError ? (
        <StatusInline
          kind="error"
          label="전략 신호"
          error={signalApiError}
          hint="백엔드 로그 또는 strategy_scan endpoint 확인"
        />
      ) : signalErrorMessage ? (
        <div className="notice notice--error">{signalErrorMessage}</div>
      ) : null}

      <section className="strategy-signal-board" aria-label="Lens Balance v1 전략 신호 그룹">
        {groupedSignalCards.map((group) => (
          <section className={`panel signal-group-panel signal-group-panel--${group.id}`} key={group.id}>
            <div className="panel-heading">
              <div className="eyebrow">전략 신호</div>
              <div className="signal-group-heading-row">
                <h2>{group.title}</h2>
                <span>{group.allCards.length}개</span>
              </div>
              <p>{group.description}</p>
            </div>
            <div className="signal-card-list">
              {signalLoading ? (
                <div className="compact-note">신호를 불러오는 중입니다.</div>
              ) : group.cards.length > 0 ? (
                group.cards.map((card) => (
                  <button
                    type="button"
                    key={`${group.id}-${card.ticker}`}
                    className={`signal-card signal-card--${group.id} ${selectedTicker === card.ticker ? "is-selected" : ""}`}
                    onClick={() => handleTickerSelect(card.ticker)}
                    aria-pressed={selectedTicker === card.ticker}
                  >
                    <div className="signal-card__top">
                      <strong>{card.ticker}</strong>
                      <span>{card.sector ?? "종목명 없음"}</span>
                    </div>
                    <div className="signal-card__label">{card.signalLabel}</div>
                    <dl>
                      {strategyNeedsLine(strategyId) ? (
                        <>
                          <div>
                            <dt>예측선</dt>
                            <dd>{formatRatioAsPercent(card.conservativeReturn)}</dd>
                          </div>
                          <div>
                            <dt>하단 위험</dt>
                            <dd>{formatRatioAsPercent(card.lowerBandReturn)}</dd>
                          </div>
                          <div>
                            <dt>밴드 폭</dt>
                            <dd>{getBandWidthState(card)}</dd>
                          </div>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                        </>
                      ) : strategyNeedsBand(strategyId) ? (
                        <>
                          <div>
                            <dt>하단 위험</dt>
                            <dd>{formatRatioAsPercent(card.lowerBandReturn)}</dd>
                          </div>
                          <div>
                            <dt>밴드 폭</dt>
                            <dd>{getBandWidthState(card)}</dd>
                          </div>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                          <div>
                            <dt>RSI</dt>
                            <dd>{formatNumber(card.rsi, 1)}</dd>
                          </div>
                        </>
                      ) : (
                        <>
                          <div>
                            <dt>60일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma60Ratio)}</dd>
                          </div>
                          <div>
                            <dt>20일 추세</dt>
                            <dd>{formatRatioAsPercent(card.ma20Ratio)}</dd>
                          </div>
                          <div>
                            <dt>RSI</dt>
                            <dd>{formatNumber(card.rsi, 1)}</dd>
                          </div>
                          <div>
                            <dt>ATR</dt>
                            <dd>{formatRatioAsPercent(card.atrRatio)}</dd>
                          </div>
                        </>
                      )}
                    </dl>
                    <p>{card.hasUsableSignal ? card.reason : "아직 신호 없음"}</p>
                  </button>
                ))
              ) : group.id === "watch" && group.allCards.length > 0 ? (
                <div className="compact-note">관망 종목은 접혀 있습니다.</div>
              ) : (
                <div className="empty-state">아직 신호 없음</div>
              )}
            </div>
            {!signalLoading && group.canToggle ? (
              <button type="button" className="signal-group-toggle" onClick={() => toggleSignalGroup(group.id)}>
                {group.isExpanded ? "접기" : group.id === "watch" ? "관망 보기" : `더 보기 (${Math.min(group.allCards.length, SIGNAL_GROUP_MAX_LIMIT)}개)`}
              </button>
            ) : null}
          </section>
        ))}
      </section>

      <section className="panel direct-ticker-panel">
        <div className="panel-heading">
          <div className="eyebrow">전략 및 티커 검색</div>
          <h2>먼저 종목을 직접 확인</h2>
          <p>AAPL처럼 현재 상위 후보에 없는 종목도 직접 조회해 전략 편입 여부와 상세 백테스트를 확인합니다.</p>
        </div>
        <form className="backtest-toolbar backtest-toolbar--compact" onSubmit={handleSubmit}>
          <label className="search-field">
            <span>티커</span>
            <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} />
          </label>
          <label className="select-field">
            <span>전략</span>
            <select value={strategyId} onChange={(event) => handleStrategyChange(event.target.value as StrategyId)}>
              {STRATEGIES.map((strategy) => (
                <option key={strategy.id} value={strategy.id}>
                  {strategy.label}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="primary-button primary-button--compact" disabled={isLoading}>
            조회
          </button>
          <div className="fee-pill">
            <span>전략</span>
            <strong>{strategyDefinition.label}</strong>
            <em>1D / {formatNumber(DEFAULT_FEE_BPS, 1)}bp</em>
          </div>
        </form>

        {suggestions.length > 0 ? (
          <div className="suggestion-row">
            {suggestions.map((item) => (
              <button key={item.ticker} type="button" onClick={() => handleTickerSelect(item.ticker)}>
                <strong>{item.ticker}</strong>
                <span>{item.sector ?? "섹터 없음"}</span>
              </button>
            ))}
          </div>
        ) : searchLoading ? (
          <div className="compact-note">검색 중입니다.</div>
        ) : null}
        {searchErrorMessage ? <div className="compact-note compact-note--warning">{searchErrorMessage}</div> : null}
      </section>

      <section className="panel strategy-rule-panel">
        <div>
          <div className="eyebrow">전략 설명</div>
          <h2>{strategyDefinition.label}</h2>
          <p>{strategyDefinition.description}</p>
          <p className="compact-note">
            이 화면은 매수·매도 추천이 아니라 룰 기반 시뮬레이션입니다. 현금 보유는 전략 기준이 약할 때 시장 참여를 잠시 줄이는 방식입니다.
          </p>
          <p className="compact-note compact-note--warning">{strategyDefinition.scopeNote}</p>
          <div className="strategy-validation-strip" aria-label={`${strategyDefinition.label} 검증 결과`}>
            {strategyDefinition.validationRows.map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="rule-grid">
          {strategyDefinition.ruleRows.map(([label, value]) => (
            <Fragment key={`${label}-${value}`}>
              <span>{label}</span>
              <strong>{value}</strong>
            </Fragment>
          ))}
        </div>
      </section>

      {backtestApiError ? (
        <StatusInline
          kind="error"
          label="백테스트 실행"
          error={backtestApiError}
          hint="ticker/strategy 조합 또는 백엔드 로그 확인"
        />
      ) : errorMessage ? (
        <div className="notice notice--error">{errorMessage}</div>
      ) : null}
      {statusMessage ? <div className="notice">{statusMessage}</div> : null}

      <div className="backtest-detail-anchor" ref={detailRef}>
        <section className="panel selected-backtest-heading">
          <div>
            <div className="eyebrow">상세 백테스트</div>
            <h2>{selectedTicker} 단일 티커 상세</h2>
            <p>
              {selectedSignalCard
                ? `${selectedSignalCard.signalLabel} · 기준일 ${selectedSignalCard.asofDate ?? "-"}`
                : "직접 입력한 종목입니다. 저장된 예측이 있으면 전략 결과를 계산합니다."}
            </p>
          </div>
          <div className="status-badge status-badge--neutral">{result ? "상세 계산 완료" : isLoading ? "계산 중" : "결과 대기"}</div>
        </section>

        <section className="metric-grid metric-grid--six">
          <MetricCard label="전략 수익률" value={formatPercent(result?.strategyReturnPct)} />
          <MetricCard label="단순 보유 수익률" value={formatPercent(result?.buyHoldReturnPct)} />
          <MetricCard label="전략 MDD" value={formatPercent(result?.maxDrawdownPct)} tone="bad" />
          <MetricCard label="단순 보유 MDD" value={formatPercent(result?.buyHoldMaxDrawdownPct)} tone="bad" />
          <MetricCard label="손실 회피율" value={result?.largeLossAvoidanceRate == null ? "-" : formatUnsignedRatioAsPercent(result.largeLossAvoidanceRate)} />
          <MetricCard label="시장 참여율" value={result ? formatUnsignedRatioAsPercent(result.marketParticipationRate) : "-"} />
        </section>
      </div>

      <section className="metric-grid metric-grid--six">
        <MetricCard label="전략 Sharpe" value={formatNumber(result?.feeAdjustedSharpe)} />
        <MetricCard label="단순 보유 Sharpe" value={formatNumber(result?.buyHoldSharpe)} />
        <MetricCard label="MDD 개선폭" value={formatPercent(result?.maxDrawdownImprovementPct)} />
        <MetricCard label="거래 횟수" value={formatNumber(result?.tradeCount, 0)} />
        <MetricCard label="평균 보유 기간" value={result?.averageHoldingDays == null ? "-" : `${formatNumber(result.averageHoldingDays, 1)}일`} />
        <MetricCard label="큰 하락 기준" value={formatPercent(result?.largeLossThresholdPct)} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div className="eyebrow">전략 평가</div>
          <h2>{strategyDefinition.label} 현재 티커 기준 점검</h2>
        </div>
        <div className="strategy-assessment-grid">
          <div>
            <span>수익 추종</span>
            <strong>{evaluateFollowReturn(result)}</strong>
          </div>
          <div>
            <span>낙폭 완화</span>
            <strong>{evaluateDrawdown(result)}</strong>
          </div>
          <div>
            <span>손실 회피</span>
            <strong>{evaluateLossAvoidance(result)}</strong>
          </div>
          <div>
            <span>거래 빈도</span>
            <strong>{evaluateTradeFrequency(result)}</strong>
          </div>
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">가격과 매매</div>
            <h2>{selectedTicker} 가격 차트</h2>
          </div>
          <MiniLineChart
            title="가격"
            lines={[{ label: selectedTicker, color: "#1f2937", values: priceSeries }]}
            markers={result?.tradeEvents}
          />
          {result ? (
            <details className="backtest-detail">
              <summary>매매 구간 보기</summary>
              <PositionStrip points={result.points} />
              <div className="position-strip__legend">
                <span><i className="is-hold" />보유 구간</span>
                <span><i className="is-cash" />현금 대기 구간</span>
                <span>가격 차트의 점은 매수/매도 시점입니다.</span>
              </div>
              {result.tradeEvents.length > 0 ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>날짜</th>
                        <th>행동</th>
                        <th>가격</th>
                        <th>이유</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.tradeEvents.slice(-10).reverse().map((event) => (
                        <tr key={`${event.kind}-${event.date}-${event.price}`}>
                          <td>{event.date}</td>
                          <td>{event.kind === "entry" ? "매수" : "매도"}</td>
                          <td>{formatNumber(event.price, 2)}</td>
                          <td>{event.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state">이 기간에는 매수/매도 전환이 없습니다.</div>
              )}
            </details>
          ) : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">성과 비교</div>
            <h2>전략 vs 단순 보유</h2>
          </div>
          {result ? (
            <MiniLineChart
              title="누적 수익률"
              lines={[
                { label: strategyDefinition.shortLabel, color: "#0f766e", values: strategyEquitySeries },
                { label: "단순 보유", color: "#64748b", values: buyHoldEquitySeries },
              ]}
            />
          ) : (
            <div className="empty-state">{isLoading ? "조회 중입니다." : "전략 결과를 계산할 수 없습니다."}</div>
          )}
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">전략 판단 요소</div>
            <h2>최근 판단 입력값</h2>
          </div>
          {latestTradeRecord ? (
            <div className="decision-factor-grid">
              {visibleDecisionFactors.map((factor) => (
                <div key={factor.id}>
                  <span>{factor.label}</span>
                  <strong>{factor.value}</strong>
                  <p>{factor.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">표시할 전략 판단 요소가 없습니다.</div>
          )}
        </div>
        <div className="panel">
          <div className="panel-heading">
            <div className="eyebrow">매매 기록</div>
            <h2>최근 매매 기록</h2>
          </div>
          {executionRecords.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>날짜</th>
                    <th>행동</th>
                    <th>가격</th>
                    <th>이유</th>
                    <th>다음 날 수익률</th>
                  </tr>
                </thead>
                <tbody>
                  {executionRecords.slice(-10).reverse().map((record) => (
                    <tr key={`${record.date}-${record.action}`}>
                      <td>{record.date}</td>
                      <td>{record.action}</td>
                      <td>{formatNumber(record.price, 2)}</td>
                      <td>{record.reason}</td>
                      <td>{formatRatioAsPercent(record.nextDayReturn)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">이 기간에는 매수/매도 전환이 없습니다.</div>
          )}
        </div>
      </section>
    </div>
  );
}
