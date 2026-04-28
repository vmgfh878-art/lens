"use client";

import { FormEvent, startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  DisplayTimeframe,
  EvaluationSummary,
  fetchAiRuns,
  fetchPrediction,
  fetchPrices,
  fetchRunEvaluations,
  fetchTickers,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
  StockSummary,
} from "@/api/client";
import Chart from "@/components/Chart";
import LayerToggle from "@/components/LayerToggle";

type ChartType = "candles" | "line";

interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

interface ApiErrorShape {
  code: string;
  message: string;
}

const TIMEFRAME_OPTIONS: Array<{ value: DisplayTimeframe; label: string }> = [
  { value: "1D", label: "1D" },
  { value: "1W", label: "1W" },
  { value: "1M", label: "1M" },
];

const AI_RUN_SCAN_LIMIT = 10;

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
}

function formatCompactNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatRatio(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  if (Math.abs(value) <= 1) {
    return `${formatNumber(value * 100, 1)}%`;
  }
  return formatNumber(value, 2);
}

function extractApiError(error: unknown): ApiErrorShape | null {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const payload = (error as { response?: { data?: { error?: { code?: string; message?: string } } } }).response?.data
      ?.error;
    if (payload?.code && payload?.message) {
      return {
        code: payload.code,
        message: payload.message,
      };
    }
  }
  return null;
}

function extractErrorMessage(error: unknown) {
  const apiError = extractApiError(error);
  if (apiError) {
    return apiError.message;
  }
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. 127.0.0.1:8000 서버가 켜져 있는지 확인해주세요.";
    }
    return error.message;
  }
  return "데이터를 불러오지 못했습니다.";
}

function getLastPrice(rows: PriceBar[]) {
  return rows.length > 0 ? rows[rows.length - 1] : null;
}

function getChangePercent(rows: PriceBar[]) {
  if (rows.length < 2) {
    return null;
  }
  const latest = rows[rows.length - 1].close;
  const previous = rows[rows.length - 2].close;
  if (!previous) {
    return null;
  }
  return ((latest - previous) / previous) * 100;
}

function getLastFinite(values: number[] | null | undefined) {
  if (!values) {
    return null;
  }
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(values[index])) {
      return values[index];
    }
  }
  return null;
}

function selectEvaluation(rows: EvaluationSummary[], prediction: PredictionResult) {
  return rows.find((row) => row.asof_date === prediction.asof_date) ?? rows[0] ?? null;
}

function buildAiState(timeframe: DisplayTimeframe): AiState {
  if (timeframe === "1M") {
    return {
      kind: "disabled",
      message: "월봉은 가격 지표만 제공됩니다.",
    };
  }
  return {
    kind: "empty",
    message: "저장된 최신 예측 결과가 아직 없습니다.",
  };
}

export default function StockView() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [timeframe, setTimeframe] = useState<DisplayTimeframe>("1D");
  const [chartType, setChartType] = useState<ChartType>("candles");
  const [layers, setLayers] = useState({
    indicators: false,
    aiBand: true,
    conservativeLine: true,
  });
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [aiState, setAiState] = useState<AiState>({ kind: "loading", message: "예측 확인 중" });
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchErrorMessage, setSearchErrorMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
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
          setSearchErrorMessage(null);
        }
      })
      .catch(() => {
        if (active) {
          setSuggestions([]);
          setSearchErrorMessage("티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 가격 조회는 계속 가능합니다.");
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

  async function loadStock(nextTicker: string, nextTimeframe: DisplayTimeframe) {
    setIsLoading(true);
    setErrorMessage(null);
    setAiState({ kind: "loading", message: "예측 확인 중" });

    try {
      const pricesResponse = await fetchPrices(nextTicker, { timeframe: nextTimeframe });
      setPriceData(pricesResponse.data.data);

      if (!isPredictionTimeframeEnabled(nextTimeframe)) {
        setPrediction(null);
        setEvaluation(null);
        setAiState(buildAiState(nextTimeframe));
        return;
      }

      const runsResponse = await fetchAiRuns({
        modelName: "patchtst",
        timeframe: nextTimeframe,
        status: "completed",
        limit: AI_RUN_SCAN_LIMIT,
      }).catch((error) => {
        setPrediction(null);
        setEvaluation(null);
        setAiState({ kind: "empty", message: `예측 조회 오류: ${extractErrorMessage(error)}` });
        return null;
      });
      if (!runsResponse) {
        return;
      }

      const completedRuns = runsResponse.data;
      if (completedRuns.length === 0) {
        setPrediction(null);
        setEvaluation(null);
        setAiState({ kind: "empty", message: "완료된 예측 run이 없어 예측 범위를 표시할 수 없습니다." });
        return;
      }

      let missingPredictionCount = 0;
      let predictionApiFailed = false;
      for (const run of completedRuns) {
        const predictionResponse = await fetchPrediction(nextTicker, {
          timeframe: nextTimeframe,
          runId: run.run_id,
        }).catch((error) => {
          const apiError = extractApiError(error);
          if (apiError?.code === "RESOURCE_NOT_FOUND") {
            missingPredictionCount += 1;
            return null;
          }
          if (apiError?.code === "INSUFFICIENT_HISTORY") {
            missingPredictionCount += 1;
            return null;
          }
          if (apiError?.code === "TIMEFRAME_DISABLED") {
            setPrediction(null);
            setEvaluation(null);
            setAiState(buildAiState(nextTimeframe));
            return null;
          }
          setPrediction(null);
          setEvaluation(null);
          setAiState({ kind: "empty", message: `예측 API 오류: ${extractErrorMessage(error)}` });
          predictionApiFailed = true;
          return null;
        });

        if (predictionApiFailed) {
          return;
        }
        if (!predictionResponse) {
          continue;
        }

        setPrediction(predictionResponse.data);
        try {
          const evaluationsResponse = await fetchRunEvaluations(predictionResponse.data.run_id, {
            ticker: nextTicker,
            timeframe: nextTimeframe,
            limit: 20,
          });
          setEvaluation(selectEvaluation(evaluationsResponse.data, predictionResponse.data));
        } catch {
          setEvaluation(null);
        }
        setAiState({ kind: "ready", message: "저장된 예측 연결됨" });
        return;
      }

      setPrediction(null);
      setEvaluation(null);
      setAiState({
        kind: "empty",
        message:
          missingPredictionCount > 0
            ? "저장된 예측 없음: completed run에 이 티커 prediction row가 없습니다."
            : "저장된 예측 없음",
      });
    } catch (error) {
      setPriceData([]);
      setPrediction(null);
      setEvaluation(null);
      setAiState({ kind: "empty", message: "가격 데이터 또는 예측 상태를 확인할 수 없습니다." });
      setErrorMessage(extractErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadStock("AAPL", "1D");
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextTicker = tickerInput.trim().toUpperCase() || "AAPL";
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTickerSelect(nextTicker: string) {
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTimeframeChange(nextTimeframe: DisplayTimeframe) {
    setTimeframe(nextTimeframe);
    setLayers((previous) => ({
      ...previous,
      aiBand: nextTimeframe !== "1M",
      conservativeLine: nextTimeframe !== "1M",
    }));
    startTransition(() => {
      void loadStock(selectedTicker, nextTimeframe);
    });
  }

  const latestPrice = getLastPrice(priceData);
  const changePercent = getChangePercent(priceData);
  const aiDisabled = timeframe === "1M";
  const conservativeValue = getLastFinite(prediction?.line_series?.length ? prediction.line_series : prediction?.conservative_series);
  const changeClass = changePercent == null ? "is-flat" : changePercent < 0 ? "is-down" : "is-up";
  const predictionStatus =
    aiState.kind === "ready" ? "표시 중" : aiState.kind === "disabled" ? "가격 전용" : aiState.kind === "loading" ? "확인 중" : "없음";

  return (
    <div className="view-stack stock-view">
      <section className="stock-topbar">
        <form className="stock-topbar__form" onSubmit={handleSubmit}>
          <div className="ticker-box">
            <label className="ticker-field">
              <span>티커</span>
              <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} />
            </label>
            <button type="submit" className="primary-button primary-button--compact" disabled={isLoading}>
              조회
            </button>
          </div>

          <div className="quote-strip" aria-label="현재 시세">
            <div className="quote-item quote-item--ticker">
              <span>종목</span>
              <strong>{selectedTicker}</strong>
            </div>
            <div className="quote-item quote-item--price">
              <span>현재가</span>
              <strong>{formatNumber(latestPrice?.close)}</strong>
            </div>
            <div className={`quote-item quote-item--change ${changeClass}`}>
              <span>등락률</span>
              <strong>{formatPercent(changePercent)}</strong>
            </div>
            <div className="quote-item">
              <span>거래량</span>
              <strong>{formatCompactNumber(latestPrice?.volume)}</strong>
            </div>
          </div>

          <div className="stock-controls">
            <div className="segmented" aria-label="타임프레임">
              {TIMEFRAME_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={timeframe === option.value ? "is-active" : ""}
                  onClick={() => handleTimeframeChange(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <div className="segmented" aria-label="차트 타입">
              <button type="button" className={chartType === "candles" ? "is-active" : ""} onClick={() => setChartType("candles")}>
                캔들
              </button>
              <button type="button" className={chartType === "line" ? "is-active" : ""} onClick={() => setChartType("line")}>
                라인
              </button>
            </div>
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

      {errorMessage && priceData.length === 0 ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="market-layout">
        <div className="chart-panel">
          {priceData.length > 0 ? (
            <Chart
              data={priceData}
              ticker={selectedTicker}
              timeframe={timeframe}
              chartType={chartType}
              prediction={prediction}
              layers={layers}
            />
          ) : (
            <div className="empty-state">가격 데이터가 없습니다.</div>
          )}
        </div>

        <aside className="panel side-panel forecast-panel">
          <div className="panel-heading">
            <div className="eyebrow">예측 범위</div>
            <h2>{predictionStatus}</h2>
          </div>
          <div className="forecast-summary">
            <div>
              <span>커버리지</span>
              <strong>{formatRatio(evaluation?.coverage)}</strong>
            </div>
            <div>
              <span>평균 밴드 폭</span>
              <strong>{formatNumber(evaluation?.avg_band_width)}</strong>
            </div>
            <div>
              <span>보수적 예측선</span>
              <strong>{formatNumber(conservativeValue)}</strong>
            </div>
            <div>
              <span>기준일</span>
              <strong>{prediction?.asof_date ?? "-"}</strong>
            </div>
          </div>
          <div className="toggle-list">
            <LayerToggle
              label="보조지표"
              checked={layers.indicators}
              onChange={(checked) => setLayers((previous) => ({ ...previous, indicators: checked }))}
            />
            <LayerToggle
              label="예측 밴드"
              checked={!aiDisabled && layers.aiBand}
              disabled={aiDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, aiBand: checked }))}
            />
            <LayerToggle
              label="보수적 예측선"
              checked={!aiDisabled && layers.conservativeLine}
              disabled={aiDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, conservativeLine: checked }))}
            />
          </div>
          <div className="notice">{aiState.message}</div>
        </aside>
      </section>
    </div>
  );
}
