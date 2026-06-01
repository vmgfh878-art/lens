"use client";

import { FormEvent, startTransition, useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";

import {
  DisplayTimeframe,
  fetchIndicators,
  fetchPrices,
  fetchTickers,
  fetchV1Band1dPrediction,
  fetchV1Band1wPrediction,
  fetchV1LinePrediction,
  getBackendConfigWarning,
  IndicatorPoint,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
  StockSummary,
  V1BandPredictionResult,
  V1LinePredictionResult,
} from "@/api/client";
import Chart from "@/components/Chart";
import IndicatorPanel, { IndicatorChartPoint, IndicatorChartSeries } from "@/components/IndicatorPanel";
import LayerToggle from "@/components/LayerToggle";
import StatusInline from "@/components/StatusInline";
import {
  ApiError,
  buildEmptyPayloadError,
  buildPriceMissingMessage,
  buildShapeMismatchError,
  classifyApiError,
  describeApiError,
  extractApiError,
  extractErrorMessage,
  isBackendConnectionError,
  isResourceNotFound,
} from "@/lib/apiErrors";
import {
  BAND_STALE_THRESHOLD_BUSINESS_DAYS,
  PRICE_LOOKBACK_LIMIT_1D,
  PRICE_LOOKBACK_LIMIT_1W,
} from "@/lib/constants";
import {
  buildDefaultPriceWindow,
  buildFullPriceWindows,
  isValidDate,
  sortPriceRows,
  sortUniqueByDate,
  uniqueDates,
} from "@/lib/dateUtils";
import {
  formatNumber,
  formatPercent,
  formatRatio,
  formatSignedPercentPoint,
  formatVolume,
} from "@/lib/formatters";
import {
  DEFAULT_INDICATORS,
  hasIndicatorValues,
  IndicatorDefinition,
  IndicatorId,
  INDICATOR_DEFINITIONS,
} from "@/lib/indicators";
import {
  buildAiBandWidthPoints,
  buildVisibleIndicatorSeries,
  checkBandOverlay,
  checkLineOverlay,
  getPredictionLineValues,
  isActualBandPrediction,
  isLegacyPrediction,
  isPredictionLineWithinPriceRange,
  PredictionOverlayCheck,
} from "@/lib/predictionOverlay";
import { getProductBandSlot, getProductLineSlot, PRODUCT_SLOT_BY_ID, ProductSlot } from "@/lib/productSlots";
import {
  evaluateBandStaleness,
  formatFreshnessStatus,
  getFreshnessClass,
  getFutureForecastDates,
  getSlotFreshness,
  SlotFreshness,
} from "@/lib/staleness";
import {
  buildBandHistoryFromV1,
  buildBandPredictionFromV1,
  buildLineHistoryFromV1,
  buildLinePredictionFromV1,
  diagnoseBandShape,
} from "@/lib/v1Adapter";

type ChartType = "candles" | "line";

interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

interface PriceState {
  kind: "loading" | "ready" | "empty" | "error";
  message: string;
}

interface PredictionProvenance {
  latestRunId: string | null;
  selectedRunId: string | null;
  isFallback: boolean;
}

const TIMEFRAME_OPTIONS: Array<{ value: DisplayTimeframe; label: string }> = [
  { value: "1D", label: "1D" },
  { value: "1W", label: "1W" },
  { value: "1M", label: "1M" },
];

const PRODUCT_HISTORY_LOOKBACK_DAYS = 370;
const FULL_PRICE_HISTORY_START_YEAR = 2015;
const LINE_OVERLAY_HELD = false;

// CP215 — 1D/1W timeframe 별 가격 lookback (calendar days).
function getPriceLookbackDays(timeframe: DisplayTimeframe) {
  if (timeframe === "1W") {
    return PRICE_LOOKBACK_LIMIT_1W;
  }
  // 1D + 1M 모두 일봉 단위 fetch. 1M 은 별도 표시 모드이지만 lookback 은 1D 와 동일.
  return PRICE_LOOKBACK_LIMIT_1D;
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

function getLastPoint(points: IndicatorChartPoint[]) {
  return points.length > 0 ? points[points.length - 1] : null;
}

function buildAiState(timeframe: DisplayTimeframe): AiState {
  if (timeframe === "1M") {
    return {
      kind: "disabled",
      message: "월간 화면은 현재 가격 전용입니다.",
    };
  }
  return {
    kind: "empty",
    message: "저장된 최신 예측 결과가 아직 없습니다.",
  };
}

// v1 adapter / price helpers 는 @/lib/v1Adapter 로 이동했다.

async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe, fullHistory = false) {
  if (!fullHistory) {
    const window = buildDefaultPriceWindow(getPriceLookbackDays(timeframe));
    const response = await fetchPrices(ticker, {
      timeframe,
      start: window.start,
      end: window.end,
    });
    return sortPriceRows(response.data.data);
  }

  const responses = await Promise.all(
    buildFullPriceWindows(FULL_PRICE_HISTORY_START_YEAR).map((window) =>
      fetchPrices(ticker, {
        timeframe,
        start: window.start,
        end: window.end,
      })
    )
  );
  return sortPriceRows(responses.flatMap((response) => response.data.data));
}

// prediction overlay helpers 는 @/lib/predictionOverlay 로 이동했다.

export default function StockView() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [timeframe, setTimeframe] = useState<DisplayTimeframe>("1D");
  const [chartType, setChartType] = useState<ChartType>("candles");
  const [layers, setLayers] = useState({
    aiBand: true,
    conservativeLine: true,
    volumeBar: true,
  });
  const [selectedIndicators, setSelectedIndicators] = useState<IndicatorId[]>(DEFAULT_INDICATORS);
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorPoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [bandPrediction, setBandPrediction] = useState<PredictionResult | null>(null);
  const [linePredictionHistory, setLinePredictionHistory] = useState<ProductLineHistoryPoint[]>([]);
  const [bandPredictionHistory, setBandPredictionHistory] = useState<ProductBandHistoryPoint[]>([]);
  const [predictionProvenance, setPredictionProvenance] = useState<PredictionProvenance>({
    latestRunId: null,
    selectedRunId: null,
    isFallback: false,
  });
  const [aiState, setAiState] = useState<AiState>({ kind: "loading", message: "예측 확인 중" });
  const [priceState, setPriceState] = useState<PriceState>({ kind: "loading", message: "가격 확인 중" });
  // CP213 — fetch/build 단계별 ApiError 가시화.
  const [linePredictionError, setLinePredictionError] = useState<ApiError | null>(null);
  const [bandPredictionError, setBandPredictionError] = useState<ApiError | null>(null);
  const [bandShapeReason, setBandShapeReason] = useState<string | null>(null);
  const [priceFetchError, setPriceFetchError] = useState<ApiError | null>(null);
  const [indicatorFetchError, setIndicatorFetchError] = useState<ApiError | null>(null);
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchErrorMessage, setSearchErrorMessage] = useState<string | null>(null);
  const [indicatorErrorMessage, setIndicatorErrorMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [visibleTimelineDates, setVisibleTimelineDates] = useState<string[]>([]);
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
    setIndicatorErrorMessage(null);
    setAiState({ kind: "loading", message: "예측 확인 중" });
    setPriceState({ kind: "loading", message: "가격 확인 중" });
    setPrediction(null);
    setBandPrediction(null);
    setLinePredictionHistory([]);
    setBandPredictionHistory([]);
    setLinePredictionError(null);
    setBandPredictionError(null);
    setBandShapeReason(null);
    setPriceFetchError(null);
    setIndicatorFetchError(null);
    setPredictionProvenance({
      latestRunId: null,
      selectedRunId: null,
      isFallback: false,
    });

    try {
      let priceRows: PriceBar[] = [];
      try {
        priceRows = await fetchPriceHistory(nextTicker, nextTimeframe);
        setPriceData(priceRows);
        setPriceState(
          priceRows.length > 0
            ? { kind: "ready", message: "가격 데이터가 연결되었습니다." }
            : { kind: "empty", message: buildPriceMissingMessage(nextTicker, nextTimeframe) }
        );
      } catch (priceError) {
        priceRows = [];
        setPriceData([]);
        if (isResourceNotFound(priceError)) {
          setPriceState({ kind: "empty", message: buildPriceMissingMessage(nextTicker, nextTimeframe) });
        } else {
          const safeMessage = extractErrorMessage(priceError);
          setPriceState({ kind: "error", message: safeMessage });
          setPriceFetchError(classifyApiError(priceError, `/api/v1/stocks/${nextTicker}/prices`));
          if (isBackendConnectionError(priceError)) {
            setErrorMessage(safeMessage);
          }
        }
      }

      try {
        const indicatorsResponse = await fetchIndicators(nextTicker, { timeframe: nextTimeframe, limit: 300 });
        setIndicatorData(sortUniqueByDate(indicatorsResponse.data.data));
        setIndicatorErrorMessage(null);
        setIndicatorFetchError(null);
      } catch (indicatorError) {
        setIndicatorData([]);
        const classified = classifyApiError(indicatorError, `/api/v1/stocks/${nextTicker}/indicators`);
        setIndicatorErrorMessage(describeApiError(classified));
        setIndicatorFetchError(classified);
      }

      if (!isPredictionTimeframeEnabled(nextTimeframe)) {
        setPrediction(null);
        setLinePredictionHistory([]);
        setBandPredictionHistory([]);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          isFallback: false,
        });
        setAiState(
          priceRows.length === 0
            ? { kind: "disabled", message: "월간 화면은 가격 데이터가 연결되면 가격 전용 차트로 표시됩니다." }
            : buildAiState(nextTimeframe)
        );
        return;
      }

      const lineSlot = getProductLineSlot(nextTimeframe);
      const bandSlot = getProductBandSlot(nextTimeframe);
      const lineEndpoint = `/api/v1/predictions/line/${nextTicker}`;
      const bandEndpoint =
        nextTimeframe === "1W"
          ? `/api/v1/predictions/band/1w/${nextTicker}`
          : `/api/v1/predictions/band/1d/${nextTicker}`;
      const linePromise =
        !LINE_OVERLAY_HELD && nextTimeframe === "1D" && lineSlot?.status === "active"
          ? fetchV1LinePrediction(nextTicker, { days: PRODUCT_HISTORY_LOOKBACK_DAYS }).catch((err) => {
              setLinePredictionError(classifyApiError(err, lineEndpoint));
              return null;
            })
          : Promise.resolve(null);
      const bandPromise =
        bandSlot?.status === "active"
          ? nextTimeframe === "1W"
            ? fetchV1Band1wPrediction(nextTicker, { days: 730 }).catch((err) => {
                setBandPredictionError(classifyApiError(err, bandEndpoint));
                return null;
              })
            : fetchV1Band1dPrediction(nextTicker, { days: PRODUCT_HISTORY_LOOKBACK_DAYS }).catch((err) => {
                setBandPredictionError(classifyApiError(err, bandEndpoint));
                return null;
              })
          : Promise.resolve(null);

      const [lineResponse, bandResponse] = await Promise.all([linePromise, bandPromise]);
      const lineHistory = lineResponse ? buildLineHistoryFromV1(lineResponse.data, priceRows) : [];
      const bandHistory = bandResponse ? buildBandHistoryFromV1(bandResponse.data, bandSlot?.horizon ?? 5, priceRows) : [];
      const bandLatest = bandResponse ? buildBandPredictionFromV1(bandResponse.data, nextTimeframe, bandSlot?.horizon ?? 5, priceRows) : null;
      const lineLatest = lineResponse
        ? buildLinePredictionFromV1(lineResponse.data, priceRows, bandLatest?.forecast_dates ?? [])
        : null;
      const bandReady = Boolean(bandLatest && isActualBandPrediction(bandLatest) && checkBandOverlay(bandLatest).ok);
      const lineReady = lineSlot?.status === "active" && (lineHistory.length > 0 || Boolean(lineLatest));

      // CP213 — 응답은 정상이지만 build 단계에서 떨어진 경우 shape_mismatch 로 분류해서 사유 노출.
      if (bandResponse && !bandReady) {
        const shapeReason = diagnoseBandShape(bandResponse.data, bandSlot?.horizon ?? 5, priceRows);
        if (shapeReason) {
          setBandShapeReason(shapeReason);
          setBandPredictionError(buildShapeMismatchError(bandEndpoint, shapeReason));
        } else if (bandLatest && !isActualBandPrediction(bandLatest)) {
          setBandShapeReason("meta.role 이 band_model 이 아님 (composite 또는 legacy)");
          setBandPredictionError(buildShapeMismatchError(bandEndpoint, "meta.role 이 band_model 이 아님"));
        } else if (bandLatest && !checkBandOverlay(bandLatest).ok) {
          const reason = checkBandOverlay(bandLatest).message ?? "overlay 검사 실패";
          setBandShapeReason(reason);
          setBandPredictionError(buildShapeMismatchError(bandEndpoint, reason));
        }
      } else if (bandResponse && bandResponse.data?.data && bandResponse.data.data.length === 0) {
        setBandPredictionError(buildEmptyPayloadError(bandEndpoint, "응답 행 0건"));
      }
      if (lineResponse && !lineReady && lineSlot?.status === "active") {
        if (!lineLatest && lineHistory.length === 0) {
          setLinePredictionError(buildEmptyPayloadError(lineEndpoint, "응답 매핑 결과 0건"));
        }
      }

      setLinePredictionHistory(lineHistory);
      setBandPredictionHistory(bandHistory);
      setPrediction(lineLatest);
      setBandPrediction(bandReady ? bandLatest : null);
      setPredictionProvenance({
        latestRunId: lineSlot?.runId ?? null,
        selectedRunId: lineLatest?.run_id ?? lineHistory.at(-1)?.run_id ?? bandLatest?.run_id ?? null,
        isFallback: false,
      });

      if (priceRows.length === 0 && (lineReady || bandReady || bandHistory.length > 0)) {
        setAiState({
          kind: "empty",
          message: "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다.",
        });
      } else if (nextTimeframe === "1W" && bandReady) {
        setAiState({ kind: "ready", message: "자동 갱신된 1W AI 밴드를 표시합니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다." });
      } else if (lineReady && (bandReady || bandHistory.length > 0)) {
        setAiState({ kind: "ready", message: "1D 보수적 기준선과 1D AI 밴드를 저장된 v1 serving 기준으로 표시합니다." });
      } else if (!lineReady && (bandReady || bandHistory.length > 0)) {
        setAiState({ kind: "ready", message: nextTimeframe === "1W" ? "자동 갱신된 1W AI 밴드를 표시합니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다." : "자동 갱신된 AI 밴드를 표시합니다. 보수적 기준선 데이터는 아직 없습니다." });
      } else if (lineReady && !bandReady) {
        setAiState({ kind: "ready", message: "1D 보수적 기준선을 저장된 v1 serving 기준으로 표시합니다. AI 밴드 데이터는 아직 없습니다." });
      } else {
        setAiState({
          kind: "empty",
          message:
            nextTimeframe === "1W"
              ? "1W AI 밴드 데이터가 아직 없습니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다."
              : "v1 제품 슬롯은 준비됐지만 표시 가능한 예측 데이터가 없습니다.",
        });
      }
    } catch (error) {
      setPriceData([]);
      setPriceState({ kind: "error", message: extractErrorMessage(error) });
      setIndicatorData([]);
        setPrediction(null);
        setLinePredictionHistory([]);
        setBandPredictionHistory([]);
        setBandPrediction(null);
      setPredictionProvenance({
        latestRunId: null,
        selectedRunId: null,
        isFallback: false,
      });
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
    setSuggestions([]);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTickerSelect(nextTicker: string) {
    setTickerInput(nextTicker);
    setSelectedTicker(nextTicker);
    setSuggestions([]);
    startTransition(() => {
      void loadStock(nextTicker, timeframe);
    });
  }

  function handleTimeframeChange(nextTimeframe: DisplayTimeframe) {
    setTimeframe(nextTimeframe);
    startTransition(() => {
      void loadStock(selectedTicker, nextTimeframe);
    });
  }

  function handleIndicatorChange(id: IndicatorId, checked: boolean) {
    setSelectedIndicators((previous) => {
      if (checked) {
        return previous.includes(id) ? previous : [...previous, id];
      }
      return previous.filter((item) => item !== id);
    });
  }

  const latestPrice = getLastPrice(priceData);
  const latestPriceDate = latestPrice?.date ?? null;
  const changePercent = getChangePercent(priceData);
  const backendConfigWarning = getBackendConfigWarning();
  const hasPriceData = priceData.length > 0;
  const aiDisabled = timeframe === "1M";
  const currentLineSlot = getProductLineSlot(timeframe);
  const currentBandSlot = getProductBandSlot(timeframe);
  const activeLineModelDisplayName =
    LINE_OVERLAY_HELD && timeframe === "1D"
      ? "보수적 기준선 보류"
      : currentLineSlot?.status === "deferred"
      ? "1W 보수적 기준선 미제공"
      : currentLineSlot?.displayName ?? "-";
  const activeBandModelDisplayName = currentBandSlot?.displayName ?? "-";
  const forecastUnitLabel = timeframe === "1W" ? "주" : "거래일";
  const rawActivePrediction = prediction && timeframe !== "1M" && checkLineOverlay(prediction).ok ? prediction : null;
  const lineOutOfPriceRange = Boolean(
    hasPriceData &&
      rawActivePrediction &&
      rawActivePrediction.meta?.value_unit !== "return" &&
      !isPredictionLineWithinPriceRange(rawActivePrediction, priceData)
  );
  const activePrediction = rawActivePrediction && !lineOutOfPriceRange ? rawActivePrediction : null;
  const activeBandPrediction =
    bandPrediction && timeframe !== "1M" && isActualBandPrediction(bandPrediction) && checkBandOverlay(bandPrediction).ok
      ? bandPrediction
      : null;
  const activeLineHistory = useMemo(
    () =>
      timeframe === "1M"
        ? []
        : linePredictionHistory
            .filter((item) => isValidDate(item.asof_date) && Number.isFinite(item.value)),
    [linePredictionHistory, timeframe]
  );
  const activeBandHistory = useMemo(
    () =>
      timeframe === "1M"
        ? []
        : bandPredictionHistory.filter(
            (item) => isValidDate(item.asof_date) && Number.isFinite(item.lower) && Number.isFinite(item.upper)
          ),
    [bandPredictionHistory, timeframe]
  );
  const chartPriceData = priceData;
  const lineFutureForecastDates = useMemo(
    () => getFutureForecastDates(activePrediction, latestPriceDate),
    [activePrediction, latestPriceDate]
  );
  const bandFutureForecastDates = useMemo(
    () => getFutureForecastDates(activeBandPrediction, latestPriceDate),
    [activeBandPrediction, latestPriceDate]
  );
  const chartTimelineDates = useMemo(
    () =>
        uniqueDates([
            ...chartPriceData.map((row) => row.date),
            ...(lineFutureForecastDates.length >= 2 ? lineFutureForecastDates : []),
            ...(bandFutureForecastDates.length >= 2 ? bandFutureForecastDates : []),
            ...activeLineHistory.map((row) => row.forecast_date ?? row.asof_date),
            ...activeBandHistory.map((row) => row.forecast_date ?? row.asof_date),
          ]),
      [activeBandHistory, activeLineHistory, bandFutureForecastDates, chartPriceData, lineFutureForecastDates]
    );
  const chartActualDateSet = useMemo(() => new Set(chartPriceData.map((row) => row.date)), [chartPriceData]);
  const indicatorRowsInChartRange = useMemo(
    () => indicatorData.filter((row) => chartActualDateSet.has(row.date)),
    [chartActualDateSet, indicatorData]
  );
  const conservativeValue = getLastFinite(getPredictionLineValues(activePrediction));
  // CP220-디버그 — 백엔드 evaluation mock 에 운영 band run 없음 (cnn_lstm 만 있음).
  // bandPrediction 의 (upper - lower) 평균 / 기준종가 % 로 직관 표시.
  // 임계값: 1D < 3% 좁음 / > 7% 넓음, 1W < 6% 좁음 / > 12% 넓음.
  const bandWidthSummary = useMemo(() => {
    const baseClose = latestPrice?.close;
    if (!bandPrediction || !Number.isFinite(baseClose) || (baseClose as number) <= 0) return null;
    const upper = bandPrediction.upper_band_series;
    const lower = bandPrediction.lower_band_series;
    const widths: number[] = [];
    for (let i = 0; i < upper.length; i++) {
      const u = upper[i];
      const l = lower[i];
      if (Number.isFinite(u) && Number.isFinite(l)) widths.push(u - l);
    }
    if (widths.length === 0) return null;
    const avgWidth = widths.reduce((s, w) => s + w, 0) / widths.length;
    const pct = (avgWidth / (baseClose as number)) * 100;
    const isWeekly = bandPrediction.timeframe === "1W";
    const [narrow, wide] = isWeekly ? [6, 12] : [3, 7];
    const label: "좁음" | "보통" | "넓음" = pct < narrow ? "좁음" : pct > wide ? "넓음" : "보통";
    return { pct, label };
  }, [bandPrediction, latestPrice]);
  const lineAsofDate = activePrediction?.asof_date ?? activeLineHistory.at(-1)?.asof_date ?? null;
  const bandAsofDate = activeBandPrediction?.asof_date ?? activeBandHistory.at(-1)?.asof_date ?? null;
  // CP215 — 토글 가드용 raw band asof. activeBandPrediction 이 overlay 검사 실패로 null 이어도
  // bandPrediction 자체의 asof 가 있으면 그걸 사용 (stale 판정은 모델 응답 기준).
  const rawBandAsofDate = bandPrediction?.asof_date ?? activeBandHistory.at(-1)?.asof_date ?? null;
  const lineFreshness = getSlotFreshness(currentLineSlot, lineAsofDate, latestPriceDate, priceData);
  const bandFreshness = getSlotFreshness(currentBandSlot, bandAsofDate, latestPriceDate, priceData);
  // CP215 — 가격 latest 와 밴드 latest asof 의 거래일 gap > 5 면 stale.
  const bandStaleness = evaluateBandStaleness(
    latestPriceDate,
    rawBandAsofDate,
    BAND_STALE_THRESHOLD_BUSINESS_DAYS
  );
  const lineFuturePointCount = lineFutureForecastDates.length;
  const bandFuturePointCount = bandFutureForecastDates.length;
  const lineFutureHidden = Boolean(activePrediction && lineFuturePointCount < 2);
  const bandFutureHidden = Boolean(activeBandPrediction && bandFuturePointCount < 2);
  const changeClass = changePercent == null ? "is-flat" : changePercent < 0 ? "is-down" : "is-up";
  // CP215 — base close 조회 실패(=adapter normalize 실패) 케이스 명시 사유.
  const bandBaseCloseMissing = Boolean(
    bandShapeReason &&
      (bandShapeReason.includes("가격 lookback 부족") || bandShapeReason.includes("환산 불가"))
  );
  // CP215 — stale 또는 base close 누락도 토글 disable.
  const bandLayerDisabled =
    aiDisabled ||
    !hasPriceData ||
    (!activeBandPrediction && activeBandHistory.length === 0) ||
    bandStaleness.isStale;
  const lineLayerDisabled = aiDisabled || !hasPriceData || (!activePrediction && activeLineHistory.length === 0);
  const volumeLayerDisabled = priceData.every((row) => !Number.isFinite(row.volume) || row.volume == null || row.volume <= 0);
  // CP213 + CP215 — disable 사유 (LayerToggle.disabledReason 으로 전달).
  // 우선순위: 환경/가격 → stale → base close 누락 → 일반 응답 에러 → shape mismatch → 슬롯 상태.
  const bandDisabledReason = !bandLayerDisabled
    ? undefined
    : aiDisabled
    ? "AI 기능 비활성"
    : !hasPriceData
    ? "가격 데이터 없음"
    : bandStaleness.isStale
    ? bandStaleness.reason
    : bandBaseCloseMissing
    ? "밴드 기준일 가격 조회 실패"
    : bandPredictionError
    ? describeApiError(bandPredictionError)
    : bandShapeReason
    ? `응답 매핑 실패 — ${bandShapeReason}`
    : currentBandSlot?.status !== "active"
    ? "슬롯이 active 가 아님"
    : "응답 없음";
  const lineDisabledReason = !lineLayerDisabled
    ? undefined
    : aiDisabled
    ? "AI 기능 비활성"
    : !hasPriceData
    ? "가격 데이터 없음"
    : linePredictionError
    ? describeApiError(linePredictionError)
    : currentLineSlot?.status !== "active"
    ? "슬롯이 active 가 아님"
    : "응답 없음";
  const volumeDisabledReason = !volumeLayerDisabled ? undefined : "가격 응답에 volume 컬럼이 0이거나 비어 있음";
  const aiBandWidthPoints = useMemo(
    () => buildAiBandWidthPoints(activeBandHistory, activeBandPrediction).filter((point) => chartActualDateSet.has(point.date)),
    [activeBandHistory, activeBandPrediction, chartActualDateSet]
  );
  const availableIndicatorDefinitions = useMemo(() => {
    const dbDefinitions = INDICATOR_DEFINITIONS.filter((definition) => hasIndicatorValues(indicatorRowsInChartRange, definition));
    const aiBandWidthDefinition = INDICATOR_DEFINITIONS.find((definition) => definition.id === "ai_band_width");
    if (timeframe !== "1M" && aiBandWidthDefinition && (aiBandWidthPoints.length > 0 || selectedIndicators.includes("ai_band_width"))) {
      return [...dbDefinitions, aiBandWidthDefinition];
    }
    return dbDefinitions;
  }, [aiBandWidthPoints, indicatorRowsInChartRange, selectedIndicators, timeframe]);
  const visibleIndicatorSeries = useMemo(
    () =>
      buildVisibleIndicatorSeries({
        selectedIndicators,
        indicatorData: indicatorRowsInChartRange,
        indicatorErrorMessage,
        availableDefinitions: availableIndicatorDefinitions,
        allowedDates: chartActualDateSet,
        aiBandWidthPoints,
      }),
    [selectedIndicators, indicatorRowsInChartRange, indicatorErrorMessage, availableIndicatorDefinitions, chartActualDateSet, aiBandWidthPoints]
  );
  const indicatorTimelineDates = visibleTimelineDates.length > 0 ? visibleTimelineDates : chartTimelineDates;
  const isLegacyArtifact = isLegacyPrediction(activePrediction);
  const lineStatusLabel = aiDisabled
    ? "-"
    : LINE_OVERLAY_HELD && timeframe === "1D"
    ? "보류"
    : currentLineSlot?.status === "deferred"
    ? "deferred"
    : !hasPriceData && (rawActivePrediction || activeLineHistory.length > 0)
    ? "가격 데이터 연결 대기"
    : lineOutOfPriceRange
    ? `${formatFreshnessStatus(lineFreshness)} · 차트 숨김`
    : activePrediction || activeLineHistory.length > 0
    ? formatFreshnessStatus(lineFreshness)
    : "데이터 없음";
  const bandStatusLabel = aiDisabled
    ? "-"
    : !hasPriceData && (activeBandPrediction || activeBandHistory.length > 0)
    ? "가격 데이터 연결 대기"
    : activeBandPrediction || activeBandHistory.length > 0
    ? formatFreshnessStatus(bandFreshness)
    : currentBandSlot?.status === "active"
    ? // CP213 — "데이터 없음" 단정 대신 실제 원인 표시.
      bandPredictionError
      ? describeApiError(bandPredictionError)
      : bandShapeReason
      ? `매핑 실패 — ${bandShapeReason}`
      : "응답 없음"
    : "-";
  const provenanceLabel = isLegacyArtifact
    ? "이전 실험 결과"
    : predictionProvenance.isFallback
    ? "저장된 예측 결과"
    : activePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0
    ? "v1 제품 슬롯"
    : "-";
  const provenanceSummary =
    rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0
      ? activeBandPrediction
        ? `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""} / ${activeBandModelDisplayName}`
        : `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""}`
      : "표시 중인 예측 결과가 없습니다.";
  const forecastHorizonCount =
    (rawActivePrediction ? currentLineSlot?.horizon : null) ??
    (activeBandPrediction ? currentBandSlot?.horizon : null) ??
    Math.max(lineFuturePointCount, bandFuturePointCount);
  const priceLatestDateLabel = latestPriceDate ?? "-";
  const bandAsofLabel = bandAsofDate ?? "-";
  const lineAsofLabel = lineAsofDate ?? "-";
  const futureForecastStatus =
    aiDisabled || !hasPriceData
      ? "-"
      : bandFuturePointCount >= 2
      ? `${bandFuturePointCount}${forecastUnitLabel}`
      : lineFuturePointCount >= 2
      ? `${lineFuturePointCount}${forecastUnitLabel}`
      : activeBandPrediction || activePrediction
      ? "미래 구간 부족"
      : "-";
  // CP215 — stale 일 땐 토글이 disable 되어 차트에 안 그려지므로 notice 도 명시적 사유로.
  const staleNotice = bandStaleness.isStale
    ? `${bandStaleness.reason} — AI 밴드 토글이 비활성화됐습니다.`
    : bandFreshness === "stale"
    ? "AI 밴드는 마지막 저장된 기준으로 표시됩니다."
    : bandFreshness === "delayed"
    ? "AI 밴드는 가격 최신일보다 조금 이전 기준으로 표시됩니다."
    : null;
  const futureHiddenNotice =
    bandFutureHidden || lineFutureHidden
      ? "가격 최신일 이후의 forecast가 2개 미만이라 최신 미래 예측선은 숨기고 저장된 이력만 표시합니다."
      : null;
  const layerStatusNotice = staleNotice ?? futureHiddenNotice;
  const displayNotice = lineOutOfPriceRange
    ? "보수적 기준선이 현재 가격 범위를 벗어나 차트에서는 숨겼습니다. AI 밴드는 유지합니다."
    : !hasPriceData && (rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0)
    ? "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다."
    : layerStatusNotice
    ? layerStatusNotice
    : aiState.message;
  const emptyIndicatorMessage =
    availableIndicatorDefinitions.length === 0
      ? "이 타임프레임에 표시할 보조지표 데이터가 없습니다."
      : "오른쪽 메뉴에서 보조지표를 선택하세요.";
  const handleVisibleDatesChange = useCallback((dates: string[]) => {
    setVisibleTimelineDates((previous) => {
      if (previous.length === dates.length && previous.every((date, index) => date === dates[index])) {
        return previous;
      }
      return dates;
    });
  }, []);

  useEffect(() => {
    setVisibleTimelineDates(chartTimelineDates);
  }, [chartTimelineDates]);

  return (
    <div className="view-stack stock-view">
      <section className="stock-topbar">
        <p className="stock-intro">
          Lens는 가격 차트 위에 보수적 기준선과 자동 갱신된 AI 밴드를 겹쳐 보며, 리스크를 먼저 확인하는 투자 보조 도구입니다.
        </p>
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
            <div className="quote-item quote-item--volume">
              <span>거래량</span>
              <strong>{formatVolume(latestPrice?.volume)}</strong>
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

      {backendConfigWarning ? <div className="notice">{backendConfigWarning}</div> : null}
      {errorMessage && priceData.length === 0 ? <div className="notice notice--error">{errorMessage}</div> : null}

      <section className="market-layout">
        <div className="chart-stack">
          <div className="chart-panel">
            {priceData.length > 0 ? (
              <Chart
                data={chartPriceData}
                ticker={selectedTicker}
                timeframe={timeframe}
                chartType={chartType}
                prediction={activePrediction}
                bandPrediction={activeBandPrediction}
                predictionHistory={activeLineHistory}
                bandPredictionHistory={activeBandHistory}
                layers={layers}
                timelineDates={chartTimelineDates}
                onVisibleDatesChange={handleVisibleDatesChange}
              />
            ) : priceFetchError ? (
              <StatusInline
                kind="error"
                label="가격 데이터"
                error={priceFetchError}
                hint="백엔드 로그 또는 네트워크 상태 확인"
                onRetry={() => void loadStock(selectedTicker, timeframe)}
                variant="block"
              />
            ) : (
              <div className="empty-state empty-state--stacked">
                <strong>가격 데이터 없음</strong>
                <span>{priceState.message}</span>
                {rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0 ? (
                  <span>저장된 AI 예측은 확인됐지만, 가격 데이터가 연결되면 차트 위에 표시됩니다.</span>
                ) : null}
              </div>
            )}
          </div>

          <IndicatorPanel
            series={visibleIndicatorSeries}
            note={indicatorErrorMessage}
            emptyMessage={emptyIndicatorMessage}
            timelineDates={indicatorTimelineDates}
          />
        </div>

        <aside className="panel side-panel forecast-panel">
          <div className="panel-heading">
            <h2>모델 정보</h2>
          </div>

          <div className="provenance-card">
            <div className="provenance-grid">
              <span>보수적 기준선</span>
              <strong>{activeLineModelDisplayName}</strong>
              <span>AI 밴드</span>
              <strong>{activeBandModelDisplayName}</strong>
              <span>가격 최신일</span>
              <strong>{priceLatestDateLabel}</strong>
              <span>AI 밴드 상태</span>
              <strong>
                <span className={`freshness-badge freshness-badge--${getFreshnessClass(bandFreshness)}`}>
                  {bandStaleness.isStale
                    ? bandStaleness.reason
                    : bandBaseCloseMissing
                    ? "기준일 가격 조회 실패"
                    : formatFreshnessStatus(bandFreshness)}
                </span>
              </strong>
              <span>보수적 기준선 상태</span>
              <strong>
                <span className={`freshness-badge freshness-badge--${getFreshnessClass(lineFreshness)}`}>
                  {formatFreshnessStatus(lineFreshness)}
                </span>
              </strong>
              <span>예측 기간</span>
              <strong>{forecastHorizonCount ? `${forecastHorizonCount}${forecastUnitLabel}` : "-"}</strong>
            </div>
            {timeframe === "1D" && (rawActivePrediction || activeLineHistory.length > 0) ? (
              <p>보수적 기준선은 5거래일 후 도착가를 보수적으로 추정합니다. 자세한 계산 방식은 지표 가이드를 참고하세요.</p>
            ) : null}
            {timeframe === "1W" ? <p>1W 보수적 기준선은 준비 중입니다. 1W AI 밴드만 표시됩니다.</p> : null}
            {!aiDisabled && (rawActivePrediction || activeBandPrediction || activeLineHistory.length > 0 || activeBandHistory.length > 0) ? (
              <details className="provenance-details">
                <summary>상세 정보</summary>
                <div className="provenance-grid provenance-grid--composite">
                  <span>표시 기준</span>
                  <strong>{provenanceLabel}</strong>
                  <span>보수적 기준선 ID</span>
                  <strong>{rawActivePrediction?.run_id ?? currentLineSlot?.runId ?? "준비 중"}</strong>
                  <span>밴드 실행 ID</span>
                  <strong>{activeBandPrediction?.run_id ?? activeBandHistory.at(-1)?.run_id ?? currentBandSlot?.runId ?? "-"}</strong>
                </div>
              </details>
            ) : null}
          </div>

          <div className="layer-group">
            <div className="layer-group__title">가격 오버레이</div>
            <LayerToggle
              label="보수적 기준선"
              checked={!lineLayerDisabled && layers.conservativeLine}
              disabled={lineLayerDisabled}
              disabledReason={lineDisabledReason}
              onChange={(checked) => setLayers((previous) => ({ ...previous, conservativeLine: checked }))}
            />
            <p className="layer-description">
              {timeframe === "1W"
                ? "1W 보수적 기준선은 준비 중입니다."
                : "앞으로 5거래일의 보수적 도착가 추정선입니다. 가격이 이 선 위에 있으면 정상 범위로 봅니다."}
            </p>
            <LayerToggle
              label="AI 밴드"
              checked={!bandLayerDisabled && layers.aiBand}
              disabled={bandLayerDisabled}
              disabledReason={bandDisabledReason}
              onChange={(checked) => setLayers((previous) => ({ ...previous, aiBand: checked }))}
            />
            <p className="layer-description">
              {timeframe === "1W"
                ? "주간 예상 변동 범위입니다. q10~q90 분위수로 실제 종가의 약 80% 가 밴드 안에 들어오도록 설계됐습니다. 넓을수록 불확실성이 큽니다."
                : "5거래일 후 예상 변동 범위입니다. q15~q85 분위수로 실제 종가의 약 70% 가 밴드 안에 들어오도록 설계됐습니다. 넓을수록 불확실성이 큽니다."}
            </p>
            {!bandLayerDisabled && layers.aiBand && bandWidthSummary ? (
              <div className="layer-metrics">
                <div>
                  <span>평균 밴드 폭</span>
                  <strong>
                    {bandWidthSummary.pct.toFixed(1)}% (<span className={`band-width-label band-width-label--${bandWidthSummary.label === "좁음" ? "narrow" : bandWidthSummary.label === "넓음" ? "wide" : "normal"}`}>{bandWidthSummary.label}</span>)
                  </strong>
                </div>
              </div>
            ) : null}
            {!lineLayerDisabled && layers.conservativeLine ? (
              <div className="layer-metrics">
                <div>
                  <span>5거래일 후 도착가</span>
                  <strong>{formatNumber(conservativeValue)}</strong>
                </div>
                <div>
                  <span>상태</span>
                  <strong>{formatFreshnessStatus(lineFreshness)}</strong>
                </div>
              </div>
            ) : null}
          </div>

          <div className="layer-group">
            <div className="layer-group__title">거래량</div>
            <LayerToggle
              label="거래량 bar"
              checked={!volumeLayerDisabled && layers.volumeBar}
              disabled={volumeLayerDisabled}
              disabledReason={volumeDisabledReason}
              onChange={(checked) => setLayers((previous) => ({ ...previous, volumeBar: checked }))}
            />
          </div>

          <div className="layer-group">
            <details className="indicator-menu" open>
              <summary>
                <span>
                  <span className="eyebrow">하단 지표</span>
                  <strong>하단 지표 선택</strong>
                </span>
              </summary>
              <div className="indicator-selector">
                {["기본", "추세", "변동성/위치", "AI"].map((category) => {
                  const options = availableIndicatorDefinitions.filter((definition) => definition.category === category);
                  if (options.length === 0) {
                    return null;
                  }
                  return (
                    <IndicatorOptionGroup
                      key={category}
                      title={category}
                      options={options}
                      selectedIndicators={selectedIndicators}
                      onChange={handleIndicatorChange}
                    />
                  );
                })}
                {availableIndicatorDefinitions.length === 0 ? (
                  <div className="compact-note">표시할 수 있는 보조지표 값이 없습니다.</div>
                ) : null}
              </div>
            </details>
          </div>

          {lineOutOfPriceRange || aiState.kind !== "ready" ? <div className="notice">{displayNotice}</div> : null}
        </aside>
      </section>
    </div>
  );
}

interface IndicatorOptionGroupProps {
  title: string;
  options: IndicatorDefinition[];
  selectedIndicators: IndicatorId[];
  onChange: (id: IndicatorId, checked: boolean) => void;
}

function IndicatorOptionGroup({ title, options, selectedIndicators, onChange }: IndicatorOptionGroupProps) {
  return (
    <div className="indicator-selector__group">
      <span>{title}</span>
      {options.map((option) => (
        <IndicatorOption
          key={option.id}
          option={option}
          checked={selectedIndicators.includes(option.id)}
          onChange={onChange}
        />
      ))}
    </div>
  );
}

interface IndicatorOptionProps {
  option: IndicatorDefinition;
  checked: boolean;
  onChange: (id: IndicatorId, checked: boolean) => void;
}

function IndicatorOption({ option, checked, onChange }: IndicatorOptionProps) {
  return (
    <label className={`indicator-option${checked ? " is-on" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(option.id, event.target.checked)}
      />
      <span className="indicator-option__dot" aria-hidden="true" />
      <span className="indicator-option__body">
        <span className="indicator-option__label">{option.label}</span>
        <span className="indicator-option__description">{option.description}</span>
      </span>
    </label>
  );
}
