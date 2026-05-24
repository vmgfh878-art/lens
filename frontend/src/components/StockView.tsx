"use client";

import { FormEvent, startTransition, useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";

import {
  DisplayTimeframe,
  EvaluationSummary,
  fetchIndicators,
  fetchPrediction,
  fetchProductPredictionHistory,
  fetchPrices,
  fetchRunEvaluations,
  fetchTickers,
  IndicatorPoint,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
  ProductBandHistoryPoint,
  ProductLineHistoryPoint,
  StockSummary,
} from "@/api/client";
import Chart from "@/components/Chart";
import IndicatorPanel, { IndicatorChartPoint, IndicatorChartSeries } from "@/components/IndicatorPanel";
import LayerToggle from "@/components/LayerToggle";

type ChartType = "candles" | "line";
type IndicatorId =
  | "rsi"
  | "macd_ratio"
  | "vol_change"
  | "ma_5_ratio"
  | "ma_20_ratio"
  | "ma_60_ratio"
  | "bb_position"
  | "atr_ratio"
  | "ai_band_width";

interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

interface PriceState {
  kind: "loading" | "ready" | "empty" | "error";
  message: string;
}

interface ApiErrorShape {
  code: string;
  message: string;
}

interface IndicatorDefinition {
  id: IndicatorId;
  label: string;
  field?: keyof IndicatorPoint;
  category: "기본" | "추세" | "변동성/위치" | "AI";
  description: string;
  color: string;
  baseline?: number;
  fixedRange?: {
    min: number;
    max: number;
  };
  transform?: (value: number) => number;
  formatLatest: (value: number | null | undefined) => string;
}

interface PredictionOverlayCheck {
  ok: boolean;
  message: string | null;
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

const PRODUCT_LINE_RUN_ID = "patchtst-1D-efad3c29d803";
const PRODUCT_BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8";
const LINE_MODEL_DISPLAY_NAME = "보수적 예측선 모델 v1";
const BAND_MODEL_DISPLAY_NAME = "AI 밴드 모델 v1";
const DEFAULT_INDICATORS: IndicatorId[] = ["rsi", "macd_ratio", "ai_band_width"];
const PRODUCT_HISTORY_LOOKBACK_DAYS = 370;
const DEFAULT_PRICE_LOOKBACK_DAYS = 365;
const FULL_PRICE_HISTORY_START_YEAR = 2015;

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatVolume(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
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

function formatSignedPercentPoint(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}%`;
}

function getMetaString(meta: Record<string, unknown> | undefined, key: string) {
  const value = meta?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function normalizeRsi(value: number) {
  return value >= 0 && value <= 1 ? value * 100 : value;
}

function ratioToPercent(value: number) {
  return value * 100;
}

const INDICATOR_DEFINITIONS: IndicatorDefinition[] = [
  {
    id: "rsi",
    label: "RSI",
    field: "rsi",
    category: "기본",
    description: "최근 상승/하락 강도를 0~100으로 본 과열·침체 참고 지표",
    color: "#0f766e",
    fixedRange: { min: 0, max: 100 },
    transform: normalizeRsi,
    formatLatest: (value) => formatNumber(value, 1),
  },
  {
    id: "macd_ratio",
    label: "MACD",
    field: "macd_ratio",
    category: "기본",
    description: "추세 전환과 모멘텀 변화를 보는 이동평균 기반 지표",
    color: "#2563eb",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "vol_change",
    label: "거래량 변화",
    field: "vol_change",
    category: "기본",
    description: "직전 구간 대비 거래량 증감을 봅니다.",
    color: "#64748b",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_5_ratio",
    label: "MA 5 괴리",
    field: "ma_5_ratio",
    category: "추세",
    description: "5일 이동평균 대비 가격 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_20_ratio",
    label: "MA 20 괴리",
    field: "ma_20_ratio",
    category: "추세",
    description: "20일 이동평균 대비 가격 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ma_60_ratio",
    label: "MA 60 괴리",
    field: "ma_60_ratio",
    category: "추세",
    description: "60일 이동평균 대비 중기 위치를 봅니다.",
    color: "#7c3aed",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "bb_position",
    label: "BB 위치",
    field: "bb_position",
    category: "변동성/위치",
    description: "볼린저 밴드 안에서 현재 위치를 봅니다.",
    color: "#9333ea",
    fixedRange: { min: 0, max: 1 },
    formatLatest: (value) => formatNumber(value, 2),
  },
  {
    id: "atr_ratio",
    label: "ATR",
    field: "atr_ratio",
    category: "변동성/위치",
    description: "최근 가격 변동 폭을 보는 전통 변동성 지표",
    color: "#d97706",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
  },
  {
    id: "ai_band_width",
    label: "AI 밴드 폭",
    category: "AI",
    description: "AI가 보는 예상 변동 범위의 넓이",
    color: "#4b5563",
    baseline: 0,
    formatLatest: (value) => formatNumber(value, 4),
  },
];

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
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

function extractHttpStatus(error: unknown) {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const status = (error as { response?: { status?: unknown } }).response?.status;
    return typeof status === "number" ? status : null;
  }
  return null;
}

function isResourceNotFound(error: unknown) {
  const apiError = extractApiError(error);
  return apiError?.code === "RESOURCE_NOT_FOUND" || extractHttpStatus(error) === 404;
}

function isBackendConnectionError(error: unknown) {
  const status = extractHttpStatus(error);
  if (status != null) {
    return status >= 500;
  }
  if (error instanceof Error) {
    return (
      error.message === "Network Error" ||
      error.message.includes("ECONNREFUSED") ||
      error.message.includes("Failed to construct 'URL'") ||
      error.message.includes("Invalid URL")
    );
  }
  return false;
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
    if (error.message.includes("Failed to construct 'URL'") || error.message.includes("Invalid URL")) {
      return "백엔드 주소 설정을 확인할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 값을 확인해주세요.";
    }
    return error.message;
  }
  return "데이터를 불러오지 못했습니다.";
}

function buildPriceMissingMessage(ticker: string, timeframe: DisplayTimeframe) {
  return `${ticker.toUpperCase()} ${timeframe} 가격 데이터가 아직 연결되지 않았습니다. 티커 또는 데이터 소스를 확인해주세요.`;
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

function selectEvaluation(rows: EvaluationSummary[], prediction: PredictionResult) {
  return rows.find((row) => row.asof_date === prediction.asof_date) ?? rows[0] ?? null;
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

function getIndicatorValue(row: IndicatorPoint, definition: IndicatorDefinition) {
  if (!definition.field) {
    return null;
  }
  const rawValue = row[definition.field];
  if (!isFiniteNumber(rawValue)) {
    return null;
  }
  return definition.transform ? definition.transform(rawValue) : rawValue;
}

function hasIndicatorValues(rows: IndicatorPoint[], definition: IndicatorDefinition) {
  if (!definition.field) {
    return false;
  }
  return rows.some((row) => isFiniteNumber(getIndicatorValue(row, definition)));
}

function buildFieldPoints(rows: IndicatorPoint[], definition: IndicatorDefinition, allowedDates?: Set<string>): IndicatorChartPoint[] {
  return sortUniqueByDate(
    rows
    .map((row) => {
      if (allowedDates && !allowedDates.has(row.date)) {
        return null;
      }
      const value = getIndicatorValue(row, definition);
      if (!isFiniteNumber(value)) {
        return null;
      }
      return {
        date: row.date,
        value,
      };
    })
    .filter((point): point is IndicatorChartPoint => point !== null)
  );
}

function isValidDate(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

function uniqueDates(dates: string[]) {
  return Array.from(new Set(dates.filter(isValidDate))).sort((left, right) => left.localeCompare(right));
}

function sortUniqueByDate<T extends { date: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidDate(row.date)) {
      deduped.set(row.date, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.date.localeCompare(right.date));
}

function sortPriceRows(rows: PriceBar[]) {
  return sortUniqueByDate(rows.filter((row) => Number.isFinite(row.close)));
}

function formatDate(value: Date) {
  return value.toISOString().slice(0, 10);
}

function buildDefaultPriceWindow() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - DEFAULT_PRICE_LOOKBACK_DAYS);
  return {
    start: formatDate(start),
    end: formatDate(end),
  };
}

function buildFullPriceWindows() {
  const currentYear = new Date().getFullYear();
  return Array.from({ length: currentYear - FULL_PRICE_HISTORY_START_YEAR + 1 }, (_, index) => {
    const year = FULL_PRICE_HISTORY_START_YEAR + index;
    return {
      start: `${year}-01-01`,
      end: `${year}-12-31`,
    };
  });
}

async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe, fullHistory = false) {
  if (!fullHistory) {
    const window = buildDefaultPriceWindow();
    const response = await fetchPrices(ticker, {
      timeframe,
      start: window.start,
      end: window.end,
    });
    return sortPriceRows(response.data.data);
  }

  const responses = await Promise.all(
    buildFullPriceWindows().map((window) =>
      fetchPrices(ticker, {
        timeframe,
        start: window.start,
        end: window.end,
      })
    )
  );
  return sortPriceRows(responses.flatMap((response) => response.data.data));
}

function isLegacyPrediction(prediction: PredictionResult | null) {
  if (!prediction) {
    return false;
  }
  const role = getMetaString(prediction.meta, "role");
  const deprecated = prediction.meta?.deprecated_for_phase1_product_contract;
  return (
    prediction.model_name.includes("composite") ||
    role === "composite_model" ||
    deprecated === true ||
    deprecated === "true"
  );
}

function checkLineOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];
  const lineValues = prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 보수적 예측선을 숨겼습니다." };
  }
  if (
    lineValues.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 시리즈 길이가 맞지 않아 보수적 예측선을 숨겼습니다." };
  }

  const allValues = [...lineValues];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "예측 데이터에 표시할 수 없는 값이 있어 보수적 예측선을 숨겼습니다." };
  }

  return { ok: true, message: null };
}

function checkBandOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 AI 밴드를 숨겼습니다." };
  }
  if (
    prediction.upper_band_series.length !== forecastDates.length ||
    prediction.lower_band_series.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 밴드 길이가 맞지 않아 AI 밴드를 숨겼습니다." };
  }

  const allValues = [...prediction.upper_band_series, ...prediction.lower_band_series];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "AI 밴드 데이터에 표시할 수 없는 값이 있어 숨겼습니다." };
  }

  return { ok: true, message: null };
}

function isActualBandPrediction(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return false;
  }

  const role = getMetaString(prediction.meta, "role");
  const bandFieldsPolicy = getMetaString(prediction.meta, "band_fields_policy");
  const bandSavedInCp140 = prediction.meta?.band_saved_in_cp140;

  if (role !== "band_model") {
    return false;
  }

  if (bandSavedInCp140 === false || bandSavedInCp140 === "false") {
    return false;
  }

  return bandFieldsPolicy !== "schema_required_degenerate_equal_to_line";
}

function getPredictionLineValues(prediction: PredictionResult | null | undefined) {
  if (!prediction) {
    return [];
  }
  return prediction.conservative_series?.length ? prediction.conservative_series : prediction.line_series ?? [];
}

function isPredictionLineWithinPriceRange(prediction: PredictionResult, rows: PriceBar[]) {
  const values = getPredictionLineValues(prediction).filter(Number.isFinite);
  const priceValues = rows
    .flatMap((row) => [row.low ?? row.close, row.high ?? row.close, row.close])
    .filter(Number.isFinite);
  if (values.length === 0 || priceValues.length < 2) {
    return false;
  }

  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const latestPrice = priceValues[priceValues.length - 1] ?? maxPrice;
  const span = Math.max(maxPrice - minPrice, Math.abs(latestPrice) * 0.04, 1);
  const lowerBound = minPrice - span * 0.35;
  const upperBound = maxPrice + span * 0.35;
  return values.every((value) => value >= lowerBound && value <= upperBound);
}

function buildAiBandWidthPoints(history: ProductBandHistoryPoint[], latestPrediction: PredictionResult | null): IndicatorChartPoint[] {
  const historyPoints = history
    .map((row) => {
      if (!row.asof_date || !Number.isFinite(row.upper) || !Number.isFinite(row.lower)) {
        return null;
      }
      return {
        date: row.asof_date,
        value: Math.max(0, row.upper - row.lower),
      };
    })
    .filter((point): point is IndicatorChartPoint => point !== null);

  const latestUpper = latestPrediction?.upper_band_series?.[0];
  const latestLower = latestPrediction?.lower_band_series?.[0];
  if (latestPrediction?.asof_date && isFiniteNumber(latestUpper) && isFiniteNumber(latestLower)) {
    historyPoints.push({
      date: latestPrediction.asof_date,
      value: Math.max(0, latestUpper - latestLower),
    });
  }

  return sortUniqueByDate(historyPoints);
}

function buildVisibleIndicatorSeries(params: {
  selectedIndicators: IndicatorId[];
  indicatorData: IndicatorPoint[];
  indicatorErrorMessage: string | null;
  availableDefinitions: IndicatorDefinition[];
  allowedDates?: Set<string>;
  aiBandWidthPoints: IndicatorChartPoint[];
}): IndicatorChartSeries[] {
  const { selectedIndicators, indicatorData, indicatorErrorMessage, availableDefinitions, allowedDates, aiBandWidthPoints } = params;
  const indicatorEmptyMessage = indicatorErrorMessage ?? "저장된 보조지표 데이터가 없습니다.";

  return availableDefinitions
    .filter((definition) => selectedIndicators.includes(definition.id))
    .map((definition) => {
      if (definition.id === "ai_band_width") {
        const latest = getLastPoint(aiBandWidthPoints)?.value;
        return {
          id: definition.id,
          label: definition.label,
          groupLabel: definition.category,
          points: aiBandWidthPoints,
          color: definition.color,
          baseline: definition.baseline,
          fixedRange: definition.fixedRange,
          latestLabel: definition.formatLatest(latest),
          emptyMessage: "저장된 AI 밴드 폭 이력이 없습니다.",
        };
      }

      const points = buildFieldPoints(indicatorData, definition, allowedDates);
      const latest = getLastPoint(points)?.value;

      return {
        id: definition.id,
        label: definition.label,
        groupLabel: definition.category,
        points,
        color: definition.color,
        baseline: definition.baseline,
        fixedRange: definition.fixedRange,
        latestLabel: definition.formatLatest(latest),
        emptyMessage: indicatorEmptyMessage,
      };
    });
}

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
  const [bandEvaluation, setBandEvaluation] = useState<EvaluationSummary | null>(null);
  const [aiState, setAiState] = useState<AiState>({ kind: "loading", message: "예측 확인 중" });
  const [priceState, setPriceState] = useState<PriceState>({ kind: "loading", message: "가격 확인 중" });
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
    setBandEvaluation(null);
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
          if (isBackendConnectionError(priceError)) {
            setErrorMessage(safeMessage);
          }
        }
      }

      try {
        const indicatorsResponse = await fetchIndicators(nextTicker, { timeframe: nextTimeframe, limit: 300 });
        setIndicatorData(sortUniqueByDate(indicatorsResponse.data.data));
        setIndicatorErrorMessage(null);
      } catch {
        setIndicatorData([]);
        setIndicatorErrorMessage("보조지표를 불러오지 못했습니다.");
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

      if (nextTimeframe === "1W") {
        setLinePredictionHistory([]);
        setBandPredictionHistory([]);
        setPrediction(null);
        setBandPrediction(null);
        setBandEvaluation(null);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          isFallback: false,
        });
        setAiState({
          kind: "empty",
          message:
            priceRows.length === 0
              ? "1W 화면은 가격 데이터가 연결되면 차트가 표시됩니다."
              : "1W 보수적 예측선은 준비 중이며, 1W AI 밴드는 AI 모델 페이지에서 평가 결과를 확인할 수 있습니다. 차트 overlay 는 다음 버전에서 추가 예정입니다.",
        });
        return;
      }

      if (nextTimeframe !== "1D") {
        setAiState(
          priceRows.length === 0
            ? { kind: "empty", message: "주간 AI 차트 overlay 는 다음 버전 예정. 가격 데이터 연결 후 표시됩니다." }
            : { kind: "empty", message: "1W AI 밴드 모델은 활성화됨 (AI 모델 페이지에서 확인). 차트 overlay 는 다음 버전 예정." }
        );
        return;
      }

      const [linePredictionResponse, bandPredictionResponse, productHistoryResponse] = await Promise.all([
        fetchPrediction(nextTicker, {
          timeframe: nextTimeframe,
          runId: PRODUCT_LINE_RUN_ID,
        }).catch(() => null),
        fetchPrediction(nextTicker, {
          timeframe: nextTimeframe,
          runId: PRODUCT_BAND_RUN_ID,
        }).catch(() => null),
        fetchProductPredictionHistory(nextTicker, {
          timeframe: "1D",
          roles: "all",
          lookbackDays: PRODUCT_HISTORY_LOOKBACK_DAYS,
        }).catch(() => null),
      ]);

      setLinePredictionHistory(productHistoryResponse?.data.line_history ?? []);
      setBandPredictionHistory(productHistoryResponse?.data.band_history ?? []);

      const lineOverlayCheck = linePredictionResponse ? checkLineOverlay(linePredictionResponse.data) : null;
      const bandOverlayCheck = bandPredictionResponse ? checkBandOverlay(bandPredictionResponse.data) : null;
      const lineReady = Boolean(linePredictionResponse && lineOverlayCheck?.ok);
      const bandReady = Boolean(bandPredictionResponse && isActualBandPrediction(bandPredictionResponse.data) && bandOverlayCheck?.ok);

      if (lineReady && linePredictionResponse) {
        setPrediction(linePredictionResponse.data);
        setPredictionProvenance({
          latestRunId: PRODUCT_LINE_RUN_ID,
          selectedRunId: linePredictionResponse.data.run_id,
          isFallback: false,
        });
      } else {
        setPrediction(null);
        setPredictionProvenance({
          latestRunId: PRODUCT_LINE_RUN_ID,
          selectedRunId: null,
          isFallback: false,
        });
      }

      if (bandReady && bandPredictionResponse) {
        setBandPrediction(bandPredictionResponse.data);
        try {
          const evaluationsResponse = await fetchRunEvaluations(bandPredictionResponse.data.run_id, {
            ticker: nextTicker,
            timeframe: nextTimeframe,
            limit: 20,
          });
          setBandEvaluation(selectEvaluation(evaluationsResponse.data, bandPredictionResponse.data));
        } catch {
          setBandEvaluation(null);
        }
      } else {
        setBandPrediction(null);
        setBandEvaluation(null);
      }

      if (priceRows.length === 0 && (lineReady || bandReady)) {
        setAiState({
          kind: "empty",
          message: "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다.",
        });
      } else if (lineReady && bandReady) {
        setAiState({ kind: "ready", message: "보수적 예측선과 AI 밴드가 저장된 모델 결과에서 연결되었습니다." });
      } else if (!lineReady && bandReady) {
        setAiState({ kind: "ready", message: "보수적 예측선 모델은 준비됨, 예측 저장 대기. AI 밴드는 표시 중입니다." });
      } else if (lineReady && !bandReady) {
        setAiState({ kind: "ready", message: "보수적 예측선은 표시 중입니다. AI 밴드는 저장 대기 상태입니다." });
      } else {
        setAiState({ kind: "empty", message: "모델 후보는 준비됐지만 표시 가능한 저장 예측이 없습니다." });
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
      setBandEvaluation(null);
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
  const changePercent = getChangePercent(priceData);
  const hasPriceData = priceData.length > 0;
  const aiDisabled = timeframe === "1M";
  const activeLineModelDisplayName = timeframe === "1W" ? "1W 보수적 예측선 비워둠" : LINE_MODEL_DISPLAY_NAME;
  const activeBandModelDisplayName = timeframe === "1W" ? "1W AI 밴드 비워둠" : BAND_MODEL_DISPLAY_NAME;
  const forecastUnitLabel = timeframe === "1W" ? "주" : "거래일";
  const rawActivePrediction = prediction && timeframe !== "1M" && checkLineOverlay(prediction).ok ? prediction : null;
  const lineOutOfPriceRange = Boolean(hasPriceData && rawActivePrediction && !isPredictionLineWithinPriceRange(rawActivePrediction, priceData));
  const activePrediction = rawActivePrediction && !lineOutOfPriceRange ? rawActivePrediction : null;
  const activeBandPrediction =
    bandPrediction && timeframe === "1D" && isActualBandPrediction(bandPrediction) && checkBandOverlay(bandPrediction).ok
      ? bandPrediction
      : null;
  const activeLineHistory = useMemo(
    () =>
      timeframe !== "1D"
        ? []
        : linePredictionHistory
            .filter((item) => isValidDate(item.asof_date) && Number.isFinite(item.value)),
    [linePredictionHistory, timeframe]
  );
  const activeBandHistory = useMemo(
    () =>
      timeframe !== "1D"
        ? []
        : bandPredictionHistory.filter(
            (item) => isValidDate(item.asof_date) && Number.isFinite(item.lower) && Number.isFinite(item.upper)
          ),
    [bandPredictionHistory, timeframe]
  );
  const chartPriceData = priceData;
  const chartTimelineDates = useMemo(
    () =>
      uniqueDates([
          ...chartPriceData.map((row) => row.date),
          ...(activePrediction?.forecast_dates ?? []),
          ...(activeBandPrediction?.forecast_dates ?? []),
          ...activeLineHistory.map((row) => row.asof_date),
          ...activeBandHistory.map((row) => row.asof_date),
        ]),
    [activeBandHistory, activeBandPrediction, activeLineHistory, activePrediction, chartPriceData]
  );
  const chartActualDateSet = useMemo(() => new Set(chartPriceData.map((row) => row.date)), [chartPriceData]);
  const indicatorRowsInChartRange = useMemo(
    () => indicatorData.filter((row) => chartActualDateSet.has(row.date)),
    [chartActualDateSet, indicatorData]
  );
  const conservativeValue = getLastFinite(getPredictionLineValues(activePrediction));
  const changeClass = changePercent == null ? "is-flat" : changePercent < 0 ? "is-down" : "is-up";
  const bandLayerDisabled = aiDisabled || !hasPriceData || !activeBandPrediction;
  const lineLayerDisabled = aiDisabled || !hasPriceData || !activePrediction;
  const volumeLayerDisabled = priceData.every((row) => !Number.isFinite(row.volume) || row.volume == null || row.volume <= 0);
  const aiBandWidthPoints = useMemo(
    () => buildAiBandWidthPoints(activeBandHistory, activeBandPrediction).filter((point) => chartActualDateSet.has(point.date)),
    [activeBandHistory, activeBandPrediction, chartActualDateSet]
  );
  const availableIndicatorDefinitions = useMemo(() => {
    const dbDefinitions = INDICATOR_DEFINITIONS.filter((definition) => hasIndicatorValues(indicatorRowsInChartRange, definition));
    const aiBandWidthDefinition = INDICATOR_DEFINITIONS.find((definition) => definition.id === "ai_band_width");
    if (timeframe === "1D" && aiBandWidthDefinition && (aiBandWidthPoints.length > 0 || selectedIndicators.includes("ai_band_width"))) {
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
    : !hasPriceData && rawActivePrediction
    ? "가격 데이터 연결 대기"
    : lineOutOfPriceRange
    ? "보수적 예측선 후보(차트 숨김)"
    : activePrediction
    ? "보수적 예측선 후보"
    : "예측 저장 대기";
  const bandStatusLabel = aiDisabled
    ? "-"
    : timeframe === "1W"
    ? "검증 중"
    : !hasPriceData && activeBandPrediction
    ? "가격 데이터 연결 대기"
    : activeBandPrediction
    ? "위험 범위 후보"
    : "예측 저장 대기";
  const provenanceLabel = isLegacyArtifact
    ? "이전 실험 결과"
    : predictionProvenance.isFallback
    ? "저장된 예측 결과"
    : activePrediction || activeBandPrediction
    ? "제품 후보"
    : "-";
  const provenanceSummary =
    rawActivePrediction || activeBandPrediction
      ? activeBandPrediction
        ? `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""} / ${activeBandModelDisplayName}`
        : `${activeLineModelDisplayName}${lineOutOfPriceRange ? " 차트 숨김" : ""}`
      : "표시 중인 예측 결과가 없습니다.";
  const forecastDates = activePrediction?.forecast_dates ?? activeBandPrediction?.forecast_dates ?? [];
  const forecastBaseDate = activePrediction?.asof_date ?? activeBandPrediction?.asof_date ?? "-";
  const displayNotice = lineOutOfPriceRange
    ? "보수적 예측선 후보가 현재 가격 범위를 벗어나 차트에서는 숨겼습니다. AI 밴드는 유지합니다."
    : !hasPriceData && (rawActivePrediction || activeBandPrediction)
    ? "예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다."
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
          Lens는 가격 차트 위에 보수적 예측선과 AI 밴드를 겹쳐 보며, 리스크를 먼저 확인하는 투자 보조 도구입니다.
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
            ) : (
              <div className="empty-state empty-state--stacked">
                <strong>가격 데이터 없음</strong>
                <span>{priceState.message}</span>
                {rawActivePrediction || activeBandPrediction ? (
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
            <div className="eyebrow">예측 레이어</div>
            <h2>모델 정보</h2>
          </div>

          <div className="provenance-card">
            <span className="eyebrow">예측 출처</span>
            <strong>{provenanceSummary}</strong>
            <div className="provenance-grid">
              <span>보수적 예측선</span>
              <strong>{activeLineModelDisplayName}</strong>
              <span>AI 밴드</span>
              <strong>{activeBandModelDisplayName}</strong>
              <span>모델 기준일</span>
              <strong>{forecastBaseDate}</strong>
              <span>예측 기간</span>
              <strong>{forecastDates.length > 0 ? `${forecastDates.length}${forecastUnitLabel}` : "-"}</strong>
            </div>
            {!aiDisabled && (rawActivePrediction || activeBandPrediction) ? (
              <details className="provenance-details">
                <summary>상세 정보</summary>
                <div className="provenance-grid provenance-grid--composite">
                  <span>표시 기준</span>
                  <strong>{provenanceLabel}</strong>
                  <span>보수적 예측선 상태</span>
                  <strong>{lineStatusLabel}</strong>
                  <span>밴드 상태</span>
                  <strong>{bandStatusLabel}</strong>
                  <span>보수적 예측선 실행 ID</span>
                  <strong>{rawActivePrediction?.run_id ?? (timeframe === "1W" ? "비워둠" : PRODUCT_LINE_RUN_ID)}</strong>
                  <span>밴드 실행 ID</span>
                  <strong>{activeBandPrediction?.run_id ?? (timeframe === "1W" ? "검증 중" : PRODUCT_BAND_RUN_ID)}</strong>
                </div>
              </details>
            ) : null}
          </div>

          <div className="layer-group">
            <div className="layer-group__title">가격 오버레이</div>
            <LayerToggle
              label="보수적 예측선"
              checked={!lineLayerDisabled && layers.conservativeLine}
              disabled={lineLayerDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, conservativeLine: checked }))}
            />
            <p className="layer-description">하방 위험을 더 조심스럽게 반영한 AI 예측선입니다.</p>
            <LayerToggle
              label="AI 밴드"
              checked={!bandLayerDisabled && layers.aiBand}
              disabled={bandLayerDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, aiBand: checked }))}
            />
            <p className="layer-description">
              {timeframe === "1W" ? "1W AI 밴드는 검증 중입니다. 저장된 1W 예측은 보수적 예측선만 표시합니다." : "예상 변동 범위를 보여주는 리스크 참고 지표입니다."}
            </p>
            {!bandLayerDisabled && layers.aiBand ? (
              <div className="layer-metrics">
                <div>
                  <span>커버리지</span>
                  <strong>{formatRatio(bandEvaluation?.coverage)}</strong>
                </div>
                <div>
                  <span>평균 밴드 폭</span>
                  <strong>{formatNumber(bandEvaluation?.avg_band_width)}</strong>
                </div>
              </div>
            ) : null}
            {!lineLayerDisabled && layers.conservativeLine ? (
              <div className="layer-metrics">
                <div>
                  <span>예측가</span>
                  <strong>{formatNumber(conservativeValue)}</strong>
                </div>
                <div>
                  <span>모델 기준일</span>
                  <strong>{activePrediction?.asof_date ?? "-"}</strong>
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
