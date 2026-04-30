"use client";

import { FormEvent, startTransition, useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";

import {
  DisplayTimeframe,
  EvaluationSummary,
  fetchAiRuns,
  fetchIndicators,
  fetchPrediction,
  fetchPrices,
  fetchRunEvaluations,
  fetchTickers,
  IndicatorPoint,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
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
  | "atr_ratio";

interface AiState {
  kind: "loading" | "ready" | "disabled" | "empty";
  message: string;
}

interface ApiErrorShape {
  code: string;
  message: string;
}

interface IndicatorDefinition {
  id: IndicatorId;
  label: string;
  field: keyof IndicatorPoint;
  category: "기본" | "추세" | "변동성/위치";
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
  selectedRunModelName: string | null;
  isFallback: boolean;
}

const TIMEFRAME_OPTIONS: Array<{ value: DisplayTimeframe; label: string }> = [
  { value: "1D", label: "1D" },
  { value: "1W", label: "1W" },
  { value: "1M", label: "1M" },
];

const AI_RUN_SCAN_LIMIT = 10;
const AI_RUN_MODEL_CANDIDATES = new Set(["line_band_composite", "patchtst"]);
const DEFAULT_INDICATORS: IndicatorId[] = ["rsi", "macd_ratio"];

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

function formatModelName(value: unknown) {
  const modelName = typeof value === "string" ? value : "";
  if (modelName === "patchtst") {
    return "PatchTST";
  }
  if (modelName === "cnn_lstm") {
    return "CNN-LSTM";
  }
  if (modelName === "line_band_composite" || modelName.includes("line__")) {
    return "Composite";
  }
  return modelName || "-";
}

function formatCalibration(value: unknown) {
  if (value === "scalar_width") {
    return "scalar width calibration";
  }
  return typeof value === "string" && value ? value : "-";
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
    description: "과열과 침체 흐름을 0~100 범위로 봅니다.",
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
    description: "단기와 장기 추세 차이를 0 기준으로 봅니다.",
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
    description: "평균 변동폭 비율을 봅니다. 값이 있을 때만 표시됩니다.",
    color: "#d97706",
    baseline: 0,
    transform: ratioToPercent,
    formatLatest: formatSignedPercentPoint,
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
      message: "월봉은 가격 지표만 제공됩니다.",
    };
  }
  return {
    kind: "empty",
    message: "저장된 최신 예측 결과가 아직 없습니다.",
  };
}

function getIndicatorValue(row: IndicatorPoint, definition: IndicatorDefinition) {
  const rawValue = row[definition.field];
  if (!isFiniteNumber(rawValue)) {
    return null;
  }
  return definition.transform ? definition.transform(rawValue) : rawValue;
}

function hasIndicatorValues(rows: IndicatorPoint[], definition: IndicatorDefinition) {
  return rows.some((row) => isFiniteNumber(getIndicatorValue(row, definition)));
}

function buildFieldPoints(rows: IndicatorPoint[], definition: IndicatorDefinition, allowedDates?: Set<string>): IndicatorChartPoint[] {
  return rows
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
    .filter((point): point is IndicatorChartPoint => point !== null);
}

function isDateBefore(left: string, right: string) {
  return left < right;
}

function uniqueDates(dates: string[]) {
  return dates.filter((date, index, source) => date && source.indexOf(date) === index);
}

function sortRunsByCreatedAt<T extends { created_at: string | null }>(runs: T[]) {
  return [...runs].sort((left, right) => {
    const leftTime = left.created_at ? Date.parse(left.created_at) : 0;
    const rightTime = right.created_at ? Date.parse(right.created_at) : 0;
    return rightTime - leftTime;
  });
}

function isCompositePrediction(prediction: PredictionResult | null) {
  if (!prediction) {
    return false;
  }
  return (
    prediction.model_name.includes("line__") ||
    prediction.model_name.includes("composite") ||
    Boolean(prediction.meta?.line_model_run_id || prediction.meta?.band_model_run_id)
  );
}

function checkPredictionOverlay(prediction: PredictionResult): PredictionOverlayCheck {
  const forecastDates = prediction.forecast_dates ?? [];
  const lineValues = prediction.line_series?.length ? prediction.line_series : prediction.conservative_series;

  if (forecastDates.length === 0) {
    return { ok: false, message: "예측 날짜가 없어 예측선을 숨겼습니다." };
  }
  if (
    lineValues.length !== forecastDates.length ||
    prediction.upper_band_series.length !== forecastDates.length ||
    prediction.lower_band_series.length !== forecastDates.length
  ) {
    return { ok: false, message: "예측 날짜와 시리즈 길이가 맞지 않아 예측선을 숨겼습니다." };
  }

  const allValues = [...lineValues, ...prediction.upper_band_series, ...prediction.lower_band_series];
  if (forecastDates.some((date) => !date) || allValues.some((value) => !Number.isFinite(value))) {
    return { ok: false, message: "예측 데이터에 표시할 수 없는 값이 있어 예측선을 숨겼습니다." };
  }

  return { ok: true, message: null };
}

function buildVisibleIndicatorSeries(params: {
  selectedIndicators: IndicatorId[];
  indicatorData: IndicatorPoint[];
  indicatorErrorMessage: string | null;
  availableDefinitions: IndicatorDefinition[];
  allowedDates?: Set<string>;
}): IndicatorChartSeries[] {
  const { selectedIndicators, indicatorData, indicatorErrorMessage, availableDefinitions, allowedDates } = params;
  const indicatorEmptyMessage = indicatorErrorMessage ?? "저장된 보조지표 데이터가 없습니다.";

  return availableDefinitions
    .filter((definition) => selectedIndicators.includes(definition.id))
    .map((definition) => {
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
  });
  const [selectedIndicators, setSelectedIndicators] = useState<IndicatorId[]>(DEFAULT_INDICATORS);
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorPoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionProvenance, setPredictionProvenance] = useState<PredictionProvenance>({
    latestRunId: null,
    selectedRunId: null,
    selectedRunModelName: null,
    isFallback: false,
  });
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [aiState, setAiState] = useState<AiState>({ kind: "loading", message: "예측 확인 중" });
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
    setPredictionProvenance({
      latestRunId: null,
      selectedRunId: null,
      selectedRunModelName: null,
      isFallback: false,
    });

    try {
      const pricesResponse = await fetchPrices(nextTicker, { timeframe: nextTimeframe });
      setPriceData(pricesResponse.data.data);

      try {
        const indicatorsResponse = await fetchIndicators(nextTicker, { timeframe: nextTimeframe, limit: 300 });
        setIndicatorData(indicatorsResponse.data.data);
        setIndicatorErrorMessage(null);
      } catch {
        setIndicatorData([]);
        setIndicatorErrorMessage("보조지표를 불러오지 못했습니다.");
      }

      if (!isPredictionTimeframeEnabled(nextTimeframe)) {
        setPrediction(null);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          selectedRunModelName: null,
          isFallback: false,
        });
        setEvaluation(null);
        setAiState(buildAiState(nextTimeframe));
        return;
      }

      const runsResponse = await fetchAiRuns({
        modelName: "",
        timeframe: nextTimeframe,
        status: "completed",
        limit: 50,
      }).catch((error) => {
        setPrediction(null);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          selectedRunModelName: null,
          isFallback: false,
        });
        setEvaluation(null);
        setAiState({ kind: "empty", message: `예측 조회 오류: ${extractErrorMessage(error)}` });
        return null;
      });
      if (!runsResponse) {
        return;
      }

      const completedRuns = sortRunsByCreatedAt(
        runsResponse.data.filter((run) => AI_RUN_MODEL_CANDIDATES.has(String(run.model_name ?? "")))
      );
      const latestRunId = completedRuns[0]?.run_id ?? null;
      if (completedRuns.length === 0) {
        setPrediction(null);
        setPredictionProvenance({
          latestRunId: null,
          selectedRunId: null,
          selectedRunModelName: null,
          isFallback: false,
        });
        setEvaluation(null);
        setAiState({ kind: "empty", message: "완료된 예측 run이 없어 예측 범위를 표시할 수 없습니다." });
        return;
      }

      let missingPredictionCount = 0;
      let invalidPredictionCount = 0;
      let lastInvalidPredictionMessage: string | null = null;
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
            setPredictionProvenance({
              latestRunId,
              selectedRunId: null,
              selectedRunModelName: null,
              isFallback: false,
            });
            setEvaluation(null);
            setAiState(buildAiState(nextTimeframe));
            return null;
          }
          setPrediction(null);
          setPredictionProvenance({
            latestRunId,
            selectedRunId: null,
            selectedRunModelName: null,
            isFallback: false,
          });
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
        const overlayCheck = checkPredictionOverlay(predictionResponse.data);
        if (!overlayCheck.ok) {
          invalidPredictionCount += 1;
          lastInvalidPredictionMessage = overlayCheck.message;
          continue;
        }

        setPrediction(predictionResponse.data);
        setPredictionProvenance({
          latestRunId,
          selectedRunId: predictionResponse.data.run_id,
          selectedRunModelName: run.model_name,
          isFallback: Boolean(latestRunId && latestRunId !== predictionResponse.data.run_id),
        });
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
      setPredictionProvenance({
        latestRunId,
        selectedRunId: null,
        selectedRunModelName: null,
        isFallback: false,
      });
      setEvaluation(null);
      setAiState({
        kind: "empty",
        message:
          invalidPredictionCount > 0
            ? lastInvalidPredictionMessage ?? "저장된 예측 row는 있으나 표시할 수 없습니다."
            : missingPredictionCount > 0
            ? "저장된 예측 없음: completed run에 이 티커 prediction row가 없습니다."
            : "저장된 예측 없음",
      });
    } catch (error) {
      setPriceData([]);
      setIndicatorData([]);
      setPrediction(null);
      setPredictionProvenance({
        latestRunId: null,
        selectedRunId: null,
        selectedRunModelName: null,
        isFallback: false,
      });
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
  const aiDisabled = timeframe === "1M";
  const activePrediction = prediction && timeframe !== "1M" && checkPredictionOverlay(prediction).ok ? prediction : null;
  const firstForecastDate = activePrediction?.forecast_dates[0] ?? null;
  const chartPriceData = useMemo(() => {
    if (!firstForecastDate) {
      return priceData;
    }
    const pastRows = priceData.filter((row) => isDateBefore(row.date, firstForecastDate));
    return pastRows.length > 0 ? pastRows : priceData;
  }, [firstForecastDate, priceData]);
  const chartTimelineDates = useMemo(
    () => uniqueDates([...chartPriceData.map((row) => row.date), ...(activePrediction?.forecast_dates ?? [])]),
    [activePrediction, chartPriceData]
  );
  const chartActualDateSet = useMemo(() => new Set(chartPriceData.map((row) => row.date)), [chartPriceData]);
  const indicatorRowsInChartRange = useMemo(
    () => indicatorData.filter((row) => chartActualDateSet.has(row.date)),
    [chartActualDateSet, indicatorData]
  );
  const conservativeValue = getLastFinite(prediction?.line_series?.length ? prediction.line_series : prediction?.conservative_series);
  const changeClass = changePercent == null ? "is-flat" : changePercent < 0 ? "is-down" : "is-up";
  const predictionStatus =
    aiState.kind === "ready" ? "표시 중" : aiState.kind === "disabled" ? "가격 전용" : aiState.kind === "loading" ? "확인 중" : "없음";
  const availableIndicatorDefinitions = useMemo(
    () => INDICATOR_DEFINITIONS.filter((definition) => hasIndicatorValues(indicatorRowsInChartRange, definition)),
    [indicatorRowsInChartRange]
  );
  const visibleIndicatorSeries = useMemo(
    () =>
      buildVisibleIndicatorSeries({
        selectedIndicators,
        indicatorData: indicatorRowsInChartRange,
        indicatorErrorMessage,
        availableDefinitions: availableIndicatorDefinitions,
        allowedDates: chartActualDateSet,
      }),
    [selectedIndicators, indicatorRowsInChartRange, indicatorErrorMessage, availableIndicatorDefinitions, chartActualDateSet]
  );
  const indicatorTimelineDates = visibleTimelineDates.length > 0 ? visibleTimelineDates : chartTimelineDates;
  const activePredictionMeta = activePrediction?.meta ?? {};
  const isComposite = isCompositePrediction(activePrediction);
  const provenanceLabel = predictionProvenance.isFallback ? "fallback run" : activePrediction ? "latest run" : "-";
  const provenanceSummary = activePrediction
    ? `저장된 예측 run: ${activePrediction.run_id} / 기준일: ${activePrediction.asof_date} / 예측 ${activePrediction.horizon}거래일`
    : "표시 중인 예측 run이 없습니다.";
  const provenanceNote =
    activePrediction && predictionProvenance.isFallback
      ? "latest completed run에 이 티커의 usable prediction이 없어 저장된 fallback run을 표시합니다."
      : "latest completed run의 저장된 prediction을 표시합니다.";
  const lineModelName = getMetaString(activePredictionMeta, "line_model_name") ?? "patchtst";
  const bandModelName = getMetaString(activePredictionMeta, "band_model_name") ?? null;
  const bandCalibration = getMetaString(activePredictionMeta, "band_calibration_method");
  const compositionPolicy = getMetaString(activePredictionMeta, "composition_policy");
  const featureContract =
    getMetaString(activePredictionMeta, "feature_contract") ?? getMetaString(activePredictionMeta, "feature_contract_version");
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
                layers={layers}
                timelineDates={chartTimelineDates}
                onVisibleDatesChange={handleVisibleDatesChange}
              />
            ) : (
              <div className="empty-state">가격 데이터가 없습니다.</div>
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
            <div className="eyebrow">예측 범위</div>
            <h2>{predictionStatus}</h2>
          </div>

          <div className="provenance-card">
            <span className="eyebrow">예측 출처</span>
            <strong>{provenanceSummary}</strong>
            {activePrediction ? <p>{provenanceNote}</p> : null}
            <div className="provenance-grid">
              <span>표시 기준</span>
              <strong>{provenanceLabel}</strong>
              <span>run 모델</span>
              <strong>{predictionProvenance.selectedRunModelName ?? activePrediction?.model_name ?? "-"}</strong>
              <span>예측 모델</span>
              <strong>{formatModelName(activePrediction?.model_name)}</strong>
            </div>
            {isComposite ? (
              <div className="provenance-grid provenance-grid--composite">
                <span>예측선 모델</span>
                <strong>{formatModelName(lineModelName)}</strong>
                <span>밴드 모델</span>
                <strong>{formatModelName(bandModelName)}</strong>
                <span>밴드 보정</span>
                <strong>{formatCalibration(bandCalibration)}</strong>
                <span>조합 정책</span>
                <strong>{compositionPolicy ?? "-"}</strong>
                <span>feature contract</span>
                <strong>{featureContract ?? "-"}</strong>
              </div>
            ) : null}
          </div>

          <div className="layer-group">
            <LayerToggle
              label="AI 밴드"
              checked={!aiDisabled && layers.aiBand}
              disabled={aiDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, aiBand: checked }))}
            />
            <p className="layer-description">모델이 예측한 향후 가격 변동 가능 범위입니다.</p>
            {!aiDisabled && layers.aiBand ? (
              <div className="layer-metrics">
                <div>
                  <span>커버리지</span>
                  <strong>{formatRatio(evaluation?.coverage)}</strong>
                </div>
                <div>
                  <span>평균 밴드 폭</span>
                  <strong>{formatNumber(evaluation?.avg_band_width)}</strong>
                </div>
              </div>
            ) : null}
          </div>

          <div className="layer-group">
            <LayerToggle
              label="보수적 예측"
              checked={!aiDisabled && layers.conservativeLine}
              disabled={aiDisabled}
              onChange={(checked) => setLayers((previous) => ({ ...previous, conservativeLine: checked }))}
            />
            <p className="layer-description">하방 리스크를 우선 반영한 방어적 예측선입니다.</p>
            {!aiDisabled && layers.conservativeLine ? (
              <div className="layer-metrics">
                <div>
                  <span>예측가</span>
                  <strong>{formatNumber(conservativeValue)}</strong>
                </div>
                <div>
                  <span>기준일</span>
                  <strong>{prediction?.asof_date ?? "-"}</strong>
                </div>
              </div>
            ) : null}
          </div>

          <div className="layer-group">
            <details className="indicator-menu" open>
              <summary>
                <span>
                  <span className="eyebrow">보조지표</span>
                  <strong>보조지표 추가</strong>
                </span>
              </summary>
              <div className="indicator-selector">
                {["기본", "추세", "변동성/위치"].map((category) => {
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

          <div className="notice">{aiState.message}</div>
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
