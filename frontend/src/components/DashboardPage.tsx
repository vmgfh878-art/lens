"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  DisplayTimeframe,
  fetchPrediction,
  fetchPrices,
  fetchTickers,
  isPredictionTimeframeEnabled,
  PredictionResult,
  PriceBar,
  StockSummary,
} from "@/api/client";
import Chart from "@/components/Chart";

const TIMEFRAME_OPTIONS = [
  { value: "1D", label: "일봉" },
  { value: "1W", label: "주봉" },
  { value: "1M", label: "월봉" },
] as const;

const MODEL_OPTIONS = [
  { value: "patchtst", label: "PatchTST" },
  { value: "cnn_lstm", label: "CNN-LSTM" },
  { value: "tide", label: "TiDE" },
] as const;

type ModelOption = (typeof MODEL_OPTIONS)[number]["value"];

interface AiPanelState {
  kind: "ready" | "disabled" | "empty";
  title: string;
  description: string;
}

interface ApiErrorShape {
  code: string;
  message: string;
}

function formatNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: 2,
  }).format(value);
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

function getSignalClassName(signal: PredictionResult["signal"] | null) {
  if (!signal) {
    return "signal--HOLD";
  }
  return `signal--${signal}`;
}

function extractApiError(error: unknown): ApiErrorShape | null {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const payload = (error as { response?: { data?: { error?: { code?: string; message?: string } } } }).response?.data?.error;
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
    return error.message;
  }
  return "요청 처리 중 알 수 없는 오류가 발생했습니다.";
}

function buildDisabledState(timeframe: DisplayTimeframe): AiPanelState {
  if (timeframe === "1M") {
    return {
      kind: "disabled",
      title: "월봉 AI는 비활성입니다",
      description:
        "월봉은 Phase 1에서 가격과 기술지표만 표시합니다. AI 예측선, 밴드, 시그널은 일봉과 주봉만 제공합니다.",
    };
  }
  return {
    kind: "empty",
    title: "AI 결과가 없습니다",
    description: "현재 조건에 맞는 예측 배치가 아직 저장되지 않았습니다.",
  };
}

export default function DashboardPage() {
  const [ticker, setTicker] = useState("AAPL");
  const [timeframe, setTimeframe] = useState<DisplayTimeframe>("1D");
  const [model, setModel] = useState<ModelOption>("patchtst");
  const [priceData, setPriceData] = useState<PriceBar[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [aiState, setAiState] = useState<AiPanelState>({
    kind: "empty",
    title: "AI 결과를 불러오는 중입니다",
    description: "종목과 타임프레임에 맞는 최신 예측 배치를 확인합니다.",
  });
  const [suggestions, setSuggestions] = useState<StockSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const deferredTicker = useDeferredValue(ticker);

  useEffect(() => {
    const keyword = deferredTicker.trim();
    if (keyword.length < 1) {
      setSuggestions([]);
      return;
    }

    let active = true;
    setSearchLoading(true);

    fetchTickers({ search: keyword, limit: 6 })
      .then((response) => {
        if (!active) {
          return;
        }
        setSuggestions(response.data);
      })
      .catch(() => {
        if (!active) {
          return;
        }
        setSuggestions([]);
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

  async function loadDashboard(nextTicker: string, nextTimeframe: DisplayTimeframe, nextModel: ModelOption) {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const pricesResponse = await fetchPrices(nextTicker, { timeframe: nextTimeframe });
      setPriceData(pricesResponse.data.data);

      if (!isPredictionTimeframeEnabled(nextTimeframe)) {
        setPrediction(null);
        setAiState(buildDisabledState(nextTimeframe));
        return;
      }

      try {
        const predictionResponse = await fetchPrediction(nextTicker, { timeframe: nextTimeframe, model: nextModel });
        setPrediction(predictionResponse.data);
        setAiState({
          kind: "ready",
          title: "최신 AI 예측 결과",
          description: "배치 추론으로 저장된 예측선과 밴드를 보여줍니다.",
        });
      } catch (error) {
        setPrediction(null);
        const apiError = extractApiError(error);
        if (apiError?.code === "TIMEFRAME_DISABLED") {
          setAiState(buildDisabledState(nextTimeframe));
          return;
        }
        if (apiError?.code === "RESOURCE_NOT_FOUND" || apiError?.code === "INSUFFICIENT_HISTORY") {
          setAiState({
            kind: "empty",
            title: "AI 결과가 아직 없습니다",
            description: apiError.message,
          });
          return;
        }
        throw error;
      }
    } catch (error) {
      setPriceData([]);
      setPrediction(null);
      setAiState({
        kind: "empty",
        title: "차트 데이터를 불러오지 못했습니다",
        description: "백엔드 연결 또는 데이터 적재 상태를 확인해주세요.",
      });
      setErrorMessage(extractErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard("AAPL", "1D", "patchtst");
  }, []);

  function handleSubmit() {
    const nextTicker = ticker.trim().toUpperCase();
    setTicker(nextTicker);
    startTransition(() => {
      loadDashboard(nextTicker, timeframe, model);
    });
  }

  function handleSuggestionClick(nextTicker: string) {
    setTicker(nextTicker);
    startTransition(() => {
      loadDashboard(nextTicker, timeframe, model);
    });
  }

  const currentClose = priceData.length > 0 ? priceData[priceData.length - 1].close : null;
  const upperBand = prediction?.upper_band_series.at(-1) ?? null;
  const lowerBand = prediction?.lower_band_series.at(-1) ?? null;
  const centerBand = prediction?.line_series.at(-1) ?? prediction?.conservative_series.at(-1) ?? null;

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <div className="hero__eyebrow">Lens v2 Dashboard</div>
          <h1 className="hero__title">가격 흐름과 AI 보조지표를 함께 보는 투자 보조 화면</h1>
          <p className="hero__body">
            Lens는 AI를 매수 지시가 아닌 보조지표로 다룹니다. 일봉과 주봉에서는 예측선과 밴드를, 월봉에서는 가격과
            기술지표만 보여줍니다.
          </p>
        </div>

        <aside className="hero-card">
          <span className="hero-card__label">현재 선택</span>
          <h2 className="hero-card__ticker">{ticker || "AAPL"}</h2>
          <div className="hero-card__meta">
            <span className={`hero-card__signal ${getSignalClassName(prediction?.signal ?? null)}`}>
              {prediction?.signal ?? (aiState.kind === "disabled" ? "비활성" : "대기 중")}
            </span>
            <div>
              <div className="section-label">모델</div>
              <div>{MODEL_OPTIONS.find((item) => item.value === model)?.label}</div>
            </div>
            <div>
              <div className="section-label">타임프레임</div>
              <div>{timeframe}</div>
            </div>
          </div>
        </aside>
      </section>

      <section className="dashboard-grid">
        <div className="stack">
          <section className="panel">
            <div className="panel__header">
              <div>
                <div className="section-label">입력</div>
                <h2 className="panel__title">종목과 모델을 선택해서 조회</h2>
              </div>
            </div>

            <div className="control-grid">
              <div className="field">
                <label htmlFor="ticker">종목 티커</label>
                <input
                  id="ticker"
                  value={ticker}
                  onChange={(event) => setTicker(event.target.value.toUpperCase())}
                  placeholder="예: AAPL"
                />
              </div>

              <div className="field">
                <label htmlFor="timeframe">차트 단위</label>
                <select
                  id="timeframe"
                  value={timeframe}
                  onChange={(event) => setTimeframe(event.target.value as DisplayTimeframe)}
                >
                  {TIMEFRAME_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label} ({option.value})
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label htmlFor="model">AI 모델</label>
                <select
                  id="model"
                  value={model}
                  onChange={(event) => setModel(event.target.value as ModelOption)}
                >
                  {MODEL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>상태</label>
                <input value={isLoading ? "조회 중" : "조회 가능"} readOnly />
              </div>
            </div>

            <div className="actions">
              <button className="button button--primary" type="button" onClick={handleSubmit} disabled={isLoading}>
                {isLoading ? "불러오는 중..." : "가격과 AI 결과 조회"}
              </button>
              <button
                className="button button--secondary"
                type="button"
                onClick={() => {
                  setTicker("AAPL");
                  setTimeframe("1D");
                  setModel("patchtst");
                  startTransition(() => {
                    loadDashboard("AAPL", "1D", "patchtst");
                  });
                }}
                disabled={isLoading}
              >
                기본값으로 되돌리기
              </button>
            </div>

            <p className="inline-note">
              월봉은 표시 전용입니다. 가격 차트와 기술지표는 보여주지만, AI 예측선·밴드·시그널은 노출하지 않습니다.
            </p>
          </section>

          {errorMessage ? (
            <div className="status-box status-box--error">
              <strong>조회 실패</strong>
              <div>{errorMessage}</div>
            </div>
          ) : null}

          <section className="panel chart-shell">
            <div className="panel__header">
              <div>
                <div className="section-label">가격 차트</div>
                <h2 className="panel__title">{ticker} 가격 흐름</h2>
              </div>
            </div>

            {priceData.length > 0 ? (
              <Chart data={priceData} ticker={ticker} timeframe={timeframe} />
            ) : (
              <div className="empty-state">
                가격 데이터가 아직 없습니다. 백엔드 수집 상태와 해당 종목 적재 여부를 먼저 확인해주세요.
              </div>
            )}
          </section>
        </div>

        <aside className="stack">
          <section className="panel">
            <div className="panel__header">
              <div>
                <div className="section-label">AI 요약</div>
                <h2 className="panel__title">{aiState.title}</h2>
              </div>
            </div>

            {prediction && aiState.kind === "ready" ? (
              <div className="metric-grid">
                <article className="metric-card">
                  <div className="section-label">시그널</div>
                  <div className="metric-card__value">{prediction.signal}</div>
                  <div className="metric-card__hint">저장된 최신 배치 추론 결과를 그대로 표시합니다.</div>
                </article>

                <article className="metric-card">
                  <div className="section-label">기준일</div>
                  <div className="metric-card__value">{prediction.asof_date}</div>
                  <div className="metric-card__hint">해당 날짜 기준으로 생성된 최신 예측 배치입니다.</div>
                </article>

                <article className="metric-card">
                  <div className="section-label">현재 종가</div>
                  <div className="metric-card__value">{formatNumber(currentClose)}</div>
                  <div className="metric-card__hint">현재 차트 데이터의 마지막 종가입니다.</div>
                </article>

                <article className="metric-card">
                  <div className="section-label">예측선</div>
                  <div className="metric-card__value">{formatNumber(centerBand)}</div>
                  <div className="metric-card__hint">보수적 기준으로 계산한 마지막 예측선 값입니다.</div>
                </article>

                <article className="metric-card">
                  <div className="section-label">상단 밴드</div>
                  <div className="metric-card__value">{formatNumber(upperBand)}</div>
                  <div className="metric-card__hint">미래 가격 범위의 상단 참고값입니다.</div>
                </article>

                <article className="metric-card">
                  <div className="section-label">하단 밴드</div>
                  <div className="metric-card__value">{formatNumber(lowerBand)}</div>
                  <div className="metric-card__hint">미래 가격 범위의 하단 참고값입니다.</div>
                </article>
              </div>
            ) : (
              <div className="empty-state">{aiState.description}</div>
            )}
          </section>

          <section className="list-card">
            <div className="section-label">티커 추천</div>
            <h2 className="panel__title">검색 결과</h2>

            {searchLoading ? <p className="inline-note">티커 목록을 찾는 중입니다.</p> : null}

            <div className="list">
              {suggestions.length > 0 ? (
                suggestions.map((item) => (
                  <button
                    key={item.ticker}
                    type="button"
                    className="list-item"
                    onClick={() => handleSuggestionClick(item.ticker)}
                  >
                    <div>
                      <div className="list-item__primary">{item.ticker}</div>
                      <div className="list-item__secondary">
                        {item.sector ?? "섹터 없음"} / {item.industry ?? "산업 없음"}
                      </div>
                    </div>
                    <div className="list-item__secondary">{formatCompactNumber(item.market_cap)}</div>
                  </button>
                ))
              ) : (
                <div className="empty-state">입력한 티커와 비슷한 종목을 여기에 보여줍니다.</div>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="section-label">사용 원칙</div>
            <div className="panel__body">
              Lens의 AI는 투자 결정을 대신하지 않습니다. 일봉과 주봉에서는 AI 보조지표를, 월봉에서는 가격과 기술지표만
              제공합니다. 데이터가 부족한 종목과 타임프레임은 결과 대신 안내 문구를 보여줍니다.
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
