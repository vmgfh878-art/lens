-- ============================================================
-- CP204 v1 Predictions: 3 flat tables for frontend display
-- Created: 2026-05-20
-- Purpose: Frontend reads predictions directly (slot-based)
-- ============================================================

-- 1D Line (flat, one row per ticker/asof_date)
CREATE TABLE IF NOT EXISTS public.predictions_line_1d (
    ticker          VARCHAR(20) NOT NULL,
    asof_date       DATE NOT NULL,
    line_score                  DOUBLE PRECISION,
    safe_line_score             DOUBLE PRECISION,
    line_rank_by_date           DOUBLE PRECISION,
    safe_line_rank_by_date      DOUBLE PRECISION,
    line_top_decile_flag        BOOLEAN,
    safe_line_top_decile_flag   BOOLEAN,
    actual_h5_return            DOUBLE PRECISION,
    model_id        VARCHAR(100),
    source_cp       VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, asof_date)
);

CREATE INDEX IF NOT EXISTS idx_predictions_line_1d_asof
    ON public.predictions_line_1d (asof_date DESC, ticker);

-- 1D Band: per-horizon flat
CREATE TABLE IF NOT EXISTS public.predictions_band_1d (
    ticker        VARCHAR(20) NOT NULL,
    asof_date     DATE NOT NULL,
    forecast_date DATE NOT NULL,
    horizon_step  INT NOT NULL,
    band_lower    DOUBLE PRECISION,
    band_upper    DOUBLE PRECISION,
    actual_return DOUBLE PRECISION,
    actual_return_available BOOLEAN,
    model_id      VARCHAR(100),
    source_cp     VARCHAR(50),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, asof_date, horizon_step)
);

CREATE INDEX IF NOT EXISTS idx_predictions_band_1d_asof
    ON public.predictions_band_1d (asof_date DESC, ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_band_1d_forecast
    ON public.predictions_band_1d (forecast_date DESC, ticker);

-- 1W Band: ensemble-averaged per-horizon flat
CREATE TABLE IF NOT EXISTS public.predictions_band_1w (
    ticker        VARCHAR(20) NOT NULL,
    asof_date     DATE NOT NULL,
    horizon_step  INT NOT NULL,
    band_lower    DOUBLE PRECISION,
    band_upper    DOUBLE PRECISION,
    actual_return DOUBLE PRECISION,
    model_id      VARCHAR(100),
    source_cp     VARCHAR(50),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, asof_date, horizon_step)
);

CREATE INDEX IF NOT EXISTS idx_predictions_band_1w_asof
    ON public.predictions_band_1w (asof_date DESC, ticker);

-- RLS disable (학교 데모 — anon key 로 frontend 직접 읽기 허용)
ALTER TABLE public.predictions_line_1d DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions_band_1d DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions_band_1w DISABLE ROW LEVEL SECURITY;
