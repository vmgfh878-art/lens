-- ============================================================
-- Lens Project - Supabase schema
-- 핵심 v1 테이블:
--   stock_info, price_data, macroeconomic_indicators,
--   market_breadth, indicators, predictions
-- 확장용 raw 테이블:
--   company_fundamentals, sector_returns
-- ============================================================

CREATE TABLE IF NOT EXISTS public.stock_info (
    ticker      VARCHAR(20) PRIMARY KEY,
    sector      VARCHAR(100),
    industry    VARCHAR(100),
    market_cap  DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.price_data (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
    date            DATE NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION NOT NULL,
    adjusted_close  DOUBLE PRECISION,
    volume          BIGINT,
    amount          DOUBLE PRECISION,
    per             DOUBLE PRECISION,
    pbr             DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_price_data_ticker_date
    ON public.price_data (ticker, date DESC);

CREATE TABLE IF NOT EXISTS public.macroeconomic_indicators (
    date                DATE PRIMARY KEY,
    cpi                 DOUBLE PRECISION,
    gdp                 DOUBLE PRECISION,
    ppi                 DOUBLE PRECISION,
    jolt                DOUBLE PRECISION,
    cci                 DOUBLE PRECISION,
    interest_rate       DOUBLE PRECISION,
    trade_balance       DOUBLE PRECISION,
    real_gdp            DOUBLE PRECISION,
    unemployment_rate   DOUBLE PRECISION,
    consumer_sentiment  DOUBLE PRECISION,
    ff_targetrate_upper DOUBLE PRECISION,
    ff_targetrate_lower DOUBLE PRECISION,
    us10y               DOUBLE PRECISION,
    us2y                DOUBLE PRECISION,
    yield_spread        DOUBLE PRECISION,
    vix_close           DOUBLE PRECISION,
    dxy_close           DOUBLE PRECISION,
    wti_price           DOUBLE PRECISION,
    gold_price          DOUBLE PRECISION,
    credit_spread_hy    DOUBLE PRECISION,
    core_cpi            DOUBLE PRECISION,
    pce                 DOUBLE PRECISION,
    core_pce            DOUBLE PRECISION,
    tradebalance_goods  DOUBLE PRECISION,
    trade_import        DOUBLE PRECISION,
    trade_export        DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS ppi DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS jolt DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS cci DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS trade_balance DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS real_gdp DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS consumer_sentiment DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS ff_targetrate_upper DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS ff_targetrate_lower DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS tradebalance_goods DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS trade_import DOUBLE PRECISION;
ALTER TABLE public.macroeconomic_indicators ADD COLUMN IF NOT EXISTS trade_export DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS public.market_breadth (
    date        DATE PRIMARY KEY,
    nh_nl_index DOUBLE PRECISION,
    ma200_pct   DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.sector_returns (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    sector      VARCHAR(100) NOT NULL,
    etf_ticker  VARCHAR(20),
    return      DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, sector)
);
CREATE INDEX IF NOT EXISTS idx_sector_returns_date
    ON public.sector_returns (date DESC);

CREATE TABLE IF NOT EXISTS public.company_fundamentals (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
    date                DATE NOT NULL,
    filing_date         DATE,
    revenue             DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    total_assets        DOUBLE PRECISION,
    total_liabilities   DOUBLE PRECISION,
    equity              DOUBLE PRECISION,
    shares_issued       DOUBLE PRECISION,
    eps                 DOUBLE PRECISION,
    roe                 DOUBLE PRECISION,
    debt_ratio          DOUBLE PRECISION,
    interest_coverage   DOUBLE PRECISION,
    operating_cash_flow DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_company_fundamentals_ticker_date
    ON public.company_fundamentals (ticker, date DESC);

CREATE TABLE IF NOT EXISTS public.indicators (
    id               BIGSERIAL PRIMARY KEY,
    ticker           VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
    timeframe        VARCHAR(4) NOT NULL DEFAULT '1D' CHECK (timeframe IN ('1D', '1W', '1M')),
    date             DATE NOT NULL,
    log_return       DOUBLE PRECISION,
    open_ratio       DOUBLE PRECISION,
    high_ratio       DOUBLE PRECISION,
    low_ratio        DOUBLE PRECISION,
    vol_change       DOUBLE PRECISION,
    ma_5_ratio       DOUBLE PRECISION,
    ma_20_ratio      DOUBLE PRECISION,
    ma_60_ratio      DOUBLE PRECISION,
    atr_ratio        DOUBLE PRECISION,
    rsi              DOUBLE PRECISION,
    macd_ratio       DOUBLE PRECISION,
    bb_position      DOUBLE PRECISION,
    us10y            DOUBLE PRECISION,
    yield_spread     DOUBLE PRECISION,
    vix_close        DOUBLE PRECISION,
    credit_spread_hy DOUBLE PRECISION,
    nh_nl_index      DOUBLE PRECISION,
    ma200_pct        DOUBLE PRECISION,
    regime_label     VARCHAR(20),
    regime_calm      DOUBLE PRECISION,
    regime_neutral   DOUBLE PRECISION,
    regime_stress    DOUBLE PRECISION,
    revenue          DOUBLE PRECISION,
    net_income       DOUBLE PRECISION,
    equity           DOUBLE PRECISION,
    eps              DOUBLE PRECISION,
    roe              DOUBLE PRECISION,
    debt_ratio       DOUBLE PRECISION,
    has_macro        BOOLEAN NOT NULL DEFAULT FALSE,
    has_breadth      BOOLEAN NOT NULL DEFAULT FALSE,
    has_fundamentals BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, timeframe, date)
);
CREATE INDEX IF NOT EXISTS idx_indicators_ticker_timeframe_date
    ON public.indicators (ticker, timeframe, date DESC);
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS atr_ratio DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS regime_label VARCHAR(20);
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS regime_calm DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS regime_neutral DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS regime_stress DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS revenue DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS net_income DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS equity DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS eps DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS roe DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS debt_ratio DOUBLE PRECISION;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS has_macro BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS has_breadth BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS has_fundamentals BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS public.predictions (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
    model_name          VARCHAR(50) NOT NULL,
    timeframe           VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
    horizon             INT NOT NULL CHECK (horizon > 0),
    asof_date           DATE NOT NULL,
    decision_time       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id              VARCHAR(100) NOT NULL,
    model_ver           VARCHAR(50) NOT NULL,
    signal              VARCHAR(10) NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    forecast_dates      JSONB NOT NULL DEFAULT '[]'::jsonb,
    line_series         JSONB NOT NULL DEFAULT '[]'::jsonb,
    upper_band_series   JSONB NOT NULL DEFAULT '[]'::jsonb,
    lower_band_series   JSONB NOT NULL DEFAULT '[]'::jsonb,
    conservative_series JSONB NOT NULL DEFAULT '[]'::jsonb,
    band_quantile_low   DOUBLE PRECISION,
    band_quantile_high  DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, ticker, model_name, timeframe, horizon, asof_date)
);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_timeframe_asof
    ON public.predictions (ticker, timeframe, asof_date DESC, decision_time DESC);
ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS line_series JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS band_quantile_low DOUBLE PRECISION;
ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS band_quantile_high DOUBLE PRECISION;
ALTER TABLE public.predictions
    DROP CONSTRAINT IF EXISTS predictions_ticker_model_name_timeframe_horizon_asof_date_key;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'predictions_run_unique'
          AND conrelid = 'public.predictions'::regclass
    ) THEN
        ALTER TABLE public.predictions
            ADD CONSTRAINT predictions_run_unique
            UNIQUE (run_id, ticker, model_name, timeframe, horizon, asof_date);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS public.model_runs (
    run_id              VARCHAR(100) PRIMARY KEY,
    wandb_run_id        VARCHAR(100),
    model_name          VARCHAR(50) NOT NULL,
    timeframe           VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
    horizon             INT NOT NULL CHECK (horizon > 0),
    feature_version     VARCHAR(50) NOT NULL DEFAULT 'v3_adjusted_ohlc',
    band_quantile_low   DOUBLE PRECISION,
    band_quantile_high  DOUBLE PRECISION,
    band_mode           VARCHAR(20) NOT NULL DEFAULT 'direct',
    alpha               DOUBLE PRECISION,
    beta                DOUBLE PRECISION,
    huber_delta         DOUBLE PRECISION,
    lambda_line         DOUBLE PRECISION,
    lambda_band         DOUBLE PRECISION,
    lambda_width        DOUBLE PRECISION,
    lambda_cross        DOUBLE PRECISION,
    train_start         DATE,
    train_end           DATE,
    val_metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    test_metrics        JSONB NOT NULL DEFAULT '{}'::jsonb,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkpoint_path     TEXT,
    status              VARCHAR(20) NOT NULL DEFAULT 'completed'
                        CHECK (status IN ('completed', 'failed_nan', 'failed_quality_gate')),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE public.model_runs
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed';
ALTER TABLE public.model_runs
    ADD COLUMN IF NOT EXISTS band_mode VARCHAR(20) NOT NULL DEFAULT 'direct';
ALTER TABLE public.model_runs
    ALTER COLUMN feature_version SET DEFAULT 'v3_adjusted_ohlc';
ALTER TABLE public.model_runs
    DROP CONSTRAINT IF EXISTS model_runs_status_check;
ALTER TABLE public.model_runs
    ADD CONSTRAINT model_runs_status_check
    CHECK (status IN ('completed', 'failed_nan', 'failed_quality_gate'));

CREATE TABLE IF NOT EXISTS public.prediction_evaluations (
    id                      BIGSERIAL PRIMARY KEY,
    run_id                  VARCHAR(100) NOT NULL REFERENCES public.model_runs(run_id),
    ticker                  VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
    timeframe               VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
    asof_date               DATE NOT NULL,
    actual_series           JSONB NOT NULL DEFAULT '[]'::jsonb,
    pinball_loss            DOUBLE PRECISION,
    coverage                DOUBLE PRECISION,
    lower_breach_rate       DOUBLE PRECISION,
    upper_breach_rate       DOUBLE PRECISION,
    avg_band_width          DOUBLE PRECISION,
    normalized_band_width   DOUBLE PRECISION,
    direction_accuracy      DOUBLE PRECISION,
    mae                     DOUBLE PRECISION,
    smape                   DOUBLE PRECISION,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, ticker, timeframe, asof_date)
);
CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_lookup
    ON public.prediction_evaluations (ticker, timeframe, asof_date DESC);
ALTER TABLE public.prediction_evaluations ADD COLUMN IF NOT EXISTS lower_breach_rate DOUBLE PRECISION;
ALTER TABLE public.prediction_evaluations ADD COLUMN IF NOT EXISTS upper_breach_rate DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS public.backtest_results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(100) NOT NULL REFERENCES public.model_runs(run_id),
    strategy_name   VARCHAR(100) NOT NULL,
    timeframe       VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
    return_pct      DOUBLE PRECISION,
    mdd             DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    num_trades      INT,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, strategy_name, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_backtest_results_lookup
    ON public.backtest_results (strategy_name, timeframe, created_at DESC);

CREATE TABLE IF NOT EXISTS public.job_runs (
    run_id        UUID PRIMARY KEY,
    job_name      VARCHAR(100) NOT NULL,
    scope_key     VARCHAR(150) NOT NULL DEFAULT '__all__',
    status        VARCHAR(20) NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    error_text    TEXT,
    meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_job_runs_lookup
    ON public.job_runs (job_name, started_at DESC);

CREATE TABLE IF NOT EXISTS public.sync_state (
    job_name         VARCHAR(100) NOT NULL,
    target_key       VARCHAR(100) NOT NULL DEFAULT '__all__',
    status           VARCHAR(20) NOT NULL DEFAULT 'pending',
    last_cursor_date DATE,
    last_success_at  TIMESTAMPTZ,
    message          TEXT,
    meta             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (job_name, target_key)
);
CREATE INDEX IF NOT EXISTS idx_sync_state_status
    ON public.sync_state (job_name, status, updated_at DESC);
