"""운영 중 필요한 스키마 차이를 실제 DB에 반영한다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")


RUNTIME_SCHEMA_STATEMENTS = [
    """
    ALTER TABLE public.predictions
    ADD COLUMN IF NOT EXISTS line_series JSONB NOT NULL DEFAULT '[]'::jsonb;
    """,
    """
    ALTER TABLE public.predictions
    ADD COLUMN IF NOT EXISTS band_quantile_low DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.predictions
    ADD COLUMN IF NOT EXISTS band_quantile_high DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS regime_label VARCHAR(20);
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS regime_calm DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS regime_neutral DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS regime_stress DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS revenue DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS net_income DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS equity DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS eps DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS roe DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS debt_ratio DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS has_macro BOOLEAN NOT NULL DEFAULT FALSE;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS has_breadth BOOLEAN NOT NULL DEFAULT FALSE;
    """,
    """
    ALTER TABLE public.indicators
    ADD COLUMN IF NOT EXISTS has_fundamentals BOOLEAN NOT NULL DEFAULT FALSE;
    """,
    """
    CREATE TABLE IF NOT EXISTS public.model_runs (
        run_id              VARCHAR(100) PRIMARY KEY,
        wandb_run_id        VARCHAR(100),
        model_name          VARCHAR(50) NOT NULL,
        timeframe           VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
        horizon             INT NOT NULL CHECK (horizon > 0),
        feature_version     VARCHAR(50) NOT NULL DEFAULT 'indicators_v1',
        band_quantile_low   DOUBLE PRECISION,
        band_quantile_high  DOUBLE PRECISION,
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
        created_at          TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public.prediction_evaluations (
        id                      BIGSERIAL PRIMARY KEY,
        run_id                  VARCHAR(100) NOT NULL REFERENCES public.model_runs(run_id),
        ticker                  VARCHAR(20) NOT NULL REFERENCES public.stock_info(ticker),
        timeframe               VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M')),
        asof_date               DATE NOT NULL,
        actual_series           JSONB NOT NULL DEFAULT '[]'::jsonb,
        pinball_loss            DOUBLE PRECISION,
        coverage                DOUBLE PRECISION,
        avg_band_width          DOUBLE PRECISION,
        normalized_band_width   DOUBLE PRECISION,
        direction_accuracy      DOUBLE PRECISION,
        mae                     DOUBLE PRECISION,
        smape                   DOUBLE PRECISION,
        created_at              TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (run_id, ticker, timeframe, asof_date)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_lookup
        ON public.prediction_evaluations (ticker, timeframe, asof_date DESC);
    """,
    """
    ALTER TABLE public.prediction_evaluations
    ADD COLUMN IF NOT EXISTS pinball_loss DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.prediction_evaluations
    ADD COLUMN IF NOT EXISTS mae DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.prediction_evaluations
    ADD COLUMN IF NOT EXISTS smape DOUBLE PRECISION;
    """,
    """
    ALTER TABLE public.model_runs
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed';
    """,
    """
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
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_backtest_results_lookup
        ON public.backtest_results (strategy_name, timeframe, created_at DESC);
    """,
]


def _connect():
    try:
        import psycopg2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("psycopg2가 필요합니다.") from exc

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"DB 연결 정보가 부족합니다: {joined}")

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["DB_NAME"],
        sslmode=os.environ.get("DB_SSLMODE", "require"),
    )


def main() -> None:
    conn = _connect()
    try:
        with conn, conn.cursor() as cursor:
            for statement in RUNTIME_SCHEMA_STATEMENTS:
                cursor.execute(statement)

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('model_runs', 'prediction_evaluations', 'backtest_results')
                ORDER BY table_name
                """
            )
            tables = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'indicators'
                  AND column_name IN (
                      'regime_label', 'regime_calm', 'regime_neutral', 'regime_stress',
                      'revenue', 'net_income', 'equity', 'eps', 'roe', 'debt_ratio',
                      'has_macro', 'has_breadth', 'has_fundamentals'
                  )
                ORDER BY column_name
                """
            )
            columns = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'predictions'
                  AND column_name IN ('line_series', 'band_quantile_low', 'band_quantile_high')
                ORDER BY column_name
                """
            )
            prediction_columns = [row[0] for row in cursor.fetchall()]

        print(f"[OK] ai_tables={tables}")
        print(f"[OK] indicator_regime_columns={columns}")
        print(f"[OK] prediction_writer_columns={prediction_columns}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
