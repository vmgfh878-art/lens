-- ============================================================
-- CP254 serving cutover: product history 테이블 + 서빙 컬럼 보강
-- Created: 2026-06-11
-- Purpose: backend/data/v1/* 서빙 parquet 를 Supabase REST 로 옮길 때
--          필요한 잔여 DDL. 멱등(재실행 안전).
-- 적용 순서: schema.sql → v1_predictions_tables.sql → 본 파일.
-- ============================================================

-- 제품 rolling prediction history (parquet: product_prediction_history_1D.parquet)
-- 컬럼 = 서빙 화이트리스트 10개 (product_prediction_history_svc._HISTORY_COLUMNS).
-- source / model_feature_hash / created_at(parquet) 은 서빙이 읽지 않으므로 미적재.
CREATE TABLE IF NOT EXISTS public.product_prediction_history_1d (
    ticker          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(4) NOT NULL,
    role            VARCHAR(10) NOT NULL CHECK (role IN ('line', 'band')),
    asof_date       DATE NOT NULL,
    display_date    DATE,
    display_horizon SMALLINT NOT NULL,
    run_id          VARCHAR(100) NOT NULL,
    line_value      DOUBLE PRECISION,
    lower_value     DOUBLE PRECISION,
    upper_value     DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    -- 현 parquet 기준 (ticker,timeframe,role,asof_date,display_horizon) 만으로도
    -- 유일(중복 0 실측)하지만, run 교체 이력이 공존할 수 있도록 run_id 포함.
    PRIMARY KEY (ticker, timeframe, role, asof_date, display_horizon, run_id)
);

-- 서빙 조회 패턴: WHERE ticker=? AND timeframe='1D' AND role IN (...) ORDER BY asof_date
CREATE INDEX IF NOT EXISTS idx_product_history_lookup
    ON public.product_prediction_history_1d (ticker, timeframe, role, asof_date DESC);

-- indicators 서빙 보강: IndicatorPoint.volume 을 단일 SELECT 로 충족
-- (없으면 market_repo 가 price_data 에서 2차 쿼리로 merge — egress 2배).
ALTER TABLE public.indicators ADD COLUMN IF NOT EXISTS volume BIGINT;

-- 서빙 /stocks 목록 전용 큐레이션 테이블 (market_stock_info.parquet = 100종목).
-- stock_info 는 FK 보호용 placeholder(~500 ticker, sector NULL)까지 품는 소스
-- 유니버스라 /stocks 목록·sector map 으로 쓰면 placeholder 가 섞인다 (parity 깨짐).
CREATE TABLE IF NOT EXISTS public.serving_stocks (
    ticker      VARCHAR(20) PRIMARY KEY,
    sector      VARCHAR(100),
    industry    VARCHAR(100),
    market_cap  DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 보안 정책 = CP177-S (security_rls_grants.sql) 일관: service_role 만 GRANT,
-- anon/authenticated 차단, RLS enable. 백엔드는 service_role key 로 read 하므로 anon 불필요.
-- (CP254 의 "RLS disable + anon read" 는 정책 위배였다 — 2026-06-18 정정. 서빙 6테이블
--  전부 적용: serving 신규 5 + v1_predictions_tables.sql 의 predictions_* 의 DISABLE 을 덮음.)
DO $$
DECLARE
    t text;
    serving_tables text[] := array[
        'serving_stocks',
        'predictions_line_1d',
        'predictions_band_1d',
        'predictions_band_1w',
        'product_prediction_history_1d'
    ];
BEGIN
    FOREACH t IN ARRAY serving_tables LOOP
        IF to_regclass(format('public.%I', t)) IS NOT NULL THEN
            EXECUTE format('alter table public.%I enable row level security', t);
            EXECUTE format('revoke all privileges on table public.%I from anon, authenticated', t);
            EXECUTE format('grant select, insert, update, delete on table public.%I to service_role', t);
        END IF;
    END LOOP;
END $$;
