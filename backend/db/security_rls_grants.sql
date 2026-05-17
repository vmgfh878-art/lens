-- CP177-S Supabase RLS / GRANT 보안 정리 초안
-- 적용 전 반드시 docs/cp177_supabase_rls_grant_plan.md를 검토하고 승인 후 실행한다.
-- 데이터 삭제, 테이블 삭제, 스키마 변경은 하지 않는다.

begin;

-- 현재 코드가 직접 접근하는 public 테이블 목록이다.
-- service_role은 Supabase/Postgres에서 BYPASSRLS=true로 확인되어 RLS policy 없이도 접근 가능하다.
-- anon/authenticated는 프론트가 Supabase를 직접 호출하지 않는 현재 구조에서 직접 Data API 권한이 필요 없다.
do $$
declare
    table_names text[] := array[
        'backtest_results',
        'company_fundamentals',
        'indicators',
        'job_runs',
        'macroeconomic_indicators',
        'market_breadth',
        'model_runs',
        'prediction_evaluations',
        'predictions',
        'price_data',
        'sector_returns',
        'stock_info',
        'sync_state'
    ];
    table_name text;
begin
    foreach table_name in array table_names loop
        if to_regclass(format('public.%I', table_name)) is not null then
            execute format('alter table public.%I enable row level security', table_name);
            execute format('revoke all privileges on table public.%I from anon, authenticated', table_name);
            execute format('grant select, insert, update, delete on table public.%I to service_role', table_name);
        end if;
    end loop;
end;
$$;

-- id/serial 시퀀스 직접 조작 권한도 anon/authenticated에서 제거한다.
-- service_role upsert/insert 경로는 유지한다.
revoke all privileges on all sequences in schema public from anon, authenticated;
grant usage, select, update on all sequences in schema public to service_role;

-- Supabase 2026-05-30/2026-10-30 기본 GRANT 변경에 맞춰 future object도 명시 GRANT 방식으로 고정한다.
-- 이후 public 테이블/시퀀스를 추가하는 migration은 필요한 role에 GRANT를 함께 선언해야 한다.
alter default privileges for role postgres in schema public
    revoke select, insert, update, delete on tables from anon, authenticated, service_role;

alter default privileges for role postgres in schema public
    revoke usage, select on sequences from anon, authenticated, service_role;

alter default privileges for role postgres in schema public
    revoke execute on functions from anon, authenticated, service_role;

alter default privileges for role postgres in schema public
    revoke execute on functions from public;

commit;
