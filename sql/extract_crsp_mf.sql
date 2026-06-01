-- CRSP mutual-fund inputs for instrumented fire-sale pressure.
-- Parameters expected from the Python extraction layer:
--   :start_month, :end_month
-- Keep all date filters server-side and cache the outputs as Parquet shards.

-- Monthly TNA and returns.
select
    coalesce(t.crsp_fundno, r.crsp_fundno) as crsp_fundno,
    coalesce(t.caldt, r.caldt) as caldt,
    t.mtna,
    r.mret
from crsp_q_mutualfunds.monthly_tna as t
full outer join crsp_q_mutualfunds.monthly_returns as r
    on t.crsp_fundno = r.crsp_fundno
   and t.caldt = r.caldt
where coalesce(t.caldt, r.caldt) between :start_month and :end_month;

-- Holdings shards are pulled year-by-year in Python to keep each query restartable:
-- select crsp_portno, report_dt, permno, percent_tna, shares, market_val
-- from crsp_q_mutualfunds.holdings
-- where report_dt between :year_start and :year_end;

-- Fund metadata used for active/index/ETF filters and leave-one-out family flows:
-- select crsp_fundno, crsp_portno, begdt, enddt, mgmt_cd, index_fund_flag, et_flag, m_fund
-- from crsp_q_mutualfunds.portnomap;
