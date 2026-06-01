-- Compustat annual controls and CRSP/Compustat link table.
-- Parameters expected from the Python extraction layer:
--   :comp_start_date, :comp_end_date

select
    gvkey,
    datadate,
    fyear,
    at,
    lt,
    seq,
    ceq,
    txditc,
    pstkrv,
    pstkl,
    pstk,
    ib,
    che,
    capx,
    sale
from comp_na_daily_all.funda
where indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
  and datadate between :comp_start_date and :comp_end_date;

select
    gvkey,
    lpermno as permno,
    linkdt,
    linkenddt,
    linktype,
    linkprim
from crsp_a_ccm.ccmxpf_linktable
where lpermno is not null
  and linktype in ('LC', 'LU')
  and linkprim in ('P', 'C');
