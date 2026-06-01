-- CRSP monthly U.S. common-equity panel.
-- Parameters expected from the Python extraction layer:
--   :start_month, :end_month

select
    m.permno,
    m.date,
    m.ret,
    m.prc,
    m.shrout,
    m.vol,
    n.shrcd,
    n.exchcd,
    n.siccd,
    d.dlret
from crsp_a_stock.msf as m
left join crsp_a_stock.msenames as n
    on m.permno = n.permno
   and m.date between n.namedt and coalesce(n.nameendt, '9999-12-31')
left join crsp_a_stock.msedelist as d
    on m.permno = d.permno
   and date_trunc('month', m.date) = date_trunc('month', d.dlstdt)
where m.date between :start_month and :end_month
  and n.shrcd in (10, 11)
  and n.exchcd in (1, 2, 3);
