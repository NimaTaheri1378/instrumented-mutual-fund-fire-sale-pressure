from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectPaths
from .io import manifest_base, month_range, write_json, write_parquet_atomic, year_range_from_months
from .wrds_access import connect, require_pgpass, safe_raw_sql


def sample_window(cfg: dict[str, Any], sample: str) -> tuple[str, str]:
    project = cfg["project"]
    if sample == "smoke":
        return project["smoke_start_month"], project["smoke_end_month"]
    return project["start_month"], project["end_month"]


def extract_all(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    paths.ensure()
    start_month, end_month = sample_window(cfg, sample)
    require_pgpass()
    manifest = manifest_base(paths.root, "extract", sample=sample, start_month=start_month, end_month=end_month, shards=[])
    with connect() as db:
        manifest["wrds_smoke"] = _sql_smoke(db)
        manifest["mutual_fund_monthly"] = extract_mutual_fund_monthly(db, cfg, paths, start_month, end_month)
        manifest["mutual_fund_metadata"] = extract_mutual_fund_metadata(db, cfg, paths)
        manifest["holdings"] = extract_holdings_shards(db, cfg, paths, start_month, end_month, sample)
        manifest["crsp_monthly"] = extract_crsp_monthly_shards(db, cfg, paths, start_month, end_month, sample)
        manifest["compustat"] = extract_compustat_shards(db, cfg, paths, start_month, end_month, sample)
        manifest["factors"] = extract_factors(db, cfg, paths, start_month, end_month)
    write_json(paths.manifests / f"extract_{sample}.json", manifest)
    return manifest


def _sql_smoke(db: Any) -> dict[str, Any]:
    df = safe_raw_sql(db, "select current_date as current_date")
    return {"ok": True, "current_date": str(df.iloc[0, 0])}


def extract_mutual_fund_monthly(db: Any, cfg: dict[str, Any], paths: ProjectPaths, start_month: str, end_month: str) -> dict[str, Any]:
    lib = cfg["wrds"]["libraries"]["mf"]
    tna = cfg["wrds"]["tables"]["mf_monthly_tna"]
    ret = cfg["wrds"]["tables"]["mf_monthly_returns"]
    start = (pd.Period(start_month, freq="M") - 13).to_timestamp("M").date()
    end = pd.Period(end_month, freq="M").to_timestamp("M").date()
    sql = f"""
        select coalesce(t.crsp_fundno, r.crsp_fundno) as crsp_fundno,
               coalesce(t.caldt, r.caldt) as caldt,
               t.mtna,
               r.mret
        from {lib}.{tna} as t
        full outer join {lib}.{ret} as r
          on t.crsp_fundno = r.crsp_fundno and t.caldt = r.caldt
        where coalesce(t.caldt, r.caldt) between '{start}' and '{end}'
    """
    df = safe_raw_sql(db, sql, date_cols=["caldt"])
    out = paths.raw / "mutual_funds" / "monthly_tna_returns.parquet"
    write_parquet_atomic(df, out)
    return {"path": str(out.relative_to(paths.root)), "rows": int(len(df)), "start": str(start), "end": str(end)}


def extract_mutual_fund_metadata(db: Any, cfg: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    lib = cfg["wrds"]["libraries"]["mf"]
    port = cfg["wrds"]["tables"]["mf_portno_map"]
    style = cfg["wrds"]["tables"]["mf_style"]
    hdr = cfg["wrds"]["tables"]["mf_header"]
    outputs = {}
    for name, table in [("portnomap", port), ("fund_style", style), ("fund_hdr", hdr)]:
        df = safe_raw_sql(db, f"select * from {lib}.{table}", date_cols=["begdt", "enddt", "mgr_dt", "first_offer_dt"])
        out = paths.raw / "mutual_funds" / f"{name}.parquet"
        write_parquet_atomic(df, out)
        outputs[name] = {"path": str(out.relative_to(paths.root)), "rows": int(len(df))}
    return outputs


def extract_holdings_shards(db: Any, cfg: dict[str, Any], paths: ProjectPaths, start_month: str, end_month: str, sample: str) -> list[dict[str, Any]]:
    lib = cfg["wrds"]["libraries"]["mf"]
    table = cfg["wrds"]["tables"]["mf_holdings"]
    start_year = cfg["project"]["holdings_start_year"] if sample == "full" else pd.Period(start_month, freq="M").year - 1
    end_year = pd.Period(end_month, freq="M").year
    shards = []
    for year in range(start_year, end_year + 1):
        out = paths.raw / "mutual_funds" / f"holdings_{sample}" / f"holdings_{year}.parquet"
        if out.exists():
            shards.append({"year": year, "path": str(out.relative_to(paths.root)), "status": "exists"})
            continue
        sql = f"""
            select crsp_portno, report_dt, eff_dt, percent_tna, nbr_shares, market_val,
                   permno, permco, cusip, ticker
            from {lib}.{table}
            where report_dt between '{year}-01-01' and '{year}-12-31'
              and permno is not null
              and (percent_tna is not null or market_val is not null or nbr_shares is not null)
        """
        df = safe_raw_sql(db, sql, date_cols=["report_dt", "eff_dt"])
        if sample == "smoke":
            df = df.head(250_000)
        write_parquet_atomic(df, out)
        shards.append({"year": year, "path": str(out.relative_to(paths.root)), "rows": int(len(df)), "status": "written"})
    return shards


def extract_crsp_monthly_shards(db: Any, cfg: dict[str, Any], paths: ProjectPaths, start_month: str, end_month: str, sample: str) -> list[dict[str, Any]]:
    lib = cfg["wrds"]["libraries"]["stock"]
    msf = cfg["wrds"]["tables"]["crsp_monthly_stock"]
    names = cfg["wrds"]["tables"]["crsp_monthly_names"]
    delist = cfg["wrds"]["tables"]["crsp_monthly_delist"]
    share_codes = ",".join(str(x) for x in cfg["universe"]["share_codes"])
    exchanges = ",".join(str(x) for x in cfg["universe"]["exchanges"])
    shards = []
    for year in year_range_from_months(start_month, end_month):
        out = paths.raw / "crsp" / "monthly" / f"crsp_monthly_{year}.parquet"
        if out.exists():
            shards.append({"year": year, "path": str(out.relative_to(paths.root)), "status": "exists"})
            continue
        sql = f"""
            select m.permno, m.permco, m.date, m.prc, m.ret, m.retx, m.vol, m.shrout,
                   m.bid, m.ask, m.spread, m.hexcd,
                   n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam, n.naics,
                   d.dlret, d.dlstcd, d.dlstdt
            from {lib}.{msf} as m
            left join {lib}.{names} as n
              on m.permno = n.permno
             and m.date between n.namedt and n.nameendt
            left join {lib}.{delist} as d
              on m.permno = d.permno
             and date_trunc('month', m.date)::date = date_trunc('month', d.dlstdt)::date
            where m.date between '{year}-01-01' and '{year}-12-31'
              and n.shrcd in ({share_codes})
              and n.exchcd in ({exchanges})
        """
        df = safe_raw_sql(db, sql, date_cols=["date", "dlstdt"])
        write_parquet_atomic(df, out)
        shards.append({"year": year, "path": str(out.relative_to(paths.root)), "rows": int(len(df)), "status": "written"})
    return shards


def extract_compustat_shards(db: Any, cfg: dict[str, Any], paths: ProjectPaths, start_month: str, end_month: str, sample: str) -> dict[str, Any]:
    comp_lib = cfg["wrds"]["libraries"]["comp"]
    ccm_lib = cfg["wrds"]["libraries"]["ccm"]
    funda = cfg["wrds"]["tables"]["comp_funda"]
    link = cfg["wrds"]["tables"]["ccm_linktable"]
    start_year = pd.Period(start_month, freq="M").year - 3
    end_year = pd.Period(end_month, freq="M").year
    comp_out = paths.raw / "compustat" / f"funda_{sample}.parquet"
    link_out = paths.raw / "compustat" / f"ccm_linktable_{sample}.parquet"
    if not comp_out.exists():
        sql = f"""
            select gvkey, datadate, fyear, fyr, indfmt, consol, popsrc, datafmt,
                   at, sale, ceq, seq, txditc, pstk, pstkrv, pstkl, lt, dltt, dlc,
                   act, lct, che, oancf, ib, dp, capx, xrd, prcc_f, csho
            from {comp_lib}.{funda}
            where datadate between '{start_year}-01-01' and '{end_year}-12-31'
              and indfmt = 'INDL' and consol = 'C' and popsrc = 'D' and datafmt = 'STD'
        """
        comp = safe_raw_sql(db, sql, date_cols=["datadate"])
        write_parquet_atomic(comp, comp_out)
    if not link_out.exists():
        links = safe_raw_sql(
            db,
            f"""
            select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim,
                   linkdt, linkenddt
            from {ccm_lib}.{link}
            where lpermno is not null
              and linktype in ('LC','LU','LS')
              and linkprim in ('P','C')
            """,
            date_cols=["linkdt", "linkenddt"],
        )
        write_parquet_atomic(links, link_out)
    return {
        "funda_path": str(comp_out.relative_to(paths.root)),
        "link_path": str(link_out.relative_to(paths.root)),
    }


def extract_factors(db: Any, cfg: dict[str, Any], paths: ProjectPaths, start_month: str, end_month: str) -> dict[str, Any]:
    lib = cfg["wrds"]["libraries"]["ff"]
    five = cfg["wrds"]["tables"]["ff_fivefactors_monthly"]
    ff3 = cfg["wrds"]["tables"]["ff_factors_monthly"]
    start = pd.Period(start_month, freq="M").to_timestamp("M").date()
    end = pd.Period(end_month, freq="M").to_timestamp("M").date()
    outputs = {}
    for name, table in [("fivefactors_monthly", five), ("factors_monthly", ff3)]:
        sql = f"select * from {lib}.{table} where date between '{start}' and '{end}'"
        df = safe_raw_sql(db, sql, date_cols=["date", "dateff"])
        out = paths.raw / "factors" / f"{name}.parquet"
        write_parquet_atomic(df, out)
        outputs[name] = {"path": str(out.relative_to(paths.root)), "rows": int(len(df))}
    return outputs
