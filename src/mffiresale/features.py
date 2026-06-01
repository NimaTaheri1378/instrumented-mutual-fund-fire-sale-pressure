from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ProjectPaths
from .extract import sample_window
from .io import manifest_base, month_range, write_json, write_parquet_atomic


def build_all_features(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    paths.ensure()
    start_month, end_month = sample_window(cfg, sample)
    manifest = manifest_base(paths.root, "features", sample=sample, start_month=start_month, end_month=end_month)
    fund_flows = build_fund_flows(cfg, paths, sample)
    portfolio_flows = build_portfolio_flows(cfg, paths, fund_flows)
    pressure = build_pressure_panel(cfg, paths, portfolio_flows, sample, lag_days=cfg["project"]["main_holdings_lag_days"])
    stock = build_stock_features(cfg, paths, sample)
    comp = build_compustat_features(cfg, paths, sample)
    panel = assemble_model_panel(cfg, paths, pressure, stock, comp, sample)
    manifest.update(
        {
            "fund_flows": _rel(paths, fund_flows),
            "portfolio_flows": _rel(paths, portfolio_flows),
            "pressure_panel": _rel(paths, pressure),
            "stock_features": _rel(paths, stock),
            "compustat_features": _rel(paths, comp),
            "model_panel": _rel(paths, panel),
        }
    )
    write_json(paths.manifests / f"features_{sample}.json", manifest)
    return manifest


def _rel(paths: ProjectPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def _month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")


def _winsor(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    qlo, qhi = s.quantile([lo, hi])
    return s.clip(qlo, qhi)


def _active_interval_join(
    left: pd.DataFrame,
    intervals: pd.DataFrame,
    key: str,
    date_col: str,
    start_col: str = "begdt",
    end_col: str = "enddt",
) -> pd.DataFrame:
    intervals = intervals.copy()
    intervals["_interval_start"] = pd.to_datetime(intervals[start_col]).fillna(pd.Timestamp("1900-01-01"))
    intervals["_interval_end"] = pd.to_datetime(intervals[end_col]).fillna(pd.Timestamp("2100-12-31"))
    intervals = intervals.drop(columns=[c for c in [start_col, end_col] if c in intervals.columns])
    merged = left.merge(intervals, on=key, how="left", suffixes=("", "_interval"))
    date = pd.to_datetime(merged[date_col])
    ok = (date >= merged["_interval_start"]) & (date <= merged["_interval_end"])
    return merged.loc[ok].drop(columns=["_interval_start", "_interval_end"]).copy()


def build_fund_flows(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> Path:
    out = paths.interim / "mutual_funds" / f"fund_flows_{sample}.parquet"
    if out.exists():
        return out
    monthly = pd.read_parquet(paths.raw / "mutual_funds" / "monthly_tna_returns.parquet")
    monthly["caldt"] = _month_end(monthly["caldt"])
    monthly = monthly.sort_values(["crsp_fundno", "caldt"])
    monthly["mtna"] = pd.to_numeric(monthly["mtna"], errors="coerce")
    monthly["mret"] = pd.to_numeric(monthly["mret"], errors="coerce")
    monthly.loc[monthly["mtna"] <= 0, "mtna"] = np.nan
    g = monthly.groupby("crsp_fundno", sort=False)
    monthly["mtna_lag1"] = g["mtna"].shift(1)
    monthly["mret_lag1"] = g["mret"].shift(1)
    denom = monthly["mtna_lag1"] * (1.0 + monthly["mret"].fillna(0.0))
    monthly["flow"] = (monthly["mtna"] - denom) / denom
    monthly.loc[~np.isfinite(monthly["flow"]), "flow"] = np.nan
    monthly["flow_capped"] = monthly["flow"].clip(-0.90, 3.00)
    monthly["flow_lag1"] = g["flow_capped"].shift(1)
    monthly["flow_lag3_mean"] = (
        g["flow_capped"]
        .apply(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
        .reset_index(level=0, drop=True)
    )
    monthly["mret_lag1"] = g["mret"].shift(1)
    monthly["log_mtna_lag1"] = np.log1p(monthly["mtna_lag1"])

    style = pd.read_parquet(paths.raw / "mutual_funds" / "fund_style.parquet")
    style_cols = ["crsp_fundno", "begdt", "enddt", "crsp_obj_cd", "lipper_class", "policy"]
    style = style[[c for c in style_cols if c in style.columns]].copy()
    monthly = _active_interval_join(monthly, style, "crsp_fundno", "caldt") if len(style) else monthly
    for col in ["lipper_class", "crsp_obj_cd", "policy"]:
        if col not in monthly:
            monthly[col] = np.nan
    monthly["style_key"] = (
        monthly["lipper_class"].fillna(monthly["crsp_obj_cd"]).fillna(monthly["policy"]).fillna("UNKNOWN").astype(str)
    )

    port = pd.read_parquet(paths.raw / "mutual_funds" / "portnomap.parquet")
    keep = ["crsp_fundno", "begdt", "enddt", "crsp_portno", "mgmt_cd", "index_fund_flag", "et_flag", "m_fund"]
    port = port[[c for c in keep if c in port.columns]].copy()
    monthly = _active_interval_join(monthly, port, "crsp_fundno", "caldt") if len(port) else monthly
    monthly["family_key"] = monthly.get("mgmt_cd", pd.Series(index=monthly.index, dtype="object")).fillna("UNKNOWN").astype(str)

    monthly = _add_leave_one_out_flow(monthly, "style_key", "style_flow_loo")
    monthly = _add_leave_one_out_flow(monthly, "family_key", "family_flow_loo")
    for col in ["style_flow_loo", "family_flow_loo"]:
        monthly[col + "_lag1"] = monthly.groupby("crsp_fundno", sort=False)[col].shift(1)

    monthly["pred_flow_exo"] = _walk_forward_predict_flow(monthly)
    write_parquet_atomic(monthly, out)
    return out


def _add_leave_one_out_flow(df: pd.DataFrame, group_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    weight = df["mtna_lag1"].where(df["mtna_lag1"].gt(0), np.nan)
    val = df["flow_capped"] * weight
    sums = df.assign(_w=weight, _v=val).groupby(["caldt", group_col], dropna=False)[["_w", "_v"]].transform("sum")
    den = sums["_w"] - weight
    num = sums["_v"] - val
    df[out_col] = num / den
    df.loc[~np.isfinite(df[out_col]), out_col] = np.nan
    return df


def _walk_forward_predict_flow(df: pd.DataFrame) -> pd.Series:
    feature_cols = [
        "flow_lag1",
        "flow_lag3_mean",
        "mret_lag1",
        "log_mtna_lag1",
        "style_flow_loo_lag1",
        "family_flow_loo_lag1",
    ]
    work = df[["crsp_fundno", "caldt", "flow_capped", *feature_cols]].copy()
    work["year"] = work["caldt"].dt.year
    pred = pd.Series(np.nan, index=work.index, dtype="float64")
    fallback = work["style_flow_loo_lag1"].fillna(work["family_flow_loo_lag1"]).fillna(work["flow_lag1"])
    for year in sorted(work["year"].dropna().unique()):
        train = (work["year"] < year) & work["flow_capped"].notna()
        test = work["year"] == year
        if train.sum() < 10_000 or test.sum() == 0:
            pred.loc[test] = pd.to_numeric(fallback.loc[test], errors="coerce").astype("float64")
            continue
        x_train = work.loc[train, feature_cols].replace([np.inf, -np.inf], np.nan)
        med = x_train.median()
        x_train = x_train.fillna(med)
        y_train = work.loc[train, "flow_capped"]
        x_test = work.loc[test, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        model = Ridge(alpha=10.0, random_state=0)
        model.fit(x_train, y_train)
        pred.loc[test] = model.predict(x_test)
    return pred.clip(-0.90, 3.00)


def build_portfolio_flows(cfg: dict[str, Any], paths: ProjectPaths, fund_flows_path: Path) -> Path:
    sample_suffix = fund_flows_path.stem.replace("fund_flows", "") or "_unknown"
    out = paths.interim / "mutual_funds" / f"portfolio_flows{sample_suffix}.parquet"
    if out.exists():
        return out
    df = pd.read_parquet(fund_flows_path)
    df = df[df["crsp_portno"].notna()].copy()
    df["flow_weight"] = df["mtna_lag1"].where(df["mtna_lag1"].gt(0), np.nan)
    for col in ["flow_capped", "pred_flow_exo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_flow_v"] = df["flow_capped"] * df["flow_weight"]
    df["_pred_v"] = df["pred_flow_exo"] * df["flow_weight"]
    grouped = df.groupby(["crsp_portno", "caldt"], dropna=False).agg(
        flow_weight=("_flow_v", lambda x: np.nan),
        flow_num=("_flow_v", "sum"),
        pred_num=("_pred_v", "sum"),
        weight_sum=("flow_weight", "sum"),
        fund_count=("crsp_fundno", "nunique"),
    )
    grouped = grouped.reset_index()
    grouped["portfolio_flow"] = grouped["flow_num"] / grouped["weight_sum"]
    grouped["portfolio_pred_flow"] = grouped["pred_num"] / grouped["weight_sum"]
    grouped = grouped.drop(columns=["flow_weight", "flow_num", "pred_num"])
    grouped.loc[~np.isfinite(grouped["portfolio_flow"]), "portfolio_flow"] = np.nan
    grouped.loc[~np.isfinite(grouped["portfolio_pred_flow"]), "portfolio_pred_flow"] = np.nan
    write_parquet_atomic(grouped, out)
    return out


def build_pressure_panel(
    cfg: dict[str, Any],
    paths: ProjectPaths,
    portfolio_flows_path: Path,
    sample: str,
    lag_days: int,
    output_sample: str | None = None,
    holdings_sample: str | None = None,
) -> Path:
    out_tag = output_sample or sample
    holdings_tag = holdings_sample or sample
    out = paths.interim / "pressure" / f"pressure_lag{lag_days}_{out_tag}.parquet"
    if out.exists():
        return out
    start_month, end_month = sample_window(cfg, sample)
    months = month_range(start_month, end_month)
    flows = pd.read_parquet(portfolio_flows_path)
    flows["caldt"] = _month_end(flows["caldt"])
    shards = []
    cache: dict[int, pd.DataFrame] = {}
    for month_end in months:
        report_cutoff = month_end - pd.Timedelta(days=lag_days)
        needed_years = range(max(cfg["project"]["holdings_start_year"], report_cutoff.year - 2), report_cutoff.year + 1)
        hold = pd.concat([_load_holdings_year(paths, y, cache, holdings_tag) for y in needed_years], ignore_index=True)
        for cached_year in list(cache):
            if cached_year not in needed_years:
                cache.pop(cached_year, None)
        if hold.empty:
            continue
        hold = hold[hold["report_dt"].le(report_cutoff)].copy()
        if hold.empty:
            continue
        recent = hold["report_dt"].ge(report_cutoff - pd.Timedelta(days=370))
        hold = hold.loc[recent]
        latest = hold.groupby("crsp_portno", sort=False)["report_dt"].transform("max")
        hold = hold.loc[hold["report_dt"].eq(latest)].copy()
        flow_m = flows.loc[flows["caldt"].eq(month_end), ["crsp_portno", "portfolio_flow", "portfolio_pred_flow"]]
        merged = hold.merge(flow_m, on="crsp_portno", how="inner")
        if merged.empty:
            continue
        merged["pressure_real_component"] = merged["holding_weight"] * merged["portfolio_flow"]
        merged["pressure_pred_component"] = merged["holding_weight"] * merged["portfolio_pred_flow"]
        merged["weight_sq"] = merged["holding_weight"] ** 2
        agg = merged.groupby("permno", sort=False).agg(
            pressure_real=("pressure_real_component", "sum"),
            pressure_pred=("pressure_pred_component", "sum"),
            owner_count=("crsp_portno", "nunique"),
            ownership_hhi=("weight_sq", "sum"),
            holding_weight_sum=("holding_weight", "sum"),
        )
        agg = agg.reset_index()
        agg["date"] = month_end
        shards.append(agg)
    panel = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
    if len(panel):
        for col in ["pressure_real", "pressure_pred", "ownership_hhi", "holding_weight_sum"]:
            panel[col + "_z"] = panel.groupby("date")[col].transform(lambda x: _zscore(_winsor(x)))
    write_parquet_atomic(panel, out)
    return out


def _load_holdings_year(paths: ProjectPaths, year: int, cache: dict[int, pd.DataFrame], sample: str) -> pd.DataFrame:
    if year in cache:
        return cache[year]
    path = paths.raw / "mutual_funds" / f"holdings_{sample}" / f"holdings_{year}.parquet"
    if not path.exists():
        cache[year] = pd.DataFrame()
        return cache[year]
    h = pd.read_parquet(path)
    h["report_dt"] = pd.to_datetime(h["report_dt"])
    h = h[h["permno"].notna() & h["crsp_portno"].notna()].copy()
    h["percent_tna"] = pd.to_numeric(h["percent_tna"], errors="coerce")
    h["market_val"] = pd.to_numeric(h["market_val"], errors="coerce")
    h["holding_weight"] = h["percent_tna"] / 100.0
    missing_weight = h["holding_weight"].isna() | h["holding_weight"].le(0)
    if missing_weight.any():
        totals = h.groupby(["crsp_portno", "report_dt"])["market_val"].transform("sum")
        h.loc[missing_weight, "holding_weight"] = h.loc[missing_weight, "market_val"] / totals.loc[missing_weight]
    h = h[h["holding_weight"].notna() & h["holding_weight"].gt(0)]
    h = h[["crsp_portno", "report_dt", "permno", "permco", "holding_weight"]]
    cache[year] = h
    return h


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def build_stock_features(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> Path:
    out = paths.interim / "stocks" / f"stock_features_{sample}.parquet"
    if out.exists():
        return out
    start_month, end_month = sample_window(cfg, sample)
    frames = []
    for year in range(pd.Period(start_month, "M").year, pd.Period(end_month, "M").year + 1):
        path = paths.raw / "crsp" / "monthly" / f"crsp_monthly_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        write_parquet_atomic(df, out)
        return out
    df["date"] = _month_end(df["date"])
    for col in ["ret", "retx", "dlret", "prc", "vol", "shrout"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ret_adj"] = (1.0 + df["ret"].fillna(0.0)) * (1.0 + df["dlret"].fillna(0.0)) - 1.0
    df.loc[df["ret"].isna() & df["dlret"].isna(), "ret_adj"] = np.nan
    df["mktcap_musd"] = (df["prc"].abs() * df["shrout"]) / 1000.0
    df["dollar_vol_musd"] = (df["prc"].abs() * df["vol"]) / 1_000_000.0
    df = df.sort_values(["permno", "date"])
    g = df.groupby("permno", sort=False)
    df["mktcap_lag1"] = g["mktcap_musd"].shift(1)
    df["price_lag1"] = g["prc"].shift(1).abs()
    df["dollar_vol_lag1"] = g["dollar_vol_musd"].shift(1)
    df["turnover"] = df["vol"] / (df["shrout"] * 1000.0)
    df["turnover_lag1"] = g["turnover"].shift(1)
    df["reversal_1m"] = g["ret_adj"].shift(1)
    logret = np.log1p(df["ret_adj"].clip(lower=-0.99))
    df["_logret"] = logret
    df["momentum_12_2"] = (
        g["_logret"]
        .apply(lambda x: x.shift(2).rolling(11, min_periods=8).sum())
        .reset_index(level=0, drop=True)
        .pipe(np.expm1)
    )
    df["volatility_12m"] = (
        g["ret_adj"]
        .apply(lambda x: x.shift(1).rolling(12, min_periods=8).std())
        .reset_index(level=0, drop=True)
    )
    df["next_ret"] = g["ret_adj"].shift(-1)
    ff = _load_five_factors(paths)
    df = df.merge(ff[["date", "rf", "mktrf"]], on="date", how="left")
    df["excess_ret"] = df["ret_adj"] - df["rf"]
    g = df.groupby("permno", sort=False)
    df["next_excess_ret"] = g["excess_ret"].shift(-1)
    df = _add_beta_features(df)
    df["implementation_universe"] = (
        df["price_lag1"].ge(cfg["universe"]["implementation_price_floor"])
        & df["mktcap_lag1"].ge(cfg["universe"]["implementation_mktcap_floor_musd"])
        & df["dollar_vol_lag1"].ge(cfg["universe"]["implementation_adv_floor_musd"])
    )
    df = df.drop(columns=["_logret"])
    write_parquet_atomic(df, out)
    return out


def _load_five_factors(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.raw / "factors" / "fivefactors_monthly.parquet"
    ff = pd.read_parquet(path)
    ff["date"] = _month_end(ff["date"])
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        if col in ff:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            if ff[col].abs().median(skipna=True) > 1:
                ff[col] = ff[col] / 100.0
    return ff


def _add_beta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["permno", "date"]).copy()
    cov = (
        df.groupby("permno", sort=False)
        .apply(lambda x: x["excess_ret"].shift(1).rolling(36, min_periods=24).cov(x["mktrf"].shift(1)))
        .reset_index(level=0, drop=True)
    )
    var = df.groupby("permno", sort=False)["mktrf"].apply(lambda x: x.shift(1).rolling(36, min_periods=24).var()).reset_index(level=0, drop=True)
    df["beta_36m"] = cov / var
    df["idio_vol_36m"] = df.groupby("permno", sort=False)["excess_ret"].apply(lambda x: x.shift(1).rolling(36, min_periods=24).std()).reset_index(level=0, drop=True)
    return df


def build_compustat_features(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> Path:
    out = paths.interim / "compustat" / f"compustat_features_{sample}.parquet"
    if out.exists():
        return out
    funda = pd.read_parquet(paths.raw / "compustat" / f"funda_{sample}.parquet")
    links = pd.read_parquet(paths.raw / "compustat" / f"ccm_linktable_{sample}.parquet")
    if funda.empty or links.empty:
        write_parquet_atomic(pd.DataFrame(), out)
        return out
    funda["datadate"] = pd.to_datetime(funda["datadate"])
    for col in ["seq", "ceq", "txditc", "pstk", "pstkrv", "pstkl", "at", "sale", "ib", "lt", "che", "capx", "csho", "prcc_f"]:
        if col in funda:
            funda[col] = pd.to_numeric(funda[col], errors="coerce")
    pref = funda["pstkrv"].fillna(funda["pstkl"]).fillna(funda["pstk"]).fillna(0.0)
    be_base = funda["seq"].fillna(funda["ceq"])
    funda["book_equity"] = be_base + funda["txditc"].fillna(0.0) - pref
    funda["profitability"] = funda["ib"] / funda["at"]
    funda["leverage"] = funda["lt"] / funda["at"]
    funda["cash_assets"] = funda["che"] / funda["at"]
    funda["capx_assets"] = funda["capx"] / funda["at"]
    funda["sales_assets"] = funda["sale"] / funda["at"]
    funda = funda.sort_values(["gvkey", "datadate"])
    funda["asset_growth"] = funda.groupby("gvkey", sort=False)["at"].pct_change(fill_method=None)
    funda["available_date"] = (funda["datadate"] + pd.DateOffset(months=6)).dt.to_period("M").dt.to_timestamp("M")
    links["linkdt"] = pd.to_datetime(links["linkdt"]).fillna(pd.Timestamp("1900-01-01"))
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(pd.Timestamp("2100-12-31"))
    merged = funda.merge(links, on="gvkey", how="inner")
    ok = (merged["datadate"] >= merged["linkdt"]) & (merged["datadate"] <= merged["linkenddt"])
    merged = merged.loc[ok].copy()
    keep = [
        "permno",
        "available_date",
        "book_equity",
        "profitability",
        "leverage",
        "cash_assets",
        "capx_assets",
        "sales_assets",
        "asset_growth",
    ]
    merged = merged[keep].dropna(subset=["permno", "available_date"])
    merged["permno"] = pd.to_numeric(merged["permno"], errors="coerce")
    merged = merged.dropna(subset=["permno"])
    merged["permno"] = merged["permno"].astype("int64")
    write_parquet_atomic(merged, out)
    return out


def assemble_model_panel(
    cfg: dict[str, Any],
    paths: ProjectPaths,
    pressure_path: Path,
    stock_path: Path,
    comp_path: Path,
    sample: str,
) -> Path:
    out = paths.processed / f"model_panel_{sample}.parquet"
    if out.exists():
        return out
    stock = pd.read_parquet(stock_path)
    pressure = pd.read_parquet(pressure_path)
    comp = pd.read_parquet(comp_path)
    if stock.empty or pressure.empty:
        write_parquet_atomic(pd.DataFrame(), out)
        return out
    stock["date"] = _month_end(stock["date"])
    pressure["date"] = _month_end(pressure["date"])
    stock["permno"] = pd.to_numeric(stock["permno"], errors="coerce")
    pressure["permno"] = pd.to_numeric(pressure["permno"], errors="coerce")
    stock = stock.dropna(subset=["permno"])
    pressure = pressure.dropna(subset=["permno"])
    stock["permno"] = stock["permno"].astype("int64")
    pressure["permno"] = pressure["permno"].astype("int64")
    panel = stock.merge(pressure, on=["permno", "date"], how="left")
    for col in ["pressure_real", "pressure_pred", "pressure_real_z", "pressure_pred_z", "owner_count", "ownership_hhi"]:
        if col in panel:
            panel[col] = panel[col].fillna(0.0)
    if not comp.empty:
        comp["available_date"] = _month_end(comp["available_date"])
        comp["permno"] = pd.to_numeric(comp["permno"], errors="coerce")
        comp = comp.dropna(subset=["permno"])
        comp["permno"] = comp["permno"].astype("int64")
        panel = panel.sort_values(["date", "permno"])
        comp = comp.sort_values(["available_date", "permno"])
        panel = pd.merge_asof(
            panel,
            comp,
            left_on="date",
            right_on="available_date",
            by="permno",
            direction="backward",
            tolerance=pd.Timedelta(days=730),
        )
    panel["book_to_market"] = panel["book_equity"] / (panel["mktcap_lag1"] * 1_000_000.0)
    panel["log_mktcap"] = np.log(panel["mktcap_lag1"].where(panel["mktcap_lag1"].gt(0)))
    panel["pressure_scaled_dvol"] = panel["pressure_pred"] / panel["dollar_vol_lag1"].replace(0, np.nan)
    panel["pressure_x_illiquidity"] = panel["pressure_pred_z"] * (1.0 / panel["dollar_vol_lag1"].replace(0, np.nan))
    panel["score_pressure_reversal"] = -panel["pressure_pred_z"]
    write_parquet_atomic(panel, out)
    return out
