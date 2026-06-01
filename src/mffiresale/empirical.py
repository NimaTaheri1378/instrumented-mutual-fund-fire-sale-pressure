from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ProjectPaths
from .io import manifest_base, write_json
from .viz import save_empirical_figures


SCORE_SPECS = {
    "instrumented": "score_pressure_reversal",
    "realized": "score_realized_reversal",
    "stale_12m": "score_stale_reversal_12m",
}


def run_empirical_suite(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "full") -> dict[str, Any]:
    paths.ensure()
    panel_path = paths.processed / f"model_panel_{sample}.parquet"
    panel = pd.read_parquet(panel_path)
    if panel.empty:
        manifest = manifest_base(paths.root, "empirical", sample=sample, status="empty_panel")
        write_json(paths.manifests / f"empirical_{sample}.json", manifest)
        return manifest
    panel = _prepare_panel(panel)
    coverage = _coverage_by_year(panel, paths, sample)
    deciles = _decile_sorts(panel, cfg, paths, sample)
    raw_vs_inst = _raw_vs_instrumented(panel, cfg, paths, sample)
    double_sorts = _double_sorts(panel, cfg, paths, sample)
    subperiods = _subperiods(deciles["timeseries_df"], paths, sample)
    fmb = _extended_fama_macbeth(panel, cfg, paths, sample)
    figs = save_empirical_figures(paths, sample)
    manifest = manifest_base(
        paths.root,
        "empirical",
        sample=sample,
        coverage=coverage,
        deciles=deciles["paths"],
        raw_vs_instrumented=raw_vs_inst,
        double_sorts=double_sorts,
        subperiods=subperiods,
        fama_macbeth_extended=fmb,
        figures=figs,
    )
    write_json(paths.manifests / f"empirical_{sample}.json", manifest)
    return manifest


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    for col in panel.columns:
        if col not in {"ticker", "comnam", "date", "available_date"}:
            try:
                panel[col] = pd.to_numeric(panel[col], errors="ignore")
            except Exception:
                pass
    panel["score_realized_reversal"] = -panel.get("pressure_real_z", 0.0)
    panel = panel.sort_values(["permno", "date"])
    panel["score_stale_reversal_12m"] = -panel.groupby("permno", sort=False)["pressure_pred_z"].shift(12)
    panel["has_pressure"] = panel.get("owner_count", 0).fillna(0).gt(0)
    return panel


def _coverage_by_year(panel: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, Any]:
    df = panel.copy()
    df["year"] = df["date"].dt.year
    out = df.groupby("year").agg(
        stock_months=("permno", "size"),
        stocks=("permno", "nunique"),
        pressure_covered=("has_pressure", "mean"),
        implementation_covered=("implementation_universe", "mean"),
        mean_owner_count=("owner_count", "mean"),
        mean_ownership_hhi=("ownership_hhi", "mean"),
        nonzero_pred_pressure=("pressure_pred_z", lambda x: x.abs().gt(1e-10).mean()),
    )
    out = out.reset_index()
    path = paths.tables / f"coverage_by_year_{sample}.csv"
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(paths.root)), "rows": int(len(out))}


def _decile_sorts(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = []
    ts_rows = []
    for score_name, score_col in SCORE_SPECS.items():
        if score_col not in panel:
            continue
        for universe_name, universe_df in _universes(panel):
            df = universe_df[["date", "permno", "next_excess_ret", score_col]].dropna()
            if df.empty:
                continue
            df["_rank"] = df.groupby("date")[score_col].rank(pct=True, method="first")
            df["decile"] = np.ceil(df["_rank"] * 10).clip(1, 10).astype(int)
            monthly = df.groupby(["date", "decile"])["next_excess_ret"].mean().reset_index()
            for decile, g in monthly.groupby("decile"):
                stat = _nw_mean(g["next_excess_ret"], int(cfg["modeling"]["newey_west_lags"]))
                rows.append(
                    {
                        "score": score_name,
                        "universe": universe_name,
                        "decile": int(decile),
                        "mean_monthly": stat["mean"],
                        "annualized_mean": stat["mean"] * 12,
                        "t": stat["t"],
                        "months": stat["n"],
                    }
                )
            spread = _spread_by_month(df, score_col)
            spread["score"] = score_name
            spread["universe"] = universe_name
            ts_rows.append(spread)
    summary = pd.DataFrame(rows)
    timeseries = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()
    if len(timeseries):
        spread_rows = []
        for (score, universe), g in timeseries.groupby(["score", "universe"]):
            stat = _nw_mean(g["long_short"], int(cfg["modeling"]["newey_west_lags"]))
            spread_rows.append(
                {
                    "score": score,
                    "universe": universe,
                    "annualized_spread": stat["mean"] * 12,
                    "monthly_spread": stat["mean"],
                    "t": stat["t"],
                    "months": stat["n"],
                    "sharpe": _ann_sharpe(g["long_short"]),
                }
            )
        spreads = pd.DataFrame(spread_rows)
    else:
        spreads = pd.DataFrame()
    summary_path = paths.tables / f"pressure_decile_summary_{sample}.csv"
    ts_path = paths.tables / f"pressure_decile_timeseries_{sample}.csv"
    spread_path = paths.tables / f"pressure_spread_summary_{sample}.csv"
    summary.to_csv(summary_path, index=False)
    timeseries.to_csv(ts_path, index=False)
    spreads.to_csv(spread_path, index=False)
    return {
        "timeseries_df": timeseries,
        "paths": {
            "decile_summary": str(summary_path.relative_to(paths.root)),
            "decile_timeseries": str(ts_path.relative_to(paths.root)),
            "spread_summary": str(spread_path.relative_to(paths.root)),
        },
    }


def _universes(panel: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = [("research", panel)]
    if "implementation_universe" in panel:
        out.append(("implementation", panel[panel["implementation_universe"].fillna(False)]))
    return out


def _spread_by_month(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby("date"):
        if len(g) < 100:
            continue
        lo = g["_rank"].le(0.10)
        hi = g["_rank"].ge(0.90)
        if lo.sum() < 20 or hi.sum() < 20:
            continue
        rows.append(
            {
                "date": date,
                "long_ret": g.loc[hi, "next_excess_ret"].mean(),
                "short_ret": g.loc[lo, "next_excess_ret"].mean(),
                "long_short": g.loc[hi, "next_excess_ret"].mean() - g.loc[lo, "next_excess_ret"].mean(),
                "long_n": int(hi.sum()),
                "short_n": int(lo.sum()),
            }
        )
    return pd.DataFrame(rows)


def _raw_vs_instrumented(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = []
    for score_name, score_col in SCORE_SPECS.items():
        if score_col not in panel:
            continue
        df = panel[["date", "permno", "next_excess_ret", score_col]].dropna()
        if df.empty:
            continue
        df["_rank"] = df.groupby("date")[score_col].rank(pct=True, method="first")
        spread = _spread_by_month(df, score_col)
        stat = _nw_mean(spread["long_short"], int(cfg["modeling"]["newey_west_lags"])) if len(spread) else {"mean": np.nan, "t": np.nan, "n": 0}
        rows.append(
            {
                "score": score_name,
                "monthly_spread": stat["mean"],
                "annualized_spread": stat["mean"] * 12,
                "t": stat["t"],
                "months": stat["n"],
                "sharpe": _ann_sharpe(spread["long_short"]) if len(spread) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    path = paths.tables / f"raw_vs_instrumented_spreads_{sample}.csv"
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(paths.root)), "rows": int(len(out))}


def _double_sorts(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = []
    base = panel.dropna(subset=["next_excess_ret", "score_pressure_reversal", "dollar_vol_lag1", "ownership_hhi"]).copy()
    if len(base):
        base["liquidity_bucket"] = base.groupby("date")["dollar_vol_lag1"].transform(lambda x: _tercile(x, reverse=True))
        base["crowding_bucket"] = base.groupby("date")["ownership_hhi"].transform(lambda x: _tercile(x, reverse=False))
        for bucket_col in ["liquidity_bucket", "crowding_bucket"]:
            for bucket, g in base.dropna(subset=[bucket_col]).groupby(bucket_col):
                g = g.copy()
                g["_rank"] = g.groupby("date")["score_pressure_reversal"].rank(pct=True, method="first")
                spread = _spread_by_month(g, "score_pressure_reversal")
                if spread.empty:
                    continue
                stat = _nw_mean(spread["long_short"], int(cfg["modeling"]["newey_west_lags"]))
                rows.append(
                    {
                        "sort": bucket_col.replace("_bucket", ""),
                        "bucket": int(bucket),
                        "monthly_spread": stat["mean"],
                        "annualized_spread": stat["mean"] * 12,
                        "t": stat["t"],
                        "months": stat["n"],
                    }
                )
    out = pd.DataFrame(rows)
    path = paths.tables / f"double_sort_spreads_{sample}.csv"
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(paths.root)), "rows": int(len(out))}


def _tercile(x: pd.Series, reverse: bool) -> pd.Series:
    pct = x.rank(pct=True, method="first")
    terc = np.ceil(pct * 3).clip(1, 3)
    if reverse:
        terc = 4 - terc
    return terc


def _subperiods(timeseries: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, Any]:
    if timeseries.empty:
        out = pd.DataFrame()
    else:
        ts = timeseries[timeseries["score"].eq("instrumented") & timeseries["universe"].eq("research")].copy()
        ts["date"] = pd.to_datetime(ts["date"])
        rows = []
        periods = [
            ("pre_gfc", "2004-01-01", "2007-12-31"),
            ("gfc", "2008-01-01", "2009-12-31"),
            ("post_gfc", "2010-01-01", "2019-12-31"),
            ("covid", "2020-01-01", "2021-12-31"),
            ("post_2021", "2022-01-01", "2025-12-31"),
        ]
        for name, start, end in periods:
            g = ts[ts["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            if g.empty:
                continue
            stat = _nw_mean(g["long_short"], 6)
            rows.append(
                {
                    "subperiod": name,
                    "start": start,
                    "end": end,
                    "monthly_spread": stat["mean"],
                    "annualized_spread": stat["mean"] * 12,
                    "t": stat["t"],
                    "months": stat["n"],
                    "sharpe": _ann_sharpe(g["long_short"]),
                }
            )
        out = pd.DataFrame(rows)
    path = paths.tables / f"subperiod_pressure_spreads_{sample}.csv"
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(paths.root)), "rows": int(len(out))}


def _extended_fama_macbeth(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    specs = {
        "pressure_only": ["score_pressure_reversal"],
        "characteristics": ["score_pressure_reversal", "log_mktcap", "book_to_market", "momentum_12_2", "reversal_1m"],
        "liquidity_crowding": [
            "score_pressure_reversal",
            "log_mktcap",
            "book_to_market",
            "momentum_12_2",
            "reversal_1m",
            "dollar_vol_lag1",
            "ownership_hhi",
            "pressure_x_illiquidity",
        ],
        "raw_and_instrumented": [
            "score_pressure_reversal",
            "score_realized_reversal",
            "log_mktcap",
            "book_to_market",
            "momentum_12_2",
            "reversal_1m",
        ],
    }
    rows = []
    lags = int(cfg["modeling"]["newey_west_lags"])
    for spec, cols in specs.items():
        cols = [c for c in cols if c in panel.columns]
        monthly_rows = []
        for date, g in panel.groupby("date"):
            x = g[cols].replace([np.inf, -np.inf], np.nan)
            y = pd.to_numeric(g["next_excess_ret"], errors="coerce")
            ok = y.notna() & x.notna().sum(axis=1).ge(max(1, len(cols) - 2))
            if ok.sum() < max(100, len(cols) + 10):
                continue
            x = x.loc[ok].apply(lambda s: s.fillna(s.median()), axis=0).astype("float64")
            x = sm.add_constant(x, has_constant="add")
            try:
                fit = sm.OLS(y.loc[ok].astype("float64"), x).fit()
                row = {"date": date}
                row.update(fit.params.to_dict())
                monthly_rows.append(row)
            except Exception:
                continue
        coefs = pd.DataFrame(monthly_rows)
        if coefs.empty:
            continue
        for term in [c for c in coefs.columns if c != "date"]:
            stat = _nw_mean(coefs[term], lags)
            rows.append({"spec": spec, "term": term, "coef": stat["mean"], "t": stat["t"], "months": stat["n"]})
    out = pd.DataFrame(rows)
    path = paths.tables / f"fama_macbeth_extended_{sample}.csv"
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(paths.root)), "rows": int(len(out))}


def _nw_mean(s: pd.Series, lags: int) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return {"mean": float(s.mean()) if len(s) else np.nan, "t": np.nan, "n": int(len(s))}
    x = np.ones((len(s), 1))
    fit = sm.OLS(s.to_numpy(dtype="float64"), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(fit.params[0]), "t": float(fit.tvalues[0]), "n": int(len(s))}


def _ann_sharpe(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    sd = s.std(ddof=1)
    return float((s.mean() / sd) * np.sqrt(12)) if len(s) > 2 and sd else np.nan
