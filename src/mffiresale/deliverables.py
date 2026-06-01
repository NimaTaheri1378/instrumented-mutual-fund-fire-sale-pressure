from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ProjectPaths
from .features import build_pressure_panel
from .io import manifest_base, write_json, write_parquet_atomic


def build_deliverables(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "full") -> dict[str, Any]:
    paths.ensure()
    manifest = manifest_base(paths.root, "deliverables", sample=sample, tables=[], figures=[], html=[])
    panel_path = paths.processed / f"model_panel_{sample}.parquet"
    rets_path = paths.processed / f"portfolio_returns_{sample}.parquet"
    panel = pd.read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
    rets = pd.read_parquet(rets_path) if rets_path.exists() else pd.DataFrame()
    if len(panel):
        panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp("M")
    if len(rets):
        rets["date"] = pd.to_datetime(rets["date"]).dt.to_period("M").dt.to_timestamp("M")

    for table in [
        _data_construction_table(cfg, paths, sample),
        _sample_summary_table(panel, paths, sample),
        _model_horse_race(paths, sample),
        _ablation_table(paths, sample),
        _robustness_table(cfg, paths, panel, sample),
        _fund_filter_robustness_table(cfg, paths, panel, sample),
        _return_horizon_table(panel, cfg, paths, sample),
    ]:
        if table:
            manifest["tables"].append(table)
    if len(rets):
        for table in [_bootstrap_table(rets, paths, sample)]:
            if table:
                manifest["tables"].append(table)
        for fig in [
            _rolling_strategy_figure(rets, paths, sample),
            _turnover_frontier_figure(paths, sample),
            _event_study_figure(panel, rets, paths, sample),
            _horizon_profile_figure(paths, sample),
        ]:
            if fig:
                manifest["figures"].append(fig)
        manifest["html"].extend(_plotly_outputs(panel, rets, paths, sample))
    shap_fig = _shap_beeswarm(paths, sample)
    if shap_fig:
        manifest["figures"].append(shap_fig)
    write_json(paths.manifests / f"deliverables_{sample}.json", manifest)
    return manifest


def _rel(paths: ProjectPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def _save(fig: plt.Figure, paths: ProjectPaths, name: str) -> dict[str, str]:
    png = paths.figures / f"{name}.png"
    svg = paths.figures / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": _rel(paths, png), "svg": _rel(paths, svg)}


def _data_construction_table(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = [
        ("Fund flow", "crsp_q_mutualfunds.monthly_tna + monthly_returns", "monthly", "uses t and t-1 TNA plus month t fund return", "Flow=(TNA_t-TNA_{t-1}(1+r_t))/(TNA_{t-1}(1+r_t))"),
        ("Predicted flow", "fund flow panel", "monthly", "walk-forward, prior-year-or-earlier training for each target year", "Ridge model using lagged flow, lagged returns, size, category and family leave-one-out flows"),
        ("Holdings weights", "crsp_q_mutualfunds.holdings", "quarterly reports", f"{cfg['project']['main_holdings_lag_days']}-day public-information lag", "percent_tna, fallback to market-value weights"),
        ("Predicted pressure", "fund predicted flows x lagged holdings", "stock-month", f"{cfg['project']['main_holdings_lag_days']}-day holdings lag", "sum_f w_{f,i,t-L} predicted_flow_{f,t}"),
        ("Realized pressure", "fund realized flows x lagged holdings", "stock-month", f"{cfg['project']['main_holdings_lag_days']}-day holdings lag", "sum_f w_{f,i,t-L} realized_flow_{f,t}"),
        ("CRSP returns", "crsp_a_stock.msf + msenames + msedelist", "stock-month", "next-month target; delisting returns included", "common shares on NYSE/AMEX/NASDAQ"),
        ("Compustat controls", "comp_na_daily_all.funda + crsp_a_ccm.ccmxpf_linktable", "annual accounting", "six-month accounting availability lag", "book-to-market, profitability, asset growth, leverage, cash/assets"),
        ("Factors", "ff_all.fivefactors_monthly", "monthly", "same month as portfolio return", "CAPM, FF3, FF5, FF5+UMD alpha attribution"),
    ]
    out = paths.tables / f"data_construction_{sample}.csv"
    pd.DataFrame(rows, columns=["variable_block", "source", "frequency", "timing_rule", "definition"]).to_csv(out, index=False)
    return {"name": "data_construction", "path": _rel(paths, out)}


def _sample_summary_table(panel: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, Any]:
    if panel.empty:
        return {}
    rows: list[dict[str, Any]] = []
    base = {
        "sample": sample,
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "stock_months": int(len(panel)),
        "unique_stocks": int(panel["permno"].nunique()),
        "pressure_coverage": float(panel.get("owner_count", pd.Series(0, index=panel.index)).fillna(0).gt(0).mean()),
        "implementation_share": float(panel.get("implementation_universe", pd.Series(False, index=panel.index)).fillna(False).mean()),
        "nonmissing_next_return": int(panel["next_ret"].notna().sum()) if "next_ret" in panel else np.nan,
        "nonmissing_compustat_controls": int(panel[["book_to_market", "profitability", "asset_growth"]].notna().all(axis=1).sum())
        if {"book_to_market", "profitability", "asset_growth"}.issubset(panel.columns)
        else np.nan,
    }
    rows.append(base)
    out = paths.tables / f"sample_summary_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "sample_summary", "path": _rel(paths, out)}


def _model_horse_race(paths: ProjectPaths, sample: str) -> dict[str, Any]:
    model_path = paths.manifests / f"models_{sample}.json"
    port_path = paths.tables / f"portfolio_summary_{sample}.csv"
    if not model_path.exists() or not port_path.exists():
        return {}
    models = json.loads(model_path.read_text(encoding="utf-8"))
    ports = pd.read_csv(port_path)
    mapping = {
        "elastic_net": "pred_elasticnet",
        "lightgbm": "pred_lightgbm",
        "xgboost": "pred_xgboost",
        "torch_mlp": "pred_torch_mlp",
        "conditional_deep": "pred_conditional_deep",
        "pressure_score": "score_pressure_reversal",
    }
    rows = []
    for model_name, prefix in mapping.items():
        metrics = models.get(model_name, {}) if model_name != "pressure_score" else {}
        for universe in ["research", "implementation"]:
            p = ports[ports["portfolio"].eq(f"{prefix}_{universe}")]
            row = {
                "model": model_name,
                "universe": universe,
                "test_r2": metrics.get("test_r2"),
                "test_rank_ic": metrics.get("test_rank_ic"),
                "test_rank_icir": metrics.get("test_rank_icir"),
                "gpu_used": metrics.get("gpu_used"),
            }
            if len(p):
                row.update(p.iloc[0].to_dict())
            rows.append(row)
    out = paths.tables / f"model_horse_race_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "model_horse_race", "path": _rel(paths, out), "rows": len(rows)}


def _ablation_table(paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = []
    spread_path = paths.tables / f"raw_vs_instrumented_spreads_{sample}.csv"
    fmb_path = paths.tables / f"fama_macbeth_extended_{sample}.csv"
    port_path = paths.tables / f"portfolio_summary_{sample}.csv"
    if spread_path.exists():
        spreads = pd.read_csv(spread_path)
        for _, row in spreads.iterrows():
            rows.append(
                {
                    "family": "pressure_score",
                    "specification": row["score"],
                    "estimate": row["annualized_spread"],
                    "t": row["t"],
                    "interpretation": "long-short spread from top-minus-bottom pressure deciles",
                }
            )
    if fmb_path.exists():
        fmb = pd.read_csv(fmb_path)
        fmb = fmb[fmb["term"].isin(["score_pressure_reversal", "score_realized_reversal"])]
        labels = {
            "pressure_only": "without Compustat controls or liquidity/crowding interactions",
            "characteristics": "adds size, book-to-market, momentum, and reversal controls",
            "liquidity_crowding": "adds liquidity, ownership concentration, and pressure-liquidity interaction",
            "raw_and_instrumented": "horse race between instrumented and realized-pressure scores",
        }
        for _, row in fmb.iterrows():
            rows.append(
                {
                    "family": "fama_macbeth",
                    "specification": f"{row['spec']}:{row['term']}",
                    "estimate": row["coef"],
                    "t": row["t"],
                    "interpretation": labels.get(row["spec"], "monthly cross-sectional coefficient with Newey-West aggregation"),
                }
            )
    if port_path.exists():
        ports = pd.read_csv(port_path)
        p = ports[ports["portfolio"].eq("score_pressure_reversal_research")]
        if len(p):
            row = p.iloc[0]
            rows.append(
                {
                    "family": "factor_neutralization",
                    "specification": "sector_beta_size_neutralized_pressure_backtest",
                    "estimate": row.get("annualized_mean"),
                    "t": row.get("t_mean"),
                    "interpretation": "monthly long-short portfolio after sector, beta, and size neutralization",
                }
            )
    if not rows:
        return {}
    out = paths.tables / f"ablation_table_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "ablation_table", "path": _rel(paths, out), "rows": len(rows)}


def _robustness_table(cfg: dict[str, Any], paths: ProjectPaths, panel: pd.DataFrame, sample: str) -> dict[str, Any]:
    rows = []
    sub_path = paths.tables / f"subperiod_pressure_spreads_{sample}.csv"
    dbl_path = paths.tables / f"double_sort_spreads_{sample}.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        for _, row in sub.iterrows():
            rows.append({"family": "subperiod", "specification": row["subperiod"], "annualized_spread": row["annualized_spread"], "t": row["t"], "months": row["months"]})
    if dbl_path.exists():
        dbl = pd.read_csv(dbl_path)
        for _, row in dbl.iterrows():
            rows.append({"family": row["sort"], "specification": f"bucket_{row['bucket']}", "annualized_spread": row["annualized_spread"], "t": row["t"], "months": row["months"]})
    if not panel.empty:
        main = _pressure_spread(panel)
        rows.append({"family": "timing", "specification": "holdings_lag_60d_main", **main})
        lag90 = _lag90_spread(cfg, paths, sample)
        if lag90:
            rows.append({"family": "timing", "specification": "holdings_lag_90d_robustness", **lag90})
    if not rows:
        return {}
    out = paths.tables / f"robustness_table_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "robustness_table", "path": _rel(paths, out), "rows": len(rows)}


def _lag90_spread(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    flows = paths.interim / "mutual_funds" / f"portfolio_flows_{sample}.parquet"
    stock_path = paths.interim / "stocks" / f"stock_features_{sample}.parquet"
    if not flows.exists() or not stock_path.exists():
        return {}
    pressure_path = build_pressure_panel(cfg, paths, flows, sample, lag_days=int(cfg["project"]["robustness_holdings_lag_days"]))
    pressure = pd.read_parquet(pressure_path)
    stock = pd.read_parquet(stock_path)
    if pressure.empty or stock.empty:
        return {}
    pressure["date"] = pd.to_datetime(pressure["date"]).dt.to_period("M").dt.to_timestamp("M")
    stock["date"] = pd.to_datetime(stock["date"]).dt.to_period("M").dt.to_timestamp("M")
    panel = stock.merge(pressure[["permno", "date", "pressure_pred_z"]], on=["permno", "date"], how="left")
    panel["pressure_pred_z"] = panel["pressure_pred_z"].fillna(0.0)
    panel["score_pressure_reversal"] = -panel["pressure_pred_z"]
    return _pressure_spread(panel)


def _fund_filter_robustness_table(cfg: dict[str, Any], paths: ProjectPaths, panel: pd.DataFrame, sample: str) -> dict[str, Any]:
    flows_path = paths.interim / "mutual_funds" / f"fund_flows_{sample}.parquet"
    stock_path = paths.interim / "stocks" / f"stock_features_{sample}.parquet"
    if panel.empty or not flows_path.exists() or not stock_path.exists():
        return {}
    flows = pd.read_parquet(flows_path)
    if flows.empty or "crsp_portno" not in flows:
        return {}
    specs: list[tuple[str, pd.Series, str]] = [("all_funds_reference", pd.Series(True, index=flows.index), "all mapped funds")]
    if "index_fund_flag" in flows:
        specs.append(("exclude_index_funds", ~_flagged(flows["index_fund_flag"]), "drops nonempty CRSP index-fund flags"))
    if "et_flag" in flows:
        specs.append(("exclude_etfs", ~_flagged(flows["et_flag"]), "drops nonempty CRSP ETF/exchange-traded flags"))
    if {"index_fund_flag", "et_flag"}.issubset(flows.columns):
        specs.append(
            (
                "active_funds_only_proxy",
                ~_flagged(flows["index_fund_flag"]) & ~_flagged(flows["et_flag"]),
                "proxy active-fund sample: drops index-fund and ETF/exchange-traded flags",
            )
        )
    rows = []
    for spec, mask, note in specs:
        filtered = flows.loc[mask.fillna(False)].copy()
        if filtered.empty:
            continue
        if spec == "all_funds_reference":
            pressure_path = paths.interim / "pressure" / f"pressure_lag{cfg['project']['main_holdings_lag_days']}_{sample}.parquet"
        else:
            portfolio_flow_path = _filtered_portfolio_flows(paths, filtered, sample, spec)
            pressure_path = build_pressure_panel(
                cfg,
                paths,
                portfolio_flow_path,
                sample,
                lag_days=int(cfg["project"]["main_holdings_lag_days"]),
                output_sample=f"{sample}_{spec}",
                holdings_sample=sample,
            )
        if not pressure_path.exists():
            continue
        pressure = pd.read_parquet(pressure_path)
        filtered_panel = _stock_panel_with_pressure(stock_path, pressure)
        stat = _pressure_spread(filtered_panel)
        rows.append(
            {
                "specification": spec,
                "note": note,
                "fund_months": int(len(filtered)),
                "unique_funds": int(filtered["crsp_fundno"].nunique()) if "crsp_fundno" in filtered else np.nan,
                "unique_portfolios": int(filtered["crsp_portno"].nunique()),
                "pressure_stock_months": int(len(pressure)),
                **stat,
            }
        )
    if not rows:
        return {}
    out = paths.tables / f"fund_filter_robustness_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "fund_filter_robustness", "path": _rel(paths, out), "rows": len(rows)}


def _flagged(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip().str.upper()
    false_like = {"", "0", "N", "NO", "FALSE", "F", "NONE", "NAN", "<NA>", "NA", "NULL"}
    return text.notna() & ~text.isin(false_like)


def _filtered_portfolio_flows(paths: ProjectPaths, flows: pd.DataFrame, sample: str, spec: str) -> Path:
    out = paths.interim / "mutual_funds" / f"portfolio_flows_{sample}_{spec}.parquet"
    df = flows[flows["crsp_portno"].notna()].copy()
    df["flow_weight"] = pd.to_numeric(df["mtna_lag1"], errors="coerce").where(lambda x: x.gt(0), np.nan)
    for col in ["flow_capped", "pred_flow_exo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_flow_v"] = df["flow_capped"] * df["flow_weight"]
    df["_pred_v"] = df["pred_flow_exo"] * df["flow_weight"]
    grouped = df.groupby(["crsp_portno", "caldt"], dropna=False).agg(
        flow_num=("_flow_v", "sum"),
        pred_num=("_pred_v", "sum"),
        weight_sum=("flow_weight", "sum"),
        fund_count=("crsp_fundno", "nunique"),
    )
    grouped = grouped.reset_index()
    grouped["portfolio_flow"] = grouped["flow_num"] / grouped["weight_sum"]
    grouped["portfolio_pred_flow"] = grouped["pred_num"] / grouped["weight_sum"]
    grouped = grouped.drop(columns=["flow_num", "pred_num"])
    grouped.loc[~np.isfinite(grouped["portfolio_flow"]), "portfolio_flow"] = np.nan
    grouped.loc[~np.isfinite(grouped["portfolio_pred_flow"]), "portfolio_pred_flow"] = np.nan
    write_parquet_atomic(grouped, out)
    return out


def _stock_panel_with_pressure(stock_path: Path, pressure: pd.DataFrame) -> pd.DataFrame:
    stock = pd.read_parquet(stock_path)
    if stock.empty or pressure.empty:
        return pd.DataFrame()
    stock["date"] = pd.to_datetime(stock["date"]).dt.to_period("M").dt.to_timestamp("M")
    pressure = pressure.copy()
    pressure["date"] = pd.to_datetime(pressure["date"]).dt.to_period("M").dt.to_timestamp("M")
    cols = [c for c in ["permno", "date", "pressure_pred_z", "owner_count", "ownership_hhi"] if c in pressure.columns]
    merged = stock.merge(pressure[cols], on=["permno", "date"], how="left")
    merged["pressure_pred_z"] = pd.to_numeric(merged["pressure_pred_z"], errors="coerce").fillna(0.0)
    merged["score_pressure_reversal"] = -merged["pressure_pred_z"]
    return merged


def _pressure_spread(panel: pd.DataFrame) -> dict[str, Any]:
    df = panel[["date", "permno", "next_excess_ret", "score_pressure_reversal"]].dropna().copy()
    if df.empty:
        return {"annualized_spread": np.nan, "t": np.nan, "months": 0}
    df["_rank"] = df.groupby("date")["score_pressure_reversal"].rank(pct=True, method="first")
    monthly = []
    for date, g in df.groupby("date"):
        hi = g["_rank"].ge(0.90)
        lo = g["_rank"].le(0.10)
        if hi.sum() < 20 or lo.sum() < 20:
            continue
        monthly.append({"date": date, "spread": g.loc[hi, "next_excess_ret"].mean() - g.loc[lo, "next_excess_ret"].mean()})
    out = pd.DataFrame(monthly)
    stat = _nw_mean(out["spread"], 6) if len(out) else {"mean": np.nan, "t": np.nan, "n": 0}
    return {"annualized_spread": stat["mean"] * 12, "t": stat["t"], "months": stat["n"]}


def _return_horizon_table(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    needed = {"date", "permno", "next_excess_ret", "score_pressure_reversal"}
    if panel.empty or not needed.issubset(panel.columns):
        return {}
    work = panel[list(needed)].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.to_period("M").dt.to_timestamp("M")
    work["_rank"] = work.groupby("date")["score_pressure_reversal"].rank(pct=True, method="first")
    rows = []
    lags = int(cfg["modeling"]["newey_west_lags"])
    for horizon in [1, 3, 6, 12]:
        target = _future_compounded_returns(work, horizon)
        spreads = []
        for date, g in work.assign(_target=target).dropna(subset=["_target", "_rank"]).groupby("date"):
            hi = g["_rank"].ge(0.90)
            lo = g["_rank"].le(0.10)
            if hi.sum() < 20 or lo.sum() < 20:
                continue
            spreads.append({"date": date, "spread": g.loc[hi, "_target"].mean() - g.loc[lo, "_target"].mean()})
        ts = pd.DataFrame(spreads)
        stat = _nw_mean(ts["spread"], lags) if len(ts) else {"mean": np.nan, "t": np.nan, "n": 0}
        rows.append(
            {
                "horizon_months": horizon,
                "mean_horizon_spread": stat["mean"],
                "annualized_spread": ((1.0 + stat["mean"]) ** (12.0 / horizon) - 1.0) if np.isfinite(stat["mean"]) and stat["mean"] > -1 else np.nan,
                "t": stat["t"],
                "months": stat["n"],
            }
        )
    out = paths.tables / f"return_horizon_profile_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "return_horizon_profile", "path": _rel(paths, out), "rows": len(rows)}


def _future_compounded_returns(panel: pd.DataFrame, horizon: int) -> pd.Series:
    df = panel[["permno", "date", "next_excess_ret"]].copy().sort_values(["permno", "date"])
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    grouped = df.groupby("permno", sort=False)
    base_period = df["date"].dt.to_period("M")
    valid = pd.Series(True, index=df.index)
    compounded = pd.Series(1.0, index=df.index, dtype="float64")
    for step in range(horizon):
        r = pd.to_numeric(grouped["next_excess_ret"].shift(-step), errors="coerce")
        d = grouped["date"].shift(-step)
        expected = (base_period + step).dt.to_timestamp("M")
        valid &= r.notna() & d.eq(expected)
        compounded *= 1.0 + r.fillna(0.0)
    out = compounded - 1.0
    out.loc[~valid] = np.nan
    return out.reindex(panel.index)


def _bootstrap_table(rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, Any]:
    rows = []
    for name, g in rets.groupby("portfolio"):
        s = pd.to_numeric(g.sort_values("date")["long_short_ret"], errors="coerce").dropna()
        lo, hi = _moving_block_mean_ci(s)
        rows.append({"portfolio": name, "monthly_mean_lo": lo, "monthly_mean_hi": hi, "annualized_mean_lo": lo * 12, "annualized_mean_hi": hi * 12, "months": len(s)})
    out = paths.tables / f"moving_block_bootstrap_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"name": "moving_block_bootstrap", "path": _rel(paths, out), "rows": len(rows)}


def _rolling_strategy_figure(rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    target = "score_pressure_reversal_research"
    if target not in set(rets["portfolio"]):
        target = str(rets["portfolio"].iloc[0])
    g = rets[rets["portfolio"].eq(target)].sort_values("date").copy()
    s = pd.to_numeric(g["long_short_ret"], errors="coerce")
    roll_mean = s.rolling(36, min_periods=24).mean() * 12
    roll_sharpe = s.rolling(36, min_periods=24).mean() / s.rolling(36, min_periods=24).std() * np.sqrt(12)
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    ax1.plot(g["date"], roll_mean, color="#315c72", label="Rolling annualized mean")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Annualized return")
    ax2 = ax1.twinx()
    ax2.plot(g["date"], roll_sharpe, color="#8f4f4f", label="Rolling Sharpe")
    ax2.set_ylabel("Sharpe")
    lines = [line for line in ax1.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper center", ncol=2, frameon=True)
    ax1.set_title("Rolling Pressure-Strategy Performance")
    return _save(fig, paths, f"rolling_strategy_metrics_{sample}")


def _turnover_frontier_figure(paths: ProjectPaths, sample: str) -> dict[str, str]:
    cost_path = paths.tables / f"transaction_costs_{sample}.csv"
    if not cost_path.exists():
        return {}
    cost = pd.read_csv(cost_path)
    if cost.empty:
        return {}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, g in cost.groupby("portfolio"):
        if "pressure" not in name and "lightgbm" not in name and "torch" not in name and "conditional" not in name:
            continue
        g = g.sort_values("cost_bps_per_turnover")
        ax.plot(g["cost_bps_per_turnover"], g["annualized_net_mean"], marker="o", linewidth=1.6, label=name.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Turnover-Cost Frontier")
    ax.set_xlabel("Cost per unit turnover (bps)")
    ax.set_ylabel("Annualized net long-short return")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5, frameon=True)
    return _save(fig, paths, f"turnover_cost_frontier_{sample}")


def _event_study_figure(panel: pd.DataFrame, rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    if panel.empty:
        return {}
    target = rets[rets["portfolio"].eq("score_pressure_reversal_research")][["date", "long_short_ret"]].copy()
    if target.empty:
        return {}
    agg = panel.groupby("date")["pressure_pred"].mean().dropna().sort_values()
    if len(agg) < 10:
        return {}
    events = list(agg.head(min(10, max(3, len(agg) // 20))).index)
    rows = []
    ret_map = target.set_index("date")["long_short_ret"]
    for event_date in events:
        event_period = pd.Period(event_date, "M")
        for k in range(-6, 7):
            date = (event_period + k).to_timestamp("M")
            if date in ret_map:
                rows.append({"event_date": event_date, "event_month": k, "long_short_ret": ret_map.loc[date]})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return {}
    table = ev.groupby("event_month")["long_short_ret"].agg(["mean", "count"]).reset_index()
    table.to_csv(paths.tables / f"event_study_extreme_pressure_{sample}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(table["event_month"], table["mean"], marker="o", color="#315c72")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="#8f4f4f", linewidth=1.0)
    ax.set_title("Pressure Strategy Around Extreme Outflow Months")
    ax.set_xlabel("Months from event")
    ax.set_ylabel("Average long-short return")
    return _save(fig, paths, f"event_study_extreme_pressure_{sample}")


def _horizon_profile_figure(paths: ProjectPaths, sample: str) -> dict[str, str]:
    table_path = paths.tables / f"return_horizon_profile_{sample}.csv"
    if not table_path.exists():
        return {}
    tbl = pd.read_csv(table_path)
    if tbl.empty:
        return {}
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    ax1.plot(tbl["horizon_months"], tbl["annualized_spread"], marker="o", color="#315c72", linewidth=1.8, label="Annualized spread")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Return horizon (months)")
    ax1.set_ylabel("Annualized top-minus-bottom spread")
    ax2 = ax1.twinx()
    ax2.plot(tbl["horizon_months"], tbl["t"], marker="s", color="#8f4f4f", linewidth=1.4, label="Newey-West t")
    ax2.set_ylabel("t-statistic")
    lines = [line for line in ax1.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", frameon=True)
    ax1.set_title("Pressure Signal Across Return Horizons")
    return _save(fig, paths, f"return_horizon_profile_{sample}")


def _shap_beeswarm(paths: ProjectPaths, sample: str) -> dict[str, str]:
    sample_path = paths.tables / f"lightgbm_shap_sample_{sample}.parquet"
    summary_path = paths.tables / f"lightgbm_shap_summary_{sample}.csv"
    if not sample_path.exists() or not summary_path.exists():
        return {}
    shap = pd.read_parquet(sample_path)
    summary = pd.read_csv(summary_path)
    order = summary["feature"].head(12).tolist()[::-1]
    shap = shap[shap["feature"].isin(order)].copy()
    if shap.empty:
        return {}
    ymap = {feature: i for i, feature in enumerate(order)}
    rng = np.random.default_rng(20260601)
    shap["_y"] = shap["feature"].map(ymap) + rng.normal(0, 0.08, size=len(shap))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sc = ax.scatter(shap["shap_value"], shap["_y"], c=shap["feature_value_z"], cmap="vlag", s=4, alpha=0.35, linewidths=0)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("LightGBM SHAP Summary")
    ax.set_xlabel("SHAP contribution to predicted return")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Feature value z-score")
    return _save(fig, paths, f"lightgbm_shap_summary_{sample}")


def _plotly_outputs(panel: pd.DataFrame, rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> list[dict[str, str]]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        return []
    out: list[dict[str, str]] = []
    html_dir = paths.reports / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    if len(rets):
        wealth_rows = []
        for name, g in rets.groupby("portfolio"):
            g = g.sort_values("date").copy()
            g["wealth"] = (1.0 + g["long_short_ret"].fillna(0.0)).cumprod()
            wealth_rows.append(g[["date", "portfolio", "wealth"]])
        wealth = pd.concat(wealth_rows, ignore_index=True)
        fig = px.line(wealth, x="date", y="wealth", color="portfolio", log_y=True, title="Cumulative Long-Short Returns")
        path = html_dir / f"cumulative_long_short_{sample}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        out.append({"name": "cumulative_long_short", "html": _rel(paths, path)})
    if len(panel) and "pressure_pred" in panel:
        agg = panel.groupby("date")["pressure_pred"].mean().reset_index()
        fig = px.line(agg, x="date", y="pressure_pred", title="Aggregate Predicted Mutual-Fund Pressure")
        path = html_dir / f"aggregate_predicted_pressure_{sample}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        out.append({"name": "aggregate_predicted_pressure", "html": _rel(paths, path)})
    heat_path = paths.tables / f"double_sort_spreads_{sample}.csv"
    if heat_path.exists():
        dbl = pd.read_csv(heat_path)
        fig = go.Figure()
        for family, g in dbl.groupby("sort"):
            fig.add_trace(go.Bar(x=g["bucket"].astype(str), y=g["annualized_spread"], name=family))
        fig.update_layout(title="Mechanism Spreads by Liquidity and Crowding", xaxis_title="Bucket", yaxis_title="Annualized spread")
        path = html_dir / f"mechanism_spreads_{sample}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        out.append({"name": "mechanism_spreads", "html": _rel(paths, path)})
    return out


def _nw_mean(s: pd.Series, lags: int) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return {"mean": float(s.mean()) if len(s) else np.nan, "t": np.nan, "n": int(len(s))}
    x = np.ones((len(s), 1))
    fit = sm.OLS(s.to_numpy(dtype="float64"), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(fit.params[0]), "t": float(fit.tvalues[0]), "n": int(len(s))}


def _moving_block_mean_ci(s: pd.Series, block: int = 6, draws: int = 500) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(x) < block + 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(20260601)
    starts = np.arange(0, len(x) - block + 1)
    blocks_needed = int(np.ceil(len(x) / block))
    means = []
    for _ in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[i : i + block] for i in chosen])[: len(x)]
        means.append(sample.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)
