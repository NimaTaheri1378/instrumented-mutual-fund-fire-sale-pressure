from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ProjectPaths
from .io import manifest_base, write_json


def make_figures(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    paths.ensure()
    sns.set_theme(style="whitegrid", context="paper")
    manifest = manifest_base(paths.root, "figures", sample=sample, figures=[])
    panel_path = paths.processed / f"model_panel_{sample}.parquet"
    rets_path = paths.processed / f"portfolio_returns_{sample}.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        if not panel.empty:
            manifest["figures"].extend(_panel_figures(panel, paths, sample))
    if rets_path.exists():
        rets = pd.read_parquet(rets_path)
        if not rets.empty:
            manifest["figures"].extend(_portfolio_figures(rets, paths, sample))
    imp_path = paths.tables / f"lightgbm_feature_importance_{sample}.csv"
    if imp_path.exists():
        manifest["figures"].append(_feature_importance_figure(pd.read_csv(imp_path), paths, sample))
    write_json(paths.manifests / f"figures_{sample}.json", manifest)
    return manifest


def _save(fig: plt.Figure, paths: ProjectPaths, name: str) -> dict[str, str]:
    png = paths.figures / f"{name}.png"
    svg = paths.figures / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.relative_to(paths.root)), "svg": str(svg.relative_to(paths.root))}


def _panel_figures(panel: pd.DataFrame, paths: ProjectPaths, sample: str) -> list[dict[str, str]]:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    figs = []
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    s = panel.groupby("date")["pressure_pred"].mean().dropna()
    ax.plot(s.index, s.values, color="#315c72", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Aggregate Predicted Mutual-Fund Pressure")
    ax.set_xlabel("")
    ax.set_ylabel("Cross-sectional mean")
    figs.append(_save(fig, paths, f"aggregate_predicted_pressure_{sample}"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = panel["pressure_pred_z"].replace([float("inf"), float("-inf")], pd.NA).dropna()
    x = x[x.abs() > 1e-10]
    x = x.clip(x.quantile(0.01), x.quantile(0.99))
    sns.histplot(x, bins=80, ax=ax, color="#5c7f67")
    ax.set_title("Distribution of Nonzero Predicted Pressure Shocks")
    ax.set_xlabel("Predicted pressure z-score")
    ax.set_ylabel("Stock-month count")
    figs.append(_save(fig, paths, f"pressure_distribution_{sample}"))

    if {"dollar_vol_lag1", "ownership_hhi", "score_pressure_reversal", "next_ret"}.issubset(panel.columns):
        work = panel.dropna(subset=["dollar_vol_lag1", "ownership_hhi", "score_pressure_reversal", "next_ret"]).copy()
        if len(work) > 100:
            work["liquidity_bucket"] = pd.qcut(work["dollar_vol_lag1"], 5, labels=False, duplicates="drop")
            work["crowding_bucket"] = pd.qcut(work["ownership_hhi"], 5, labels=False, duplicates="drop")
            heat = work.groupby(["liquidity_bucket", "crowding_bucket"])["next_ret"].mean().unstack()
            heat = heat.apply(pd.to_numeric, errors="coerce").astype("float64")
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            sns.heatmap(heat, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Mean next return"})
            ax.set_title("Return Strength by Liquidity and Crowding")
            ax.set_xlabel("Crowding bucket")
            ax.set_ylabel("Liquidity bucket")
            if heat.shape[1] == 5:
                ax.set_xticklabels(["Low", "2", "3", "4", "High"], rotation=0)
            if heat.shape[0] == 5:
                ax.set_yticklabels(["Low", "2", "3", "4", "High"], rotation=0)
            figs.append(_save(fig, paths, f"liquidity_crowding_heatmap_{sample}"))
    return figs


def _portfolio_figures(rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> list[dict[str, str]]:
    rets = rets.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    figs = []
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    wealth_by_name: dict[str, pd.Series] = {}
    for name, g in rets.groupby("portfolio"):
        g = g.sort_values("date")
        wealth = (1.0 + g["long_short_ret"].fillna(0.0)).cumprod()
        wealth_by_name[name] = wealth
        ax.plot(g["date"], wealth, linewidth=1.45, label=_pretty_portfolio_label(name))
    max_wealth = max(float(w.max()) for w in wealth_by_name.values() if len(w)) if wealth_by_name else 1.0
    min_positive = min(float(w[w.gt(0)].min()) for w in wealth_by_name.values() if w.gt(0).any()) if wealth_by_name else 1.0
    if max_wealth / max(min_positive, 1e-12) > 100:
        ax.set_yscale("log")
        ax.set_title("Cumulative Long-Short Portfolio Returns (Log Scale)")
        ax.set_ylabel("Growth of $1 (log scale)")
    else:
        ax.set_title("Cumulative Long-Short Portfolio Returns")
        ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5, frameon=True)
    figs.append(_save(fig, paths, f"cumulative_long_short_{sample}"))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    target = _headline_portfolio(rets, paths, sample)
    g = rets[rets["portfolio"].eq(target)].sort_values("date")
    wealth = (1.0 + g["long_short_ret"].fillna(0.0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    ax.fill_between(g["date"], dd, 0, color="#8f4f4f", alpha=0.45)
    ax.set_title(f"Drawdown: {_pretty_portfolio_label(target)}")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    figs.append(_save(fig, paths, f"drawdown_{sample}"))
    return figs


def _headline_portfolio(rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> str:
    summary_path = paths.tables / f"portfolio_summary_{sample}.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["portfolio"].astype(str).str.endswith("_research")].copy()
        summary["sharpe"] = pd.to_numeric(summary.get("sharpe"), errors="coerce")
        if summary["sharpe"].notna().any():
            return str(summary.sort_values("sharpe").iloc[-1]["portfolio"])
    preferred = "pred_lightgbm_research"
    if preferred in set(rets["portfolio"]):
        return preferred
    return str(rets.groupby("portfolio")["long_short_ret"].mean().sort_values().index[-1])


def _feature_importance_figure(imp: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    if imp.empty or "feature" not in imp:
        return {}
    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce").fillna(0)
    imp = imp[imp["importance"].gt(0)].sort_values("importance", ascending=False).head(18).sort_values("importance")
    if imp.empty:
        return {}
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.barh(imp["feature"], imp["importance"], color="#6f7d9a")
    ax.set_title("LightGBM Feature Importance")
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    return _save(fig, paths, f"lightgbm_feature_importance_{sample}")


def _pretty_portfolio_label(name: str) -> str:
    label = name.replace("score_pressure_reversal", "Pressure")
    label = label.replace("pred_elasticnet", "Elastic Net")
    label = label.replace("pred_lightgbm", "LightGBM")
    label = label.replace("pred_xgboost", "XGBoost")
    label = label.replace("pred_torch_mlp", "Torch MLP")
    label = label.replace("pred_conditional_deep", "Conditional Deep")
    label = label.replace("_implementation", " impl.")
    label = label.replace("_research", " research")
    return label.replace("_", " ")


def save_empirical_figures(paths: ProjectPaths, sample: str) -> list[dict[str, str]]:
    sns.set_theme(style="whitegrid", context="paper")
    figs: list[dict[str, str]] = []
    decile_path = paths.tables / f"pressure_decile_summary_{sample}.csv"
    spread_path = paths.tables / f"raw_vs_instrumented_spreads_{sample}.csv"
    sub_path = paths.tables / f"subperiod_pressure_spreads_{sample}.csv"
    cov_path = paths.tables / f"coverage_by_year_{sample}.csv"
    if decile_path.exists():
        dec = pd.read_csv(decile_path)
        figs.append(_decile_figure(dec, paths, sample))
    if spread_path.exists():
        spreads = pd.read_csv(spread_path)
        figs.append(_spread_comparison_figure(spreads, paths, sample))
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        figs.append(_subperiod_figure(sub, paths, sample))
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        figs.append(_coverage_figure(cov, paths, sample))
    return [f for f in figs if f]


def _decile_figure(dec: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    work = dec[(dec["score"].eq("instrumented")) & (dec["universe"].eq("research"))].copy()
    if work.empty:
        return {}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(work["decile"], work["annualized_mean"], color="#5c7f67")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Instrumented Pressure Decile Returns")
    ax.set_xlabel("Pressure reversal score decile")
    ax.set_ylabel("Annualized next-month excess return")
    ax.set_xticks(range(1, 11))
    return _save(fig, paths, f"pressure_decile_returns_{sample}")


def _spread_comparison_figure(spreads: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    if spreads.empty:
        return {}
    order = ["instrumented", "realized", "stale_12m"]
    work = spreads.set_index("score").reindex(order).dropna(subset=["annualized_spread"]).reset_index()
    if work.empty:
        return {}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = ["#315c72", "#8f4f4f", "#8a8a8a"][: len(work)]
    ax.bar(work["score"].str.replace("_", " ").str.title(), work["annualized_spread"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Raw, Instrumented, and Placebo Pressure Spreads")
    ax.set_ylabel("Annualized long-short excess return")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, paths, f"raw_vs_instrumented_spreads_{sample}")


def _subperiod_figure(sub: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    if sub.empty:
        return {}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = sub["subperiod"].str.replace("_", " ").str.title()
    ax.bar(labels, sub["annualized_spread"], color="#6f7d9a")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Instrumented Pressure Spreads by Subperiod")
    ax.set_ylabel("Annualized long-short excess return")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, paths, f"subperiod_pressure_spreads_{sample}")


def _coverage_figure(cov: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, str]:
    if cov.empty:
        return {}
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )
    ax1.plot(cov["year"], cov["pressure_covered"], color="#315c72", linewidth=1.8, label="Pressure coverage")
    ax1.plot(cov["year"], cov["implementation_covered"], color="#5c7f67", linewidth=1.8, label="Implementation universe")
    ax1.set_ylim(0, 1)
    ax1.set_title("Coverage by Year")
    ax1.set_ylabel("Share of stock-months")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=True)
    ax2.bar(cov["year"], cov["stocks"], alpha=0.55, color="#8d98ad")
    ax2.set_ylabel("Unique stocks")
    ax2.set_xlabel("")
    return _save(fig, paths, f"coverage_by_year_{sample}")
