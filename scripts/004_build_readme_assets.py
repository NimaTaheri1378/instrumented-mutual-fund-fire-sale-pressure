from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES = [
    "coverage_by_year_full.png",
    "aggregate_predicted_pressure_full.png",
    "lightgbm_feature_importance_full.png",
    "turnover_cost_frontier_full.png",
    "strategy_candidate_locked_test_full.png",
    "return_horizon_profile_full.png",
]

TABLES = [
    "sample_summary_full.csv",
    "model_horse_race_full.csv",
    "iv_2sls_summary_full.csv",
    "dml_iv_summary_full.csv",
    "strategy_candidate_selected_full.csv",
    "raw_vs_instrumented_spreads_full.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build README-safe aggregate assets.")
    parser.add_argument("--artifact-root", default="reports/remote_proposal_grade")
    args = parser.parse_args()
    root = Path(".").resolve()
    artifact = (root / args.artifact_root).resolve()
    fig_src = artifact / "reports" / "figures"
    table_src = artifact / "reports" / "tables"
    fig_dst = root / "docs" / "figures"
    table_dst = root / "docs" / "assets" / "tables"
    fig_dst.mkdir(parents=True, exist_ok=True)
    table_dst.mkdir(parents=True, exist_ok=True)

    for name in FIGURES:
        src = fig_src / name
        if src.exists():
            shutil.copy2(src, fig_dst / name)
    for name in TABLES:
        src = table_src / name
        if src.exists():
            shutil.copy2(src, table_dst / name)

    _headline_table(table_src, table_dst)
    _headline_figure(artifact, fig_dst)
    return 0


def _headline_table(table_src: Path, table_dst: Path) -> None:
    rows = []
    sample = _read_csv(table_src / "sample_summary_full.csv")
    if not sample.empty:
        row = sample.iloc[0]
        rows.append({"block": "Panel", "metric": "Stock-months", "value": f"{int(row['stock_months']):,}"})
        rows.append({"block": "Panel", "metric": "Stocks", "value": f"{int(row['unique_stocks']):,}"})
        rows.append({"block": "Panel", "metric": "Pressure coverage", "value": f"{row['pressure_coverage']:.1%}"})
    models = _read_csv(table_src / "model_horse_race_full.csv")
    if not models.empty:
        impl = models[models["universe"].eq("implementation")].copy()
        impl = impl.sort_values("ff5_umd_alpha_t", ascending=False)
        if len(impl):
            row = impl.iloc[0]
            rows.append({"block": "Implementation", "metric": "Best FF5+UMD alpha t", "value": f"{row['ff5_umd_alpha_t']:.2f}"})
            rows.append({"block": "Implementation", "metric": "Best annualized mean", "value": f"{row['annualized_mean']:.1%}"})
        rank = models.dropna(subset=["test_rank_ic"]).sort_values("test_rank_ic", ascending=False)
        if len(rank):
            row = rank.iloc[0]
            rows.append({"block": "Prediction", "metric": "Best locked-test rank IC", "value": f"{row['test_rank_ic']:.2%}"})
    selected = _read_csv(table_src / "strategy_candidate_selected_full.csv")
    if not selected.empty:
        test = selected[selected["window"].eq("locked_test")]
        if len(test):
            row = test.iloc[0]
            rows.append({"block": "Strategy audit", "metric": "Validation-selected test net 25 bps", "value": f"{row['net25_annualized_mean']:.1%}"})
    pd.DataFrame(rows).to_csv(table_dst / "headline_metrics.csv", index=False)


def _headline_figure(artifact: Path, fig_dst: Path) -> None:
    returns_path = artifact / "data" / "processed" / "portfolio_returns_full.parquet"
    if not returns_path.exists():
        return
    rets = pd.read_parquet(returns_path)
    rets["date"] = pd.to_datetime(rets["date"]).dt.to_period("M").dt.to_timestamp("M")
    keep = {
        "pred_xgboost_implementation": "XGBoost implementation",
        "pred_torch_mlp_implementation": "Torch MLP implementation",
        "pred_conditional_deep_implementation": "Conditional deep implementation",
        "score_pressure_reversal_implementation": "Pressure score implementation",
    }
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    colors = ["#315c72", "#8f4f4f", "#557a46", "#7a6f3d"]
    for color, (portfolio, label) in zip(colors, keep.items()):
        g = rets[rets["portfolio"].eq(portfolio)].sort_values("date")
        if g.empty:
            continue
        wealth = (1.0 + pd.to_numeric(g["long_short_ret"], errors="coerce").fillna(0.0)).cumprod()
        ax.plot(g["date"], wealth, label=label, linewidth=1.8, color=color)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Implementation-Universe Long-Short Signals")
    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(fig_dst / "readme_headline_result.png", dpi=190, bbox_inches="tight")
    fig.savefig(fig_dst / "readme_headline_result.svg", bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())
