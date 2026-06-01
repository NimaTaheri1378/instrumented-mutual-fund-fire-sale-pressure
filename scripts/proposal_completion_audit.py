from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_SOURCE_FILES = [
    "README.md",
    "LICENSE",
    "environment.yml",
    "Dockerfile",
    "pyproject.toml",
    "mkdocs.yml",
    ".gitignore",
    ".github/workflows/ci.yml",
    ".github/workflows/pages.yml",
    "configs/project.yml",
    "configs/schema_map.yml",
    "configs/data_paths.yml",
    "configs/model_configs/main.yml",
    "sql/extract_crsp_mf.sql",
    "sql/extract_crsp_stock.sql",
    "sql/extract_ccm_comp.sql",
    "docs/index.md",
    "docs/methodology.md",
    "docs/results.md",
    "docs/runbook.md",
    "docs/release_checklist.md",
    "docs/claim_ledger.md",
    "docs/proposal_completion_audit.md",
    "scripts/public_safety_check.py",
    "scripts/proposal_completion_audit.py",
    "scripts/strategy_candidate_sweep.py",
    "scripts/001_smoke.py",
    "scripts/002_full_pipeline.py",
    "scripts/003_release_audit.py",
    "scripts/004_build_readme_assets.py",
    "jobs/run_stage_on_allocation.sh",
    "jobs/run_post_full_on_allocation.sh",
    "jobs/run_full_deep_optuna_on_allocation.sh",
    "src/mffiresale/__init__.py",
    "src/mffiresale/backtest.py",
    "src/mffiresale/config.py",
    "src/mffiresale/deliverables.py",
    "src/mffiresale/empirical.py",
    "src/mffiresale/extract.py",
    "src/mffiresale/features.py",
    "src/mffiresale/io.py",
    "src/mffiresale/models.py",
    "src/mffiresale/pipeline.py",
    "src/mffiresale/schema.py",
    "src/mffiresale/validate.py",
    "src/mffiresale/viz.py",
    "src/mffiresale/wrds_access.py",
    "tests/test_backtest_constraints.py",
    "tests/test_deliverable_horizons.py",
    "tests/test_linkage_and_returns.py",
    "tests/test_model_tuning_helpers.py",
    "tests/test_model_windows.py",
    "tests/test_timing_and_flows.py",
]

REQUIRED_NOTEBOOKS = [
    "notebooks/00_schema_audit.ipynb",
    "notebooks/10_feature_store.ipynb",
    "notebooks/20_baselines.ipynb",
    "notebooks/30_ml_models.ipynb",
    "notebooks/40_figures.ipynb",
]

REQUIRED_MANIFESTS = [
    "schema_audit.json",
    "extract_full.json",
    "features_full.json",
    "models_full.json",
    "backtest_full.json",
    "empirical_full.json",
    "figures_full.json",
    "deliverables_full.json",
    "validation_full.json",
]

REQUIRED_TABLES = [
    "data_construction_full.csv",
    "sample_summary_full.csv",
    "fama_macbeth_summary_full.csv",
    "fama_macbeth_extended_full.csv",
    "iv_2sls_summary_full.csv",
    "dml_iv_summary_full.csv",
    "model_horse_race_full.csv",
    "ablation_table_full.csv",
    "robustness_table_full.csv",
    "fund_filter_robustness_full.csv",
    "return_horizon_profile_full.csv",
    "moving_block_bootstrap_full.csv",
    "transaction_costs_full.csv",
    "lightgbm_optuna_trials_full.csv",
    "lightgbm_shap_summary_full.csv",
    "conditional_deep_pricing_errors_full.csv",
    "strategy_candidate_sweep_full.csv",
    "strategy_candidate_selected_full.csv",
]

REQUIRED_FIGURES = [
    "aggregate_predicted_pressure_full.png",
    "aggregate_predicted_pressure_full.svg",
    "cumulative_long_short_full.png",
    "cumulative_long_short_full.svg",
    "pressure_distribution_full.png",
    "pressure_distribution_full.svg",
    "liquidity_crowding_heatmap_full.png",
    "liquidity_crowding_heatmap_full.svg",
    "rolling_strategy_metrics_full.png",
    "rolling_strategy_metrics_full.svg",
    "lightgbm_feature_importance_full.png",
    "lightgbm_feature_importance_full.svg",
    "lightgbm_shap_summary_full.png",
    "lightgbm_shap_summary_full.svg",
    "event_study_extreme_pressure_full.png",
    "event_study_extreme_pressure_full.svg",
    "turnover_cost_frontier_full.png",
    "turnover_cost_frontier_full.svg",
    "drawdown_full.png",
    "drawdown_full.svg",
    "strategy_candidate_locked_test_full.png",
    "strategy_candidate_locked_test_full.svg",
]

REQUIRED_HTML = [
    "aggregate_predicted_pressure_full.html",
    "cumulative_long_short_full.html",
    "mechanism_spreads_full.html",
]

DELIVERABLE_ROW_MINIMUMS = {
    "model_horse_race": 12,
    "ablation_table": 9,
    "robustness_table": 13,
    "fund_filter_robustness": 4,
    "return_horizon_profile": 4,
    "moving_block_bootstrap": 12,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit proposal completion evidence.")
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--artifact-root", default="reports/remote_proposal_grade", help="downloaded artifact bundle")
    parser.add_argument("--write-csv", default=None, help="optional CSV report path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifact = (root / args.artifact_root).resolve()
    rows: list[dict[str, Any]] = []

    for rel in REQUIRED_SOURCE_FILES + REQUIRED_NOTEBOOKS:
        rows.append(_file_check(root / rel, "source", rel))
    for rel in REQUIRED_MANIFESTS:
        rows.append(_file_check(artifact / "manifests" / rel, "manifest", rel))
    for rel in REQUIRED_TABLES:
        rows.append(_file_check(artifact / "reports" / "tables" / rel, "table", rel, require_nonempty=True))
    for rel in REQUIRED_FIGURES:
        rows.append(_file_check(artifact / "reports" / "figures" / rel, "figure", rel, min_bytes=1000))
    for rel in REQUIRED_HTML:
        rows.append(_file_check(artifact / "reports" / "html" / rel, "html", rel, min_bytes=1000))

    rows.extend(_semantic_checks(artifact))

    if args.write_csv:
        out = root / args.write_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["category", "item", "status", "detail"])
            writer.writeheader()
            writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "pass"]
    for row in rows:
        print(f"{row['status']:>4} | {row['category']:<12} | {row['item']} | {row['detail']}")
    if failures:
        print(f"proposal_completion_audit: {len(failures)} failing checks", file=sys.stderr)
        return 1
    print(f"proposal_completion_audit: pass ({len(rows)} checks)")
    return 0


def _file_check(path: Path, category: str, item: str, require_nonempty: bool = False, min_bytes: int = 1) -> dict[str, Any]:
    if not path.exists():
        return {"category": category, "item": item, "status": "fail", "detail": "missing"}
    size = path.stat().st_size
    if size < min_bytes:
        return {"category": category, "item": item, "status": "fail", "detail": f"too small: {size} bytes"}
    if require_nonempty and path.suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            return {"category": category, "item": item, "status": "fail", "detail": f"unreadable csv: {type(exc).__name__}"}
        if df.empty:
            return {"category": category, "item": item, "status": "fail", "detail": "empty csv"}
        return {"category": category, "item": item, "status": "pass", "detail": f"rows={len(df)}"}
    return {"category": category, "item": item, "status": "pass", "detail": f"bytes={size}"}


def _semantic_checks(artifact: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation = _read_json(artifact / "manifests" / "validation_full.json")
    rows.append(_check("validation", "validation_full_status", validation.get("status") == "pass", validation.get("status", "missing")))
    for check in validation.get("checks", []):
        rows.append(_check("validation", f"validation:{check.get('check')}", bool(check.get("pass")), check.get("detail", "")))

    models = _read_json(artifact / "manifests" / "models_full.json")
    rows.append(_check("models", "lightgbm_optuna_trials", models.get("lightgbm", {}).get("optuna_tuning", {}).get("trials", 0) >= 1, models.get("lightgbm", {}).get("optuna_tuning", {})))
    rows.append(_check("models", "lightgbm_gpu_attempt", models.get("lightgbm", {}).get("gpu_used") is True, models.get("lightgbm", {}).get("gpu_used")))
    rows.append(_check("models", "torch_mlp_cuda", models.get("torch_mlp", {}).get("gpu_used") is True, models.get("torch_mlp", {}).get("device")))
    rows.append(_check("models", "conditional_deep_cuda", models.get("conditional_deep", {}).get("gpu_used") is True, models.get("conditional_deep", {}).get("device")))

    deliverables = _read_json(artifact / "manifests" / "deliverables_full.json")
    rows.append(_check("deliverables", "deliverable_table_count", len(deliverables.get("tables", [])) >= 8, len(deliverables.get("tables", []))))
    rows.append(_check("deliverables", "deliverable_figure_count", len(deliverables.get("figures", [])) >= 5, len(deliverables.get("figures", []))))
    rows.append(_check("deliverables", "deliverable_html_count", len(deliverables.get("html", [])) >= 3, len(deliverables.get("html", []))))
    table_rows = {table.get("name"): int(table.get("rows", 0)) for table in deliverables.get("tables", []) if "rows" in table}
    for name, minimum in DELIVERABLE_ROW_MINIMUMS.items():
        rows.append(_check("deliverables", f"{name}_rows", table_rows.get(name, 0) >= minimum, table_rows.get(name, 0)))

    returns_path = artifact / "data" / "processed" / "portfolio_returns_full.parquet"
    if returns_path.exists():
        rets = pd.read_parquet(returns_path)
        rows.append(_check("backtest", "portfolio_count", rets["portfolio"].nunique() >= 12, rets["portfolio"].nunique()))
        rows.append(_check("backtest", "turnover_cap", float(rets["turnover"].max()) <= 1.500001, float(rets["turnover"].max())))
        rows.append(_check("backtest", "max_name_weight", float(rets["max_abs_weight"].max()) <= 0.050001, float(rets["max_abs_weight"].max())))
    else:
        rows.append(_check("backtest", "portfolio_returns_full_parquet", False, "missing"))

    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check(category: str, item: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"category": category, "item": item, "status": "pass" if passed else "fail", "detail": str(detail)}


if __name__ == "__main__":
    raise SystemExit(main())
