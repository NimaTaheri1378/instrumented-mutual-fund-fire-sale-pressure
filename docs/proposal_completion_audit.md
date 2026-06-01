# Proposal Completion Audit

This page records what the release bundle is expected to prove before any public push. It is not a manuscript section. It is a reproducibility and claim-control checklist for the code, empirical artifacts, tables, figures, and safety guardrails.

## Completion Gates

| Gate | Required evidence | Current release standard |
|---|---|---|
| Source package | Package modules, tests, configs, SQL templates, notebooks, docs, CI, and allocation job scripts are present. | `scripts/proposal_completion_audit.py` checks the expected source surface. |
| WRDS construction | Schema audit, extraction, and feature manifests exist for the full sample. | `manifests/schema_audit.json`, `extract_full.json`, and `features_full.json`. |
| Timing discipline | Holdings and controls are lagged before prediction and regression targets. | Unit tests plus robustness tables for the main lag and 90-day lag variant. |
| Empirical specifications | Fama-MacBeth, IV, DML, robustness, ablation, fund-filter, horizon, and bootstrap tables are nonempty. | Checked by required table files and row-count gates. |
| ML and GPU models | Elastic Net, LightGBM, XGBoost, Torch MLP, and conditional deep models are represented in the horse race. | `models_full.json` records CUDA use for the neural models and Optuna trials for LightGBM. |
| Portfolio realism | Portfolio output includes turnover, max-name-weight, transaction-cost, and capacity diagnostics. | `portfolio_returns_full.parquet` must pass turnover and weight caps. |
| Strategy readiness | Validation-selected implementation strategy diagnostics are present and separated from ex-post exploratory candidates. | `strategy_candidate_sweep_full.csv`, `strategy_candidate_selected_full.csv`, and the locked-test figure. |
| Visual QA | PNG/SVG figures and selected interactive HTML files exist and are nontrivial in size. | The audit checks files; manual visual review remains required after plot changes. |
| Public safety | Credentials, WRDS data, private paths, logs, generated bundles, and allocation identifiers are excluded. | `scripts/public_safety_check.py` and `.gitignore` enforce the first pass. |
| Publishing | No GitHub push happens until explicitly approved. | This gate is intentionally outside automation and remains user-controlled. |

## Manifest Catalog

The full proposal run is expected to leave these manifests in the artifact root:

| Manifest | Purpose |
|---|---|
| `schema_audit.json` | WRDS library/table/column visibility and schema checks. |
| `extract_full.json` | Data extraction status, source coverage, and cached output references. |
| `features_full.json` | Feature-panel construction status and row counts. |
| `models_full.json` | Regression, IV/DML, ML, tuning, GPU, and prediction artifacts. |
| `backtest_full.json` | Portfolio-count and return-file summary. |
| `empirical_full.json` | Main empirical table construction status. |
| `figures_full.json` | Static and interactive figure construction status. |
| `deliverables_full.json` | Manuscript-facing table, figure, and HTML inventory. |
| `validation_full.json` | Final panel, duplicate-key, return, signal, and portfolio validation checks. |

## Table Inventory

The audit requires core tables for construction, summary statistics, Fama-MacBeth, IV, DML, model comparison, ablation, robustness, fund filters, return horizons, moving-block bootstrap, transaction costs, SHAP, Optuna tuning, and conditional-deep pricing diagnostics. Row-count gates are intentionally conservative: they check that expected table families are present without treating mixed empirical signs as failures.

## Allowed Claims

The release can say that the project builds a point-in-time WRDS/CRSP/Compustat panel, estimates instrumented mutual-fund fire-sale pressure, compares linear/tree/deep out-of-sample predictors, and evaluates implementation-aware portfolio diagnostics. It can also say the evidence is consistent with temporary price pressure and subsequent reversal in the tested specifications.

The release should not say that the evidence proves causal fire-sale mispricing, establishes a directly tradable arbitrage, or eliminates all information-timing concerns. Those are manuscript-level claims that would require additional identification and execution evidence.

## How To Run

With a downloaded artifact bundle:

```bash
python scripts/proposal_completion_audit.py --artifact-root reports/remote_proposal_grade
```

On the Amarel compute allocation after a full run, use the project root as the artifact root:

```bash
python scripts/proposal_completion_audit.py --artifact-root .
```

The audit is deliberately loud: any missing required file, nonpassing manifest gate, missing table family, missing visual family, failed GPU/tuning evidence gate, or failed portfolio constraint exits nonzero.
