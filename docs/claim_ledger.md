# Claim Ledger

This ledger maps empirical claims to artifacts and states the wording discipline for a public release. It is deliberately conservative: the repository can report evidence, robustness, and portfolio diagnostics, but it should not convert descriptive or predictive evidence into causal certainty.

| Module | Evidence-bound claim | Evidence strength | Source artifacts | Allowed wording | Forbidden wording | Caveats and next gates |
|---|---|---:|---|---|---|---|
| Data construction | WRDS inputs are transformed into a stock-month panel with lagged mutual-fund pressure, CRSP returns, delisting returns, and Compustat controls. | High | `manifests/extract_full.json`, `manifests/features_full.json`, `reports/tables/data_construction_full.csv`, `reports/tables/sample_summary_full.csv` | The pipeline constructs a point-in-time stock-month panel using documented WRDS sources and lag rules. | The data are error-free or universally complete. | Recheck schema and sample coverage after any WRDS schema or access change. |
| Timing | Main holdings are activated after the configured public-information lag, with 90-day robustness. | High | `configs/project.yml`, `reports/tables/robustness_table_full.csv`, timing tests | The timing design uses a conservative holdings lag and reports a 90-day robustness check. | The timing rule eliminates every possible information concern. | Any post-2024 monthly-reporting appendix should be clearly labeled as an add-on. |
| Instrumented pressure | Forecastable fund-flow pressure has predictive and mechanism content in portfolio sorts and robustness tables. | Medium | `reports/tables/raw_vs_instrumented_spreads_full.csv`, `reports/tables/pressure_spread_summary_full.csv`, `reports/tables/robustness_table_full.csv` | The results are consistent with forecastable flow-driven pressure predicting subsequent reversal in several specifications. | Instrumented pressure proves causal mispricing or arbitrage. | Fama-MacBeth signs differ across controls; do not oversell monotonicity. |
| IV and DML | IV and DML specifications provide high-dimensional and instrumented robustness layers. | Medium | `reports/tables/iv_2sls_summary_full.csv`, `reports/tables/dml_iv_summary_full.csv`, `manifests/models_full.json` | IV and DML estimates provide robustness evidence under the stated shift-share-style design. | The design definitively identifies causal price pressure. | Interpret as robustness evidence unless additional exclusion and placebo tests are added. |
| ML horse race | Elastic Net, LightGBM, XGBoost, Torch MLP, and conditional deep models are compared out of sample. | High | `reports/tables/model_horse_race_full.csv`, `manifests/models_full.json`, prediction artifacts | The ML horse race compares sparse, tree, and deep predictors using chronological validation and locked test windows. | The best model is guaranteed to generalize to future regimes. | Treat ML results as predictive benchmarks, not the identification core. |
| LightGBM tuning and SHAP | LightGBM uses Optuna tuning and has SHAP-based interpretation artifacts. | High | `reports/tables/lightgbm_optuna_trials_full.csv`, `reports/tables/lightgbm_shap_summary_full.csv`, `reports/figures/lightgbm_shap_summary_full.*` | The production tree model has recorded tuning trials and feature-attribution diagnostics. | SHAP proves the true structural mechanism. | SHAP is a model explanation, not an economic identification proof. |
| Conditional deep appendix | A CUDA-backed conditional deep asset-pricing appendix model is implemented with pricing-error diagnostics. | Medium | `reports/tables/conditional_deep_pricing_errors_full.csv`, `manifests/models_full.json` | The appendix model learns characteristic-conditioned exposures and reports pricing-error diagnostics. | The deep model is a full no-arbitrage proof. | Treat as an appendix benchmark unless a formal SDF or IPCA validation is added. |
| Implementation realism | Portfolio tests report transaction costs, turnover, max-name-weight, and ADV-capacity diagnostics. | High | `reports/tables/portfolio_summary_full.csv`, `reports/tables/transaction_costs_full.csv`, `data/processed/portfolio_returns_full.parquet` | Portfolio results are evaluated with explicit turnover, cost, and capacity diagnostics. | The strategies are directly tradable at institutional scale. | Capacity is a proxy; small research-universe capacity estimates require caution. |
| Strategy-forward lead | A validation-selected implementation strategy sweep does not yet produce a sharp locked-test strategy result. | Low | `reports/tables/strategy_candidate_sweep_full.csv`, `reports/tables/strategy_candidate_selected_full.csv`, `reports/figures/strategy_candidate_locked_test_full.*` | The current implementation sweep is useful as a diagnostic and shows where candidate strategies weaken out of sample. | The project has a validated high-return implementable trading strategy. | The selected strategy fades in the locked test after costs; ex-post stronger candidates require a new validation design before being promoted. |
| Robustness | Results include timing, subperiod, liquidity/crowding, fund-filter, horizon, bootstrap, and placebo-style comparisons. | High | `reports/tables/robustness_table_full.csv`, `reports/tables/fund_filter_robustness_full.csv`, `reports/tables/return_horizon_profile_full.csv`, `reports/tables/moving_block_bootstrap_full.csv`, `reports/tables/ablation_table_full.csv` | The release includes a broad robustness and ablation package. | All robustness tests uniformly support the same conclusion. | Report mixed signs and weak t-statistics directly. |
| Visuals | Figure outputs exist in PNG/SVG and selected Plotly HTML formats and have been visually reviewed. | High | `reports/figures/*.png`, `reports/figures/*.svg`, `reports/html/*.html` | The project ships publication-oriented static and interactive visuals. | Figures alone establish economic significance. | Re-run visual QA whenever labels, models, or plotted series change. |
| Public safety | The releasable tree excludes credentials, private allocation IDs, raw WRDS data, logs, and generated result bundles. | High | `scripts/public_safety_check.py`, `.gitignore`, CI workflow | Public-release guardrails are enforced by a scanner and CI. | The scanner catches every possible secret. | Run a manual review before any push. |

## Release Wording Guardrails

Use:

- "consistent with temporary price pressure";
- "forecastable fund-flow pressure predicts subsequent reversal";
- "survives these robustness layers";
- "implementation-aware diagnostics show turnover, cost, and capacity constraints."
- "the validation-selected implementable strategy does not yet support a strategy-forward lead claim."

Avoid:

- "proves fire-sale causality";
- "arbitrage opportunity";
- "fully tradable at scale";
- "validated high-return strategy";
- "orthogonal to all fundamentals";
- "no-arbitrage model proves the mechanism."
