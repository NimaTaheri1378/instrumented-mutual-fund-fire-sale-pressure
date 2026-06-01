# Results

The refreshed full run validates the core panel and writes the proposal-grade tables and figures:

- model and prediction manifests in `manifests`;
- portfolio summaries, IV/DML, Fama-MacBeth, ablation, robustness, fund-filter, return-horizon, Optuna tuning, conditional-deep diagnostics, and transaction-cost tables in `reports/tables`;
- PNG/SVG figures in `reports/figures`;
- interactive Plotly HTML outputs in `reports/html`.

Interpretation should remain evidence-bound. The raw pressure-reversal sort is positive in the full sample, while controlled IV/DML and some Fama-MacBeth specifications can differ in sign. Those differences are part of the empirical record, not something to smooth away.

## Strategy Readiness

The implementation-strategy evidence is not yet sharp enough to lead a public paper claim. A validation-selected candidate sweep chooses the best implementation-universe sort using 2015-2018 only, then evaluates that rule on the locked 2019-2024 test window. The selected candidate is a 5% long-short sort on a fixed XGBoost-plus-pressure blend without ex-post test-window tuning.

Selected validation result:

- gross annualized mean: 14.1%;
- gross Sharpe: 1.11;
- 25 bps net annualized mean: 10.4%;
- FF5+UMD monthly alpha t-statistic: 3.46.

Locked-test result:

- gross annualized mean: 4.8%;
- gross Sharpe: 0.23;
- 25 bps net annualized mean: 0.9%;
- 50 bps net annualized mean: -3.0%;
- FF5+UMD monthly alpha t-statistic: 0.80;
- max drawdown: -48.1%.

Ex post, some test-window candidates look better, especially a LightGBM 5% no-neutralization sort, but those were not selected by the validation rule. They should be treated as exploratory leads, not as publishable strategy evidence.
