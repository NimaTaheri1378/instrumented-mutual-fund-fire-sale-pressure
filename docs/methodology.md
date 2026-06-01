# Methodology

The project uses a conservative public-information design.

- Mutual-fund holdings are activated only after a uniform 60-day lag in the main specification.
- Fund flows are computed from TNA and monthly returns, then forecast with only lagged public fund-level, style-level, and family-level information.
- Stock-level pressure is the holdings-weighted sum of predicted fund flows.
- Monthly stock returns include CRSP delisting returns.
- Compustat controls use a six-month accounting availability lag.
- Prediction windows are calendar ordered: development through 2014, validation from 2015 to 2018, and locked test results from 2019 onward.

The empirical hierarchy is descriptive sorts, Fama-MacBeth regressions, 2SLS, DML, sparse prediction, Optuna-tuned LightGBM, SHAP interpretation, and CUDA-backed deep benchmarks. The appendix layer includes a conditional deep asset-pricing model with learned characteristic betas and train-window latent factor premia.

Portfolio tests are market-neutral and rank-based. The backtest neutralizes scores by sector, beta, and size, then applies max-name-weight and turnover-cap constraints before computing factor alphas, transaction-cost variants, turnover, and ADV-based capacity.
