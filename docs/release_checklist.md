# Release Checklist

Before publishing, run the checks below from the project root:

```bash
python scripts/public_safety_check.py
pytest -q
mkdocs build --strict
```

When a downloaded empirical artifact bundle is present, also run:

```bash
python scripts/proposal_completion_audit.py
```

The completion audit is documented in `docs/proposal_completion_audit.md`. It checks the source package, manifests, table families, visual families, GPU/tuning evidence, validation status, and portfolio implementation constraints.

The public repository should contain code, configuration, documentation, SQL templates, tests, notebooks, and workflow files. It should not contain WRDS-derived raw/interim/processed data, generated logs, generated manifests, model artifacts, downloaded result bundles, credentials, `.pgpass`, private server paths, or allocation identifiers.

Empirical release artifacts should be regenerated from cached WRDS extracts on an approved compute allocation, then reviewed locally for:

- passing validation manifests;
- no duplicate `permno`-month observations;
- lagged holdings only after the configured public-information delay;
- model predictions produced from chronological train/validation/test windows;
- constrained portfolio turnover and max-name-weight diagnostics;
- readable PNG/SVG/HTML figures with non-overlapping labels;
- tables covering construction, sample summary, Fama-MacBeth, IV/DML, model horse race, ablation, robustness, fund filters, return horizons, bootstrap, and transaction costs.

Only after the safety check, tests, docs build, validation, and visual review pass should the branch be committed and pushed.
