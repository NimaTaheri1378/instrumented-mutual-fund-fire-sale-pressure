# Instrumented Mutual-Fund Fire-Sale Pressure

This repository builds a point-in-time WRDS pipeline that maps mutual-fund flows through lagged portfolio holdings into stock-level fire-sale pressure signals, then evaluates those signals with asset-pricing regressions, IV/DML estimates, machine-learning prediction, and implementation-aware long-short portfolios.

The main full-sample artifacts are written under `reports/figures`, `reports/tables`, `reports/html`, and `manifests`. Raw WRDS extracts and credentials are intentionally excluded from version control.

## Reproduction

Run stages through the configured research environment:

```bash
python -m mffiresale.pipeline schema-audit --sample full
python -m mffiresale.pipeline all --sample full
```

On Amarel, use the SLURM wrapper in `jobs/run_stage_on_allocation.sh` so extraction, model fitting, and validation run on the compute allocation rather than a login node.
