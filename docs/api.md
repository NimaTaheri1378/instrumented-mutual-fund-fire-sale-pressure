# API

The package entry point is:

```bash
python -m mffiresale.pipeline STAGE --sample smoke
python -m mffiresale.pipeline STAGE --sample full
```

Available stages are:

- `schema-audit`
- `extract`
- `features`
- `models`
- `backtest`
- `empirical`
- `figures`
- `deliverables`
- `validate`
- `all`

Use `configs/project.yml`, `configs/schema_map.yml`, `configs/data_paths.yml`, and `configs/model_configs/main.yml` for reproducible configuration.

SQL templates in `sql/` document the CRSP mutual-fund, CRSP stock, and CCM/Compustat extracts used by the Python extraction layer.
