from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ProjectPaths
from .io import manifest_base, write_json


def run_validation(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    panel_path = paths.processed / f"model_panel_{sample}.parquet"
    if not panel_path.exists():
        checks.append(_check("model_panel_exists", False, "Missing model panel"))
    else:
        panel = pd.read_parquet(panel_path)
        checks.extend(_panel_checks(panel))
    ret_path = paths.processed / f"portfolio_returns_{sample}.parquet"
    checks.append(_check("portfolio_returns_exists", ret_path.exists(), str(ret_path)))
    if ret_path.exists():
        rets = pd.read_parquet(ret_path)
        checks.append(_check("portfolio_returns_nonempty", len(rets) > 0, f"rows={len(rets)}"))
    status = "pass" if all(x["pass"] for x in checks) else "fail"
    manifest = manifest_base(paths.root, "validation", sample=sample, status=status, checks=checks)
    write_json(paths.manifests / f"validation_{sample}.json", manifest)
    pd.DataFrame(checks).to_csv(paths.tables / f"validation_checks_{sample}.csv", index=False)
    return manifest


def _panel_checks(panel: pd.DataFrame) -> list[dict[str, Any]]:
    checks = []
    checks.append(_check("model_panel_nonempty", len(panel) > 0, f"rows={len(panel)}"))
    if len(panel):
        dupes = panel.duplicated(["permno", "date"]).sum()
        checks.append(_check("unique_permno_month", dupes == 0, f"duplicates={dupes}"))
        checks.append(_check("has_next_return", panel["next_ret"].notna().sum() > 0, f"nonmissing={panel['next_ret'].notna().sum()}"))
        for col in ["pressure_pred_z", "score_pressure_reversal", "mktcap_lag1"]:
            checks.append(_check(f"has_{col}", col in panel.columns and panel[col].notna().sum() > 0, f"nonmissing={panel[col].notna().sum() if col in panel else 0}"))
    return checks


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "pass": bool(passed), "detail": detail}
