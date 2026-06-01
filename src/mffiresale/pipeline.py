from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backtest import run_backtests
from .config import ProjectPaths, find_project_root, load_config
from .extract import extract_all
from .features import build_all_features
from .empirical import run_empirical_suite
from .deliverables import build_deliverables
from .models import run_all_models
from .schema import run_schema_audit
from .validate import run_validation
from .viz import make_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mutual-fund fire-sale pressure pipeline")
    parser.add_argument(
        "stage",
        choices=[
            "schema-audit",
            "smoke",
            "extract",
            "features",
            "models",
            "backtest",
            "empirical",
            "figures",
            "deliverables",
            "validate",
            "all",
        ],
    )
    parser.add_argument("--sample", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else find_project_root()
    cfg = load_config(root)
    paths = ProjectPaths.from_config(cfg, root)
    paths.ensure()

    result: dict[str, Any]
    if args.stage == "schema-audit":
        result = run_schema_audit(cfg, paths)
    elif args.stage == "smoke":
        result = run_all(cfg, paths, "smoke")
    elif args.stage == "extract":
        result = extract_all(cfg, paths, args.sample)
    elif args.stage == "features":
        result = build_all_features(cfg, paths, args.sample)
    elif args.stage == "models":
        result = run_all_models(cfg, paths, args.sample)
    elif args.stage == "backtest":
        result = run_backtests(cfg, paths, args.sample)
    elif args.stage == "empirical":
        result = run_empirical_suite(cfg, paths, args.sample)
    elif args.stage == "figures":
        result = make_figures(cfg, paths, args.sample)
    elif args.stage == "deliverables":
        result = build_deliverables(cfg, paths, args.sample)
    elif args.stage == "validate":
        result = run_validation(cfg, paths, args.sample)
    elif args.stage == "all":
        result = run_all(cfg, paths, args.sample)
    else:
        raise AssertionError(args.stage)
    print(json.dumps(_compact(result), indent=2, default=str))
    return 0


def run_all(cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    return {
        "schema_audit": run_schema_audit(cfg, paths),
        "extract": extract_all(cfg, paths, sample),
        "features": build_all_features(cfg, paths, sample),
        "models": run_all_models(cfg, paths, sample),
        "backtest": run_backtests(cfg, paths, sample),
        "empirical": run_empirical_suite(cfg, paths, sample),
        "figures": make_figures(cfg, paths, sample),
        "deliverables": build_deliverables(cfg, paths, sample),
        "validation": run_validation(cfg, paths, sample),
    }


def _compact(obj: Any) -> Any:
    if isinstance(obj, dict):
        compacted = {}
        for key, value in obj.items():
            if key in {"libraries", "columns"} and isinstance(value, list) and len(value) > 20:
                compacted[key] = value[:20] + [f"... {len(value) - 20} more"]
            else:
                compacted[key] = _compact(value)
        return compacted
    if isinstance(obj, list):
        if len(obj) > 12:
            return [_compact(x) for x in obj[:12]] + [f"... {len(obj) - 12} more"]
        return [_compact(x) for x in obj]
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
