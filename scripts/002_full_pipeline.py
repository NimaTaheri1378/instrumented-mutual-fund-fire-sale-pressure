from __future__ import annotations

import argparse

from mffiresale.pipeline import main


STAGES = [
    "schema-audit",
    "extract",
    "features",
    "models",
    "backtest",
    "empirical",
    "figures",
    "deliverables",
    "validate",
]


def run() -> int:
    parser = argparse.ArgumentParser(description="Run the full proposal-grade pipeline.")
    parser.add_argument("--sample", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    for stage in STAGES:
        code = main([stage, "--sample", args.sample])
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
