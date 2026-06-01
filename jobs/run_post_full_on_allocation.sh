#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RUNNER="${PROJECT_ROOT}/jobs/run_stage_on_allocation.sh"

export PROJECT_ROOT
cd "${PROJECT_ROOT}"

bash "${RUNNER}" models full
bash "${RUNNER}" backtest full
bash "${RUNNER}" empirical full
bash "${RUNNER}" figures full
bash "${RUNNER}" deliverables full
bash "${RUNNER}" validate full
