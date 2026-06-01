#!/usr/bin/env bash
set -euo pipefail

sample="${1:-full}"
python_bin="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p logs

for stage in models backtest empirical figures deliverables validate; do
  ts="$(date -u +%Y%m%d_%H%M%S)"
  log="logs/${stage}_${sample}_deep_optuna_${ts}.log"
  echo "RUNNING ${stage} ${sample} ${ts}"
  "${python_bin}" -m mffiresale.pipeline "${stage}" --sample "${sample}" >"${log}" 2>&1
  tail -n 40 "${log}"
done
