#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-smoke}"
SAMPLE="${2:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

mkdir -p logs manifests
LOG="logs/${STAGE}_${SAMPLE}_$(date +%Y%m%d_%H%M%S).log"

echo "stage=${STAGE} sample=${SAMPLE}" | tee "${LOG}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cuda=${CUDA_VISIBLE_DEVICES:-none}" | tee -a "${LOG}"
"${PYTHON_BIN}" -m mffiresale.pipeline "${STAGE}" --sample "${SAMPLE}" --root "${PROJECT_ROOT}" 2>&1 | tee -a "${LOG}"
