#!/bin/bash
#SBATCH --job-name=ca_intervals
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/intervals_%j.out
#
# The chunk grid, the window tables, and the train/val/test split.
#
# Serial and single-node: it loads sequence.pkl once (~3 GB resident for hg38)
# to find the N runs, and every window size is then a cheap pass over the same
# run arrays. Splitting it across tasks would mean loading that pickle N times.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/08_intervals.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE started=$(date)"
$PY -m chipatlas_forge.intervals \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --windows ${WINDOWS:-8192 65536 1048576} \
    --val-chroms ${VAL_CHROMS:-chr8 chr9} \
    --test-fraction "${TEST_FRACTION:-0.1}"
echo "finished=$(date)"
