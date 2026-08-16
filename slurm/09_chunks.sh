#!/bin/bash
#SBATCH --job-name=ca_chunks
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/chunks_%A_%a.out
#
# Cut the signal into per-(tissue, chromosome) parquet. One task per tissue, so
# every bedGraph is read exactly once -- sharding by chromosome as well would
# read all 18.8 GB twenty-five times over.
#
# TASKS must match the array width, or tissues are skipped. There are 19 usable
# tissues on hg38 and 18 on mm10, so the array is small; the work is skewed
# (Blood is far larger than the rest) and that is fine, tasks are independent.
#
#   ORG=hg38 RELEASE=2026-08 TASKS=19 sbatch --array=0-18 slurm/09_chunks.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE task=$SLURM_ARRAY_TASK_ID started=$(date)"
$PY -m chipatlas_forge.chunks \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --what omics --task env --tasks "${TASKS:?set TASKS to the array width}"
echo "finished=$(date)"
