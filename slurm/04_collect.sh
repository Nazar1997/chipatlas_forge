#!/bin/bash
#SBATCH --job-name=ca_collect
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/collect_%A_%a.out
#
# One task per bucket. --buckets MUST match what stage 2 ran with, or a task
# will find rows for a group that does not hash to it and abort loudly rather
# than write a half-empty BED.
#
#   ORG=hg38 sbatch --array=0-127%100 slurm/04_collect.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG task=$SLURM_ARRAY_TASK_ID started=$(date)"
$PY -m chipatlas_forge.collect \
    --root "$ROOT" --org "$ORG" --bucket env --buckets "$BUCKETS" \
    --compress "${COMPRESS:-none}"
echo "finished=$(date)"
