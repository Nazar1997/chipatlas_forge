#!/bin/bash
#SBATCH --job-name=ca_binmax
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/binmax_%A_%a.out
#
# Collapse overlapping peaks into a max-score signal track. Reads out/, writes
# out_binmax<bin>/ -- so it never touches the peak BEDs and can be rerun at a
# different resolution without redoing stages 1-4.
#
# Work is split by file, strided over a size-sorted list so the handful of
# multi-gigabyte files land in different tasks. TASKS must match the array width
# or files will be skipped.
#
#   ORG=hg38 TASKS=100 sbatch --array=0-99%50 slurm/05_binmax.sh
set -euo pipefail
# NOT `dirname "$0"`: sbatch copies the batch script to the node's spool
# directory, so inside a job $0 is /var/spool/slurm/d/jobNNNN/slurm_script and
# _env.sh is not beside it. run_all.sh exports FORGE_ENV; the fallback covers
# running this script by hand.
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG task=$SLURM_ARRAY_TASK_ID started=$(date)"
$PY -m chipatlas_forge.binmax \
    --root "$ROOT" --org "$ORG" --bin-size "${BIN_SIZE:-1}" \
    --task env --tasks "${TASKS:?set TASKS to the array width}"
echo "finished=$(date)"
