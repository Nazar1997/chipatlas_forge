#!/bin/bash
#SBATCH --job-name=ca_pantissue
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pantissue_%A_%a.out
#
# Derive the all-cell-types track: every antigen, max-merged across tissues.
#
# ChIP-Atlas's cell-type classes are real tissues, so nothing grouped from
# per-experiment metadata can produce this bucket -- but the 2021 tree had one,
# and it is what `only_one_tissue` runs train on and what the pair builder
# profiles against.
#
# Runs after 05b_adopt (it merges the adopted tracks) and after 07_genome (it
# needs chrom_sizes), and before 06_vocab, which lists it as a tissue.
#
#   ORG=hg38 RELEASE=2026-08 TASKS=40 sbatch --array=0-39 slurm/05c_pantissue.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE task=$SLURM_ARRAY_TASK_ID started=$(date)"
$PY -m chipatlas_forge.pantissue \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --task env --tasks "${TASKS:?set TASKS to the array width}"
echo "finished=$(date)"
