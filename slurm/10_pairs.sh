#!/bin/bash
#SBATCH --job-name=ca_pairs
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/pairs_%j.out
#
# IDF track weights and the omics-similarity pair index.
#
# The scoring pass is one big BLAS GEMM per block, so it wants cores rather than
# tasks; -c 16 lets OpenBLAS thread it. Held-out windows are excluded by reading
# the `split` column the intervals stage wrote, so this can no longer disagree
# with the window tables about what is held out.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/10_pairs.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE started=$(date)"
$PY -m chipatlas_forge.pairs \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --tissues ${PAIR_TISSUES:-"All cell types"} \
    --pair_bp "${PAIR_BP:-8192}"
echo "finished=$(date)"
