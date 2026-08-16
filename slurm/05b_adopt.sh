#!/bin/bash
#SBATCH --job-name=ca_adopt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/adopt_%j.out
#
# Carry the peak pipeline's outputs into a release: the binmax signal tracks and
# groups.tsv. Hard links, so 18.8 GB costs nothing.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/05b_adopt.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE started=$(date)"
$PY -m chipatlas_forge.adopt \
    --root "$ROOT" --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --binmax "${BINMAX:-out_binmax1}"
echo "finished=$(date)"
