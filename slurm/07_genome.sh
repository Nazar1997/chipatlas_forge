#!/bin/bash
#SBATCH --job-name=ca_genome
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/genome_%j.out
#
# Adopt the reference sequence and the blacklist into a release, and derive the
# chromosome lengths from the FASTA rather than from a hardcoded table.
#
# Hard links, so this costs nothing even though the two files are 6 GB together.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/07_genome.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE donor=$DONOR_RELEASE started=$(date)"
$PY -m chipatlas_forge.genome \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --from-release "$DONOR_RELEASE"
echo "finished=$(date)"
