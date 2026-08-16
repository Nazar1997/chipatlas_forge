#!/bin/bash
#SBATCH --job-name=ca_verify
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/verify_%j.out
#
# Prove the release is complete before promoting it. Exits non-zero on any
# problem, so `sbatch --dependency=afterok:` on this is what gates promotion.
#
# The sampled check re-derives chunks straight from signal/ and compares them
# row for row; it is the only check that can catch a wrong value rather than a
# missing file, and the only expensive one.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/11_verify.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE started=$(date)"
$PY -m chipatlas_forge.verify \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --sample "${SAMPLE:-100}"
echo "finished=$(date)"
