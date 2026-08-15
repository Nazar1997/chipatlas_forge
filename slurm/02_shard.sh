#!/bin/bash
#SBATCH --job-name=ca_shard
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/shard_%j.out
#
# The one serial stage: a plain-gzip stream cannot be inflated by more than one
# core (see shard.py). 6 CPUs, not 1, because pigz keeps reader and writer on
# their own threads and the zstd filter needs a core of its own.
#
# Submit one of these per organism -- they are independent and run concurrently.
#   ORG=hg38 sbatch slurm/02_shard.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG chunk=$CHUNK started=$(date)"
$PY -m chipatlas_forge.shard --root "$ROOT" --org "$ORG" --chunk-size "$CHUNK"
echo "finished=$(date)"
