#!/bin/bash
#SBATCH --job-name=ca_manifest
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/manifest_%j.out
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT"
echo "host=$(hostname) started=$(date)"
$PY -m chipatlas_forge.manifest --root "$ROOT" --org ${ORGS:-hg38 mm10}
echo "finished=$(date)"
