#!/bin/bash
#SBATCH --job-name=ca_manifest
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/manifest_%j.out
set -euo pipefail
# NOT `dirname "$0"`: sbatch copies the batch script to the node's spool
# directory, so inside a job $0 is /var/spool/slurm/d/jobNNNN/slurm_script and
# _env.sh is not beside it. run_all.sh exports FORGE_ENV; the fallback covers
# running this script by hand.
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
echo "host=$(hostname) started=$(date)"
$PY -m chipatlas_forge.manifest --root "$ROOT" --org ${ORGS:-hg38 mm10}
echo "finished=$(date)"
