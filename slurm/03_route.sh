#!/bin/bash
#SBATCH --job-name=ca_route
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/route_%A_%a.out
#
# The wide stage: one task per shard, no coordination between them.
#
# The array is submitted wider than the shard count because that count is only
# known once stage 1 finishes; tasks past the end exit 0 with "nothing to do"
# (see route.resolve_shards). %100 throttles to 100 running at a time, which is
# what cpu-e-quick's 640 cores support at 4 CPUs each.
#
#   ORG=hg38 sbatch --array=0-199%100 slurm/03_route.sh
set -euo pipefail
# NOT `dirname "$0"`: sbatch copies the batch script to the node's spool
# directory, so inside a job $0 is /var/spool/slurm/d/jobNNNN/slurm_script and
# _env.sh is not beside it. run_all.sh exports FORGE_ENV; the fallback covers
# running this script by hand.
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG task=$SLURM_ARRAY_TASK_ID started=$(date)"
$PY -m chipatlas_forge.route \
    --root "$ROOT" --org "$ORG" --shard env --buckets "$BUCKETS"
echo "finished=$(date)"
