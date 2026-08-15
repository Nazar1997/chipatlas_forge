#!/bin/bash
#
# Submit the whole pipeline for one organism as a dependency chain, then exit.
# Nothing here blocks -- check progress with `squeue -u $USER` or `logs/`.
#
#   ORG=hg38 ./slurm/run_all.sh
#   ORG=mm10 BUCKETS=128 COMPRESS=gzip ./slurm/run_all.sh
#
# manifest and shard are independent (one reads meta/, the other raw/), so they
# are submitted without a dependency between them and run at the same time.
# route waits on both; collect waits on route.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
MAX_SHARDS="${MAX_SHARDS:-199}"
THROTTLE="${THROTTLE:-100}"

cd "$ROOT"

manifest=$(sbatch --parsable --partition="$PARTITION" \
    --export=ALL,ORGS="$ORG" "$HERE/01_manifest.sh")
echo "manifest : $manifest"

shard=$(sbatch --parsable --partition="$PARTITION" \
    --export=ALL,ORG="$ORG",CHUNK="$CHUNK" "$HERE/02_shard.sh")
echo "shard    : $shard"

route=$(sbatch --parsable --partition="$PARTITION" \
    --dependency="afterok:$manifest:$shard" \
    --array="0-${MAX_SHARDS}%${THROTTLE}" \
    --export=ALL,ORG="$ORG",BUCKETS="$BUCKETS" "$HERE/03_route.sh")
echo "route    : $route  (array 0-${MAX_SHARDS}%${THROTTLE})"

collect=$(sbatch --parsable --partition="$PARTITION" \
    --dependency="afterok:$route" \
    --array="0-$((BUCKETS - 1))%${THROTTLE}" \
    --export=ALL,ORG="$ORG",BUCKETS="$BUCKETS",COMPRESS="${COMPRESS:-none}" \
    "$HERE/04_collect.sh")
echo "collect  : $collect  (array 0-$((BUCKETS - 1))%${THROTTLE})"

echo
echo "watch:   squeue -u $USER"
echo "report:  $PY -m chipatlas_forge.report --root $ROOT --org $ORG"
