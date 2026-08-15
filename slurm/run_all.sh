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
# Submitted before the shard count is known, so it has to be an upper bound:
# tasks past the end exit 0 immediately (route.resolve_shards). Too LOW is the
# dangerous direction -- shards silently never routed and BEDs quietly short --
# but too high is not free either, because **every pending array task counts
# against the QOS submit limit** (MaxSubmitJobsPU=500 on `normal` here). A
# 401-wide array consumed the whole budget on the first attempt and the collect
# array simply failed to submit.
#
# At CHUNK=2G the current archives need ~60 shards for hg38 and ~43 for mm10, so
# 80 leaves real margin and keeps the whole plan inside the cap. collect refuses
# to run if any shard was missed, so an undersized value fails loudly rather
# than truncating.
MAX_SHARDS="${MAX_SHARDS:-80}"
THROTTLE="${THROTTLE:-50}"

cd "$ROOT"

# --- stay inside the QOS submit limit -------------------------------------
# Exceeding it does not queue, it *rejects*, and sbatch keeps going -- so a
# chain can end up with its first jobs submitted and its last ones missing,
# which looks like a pipeline that ran and produced nothing.
planned=$(( 2 + (MAX_SHARDS + 1) + BUCKETS ))
limit=$(sacctmgr -n show qos normal format=MaxSubmitJobsPU 2>/dev/null | tr -d ' ')
in_queue=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
if [[ -n "$limit" && "$limit" =~ ^[0-9]+$ ]]; then
    if (( in_queue + planned > limit )); then
        echo "refusing to submit: $planned jobs on top of $in_queue already queued" >&2
        echo "would exceed the QOS limit of $limit." >&2
        echo "Wait for the queue to drain, or lower BUCKETS / MAX_SHARDS." >&2
        exit 1
    fi
    echo "submit budget: $planned planned + $in_queue queued of $limit"
fi

manifest=$(sbatch --parsable --partition="$PARTITION" \
    --export=ALL,FORGE_ENV="$HERE/_env.sh",ORGS="$ORG" "$HERE/01_manifest.sh")
echo "manifest : $manifest"

shard=$(sbatch --parsable --partition="$PARTITION" \
    --export=ALL,FORGE_ENV="$HERE/_env.sh",ORG="$ORG",CHUNK="$CHUNK" "$HERE/02_shard.sh")
echo "shard    : $shard"

route=$(sbatch --parsable --partition="$PARTITION" \
    --dependency="afterok:$manifest:$shard" \
    --array="0-${MAX_SHARDS}%${THROTTLE}" \
    --export=ALL,FORGE_ENV="$HERE/_env.sh",ORG="$ORG",BUCKETS="$BUCKETS" "$HERE/03_route.sh")
echo "route    : $route  (array 0-${MAX_SHARDS}%${THROTTLE})"

collect=$(sbatch --parsable --partition="$PARTITION" \
    --dependency="afterok:$route" \
    --array="0-$((BUCKETS - 1))%${THROTTLE}" \
    --export=ALL,FORGE_ENV="$HERE/_env.sh",ORG="$ORG",BUCKETS="$BUCKETS",COMPRESS="${COMPRESS:-none}" \
    "$HERE/04_collect.sh")
echo "collect  : $collect  (array 0-$((BUCKETS - 1))%${THROTTLE})"

echo
echo "watch:   squeue -u $USER"
echo "report:  $PY -m chipatlas_forge.report --root $ROOT --org $ORG"
