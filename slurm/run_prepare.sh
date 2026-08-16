#!/bin/bash
# Build a release from finished peak output, and gate promotion on verification.
#
#   ORG=hg38 RELEASE=2026-08 slurm/run_prepare.sh
#   ORG=hg38 RELEASE=2026-08 PROMOTE=1 slurm/run_prepare.sh
#
# Stages 0-5 (run_all.sh) must have finished first -- this starts from
# out_binmax1/. Everything is chained with --dependency=afterok, so a stage that
# fails stops the ones after it rather than building on a half-written release.
#
# Promotion is deliberately opt-in and deliberately last. `latest` is what every
# training job resolves, so moving it is the single action that changes what
# runs read; it happens only after verify has exited zero.
set -euo pipefail
export FORGE_ENV="$(cd "$(dirname "$0")" && pwd)/_env.sh"
source "$FORGE_ENV"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"

# One task per tissue. Read from the donor's vocabulary, because this release's
# own tissues.json does not exist until stage 06 runs -- and the count only
# changes when the vocabulary is refreshed, which is not what this pipeline does.
TISSUE_JSON="$DATA_DIR/$ORG/releases/${DONOR_RELEASE}/index/tissues.json"
TISSUE_PKL="$DATA_DIR/$ORG/SupportFiles/target_tissues.pkl"
if [[ -f "$TISSUE_JSON" ]]; then
    N_TISSUES=$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tissues']))" "$TISSUE_JSON")
elif [[ -f "$TISSUE_PKL" ]]; then
    N_TISSUES=$($PY -c "from joblib import load;import sys;print(len(load(sys.argv[1])))" "$TISSUE_PKL")
else
    N_TISSUES="${N_TISSUES:-24}"
fi

# Every pending array task counts against MaxSubmitJobsPU, and a rejected
# submission does not stop sbatch -- stage 09's array vanished silently that way
# once already. Check before submitting rather than after.
planned=$(( 7 + N_TISSUES ))
limit=$(sacctmgr -n show qos normal format=MaxSubmitJobsPU | tr -d ' ')
in_queue=$(squeue -u "$USER" -h -r | wc -l)
if [[ -n "$limit" ]] && (( in_queue + planned > limit )); then
    echo "refusing to submit: $in_queue queued + $planned planned > $limit allowed" >&2
    exit 1
fi

echo "org=$ORG release=$RELEASE donor=$DONOR_RELEASE tissues=$N_TISSUES"
submit() { sbatch --parsable "$@"; }

adopt=$(ORG=$ORG   submit slurm/05b_adopt.sh)
vocab=$(ORG=$ORG   submit --dependency=afterok:$adopt  slurm/06_vocab.sh)
genome=$(ORG=$ORG  submit --dependency=afterok:$vocab  slurm/07_genome.sh)
ivals=$(ORG=$ORG   submit --dependency=afterok:$genome slurm/08_intervals.sh)
dna=$(ORG=$ORG     submit --dependency=afterok:$ivals  slurm/09_chunks_dna.sh)
omics=$(ORG=$ORG TASKS=$N_TISSUES submit --dependency=afterok:$ivals \
        --array=0-$((N_TISSUES - 1)) slurm/09_chunks.sh)
pairs=$(ORG=$ORG   submit --dependency=afterok:$omics  slurm/10_pairs.sh)
verify=$(ORG=$ORG  submit --dependency=afterok:$dna:$omics:$pairs slurm/11_verify.sh)

echo "  05b adopt     $adopt"
echo "  06  vocab     $vocab"
echo "  07  genome    $genome"
echo "  08  intervals $ivals"
echo "  09  dna       $dna"
echo "  09  omics     $omics  (array 0-$((N_TISSUES - 1)))"
echo "  10  pairs     $pairs"
echo "  11  verify    $verify"

if [[ "${PROMOTE:-0}" == "1" ]]; then
    promote=$(submit --dependency=afterok:$verify --job-name=ca_promote \
        --time=0:05:00 --output=logs/promote_%j.out \
        --wrap "$PY -c \"import sys; sys.path.insert(0,'$ROOT'); \
from chipatlas_forge import layout; \
print(layout.promote('$DATA_DIR', '$ORG', '$RELEASE'))\"")
    echo "  --  promote   $promote  (only if verify exits 0)"
else
    echo
    echo "not promoting. After verify passes:"
    echo "  $PY -c \"from chipatlas_forge import layout; layout.promote('$DATA_DIR','$ORG','$RELEASE')\""
fi
