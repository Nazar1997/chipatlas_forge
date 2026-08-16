#!/bin/bash
#SBATCH --job-name=ca_vocab
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/vocab_%j.out
#
# Decide the tissue and antigen vocabulary for a release. Seconds of work, but a
# separate stage because everything after it is shaped by what it picks.
#
# FREEZE_FEATURES pins the feature list to an existing release's. Leave it set:
# the omics head's output dimension is the feature count rounded to a power of
# two, so refreshing the vocabulary makes every existing checkpoint
# architecturally unloadable. Refreshing is a deliberate act with a retraining
# budget attached.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/06_vocab.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
if [[ -z "${FREEZE_FEATURES:-}" ]]; then
    if have_donor_release "$ORG"; then
        FREEZE_FEATURES="$DATA_DIR/$ORG/releases/$DONOR_RELEASE/index/features.json"
    else
        # The pre-release vocabulary, as a joblib pickle. Same 1,009 columns.
        FREEZE_FEATURES="$DATA_DIR/$ORG/SupportFiles/target_features.pkl"
    fi
fi
echo "host=$(hostname) org=$ORG release=$RELEASE started=$(date)"
freeze=()
[[ -n "$FREEZE_FEATURES" && -f "$FREEZE_FEATURES" ]] && freeze=(--freeze-features "$FREEZE_FEATURES")
$PY -m chipatlas_forge.vocab \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --meta-dir "$ROOT/meta" "${freeze[@]}"
echo "finished=$(date)"
