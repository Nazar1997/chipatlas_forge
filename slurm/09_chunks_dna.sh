#!/bin/bash
#SBATCH --job-name=ca_chunks_dna
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/chunks_dna_%j.out
#
# The DNA half of stage 9, separate because it is a different shape of work: one
# serial pass that hard-links ~47,000 chunk files from the donor release. The
# reference does not change between releases, so this is a directory-entry
# update per chunk rather than 3.1 GB of slicing.
#
# Unset DONOR_RELEASE to cut the chunks out of sequence.pkl instead.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/09_chunks_dna.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE donor=${DONOR_RELEASE:-none} started=$(date)"
donor=()
if have_donor_release "$ORG"; then
    donor=(--from-release "$DONOR_RELEASE")
elif [[ -d "$DATA_DIR/$ORG/Subtables/dna" ]]; then
    # Links from the tree the running jobs are reading. Read-only on the source,
    # and a hard link, so this neither disturbs them nor duplicates 3 GB.
    donor=(--dna-dir "$DATA_DIR/$ORG/Subtables/dna")
fi
$PY -m chipatlas_forge.chunks \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    --what dna "${donor[@]}"
echo "finished=$(date)"
