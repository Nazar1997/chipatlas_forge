#!/bin/bash
#SBATCH --job-name=ca_genome
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/genome_%j.out
#
# Adopt the reference sequence and the blacklist into a release, and derive the
# chromosome lengths from the FASTA rather than from a hardcoded table.
#
# Hard links, so this costs nothing even though the two files are 6 GB together.
#
#   ORG=hg38 RELEASE=2026-08 sbatch slurm/07_genome.sh
set -euo pipefail
source "${FORGE_ENV:-$(cd "$(dirname "$0")" && pwd)/_env.sh}"
cd "$ROOT"
ORG="${ORG:?set ORG=hg38 or ORG=mm10}"
echo "host=$(hostname) org=$ORG release=$RELEASE donor=$DONOR_RELEASE started=$(date)"
if have_donor_release "$ORG"; then
    source_args=(--from-release "$DONOR_RELEASE")
else
    source_args=(--fasta     "$DATA_DIR/$ORG/DNA/$ORG.fa"
                 --blacklist "$DATA_DIR/$ORG/DNA/$ORG-blacklist.v2.bed"
                 --sequence  "$DATA_DIR/$ORG/SupportFiles/${ORG}_DNA_seq.pkl")
fi
$PY -m chipatlas_forge.genome \
    --data-dir "$DATA_DIR" --org "$ORG" --release "$RELEASE" \
    "${source_args[@]}"
echo "finished=$(date)"
