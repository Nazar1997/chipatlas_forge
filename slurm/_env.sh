# Shared settings for every stage. Sourced, not executed.
#
# The interpreter is named by absolute path on purpose: /usr/bin/python on the
# cHARISMa nodes is 2.7.5, and `module load Python` + `conda activate` is two
# more things that can silently no-op inside a batch job and leave you running
# the system interpreter. myenv has the pyarrow 20 / numpy / pandas this needs.
PY="${PY:-$HOME/.conda/envs/myenv/bin/python}"
ROOT="${ROOT:-$HOME/HyenaProject/data/chipatlas_forge}"
PARTITION="${PARTITION:-cpu-e-quick}"
BUCKETS="${BUCKETS:-128}"
# 2G of text per shard puts the current archives at ~60 shards for hg38 and ~43
# for mm10. Sized against the QOS submit cap rather than against throughput:
# every pending array task counts toward MaxSubmitJobsPU=500, so a smaller chunk
# (512M -> 238 shards for hg38) leaves no budget for the collect array. Each
# route task then holds ~60M rows, a few GB, which these nodes have in abundance.
CHUNK="${CHUNK:-2G}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs"

# Never pass --mem on this cluster: cpu-e-quick nodes report RealMemory=1
# because memory is not a scheduled resource, so any --mem request is rejected
# outright even though the node has ~790 GB free.

# --- prepare stages (06-11) -------------------------------------------------
# Where releases live. The peak stages write into $ROOT (forge's own working
# tree); the prepare stages write into the data directory beside it, because a
# release is training input and not a build artifact of this repository.
DATA_DIR="${DATA_DIR:-$(cd "$ROOT/.." && pwd)}"
# The release being built. Dated, and it becomes a directory name -- see
# chipatlas_forge.layout.check_release_id for what is allowed.
RELEASE="${RELEASE:-$(date -u +%Y-%m)}"
# The release whose genome and DNA chunks a new one adopts. hg38 is hg38
# whichever ChIP-Atlas snapshot the peaks came from, so this is a hard link
# rather than 3 GB of copying. Unset it to rebuild from the FASTA.
DONOR_RELEASE="${DONOR_RELEASE:-2021-10}"

# Whether $DONOR_RELEASE actually exists for $ORG. Until the 2021 tree is
# migrated there is no donor *release* -- only the pre-release
# data/<org>/{DNA,SupportFiles,Subtables} layout -- and the stages that adopt
# the genome and the DNA chunks have to read that instead. Migration is blocked
# on training being idle, and building a release should not have to wait for it.
have_donor_release() {
    local org="$1"
    [[ -n "${DONOR_RELEASE:-}" \
       && -f "$DATA_DIR/$org/releases/$DONOR_RELEASE/MANIFEST.json" ]]
}
