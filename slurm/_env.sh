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
