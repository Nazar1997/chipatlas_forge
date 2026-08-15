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
# 1G of text per shard puts the current archives at ~119 shards for hg38 and
# ~85 for mm10 -- close to one full round at THROTTLE=100. 512M would give 238
# and 171, which is more scheduling churn for no gain. Each route task then
# holds ~30M rows, a few GB.
CHUNK="${CHUNK:-1G}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs"

# Never pass --mem on this cluster: cpu-e-quick nodes report RealMemory=1
# because memory is not a scheduled resource, so any --mem request is rejected
# outright even though the node has ~790 GB free.
