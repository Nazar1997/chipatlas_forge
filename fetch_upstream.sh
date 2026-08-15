#!/bin/bash
#
# Fetch current allPeaks_light archives + metadata straight from ChIP-Atlas.
#
# The copies that came off the Yandex share are an older release: hg38 is
# 10.24 GB against upstream's 22.17 GB (2.17x) and mm10 is 7.70 GB against
# 15.88 GB (2.06x), both stamped 2024-11-13 upstream. The bundled
# chip_atlas_experiment_list.zip is worse -- October 2021, missing ~3.2% of the
# accessions even the *old* peaks cite.
#
# Downloads land in raw_fresh/ rather than raw/, so the pipeline keeps running
# against the current data until the new files are verified and swapped in.
#
# Verification is Content-Length plus a full gzip CRC check (`pigz -t`), because
# ChIP-Atlas publishes no checksums. Size alone would accept a truncated file.
#
#   ORGS="hg38 mm10" ./fetch_upstream.sh
set -uo pipefail

ROOT="${ROOT:-$HOME/HyenaProject/data/chipatlas_forge}"
ORGS="${ORGS:-hg38 mm10}"
BASE="https://chip-atlas.dbcls.jp/data"
DEST="$ROOT/raw_fresh"
META="$ROOT/meta"
MAX_TRIES=5

mkdir -p "$DEST" "$META"

remote_size() {
    curl -sIL --max-time 120 "$1" | awk 'tolower($1)=="content-length:"{n=$2} END{gsub(/\r/,"",n); print n+0}'
}

fetch() {
    local url="$1" out="$2" want try=1 have
    want=$(remote_size "$url")
    if [[ "$want" -le 0 ]]; then
        echo "!! cannot size $url" >&2; return 1
    fi
    echo "-> $(basename "$out")  $(numfmt --to=iec "$want")  $(date '+%H:%M:%S')"

    while (( try <= MAX_TRIES )); do
        # -c resumes; a 22 GB restart-from-zero on a blip is not acceptable.
        aria2c -c -x 8 -s 8 -k 16M --auto-file-renaming=false --allow-overwrite=true \
               --summary-interval=120 --console-log-level=warn \
               -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
        have=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [[ "$have" == "$want" ]]; then
            echo "   size OK ($have); checking gzip CRC ..."
            if [[ "$out" == *.gz ]] && ! pigz -t "$out" 2>/dev/null; then
                echo "   CRC FAILED -- refetching" >&2
                rm -f "$out"; (( try++ )); continue
            fi
            echo "OK $(basename "$out")  $(date '+%H:%M:%S')"
            return 0
        fi
        echo "   have $have of $want, retrying" >&2
        (( try++ ))
    done
    echo "!! $(basename "$out") FAILED after $MAX_TRIES tries" >&2
    return 1
}

echo "### upstream fetch started $(date)"
echo "### orgs: $ORGS  ->  $DEST"
failed=0

# Metadata first: small, and it is what decides how many peaks are mappable.
for f in experimentList.tab fileList.tab analysisList.tab; do
    fetch "$BASE/metadata/$f" "$META/$f" || failed=1
done

for org in $ORGS; do
    fetch "$BASE/$org/allPeaks_light/allPeaks_light.$org.05.bed.gz" \
          "$DEST/allPeaks_light.$org.05.bed.gz" || failed=1
done

echo "### finished $(date)  (failed=$failed)"
ls -la "$DEST"
exit $failed
