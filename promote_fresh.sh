#!/bin/bash
#
# Swap the freshly-downloaded archives into raw/ once they verify.
#
# Nothing is deleted. The superseded archives move to raw_superseded/ so the
# swap is reversible and so a half-finished download can never quietly replace
# good data -- every file is re-checked here even though fetch_upstream.sh
# already checked it, because this is the step that is hard to undo.
#
# Any existing work/ and out/ are cleared: they were derived from the old
# archives, and shards left behind would be silently mixed with new ones by
# stage 2, which indexes shards positionally.
#
#   ORGS="hg38 mm10" ./promote_fresh.sh
set -uo pipefail

ROOT="${ROOT:-$HOME/HyenaProject/data/chipatlas_forge}"
ORGS="${ORGS:-hg38 mm10}"
FRESH="$ROOT/raw_fresh"
RAW="$ROOT/raw"
OLD="$ROOT/raw_superseded"

# A full CRC pass inflates the whole archive: ~10 min for hg38, ~7 for mm10,
# since plain gzip is single-threaded (see shard.py). fetch_upstream.sh already
# did exactly this check, so VERIFY=0 skips it when the two run back to back.
# It defaults on because this is the step that is hard to undo.
VERIFY="${VERIFY:-1}"

for org in $ORGS; do
    f="$FRESH/allPeaks_light.$org.05.bed.gz"
    [[ -f "$f" ]] || { echo "!! missing $f" >&2; exit 1; }
    [[ -f "$f.aria2" ]] && { echo "!! $org still downloading (.aria2 present)" >&2; exit 1; }
    if [[ "$VERIFY" == "1" ]]; then
        echo "checking $org gzip CRC (VERIFY=0 to skip) ..."
        pigz -t "$f" || { echo "!! $org failed CRC; not promoting" >&2; exit 1; }
    fi
    echo "  ok  $org  $(numfmt --to=iec "$(stat -c %s "$f")")"
done

mkdir -p "$OLD" "$RAW"
for org in $ORGS; do
    if [[ -f "$RAW/allPeaks_light.$org.05.bed.gz" ]]; then
        mv -v "$RAW/allPeaks_light.$org.05.bed.gz" "$OLD/"
    fi
    mv -v "$FRESH/allPeaks_light.$org.05.bed.gz" "$RAW/"
done

# Clear derived state for the promoted organisms ONLY.
#
# This used to wipe all of work/ and out/. That is correct when an archive is
# replaced -- stage 2 addresses shards positionally, so leftovers from the old
# archive would be silently mixed with new ones -- but catastrophic when the
# promotion merely ADDS assemblies: it would delete finished output for every
# organism already built, which at the time of writing was 152 GB of hg38 and
# mm10 BEDs that have nothing to do with this promotion.
for org in $ORGS; do
    for stale in "$ROOT/work/shards/$org" "$ROOT/work/parts/$org" \
                 "$ROOT/work/stats/$org" "$ROOT/work/collect_stats/$org" \
                 "$ROOT/work/manifest/$org" "$ROOT/out/$org"; do
        if [[ -d "$stale" ]]; then
            echo "clearing $stale (derived from the superseded $org archive)"
            rm -rf "${stale:?}"
        fi
    done
done

echo
echo "promoted. superseded archives kept in $OLD -- delete when you are happy."
ls -la "$RAW"
