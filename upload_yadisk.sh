#!/bin/bash
#
# Publish the current archives + metadata to the existing Yandex Disk share.
#
#   https://disk.yandex.ru/d/RTrDMXJrszil8Q  ==  disk:/chipatlas_05
#
# **REST API, not WebDAV, and not rclone.** WebDAV writes on this account are
# throttled to ~16 KB/s -- measured from two networks, so it is Yandex-side --
# which puts a 22 GB file at roughly two weeks, and large PUTs stall outright
# rather than merely crawling. The REST upload flow runs at ~20 MB/s, about
# 1200x faster. rclone's native `yandex` backend cannot be used either: it
# demands a refresh_token that the OAuth polygon does not issue.
#
# Credentials: an OAuth token (`y0__...`) at ~/.yandex_oauth, mode 0600. The
# 16-char app password is Basic-auth only and returns 401 here -- wrong
# mechanism, not a missing scope.
#
# Uploads land in a DATED SUBFOLDER, leaving the superseded release in place.
# The old files carry identical names, so writing at the top level would
# overwrite a release that ChIP-Atlas no longer serves -- and the point of this
# refresh was that stale copies are dangerous, not that they are worthless.
#
#   ./upload_yadisk.sh
set -uo pipefail

ROOT="${ROOT:-$HOME/HyenaProject/data/chipatlas_forge}"
REMOTE="${REMOTE:-disk:/chipatlas_05/2024-11_q05}"
API="https://cloud-api.yandex.net/v1/disk"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.yandex_oauth}"
# Generous, because each retry is a fresh roll of the uploader-host dice rather
# than a repeat of the same failing thing.
MAX_TRIES=8

[[ -r "$TOKEN_FILE" ]] || { echo "no OAuth token at $TOKEN_FILE" >&2; exit 1; }
TOKEN=$(tr -d ' \n' < "$TOKEN_FILE")

# --raw so spaces/slashes survive; the token never appears in argv.
enc() { python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
api() { curl -sS -H "Authorization: OAuth $TOKEN" "$@"; }

mkdir_remote() {
    local code
    code=$(api -o /dev/null -w '%{http_code}' -X PUT "$API/resources?path=$(enc "$1")")
    # 201 created, 409 already there -- both fine.
    [[ "$code" == "201" || "$code" == "409" ]] || { echo "mkdir $1 -> HTTP $code" >&2; return 1; }
}

remote_md5() {
    api "$API/resources?path=$(enc "$1")&fields=md5" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("md5",""))' 2>/dev/null
}

put() {
    local local_path="$1" remote_path="$2" want href code try=1
    want=$(md5sum "$local_path" | cut -d' ' -f1)

    if [[ "$(remote_md5 "$remote_path")" == "$want" ]]; then
        echo "== $(basename "$local_path") already there and md5-clean"
        return 0
    fi

    while (( try <= MAX_TRIES )); do
        echo "-> $(basename "$local_path")  $(numfmt --to=iec "$(stat -c %s "$local_path")")  try $try  $(date '+%H:%M:%S')"
        href=$(api "$API/resources/upload?path=$(enc "$remote_path")&overwrite=true" \
               | python3 -c 'import sys,json; print(json.load(sys.stdin).get("href",""))')
        if [[ -z "$href" ]]; then
            echo "   no upload href; retrying in 30s" >&2; sleep 30; (( try++ )); continue
        fi
        # A one-time href on an uploader*.disk.yandex.net host, and WHICH host
        # you get decides everything. Throughput varies by orders of magnitude
        # between them, and a bad one does not merely crawl -- it accepts the
        # connection and then sends nothing at all. Observed: uploader746klg
        # held a 329 MB PUT for 13 minutes at wchar=0, while uploader13klg has
        # done 20 MB/s on a 3.8 GB file.
        #
        # So the real defence is a stall detector, not a timeout: abort once
        # throughput sits under 100 KB/s for 2 minutes, then loop round and ask
        # for a NEW href, which usually lands on a different host. --max-time is
        # only a final ceiling -- at 20 MB/s the 22 GB archives need ~18 min.
        code=$(curl -sS -o /dev/null -w '%{http_code}' \
                    --speed-limit 100000 --speed-time 120 \
                    --max-time 7200 -T "$local_path" "$href")
        if [[ "$code" == "201" || "$code" == "202" ]]; then
            sleep 5                                   # let Yandex finish hashing
            if [[ "$(remote_md5 "$remote_path")" == "$want" ]]; then
                echo "OK $(basename "$local_path")  md5 $want  $(date '+%H:%M:%S')"
                return 0
            fi
            echo "   uploaded but md5 mismatch" >&2
        else
            echo "   PUT returned $code" >&2
        fi
        (( try++ ))
    done
    echo "!! $(basename "$local_path") FAILED" >&2
    return 1
}

echo "### upload started $(date)"
echo "### $ROOT  ->  $REMOTE"
mkdir_remote "$REMOTE"     || exit 1
mkdir_remote "$REMOTE/meta" || exit 1

failed=0

# Metadata first: small, and it is the piece whose staleness silently drops peaks.
for f in experimentList.tab fileList.tab analysisList.tab; do
    [[ -f "$ROOT/meta/$f" ]] && { put "$ROOT/meta/$f" "$REMOTE/meta/$f" || failed=1; }
done

# Archives, smallest first -- a credential or quota problem surfaces in seconds
# rather than after 22 GB.
while read -r size path; do
    put "$path" "$REMOTE/$(basename "$path")" || failed=1
done < <(find "$ROOT/raw" -name 'allPeaks_light.*.05.bed.gz' -printf '%s %p\n' | sort -n)

echo "### finished $(date)  (failed=$failed)"
exit $failed
