#!/usr/bin/env bash
# backup-data.sh — consistent, secret-free snapshot of a realtimevoice data dir.
#
# Usage:
#   deploy/backup-data.sh <data_dir> <backup_dir> \
#       [--stop-container <name>]... [--allow-running]
#
# Produces in <backup_dir>:
#   rtv-data-<UTC timestamp>.tar.gz           0600 archive of <data_dir>
#   rtv-data-<UTC timestamp>.tar.gz.sha256    0600 checksum
#   rtv-data-<UTC timestamp>.manifest.json    0600 secret-free inventory
#
# Consistency: pass one or more `--stop-container <name>` (the app container
# that writes data, normally `somora-realtimevoice`) to stop them with
# `docker stop --time 20` immediately before archiving. Exactly the containers
# that were running are started again afterwards (trap on EXIT/ERR), and the
# script verifies they are running before returning.
#
# If stopping a requested container FAILS, the script aborts with a non-zero
# exit and creates NO archive — a "consistent snapshot" must never silently
# become an inconsistent one. `--allow-running` opts into archiving anyway and
# records `"consistent": false` in the manifest (the operator accepts that the
# data may be mid-write).
#
# The manifest records `"consistent": true` when the app was stopped for the
# archive, and also when no stop was requested (a plain offline copy).
#
# Retention: only files inside <backup_dir> matching this script's own naming
# pattern AND older than RTV_BACKUP_KEEP_DAYS (default 30) are removed. Nothing
# else is ever deleted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_CONTAINER_DEFAULT="somora-realtimevoice"

usage() {
    echo "usage: $0 <data_dir> <backup_dir> [--stop-container <name>]... [--allow-running]" >&2
    echo "  stop the given containers (e.g. $APP_CONTAINER_DEFAULT) for a consistent archive" >&2
}

DATA_DIR="${1:-}"
BACKUP_DIR="${2:-}"
shift 2 2>/dev/null || true

STOP_CONTAINERS=()
ALLOW_RUNNING=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stop-container)
            name="${2:-}"
            if [ -z "$name" ]; then
                usage
                exit 1
            fi
            STOP_CONTAINERS+=("$name")
            shift 2
            ;;
        --allow-running)
            ALLOW_RUNNING=1
            shift
            ;;
        --stop)
            # legacy docker-compose-directory flag: removed on purpose so a
            # misconfigured compose interpolation can never yield an
            # inconsistent archive. No silent fallback.
            echo "error: --stop <compose-project-dir> is no longer supported;" >&2
            echo "       use --stop-container ${APP_CONTAINER_DEFAULT} instead." >&2
            exit 1
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [ -z "$DATA_DIR" ] || [ -z "$BACKUP_DIR" ]; then
    usage
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "error: data dir '$DATA_DIR' is not a directory" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/settings.json" ]; then
    echo "error: '$DATA_DIR/settings.json' not found" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
chmod 0700 "$BACKUP_DIR" 2>/dev/null || true

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASENAME="rtv-data-${STAMP}"
ARCHIVE="$BACKUP_DIR/${BASENAME}.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"
MANIFEST="$BACKUP_DIR/${BASENAME}.manifest.json"
TMP_ARCHIVE="${ARCHIVE}.partial"

# containers we actually stopped, so we restart exactly those
STOPPED=()
CONSISTENT="true"

running_now() {
    # true when the named container is currently running (docker absent -> false)
    local name="$1"
    command -v docker >/dev/null 2>&1 || return 1
    [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = "true" ]
}

restart_stopped() {
    [ "${#STOPPED[@]}" -eq 0 ] && return 0
    echo "restarting stopped containers: ${STOPPED[*]}" >&2
    docker start "${STOPPED[@]}" >&2 || \
        echo "warning: could not restart ${STOPPED[*]}; check manually" >&2
    local name
    for name in "${STOPPED[@]}"; do
        if running_now "$name"; then
            echo "container running again: $name" >&2
        else
            echo "warning: container '$name' is NOT running after restart" >&2
        fi
    done
    STOPPED=()
}

cleanup_tmp() {
    rm -f "$TMP_ARCHIVE"
}

on_exit() {
    local status=$?
    restart_stopped
    cleanup_tmp
    return "$status"
}
trap on_exit EXIT

stop_requested_containers() {
    [ "${#STOP_CONTAINERS[@]}" -eq 0 ] && return 0
    if ! command -v docker >/dev/null 2>&1; then
        echo "error: docker not on PATH; cannot stop ${STOP_CONTAINERS[*]}" >&2
        return 1
    fi

    local name
    for name in "${STOP_CONTAINERS[@]}"; do
        if ! running_now "$name"; then
            echo "container '$name' is not running; nothing to stop" >&2
            continue
        fi
        echo "stopping container '$name' for a consistent snapshot" >&2
        if ! docker stop --time 20 "$name" >&2; then
            echo "error: failed to stop container '$name'" >&2
            return 1
        fi
        STOPPED+=("$name")
    done
    return 0
}

if [ "${#STOP_CONTAINERS[@]}" -gt 0 ]; then
    if stop_requested_containers; then
        CONSISTENT="true"
    else
        if [ "$ALLOW_RUNNING" -eq 1 ]; then
            echo "warning: stop failed; --allow-running given, archiving a running data dir" >&2
            CONSISTENT="false"
        else
            echo "error: refusing to archive an inconsistent data dir; pass --allow-running to override" >&2
            exit 3
        fi
    fi
fi

# manifest is produced from the same (possibly stopped) data dir. data_manifest
# prints only counts/ids/hashes; the consistency flag is added by this script.
manifests_tmp="$MANIFEST.partial"
python3 "$SCRIPT_DIR/data_manifest.py" "$DATA_DIR" > "$manifests_tmp"
python3 - "$manifests_tmp" "$CONSISTENT" > "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
data["consistent"] = sys.argv[2] == "true"
json.dump(data, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY
rm -f "$manifests_tmp"
chmod 0600 "$MANIFEST"

# archive the data dir contents (top-level entries), no absolute paths
tar -C "$DATA_DIR" -czf "$TMP_ARCHIVE" .
chmod 0600 "$TMP_ARCHIVE"
mv "$TMP_ARCHIVE" "$ARCHIVE"

if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$BACKUP_DIR" && sha256sum "$(basename "$ARCHIVE")" > "$CHECKSUM" )
else
    ( cd "$BACKUP_DIR" && shasum -a 256 "$(basename "$ARCHIVE")" > "$CHECKSUM" )
fi
chmod 0600 "$CHECKSUM"

restart_stopped

# retention: only this script's own archives, only older than the keep window
KEEP_DAYS="${RTV_BACKUP_KEEP_DAYS:-30}"
if [ "$KEEP_DAYS" -gt 0 ] 2>/dev/null; then
    find "$BACKUP_DIR" -maxdepth 1 -type f \
        \( -name 'rtv-data-*.tar.gz' -o -name 'rtv-data-*.tar.gz.sha256' \
           -o -name 'rtv-data-*.manifest.json' \) \
        -mtime "+${KEEP_DAYS}" -delete
fi

echo "$ARCHIVE"