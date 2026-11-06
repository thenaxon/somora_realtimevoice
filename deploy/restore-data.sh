#!/usr/bin/env bash
# restore-data.sh — restore a backup into a NEW, empty data directory.
#
# Usage:
#   deploy/restore-data.sh <backup.tar.gz> <new_empty_dir>
#
# Safety properties:
#   * refuses a target that exists and is not empty (never overwrites live data);
#   * verifies <backup>.sha256 before extracting anything;
#   * extracts, then re-runs deploy/data_manifest.py on the extracted data and
#     compares it with the manifest stored beside the archive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "usage: $0 <backup.tar.gz> <new_empty_dir>" >&2
}

ARCHIVE="${1:-}"
TARGET="${2:-}"
if [ -z "$ARCHIVE" ] || [ -z "$TARGET" ]; then
    usage
    exit 1
fi
if [ ! -f "$ARCHIVE" ]; then
    echo "error: backup archive '$ARCHIVE' not found" >&2
    exit 1
fi

CHECKSUM="${ARCHIVE}.sha256"
if [ ! -f "$CHECKSUM" ]; then
    echo "error: checksum file '$CHECKSUM' not found" >&2
    exit 1
fi

# 1) checksum verification BEFORE touching the target
echo "verifying sha256 checksum" >&2
if command -v sha256sum >/dev/null 2>&1; then
    if ! ( cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$CHECKSUM")" >/dev/null 2>&1 ); then
        echo "error: sha256 checksum mismatch for '$ARCHIVE'" >&2
        exit 1
    fi
else
    recorded="$(awk '{print $1}' "$CHECKSUM")"
    actual="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
    if [ "$recorded" != "$actual" ]; then
        echo "error: sha256 checksum mismatch for '$ARCHIVE'" >&2
        exit 1
    fi
fi

# 2) the target must not exist or must be empty
if [ -e "$TARGET" ]; then
    if [ ! -d "$TARGET" ]; then
        echo "error: restore target '$TARGET' exists and is not a directory" >&2
        exit 1
    fi
    if [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
        echo "error: restore target '$TARGET' is non-empty; refusing to overwrite" >&2
        exit 1
    fi
else
    mkdir -p "$TARGET"
fi

# 3) extract
echo "extracting into $TARGET" >&2
tar -C "$TARGET" -xzf "$ARCHIVE"

# 4) manifest check against the stored manifest (when present)
MANIFEST="${ARCHIVE%.tar.gz}.manifest.json"
if [ -f "$MANIFEST" ]; then
    echo "verifying restored data against stored manifest" >&2
    python3 "$SCRIPT_DIR/data_manifest.py" "$TARGET" --compare "$MANIFEST" >/dev/null
else
    echo "warning: no stored manifest beside '$ARCHIVE'; skipping manifest check" >&2
    python3 "$SCRIPT_DIR/data_manifest.py" "$TARGET" >/dev/null
fi

echo "restored to $TARGET" >&2
