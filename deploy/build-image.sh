#!/usr/bin/env bash
# build-image.sh — build the app image from an exact COMMIT (never the worktree).
#
# Usage:
#   deploy/build-image.sh <commit>
#
# The build context is `git archive <commit>`, so uncommitted working-tree
# changes can never leak into a released image. The script refuses a dirty
# checkout or an unknown revision. It tags `somora-realtimevoice:<shortsha>` and
# passes `--build-arg REVISION=<shortsha>`, then prints the resulting image id.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    echo "usage: $0 <commit>" >&2
}

COMMIT="${1:-}"
if [ -z "$COMMIT" ]; then
    usage
    exit 1
fi

if ! git rev-parse --verify --quiet "${COMMIT}^{commit}" >/dev/null; then
    echo "error: unknown commit '$COMMIT'" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is dirty; commit or stash before building" >&2
    exit 1
fi

SHORT_SHA="$(git rev-parse --short=12 "${COMMIT}^{commit}")"
IMAGE="somora-realtimevoice:${SHORT_SHA}"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not on PATH" >&2
    exit 1
fi

CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "$CONTEXT_DIR"' EXIT

echo "exporting $COMMIT to build context" >&2
git archive "${COMMIT}^{commit}" | tar -x -C "$CONTEXT_DIR"

echo "building ${IMAGE} from ${COMMIT}" >&2
docker build \
    --build-arg "REVISION=${SHORT_SHA}" \
    -t "$IMAGE" \
    "$CONTEXT_DIR"

echo "image: ${IMAGE}"
docker image inspect --format '{{.Id}}' "$IMAGE"
