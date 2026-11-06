#!/usr/bin/env bash
# smoke-image.sh — start an image on loopback with an empty data dir and check
# that the unauthenticated /api/health path answers. Documented for manual use
# on the target host; not run in CI.
#
# Usage:
#   deploy/smoke-image.sh <image> [host_port]
#
# The container is a throwaway: it is removed on exit and its data dir lives in
# a temp directory that is deleted afterwards. No host data is touched.
set -euo pipefail

IMAGE="${1:-}"
HOST_PORT="${2:-18766}"

if [ -z "$IMAGE" ]; then
    echo "usage: $0 <image> [host_port]" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not on PATH" >&2
    exit 1
fi

DATA_DIR="$(mktemp -d)"
CONTAINER="rtv-smoke-$$"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    rm -rf "$DATA_DIR"
}
trap cleanup EXIT

echo "starting $IMAGE on 127.0.0.1:${HOST_PORT}" >&2
docker run -d --rm \
    --name "$CONTAINER" \
    -p "127.0.0.1:${HOST_PORT}:8000" \
    -v "$DATA_DIR:/app/data" \
    -e REALTIMEVOICE_DATA_DIR=/app/data \
    "$IMAGE" >/dev/null

for _ in $(seq 1 30); do
    if python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${HOST_PORT}/api/health', timeout=2).status == 200 else 1)" 2>/dev/null; then
        echo "health: ok" >&2
        exit 0
    fi
    if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
        echo "error: container exited during startup; logs:" >&2
        docker logs "$CONTAINER" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "error: /api/health did not become ready in time; logs:" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
