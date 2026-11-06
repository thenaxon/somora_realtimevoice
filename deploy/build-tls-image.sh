#!/usr/bin/env bash
# build-tls-image.sh — build the DNS-01 TLS terminator image.
#
# Usage:
#   deploy/build-tls-image.sh [tag]
#
# Builds deploy/tls/Dockerfile.dns-cloudflare with xcaddy and the pinned
# caddy-dns/cloudflare module. Default tag: the Caddy version in the
# Dockerfile (e.g. somora-realtimevoice-tls:2.10.0). Needs a local Docker
# daemon; the module and base digests are pinned so the build is reproducible.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/tls/Dockerfile.dns-cloudflare"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not on PATH" >&2
    exit 1
fi

# default tag = the runtime base Caddy version (last FROM line)
CADDY_VERSION="$(grep -E '^FROM caddy:[0-9.]+-alpine@sha256:' "$DOCKERFILE" \
    | sed -E 's/^FROM caddy:([0-9.]+)-alpine.*/\1/' | tail -1)"
TAG="${1:-$CADDY_VERSION}"
IMAGE="somora-realtimevoice-tls:${TAG}"

echo "building ${IMAGE} from ${DOCKERFILE}" >&2
docker build -t "$IMAGE" -f "$DOCKERFILE" "$SCRIPT_DIR/tls"

echo "$IMAGE"
