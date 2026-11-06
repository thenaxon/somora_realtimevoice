#!/usr/bin/env bash
# check-cert.sh — inspect the certificate a TLS endpoint actually serves.
#
# Usage:
#   deploy/check-cert.sh <hostname> <addr:port>
#
# Prints subject, SAN and notAfter, plus days remaining. Exits non-zero when
# fewer than RTV_CERT_MIN_DAYS days (default 21) are left, so a cron job can
# warn before expiry. Only public certificate metadata is printed — never any
# private key material.
set -euo pipefail

HOSTNAME="${1:-}"
ENDPOINT="${2:-}"
MIN_DAYS="${RTV_CERT_MIN_DAYS:-21}"

usage() {
    echo "usage: $0 <hostname> <addr:port>" >&2
}

if [ -z "$HOSTNAME" ] || [ -z "$ENDPOINT" ]; then
    usage
    exit 2
fi
ADDR="${ENDPOINT%:*}"
PORT="${ENDPOINT##*:}"

if ! command -v openssl >/dev/null 2>&1; then
    echo "error: openssl not on PATH" >&2
    exit 1
fi

PEM="$(openssl s_client -servername "$HOSTNAME" -connect "$ADDR:$PORT" \
    -showcerts </dev/null 2>/dev/null \
    | sed -n '/BEGIN CERTIFICATE/,/END CERTIFICATE/p' | head -n 40)"

if [ -z "$PEM" ]; then
    echo "error: could not retrieve a certificate from $ADDR:$PORT" >&2
    exit 1
fi

SUBJECT="$(printf '%s\n' "$PEM" | openssl x509 -noout -subject -nameopt RFC2253 2>/dev/null \
    | sed 's/^subject=//')"
SAN="$(printf '%s\n' "$PEM" | openssl x509 -noout -ext subjectAltName 2>/dev/null \
    | sed -n '2p' | sed 's/^[[:space:]]*//')"
NOT_AFTER="$(printf '%s\n' "$PEM" | openssl x509 -noout -enddate 2>/dev/null | sed 's/^notAfter=//')"

echo "subject:   ${SUBJECT}"
echo "SAN:       ${SAN}"
echo "notAfter:  ${NOT_AFTER}"

END_EPOCH="$(date -d "$NOT_AFTER" +%s 2>/dev/null || echo "")"
NOW_EPOCH="$(date +%s)"
if [ -z "$END_EPOCH" ]; then
    echo "error: cannot parse notAfter '${NOT_AFTER}'" >&2
    exit 1
fi
DAYS_LEFT=$(( (END_EPOCH - NOW_EPOCH) / 86400 ))
echo "days left: ${DAYS_LEFT} (warn below ${MIN_DAYS})"

if [ "$DAYS_LEFT" -lt "$MIN_DAYS" ]; then
    echo "error: certificate expires in ${DAYS_LEFT} days (< ${MIN_DAYS})" >&2
    exit 1
fi
