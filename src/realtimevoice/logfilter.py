"""Access-log redaction: instance tokens must never reach the journal.

uvicorn's access log includes the raw request line (with query string), and
WebSocket URLs carry `?token=...` for non-browser clients. This filter
scrubs token values and Authorization headers from every access record.
"""

from __future__ import annotations

import re

_TOKEN_QUERY = re.compile(r"(token=)[^&\s\"']+", re.IGNORECASE)
_BEARER = re.compile(r"(bearer\s+)\S+", re.IGNORECASE)

# patterns for error-text sanitization: anything secret- or transport-shaped
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"(?i)(token|api[_-]?key|access[_-]?token|authorization|password)\s*[=:]\s*\S+"),
    re.compile(r"(?i)(token=)[^&\s\"']+"),
    # opaque backend-internal reference ids that may appear in error text
    re.compile(r"(?i)\b(trace|request[_-]?id|correlation[_-]?id|run[_-]?id|task[_-]?id|span[_-]?id)=[^\s\"']+"),
)
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_B64_BLOB = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def sanitize_error_text(text: Any, limit: int = 300) -> str:
    """Make a client-facing error message content-safe: strip URLs, secret
    patterns and base64 blobs, collapse whitespace, cap length. Protocol
    error strings pass through unchanged in practice (nothing matches)."""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    out = _URL.sub("[url]", out)
    out = _B64_BLOB.sub("[redacted]", out)
    out = re.sub(r"\s+", " ", out.replace("\n", " ").replace("\r", " ")).strip()
    return out[:limit]


class TokenRedactingFilter:
    def __init__(self, name: str = "") -> None:
        self.name = name

    def filter(self, record) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if "token=" not in message and "bearer" not in message.lower():
            return True
        scrubbed = _TOKEN_QUERY.sub(r"\1[redacted]", message)
        scrubbed = _BEARER.sub(r"\1[redacted]", scrubbed)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = None
        return True
