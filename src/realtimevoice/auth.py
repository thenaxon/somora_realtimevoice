"""Bearer-token authentication for HTTP routes and WebSockets."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, WebSocket, status


def token_ok(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided, expected)


def extract_http_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


def extract_ws_token(websocket: WebSocket) -> str | None:
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    token = websocket.query_params.get("token")
    if token:
        return token
    # browser-compatible path: the client offers its token as a second
    # WebSocket subprotocol ("rtv.bearer.<token>"), keeping tokens out of
    # request-line logs entirely
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in protocols.split(","):
        protocol = protocol.strip()
        if protocol.startswith(SUBPROTOCOL_PREFIX):
            return protocol[len(SUBPROTOCOL_PREFIX) :]
    return None


SUBPROTOCOL = "rtv.bearer"
SUBPROTOCOL_PREFIX = SUBPROTOCOL + "."


def ws_subprotocol_to_accept(websocket: WebSocket) -> str | None:
    """Echo 'rtv.bearer' only if the client offered the auth subprotocols."""
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    offered = {p.strip() for p in protocols.split(",") if p.strip()}
    if SUBPROTOCOL in offered or any(
        p.startswith(SUBPROTOCOL_PREFIX) for p in offered
    ):
        return SUBPROTOCOL
    return None


def require_http_auth(request: Request) -> None:
    expected: str = request.app.state.auth_token
    if not token_ok(extract_http_token(request), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_ws_auth(websocket: WebSocket, expected: str) -> bool:
    return token_ok(extract_ws_token(websocket), expected)
