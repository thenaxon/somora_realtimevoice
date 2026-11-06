"""Server settings, health and version. Every route but health requires the
bearer token."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from . import __version__
from .auth import require_http_auth
from .config import SECRET_MASK

router = APIRouter(prefix="/api", dependencies=[])


def _guard(request: Request) -> None:
    require_http_auth(request)


SERVER_SETTINGS_KEYS = ("session", "default_set", "voice_library")


def _server_view(settings) -> dict[str, Any]:
    """The server's own settings: call defaults, the set for other apps, the
    model that transcribes recordings. Models, voices and sets have their own
    routes."""
    data = settings.model_dump()
    return {key: data[key] for key in SERVER_SETTINGS_KEYS}


@router.get("/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    _guard(request)
    settings = await request.app.state.settings_store.get()
    return _server_view(settings)


@router.put("/settings")
async def put_settings(request: Request) -> dict[str, Any]:
    _guard(request)
    update = await request.json()
    if not isinstance(update, dict):
        raise ValueError("settings payload must be a JSON object")
    unknown = sorted(k for k in update if k not in SERVER_SETTINGS_KEYS and k != "version")
    if unknown:
        raise ValueError(f"not a server setting: {', '.join(unknown)}")
    store = request.app.state.settings_store
    current = await store.get()
    default_set = update.get("default_set")
    if default_set and default_set not in current.conversation_profiles:
        raise ValueError(f"no set is called '{default_set}'")
    stt = (update.get("voice_library") or {}).get("transcription_stt_profile")
    if stt and stt not in current.profiles.get("stt", {}):
        raise ValueError(f"no listening model '{stt}'")
    try:
        settings = await store.update(update)
    except ValidationError as exc:
        parts = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            parts.append(f"{loc or 'settings'}: {error.get('msg', 'invalid value')}")
        raise ValueError("; ".join(parts)) from exc
    return _server_view(settings)


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@router.get("/version")
async def version(request: Request) -> dict[str, Any]:
    """Build identity of the served checkout (determined once at startup)."""
    _guard(request)
    return {"version": __version__, "revision": request.app.state.revision}


@router.get("/session-auth-check")
async def session_auth_check(request: Request) -> dict[str, Any]:
    _guard(request)
    sessions = request.app.state.sessions
    return {"active_sessions": len(sessions), "secret_mask": SECRET_MASK}


def register_settings_routes(app) -> None:
    app.include_router(router)
