"""FastAPI application: public realtime WS, settings API, static test page."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from . import __version__

logger = logging.getLogger("realtimevoice")
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import require_ws_auth, ws_subprotocol_to_accept
from .backends.base import Backends
from .backends.openai_http import build_backends
from .catalog import ResolvedSelection, SelectionError
from .config import Settings, SettingsLoadError, SettingsStore
from .errors import BackendError
from .profiles import (
    PROFILE_PREFIX,
    PROFILE_TEST_PREFIX,
    VOICE_ROUTE_PREFIX,
    TestTicketError,
    pending_selection,
    resolve_default_set,
    resolve_profile,
    resolve_test,
    resolve_voice_route_full,
)
from .session import RealtimeSession

WEB_DIR = Path(__file__).parent / "web"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_WS_CLOSE_REASON_MAX = 100

# Build identity injected by the container image (Dockerfile ARG REVISION ->
# ENV REALTIMEVOICE_REVISION). A container has no .git checkout, so this is the
# only way ``/api/version`` can report the deployed revision. Restricted to a
# conservative character class so arbitrary env content is never echoed back.
_REVISION_ENV = "REALTIMEVOICE_REVISION"
_REVISION_RE = re.compile(r"[0-9A-Za-z._-]{1,64}")


def detect_revision(repo_root: Path = _REPO_ROOT) -> str:
    """Build identity of the served checkout, ``"unknown"`` on any failure.

    The ``REALTIMEVOICE_REVISION`` env var (set from the image build arg) is
    authoritative when present and valid. Otherwise the short git revision of
    the served checkout is used. Determined once at startup and never allowed
    to abort it: a missing git, a non-repo checkout or a dirty tree all fall
    back to ``"unknown"``.
    """
    env_revision = os.environ.get(_REVISION_ENV, "").strip()
    if _REVISION_RE.fullmatch(env_revision):
        return env_revision
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        return "unknown"
    return revision


class NoCacheStaticFiles(StaticFiles):
    """Static web assets that always revalidate.

    Browsers apply heuristic freshness to static responses that lack an
    explicit ``Cache-Control`` and then reuse ES modules from the disk cache
    without asking the server — a human retest can silently run pre-deploy
    client code. ``no-cache`` forces a conditional request on every load; the
    ``ETag`` / ``Last-Modified`` response stays a cheap 304 when unchanged.
    API responses are not served here and keep their own headers.
    """

    async def get_response(self, path: str, scope) -> object:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def sanitize_close_reason(text: str) -> str:
    """WS close reasons are limited (~100 bytes) and must not leak secrets."""
    from .logfilter import sanitize_error_text

    reason = sanitize_error_text(text)
    return reason.encode("utf-8")[:_WS_CLOSE_REASON_MAX].decode("utf-8", "ignore")


def default_backend_factory(settings: Settings, client: httpx.AsyncClient | None = None) -> Backends:
    stt, llm, tts = build_backends(settings.stt, settings.llm, settings.tts, client)
    return Backends(stt=stt, llm=llm, tts=tts)


BackendFactory = Callable[[Settings, httpx.AsyncClient | None], Backends]


async def _refuse(websocket: WebSocket, status: int, code: str, message: str, *, close_code: int) -> None:
    """Refuse a realtime connection before it is accepted, so the client can
    tell why: an HTTP status with an OpenAI-style error body where the server
    supports it (a close before accept reaches clients as a bare 403), the
    close code and reason otherwise."""
    if "websocket.http.response" in (websocket.scope.get("extensions") or {}):
        body = {"error": {"type": "invalid_request_error", "code": code, "message": message}}
        await websocket.send_denial_response(JSONResponse(status_code=status, content=body))
        return
    await websocket.close(code=close_code, reason=message)


def create_app(
    *,
    settings_store: SettingsStore | None = None,
    backend_factory: BackendFactory | None = None,
    auth_token: str = "",
    web_dir: Path | None = None,
) -> FastAPI:
    store = settings_store or SettingsStore()
    factory = backend_factory or default_backend_factory
    web = web_dir or WEB_DIR
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # One HTTP client for every backend call; open sessions are closed
        # and the client with them when the server stops.
        app.state.http_client = httpx.AsyncClient(timeout=60.0)
        try:
            yield
        finally:
            for session in list(app.state.sessions.values()):
                session.close()
            await app.state.http_client.aclose()

    app = FastAPI(title="somora_realtimevoice", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings_store = store
    app.state.backend_factory = factory
    app.state.auth_token = auth_token
    app.state.revision = detect_revision()
    app.state.sessions = {}
    app.state.http_client = None
    from .profiles import Base64LRU, ProfileTestStore
    from .voices import SampleStore

    app.state.sample_store = SampleStore(store.data_dir)
    app.state.b64_cache = Base64LRU()
    app.state.profile_tests = ProfileTestStore()
    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        status_code = getattr(exc, "status_code", None) or 422
        # content-safe detail: a raw pydantic ValidationError embeds the
        # submitted input values (credentials / reference material must
        # never be echoed back in the error body)
        from .logfilter import sanitize_error_text

        detail = sanitize_error_text(str(exc))
        content: dict[str, Any] = {"detail": detail, "code": getattr(exc, "code", None)}
        issues = getattr(exc, "issues", None)
        if issues is not None:
            content["issues"] = issues
        probes = getattr(exc, "probes", None)
        if probes is not None:
            content["probes"] = probes
        return JSONResponse(status_code=status_code, content=content)

    @app.exception_handler(SettingsLoadError)
    async def _settings_load_error_handler(_request: Request, exc: SettingsLoadError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": "settings_unavailable"})

    @app.exception_handler(BackendError)
    async def _backend_error_handler(_request: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": exc.message, "code": exc.code})

    from .routes_settings import register_settings_routes
    from .routes_catalog import register_catalog_routes
    from .routes_profiles import register_profile_routes

    register_settings_routes(app)
    register_catalog_routes(app)
    register_profile_routes(app)

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        from .auth import require_http_auth

        require_http_auth(request)
        try:
            settings = await store.get()
        except SettingsLoadError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=str(exc))
        data: list[dict] = []
        for pid, profile in sorted(settings.conversation_profiles.items()):
            if not profile.enabled:
                continue
            data.append(
                {
                    "id": f"profile:{pid}",
                    "object": "model",
                    "owned_by": "realtimevoice",
                    "x_realtimevoice": {
                        "display_name": profile.display_name,
                        "language": profile.spec.language,
                        "default": pid == settings.default_set,
                    },
                }
            )
        data.append(
            {
                "id": "voice-route:auto",
                "object": "model",
                "owned_by": "realtimevoice",
                "x_realtimevoice": {
                    "display_name": "The set named by the call's voice",
                    "language": "",
                    "default": False,
                },
            }
        )
        return {"object": "list", "data": data}

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(websocket: WebSocket, model: str = "") -> None:
        if not require_ws_auth(websocket, app.state.auth_token):
            await _refuse(websocket, 401, "invalid_token", "invalid or missing token", close_code=4401)
            return
        try:
            settings = await store.get()
        except SettingsLoadError as exc:
            await _refuse(websocket, 503, "settings_unavailable", sanitize_close_reason(str(exc)), close_code=1011)
            return
        # resolve the model to a frozen snapshot BEFORE any backend exists;
        # an unknown set is rejected explicitly (no silent default)
        voice_route: dict | None = None
        route_pending = False
        route_resolver = None
        selection: ResolvedSelection | None = None
        connect_warnings: list[str] = []
        try:
            if model.startswith(PROFILE_TEST_PREFIX):
                ticket = model[len(PROFILE_TEST_PREFIX):]
                entry = app.state.profile_tests.get(ticket)
                if entry is None:
                    raise TestTicketError(ticket)
                pid, _expires = entry
                selection, voice_route = resolve_test(
                    settings, pid,
                    sample_store=app.state.sample_store, b64_cache=app.state.b64_cache,
                    # F10: the unsaved editor selection captured with the ticket
                    spec_override=app.state.profile_tests.spec_for(ticket),
                )
                voice_route = dict(voice_route)
            elif model.startswith(PROFILE_PREFIX):
                pid = model[len(PROFILE_PREFIX):]
                selection, voice_route = resolve_profile(
                    settings, pid,
                    sample_store=app.state.sample_store, b64_cache=app.state.b64_cache,
                )
                voice_route = dict(voice_route)
            elif model.startswith(VOICE_ROUTE_PREFIX):
                target = model[len(VOICE_ROUTE_PREFIX):]
                if target == "auto" or not target:
                    # late binding: accept, no backends bound until the first
                    # voice-carrying session.update resolves the full profile
                    selection = pending_selection(settings)
                    route_pending = True
                    voice_route = {
                        "mode": "auto",
                        "requested": "",
                        "profile": "",
                        "revision": 0,
                    }

                    def _resolver(voice: str, _settings=settings):
                        return resolve_voice_route_full(
                            _settings, voice,
                            sample_store=app.state.sample_store, b64_cache=app.state.b64_cache,
                        )

                    route_resolver = _resolver
                else:
                    selection, voice_route = resolve_voice_route_full(
                        settings, target,
                        sample_store=app.state.sample_store, b64_cache=app.state.b64_cache,
                    )
                    voice_route = dict(voice_route)
            else:
                # a client naming no set: the set for other apps
                selection, voice_route = resolve_default_set(
                    settings, sample_store=app.state.sample_store, b64_cache=app.state.b64_cache,
                )
                voice_route = dict(voice_route)
                voice_route["requested"] = model
        except SelectionError as exc:
            reason = sanitize_close_reason(str(exc))
            await _refuse(websocket, 404, getattr(exc, "code", None) or "unknown_model", reason, close_code=4404)
            return
        assert selection is not None
        await websocket.accept(subprotocol=ws_subprotocol_to_accept(websocket))
        if route_pending:
            # no STT/LLM/TTS backend may be bound before the route resolves
            backends = Backends()
        else:
            backends = factory(selection.settings, app.state.http_client)
        profile_mode = bool(voice_route and voice_route.get("mode") == "profile")
        session = RealtimeSession(
            backends=backends,
            settings=selection.settings,
            model=model,
            voice_frozen=profile_mode,
            resolved_info=None if route_pending else selection.info(),
            voice_route=voice_route if not route_pending else None,
            route_pending=route_pending,
            route_resolver=route_resolver,
            full_backend_factory=(
                (lambda s: factory(s, app.state.http_client)) if route_pending else None
            ),
            closed_cb=lambda sid: app.state.sessions.pop(sid, None),
            warnings=connect_warnings,
        )
        app.state.sessions[session.id] = session

        async def sender() -> None:
            async for event in session.events():
                await websocket.send_json(event)

        sender_task = asyncio.get_running_loop().create_task(sender())
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    session._emit_error(
                        "binary frames are not supported; send JSON events",
                        code="binary_frame_not_supported",
                    )
                    continue
                try:
                    event = json.loads(text)
                except ValueError:
                    session._emit_error("invalid JSON frame", code="invalid_json")
                    continue
                if isinstance(event, dict):
                    session.handle_event(event)
                else:
                    session._emit_error("event must be a JSON object", code="invalid_json")
        except WebSocketDisconnect:
            pass
        finally:
            # External teardown (TestClient, server shutdown) may cancel this
            # task mid-cleanup; keep the cleanup itself running shielded so
            # no task or async generator is destroyed while pending.
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender_task
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(session.aclose())
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(backends.aclose())

    if web and web.exists():
        # `/` leads to the management UI; the static mount below also serves
        # the shared audio worklet the UI loads
        @app.get("/", include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/", status_code=307)

        app.mount("/", NoCacheStaticFiles(directory=str(web), html=True), name="web")

    return app
