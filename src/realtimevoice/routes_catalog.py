"""Models API: the model entries sets listen, think and speak with.

Every route requires the bearer token. All outward views mask secrets
(api_key and secret-looking extra values); requests carrying the mask keep
the stored secret.
"""

from __future__ import annotations

import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import ValidationError

from .audio import pcm16_to_wav
from .auth import require_http_auth
from .backends.openai_http import OpenAIHTTPLLM, OpenAIHTTPSTT, OpenAIHTTPTTS
from .catalog import (
    mask_profile,
    is_masked_secret,
    merge_profile_masked,
)
from .config import (
    PROFILE_ID_RE,
    ROLES,
    BackendProfile,
    Settings,
    _model_id,
    _validation_message,
    is_sensitive_extra_key,
    reference_fingerprint,
)
from .engine_groups import group_engine_view, group_engines
from .errors import BackendError
from .logfilter import sanitize_error_text
from .profiles import breaks_sets_error, broken_sets, profile_backend_settings, sets_using

router = APIRouter(prefix="/api")

PROBE_NOTE = (
    "model list comes from GET /models; voices/language/streaming are not "
    "verified by this probe"
)


class ApiError(ValueError):
    """ValueError with an HTTP status and machine-readable code (the app's
    ValueError handler turns it into {"detail", "code"})."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _guard(request: Request) -> None:
    require_http_auth(request)


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")


def _conflict(code: str, message: str) -> ApiError:
    return ApiError(409, code, message)


# ------------------------------------------------------------------ catalog


@router.get("/catalog")
async def get_catalog(request: Request) -> dict[str, Any]:
    _guard(request)
    settings = await request.app.state.settings_store.get()
    roles: dict[str, Any] = {}
    for role in ROLES:
        profiles = sorted(settings.profiles.get(role, {}).values(), key=lambda p: p.id)
        # Round 2 / CP1: every stored profile id stays in `profiles` (no data
        # change, exact id sets preserved). `engine_groups` is ADDITIVE
        # presentation metadata: one entry per engine, listing its row ids so
        # the selectors can show each engine once without losing any id.
        groups = group_engines(profiles)
        roles[role] = {
            "profiles": [_profile_view(p, settings) for p in profiles],
            "engine_groups": [group_engine_view(g, role=role) for g in groups],
        }
    return {"roles": roles}


def _profile_view(profile: BackendProfile, settings: Settings | None = None) -> dict[str, Any]:
    """Masked outward view plus SAFE reference metadata: presence and
    fingerprint only — never the stored secret/reference data itself — and
    the sets that use it."""
    view = mask_profile(profile)
    if settings is not None:
        view["used_by"] = sets_using(settings, role=profile.role, profile_id=profile.id)
    ref = reference_fingerprint(profile.extra)
    if ref:
        view["ref_sha256"] = ref
    sensitive = sorted(str(k) for k in (profile.extra or {}) if is_sensitive_extra_key(str(k)))
    if sensitive:
        view["sensitive_extra_keys"] = sensitive
    return view


def _validation_value_error(exc: Exception) -> ValueError:
    from .config import _validation_message

    return ValueError(_validation_message(exc) if hasattr(exc, "errors") else str(exc))


@router.get("/tts-engines")
async def get_tts_engines(request: Request) -> dict[str, Any]:
    """TTS engines with honestly-labelled cloning capabilities.

    Round 2 / CP1 (Option A): rows that materialise the SAME engine (same
    adapter + endpoint + model + engine params) are presented ONCE, so
    `chatterbox-clara` and `chatterbox-jonas` no longer appear as two
    engines. The stored rows and ids are untouched; every top-level field is
    kept and `variants` (the grouped row ids) is added. Metadata (model,
    capabilities, evidence, languages, streaming) comes from the row that is
    actually used — the primary row — never a sibling's better metadata (C4).

    The voice editor lists ONLY `reference_audio` engines for cloning preview;
    `unknown` is shown disabled ("capability not verified"); `none` is never
    offered. This endpoint lets the UI avoid 409 round trips."""
    _guard(request)
    settings = await request.app.state.settings_store.get()
    profiles = sorted(settings.profiles.get("tts", {}).values(), key=lambda p: p.id)
    engines = [
        group_engine_view(group, role="tts")
        for group in group_engines(profiles)
    ]
    return {"engines": engines}


# ----------------------------------------------------------------- profiles


def _new_id(settings: Settings, role: str, body: dict[str, Any]) -> str:
    """A new entry's id: the one given, else derived from its model name."""
    taken = set(settings.profiles.get(role, {}))
    given = body.get("id")
    if isinstance(given, str) and given:
        if not PROFILE_ID_RE.match(given):
            raise ValueError("model id must match [a-z0-9][a-z0-9._-]{0,63}")
        if given in taken:
            raise _conflict("profile_exists", f"model '{given}' already exists")
        return given
    return _model_id(str(body.get("model") or ""), str(body.get("display_name") or ""), taken)


@router.post("/profiles/{role}")
async def create_profile(role: str, request: Request) -> dict[str, Any]:
    _guard(request)
    _check_role(role)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("profile payload must be a JSON object")
    data = dict(body)
    data.pop("version", None)
    created: list[str] = []

    def mutate(settings: Settings) -> None:
        pid = _new_id(settings, role, data)
        data["id"] = pid
        data["role"] = role
        settings.profiles.setdefault(role, {})[pid] = BackendProfile.model_validate(data)
        created.append(pid)

    settings = await request.app.state.settings_store.mutate(mutate)
    return _profile_view(settings.profiles[role][created[0]], settings)


@router.put("/profiles/{role}/{profile_id}")
async def update_profile(role: str, profile_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    _check_role(role)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("profile payload must be a JSON object")
    store = request.app.state.settings_store
    sample_store = request.app.state.sample_store
    before = await store.get()

    def mutate(settings: Settings) -> None:
        role_profiles = settings.profiles.setdefault(role, {})
        current = role_profiles.get(profile_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"profile '{profile_id}' not found")
        # C3 (security): a changed endpoint must not silently keep sending the
        # stored secret there. If the endpoint changes and no new key is sent,
        # refuse unless the request explicitly confirms keeping the stored key.
        if "base_url" in body and not _same_endpoint(body.get("base_url"), current.base_url):
            new_key = body.get("api_key")
            confirmed = body.get("keep_api_key_for_new_endpoint") is True
            if is_masked_secret(new_key) and not confirmed:
                raise _conflict(
                    "key_required_for_changed_endpoint",
                    "The endpoint changed — enter the API key for the new endpoint, "
                    "or confirm keeping the existing key for it.",
                )
        # control-only; never stored on the row
        clean_body = {k: v for k, v in body.items() if k != "keep_api_key_for_new_endpoint"}
        role_profiles[profile_id] = merge_profile_masked(current, clean_body)
        # a saved set speaks with this model at once: refuse a change that
        # would stop a switched-on set from working
        broken = broken_sets(before, settings, sample_store=sample_store)
        if broken:
            raise breaks_sets_error(broken)

    settings = await store.mutate(mutate)
    return _profile_view(settings.profiles[role][profile_id], settings)


@router.post("/profiles/{role}/{profile_id}/clone", status_code=201)
async def clone_profile(role: str, profile_id: str, request: Request) -> dict[str, Any]:
    """Duplicate a backend profile server-side, keeping the secret server-side.

    The clone copies the source profile's `api_key` and reference material
    (raw `extra`) verbatim from the STORED profile, so an operator never has
    to retype a credential. Only non-secret overrides (new id/display name/
    model) are accepted from the client; a client-supplied `api_key` is
    ignored. Returns the masked profile view."""
    _guard(request)
    _check_role(role)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("clone payload must be a JSON object")
    created: list[str] = []

    def mutate(settings: Settings) -> None:
        role_profiles = settings.profiles.setdefault(role, {})
        source = role_profiles.get(profile_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"profile '{profile_id}' not found")
        data = source.model_dump()
        new_id = _new_id(settings, role, {"model": body.get("model") or source.model, **body})
        data["id"] = new_id
        data["role"] = role
        created.append(new_id)
        # only non-secret metadata may be overridden; api_key/extra (which may
        # carry reference audio) always come from the stored source
        for key in ("display_name", "model", "language_format", "voice"):
            if key in body:
                data[key] = body[key]
        if "capabilities" in body and isinstance(body["capabilities"], dict):
            caps = dict(data.get("capabilities") or {})
            caps.update(body["capabilities"])
            data["capabilities"] = caps
        role_profiles[new_id] = BackendProfile.model_validate(data)

    settings = await request.app.state.settings_store.mutate(mutate)
    return _profile_view(settings.profiles[role][created[0]], settings)


@router.delete("/profiles/{role}/{profile_id}")
async def delete_profile(role: str, profile_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    _check_role(role)

    def mutate(settings: Settings) -> None:
        role_profiles = settings.profiles.setdefault(role, {})
        if profile_id not in role_profiles:
            raise HTTPException(status_code=404, detail=f"profile '{profile_id}' not found")
        users = sets_using(settings, role=role, profile_id=profile_id)
        if users:
            names = ", ".join(settings.conversation_profiles[u].display_name or u for u in users)
            raise _conflict("model_in_use", f"{names} use this model; choose another model there first")
        if role == "stt" and settings.voice_library.transcription_stt_profile == profile_id:
            raise _conflict(
                "model_in_use",
                "recordings are transcribed with this model; choose another one in Settings first",
            )
        del role_profiles[profile_id]

    await request.app.state.settings_store.mutate(mutate)
    return {"deleted": profile_id}


@router.post("/probe-profile")
async def probe_profile(request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    role = body.get("role")
    pid = body.get("id")
    _check_role(str(role) if role else "")
    settings = await request.app.state.settings_store.get()
    stored = settings.profiles.get(role, {}).get(pid) if pid else None

    # R2 / CP3: probe an UNSAVED row. The add-backend form checks the connection
    # BEFORE saving; the body carries the form's `profile` object.
    #
    # C1 (security): a stored key is attached ONLY when this is an edit of the
    # row's OWN, UNCHANGED endpoint. Otherwise an operator (or anyone holding
    # the UI token) could point an existing row at an arbitrary host and make
    # the server send that row's secret there. A NEW backend (no top-level id)
    # never receives a stored key, even when its derived id collides.
    raw_profile = body.get("profile")
    if isinstance(raw_profile, dict):
        data = dict(raw_profile)
        data.pop("role", None)
        data.setdefault("role", role)
        # an unsaved row has no id yet; BackendProfile requires a valid one, so
        # validate against a neutral placeholder when none was derived.
        data.setdefault("id", pid or "")
        if not isinstance(data.get("id"), str) or not data.get("id"):
            data["id"] = "probe-unsaved"
        key_attached = False
        if not data.get("api_key"):
            source = stored if (stored is not None and stored.id == data["id"]) else None
            # only an explicit EDIT (top-level id names the row) may use the row's
            # key, and only for the row's own endpoint
            if source is not None and pid and data["id"] == pid:
                if _same_endpoint(data.get("base_url"), source.base_url):
                    data["api_key"] = source.api_key
                    key_attached = True
        if not data.get("base_url"):
            return {"ok": False, "detail": f"no {role} base_url configured", "models": [], "note": PROBE_NOTE}
        try:
            profile = BackendProfile.model_validate(data)
        except ValidationError as exc:
            return {"ok": False, "detail": _validation_message(exc), "models": [], "note": PROBE_NOTE}
        result = await _probe_profile_row(request, role, profile)
        # a key-less probe that meets an auth wall on a changed/new endpoint gets
        # a plain, secret-free explanation instead of the raw HTTP line
        if (
            not key_attached
            and isinstance(result, dict)
            and not result.get("ok")
            and _http_status(result.get("detail", "")) in (401, 403)
        ):
            return {
                **result,
                "error_code": "key_required_for_changed_endpoint",
                "detail": "The endpoint changed — enter the API key to check it.",
            }
        return result

    if stored is None:
        raise HTTPException(status_code=404, detail=f"profile '{pid}' not found")
    if not stored.base_url:
        return {"ok": False, "detail": f"no {role} base_url configured", "models": [], "note": PROBE_NOTE}
    return await _probe_profile_row(request, role, stored)


def _normalise_endpoint(url: str | None) -> tuple[str, str, int | None, str] | None:
    """(scheme, lowercased host, port, path) with a trailing `/` ignored.

    Port `None` means the scheme default. Returns None for an unusable URL, so
    an unparseable endpoint is never treated as "the same" as a stored one.
    """
    if not url:
        return None
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return (parts.scheme.lower(), parts.hostname.lower(), parts.port, path)


def _same_endpoint(candidate: str | None, stored: str | None) -> bool:
    """True when two endpoints are the same host/port/path after normalisation.

    Trailing `/` differences and host case differences count as unchanged; any
    change of scheme, host, port or path does not.
    """
    a = _normalise_endpoint(candidate)
    b = _normalise_endpoint(stored)
    return a is not None and a == b


def _http_status(detail: str) -> int:
    """The HTTP status the adapter reported, or 0.

    The adapter's probe detail is `HTTP <code> from <url>` (openai_http._probe);
    this extracts just the code so a key-less auth wall can be explained.
    """
    match = re.search(r"HTTP\s+(\d{3})", str(detail or ""))
    return int(match.group(1)) if match else 0


async def _probe_profile_row(request: Request, role: str, profile: BackendProfile) -> dict[str, Any]:
    backend_settings = profile_backend_settings(profile)
    client = request.app.state.http_client
    if role == "stt":
        backend = OpenAIHTTPSTT(backend_settings, client)
    elif role == "llm":
        backend = OpenAIHTTPLLM(backend_settings, client)
    else:
        backend = OpenAIHTTPTTS(backend_settings, client)
    try:
        result = await backend.probe()
    except Exception as exc:  # redaction happens inside the adapters
        return {"ok": False, "detail": f"probe failed: {exc}", "models": [], "note": PROBE_NOTE}
    return {
        "ok": result.ok,
        "detail": result.detail,
        "models": result.models,
        "error_code": getattr(result, "error_code", ""),
        "note": PROBE_NOTE,
    }


# ---------------------------------------------------------------- synthesis

MAX_TTS_TEXT_CHARS = 1000
MAX_TTS_WAV_BYTES = 40 * 1024 * 1024


@router.post("/tts-synthesis")
async def tts_synthesis(request: Request):
    """Direct TTS synthesis from ONE tts profile, server-side.

    Used by the controlled TTS comparison: credentials and voice-reference
    material (extra.ref_audio / extra.ref_text) stay on the server; the
    response is a raw WAV plus safe proof headers (profile id, model,
    effective voice, echoed text id). No settings mutation.

    X-Rtv-First-Pcm-Ms is measured at the server PCM decoder boundary (first
    nonempty adapter PCM after WAV decode/resample) — NOT acoustic playback;
    browser buffering/VAD are not included.
    """
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("synthesis payload must be a JSON object")
    pid = str(body.get("profile_id") or "")
    text = str(body.get("text") or "")
    text_id = str(body.get("text_id") or "")
    voice = str(body.get("voice") or "")
    language = str(body.get("language") or "")
    if not pid:
        raise ValueError("profile_id is required")
    if not text.strip():
        raise ValueError("text is required")
    if len(text) > MAX_TTS_TEXT_CHARS:
        raise ValueError(f"text too long: {len(text)} > {MAX_TTS_TEXT_CHARS} characters")
    # user-controlled proof metadata must be bounded safe ASCII: header
    # values with CRLF/unicode would break the response (HTTP 500)
    if text_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", text_id):
        raise ValueError("text_id must match [A-Za-z0-9._-]{1,64}")

    settings = await request.app.state.settings_store.get()
    profile = settings.profiles.get("tts", {}).get(pid)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile '{pid}' not found")
    effective_voice = voice or profile.voice
    if effective_voice and effective_voice != "default" and profile.capabilities.voices:
        if effective_voice not in profile.capabilities.voices:
            raise ApiError(
                422,
                "voice_mismatch",
                f"voice '{effective_voice}' is not one of this model's voices "
                f"({', '.join(profile.capabilities.voices)})",
            )

    backend_settings = profile_backend_settings(profile, language)
    client = request.app.state.http_client
    backend = OpenAIHTTPTTS(backend_settings, client)
    pcm_parts: list[bytes] = []
    total = 0
    first_pcm_ms: int | None = None
    started = time.perf_counter()
    try:
        async for pcm in backend.stream_speech(
            text, voice=effective_voice or None, language=language or None
        ):
            if first_pcm_ms is None and pcm:
                # backend TTFA proxy: first nonempty PCM from the adapter
                first_pcm_ms = int((time.perf_counter() - started) * 1000)
            total += len(pcm)
            if total > MAX_TTS_WAV_BYTES:
                raise ValueError("synthesis output exceeded the size limit")
            pcm_parts.append(pcm)
    except Exception as exc:
        if isinstance(exc, BackendError):
            raise ApiError(502, "tts_backend_error", exc.message) from exc
        # never surface raw backend/transport text: sanitize (URLs/hosts,
        # keys, bearer, base64 blobs); fall back to the class name only
        detail = sanitize_error_text(str(exc))[:300] or exc.__class__.__name__
        raise ApiError(502, "tts_synthesis_failed", f"synthesis failed: {detail}") from exc
    synth_ms = int((time.perf_counter() - started) * 1000)
    wav = pcm16_to_wav(b"".join(pcm_parts), 24000)
    headers = {
        "X-Rtv-Profile-Id": _safe_header_value(profile.id),
        "X-Rtv-Model": _safe_header_value(profile.model),
        "X-Rtv-Voice": _safe_header_value(effective_voice),
        "X-Rtv-Text-Id": _safe_header_value(text_id),
        "X-Rtv-First-Pcm-Ms": "" if first_pcm_ms is None else str(first_pcm_ms),
        "X-Rtv-Synth-Ms": str(synth_ms),
        "Cache-Control": "no-store",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


def _safe_header_value(value: str) -> str:
    """Proof headers carry server/profile/user metadata; enforce bounded
    printable ASCII so CRLF/unicode can never break the response."""
    return "".join(ch if 0x20 <= ord(ch) <= 0x7E else "?" for ch in value)[:128]


def register_catalog_routes(app) -> None:
    app.include_router(router)
