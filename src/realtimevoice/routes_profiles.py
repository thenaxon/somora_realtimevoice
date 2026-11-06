"""Sets & voice library REST API.

Every route requires the bearer token. Responses never carry credentials or
reference material; only safe proof metadata is returned.

A saved set is what calls use. A change elsewhere that would stop a
switched-on set from working (a model edit, a new reference recording, an
edited reference transcript) is refused with 409 `breaks_sets`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError

from .auth import require_http_auth
from .backends.openai_http import OpenAIHTTPSTT, OpenAIHTTPTTS
from .config import (
    PROFILE_ID_RE,
    SET_HISTORY_MAX,
    BackendProfile,
    ConversationProfile,
    ProfileSpec,
    SetVersion,
    Settings,
    Voice,
)
from .logfilter import sanitize_error_text
from .profiles import (
    blocking_issues,
    breaks_sets_error,
    broken_sets,
    compatibility_warnings,
    profile_backend_settings,
    resolve_test,
    sets_using,
    validate_spec,
)
from .symbol_filter import strip_emoji_for_speech
from .voices import (
    SampleValidationError,
    canonicalise_wav,
    gc_eligible,
    new_sample,
    speech_duration_ms,
)

router = APIRouter(prefix="/api")

PROBE_TIMEOUT_S = 10.0
MAX_PROBE_CONCURRENCY = 3
# N-3: fewer than this many voiced milliseconds means "no speech" and the
# sample is never sent to STT (silence makes STT models hallucinate)
MIN_SPEECH_MS_FOR_STT = 300.0


class ApiError(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _guard(request: Request) -> None:
    require_http_auth(request)


def _conflict(code: str, message: str) -> ApiError:
    return ApiError(409, code, message)


def _state(request: Request) -> tuple[Any, Any, Any]:
    app = request.app
    return app.state.settings_store, app.state.sample_store, app.state.b64_cache


# ---------------------------------------------------------------- masking

def _profile_view(settings: Settings, profile: ConversationProfile) -> dict[str, Any]:
    view = profile.model_dump()
    view["is_default"] = profile.id == settings.default_set
    # what would stop it from working now (no file reads: cheap for lists)
    view["issues"] = blocking_issues(validate_spec(settings, profile.spec))
    view["compatibility_warnings"] = compatibility_warnings(settings, profile.spec)
    return view


def _set_invalid(issues: list[dict[str, Any]], what: str) -> ApiError:
    exc = ApiError(
        409,
        "set_invalid",
        f"{what}: " + "; ".join(str(i.get("message", i.get("code"))) for i in issues),
    )
    exc.issues = issues  # type: ignore[attr-defined]
    return exc


def _recheck(request: Request, settings: Settings, spec: ProfileSpec, what: str) -> None:
    """The validity check again, against the settings the save commits to:
    the models or the recording may have changed while the probes ran."""
    issues = blocking_issues(
        validate_spec(settings, spec, sample_store=request.app.state.sample_store, check_files=True)
    )
    if issues:
        raise _set_invalid(issues, what)


def _parse_spec(raw: Any) -> ProfileSpec:
    try:
        return ProfileSpec.model_validate(raw or {})
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from exc


# ------------------------------------------------------------- probes

async def _probe_role(request: Request, role: str, pid: str, settings: Settings) -> dict[str, Any]:
    profile = settings.profiles.get(role, {}).get(pid)
    started = time.perf_counter()
    timeout_ms = int(PROBE_TIMEOUT_S * 1000)
    if profile is None:
        return {"ok": False, "latency_ms": 0, "error_code": "unknown_profile", "timeout_ms": timeout_ms}
    if not profile.base_url:
        return {"ok": False, "latency_ms": 0, "error_code": "no_base_url", "timeout_ms": timeout_ms}
    backend_settings = _profile_to_backend_settings(profile)
    client = request.app.state.http_client
    if role == "stt":
        backend = OpenAIHTTPSTT(backend_settings, client)
    elif role == "llm":
        from .backends.openai_http import OpenAIHTTPLLM

        backend = OpenAIHTTPLLM(backend_settings, client)
    else:
        backend = OpenAIHTTPTTS(backend_settings, client)
    try:
        result = await asyncio.wait_for(backend.probe(), timeout=PROBE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_code": "backend_timeout",
            "timeout_ms": timeout_ms,
        }
    except Exception as exc:  # adapters redact their own text
        return {
            "ok": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_code": "backend_unreachable",
            "detail": sanitize_error_text(str(exc))[:200],
            "timeout_ms": timeout_ms,
        }
    error_code = ""
    if not result.ok:
        # P1: prefer the adapter's machine-readable reason (e.g.
        # model_not_listed) over the generic fallback
        error_code = getattr(result, "error_code", "") or "probe_failed"
    return {
        "ok": bool(result.ok),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error_code": error_code,
        "timeout_ms": timeout_ms,
        # the configured model id, so the UI can name it in the
        # model_not_listed message ("model <id> is not in its model list")
        "model": profile.model,
    }


async def _probe_spec(request: Request, settings: Settings, spec: ProfileSpec) -> dict[str, Any]:
    results = await asyncio.gather(
        *[ _probe_role(request, role, getattr(spec, f"{role}_profile"), settings) for role in ("stt", "llm", "tts") ]
    )
    return dict(zip(("stt", "llm", "tts"), results))


def _profile_to_backend_settings(profile: BackendProfile):
    return profile_backend_settings(profile)


# ============================================================ conversation

@router.get("/conversation-profiles")
async def list_conversation_profiles(request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)
    settings = await store.get()
    return {
        "profiles": [
            _profile_view(settings, p)
            for p in sorted(settings.conversation_profiles.values(), key=lambda p: p.id)
        ]
    }


async def _checked_spec(request: Request, settings: Settings, spec: ProfileSpec, what: str) -> dict[str, Any]:
    """Refuse a spec that cannot work (409 set_invalid); otherwise probe its
    three models. A probe failure is reported, never blocking: an offline
    server must not prevent saving a set."""
    issues = blocking_issues(
        validate_spec(settings, spec, sample_store=request.app.state.sample_store, check_files=True)
    )
    if issues:
        raise _set_invalid(issues, what)
    return await _probe_spec(request, settings, spec)


@router.post("/conversation-profiles", status_code=201)
async def create_conversation_profile(request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("set payload must be a JSON object")
    pid = body.get("id")
    if not isinstance(pid, str) or not pid:
        raise ValueError("id is required")
    if not PROFILE_ID_RE.match(pid):
        raise ValueError("id must match [a-z0-9][a-z0-9-]{1,62}")
    spec = _parse_spec(body.get("spec"))
    enabled = bool(body.get("enabled", False))
    store, _, _ = _state(request)
    probes: dict[str, Any] = {}
    if enabled:
        probes = await _checked_spec(request, await store.get(), spec, "the set cannot be switched on")

    def mutate(settings: Settings) -> None:
        if pid in settings.conversation_profiles:
            raise _conflict("conversation_profile_exists", f"a set called '{pid}' already exists")
        if enabled:
            _recheck(request, settings, spec, "the set cannot be switched on")
        now = time.time()
        settings.conversation_profiles[pid] = ConversationProfile(
            id=pid,
            display_name=str(body.get("display_name") or pid),
            description=str(body.get("description") or ""),
            spec=spec,
            enabled=enabled,
            created_at=now,
            updated_at=now,
            spec_saved_at=now,
        )

    settings = await store.mutate(mutate)
    view = _profile_view(settings, settings.conversation_profiles[pid])
    if probes:
        view["probes"] = probes
    return view


@router.get("/conversation-profiles/{profile_id}")
async def get_conversation_profile(profile_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
    return _profile_view(settings, profile)


@router.put("/conversation-profiles/{profile_id}")
async def update_conversation_profile(profile_id: str, request: Request) -> dict[str, Any]:
    """Save a set: what calls use from now on (running calls keep theirs).
    A switched-on set must be valid; the old spec goes to its versions."""
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("set payload must be a JSON object")
    store, _, _ = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
    spec = _parse_spec(body["spec"]) if "spec" in body else profile.spec
    enabled = bool(body["enabled"]) if "enabled" in body else profile.enabled
    probes: dict[str, Any] = {}
    if enabled:
        probes = await _checked_spec(
            request, settings, spec,
            "the set cannot be saved while it is on" if profile.enabled else "the set cannot be switched on",
        )

    def mutate(settings: Settings) -> None:
        profile = settings.conversation_profiles.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
        # fields the request leaves out keep their CURRENT value (another save
        # may have changed them while the probes ran)
        new_spec = spec if "spec" in body else profile.spec
        new_enabled = enabled if "enabled" in body else profile.enabled
        if new_enabled:
            _recheck(request, settings, new_spec, "the set cannot be saved while it is on")
        if "display_name" in body:
            profile.display_name = str(body["display_name"]) or profile_id
        if "description" in body:
            profile.description = str(body["description"])
        now = time.time()
        if new_spec != profile.spec:
            profile.history.insert(0, SetVersion(saved_at=profile.spec_saved_at or profile.updated_at, spec=profile.spec))
            profile.history = profile.history[:SET_HISTORY_MAX]
            profile.spec = new_spec
            profile.spec_saved_at = now
        profile.enabled = new_enabled
        profile.updated_at = now

    settings = await store.mutate(mutate)
    view = _profile_view(settings, settings.conversation_profiles[profile_id])
    if probes:
        view["probes"] = probes
    return view


@router.delete("/conversation-profiles/{profile_id}")
async def delete_conversation_profile(profile_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)

    def mutate(settings: Settings) -> None:
        profile = settings.conversation_profiles.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
        if profile.enabled:
            raise _conflict(
                "conversation_profile_active",
                f"set '{profile_id}' is on; switch it off first",
            )
        del settings.conversation_profiles[profile_id]
        if settings.default_set == profile_id:
            settings.default_set = ""

    await store.mutate(mutate)
    return {"deleted": profile_id}


@router.post("/conversation-profiles/{profile_id}/validate")
async def validate_conversation_profile(profile_id: str, request: Request) -> dict[str, Any]:
    """Check a set (as saved, or the `spec` in the body) without saving it."""
    _guard(request)
    body = await request.json() if await request.body() else {}
    store, sample_store, _ = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
    spec = _parse_spec(body["spec"]) if isinstance(body, dict) and body.get("spec") is not None else profile.spec
    issues = validate_spec(settings, spec, sample_store=sample_store, check_files=True)
    probes = await _probe_spec(request, settings, spec)
    return {"ok": not blocking_issues(issues), "issues": issues, "probes": probes}


@router.post("/conversation-profiles/{profile_id}/restore")
async def restore_conversation_profile(profile_id: str, request: Request) -> dict[str, Any]:
    """Make an earlier version (`index` into versions, 0 = newest) the set
    again; the current one goes to the versions."""
    _guard(request)
    body = await request.json() if await request.body() else {}
    index = int((body or {}).get("index") or 0)
    store, _, _ = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
    if not 0 <= index < len(profile.history):
        raise _conflict("no_history", f"set '{profile_id}' has no version {index}")
    target = profile.history[index].spec
    target_saved_at = profile.history[index].saved_at
    probes: dict[str, Any] = {}
    if profile.enabled:
        probes = await _checked_spec(request, settings, target, "that version cannot be used")

    def mutate(settings: Settings) -> None:
        profile = settings.conversation_profiles.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
        # the versions may have moved while the probes ran: restore exactly
        # the version that was chosen, or nothing
        if not (
            0 <= index < len(profile.history)
            and profile.history[index].spec == target
            and profile.history[index].saved_at == target_saved_at
        ):
            raise _conflict("history_changed", "the set was saved meanwhile; reload it and choose the version again")
        if profile.enabled:
            _recheck(request, settings, target, "that version cannot be used")
        restored = profile.history.pop(index)
        profile.history.insert(0, SetVersion(saved_at=profile.spec_saved_at or profile.updated_at, spec=profile.spec))
        profile.history = profile.history[:SET_HISTORY_MAX]
        profile.spec = restored.spec
        profile.updated_at = profile.spec_saved_at = time.time()

    settings = await store.mutate(mutate)
    view = _profile_view(settings, settings.conversation_profiles[profile_id])
    if probes:
        view["probes"] = probes
    return view


@router.post("/conversation-profiles/{profile_id}/test-session")
async def create_test_session(profile_id: str, request: Request) -> dict[str, Any]:
    """A ticket for a test call with the set as the editor shows it (`spec`,
    never persisted) or as saved; the set need not be on."""
    _guard(request)
    body = await request.json() if await request.body() else {}
    store, sample_store, _ = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"set '{profile_id}' not found")
    unsaved_spec: ProfileSpec | None = None
    if isinstance(body, dict) and body.get("spec") is not None:
        unsaved_spec = _parse_spec(body["spec"])
    tested = unsaved_spec if unsaved_spec is not None else profile.spec
    issues = blocking_issues(validate_spec(settings, tested, sample_store=sample_store, check_files=True))
    if issues:
        exc = _conflict(
            "set_incomplete",
            "the set is incomplete and cannot be tested: "
            + "; ".join(str(i.get("message", i.get("code"))) for i in issues),
        )
        exc.issues = issues  # type: ignore[attr-defined]
        raise exc
    ticket = request.app.state.profile_tests.create(profile_id, spec=unsaved_spec)
    expires_at = time.time() + request.app.state.profile_tests.ttl_s
    return {"model": f"profile-test:{ticket}", "expires_at": expires_at}


# R2 / CP2: the audible "test this sentence" path. It speaks the typed text
# with the CURRENT (possibly unsaved) selection through EXACTLY the
# conversation materialisation, so what the operator hears is what a real
# conversation would send. No STT, no LLM, nothing persisted.
MAX_TEST_SPEECH_CHARS = 1000
MAX_TEST_SPEECH_BYTES = 40 * 1024 * 1024
TEST_SPEECH_SAMPLE_RATE = 24000


@router.post("/conversation-profiles/{profile_id}/test-speech")
async def test_profile_speech(profile_id: str, request: Request) -> Response:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("test-speech payload must be a JSON object")
    text = str(body.get("text") or "")
    if not text.strip():
        raise ValueError("text is required")
    if len(text) > MAX_TEST_SPEECH_CHARS:
        raise ValueError(f"text too long: {len(text)} > {MAX_TEST_SPEECH_CHARS} characters")
    store, sample_store, b64_cache = _state(request)
    settings = await store.get()
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"conversation profile '{profile_id}' not found")

    # `spec` = the editor's CURRENT selection; unsaved allowed, never persisted.
    # Without it the saved set is used.
    spec = profile.spec
    if body.get("spec") is not None:
        spec = _parse_spec(body["spec"])

    issues = validate_spec(settings, spec, sample_store=sample_store, check_files=True)
    blocking = [i for i in issues if i.get("severity") != "warning"]
    if blocking:
        exc = _conflict(
            "set_incomplete",
            "the selection is incomplete and cannot be spoken: "
            + "; ".join(str(i.get("message", i.get("code"))) for i in blocking),
        )
        exc.issues = blocking  # type: ignore[attr-defined]
        raise exc

    # EXACTLY the conversation materialisation, reusing the code the test
    # session and `profile-test:<ticket>` connections use.
    try:
        selection, route = resolve_test(
            settings,
            profile_id,
            sample_store=sample_store,
            b64_cache=b64_cache,
            spec_override=spec,
        )
    except ValueError as exc:
        # a resolution failure is not a synthesis failure: surface it cleanly
        # (never a 500) and never call the engine
        raise ApiError(409, getattr(exc, "code", "profile_unresolved"), str(exc)) from exc

    tts_settings = selection.settings.tts
    voice = selection.settings.session.voice or None
    language = selection.settings.session.language or None
    # The conversation strips emoji sequences from the SPOKEN text (LLM output)
    # before synthesis. The typed sentence goes through the SAME filter so the
    # sample matches a conversation; nothing else is rewording it (no LLM).
    cleaned = strip_emoji_for_speech(text)
    spoken = cleaned.text
    if not spoken.strip():
        raise ApiError(
            409,
            "nothing_to_speak",
            "the sentence contains only symbols or emoji that the speech path skips; type a sentence with words",
        )

    backend = OpenAIHTTPTTS(tts_settings, request.app.state.http_client)
    pcm_parts: list[bytes] = []
    first_pcm_ms: int | None = None
    total = 0
    started = time.perf_counter()
    try:
        async for pcm in backend.stream_speech(spoken, voice=voice, language=language):
            if first_pcm_ms is None and pcm:
                first_pcm_ms = int((time.perf_counter() - started) * 1000)
            total += len(pcm)
            if total > MAX_TEST_SPEECH_BYTES:
                raise ValueError("test-speech output exceeded the size limit")
            pcm_parts.append(pcm)
    except Exception as exc:
        from .errors import BackendError

        if isinstance(exc, BackendError):
            raise ApiError(502, "tts_backend_error", exc.message) from exc
        detail = sanitize_error_text(str(exc))[:300] or exc.__class__.__name__
        raise ApiError(502, "tts_preview_failed", f"synthesis failed: {detail}") from exc
    total_ms = int((time.perf_counter() - started) * 1000)

    from .audio import pcm16_to_wav

    pcm = b"".join(pcm_parts)
    audio_ms = int(len(pcm) / 2 / TEST_SPEECH_SAMPLE_RATE * 1000)
    wav_out = pcm16_to_wav(pcm, TEST_SPEECH_SAMPLE_RATE)
    # Provenance: safe metadata only. voice kind + the honest display voice and,
    # for a library voice, the sha8 of the materialised reference the backend
    # actually received (never the row's embedded legacy reference). No host, no
    # secret.
    voice_kind = spec.voice.kind
    ref_sha8 = ""
    if voice_kind == "library":
        ref_sha8 = (route.get("voice") or {}).get("ref_sha256", "")[:8]
    headers = {
        "X-Rtv-First-Pcm-Ms": "" if first_pcm_ms is None else str(first_pcm_ms),
        "X-Rtv-Total-Ms": str(total_ms),
        "X-Rtv-Audio-Ms": str(audio_ms),
        "X-Rtv-Model": _safe_header(tts_settings.model),
        "X-Rtv-Voice-Kind": _safe_header(voice_kind),
        "X-Rtv-Voice": _safe_header(selection.display_voice),
        "X-Rtv-Sample-Sha8": _safe_header(ref_sha8),
        "X-Rtv-Reference-Sha8": _safe_header(ref_sha8),
        "X-Rtv-Emoji-Removed": str(cleaned.removed),
        "Cache-Control": "no-store",
    }
    return Response(content=wav_out, media_type="audio/wav", headers=headers)


# ================================================================ voices

def _voice_view(settings: Settings, voice: Voice) -> dict[str, Any]:
    view = voice.model_dump()
    # the sets that speak with this voice (as saved)
    view["used_by"] = [
        {"profile": pid, "display_name": settings.conversation_profiles[pid].display_name,
         "enabled": settings.conversation_profiles[pid].enabled}
        for pid in sets_using(settings, voice_id=voice.id)
    ]
    return view


async def _mutate_voice(request: Request, mutator) -> Settings:
    """Apply a voice change, refusing it (409 breaks_sets) when it would stop
    a switched-on set that speaks with the voice from working."""
    store, sample_store, _ = _state(request)
    before = await store.get()

    def mutate(settings: Settings) -> None:
        mutator(settings)
        broken = broken_sets(before, settings, sample_store=sample_store)
        if broken:
            raise breaks_sets_error(broken)

    return await store.mutate(mutate)


@router.get("/voices")
async def list_voices(request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)
    settings = await store.get()
    return {"voices": [_voice_view(settings, v) for v in sorted(settings.voices.values(), key=lambda v: v.id)]}


@router.post("/voices", status_code=201)
async def create_voice(request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("voice payload must be a JSON object")
    vid = body.get("id")
    if not isinstance(vid, str) or not vid:
        raise ValueError("id is required")
    if not PROFILE_ID_RE.match(vid):
        raise ValueError("id must match [a-z0-9][a-z0-9-]{1,62}")
    store, _, _ = _state(request)

    def mutate(settings: Settings) -> None:
        if vid in settings.voices:
            raise _conflict("voice_exists", f"voice '{vid}' already exists")
        settings.voices[vid] = Voice(
            id=vid,
            display_name=str(body.get("display_name") or vid),
            language=str(body.get("language") or ""),
            description=str(body.get("description") or ""),
            created_at=time.time(),
            updated_at=time.time(),
        )

    settings = await store.mutate(mutate)
    return _voice_view(settings, settings.voices[vid])


@router.get("/voices/{voice_id}")
async def get_voice(voice_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)
    settings = await store.get()
    voice = settings.voices.get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    return _voice_view(settings, voice)


@router.patch("/voices/{voice_id}")
async def update_voice(voice_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("voice payload must be a JSON object")
    store, _, _ = _state(request)

    def mutate(settings: Settings) -> None:
        voice = settings.voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
        for field in ("display_name", "language", "description"):
            if field in body:
                setattr(voice, field, str(body[field]))
        voice.updated_at = time.time()

    settings = await store.mutate(mutate)
    return _voice_view(settings, settings.voices[voice_id])


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)

    def mutate(settings: Settings) -> None:
        voice = settings.voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
        users = sets_using(settings, voice_id=voice_id)
        if users:
            names = ", ".join(settings.conversation_profiles[u].display_name or u for u in users)
            raise _conflict("voice_referenced", f"{names} speak with this voice; choose another voice there first")
        del settings.voices[voice_id]

    await store.mutate(mutate)
    request.app.state.sample_store.remove_voice_dir(voice_id)
    return {"deleted": voice_id}


# --------------------------------------------------------------- samples

def _find_sample(settings: Settings, voice_id: str, sample_id: str):
    voice = settings.voices.get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    sample = next((s for s in voice.samples if s.id == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"sample '{sample_id}' not found")
    return voice, sample


@router.post("/voices/{voice_id}/samples")
async def upload_sample(
    voice_id: str,
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
) -> dict[str, Any]:
    _guard(request)
    store, sample_store, _ = _state(request)
    if source not in ("upload", "mic"):
        raise ApiError(400, "invalid_source", "source must be 'upload' or 'mic'")
    settings = await store.get()
    if voice_id not in settings.voices:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    raw = await file.read()
    try:
        canonical = canonicalise_wav(raw)
    except SampleValidationError as exc:
        status = 413 if exc.code == "too_large" else (415 if exc.code == "unsupported_format" else 400)
        raise ApiError(status, exc.code, str(exc)) from exc
    sample = new_sample(canonical, source=source, original_filename=file.filename or "")
    sample_store.write(voice_id, canonical.sha256, canonical.data)

    def mutate(settings: Settings) -> None:
        voice = settings.voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
        if not any(s.sha256 == canonical.sha256 for s in voice.samples):
            voice.samples.append(sample)
        if not voice.reference_sample_id:
            voice.reference_sample_id = sample.id
        voice.updated_at = time.time()

    settings = await store.mutate(mutate)
    voice, stored = _find_sample(settings, voice_id, sample.id)
    return stored.model_dump()


@router.post("/voices/{voice_id}/samples/{sample_id}/excerpt")
async def create_sample_excerpt(voice_id: str, sample_id: str, request: Request) -> dict[str, Any]:
    """F10: cut [start_ms, end_ms) out of an existing sample into a NEW sample.

    The original sample, its file and its transcript are never modified. The
    excerpt is a separate content-addressed sample with an EMPTY, unconfirmed
    transcript — the original's text does not match the cut audio, so the
    excerpt goes through transcription and user confirmation like any new
    recording. It is not made the voice's reference automatically; the user
    chooses that explicitly.
    """
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("excerpt payload must be a JSON object")
    try:
        start_ms = int(body.get("start_ms"))
        end_ms = int(body.get("end_ms"))
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "invalid_range", "start_ms and end_ms must be integers") from exc
    store, sample_store, _ = _state(request)
    settings = await store.get()
    _, source_sample = _find_sample(settings, voice_id, sample_id)
    try:
        original = sample_store.read(voice_id, source_sample.sha256)
    except OSError:
        raise HTTPException(status_code=404, detail="sample audio file is missing")

    import io
    import wave

    from .voices import MIN_SAMPLE_SECONDS

    with wave.open(io.BytesIO(original)) as src:
        rate = src.getframerate()
        channels = src.getnchannels()
        width = src.getsampwidth()
        total_frames = src.getnframes()
        duration_ms = int(round(total_frames * 1000 / rate)) if rate else 0
        if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
            raise ApiError(
                400,
                "invalid_range",
                f"range {start_ms}-{end_ms} ms is outside the sample (0-{duration_ms} ms)",
            )
        if end_ms - start_ms < int(MIN_SAMPLE_SECONDS * 1000):
            raise ApiError(
                400,
                "excerpt_too_short",
                f"an excerpt must be at least {MIN_SAMPLE_SECONDS:.0f} s long",
            )
        start_frame = int(round(start_ms * rate / 1000))
        end_frame = min(total_frames, int(round(end_ms * rate / 1000)))
        src.setpos(start_frame)
        frames = src.readframes(end_frame - start_frame)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(width)
        dst.setframerate(rate)
        dst.writeframes(frames)
    try:
        canonical = canonicalise_wav(buf.getvalue())
    except SampleValidationError as exc:
        status = 413 if exc.code == "too_large" else (415 if exc.code == "unsupported_format" else 400)
        raise ApiError(status, exc.code, str(exc)) from exc

    base_name = source_sample.original_filename or source_sample.id
    label = f"excerpt {start_ms / 1000:.1f}-{end_ms / 1000:.1f} s of {base_name}"[:128]
    excerpt = new_sample(canonical, source=source_sample.source, original_filename=label)
    excerpt.derived_from = {
        "sample_id": source_sample.id,
        "sha256": source_sample.sha256,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }
    sample_store.write(voice_id, canonical.sha256, canonical.data)

    def mutate(settings: Settings) -> None:
        voice = settings.voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
        if not any(s.sha256 == canonical.sha256 for s in voice.samples):
            voice.samples.append(excerpt)
        voice.updated_at = time.time()

    settings = await store.mutate(mutate)
    _, stored = _find_sample(settings, voice_id, excerpt.id)
    return stored.model_dump()


@router.get("/voices/{voice_id}/samples/{sample_id}/audio")
async def get_sample_audio(voice_id: str, sample_id: str, request: Request) -> Response:
    _guard(request)
    store, sample_store, _ = _state(request)
    settings = await store.get()
    _, sample = _find_sample(settings, voice_id, sample_id)
    try:
        data = sample_store.read(voice_id, sample.sha256)
    except OSError:
        raise HTTPException(status_code=404, detail="sample audio file is missing")
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@router.patch("/voices/{voice_id}/samples/{sample_id}")
async def patch_sample(voice_id: str, sample_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("sample payload must be a JSON object")

    def mutate(settings: Settings) -> None:
        _, sample = _find_sample(settings, voice_id, sample_id)
        if "transcript" in body:
            sample.transcript = str(body["transcript"])
            sample.transcript_source = "manual"
            # a manual edit resets confirmation unless confirmed in the same request
            sample.transcript_confirmed = bool(body.get("transcript_confirmed", False))
        elif "transcript_confirmed" in body:
            sample.transcript_confirmed = bool(body["transcript_confirmed"])

    settings = await _mutate_voice(request, mutate)
    _, sample = _find_sample(settings, voice_id, sample_id)
    return sample.model_dump()


@router.delete("/voices/{voice_id}/samples/{sample_id}")
async def delete_sample(voice_id: str, sample_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    store, sample_store, _ = _state(request)

    def mutate(settings: Settings) -> None:
        voice, sample = _find_sample(settings, voice_id, sample_id)
        if voice.reference_sample_id == sample_id and sets_using(settings, voice_id=voice_id):
            raise _conflict(
                "sample_is_reference",
                "sets speak with this recording; choose another recording for the voice first",
            )
        voice.samples = [s for s in voice.samples if s.id != sample_id]
        if voice.reference_sample_id == sample_id:
            voice.reference_sample_id = voice.samples[0].id if voice.samples else ""
        voice.updated_at = time.time()
        if gc_eligible(settings, voice_id, sample.sha256):
            sample_store.delete(voice_id, sample.sha256)

    await store.mutate(mutate)
    return {"deleted": sample_id}


@router.put("/voices/{voice_id}/reference")
async def set_reference(voice_id: str, request: Request) -> dict[str, Any]:
    """Make a recording the voice's reference: every set that speaks with the
    voice uses it from the next call. Refused (409 breaks_sets) when a
    switched-on set could not use it (unconfirmed text, too long for its
    speaking model)."""
    _guard(request)
    body = await request.json()
    sample_id = str((body or {}).get("sample_id") or "")
    if not sample_id:
        raise ValueError("sample_id is required")

    def mutate(settings: Settings) -> None:
        voice, _ = _find_sample(settings, voice_id, sample_id)
        voice.reference_sample_id = sample_id
        voice.updated_at = time.time()

    settings = await _mutate_voice(request, mutate)
    return _voice_view(settings, settings.voices[voice_id])


# ------------------------------------------------------------ transcription

@router.post("/voices/{voice_id}/samples/{sample_id}/transcribe")
async def transcribe_sample(voice_id: str, sample_id: str, request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json() if await request.body() else {}
    if not isinstance(body, dict):
        body = {}
    store, sample_store, _ = _state(request)
    settings = await store.get()
    voice, sample = _find_sample(settings, voice_id, sample_id)
    stt_profile_id = str(body.get("stt_profile") or settings.voice_library.transcription_stt_profile or "")
    if not stt_profile_id:
        raise ApiError(
            409,
            "no_transcription_model",
            "no listening model is chosen to transcribe recordings; choose one in Settings",
        )
    profile = settings.profiles.get("stt", {}).get(stt_profile_id)
    if profile is None:
        raise ApiError(400, "unknown_stt_profile", f"unknown STT profile: {stt_profile_id}")
    language = voice.language or "de"
    backend_settings = profile_backend_settings(profile, language)
    model = profile.model
    timeout = min(max(1.0, profile.request_timeout_s or 60.0), 60.0)
    try:
        wav = sample_store.read(voice_id, sample.sha256)
    except OSError:
        raise HTTPException(status_code=404, detail="sample audio file is missing")

    backend = OpenAIHTTPSTT(backend_settings, request.app.state.http_client)
    started = time.perf_counter()
    error_code = ""
    text = ""
    state = "done"

    # N-3 pre-check: never send a silent sample to STT (models hallucinate on
    # silence). Fewer than 300 ms of voiced frames is treated as no speech.
    speech_ms = speech_duration_ms(wav)
    if speech_ms < MIN_SPEECH_MS_FOR_STT:
        error_code, state = "no_speech_detected", "failed"
        duration_ms = 0
        text = ""
    else:
        try:
            text = await asyncio.wait_for(backend.transcribe(wav, language=language), timeout=timeout)
        except asyncio.TimeoutError:
            error_code, state = "backend_timeout", "failed"
        except Exception as exc:
            from .errors import BackendError

            if isinstance(exc, BackendError):
                error_code = {"stt_timeout": "backend_timeout"}.get(exc.code, "stt_failed")
            else:
                error_code = "stt_failed"
            state = "failed"
        duration_ms = int((time.perf_counter() - started) * 1000)
        text = _strip_asr_prefix(text)
        # N-3 post-check: an STT that returns text for audio with no voiced
        # frames is hallucinating; discard it and report no speech.
        if state == "done" and text.strip() and speech_ms < MIN_SPEECH_MS_FOR_STT:
            error_code, state, text = "no_speech_detected", "failed", ""

    def mutate(settings: Settings) -> None:
        _, s = _find_sample(settings, voice_id, sample_id)
        if state == "done" and text.strip():
            s.transcript = text.strip()
            s.transcript_source = "stt"
            s.transcript_confirmed = False
            s.transcription = {
                "state": "done", "stt_profile": stt_profile_id, "model": model,
                "language": language, "duration_ms": duration_ms, "updated_at": time.time(),
            }
        else:
            code = error_code or "no_speech_detected"
            s.transcription = {
                "state": "failed", "stt_profile": stt_profile_id, "model": model,
                "language": language, "duration_ms": duration_ms, "error_code": code,
                "updated_at": time.time(),
            }

    settings = await _mutate_voice(request, mutate)
    _, sample = _find_sample(settings, voice_id, sample_id)
    return sample.model_dump()


def _strip_asr_prefix(text: str) -> str:
    """Qwen3-ASR prefixes like `language German<asr_text>` must be stripped."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.strip()


# ---------------------------------------------------------------- preview

@router.post("/voices/{voice_id}/preview")
async def preview_voice(voice_id: str, request: Request) -> Response:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("preview payload must be a JSON object")
    text = str(body.get("text") or "")
    if not text.strip():
        raise ValueError("text is required")
    if len(text) > 1000:
        raise ValueError("text too long: max 1000 characters")
    tts_profile_id = str(body.get("tts_profile") or "")
    if not tts_profile_id:
        raise ValueError("tts_profile is required")
    store, sample_store, b64_cache = _state(request)
    settings = await store.get()
    voice = settings.voices.get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    profile = settings.profiles.get("tts", {}).get(tts_profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile '{tts_profile_id}' not found")
    sample_id = str(body.get("sample_id") or voice.reference_sample_id)
    sample = next((s for s in voice.samples if s.id == sample_id), None)
    if sample is None:
        raise ApiError(400, "unknown_sample", "sample_id is not part of this voice")
    if profile.capabilities.voice_cloning != "reference_audio":
        raise ApiError(409, "voice_incompatible", "TTS profile does not support reference_audio cloning")
    if not sample.transcript.strip() or not sample.transcript_confirmed:
        raise ApiError(409, "transcript_not_confirmed", "reference transcript must be non-empty and confirmed")
    try:
        wav = sample_store.read(voice_id, sample.sha256)
    except OSError:
        raise HTTPException(status_code=404, detail="sample audio file is missing")
    # F10: same limit as test sessions and activation. Previously the preview
    # skipped it and the engine silently used only the first part of the
    # reference (chatterbox keeps 10 s), so what the user heard was not
    # what they had recorded. Refuse with the numbers a UI needs to offer a
    # fitting excerpt instead.
    from .profiles import materialise_tts_extra, sample_duration_ms

    limit = int(profile.capabilities.max_reference_ms or 0)
    if limit:
        duration = sample_duration_ms(voice_id, sample, sample_store)
        if duration > limit:
            exc = ApiError(
                409,
                "reference_too_long",
                f"reference sample is {duration / 1000:.1f} s but engine '{profile.id}' uses at most "
                f"{limit / 1000:.1f} s of a reference; choose a shorter excerpt as the reference",
            )
            exc.issues = [  # type: ignore[attr-defined]
                {
                    "code": "reference_too_long",
                    "voice_id": voice_id,
                    "sample_id": sample.id,
                    "sample_ms": duration,
                    "max_ms": limit,
                    "tts_profile": profile.id,
                }
            ]
            raise exc

    backend_settings = profile_backend_settings(profile, voice.language)
    backend_settings.extra = materialise_tts_extra(backend_settings, voice, sample, sample_bytes=wav, b64_cache=b64_cache)
    backend = OpenAIHTTPTTS(backend_settings, request.app.state.http_client)
    pcm_parts: list[bytes] = []
    first_pcm_ms: int | None = None
    total = 0
    started = time.perf_counter()
    try:
        async for pcm in backend.stream_speech(text, voice=profile.voice or None, language=voice.language or None):
            if first_pcm_ms is None and pcm:
                first_pcm_ms = int((time.perf_counter() - started) * 1000)
            total += len(pcm)
            if total > 40 * 1024 * 1024:
                raise ValueError("preview output exceeded the size limit")
            pcm_parts.append(pcm)
    except Exception as exc:
        from .errors import BackendError

        if isinstance(exc, BackendError):
            raise ApiError(502, "tts_backend_error", exc.message) from exc
        detail = sanitize_error_text(str(exc))[:300] or exc.__class__.__name__
        raise ApiError(502, "tts_preview_failed", f"preview failed: {detail}") from exc
    total_ms = int((time.perf_counter() - started) * 1000)
    from .audio import pcm16_to_wav

    pcm = b"".join(pcm_parts)
    audio_ms = int(len(pcm) / 2 / 24000 * 1000)
    wav_out = pcm16_to_wav(pcm, 24000)
    # Provenance: prove WHICH reference the backend actually received. The
    # sha8 is of the user's sample bytes (the materialised ref_audio), so it
    # is the sample's own sha even when the TTS profile carries a legacy
    # embedded reference that materialisation stripped and replaced.
    headers = {
        "X-Rtv-First-Pcm-Ms": "" if first_pcm_ms is None else str(first_pcm_ms),
        "X-Rtv-Total-Ms": str(total_ms),
        "X-Rtv-Audio-Ms": str(audio_ms),
        "X-Rtv-Profile-Id": profile.id,
        "X-Rtv-Model": profile.model,
        "X-Rtv-Voice-Id": _safe_header(voice_id),
        "X-Rtv-Sample-Id": _safe_header(sample.id),
        "X-Rtv-Reference-Sha8": _safe_header(sample.sha256[:8]),
        "Cache-Control": "no-store",
    }
    return Response(content=wav_out, media_type="audio/wav", headers=headers)


def _safe_header(value: str) -> str:
    """Bound provenance header values to printable ASCII (no CRLF/unicode)."""
    return "".join(ch if 0x20 <= ord(ch) <= 0x7E else "?" for ch in str(value))[:128]


# ------------------------------------------------------- voice library settings

@router.get("/voice-library/settings")
async def get_voice_library_settings(request: Request) -> dict[str, Any]:
    _guard(request)
    store, _, _ = _state(request)
    settings = await store.get()
    return settings.voice_library.model_dump()


@router.put("/voice-library/settings")
async def put_voice_library_settings(request: Request) -> dict[str, Any]:
    _guard(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("settings payload must be a JSON object")
    stt_profile = str(body.get("transcription_stt_profile") or "")
    store, _, _ = _state(request)

    def mutate(settings: Settings) -> None:
        if stt_profile and stt_profile not in settings.profiles.get("stt", {}):
            raise ApiError(400, "unknown_stt_profile", f"unknown STT profile: {stt_profile}")
        settings.voice_library.transcription_stt_profile = stt_profile

    settings = await store.mutate(mutate)
    return settings.voice_library.model_dump()


# ------------------------------------------------------------------ helpers

def _validation_message(exc: ValidationError) -> str:
    from .config import _validation_message as msg

    return msg(exc)


def _validation_error(exc: ValidationError) -> ValueError:
    return ValueError(_validation_message(exc))


def register_profile_routes(app) -> None:
    app.include_router(router)
