"""Sets: validation and resolution of a call.

A **set** (conversation profile) names three model entries, one voice, a
language and optional call defaults. What is saved is what calls use: a call
resolves the set against the models and the voice's reference recording as
they are when it starts, and keeps that copy for its lifetime.

Resolution happens BEFORE any backend call: `profile:<id>`,
`profile-test:<ticket>`, `voice-route:auto` / `voice-route:<id>`, and the set
for other apps. Nothing here logs or returns reference material or
credentials.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import secrets
import time
from collections import OrderedDict
from typing import Any

from .config import (
    BackendProfile,
    BackendSettings,
    ConversationProfile,
    ProfileSpec,
    Settings,
    Voice,
    reference_fingerprint,
)
from .catalog import ResolvedSelection, SelectionError

PROFILE_PREFIX = "profile:"
PROFILE_TEST_PREFIX = "profile-test:"
VOICE_ROUTE_PREFIX = "voice-route:"
PROFILE_TEST_TTL_S = 900.0
PROFILE_TEST_MAX = 32
B64_LRU_MAX = 16

REFERENCE_EXTRA_KEYS = ("ref_audio", "ref_text")
TTS_REFERENCE_EXTRA_KEYS = ("ref_audio", "ref_text", "ref_wav", "ref_transcript", "reference")


class ProfileResolutionError(SelectionError):
    def __init__(self, message: str, code: str):
        super().__init__(message, code=code)


class UnknownConversationProfileError(ProfileResolutionError):
    def __init__(self, pid: str):
        super().__init__(f"no set is called '{pid}'", code="unknown_conversation_profile")


class ProfileNotAvailableError(ProfileResolutionError):
    def __init__(self, pid: str):
        super().__init__(
            f"set '{pid}' is switched off",
            code="profile_not_available",
        )


class TestTicketError(ProfileResolutionError):
    def __init__(self, ticket: str):
        super().__init__(f"unknown or expired profile test ticket: {ticket}", code="unknown_test_ticket")


class ProfileActivationError(ValueError):
    """Set validation failure carrying per-issue structure."""

    def __init__(self, code: str, message: str, issues: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.issues = issues or []


# ----------------------------------------------------------------- tickets

class ProfileTestStore:
    """In-memory bounded, expiring profile-test tickets: a call with what the
    set editor shows, saved or not."""

    def __init__(self, ttl_s: float = PROFILE_TEST_TTL_S, max_alive: int = PROFILE_TEST_MAX):
        self.ttl_s = ttl_s
        self.max_alive = max_alive
        self._tickets: dict[str, tuple[str, float]] = {}
        # the editor's unsaved spec per ticket (None = the saved set)
        self._specs: dict[str, "ProfileSpec"] = {}

    def create(
        self,
        profile_id: str,
        now: float | None = None,
        *,
        spec: "ProfileSpec | None" = None,
    ) -> str:
        now = time.time() if now is None else now
        self._evict(now)
        while len(self._tickets) >= self.max_alive:
            oldest = min(self._tickets.items(), key=lambda kv: kv[1][1])[0]
            del self._tickets[oldest]
            self._specs.pop(oldest, None)
        ticket = secrets.token_hex(12)
        while ticket in self._tickets:
            ticket = secrets.token_hex(12)
        self._tickets[ticket] = (profile_id, now + self.ttl_s)
        if spec is not None:
            # snapshot: later edits in the editor must not change a running test
            self._specs[ticket] = spec.model_copy(deep=True)
        return ticket

    def spec_for(self, ticket: str) -> "ProfileSpec | None":
        """The unsaved spec snapshot for a ticket, or None (= use the saved set)."""
        if ticket not in self._tickets:
            self._specs.pop(ticket, None)
            return None
        return self._specs.get(ticket)

    def get(self, ticket: str, now: float | None = None) -> tuple[str, float] | None:
        entry = self._tickets.get(ticket)
        if entry is None:
            return None
        now = time.time() if now is None else now
        if now >= entry[1]:
            del self._tickets[ticket]
            self._specs.pop(ticket, None)
            return None
        return entry

    def _evict(self, now: float) -> None:
        for ticket in [t for t, e in self._tickets.items() if now >= e[1]]:
            del self._tickets[ticket]
            self._specs.pop(ticket, None)


# ------------------------------------------------------------ base64 cache

class Base64LRU:
    """Bounded in-memory cache of base64-encoded sample bytes per sha."""

    def __init__(self, max_entries: int = B64_LRU_MAX):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, str] = OrderedDict()

    def get(self, sha256: str) -> str | None:
        value = self._cache.get(sha256)
        if value is not None:
            self._cache.move_to_end(sha256)
        return value

    def put(self, sha256: str, encoded: str) -> None:
        self._cache[sha256] = encoded
        self._cache.move_to_end(sha256)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------- helpers

def _strip_reference_extra(extra: dict[str, Any] | None, also: tuple[str, ...] = ()) -> dict[str, Any]:
    """`extra` without stored reference material (the usual field names, plus
    the ones a model is configured to use)."""
    names = set(TTS_REFERENCE_EXTRA_KEYS) | {a.lower() for a in also}
    data = dict(extra or {})
    for key in list(data):
        if str(key).lower() in names:
            del data[key]
    return data


def profile_backend_settings(profile: BackendProfile, language: str = "") -> BackendSettings:
    """A model entry as one call uses it."""
    return BackendSettings(
        adapter=profile.adapter,
        base_url=profile.base_url,
        api_key=profile.api_key,
        model=profile.model,
        language=language,
        language_format=profile.language_format,
        voice=profile.voice,
        request_timeout_s=profile.request_timeout_s,
        extra=copy.deepcopy(profile.extra),
        reference_audio_field=profile.reference_audio_field,
        reference_text_field=profile.reference_text_field,
        reference_audio_encoding=profile.reference_audio_encoding,
        audio_format=profile.audio_format,
        pcm_sample_rate=profile.pcm_sample_rate,
        start_buffer_ms=profile.start_buffer_ms,
    )


def sets_using(settings: Settings, *, role: str = "", profile_id: str = "", voice_id: str = "") -> list[str]:
    """Ids of the sets that use a model entry or a library voice."""
    out = []
    for pid, profile in sorted(settings.conversation_profiles.items()):
        spec = profile.spec
        if role and getattr(spec, f"{role}_profile", "") == profile_id:
            out.append(pid)
        elif voice_id and spec.voice.kind == "library" and spec.voice.voice_id == voice_id:
            out.append(pid)
    return out


def broken_sets(before: Settings, after: Settings, *, sample_store: Any | None = None) -> list[dict[str, Any]]:
    """Switched-on sets that a change (before -> after) would make unusable:
    each with the blocking issues it did not have before. Used to refuse a
    model, voice or recording change that would silently break a set."""
    out = []
    for pid, profile in sorted(after.conversation_profiles.items()):
        if not profile.enabled:
            continue
        old = before.conversation_profiles.get(pid)
        new_issues = blocking_issues(validate_spec(after, profile.spec, sample_store=sample_store))
        if not new_issues:
            continue
        old_codes = set()
        if old is not None:
            old_codes = {
                (i.get("path"), i.get("code"))
                for i in blocking_issues(validate_spec(before, old.spec, sample_store=sample_store))
            }
        fresh = [i for i in new_issues if (i.get("path"), i.get("code")) not in old_codes]
        if fresh:
            out.append({"profile": pid, "display_name": profile.display_name, "issues": fresh})
    return out


def breaks_sets_error(broken: list[dict[str, Any]]) -> ValueError:
    """409 breaks_sets: names the sets and what would go wrong for each."""
    names = ", ".join(b["display_name"] or b["profile"] for b in broken)
    first = broken[0]["issues"][0].get("message", broken[0]["issues"][0].get("code"))
    exc = ValueError(f"this would stop {names} from working: {first}")
    exc.status_code = 409  # type: ignore[attr-defined]
    exc.code = "breaks_sets"  # type: ignore[attr-defined]
    exc.issues = broken  # type: ignore[attr-defined]
    return exc


# ------------------------------------------------------------- validation

def _sample_for_reference(settings: Settings, spec: ProfileSpec) -> tuple[Voice, Any] | None:
    voice = settings.voices.get(spec.voice.voice_id)
    if voice is None:
        return None
    sample = next((s for s in voice.samples if s.id == voice.reference_sample_id), None)
    if sample is None:
        return None
    return voice, sample


def sample_duration_ms(voice_id: str, sample: Any, sample_store: Any | None = None) -> int:
    """Duration of a stored sample in ms.

    Uses the stored metadata; when that is 0 (unknown — e.g. samples imported
    by the profile migration were never measured) the WAV header of the real
    file is read instead, so a length check is never silently skipped for
    lack of metadata. Returns 0 only if the duration genuinely cannot be
    determined. Never modifies the sample or its metadata.
    """
    duration = int(getattr(sample, "duration_ms", 0) or 0)
    if duration > 0 or sample_store is None:
        return duration
    try:
        import io
        import wave

        data = sample_store.read(voice_id, sample.sha256)
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            if rate > 0:
                return int(round(w.getnframes() * 1000 / rate))
    except Exception:  # noqa: BLE001 — unreadable file: duration unknown
        return 0
    return 0


def reference_length_issue(
    settings: Settings, spec: ProfileSpec, *, sample_store: Any | None = None
) -> dict[str, Any] | None:
    """The one place that decides whether a library voice's reference sample
    is too long for the selected TTS engine.

    Why it blocks (F10): the limit is not an arbitrary number. For chatterbox
    it mirrors the model itself — mlx_audio's chatterbox keeps only the first
    DEC_COND_LEN = 10 s of the reference for the decoder (and 6 s for the
    speaker encoder) and silently drops the rest. Letting a longer sample
    through therefore means the engine quietly ignores part of what the user
    recorded. Engines without a known limit advertise max_reference_ms = 0
    and are never checked.

    Returns a structured, blocking issue (with the numbers a UI needs to offer
    a fitting excerpt) or None.
    """
    if spec.voice.kind != "library" or not spec.tts_profile:
        return None
    tts = settings.profiles["tts"].get(spec.tts_profile)
    if tts is None or not tts.capabilities.max_reference_ms:
        return None
    entry = _sample_for_reference(settings, spec)
    if entry is None:
        return None
    voice, sample = entry
    duration = sample_duration_ms(voice.id, sample, sample_store)
    limit = int(tts.capabilities.max_reference_ms)
    if duration <= limit:
        return None
    return {
        "path": "voice.voice_id",
        "code": "reference_too_long",
        "message": (
            f"reference sample {sample.id} is {duration / 1000:.1f} s but engine "
            f"'{spec.tts_profile}' uses at most {limit / 1000:.1f} s of a reference; "
            "choose a shorter excerpt as the reference"
        ),
        "voice_id": voice.id,
        "sample_id": sample.id,
        "sample_ms": duration,
        "max_ms": limit,
        "tts_profile": spec.tts_profile,
    }


def validate_spec(
    settings: Settings,
    spec: ProfileSpec,
    *,
    sample_store: Any | None = None,
    check_files: bool = True,
) -> list[dict[str, Any]]:
    """Return validation issues (each {path, code, message}). Empty = valid.
    Never probes; touches files (presence/sha) only when check_files is True
    and a sample store is given."""
    issues: list[dict[str, Any]] = []
    for role, pid in (
        ("stt", spec.stt_profile),
        ("llm", spec.llm_profile),
        ("tts", spec.tts_profile),
    ):
        if not pid:
            issues.append({"path": f"{role}_profile", "code": "missing", "message": f"{role} profile is required"})
        elif pid not in settings.profiles.get(role, {}):
            issues.append(
                {"path": f"{role}_profile", "code": "unknown_profile", "message": f"unknown {role} profile: {pid}"}
            )
    if not spec.language:
        issues.append({"path": "language", "code": "missing", "message": "language is required"})
    voice = spec.voice
    if voice.kind == "library":
        if not voice.voice_id:
            issues.append({"path": "voice.voice_id", "code": "missing", "message": "voice_id is required"})
        else:
            entry = _sample_for_reference(settings, spec)
            if entry is None:
                issues.append(
                    {
                        "path": "voice.reference_sample_id",
                        "code": "missing_reference",
                        "message": "voice reference sample is not set or missing",
                    }
                )
            else:
                stored_voice, sample = entry
                if not sample.transcript.strip():
                    issues.append(
                        {"path": "voice.transcript", "code": "transcript_missing", "message": "reference transcript is empty"}
                    )
                elif not sample.transcript_confirmed:
                    issues.append(
                        {
                            "path": "voice.transcript",
                            "code": "transcript_unconfirmed",
                            "message": "reference transcript is not confirmed",
                        }
                    )
                if check_files and sample_store is not None:
                    if not sample_store.exists(stored_voice.id, sample.sha256):
                        issues.append(
                            {"path": "voice.reference_sample_id", "code": "file_missing", "message": "reference WAV file is missing"}
                        )
                    else:
                        data = sample_store.read(stored_voice.id, sample.sha256)
                        if hashlib.sha256(data).hexdigest() != sample.sha256:
                            issues.append(
                                {"path": "voice.reference_sample_id", "code": "sha_mismatch", "message": "reference WAV does not match its sha256"}
                            )
        role_tts = settings.profiles["tts"].get(spec.tts_profile) if spec.tts_profile else None
        if role_tts is not None and role_tts.capabilities.voice_cloning != "reference_audio":
            issues.append(
                {
                    "path": "voice",
                    "code": "voice_incompatible",
                    "message": "selected TTS profile does not support reference_audio cloning",
                }
            )
    elif voice.kind == "native":
        if not voice.native_voice:
            issues.append({"path": "voice.native_voice", "code": "missing", "message": "native_voice is required"})
        role_tts = settings.profiles["tts"].get(spec.tts_profile) if spec.tts_profile else None
        if role_tts is not None and role_tts.capabilities.voice_cloning == "reference_audio":
            # native voice against a cloning engine: allowed (engine can take a
            # native label) but flagged for the operator
            issues.append(
                {
                    "path": "voice",
                    "code": "native_on_cloning_engine",
                    "message": "native voice selected on a reference-audio TTS profile",
                    "severity": "warning",
                }
            )
    # F10: blocking for test calls and for a switched-on set; a switched-off
    # set may be saved with it while the user fixes the recording
    length_issue = reference_length_issue(settings, spec, sample_store=sample_store)
    if length_issue is not None:
        issues.append(length_issue)
    return issues


def compatibility_warnings(settings: Settings, spec: ProfileSpec) -> list[str]:
    """Non-blocking compatibility warnings shown with a set."""
    warnings: list[str] = []
    tts = settings.profiles["tts"].get(spec.tts_profile) if spec.tts_profile else None
    if tts is not None:
        if spec.voice.kind == "library" and tts.capabilities.voice_cloning != "reference_audio":
            warnings.append("voice_incompatible: TTS profile does not advertise reference_audio cloning")
        # same decision as the blocking check (single source of truth); the
        # the stored duration is used here: views never read files
        length_issue = reference_length_issue(settings, spec)
        if length_issue is not None:
            warnings.append(
                f"reference_length: sample {length_issue['sample_ms']} ms exceeds max {length_issue['max_ms']} ms"
            )
    return warnings


def blocking_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in issues if i.get("severity") != "warning"]


# ---------------------------------------------------------- materialisation

def materialise_tts_extra(
    tts: BackendSettings,
    voice: Voice | None,
    sample: Any | None,
    *,
    sample_bytes: bytes | None,
    b64_cache: Base64LRU | None = None,
) -> dict[str, Any]:
    """The speaking model's `extra` for a call: the entry's own extra WITHOUT
    any stored reference material, plus the voice's reference recording and
    its transcript in the fields and the encoding the model is configured for,
    when the set speaks with a library voice."""
    also = (tts.reference_audio_field, tts.reference_text_field)
    extra = _strip_reference_extra(tts.extra, also)
    if voice is None or sample is None or sample_bytes is None:
        return extra
    encoded = b64_cache.get(sample.sha256) if b64_cache is not None else None
    if encoded is None:
        encoded = base64.b64encode(sample_bytes).decode("ascii")
        if b64_cache is not None:
            b64_cache.put(sample.sha256, encoded)
    if tts.reference_audio_encoding == "data_uri":
        encoded = "data:audio/wav;base64," + encoded
    extra[tts.reference_audio_field] = encoded
    extra[tts.reference_text_field] = sample.transcript
    return extra


# -------------------------------------------------------------- resolution

def _profile_ids(spec: ProfileSpec) -> dict[str, str]:
    return {"stt": spec.stt_profile, "llm": spec.llm_profile, "tts": spec.tts_profile}


def resolve_spec(
    settings: Settings,
    profile: ConversationProfile,
    spec: ProfileSpec,
    *,
    sample_store: Any | None = None,
    b64_cache: Base64LRU | None = None,
) -> tuple[ResolvedSelection, dict[str, Any]]:
    """Resolve a set's spec to a frozen per-call snapshot: the models and the
    voice's reference recording as they are now. Returns (selection, route)."""
    resolved = settings.model_copy(deep=True)
    for role in ("stt", "llm", "tts"):
        pid = getattr(spec, f"{role}_profile")
        row = settings.profiles.get(role, {}).get(pid)
        if row is None:
            raise ProfileResolutionError(
                f"set '{profile.id}' uses a {role} model that does not exist ('{pid}')",
                code="unknown_profile",
            )
        setattr(resolved, role, profile_backend_settings(row, spec.language))
    tts = resolved.tts
    voice_sel = spec.voice
    route_voice: dict[str, Any] = {"kind": voice_sel.kind}
    if voice_sel.kind == "library":
        entry = _sample_for_reference(settings, spec)
        if entry is None:
            raise ProfileResolutionError(
                f"voice '{voice_sel.voice_id}' of set '{profile.id}' has no reference recording",
                code="missing_reference",
            )
        voice, sample = entry
        sample_bytes = None
        if sample_store is not None:
            try:
                sample_bytes = sample_store.read(voice.id, sample.sha256)
            except OSError:
                raise ProfileResolutionError(
                    f"voice reference file for '{voice.id}' is missing",
                    code="reference_file_missing",
                )
        tts.extra = materialise_tts_extra(tts, voice, sample, sample_bytes=sample_bytes, b64_cache=b64_cache)
        resolved.session.voice = tts.voice or voice.display_name
        display_voice = voice.display_name
        route_voice.update({"voice_id": voice.id, "sample_id": sample.id, "ref_sha256": sample.sha256})
    elif voice_sel.kind == "native":
        tts.extra = _strip_reference_extra(tts.extra, (tts.reference_audio_field, tts.reference_text_field))
        resolved.session.voice = voice_sel.native_voice
        display_voice = voice_sel.native_voice
        route_voice["native_voice"] = voice_sel.native_voice
    else:
        tts.extra = _strip_reference_extra(tts.extra, (tts.reference_audio_field, tts.reference_text_field))
        resolved.session.voice = tts.voice or resolved.session.voice
        display_voice = tts.voice or ""
    resolved.session.language = spec.language or resolved.session.language
    if spec.instructions_default:
        resolved.session.instructions = spec.instructions_default
    if spec.turn_detection_default:
        merged = resolved.session.turn_detection.model_dump()
        merged.update(spec.turn_detection_default)
        from .config import TurnDetectionConfig

        resolved.session.turn_detection = TurnDetectionConfig.model_validate(merged)
    if spec.output_pacing:
        resolved.session.output_pacing = spec.output_pacing
    selection = ResolvedSelection(
        settings=resolved,
        set_id=profile.id,
        profile_ids=_profile_ids(spec),
        voice=resolved.session.voice,
        language=spec.language,
        display_voice=display_voice,
        ref_sha256=reference_fingerprint(tts.extra, tts.reference_audio_field),
    )
    route = {
        "mode": "profile",
        "requested": "",
        "resolved": True,
        "profile": profile.id,
        "profile_id": profile.id,
        "actual_voice": resolved.session.voice,
        "display_voice": display_voice,
        "stt_model": resolved.stt.model,
        "llm_model": resolved.llm.model,
        "tts_model": resolved.tts.model,
        "voice": route_voice,
        "language": spec.language,
    }
    return selection, route


def resolve_profile(
    settings: Settings,
    pid: str,
    *,
    sample_store: Any | None = None,
    b64_cache: Base64LRU | None = None,
) -> tuple[ResolvedSelection, dict[str, Any]]:
    """Resolve `profile:<id>`: a switched-on set."""
    profile = settings.conversation_profiles.get(pid)
    if profile is None:
        raise UnknownConversationProfileError(pid)
    if not profile.enabled:
        raise ProfileNotAvailableError(pid)
    return resolve_spec(settings, profile, profile.spec, sample_store=sample_store, b64_cache=b64_cache)


def resolve_test(
    settings: Settings,
    profile_id: str,
    *,
    sample_store: Any | None = None,
    b64_cache: Base64LRU | None = None,
    spec_override: ProfileSpec | None = None,
) -> tuple[ResolvedSelection, dict[str, Any]]:
    """Resolve a `profile-test:<ticket>` connection: the set as the editor
    shows it (`spec_override`, never persisted) or as saved, switched on or
    not."""
    profile = settings.conversation_profiles.get(profile_id)
    if profile is None:
        raise UnknownConversationProfileError(profile_id)
    spec = spec_override if spec_override is not None else profile.spec
    issues = blocking_issues(validate_spec(settings, spec, sample_store=sample_store, check_files=True))
    if issues:
        raise ProfileResolutionError(
            "the set is incomplete and cannot be tested: "
            + "; ".join(str(i.get("message", i.get("code"))) for i in issues),
            code="set_incomplete",
        )
    selection, route = resolve_spec(settings, profile, spec, sample_store=sample_store, b64_cache=b64_cache)
    route["requested"] = "profile-test"
    return selection, route


def resolve_voice_route_full(
    settings: Settings,
    voice: str,
    *,
    sample_store: Any | None = None,
    b64_cache: Base64LRU | None = None,
) -> tuple[ResolvedSelection, dict[str, Any]]:
    """`voice-route:<name>` / the voice of a `voice-route:auto` call: the set
    with that name."""
    if voice not in settings.conversation_profiles:
        raise ProfileResolutionError(f"no set is called '{voice}'", code="unknown_voice_route")
    selection, route = resolve_profile(settings, voice, sample_store=sample_store, b64_cache=b64_cache)
    route["requested"] = voice
    return selection, route


def resolve_default_set(
    settings: Settings,
    *,
    sample_store: Any | None = None,
    b64_cache: Base64LRU | None = None,
) -> tuple[ResolvedSelection, dict[str, Any]]:
    """A client that names no set (model "" or its own model name): the set
    for other apps, if one is chosen."""
    pid = settings.default_set
    if not pid or pid not in settings.conversation_profiles:
        raise ProfileResolutionError(
            "no set selected: connect with model profile:<set> or voice-route:auto, "
            "or choose a set for other apps in the server settings",
            code="no_set_selected",
        )
    return resolve_profile(settings, pid, sample_store=sample_store, b64_cache=b64_cache)


def pending_selection(settings: Settings) -> ResolvedSelection:
    """The base of a `voice-route:auto` call before its set is known: the
    server's call defaults, no models."""
    return ResolvedSelection(
        settings=settings.model_copy(deep=True),
        set_id="",
        profile_ids={"stt": "", "llm": "", "tts": ""},
    )


__all__ = [
    "Base64LRU",
    "PROFILE_PREFIX",
    "PROFILE_TEST_PREFIX",
    "ProfileActivationError",
    "ProfileNotAvailableError",
    "ProfileResolutionError",
    "ProfileTestStore",
    "TestTicketError",
    "UnknownConversationProfileError",
    "breaks_sets_error",
    "broken_sets",
    "compatibility_warnings",
    "materialise_tts_extra",
    "pending_selection",
    "profile_backend_settings",
    "resolve_default_set",
    "resolve_profile",
    "resolve_spec",
    "resolve_test",
    "resolve_voice_route_full",
    "sets_using",
    "validate_spec",
]
