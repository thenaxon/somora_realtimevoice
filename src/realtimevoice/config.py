"""Settings model, persistence and secret masking.

Backends are generic: no URLs, hosts, keys or model IDs are hard-coded. All
deployment-specific values arrive via the Settings UI/API or environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

SECRET_MASK = "********"
DATA_DIR_ENV = "REALTIMEVOICE_DATA_DIR"
AUTH_TOKEN_ENV = "REALTIMEVOICE_AUTH_TOKEN"
SETTINGS_VERSION = 3
SETTINGS_V1_BACKUP_NAME = "settings.v1.backup.json"
SETTINGS_V2_BACKUP_NAME = "settings.v2.backup.json"

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# conversation profiles / voices: immutable, client-facing, hyphens only
PUBLIC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
# prefixes a voice or set id must never use (the profile test ticket namespace)
RESERVED_ID_PREFIXES = ("tmp_", "profile-test-")
ROLES = ("stt", "llm", "tts")
SET_HISTORY_MAX = 10
# the request fields a speaking model always gets; a recording field may not
# take one of these names
TTS_BASE_FIELDS = ("model", "input", "voice", "response_format", "stream", "language")
REFERENCE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# How a model wants the language of a call: "code" ("de"), "name" ("German",
# e.g. Qwen3-TTS), or "none" (the model is not told a language).
LANGUAGE_FORMATS = ("code", "name", "none")
LANGUAGE_NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sv": "Swedish", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
}


def language_for_model(language: str, language_format: str) -> str:
    """The call language as one model wants it: "" means send none. A code
    with a region ("de-AT") is sent as its base language ("de" / "German")."""
    code = (language or "").strip()
    if not code or code == "auto" or language_format == "none":
        return ""
    base = code.split("-")[0].split("_")[0].lower()
    if language_format == "name":
        return LANGUAGE_NAMES.get(base, code)
    return base


class SettingsLoadError(RuntimeError):
    """Stored settings are malformed or use an unsupported schema version.

    The store refuses to fall back to empty defaults: raising is the safe
    behavior; operators restore the private backup instead.
    """


def default_data_dir() -> str:
    return os.environ.get(DATA_DIR_ENV) or os.path.join(os.getcwd(), "data")


class TurnDetectionConfig(BaseModel):
    type: str = "server_vad"
    threshold_rms: float = 350.0
    threshold: float = 0.5
    prefix_padding_ms: int = Field(default=200, ge=0, le=2000)
    silence_duration_ms: int = Field(default=420, ge=50, le=10_000)
    interrupt_response: bool = True
    create_response: bool = True
    # H2 soft barge-in: on speech_started during an active response, wait this
    # long for the burst to confirm before cancelling. 0 restores the old
    # immediate behaviour. Held audio is flushed when the burst is rejected.
    barge_in_confirm_ms: int = Field(default=250, ge=0, le=5000)
    # how speech is told from other sound: "model" asks a small speech
    # classifier as well as the loudness (a cough is loud but not speech);
    # "loudness" uses the level alone
    speech_check: Literal["model", "loudness"] = "model"

    @field_validator("type")
    @classmethod
    def _check_type(cls, value: str) -> str:
        if value not in ("server_vad", "none"):
            raise ValueError("turn_detection.type must be 'server_vad' or 'none'")
        return value

    @field_validator("threshold_rms")
    @classmethod
    def _check_rms(cls, value: float) -> float:
        if not 1.0 <= value <= 10_000.0:
            raise ValueError("threshold_rms must be between 1 and 10000")
        return value

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("turn_detection.threshold must be between 0 and 1")
        return value


class BackendSettings(BaseModel):
    """One model as a call uses it (built from a model entry and the set)."""

    adapter: str = "openai_http"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    language: str = ""          # the call's language code; sent per language_format
    language_format: Literal["code", "name", "none"] = "code"
    voice: str = ""
    request_timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    extra: dict[str, Any] = Field(default_factory=dict)
    # how a cloning model takes the voice recording: the request fields for
    # the recording and its transcript, and whether the recording is plain
    # base64 or a data URL ("data:audio/wav;base64,…")
    reference_audio_field: str = "ref_audio"
    reference_text_field: str = "ref_text"
    reference_audio_encoding: Literal["base64", "data_uri"] = "base64"
    # what a speaking model answers with: WAV (any rate), raw 16-bit PCM at
    # pcm_sample_rate (mono), or MP3 (decoded with ffmpeg)
    audio_format: Literal["wav", "pcm", "mp3"] = "wav"
    pcm_sample_rate: int = Field(default=24000, ge=8000, le=192000)
    # audio held back before a response starts speaking, for a model that
    # delivers its first pieces barely in real time (0 = speak at once)
    start_buffer_ms: int = Field(default=0, ge=0, le=5000)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        value = value.strip()
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class SessionDefaults(BaseModel):
    instructions: str = (
        "You are a helpful voice assistant. Keep answers short and suitable for "
        "speech. Answer in the language the user speaks."
    )
    voice: str = "default"
    language: str = "de"
    turn_detection: TurnDetectionConfig = Field(default_factory=TurnDetectionConfig)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    max_history_items: int = Field(default=200, ge=10, le=1000)
    # H1 opt-in output pacing: "off" streams TTS as fast as it is synthesized;
    # "realtime" queues synthesized audio server-side and emits it at ~1x with
    # a lead, so response.done lands when the client's playback is nearly over.
    output_pacing: Literal["off", "realtime"] = "off"
    output_pacing_lead_ms: int = Field(default=600, ge=0, le=10_000)


class ProfileCapabilities(BaseModel):
    """Honestly-evidenced capabilities; empty/unknown means "nothing claimed"."""

    voices: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    streaming: str = "unknown"
    evidence: str = "unverified"
    voice_cloning: Literal["none", "reference_audio", "unknown"] = "none"
    reference_text_required: bool = False
    max_reference_ms: int = Field(default=0, ge=0)
    # Round 2 / CP3: whether this engine actually changes its voice when the
    # native label changes. "honoured" -> the labels in `voices` are a durable
    # identity and may be offered as named presets; "ignored" -> the label is
    # silently ignored (e.g. Fish); "unknown" -> treated like ignored. This
    # replaces the old hard-coded TRUSTWORTHY_NATIVE_ENGINE_MODELS list with
    # row data (a newly added TTS backend can now earn the status).
    native_voice_labels: Literal["honoured", "ignored", "unknown"] = "unknown"

    @field_validator("streaming")
    @classmethod
    def _check_streaming(cls, value: str) -> str:
        if value not in ("unknown", "supported", "unsupported"):
            raise ValueError("streaming must be 'unknown', 'supported' or 'unsupported'")
        return value

    @field_validator("evidence")
    @classmethod
    def _check_evidence(cls, value: str) -> str:
        if value not in ("unverified", "probe", "runtime"):
            raise ValueError("evidence must be 'unverified', 'probe' or 'runtime'")
        return value


class BackendProfile(BaseModel):
    id: str
    role: str = ""
    display_name: str = ""
    adapter: str = "openai_http"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    language_format: Literal["code", "name", "none"] = "code"
    voice: str = ""
    request_timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    extra: dict[str, Any] = Field(default_factory=dict)
    # how a cloning model takes the voice recording: the request fields for
    # the recording and its transcript, and whether the recording is plain
    # base64 or a data URL ("data:audio/wav;base64,…")
    reference_audio_field: str = "ref_audio"
    reference_text_field: str = "ref_text"
    reference_audio_encoding: Literal["base64", "data_uri"] = "base64"
    # what a speaking model answers with: WAV (any rate), raw 16-bit PCM at
    # pcm_sample_rate (mono), or MP3 (decoded with ffmpeg)
    audio_format: Literal["wav", "pcm", "mp3"] = "wav"
    pcm_sample_rate: int = Field(default=24000, ge=8000, le=192000)
    # audio held back before a response starts speaking, for a model that
    # delivers its first pieces barely in real time (0 = speak at once)
    start_buffer_ms: int = Field(default=0, ge=0, le=5000)
    capabilities: ProfileCapabilities = Field(default_factory=ProfileCapabilities)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not PROFILE_ID_RE.match(value):
            raise ValueError("model id must match [a-z0-9][a-z0-9._-]{0,63}")
        return value

    @field_validator("reference_audio_field", "reference_text_field")
    @classmethod
    def _check_reference_field(cls, value: str) -> str:
        value = value.strip()
        if not REFERENCE_FIELD_RE.match(value) or value in TTS_BASE_FIELDS:
            raise ValueError(
                "a recording field name must be a plain name (letters, digits, _ . -) "
                "other than " + ", ".join(TTS_BASE_FIELDS)
            )
        return value

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        value = value.strip()
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _distinct_reference_fields(self) -> "BackendProfile":
        if self.reference_audio_field == self.reference_text_field:
            raise ValueError("the recording and its text need two different field names")
        return self

    @model_validator(mode="after")
    def _migrate_native_voice_trust(self) -> "BackendProfile":
        """Round 2 / CP3 migration, applied when a row is materialised.

        Only a row that does NOT carry ``native_voice_labels`` at all (legacy
        data / a payload without the key) is migrated: a legacy-trusted model
        gets ``"honoured"``, every other model keeps the default ``"unknown"``.
        An EXPLICIT value — including ``"unknown"`` — is always preserved, so an
        operator can deliberately set a legacy-trusted model back to
        ``"unknown"`` and save it (C2). The check uses the nested model's
        ``model_fields_set``, which lists exactly the keys the payload provided.
        """
        if "native_voice_labels" in self.capabilities.model_fields_set:
            return self
        if self.model in LEGACY_TRUSTWORTHY_NATIVE_ENGINE_MODELS:
            self.capabilities.native_voice_labels = "honoured"
        return self


def _check_public_id(value: str) -> str:
    if not PUBLIC_ID_RE.match(value):
        raise ValueError("id must match [a-z0-9][a-z0-9-]{1,62}")
    for prefix in RESERVED_ID_PREFIXES:
        if value.startswith(prefix):
            raise ValueError(f"id must not use the reserved prefix '{prefix}'")
    return value


class VoiceSample(BaseModel):
    id: str                      # "smp_" + first 16 hex of sha256
    sha256: str                  # of the stored WAV bytes
    source: Literal["upload", "mic", "migration"]
    original_filename: str = ""  # display only, sanitized, max 128 chars
    duration_ms: int
    sample_rate: int
    channels: int = 1
    size_bytes: int
    transcript: str = ""
    transcript_confirmed: bool = False
    transcript_source: Literal["", "stt", "manual", "migration"] = ""
    transcription: dict[str, Any] = Field(default_factory=dict)
    created_at: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    # F10: provenance of an excerpt cut from another sample:
    # {sample_id, sha256, start_ms, end_ms}. Empty for recordings/uploads.
    # Deliberately an optional extra field instead of a new `source` literal:
    # models ignore unknown fields, so data written here stays loadable by
    # older builds (a code rollback does not break settings.json).
    derived_from: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not value.startswith("smp_"):
            raise ValueError("sample id must start with 'smp_'")
        return value

    @field_validator("original_filename")
    @classmethod
    def _check_original_filename(cls, value: str) -> str:
        value = value.strip()
        if len(value) > 128:
            value = value[:128]
        # display only: strip path separators and control characters
        return "".join(ch for ch in value if ch.isprintable() and ch not in "/\\")


class Voice(BaseModel):
    id: str                      # ^[a-z0-9][a-z0-9-]{1,62}$, immutable
    display_name: str
    language: str = ""           # hint, e.g. "de"
    description: str = ""
    samples: list[VoiceSample] = Field(default_factory=list)
    reference_sample_id: str = ""  # exactly one explicit reference
    created_at: float
    updated_at: float

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _check_public_id(value)


class VoiceSelection(BaseModel):
    kind: Literal["library", "native", "none"] = "none"
    voice_id: str = ""           # kind=library
    native_voice: str = ""       # kind=native


class ProfileSpec(BaseModel):
    stt_profile: str = ""        # BackendProfile ids
    llm_profile: str = ""
    tts_profile: str = ""
    voice: VoiceSelection = Field(default_factory=VoiceSelection)
    language: str = ""           # session/STT language code, e.g. "de"
    instructions_default: str = ""       # used only if the client sends none
    turn_detection_default: dict[str, Any] = Field(default_factory=dict)
    # H1 opt-in: "off" (default) or "realtime" output pacing for sessions
    # resolved from this profile (additive; old profiles load unchanged)
    output_pacing: Literal["off", "realtime"] = "off"


class SetVersion(BaseModel):
    """An earlier saved state of a set, kept for Restore."""

    saved_at: float
    spec: ProfileSpec


class ConversationProfile(BaseModel):
    """A set: what an agent speaks with. Saved = what calls use: a call
    resolves the spec against the models and the voice recording as they are
    when it starts."""

    id: str                      # immutable, the name clients use
    display_name: str
    description: str = ""
    spec: ProfileSpec = Field(default_factory=ProfileSpec)
    enabled: bool = False        # usable by clients (needs a valid spec)
    history: list[SetVersion] = Field(default_factory=list)
    created_at: float
    updated_at: float
    spec_saved_at: float = 0.0   # when the current spec was stored

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _check_public_id(value)


class VoiceLibrarySettings(BaseModel):
    transcription_stt_profile: str = ""   # the listening model that transcribes recordings


class Settings(BaseModel):
    version: int = SETTINGS_VERSION
    # the three models of ONE call, filled when a call is resolved from a set;
    # never stored and never part of the settings API
    stt: BackendSettings = Field(default_factory=BackendSettings, exclude=True)
    llm: BackendSettings = Field(default_factory=BackendSettings, exclude=True)
    tts: BackendSettings = Field(default_factory=BackendSettings, exclude=True)
    session: SessionDefaults = Field(default_factory=SessionDefaults)
    profiles: dict[str, dict[str, BackendProfile]] = Field(
        default_factory=lambda: {"stt": {}, "llm": {}, "tts": {}}
    )
    voices: dict[str, Voice] = Field(default_factory=dict)
    voice_library: VoiceLibrarySettings = Field(default_factory=VoiceLibrarySettings)
    conversation_profiles: dict[str, ConversationProfile] = Field(default_factory=dict)
    # the set a client gets that names none (another app sending model "" or
    # its own model name); "" = such a call is refused
    default_set: str = ""


def mask_secrets(settings: Settings) -> dict[str, Any]:
    """The settings as the API shows them: model keys and secret-looking
    extras masked."""
    data = settings.model_dump()
    for role_profiles in (data.get("profiles") or {}).values():
        if not isinstance(role_profiles, dict):
            continue
        for profile in role_profiles.values():
            if not isinstance(profile, dict):
                continue
            if profile.get("api_key"):
                profile["api_key"] = SECRET_MASK
            extra = profile.get("extra")
            if isinstance(extra, dict):
                profile["extra"] = _mask_extra_secrets(extra)
    return data


def _is_mask_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value == SECRET_MASK


def _strip_mask_placeholders(value: Any) -> Any:
    """A mask is a keep-marker, not data: on a genuinely NEW value (nothing
    stored to keep) masked leaves are dropped instead of stored literally."""
    if isinstance(value, dict):
        return {
            str(k): _strip_mask_placeholders(v)
            for k, v in value.items()
            if not _is_mask_placeholder(v)
        }
    if isinstance(value, list):
        return [_strip_mask_placeholders(v) for v in value if not _is_mask_placeholder(v)]
    return value


def _merge_masked(stored: Any, incoming: Any) -> Any:
    """Recursive merge that preserves stored values under mask placeholders
    at ANY depth (list items included): a mask keeps its stored counterpart
    or drops out entirely — it never becomes stored data."""
    if _is_mask_placeholder(incoming):
        return stored
    if isinstance(incoming, dict):
        base = dict(stored) if isinstance(stored, dict) else {}
        for key, value in incoming.items():
            k = str(key)
            if value is None:
                base.pop(k, None)  # explicit remove at this path
            elif is_sensitive_extra_key(k) and value == "":
                continue  # empty sensitive value keeps the stored one
            elif _is_mask_placeholder(value) and k not in base:
                continue  # mask with nothing stored: never store the literal mask
            else:
                merged = _merge_masked(base.get(k), value)
                base[k] = merged
        return base
    if isinstance(incoming, list):
        base = list(stored) if isinstance(stored, list) else []
        out: list[Any] = []
        for i, item in enumerate(incoming):
            if _is_mask_placeholder(item):
                if i < len(base):
                    out.append(base[i])  # keep the stored item
                continue  # mask with nothing stored: dropped
            if i < len(base):
                # generic positional recursion: dict/dict and list/list pairs
                # merge (masks keep stored counterparts at any depth),
                # scalars replace, null removes
                if item is None:
                    continue
                out.append(_merge_masked(base[i], item))
            else:
                stripped = _strip_mask_placeholders(item)
                if stripped != {} and stripped != []:
                    out.append(stripped)
        return out
    return incoming


def merge_extra_values(stored: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible recursive extra merge.

    Nested dicts merge recursively (existing `stop` arrays and
    `response_format` objects keep working); arrays merge positionally and
    scalars replace; JSON null removes the key at its path (explicit clear —
    applied directly to the merged result, so a later deep-merge can never
    resurrect it); a masked/empty sensitive value at ANY depth keeps the
    stored value — a mask placeholder never becomes stored data (with
    nothing stored it is dropped, and masked leaves of genuinely new values
    are stripped).
    """
    return _merge_masked(
        stored if isinstance(stored, dict) else {},
        incoming if isinstance(incoming, dict) else {},
    )


def mask_sensitive_value(value: Any) -> Any:
    """Recursively mask sensitive-keyed entries at ANY depth (nested objects
    in manually edited settings must not leak through views either); the
    structure itself is retained."""
    if isinstance(value, dict):
        return {
            key: (SECRET_MASK if v not in (None, "") and is_sensitive_extra_key(str(key)) else mask_sensitive_value(v))
            for key, v in value.items()
        }
    if isinstance(value, list):
        return [mask_sensitive_value(item) for item in value]
    return value


def _mask_extra_secrets(extra: dict[str, Any]) -> dict[str, Any]:
    """View helper: secret-looking extra values never leave the server raw."""
    return mask_sensitive_value(extra)


def merge_masked_update(current: Settings, update: dict[str, Any]) -> Settings:
    """Apply a settings update; SECRET_MASK values keep the stored secret.

    Model entries in the same payload keep their stored api_key under a mask,
    and their `extra` merges recursively (JSON null removes a key; a
    masked/empty sensitive value never becomes the stored value).
    """
    current_data = current.model_dump()
    update, profile_extra_merges = _preserve_profile_secrets(current_data, update)
    merged = _deep_merge(current_data, update)
    for (role, pid), extra in profile_extra_merges.items():
        merged.setdefault("profiles", {}).setdefault(role, {}).setdefault(pid, {})["extra"] = extra
    return Settings.model_validate(merged)


SENSITIVE_EXTRA_RE = re.compile(
    r"(?i)(key|token|secret|password|authorization|credential"
    r"|ref[_-]?audio|ref[_-]?wav|ref[_-]?text|ref[_-]?transcript|reference)"
)


def is_sensitive_extra_key(key: str) -> bool:
    """Extra keys that must never reach browser-visible views.

    Covers credentials AND voice-reference material (ref_audio base64 WAVs,
    ref_text reference transcripts — e.g. Chatterbox voice cloning inputs).
    Masked in views; masked/empty updates keep the stored server-side value;
    real values are still sent to the backend requests.
    """
    return bool(SENSITIVE_EXTRA_RE.search(str(key)))


def reference_fingerprint(extra: dict[str, Any] | None, field: str = "ref_audio") -> str:
    """SHA256 fingerprint of the EFFECTIVE TTS voice-reference material.

    Convention (fixed 2026-09-12): the hash runs over the LITERAL
    string stored under ``extra["ref_audio"]`` (its UTF-8 bytes) — not over
    decoded audio. The fingerprint is safe metadata: it proves WHICH reference
    a frozen session / first TTS request uses (same-model profiles such as two
    ``chatterbox-fp16`` voices are distinguishable) without exposing the
    reference data itself. Returns "" when no reference material is
    configured (e.g. Qwen native-voice profiles).
    """
    if not isinstance(extra, dict):
        return ""
    value = extra.get(field)
    if not isinstance(value, str) or not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_secretish_key(key: str) -> bool:
    return is_sensitive_extra_key(key)


def _preserve_profile_secrets(
    current_data: dict[str, Any],
    update: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    """Masked profile api_keys / nested secret extras keep the stored values;
    profile `extra` payloads merge recursively (masks retained at any depth,
    JSON null removes at its path). Returns the sanitized update plus the
    fully merged profile extras keyed (role, pid) for the post-deep-merge
    assign in merge_masked_update (removals must not be resurrected)."""
    update = dict(update)
    current_profiles = current_data.get("profiles") or {}
    incoming_profiles = update.get("profiles")
    profile_extra_merges: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(incoming_profiles, dict):
        merged_profiles: dict[str, Any] = {}
        for role, profiles in incoming_profiles.items():
            if not isinstance(profiles, dict):
                merged_profiles[role] = profiles
                continue
            merged_role: dict[str, Any] = {}
            for pid, profile_update in profiles.items():
                if not isinstance(profile_update, dict):
                    merged_role[pid] = profile_update
                    continue
                stored = (current_profiles.get(role) or {}).get(pid)
                profile_update = dict(profile_update)
                if profile_update.get("api_key") in (None, "", SECRET_MASK):
                    profile_update.pop("api_key", None)
                extra_update = profile_update.get("extra")
                if isinstance(extra_update, dict):
                    stored_extra = (stored or {}).get("extra") or {}
                    profile_extra_merges[(str(role), str(pid))] = merge_extra_values(
                        stored_extra, extra_update
                    )
                    profile_update["extra"] = {}  # placeholder until post-merge assign
                merged_role[pid] = profile_update
            merged_profiles[role] = merged_role
        update["profiles"] = merged_profiles
    return update, profile_extra_merges


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class SettingsStore:
    def __init__(self, data_dir: str | None = None, settings: Settings | None = None):
        self.data_dir = data_dir or default_data_dir()
        self.path = os.path.join(self.data_dir, "settings.json")
        self._settings = settings or Settings()
        self._loaded = False
        self._error: SettingsLoadError | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> Settings:
        async with self._lock:
            if not self._loaded:
                await self._load()
            if self._error is not None:
                raise self._error
            return self._settings.model_copy(deep=True)

    @property
    def load_error(self) -> SettingsLoadError | None:
        return self._error

    async def update(self, update: dict[str, Any]) -> Settings:
        async with self._lock:
            if not self._loaded:
                await self._load()
            if self._error is not None:
                raise self._error
            update = dict(update)
            requested_version = update.pop("version", None)
            if requested_version is not None and requested_version != self._settings.version:
                raise ValueError(
                    f"settings version cannot be changed through updates (stored: {self._settings.version})"
                )
            try:
                merged = merge_masked_update(self._settings, update)
            except ValidationError as exc:
                raise ValueError(_validation_message(exc)) from exc
            await self._persist(merged)
            # commit in memory only after the atomic write succeeded
            self._settings = merged
            return self._settings.model_copy(deep=True)

    async def mutate(self, mutator) -> Settings:
        """Apply a mutation (models, voices, sets) atomically under the store
        lock; revalidates and saves once.
        The in-memory state is committed only after the write succeeded."""
        async with self._lock:
            if not self._loaded:
                await self._load()
            if self._error is not None:
                raise self._error
            candidate = self._settings.model_copy(deep=True)
            try:
                mutator(candidate)
                candidate = Settings.model_validate(candidate.model_dump())
            except ValidationError as exc:
                # msg-only validation text: pydantic error strings embed the
                # submitted input values (secrets/reference material must
                # never be echoed in the error body)
                raise ValueError(_validation_message(exc)) from exc
            await self._persist(candidate)
            self._settings = candidate
            return self._settings.model_copy(deep=True)

    async def _load(self) -> None:
        self._loaded = True
        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None, self._read_file
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            self._error = SettingsLoadError(f"cannot read settings file: {exc}")
            return
        if raw is None or not raw.strip():
            self._error = SettingsLoadError(
                "settings file is empty (malformed); restore the private backup or delete it to start fresh"
            )
            return
        try:
            data = json.loads(raw)
        except ValueError as exc:
            self._error = SettingsLoadError(f"malformed settings file ({exc}); restore the private backup")
            return
        if not isinstance(data, dict):
            self._error = SettingsLoadError("malformed settings file: JSON object expected")
            return
        version = data.get("version")
        if version in (1, 2):
            # any migration failure must be sticky: without this, later
            # get() calls would silently return empty defaults
            try:
                await self._migrate(data, version)
            except SettingsLoadError as exc:
                self._error = exc
            return
        if version != SETTINGS_VERSION:
            self._error = SettingsLoadError(
                f"unsupported settings version {version!r} (supported: 1, 2, {SETTINGS_VERSION}); "
                "not falling back to empty defaults"
            )
            return
        try:
            self._settings = Settings.model_validate(data)
        except ValidationError as exc:
            self._error = SettingsLoadError(
                f"invalid settings file: {_validation_message(exc)}; restore the private backup"
            )

    def _read_file(self) -> str | None:
        with open(self.path, "r", encoding="utf-8") as fh:
            return fh.read()

    async def _migrate(self, data: dict[str, Any], version: int) -> None:
        """v1/v2 -> v3, with an exact private backup of the original taken
        BEFORE any write (never overwriting an earlier backup of the same
        version). Any failure raises; the original file stays untouched and
        the store records a sticky load error."""
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, self._read_file)
        except OSError as exc:
            raise SettingsLoadError(f"cannot re-read settings file for backup: {exc}") from exc
        name = SETTINGS_V1_BACKUP_NAME if version == 1 else SETTINGS_V2_BACKUP_NAME
        backup_path = os.path.join(self.data_dir, name)
        await loop.run_in_executor(None, _backup_once, backup_path, raw or "")
        migrated = migrate_v2_data(migrate_v1_data(data) if version == 1 else data)
        try:
            settings = Settings.model_validate(migrated)
        except ValidationError as exc:
            raise SettingsLoadError(
                f"settings migration produced invalid settings: {_validation_message(exc)}"
            ) from exc
        try:
            await loop.run_in_executor(
                None,
                _atomic_write_private,
                self.path,
                json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            )
        except OSError as exc:
            raise SettingsLoadError(
                f"migration write failed, original file kept: {exc}; backup at {backup_path}"
            ) from exc
        # commit only after the atomic write succeeded
        self._settings = settings

    async def _persist(self, settings: Settings) -> None:
        payload = json.dumps(settings.model_dump(), ensure_ascii=False, indent=2)
        await asyncio.get_running_loop().run_in_executor(None, _atomic_write_private, self.path, payload)


def _atomic_write_private(path: str, payload: str) -> None:
    """Create-and-replace atomically; the file is mode 0600 from creation on."""
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(path + ".tmp", path)
    except BaseException:
        try:
            os.unlink(path + ".tmp")
        except OSError:
            pass
        raise


def _backup_once(path: str, payload: str) -> None:
    """Persist an original exactly once: a previously saved backup is never
    overwritten (the first original wins)."""
    if os.path.exists(path):
        return
    _atomic_write_private(path, payload)


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{loc or 'settings'}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)


# Round 2 / CP3: the models whose native voice labels the old code trusted
# unconditionally. Kept here ONLY as the migration source: rows carrying one of
# these models are moved to capabilities.native_voice_labels == "honoured";
# every other existing row becomes "unknown" (today's "not trustworthy").
# New code must never consult this list for behaviour.
LEGACY_TRUSTWORTHY_NATIVE_ENGINE_MODELS = (
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "mlx-community--Voxtral-4B-TTS-2603-mlx-4bit",
)


def migrate_native_voice_labels(data: dict[str, Any]) -> bool:
    """Round 2 / CP3: replace the hard-coded trust list with row data.

    Only rows that do NOT carry ``native_voice_labels`` at all are filled: a
    legacy-list model becomes ``"honoured"``, every other model ``"unknown"``.
    A row with ANY explicit value (including ``"unknown"``) is never touched, so
    an operator's deliberate choice survives a reload (C2). Ids, references and
    every other field stay untouched. Returns True when the data was changed
    (so a second run is a no-op / idempotent).
    """
    changed = False
    tts = (data.get("profiles") or {}).get("tts")
    if not isinstance(tts, dict):
        return False
    for row in tts.values():
        if not isinstance(row, dict):
            continue
        caps = row.get("capabilities")
        if not isinstance(caps, dict):
            continue
        if "native_voice_labels" in caps:
            continue  # explicit value (even "unknown") is authoritative
        model = str(row.get("model") or "")
        caps["native_voice_labels"] = (
            "honoured"
            if model in LEGACY_TRUSTWORTHY_NATIVE_ENGINE_MODELS
            else "unknown"
        )
        changed = True
    return changed


def migrate_v1_data(data: dict[str, Any]) -> dict[str, Any]:
    """v1 dict -> v2 dict: the single backend per role becomes the three
    baseline model entries."""
    migrated = dict(data)
    migrated["version"] = 2
    profiles: dict[str, dict[str, Any]] = {"stt": {}, "llm": {}, "tts": {}}
    for role in ROLES:
        role_data = data.get(role)
        if isinstance(role_data, dict) and (role_data.get("model") or role_data.get("base_url")):
            pid = f"{role}-baseline"
            profile = dict(role_data)
            profile["id"] = pid
            profile["role"] = role
            profile.setdefault("display_name", f"Baseline {role.upper()}")
            profiles[role][pid] = profile
    migrated["profiles"] = profiles
    return migrated


def _model_id(model: str, display_name: str, taken: set[str]) -> str:
    """A model entry's id, derived from its model name (ids are never shown)."""
    base = re.sub(r"[^a-z0-9._-]+", "-", (model or display_name or "model").lower()).strip("-._")
    base = (base or "model")[:60]
    if not base[0].isalnum():
        base = "m" + base[:59]
    candidate, n = base, 2
    while candidate in taken:
        candidate = f"{base[:57]}-{n}"
        n += 1
    taken.add(candidate)
    return candidate


def _language_format(value: str) -> str:
    """A v2 model language ("de", "German", "") as a v3 language_format."""
    value = (value or "").strip()
    if not value or re.fullmatch(r"[a-zA-Z]{2,3}([-_][a-zA-Z]{2,4})?", value) or value == "auto":
        return "code"
    return "name"


def migrate_v2_data(data: dict[str, Any]) -> dict[str, Any]:
    """v2 dict -> v3 dict.

    - Model entries: ids derived from the model name (every reference
      follows); the stored language becomes language_format.
    - Sets: the active revision's spec (else the draft) is the set; enabled =
      available to clients; earlier revisions become Restore versions. A set
      now speaks with the models and recording as they are.
    - Dropped: presets, the default preset, voice-routing mappings, the
      legacy backend triple.
    - Recordings are transcribed with the listening model the old baseline
      pointed at, unless one was chosen.
    """
    migrate_native_voice_labels(data)
    out: dict[str, Any] = {"version": SETTINGS_VERSION}
    if isinstance(data.get("session"), dict):
        out["session"] = data["session"]
    renames: dict[str, dict[str, str]] = {role: {} for role in ROLES}
    profiles: dict[str, dict[str, Any]] = {role: {} for role in ROLES}
    for role in ROLES:
        rows = ((data.get("profiles") or {}).get(role) or {})
        taken: set[str] = set()
        for old_id in sorted(rows):
            row = dict(rows[old_id])
            new_id = _model_id(str(row.get("model") or ""), str(row.get("display_name") or ""), taken)
            renames[role][old_id] = new_id
            row["id"] = new_id
            row["role"] = role
            row["language_format"] = _language_format(str(row.pop("language", "") or ""))
            profiles[role][new_id] = row
    out["profiles"] = profiles

    def spec_v3(spec: Any) -> dict[str, Any]:
        spec = dict(spec or {})
        for role in ROLES:
            key = f"{role}_profile"
            if spec.get(key):
                spec[key] = renames[role].get(spec[key], spec[key])
        return spec

    sets: dict[str, Any] = {}
    for pid, cp in (data.get("conversation_profiles") or {}).items():
        if not isinstance(cp, dict):
            continue
        active = cp.get("active") if isinstance(cp.get("active"), dict) else None
        spec = active["spec"] if active else cp.get("draft")
        history = [
            {"saved_at": float(rev.get("activated_at") or 0), "spec": spec_v3(rev.get("spec"))}
            for rev in (cp.get("history") or [])
            if isinstance(rev, dict) and isinstance(rev.get("spec"), dict)
        ][:SET_HISTORY_MAX]
        sets[pid] = {
            "id": pid,
            "display_name": cp.get("display_name") or pid,
            "description": cp.get("description") or "",
            "spec": spec_v3(spec),
            "enabled": bool(cp.get("available_to_clients")) and active is not None,
            "history": history,
            "created_at": cp.get("created_at") or time.time(),
            "updated_at": (active or {}).get("activated_at") or cp.get("draft_updated_at") or time.time(),
            "spec_saved_at": (active or {}).get("activated_at") or cp.get("draft_updated_at") or time.time(),
        }
    out["conversation_profiles"] = sets
    out["voices"] = data.get("voices") or {}
    library = dict(data.get("voice_library") or {})
    stt = library.get("transcription_stt_profile") or ""
    stt = renames["stt"].get(stt or "stt-baseline", "")
    if not stt and profiles["stt"]:
        stt = sorted(profiles["stt"])[0]
    library["transcription_stt_profile"] = stt
    out["voice_library"] = library
    out["default_set"] = ""
    return out


class ServerConfig:
    """Process-level configuration that never ships in the repository."""

    def __init__(self, *, host: str, port: int, data_dir: str, auth_token: str):
        self.host = host
        self.port = port
        self.data_dir = data_dir
        self.auth_token = auth_token

    @classmethod
    def load(cls, *, host: str = "127.0.0.1", port: int = 8000, data_dir: str | None = None) -> "ServerConfig":
        data_dir = data_dir or default_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        token = os.environ.get(AUTH_TOKEN_ENV, "").strip()
        server_file = os.path.join(data_dir, "server.json")
        if not token:
            stored = cls._read_token(server_file)
            if stored:
                token = stored
            else:
                token = secrets.token_urlsafe(32)
                cls._write_token(server_file, token)
        return cls(host=host, port=port, data_dir=data_dir, auth_token=token)

    @staticmethod
    def _read_token(path: str) -> str | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            token = data.get("auth_token")
            return token if isinstance(token, str) and token else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_token(path: str, token: str) -> None:
        _atomic_write_private(path, json.dumps({"auth_token": token}))
