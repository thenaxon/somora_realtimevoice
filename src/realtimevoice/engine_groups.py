"""Presentation-time engine grouping (Round 2 / CP1, Option A).

The stored ``BackendProfile`` row conflates three concepts: the ENGINE
(endpoint + model + engine params), a VOICE (native preset or library
reference) and the stored config id. Two rows can therefore be the *same
engine* while differing only in voice/capability metadata — e.g.
``chatterbox-clara`` and ``chatterbox-jonas`` (same model, same endpoint,
different legacy embedded reference) or ``stt-baseline`` and its migration
snapshot (0 field diffs).

This module computes an **engine key** per row and groups rows by it. Nothing
is stored differently: the ids and rows are untouched, resolution is untouched,
and the grouping feeds ONLY presentation (the profile-editor selectors and the
``/api/tts-engines`` projection). It is a pure function of the rows, so it can
be introduced and rolled back with zero data risk.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .config import BackendProfile, is_sensitive_extra_key

__all__ = [
    "engine_key",
    "engine_id",
    "EngineGroup",
    "group_engines",
    "primary_row",
    "group_engine_view",
    "native_presets_trustworthy",
]

# Engine-identity extra keys. Everything else in `extra` is either a secret /
# voice reference (excluded via is_sensitive_extra_key) or an engine param that
# must split the key. Kept explicit so the contract is auditable.
_IDENTITY_EXTRA_EXCLUDE = ()

# Capabilities that describe voices, not the engine. A different value here
# must never split an engine group (design §3, C4).
_VOICE_CAPABILITY_KEYS = (
    "voices",
    "voice_cloning",
    "reference_text_required",
    "max_reference_ms",
    "evidence",
    "languages",
    "streaming",
)

# Engines whose native voice labels the model actually honours are identified
# by ROW DATA (``capabilities.native_voice_labels == "honoured"``), not by a
# hard-coded model list (Round 2 / CP3). The old list survives only in
# config.LEGACY_TRUSTWORTHY_NATIVE_ENGINE_MODELS as a one-time migration source.


def _non_voice_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """The engine-param part of `extra`: drop secrets and voice-reference keys."""
    out: dict[str, Any] = {}
    for key, value in (extra or {}).items():
        if is_sensitive_extra_key(str(key)):
            continue
        out[str(key)] = value
    return out


def engine_key(profile: BackendProfile) -> tuple:
    """The identity of an *engine*: endpoint + model + engine params.

    Deliberately excludes ``id``, ``display_name``, ``role`` (storage/label),
    ``api_key`` (masked, rotation must not fork the engine), ``voice`` and all
    ``capabilities`` (voice/capability metadata). A changed ``base_url``,
    ``model`` or non-secret/non-voice engine ``extra`` value yields a different
    key; a changed ``api_key``, ``voice``, ``capabilities.voices`` or
    ``extra.ref_audio`` does not.
    """
    params = {
        k: v
        for k, v in _non_voice_extra(profile.extra).items()
        if k not in (profile.reference_audio_field, profile.reference_text_field)
    }
    return (
        profile.adapter,
        profile.base_url,
        profile.model,
        profile.language_format,
        profile.request_timeout_s,
        profile.audio_format,
        profile.pcm_sample_rate,
        profile.reference_audio_field,
        profile.reference_text_field,
        profile.reference_audio_encoding,
        json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str),
    )


def engine_id(key: tuple) -> str:
    """A SAFE, stable identifier for an engine key.

    The raw key carries `base_url` (internal hostnames/ports must never leave
    the server), so outward views use this opaque SHA-derived id instead. It is
    stable for the lifetime of the underlying engine key and collision-resistant
    enough to group UI options; it reveals nothing about the endpoint.
    """
    payload = json.dumps(list(key), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "eng_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class EngineGroup:
    """One engine plus the stored rows that materialise it."""

    def __init__(self, key: tuple, rows: list[BackendProfile]):
        self.key = key
        # deterministic order: lexicographically smallest id first
        self.rows = sorted(rows, key=lambda p: p.id)

    @property
    def model(self) -> str:
        return self.rows[0].model if self.rows else ""

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


def _is_snapshot_or_baseline(profile: BackendProfile) -> bool:
    pid = profile.id
    return pid.endswith("-snapshot") or pid in (
        "stt-baseline",
        "llm-baseline",
        "tts-baseline",
    )


def primary_row(group: EngineGroup | Iterable[BackendProfile], referenced: str = "") -> BackendProfile:
    """The row that is actually written/used for a group (deterministic).

    Preference, in order (design §2, accepted by the owner):
      a) the row already referenced by the current spec/selection,
      b) a row that is neither a ``*-snapshot`` migration artifact nor a
         ``*-baseline`` mirror,
      c) the lexicographically smallest id.

    The returned row is the one whose metadata must be shown (C4).
    """
    rows = group.rows if isinstance(group, EngineGroup) else list(group)
    if not rows:
        raise ValueError("cannot pick a primary row from an empty group")
    if referenced:
        for row in rows:
            if row.id == referenced:
                return row
    non_artifact = [r for r in rows if not _is_snapshot_or_baseline(r)]
    if non_artifact:
        return sorted(non_artifact, key=lambda p: p.id)[0]
    return sorted(rows, key=lambda p: p.id)[0]


def group_engines(profiles: Iterable[BackendProfile]) -> list[EngineGroup]:
    """Group stored rows by engine key, preserving first-seen order of keys."""
    grouped: dict[tuple, list[BackendProfile]] = {}
    for profile in profiles:
        grouped.setdefault(engine_key(profile), []).append(profile)
    return [EngineGroup(key, rows) for key, rows in grouped.items()]


def native_presets_trustworthy(profile: BackendProfile) -> bool:
    """True when an engine's ``capabilities.voices`` labels are a durable
    identity. Fish-style engines ignore the label, so they must not be offered
    as named presets (C2/design §5).

    Round 2 / CP3: decided by ROW DATA — an operator promises the label is
    honoured via ``capabilities.native_voice_labels == "honoured"``. The default
    ``"unknown"`` is treated exactly like today's "not trustworthy", so a
    migrated row keeps the previous behaviour until it is explicitly trusted.
    """
    return profile.capabilities.native_voice_labels == "honoured"


def group_engine_view(
    group: EngineGroup,
    *,
    referenced: str = "",
    role: str = "",
) -> dict[str, Any]:
    """Safe presentation descriptor for one engine group.

    The descriptor is derived from the PRIMARY row (the one that is actually
    written) for every field that claims a real property — model, endpoint
    label, capabilities, evidence, languages, streaming. Sibling rows are
    listed only as ``variants`` (ids) and ``embeds_reference`` (whether ANY
    variant carries legacy reference material); no sibling's better metadata is
    borrowed (C4).

    Library voices are usable when the primary row advertises
    ``voice_cloning == "reference_audio"``. Native presets are offered from
    ``capabilities.voices`` ONLY when the engine's labels are trustworthy
    (C2); for a legacy speaker-named row the embedded reference is stripped in
    the profile path, so its label must never become a native preset.
    """
    primary = primary_row(group, referenced=referenced)
    cloning = primary.capabilities.voice_cloning
    if cloning not in ("reference_audio", "none", "unknown"):
        cloning = "unknown"
    native_voices: list[str] = []
    if native_presets_trustworthy(primary):
        native_voices = list(primary.capabilities.voices or [])
    embeds_reference = any(
        bool((r.extra or {}).get("ref_audio")) for r in group.rows
    )
    return {
        "engine_id": engine_id(group.key),
        "role": role or primary.role,
        "primary_id": primary.id,
        "id": primary.id,
        "display_name": primary.display_name or primary.id,
        "backend_label": primary.display_name or primary.id,
        "model": primary.model,
        "voice_cloning": cloning,
        "reference_text_required": bool(primary.capabilities.reference_text_required),
        "max_reference_ms": int(primary.capabilities.max_reference_ms or 0),
        "embeds_reference": embeds_reference,
        "native_voices": native_voices,
        "native_voice_labels": primary.capabilities.native_voice_labels,
        "evidence": primary.capabilities.evidence,
        "languages": list(primary.capabilities.languages or []),
        "streaming": primary.capabilities.streaming,
        "variants": group.ids,
    }