"""Model entries: masking, partial updates, and the per-call selection.

Security properties:
- Credentials live server-side only; every outward view masks the api_key and
  secret-looking `extra` values.
- A call's selection is a deep copy, so a running call is frozen: later edits
  to models, voices or sets never modify an existing call.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .config import (
    ROLES,
    SECRET_MASK,
    BackendProfile,
    Settings,
    mask_sensitive_value,
    merge_extra_values,
)

__all__ = [
    "SelectionError",
    "ResolvedSelection",
    "mask_profile",
    "is_masked_secret",
    "merge_profile_masked",
]


class SelectionError(ValueError):
    """A client's model/set selection cannot be resolved (machine-readable code)."""

    def __init__(self, message: str, code: str = "unknown_model"):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- masking

def mask_profile(profile: BackendProfile | dict[str, Any]) -> dict[str, Any]:
    data = profile.model_dump() if isinstance(profile, BaseModel) else dict(profile)
    if data.get("api_key"):
        data["api_key"] = SECRET_MASK
    extra = data.get("extra")
    if isinstance(extra, dict):
        # recursive: sensitive entries are masked at ANY depth; raw data
        # never leaves the server
        masked = mask_sensitive_value(extra)
        # a recording stored under the model's own field names is reference
        # material too
        for name in _reference_fields(data):
            if masked.get(name) not in (None, ""):
                masked[name] = SECRET_MASK
        data["extra"] = masked
    return data


def _reference_fields(data: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(data[k]) for k in ("reference_audio_field", "reference_text_field") if data.get(k)
    )


def is_masked_secret(value: Any) -> bool:
    return value in (None, "", SECRET_MASK)


def merge_profile_masked(current: BackendProfile, update: dict[str, Any]) -> BackendProfile:
    """Partial profile update; masked api_key / secret-ish extra values keep
    the stored secret. id/role are immutable. In `extra`, JSON null is an
    explicit REMOVE of the stored key; a masked/empty sensitive value is
    never stored as data (it keeps the stored secret; with nothing stored it
    adds nothing)."""
    update = {k: v for k, v in update.items() if k not in ("id", "role")}
    data = current.model_dump()
    incoming_key = update.get("api_key")
    if is_masked_secret(incoming_key):
        update.pop("api_key", None)
    incoming_extra = update.get("extra")
    if isinstance(incoming_extra, dict):
        # a mask (or empty value) under a reference field name keeps what is
        # stored, like any other secret; with nothing stored it adds nothing
        stored_extra = data.get("extra") or {}
        incoming_extra = dict(incoming_extra)
        for name in _reference_fields(data) + _reference_fields(update):
            if name in incoming_extra and incoming_extra[name] in (SECRET_MASK, ""):
                if name in stored_extra:
                    incoming_extra[name] = stored_extra[name]
                else:
                    del incoming_extra[name]
        update = {**update, "extra": incoming_extra}
    merged = _deep_merge(data, update)
    if isinstance(incoming_extra, dict):
        # recursive merge (backward compatible: nested dicts merge, arrays
        # replace, null removes at its path, masks keep stored values at any
        # depth) — assigned AFTER the deep merge so explicit removals can
        # never be resurrected from the original stored extras
        merged["extra"] = merge_extra_values(data.get("extra") or {}, incoming_extra)
    # a reference field that was renamed away: what is stored under the old
    # name is reference material nothing recognises any more; drop it
    # (also when the form sent it back masked)
    for name in set(_reference_fields(data)) - set(_reference_fields(merged)):
        (merged.get("extra") or {}).pop(name, None)
    return BackendProfile.model_validate(merged)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class ResolvedSelection:
    """A frozen configuration snapshot for one call."""

    def __init__(
        self,
        *,
        settings: Settings,
        set_id: str,
        profile_ids: dict[str, str],
        voice: str = "",
        language: str = "",
        display_voice: str = "",
        ref_sha256: str = "",
    ):
        self.settings = settings
        self.set_id = set_id
        self.profile_ids = profile_ids
        # the voice the speaking model is asked for
        self.voice = voice
        self.language = language
        # the voice's name for display: a recording's voice by its library
        # name, a built-in voice by its label
        self.display_voice = display_voice or voice
        # sha256 of the voice recording the speaking model gets ("" = none)
        self.ref_sha256 = ref_sha256

    def info(self) -> dict[str, Any]:
        """Safe runtime metadata for session.created."""
        roles: dict[str, Any] = {}
        for role in ROLES:
            pid = self.profile_ids.get(role, "")
            profile = self.settings.profiles.get(role, {}).get(pid) if pid else None
            if profile is None:
                roles[role] = {"profile": "", "model": getattr(self.settings, role).model}
            else:
                roles[role] = {"profile": pid, "display_name": profile.display_name, "model": profile.model}
        tts_block = roles.get("tts", {})
        if self.voice:
            tts_block["voice"] = self.voice
        if self.display_voice and self.display_voice != self.voice:
            tts_block["display_voice"] = self.display_voice
        if self.ref_sha256:
            tts_block["ref_sha256"] = self.ref_sha256
        return {
            "set": self.set_id,
            "stt": roles.get("stt", {}),
            "llm": roles.get("llm", {}),
            "tts": tts_block,
            "language": self.language or self.settings.session.language,
        }
