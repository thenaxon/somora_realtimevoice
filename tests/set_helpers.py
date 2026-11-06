"""Test helper: a minimal enabled set, so a realtime call has models.

A call that names no set uses the server's `default_set`; without one it is
refused (`no_set_selected`). Tests that only exercise the protocol add this
set and make it the default.
"""

from __future__ import annotations

import time

from realtimevoice.config import BackendProfile, ConversationProfile, ProfileSpec, Settings, VoiceSelection


def add_set(
    settings: Settings,
    set_id: str = "test-set",
    *,
    display_name: str = "",
    language: str = "de",
    enabled: bool = True,
    base_url: str = "http://fake.test/v1",
    models: dict[str, str] | None = None,
    language_formats: dict[str, str] | None = None,
    native_voice: str = "",
    make_default: bool = False,
) -> Settings:
    """Add one model per role (ids `<set_id>-stt` etc.) and a set using them.
    Returns the same Settings object."""
    models = models or {}
    language_formats = language_formats or {}
    ids: dict[str, str] = {}
    for role in ("stt", "llm", "tts"):
        pid = f"{set_id}-{role}"
        settings.profiles.setdefault(role, {})[pid] = BackendProfile(
            id=pid,
            role=role,
            display_name=f"{set_id} {role}",
            base_url=base_url,
            model=models.get(role, f"{role}-model"),
            language_format=language_formats.get(role, "code"),
        )
        ids[role] = pid
    now = time.time()
    voice = VoiceSelection(kind="native", native_voice=native_voice) if native_voice else VoiceSelection()
    settings.conversation_profiles[set_id] = ConversationProfile(
        id=set_id,
        display_name=display_name or set_id,
        spec=ProfileSpec(
            stt_profile=ids["stt"],
            llm_profile=ids["llm"],
            tts_profile=ids["tts"],
            voice=voice,
            language=language,
        ),
        enabled=enabled,
        created_at=now,
        updated_at=now,
    )
    if make_default:
        settings.default_set = set_id
    return settings


def settings_with_default_set(settings: Settings | None = None, **kwargs) -> Settings:
    """Settings whose calls without a named set get a minimal enabled set."""
    return add_set(settings or Settings(), make_default=True, **kwargs)
