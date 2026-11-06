"""A3: TTS materialisation (contract §4) — library/native/none, reference
fields never leak from the speaking model's entry, bounded base64 LRU; a call
resolves the set against the CURRENT model entries.

Unit-level, no network.
"""

import base64

import pytest

from realtimevoice.config import (
    BackendSettings,
    BackendProfile,
    ConversationProfile,
    ProfileSpec,
    Settings,
    Voice,
    VoiceSelection,
    language_for_model,
)
from realtimevoice.profiles import (
    Base64LRU,
    ProfileResolutionError,
    materialise_tts_extra,
    resolve_profile,
)
from realtimevoice.voices import SampleStore, new_sample

from test_profiles_store import canonicalise_wav, sine, wav_bytes  # noqa: E402


def _settings():
    s = Settings()
    for role in ("stt", "llm", "tts"):
        s.profiles[role]["alt"] = BackendProfile(
            id="alt", role=role, base_url="http://b/v1", model=f"{role}-m"
        )
    s.profiles["tts"]["alt"].capabilities.voice_cloning = "reference_audio"
    s.profiles["tts"]["alt"].voice = "backend-voice"
    s.profiles["tts"]["alt"].extra = {"ref_audio": "LEAKME", "ref_text": "leak", "temperature": 0.5}
    return s


def _voice(settings):
    c = canonicalise_wav(wav_bytes(sine(1500)))
    sample = new_sample(c, source="upload", created_at=0.0)
    sample.transcript = "hallo welt"
    sample.transcript_confirmed = True
    settings.voices["anna"] = Voice(
        id="anna", display_name="Anna", samples=[sample],
        reference_sample_id=sample.id, created_at=0.0, updated_at=0.0,
    )
    return c, sample


def _set(settings, *, enabled=True, **spec_fields):
    fields = {"stt_profile": "alt", "llm_profile": "alt", "tts_profile": "alt", "language": "de"}
    fields.update(spec_fields)
    profile = ConversationProfile(id="prof", display_name="p", spec=ProfileSpec(**fields), enabled=enabled,
                                  created_at=0.0, updated_at=0.0)
    settings.conversation_profiles["prof"] = profile
    return profile


def test_resolve_library_materialises_reference_and_swaps_out_entry_ref(tmp_path):
    s = _settings()
    c, _ = _voice(s)
    store = SampleStore(str(tmp_path))
    store.write("anna", c.sha256, c.data)
    _set(s, voice=VoiceSelection(kind="library", voice_id="anna"))
    selection, route = resolve_profile(s, "prof", sample_store=store, b64_cache=Base64LRU())
    assert selection.settings.tts.extra["ref_text"] == "hallo welt"
    assert base64.b64decode(selection.settings.tts.extra["ref_audio"]) == c.data
    assert selection.settings.tts.extra["temperature"] == 0.5
    assert "LEAKME" not in selection.settings.tts.extra.get("ref_audio", "")
    assert selection.settings.session.voice == "backend-voice"
    assert route["voice"]["ref_sha256"] == c.sha256
    # the stored entry is never changed by resolution
    assert s.profiles["tts"]["alt"].extra["ref_audio"] == "LEAKME"


def test_resolve_native_drops_entry_reference():
    s = _settings()
    _set(s, voice=VoiceSelection(kind="native", native_voice="ryan"))
    selection, _ = resolve_profile(s, "prof")
    assert "ref_audio" not in selection.settings.tts.extra
    assert "ref_text" not in selection.settings.tts.extra
    assert selection.settings.session.voice == "ryan"
    assert selection.settings.tts.extra["temperature"] == 0.5


def test_resolve_none_strips_entry_reference():
    s = _settings()
    _set(s)
    selection, _ = resolve_profile(s, "prof")
    # kind=none: the entry as-is, but reference material never comes from it
    assert "ref_audio" not in selection.settings.tts.extra
    assert selection.settings.session.voice == "backend-voice"


def test_the_entry_is_read_when_a_call_starts():
    """Key rotation and any other entry edit reach the next call; a call
    already resolved keeps its copy."""
    s = _settings()
    s.profiles["tts"]["alt"].api_key = "sk-rotation-1"
    _set(s)
    first, _ = resolve_profile(s, "prof")
    assert first.settings.tts.api_key == "sk-rotation-1"
    s.profiles["tts"]["alt"].api_key = "sk-rotation-2"
    s.profiles["tts"]["alt"].model = "tts-m2"
    s.profiles["tts"]["alt"].extra["temperature"] = 0.9
    second, _ = resolve_profile(s, "prof")
    assert second.settings.tts.api_key == "sk-rotation-2"
    assert second.settings.tts.model == "tts-m2"
    assert second.settings.tts.extra["temperature"] == 0.9
    # the first call's copy is untouched
    assert first.settings.tts.api_key == "sk-rotation-1"
    assert first.settings.tts.model == "tts-m"
    assert first.settings.tts.extra["temperature"] == 0.5


def test_a_switched_off_or_unknown_set_is_refused():
    s = _settings()
    _set(s, enabled=False)
    with pytest.raises(ProfileResolutionError) as off:
        resolve_profile(s, "prof")
    assert off.value.code == "profile_not_available"
    with pytest.raises(ProfileResolutionError) as unknown:
        resolve_profile(s, "nobody")
    assert unknown.value.code == "unknown_conversation_profile"


def test_a_missing_model_entry_is_refused_at_resolution():
    s = _settings()
    _set(s, llm_profile="gone")
    with pytest.raises(ProfileResolutionError) as exc:
        resolve_profile(s, "prof")
    assert exc.value.code == "unknown_profile"


def test_each_model_gets_the_sets_language_in_its_own_format():
    s = _settings()
    s.profiles["stt"]["alt"].language_format = "code"
    s.profiles["llm"]["alt"].language_format = "none"
    s.profiles["tts"]["alt"].language_format = "name"
    _set(s, language="de")
    selection, _ = resolve_profile(s, "prof")
    sent = {
        role: language_for_model(getattr(selection.settings, role).language,
                                 getattr(selection.settings, role).language_format)
        for role in ("stt", "llm", "tts")
    }
    assert sent == {"stt": "de", "llm": "", "tts": "German"}
    assert selection.settings.session.language == "de"


def test_base64_lru_reuses_encoding_across_two_calls():
    s = Settings()
    _, sample = _voice(s)
    voice = s.voices["anna"]
    cache = Base64LRU(max_entries=2)
    extra1 = materialise_tts_extra(BackendSettings(), voice, sample, sample_bytes=b"DATA", b64_cache=cache)
    extra2 = materialise_tts_extra(BackendSettings(), voice, sample, sample_bytes=b"DATA", b64_cache=cache)
    assert extra2["ref_audio"] == extra1["ref_audio"] == base64.b64encode(b"DATA").decode()
    assert len(cache) == 1  # same sha -> one entry, encoded once


def test_set_output_pacing_and_turn_detection_defaults_apply():
    s = _settings()
    _set(s, output_pacing="realtime", turn_detection_default={"barge_in_confirm_ms": 120})
    selection, _ = resolve_profile(s, "prof")
    assert selection.settings.session.output_pacing == "realtime"
    assert selection.settings.session.turn_detection.barge_in_confirm_ms == 120


def test_set_default_output_pacing_is_off():
    s = _settings()
    _set(s)
    selection, _ = resolve_profile(s, "prof")
    assert selection.settings.session.output_pacing == "off"
