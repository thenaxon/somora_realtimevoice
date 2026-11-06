"""A1: set & voice data model, sample store, WAV validation, GC rules.

All audio is synthetic (numpy sine/noise); no private runtime data.
"""

import io
import math
import os
import stat
import struct
import wave

import numpy as np
import pytest

from realtimevoice.config import (
    BackendSettings,
    SET_HISTORY_MAX,
    ConversationProfile,
    ProfileSpec,
    SetVersion,
    Settings,
    Voice,
    VoiceLibrarySettings,
    VoiceSample,
    VoiceSelection,
)
from realtimevoice.profiles import (
    Base64LRU,
    materialise_tts_extra,
    validate_spec,
)
from realtimevoice.voices import (
    MAX_SAMPLE_BYTES,
    SampleStore,
    SampleValidationError,
    canonicalise_wav,
    gc_eligible,
    new_sample,
    sample_id_for,
)


def wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def sine(ms: int, rate: int = 24000, freq: float = 440.0, amp: float = 8000.0) -> bytes:
    n = int(rate * ms / 1000)
    x = (amp * np.sin(2 * math.pi * freq * np.arange(n) / rate)).astype("<i2")
    return x.tobytes()


def stereo_sine(ms: int, rate: int = 24000) -> bytes:
    n = int(rate * ms / 1000)
    x = (8000 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2")
    return np.repeat(x, 2).astype("<i2").tobytes()


def float_wav(ms: int, rate: int = 24000) -> bytes:
    n = int(rate * ms / 1000)
    x = (0.3 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<f4")
    return wav_bytes(x.tobytes(), rate=rate, sampwidth=4)


# ------------------------------------------------------------------- model

def test_settings_additive_fields_default_and_roundtrip():
    s = Settings()
    assert s.voices == {}
    assert s.conversation_profiles == {}
    assert s.voice_library == VoiceLibrarySettings()
    assert Settings.model_validate(s.model_dump()) == s


def test_a_set_holds_its_spec_switch_and_versions():
    p = ConversationProfile(id="anna-agent", display_name="Anna", created_at=0.0, updated_at=0.0)
    assert p.enabled is False and p.history == [] and p.spec == ProfileSpec()
    p.history.append(SetVersion(saved_at=1.0, spec=ProfileSpec(language="de")))
    dumped = p.model_dump()
    assert set(dumped) == {
        "id", "display_name", "description", "spec", "enabled", "history", "created_at", "updated_at",
        "spec_saved_at",
    }
    assert ConversationProfile.model_validate(dumped) == p
    assert SET_HISTORY_MAX == 10


def test_the_call_triple_is_never_stored():
    """settings.stt/llm/tts are the models of ONE resolved call: never part of
    what is saved."""
    s = Settings()
    s.tts.base_url = "http://b/v1"
    s.tts.api_key = "sk-never-stored"
    dumped = s.model_dump()
    assert not {"stt", "llm", "tts"} & set(dumped)
    assert "sk-never-stored" not in repr(dumped)


def test_public_id_rules_reject_reserved_and_bad_ids():
    for bad in ("tmp_foo", "profile-test-1", "A", "1", "has_underscore", "-leading", "x" * 65):
        with pytest.raises(Exception):
            Voice(id=bad, display_name="x", created_at=0.0, updated_at=0.0)
    ok = Voice(id="anna-de", display_name="Anna", created_at=0.0, updated_at=0.0)
    assert ok.id == "anna-de"


def test_voice_sample_original_filename_sanitised():
    s = VoiceSample(
        id="smp_" + "a" * 16,
        sha256="a" * 64,
        source="upload",
        original_filename="../../etc/passwd\x00name",
        duration_ms=1000,
        sample_rate=24000,
        size_bytes=100,
        created_at=0.0,
    )
    assert "/" not in s.original_filename and "\\" not in s.original_filename
    assert len(s.original_filename) <= 128


def test_capabilities_voice_cloning_fields():
    from realtimevoice.config import ProfileCapabilities

    c = ProfileCapabilities()
    assert c.voice_cloning == "none" and c.reference_text_required is False and c.max_reference_ms == 0
    c2 = ProfileCapabilities(voice_cloning="reference_audio", reference_text_required=True, max_reference_ms=10000)
    assert c2.voice_cloning == "reference_audio" and c2.max_reference_ms == 10000


# ------------------------------------------------------------- wav validation

def test_canonicalise_pcm16_mono_roundtrip():
    raw = wav_bytes(sine(1500))
    c = canonicalise_wav(raw)
    assert c.sample_rate == 24000 and c.channels == 1
    assert c.duration_ms == pytest.approx(1500, abs=2)
    assert c.sha256 and c.metrics["peak_dbfs"] < 0


def test_canonicalise_stereo_downmix_reduces_channels():
    c = canonicalise_wav(wav_bytes(stereo_sine(1500), channels=2))
    assert c.channels == 1
    # downmix keeps sample count (frames), not doubled samples
    assert c.duration_ms == pytest.approx(1500, abs=2)


def test_canonicalise_float32_converts_to_pcm16():
    c = canonicalise_wav(float_wav(1500))
    assert c.channels == 1 and c.duration_ms == pytest.approx(1500, abs=2)


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"not a wave at all", "unsupported_format"),
        (wav_bytes(sine(200)), "sample_too_short"),
        (wav_bytes(sine(31000)), "sample_too_long"),
        (wav_bytes(sine(1000), rate=8000), "unsupported_format"),
        (wav_bytes(sine(1000), rate=96000), "unsupported_format"),
        (b"RIFF" + b"\x00" * 40 + b"tail", "unsupported_format"),
    ],
)
def test_canonicalise_rejection_matrix(raw, code):
    with pytest.raises(SampleValidationError) as exc:
        canonicalise_wav(raw)
    assert exc.value.code == code


def test_canonicalise_too_large():
    with pytest.raises(SampleValidationError) as exc:
        canonicalise_wav(b"RIFF" + b"\x00" * (MAX_SAMPLE_BYTES + 1))
    assert exc.value.code == "too_large"


def test_sample_id_matches_sha_prefix():
    c = canonicalise_wav(wav_bytes(sine(1200)))
    assert sample_id_for(c.sha256) == "smp_" + c.sha256[:16]


# ---------------------------------------------------------------- file store

def test_sample_store_permissions_and_content_addressing(tmp_path):
    store = SampleStore(str(tmp_path))
    c = canonicalise_wav(wav_bytes(sine(1200)))
    path = store.write("anna", c.sha256, c.data)
    assert os.path.isfile(path)
    assert stat.S_IMODE(os.stat(tmp_path / "voices").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    # second write of identical bytes is a no-op, same path
    assert store.write("anna", c.sha256, c.data) == path
    assert store.read("anna", c.sha256) == c.data


def test_sample_store_atomic_no_temp_left(tmp_path):
    store = SampleStore(str(tmp_path))
    c = canonicalise_wav(wav_bytes(sine(1200)))
    store.write("anna", c.sha256, c.data)
    leftovers = [n for n in os.listdir(tmp_path / "voices" / "anna") if n.endswith(".tmp")]
    assert leftovers == []


def test_gc_eligible_blocks_referenced_shas(tmp_path):
    store = SampleStore(str(tmp_path))
    c = canonicalise_wav(wav_bytes(sine(1200)))
    store.write("anna", c.sha256, c.data)
    settings = Settings()
    sample = new_sample(c, source="upload")
    settings.voices["anna"] = Voice(
        id="anna",
        display_name="Anna",
        samples=[sample],
        reference_sample_id=sample.id,
        created_at=0.0,
        updated_at=0.0,
    )
    # referenced by the voice sample metadata -> not eligible
    assert gc_eligible(settings, "anna", c.sha256) is False
    # a live session reference also blocks
    assert gc_eligible({}, "anna", c.sha256, live_shas={c.sha256}) is False
    # a sha no longer referenced anywhere -> eligible
    assert gc_eligible(Settings(), "anna", "b" * 64) is True


# ------------------------------------------------------------ materialisation

def _library_voice(transcript="hallo welt"):
    c = canonicalise_wav(wav_bytes(sine(1200)))
    sample = new_sample(c, source="upload", created_at=0.0)
    sample.transcript = transcript
    sample.transcript_confirmed = True
    voice = Voice(id="anna", display_name="Anna", samples=[sample],
                  reference_sample_id=sample.id, created_at=0.0, updated_at=0.0)
    return voice, sample


def test_materialise_library_sets_ref_fields_and_uses_cache():
    entry_extra = {"ref_audio": "old", "ref_text": "old", "temperature": 0.7}
    voice, sample = _library_voice()
    cache = Base64LRU()
    extra = materialise_tts_extra(BackendSettings(extra=entry_extra), voice, sample, sample_bytes=b"WAVDATA", b64_cache=cache)
    assert "temperature" in extra and extra["temperature"] == 0.7
    assert extra["ref_text"] == "hallo welt"
    assert len(cache) == 1
    import base64

    assert base64.b64decode(extra["ref_audio"]) == b"WAVDATA"
    assert entry_extra["ref_audio"] == "old", "the entry's own extra is never modified"


def test_materialise_native_drops_reference_material():
    entry_extra = {"ref_audio": "old", "ref_text": "old", "temperature": 0.5}
    extra = materialise_tts_extra(BackendSettings(extra=entry_extra), None, None, sample_bytes=None)
    assert "ref_audio" not in extra and "ref_text" not in extra
    assert extra["temperature"] == 0.5


def test_base64_lru_is_bounded():
    cache = Base64LRU(max_entries=16)
    for i in range(20):
        cache.put(f"{i:064x}", "x")
    assert len(cache) == 16
    assert cache.get("0".zfill(64)) is None  # oldest evicted


# ------------------------------------------------------------- spec validation

def _settings_with_profiles():
    from realtimevoice.config import BackendProfile

    s = Settings()
    for role in ("stt", "llm", "tts"):
        s.profiles[role][f"{role}-x"] = BackendProfile(
            id=f"{role}-x", role=role, base_url="http://b/v1", model=f"{role}-model"
        )
    s.profiles["tts"]["tts-x"].capabilities.voice_cloning = "reference_audio"
    return s


def test_validate_spec_reports_missing_reference_and_transcript(tmp_path):
    s = _settings_with_profiles()
    spec = ProfileSpec(stt_profile="stt-x", llm_profile="llm-x", tts_profile="tts-x", language="de",
                       voice=VoiceSelection(kind="library", voice_id="anna"))
    issues = validate_spec(s, spec)
    codes = {i["code"] for i in issues}
    assert "missing_reference" in codes

    c = canonicalise_wav(wav_bytes(sine(1200)))
    store = SampleStore(str(tmp_path))
    store.write("anna", c.sha256, c.data)
    sample = new_sample(c, source="upload")
    s.voices["anna"] = Voice(id="anna", display_name="Anna", samples=[sample],
                             reference_sample_id=sample.id, created_at=0.0, updated_at=0.0)
    issues = validate_spec(s, spec, sample_store=store)
    codes = {i["code"] for i in issues}
    assert "transcript_missing" in codes
    s.voices["anna"].samples[0].transcript = "hallo"
    s.voices["anna"].samples[0].transcript_confirmed = True
    assert validate_spec(s, spec, sample_store=store) == []


def test_a_set_stores_only_names_never_model_data_or_reference_material():
    """A set names its models and voice; the key, the entry's extra and the
    recording are read when a call starts, so none of them is in the set."""
    s = _settings_with_profiles()
    s.profiles["tts"]["tts-x"].api_key = "sk-secret"
    s.profiles["tts"]["tts-x"].extra = {"ref_audio": "SECRETREF", "temperature": 0.6}
    voice, sample = _library_voice("hallo")
    s.voices["anna"] = voice
    spec = ProfileSpec(stt_profile="stt-x", llm_profile="llm-x", tts_profile="tts-x",
                       language="de", voice=VoiceSelection(kind="library", voice_id="anna"))
    profile = ConversationProfile(id="anna-agent", display_name="Anna", spec=spec, enabled=True,
                                  created_at=0.0, updated_at=0.0)
    s.conversation_profiles["anna-agent"] = profile
    dumped = repr(profile.model_dump())
    for leaked in ("sk-secret", "SECRETREF", "http://b/v1", sample.sha256, "tts-model", "temperature"):
        assert leaked not in dumped, leaked
