"""Round 2 / CP1 (Option A): API projection + resolution regressions.

Covers:
- GET /api/tts-engines returns one Chatterbox and one Qwen3-TTS Base, keeps the
  existing fields and adds `variants`; no raw endpoint leaks (security).
- GET /api/catalog keeps the EXACT profile id sets (grouping is additive).
- C1: for each engine group, resolving the same (engine, voice selection)
  through each row of the group yields identical model, endpoint, engine params
  and materialised extra (ref_sha256 library; no ref_audio native/none).
- C1 exact-id regression for four sets saved switched on through the API.
- C2: legacy speaker rows are not offered as native presets.
- C3: the display names the library voice; the engine request is unchanged.

Synthetic audio only; capturing fake HTTP backend — never a real endpoint.
"""

import base64
import io
import math
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import BackendProfile, ProfileSpec, Settings, SettingsStore, VoiceSelection
from realtimevoice.profiles import resolve_profile, resolve_spec

AUTH = {"Authorization": "Bearer tok"}
SPEECH = b"\x01\x02" * 600
OLLAMA = "http://cap.test/v1"

REF_CLARA = b"RIFF-clara-reference"
REF_JONAS = b"RIFF-jonas-reference"


def wav_bytes(pcm: bytes, rate=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def sine_pcm(ms=1500, rate=24000) -> bytes:
    n = int(rate * ms / 1000)
    return (8000 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2").tobytes()


class Capture:
    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Hallo Welt"})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(SPEECH))
        if path.endswith("/models"):
            # empty list = "no model enumeration": the probe only fails when a
            # non-empty list lacks the configured model
            return httpx.Response(200, json={"data": []})
        if path.endswith("/chat/completions"):
            return httpx.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"error": {"message": "nf"}})


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def make_settings() -> Settings:
    """The brief's catalog shape: two rows of one engine per role where the
    brief had them, so the grouping is reproduced."""
    s = Settings()
    # STT
    s.profiles["stt"]["stt-qwen3-asr-0.6b"] = BackendProfile(
        id="stt-qwen3-asr-0.6b", role="stt", base_url=OLLAMA, model="Qwen3-ASR-0.6B-8bit"
    )
    s.profiles["stt"]["stt-qwen3-asr-0.6b-8bit-snapshot"] = BackendProfile(
        id="stt-qwen3-asr-0.6b-8bit-snapshot", role="stt", base_url=OLLAMA, model="Qwen3-ASR-0.6B-8bit"
    )
    s.profiles["stt"]["stt-qwen3-asr-1.7b"] = BackendProfile(
        id="stt-qwen3-asr-1.7b", role="stt", base_url=OLLAMA, model="Qwen3-ASR-1.7B-8bit"
    )
    s.profiles["stt"]["whisper-existing"] = BackendProfile(
        id="whisper-existing", role="stt", base_url=OLLAMA, model="whisper-large-v3-turbo"
    )
    # LLM
    s.profiles["llm"]["llm-gemma"] = BackendProfile(
        id="llm-gemma", role="llm", base_url=OLLAMA, model="gemma-4-26b-a4b-it-4bit"
    )
    s.profiles["llm"]["llm-gemma-4-26b-a4b-it-4bit-snapshot"] = BackendProfile(
        id="llm-gemma-4-26b-a4b-it-4bit-snapshot", role="llm", base_url=OLLAMA, model="gemma-4-26b-a4b-it-4bit"
    )
    s.profiles["llm"]["qwen14b-existing"] = BackendProfile(
        id="qwen14b-existing", role="llm", base_url=OLLAMA, model="qwen3-14b-4bit"
    )
    # TTS
    s.profiles["tts"]["qwen-default"] = BackendProfile(
        id="qwen-default", role="tts", base_url=OLLAMA, model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        capabilities={"voice_cloning": "none", "voices": ["ryan", "serena"]},
    )
    cb_caps = {"voice_cloning": "reference_audio", "max_reference_ms": 10000}
    s.profiles["tts"]["chatterbox-clara"] = BackendProfile(
        id="chatterbox-clara", role="tts", display_name="Chatterbox / Clara",
        base_url=OLLAMA, model="chatterbox-fp16", voice="clara-reference",
        extra={"ref_audio": _b64(REF_CLARA), "ref_text": "ich bin clara", "temperature": 0.8},
        capabilities=cb_caps,
    )
    s.profiles["tts"]["chatterbox-jonas"] = BackendProfile(
        id="chatterbox-jonas", role="tts", display_name="Chatterbox / Jonas",
        base_url=OLLAMA, model="chatterbox-fp16", voice="jonas-reference",
        extra={"ref_audio": _b64(REF_JONAS), "ref_text": "ich bin jonas", "temperature": 0.8},
        capabilities=cb_caps,
    )
    s.profiles["tts"]["tts-qwen3-base-clone"] = BackendProfile(
        id="tts-qwen3-base-clone", role="tts", display_name="Qwen3-TTS Base",
        base_url=OLLAMA, model="Qwen3-TTS-12Hz-1.7B-Base-8bit",
        capabilities={"voice_cloning": "reference_audio", "reference_text_required": True, "max_reference_ms": 0},
    )
    cv_caps = {"voice_cloning": "none", "voices": ["ryan", "serena"]}
    s.profiles["tts"]["qwen-ryan"] = BackendProfile(
        id="qwen-ryan", role="tts", display_name="Qwen Ryan", base_url=OLLAMA,
        model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", voice="ryan", capabilities=cv_caps,
    )
    s.profiles["tts"]["qwen-serena"] = BackendProfile(
        id="qwen-serena", role="tts", display_name="Qwen Serena", base_url=OLLAMA,
        model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", voice="serena", capabilities=cv_caps,
    )
    vox_caps = {"voice_cloning": "none", "voices": ["de_male", "de_female", "casual_male"]}
    s.profiles["tts"]["voxtral-casual"] = BackendProfile(
        id="voxtral-casual", role="tts", display_name="Voxtral casual", base_url=OLLAMA,
        model="mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", voice="casual_male", capabilities=vox_caps,
    )
    s.profiles["tts"]["voxtral-de-female"] = BackendProfile(
        id="voxtral-de-female", role="tts", display_name="Voxtral female", base_url=OLLAMA,
        model="mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", voice="de_female", capabilities=vox_caps,
    )
    s.profiles["tts"]["fish-native"] = BackendProfile(
        id="fish-native", role="tts", display_name="Fish", base_url=OLLAMA,
        model="fish-audio-s2-pro-8bit", voice="native",
        capabilities={"voice_cloning": "none", "voices": ["native"]},
    )
    s.voice_library.transcription_stt_profile = "stt-qwen3-asr-0.6b"
    return s


@pytest.fixture
def env(tmp_path):
    client = httpx.AsyncClient(transport=httpx.MockTransport(Capture().handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        yield tc, store, tmp_path


def _make_voice(tc, vid, raw_ref, display, transcript="Hallo Welt"):
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": display}).status_code == 201
    up = tc.post(
        f"/api/voices/{vid}/samples", headers=AUTH,
        files={"file": ("ref.wav", wav_bytes(sine_pcm()), "audio/wav")},
        data={"source": "upload"},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    tc.patch(f"/api/voices/{vid}/samples/{sid}", headers=AUTH,
             json={"transcript": transcript, "transcript_confirmed": True})
    tc.put(f"/api/voices/{vid}/reference", headers=AUTH, json={"sample_id": sid})
    return sid


# ------------------------------------------------- /api/tts-engines

def test_tts_engines_one_entry_per_engine_with_variants(env):
    tc, _, _ = env
    r = tc.get("/api/tts-engines", headers=AUTH)
    assert r.status_code == 200, r.text
    engines = r.json()["engines"]
    by_model: dict[str, list] = {}
    for e in engines:
        by_model.setdefault(e["model"], []).append(e)
    # one Chatterbox entry grouping both speaker-named rows
    cb = by_model["chatterbox-fp16"]
    assert len(cb) == 1, "Chatterbox must appear exactly once"
    assert set(cb[0]["variants"]) == {"chatterbox-clara", "chatterbox-jonas"}
    assert cb[0]["id"] in (cb[0]["variants"])
    # one Qwen3-TTS Base entry, distinct from CustomVoice
    base = by_model["Qwen3-TTS-12Hz-1.7B-Base-8bit"]
    assert len(base) == 1
    cv = by_model["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"]
    assert len(cv) == 1
    assert base[0]["id"] != cv[0]["id"]
    # existing fields kept
    for e in engines:
        for f in ("id", "display_name", "backend_label", "model", "voice_cloning",
                  "reference_text_required", "max_reference_ms", "embeds_reference"):
            assert f in e, f"field {f} must be kept"
    # C2: no native presets from the legacy speaker rows
    assert cb[0]["native_voices"] == []
    # trustworthy native presets are offered
    assert sorted(cv[0]["native_voices"]) == ["ryan", "serena"]
    assert by_model["fish-audio-s2-pro-8bit"][0]["native_voices"] == []


def test_tts_engines_never_leak_endpoint(env):
    tc, _, _ = env
    blob = tc.get("/api/tts-engines", headers=AUTH).text
    assert "cap.test" not in blob, "internal endpoint must never leave the server"
    assert "base_url" not in blob


# ------------------------------------------------------ /api/catalog

def test_catalog_profile_id_sets_unchanged(env):
    tc, _, _ = env
    cat = tc.get("/api/catalog", headers=AUTH).json()
    got = {role: {p["id"] for p in cat["roles"][role]["profiles"]} for role in ("stt", "llm", "tts")}
    assert got["stt"] == {
        "stt-qwen3-asr-0.6b", "stt-qwen3-asr-0.6b-8bit-snapshot",
        "stt-qwen3-asr-1.7b", "whisper-existing",
    }
    assert got["llm"] == {
        "llm-gemma", "llm-gemma-4-26b-a4b-it-4bit-snapshot", "qwen14b-existing",
    }
    assert got["tts"] == {
        "qwen-default", "chatterbox-clara", "chatterbox-jonas",
        "tts-qwen3-base-clone", "qwen-ryan", "qwen-serena",
        "voxtral-casual", "voxtral-de-female", "fish-native",
    }


def test_catalog_engine_groups_are_additive(env):
    tc, _, _ = env
    cat = tc.get("/api/catalog", headers=AUTH).json()
    tts_groups = cat["roles"]["tts"]["engine_groups"]
    assert len(tts_groups) == 5
    models = sorted(g["model"] for g in tts_groups)
    assert models == sorted([
        "chatterbox-fp16", "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", "fish-audio-s2-pro-8bit",
    ])
    assert len(cat["roles"]["stt"]["engine_groups"]) == 3
    assert len(cat["roles"]["llm"]["engine_groups"]) == 2
    # every grouped variant id is a real profile id (nothing invented)
    for role in ("stt", "llm", "tts"):
        ids = {p["id"] for p in cat["roles"][role]["profiles"]}
        for g in cat["roles"][role]["engine_groups"]:
            assert set(g["variants"]) <= ids


# ------------------------------------------------------------------- C1

def _resolve_row(settings, tts_pid, voice_kind, voice_id="", native_voice="", sample_store=None):
    """Resolve a synthetic set pointing at one TTS row."""
    from types import SimpleNamespace

    spec = ProfileSpec(
        stt_profile="stt-qwen3-asr-0.6b", llm_profile="llm-gemma", tts_profile=tts_pid,
        voice=VoiceSelection(kind=voice_kind, voice_id=voice_id, native_voice=native_voice),
        language="de",
    )
    selection, _route = resolve_spec(settings, SimpleNamespace(id="probe"), spec, sample_store=sample_store)
    return selection


def test_c1_rows_in_one_engine_group_are_interchangeable_library(env):
    tc, store, _ = env
    _make_voice(tc, "voice-1", b"RIFF-voice-1", "Speaker")
    settings = tc.app.state.settings_store._settings

    store_ = tc.app.state.sample_store

    def resolved(pid):
        return _resolve_row(settings, pid, "library", voice_id="voice-1", sample_store=store_)

    a = resolved("chatterbox-clara")
    b = resolved("chatterbox-jonas")
    # identical engine request material: model, endpoint, engine params
    assert a.settings.tts.model == b.settings.tts.model == "chatterbox-fp16"
    assert a.settings.tts.base_url == b.settings.tts.base_url
    assert a.settings.tts.adapter == b.settings.tts.adapter
    assert a.settings.tts.language == b.settings.tts.language
    assert a.settings.tts.request_timeout_s == b.settings.tts.request_timeout_s
    # engine params: same non-voice extra; library sample replaces ref material
    non_voice = lambda extra: {k: v for k, v in extra.items() if k not in ("ref_audio", "ref_text")}
    assert non_voice(a.settings.tts.extra) == non_voice(b.settings.tts.extra)
    # materialised library extra is identical (the user's sample)
    assert a.settings.tts.extra["ref_audio"] == b.settings.tts.extra["ref_audio"]
    assert a.settings.tts.extra["ref_text"] == b.settings.tts.extra["ref_text"]
    # the embedded legacy references are gone
    assert a.settings.tts.extra["ref_audio"] not in (_b64(REF_CLARA), _b64(REF_JONAS))


def test_c1_exact_id_regression_for_four_sets(env):
    """Same resolved model/session.voice/ref_sha256 before and after grouping;
    grouping is presentation-only and does not feed resolution."""
    tc, store, _ = env
    _make_voice(tc, "voice-1", b"RIFF-voice-1", "Speaker")
    settings = tc.app.state.settings_store._settings
    # save the four sets switched on via the API, then compare resolution
    # (profile id, tts row, voice kind, voice id, native voice,
    #  expected request voice, expected display voice)
    cases = [
        ("clara-reference", "chatterbox-clara", "library", "voice-1", "", "clara-reference", "Speaker"),
        ("jonas-reference", "chatterbox-jonas", "library", "voice-1", "", "jonas-reference", "Speaker"),
        # this row has an empty stored voice -> the library display name is the
        # request value (existing fallback, unchanged)
        ("speaker-test", "tts-qwen3-base-clone", "library", "voice-1", "", "Speaker", "Speaker"),
        ("test-alt-de", "qwen-serena", "native", "", "serena", "serena", "serena"),
    ]
    sample_store = tc.app.state.sample_store
    for pid, tts, kind, vid, nv, want_voice, want_display in cases:
        body = {
            "id": pid, "display_name": pid, "enabled": True,
            "spec": {
                "stt_profile": "stt-qwen3-asr-0.6b", "llm_profile": "llm-gemma",
                "tts_profile": tts, "language": "de",
                "voice": {"kind": kind, "voice_id": vid, "native_voice": nv},
            },
        }
        created = tc.post("/api/conversation-profiles", headers=AUTH, json=body)
        assert created.status_code == 201, (pid, created.text)
        sel, route = resolve_profile(
            tc.app.state.settings_store._settings, pid, sample_store=sample_store
        )
        assert sel.settings.tts.model == settings.profiles["tts"][tts].model
        assert sel.settings.tts.language == "de"
        assert sel.settings.session.voice == want_voice, pid
        assert sel.display_voice == want_display, pid
        if kind == "library":
            # the library sample replaces the row's embedded legacy reference
            assert sel.settings.tts.extra.get("ref_audio")
            assert sel.settings.tts.extra["ref_audio"] not in (_b64(REF_CLARA), _b64(REF_JONAS))
        else:
            assert "ref_audio" not in sel.settings.tts.extra
        # grouping metadata is presentation only: never part of the set
        stored = tc.get(f"/api/conversation-profiles/{pid}", headers=AUTH).json()
        assert "engine_groups" not in stored
        assert "engine_key" not in stored


# ------------------------------------------------------------------- C3

def test_c3_display_names_library_voice_but_request_is_unchanged(env):
    tc, store, _ = env
    _make_voice(tc, "voice-1", b"RIFF-voice-1", "Speaker")
    # chatterbox-clara row voice label is "clara-reference"
    settings = tc.app.state.settings_store._settings
    sel = _resolve_row(settings, "chatterbox-clara", "library", voice_id="voice-1")
    # request value unchanged (the row's stored voice label)
    assert sel.settings.session.voice == "clara-reference"
    # display names the library voice
    assert sel.display_voice == "Speaker"
    info = sel.info()
    assert info["tts"]["voice"] == "clara-reference"
    assert info["tts"]["display_voice"] == "Speaker"


def test_c3_route_block_exposes_display_voice_additively(env):
    tc, _, _ = env
    _make_voice(tc, "voice-1", b"RIFF-voice-1", "Speaker")
    body = {
        "id": "speaker-test", "display_name": "Speaker",
        "enabled": True,
        "spec": {
            "stt_profile": "stt-qwen3-asr-0.6b", "llm_profile": "llm-gemma",
            "tts_profile": "chatterbox-clara", "language": "de",
            "voice": {"kind": "library", "voice_id": "voice-1"},
        },
    }
    created = tc.post("/api/conversation-profiles", headers=AUTH, json=body)
    assert created.status_code == 201, created.text

    settings = tc.app.state.settings_store._settings
    sel, route = resolve_profile(settings, "speaker-test", sample_store=tc.app.state.sample_store)
    # the engine request value is unchanged (the row's stored voice label)
    assert sel.settings.session.voice == "clara-reference"
    assert route["actual_voice"] == "clara-reference"
    # the display names the library voice, additively
    assert sel.display_voice == "Speaker"
    assert route["display_voice"] == "Speaker"