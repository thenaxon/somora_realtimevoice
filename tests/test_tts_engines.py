"""F1: TTS engine discovery and preview provenance.

Regression tests for operator feedback round 1. Each assertion would
have failed before the F1 fix:
- GET /api/tts-engines did not exist (404),
- preview did not return X-Rtv-Reference-Sha8 / X-Rtv-Voice-Id / X-Rtv-Sample-Id,
- legacy speaker-named profiles were indistinguishable from neutral engines.

Synthetic audio only; capturing fake HTTP backend — never a real endpoint.
"""

import base64
import io
import json
import math
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import BackendProfile, Settings, SettingsStore

AUTH = {"Authorization": "Bearer tok"}
SPEECH = b"\x01\x02" * 600


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
    def __init__(self):
        self.calls = []
        self.transcript = "Hallo Welt"

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append({"path": path, "content": request.read()})
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": self.transcript})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(SPEECH))
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "x"}, {"id": "tts-alt"}, {"id": "tts-legacy"}]})
        return httpx.Response(404, json={"error": {"message": "nf"}})


def make_settings() -> Settings:
    s = Settings()
    s.profiles["stt"]["alt"] = BackendProfile(id="alt", role="stt", base_url="http://cap.test/v1", model="stt-alt")
    s.profiles["llm"]["alt"] = BackendProfile(id="alt", role="llm", base_url="http://cap.test/v1", model="llm-alt")
    # neutral cloning engine
    s.profiles["tts"]["alt"] = BackendProfile(
        id="alt", role="tts", base_url="http://cap.test/v1", model="tts-alt",
        voice="alt-voice",
    )
    s.profiles["tts"]["alt"].capabilities.voice_cloning = "reference_audio"
    s.profiles["tts"]["alt"].extra = {"temperature": 0.5}
    # legacy speaker-named profile with an embedded reference
    s.profiles["tts"]["tts-legacy"] = BackendProfile(
        id="tts-legacy", role="tts", display_name="Chatterbox Jonas",
        base_url="http://cap.test/v1", model="tts-legacy",
        extra={"ref_audio": "LEGACYREF", "ref_text": "legacy text"},
    )
    s.profiles["tts"]["tts-legacy"].capabilities.voice_cloning = "reference_audio"
    # a non-cloning preset engine
    s.profiles["tts"]["preset"] = BackendProfile(
        id="preset", role="tts", display_name="Preset voices",
        base_url="http://cap.test/v1", model="tts-base",
        capabilities={"voices": ["ryan"], "voice_cloning": "none"},
    )
    # an engine whose capability was never verified
    s.profiles["tts"]["mystery"] = BackendProfile(
        id="mystery", role="tts", display_name="Mystery TTS",
        base_url="http://secret-host.example:9000/v1", model="mystery",
    )
    s.profiles["tts"]["mystery"].capabilities.voice_cloning = "unknown"
    return s


@pytest.fixture
def env(tmp_path):
    capture = Capture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        yield tc, capture, tmp_path


def _setup_voice(tc, vid="anna", confirmed=True):
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": "Anna"}).status_code == 201
    up = tc.post(
        f"/api/voices/{vid}/samples", headers=AUTH,
        files={"file": ("ref.wav", wav_bytes(sine_pcm()), "audio/wav")}, data={"source": "upload"},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    if confirmed:
        patched = tc.patch(
            f"/api/voices/{vid}/samples/{sid}", headers=AUTH,
            json={"transcript": "Hallo Welt", "transcript_confirmed": True},
        )
        assert patched.status_code == 200
    return sid


# ------------------------------------------------------------ engine discovery

def test_tts_engines_route_exists_and_labels_capabilities(env):
    tc, _, _ = env
    r = tc.get("/api/tts-engines", headers=AUTH)
    assert r.status_code == 200, r.text
    engines = {e["id"]: e for e in r.json()["engines"]}
    # every tts profile is offered, capabilities normalised
    assert engines["alt"]["voice_cloning"] == "reference_audio"
    assert engines["preset"]["voice_cloning"] == "none"
    assert engines["mystery"]["voice_cloning"] == "unknown"
    # neutral label only (never a hostname/port, never the full URL) and no secrets
    assert "host" not in engines["mystery"], "internal host must not be exposed"
    assert engines["mystery"]["backend_label"] == "Mystery TTS"
    assert "secret-host.example" not in json.dumps(r.json())
    # legacy embedded reference is flagged
    assert engines["tts-legacy"]["embeds_reference"] is True
    assert engines["alt"]["embeds_reference"] is False
    # model names are present for honest labelling
    assert engines["tts-legacy"]["model"] == "tts-legacy"


def test_tts_engines_requires_auth(env):
    tc, _, _ = env
    assert tc.get("/api/tts-engines").status_code == 401


# --------------------------------------------------------------- provenance

def test_preview_headers_carry_sample_provenance(env):
    tc, _, _ = env
    sid = _setup_voice(tc)
    r = tc.post("/api/voices/anna/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Rtv-Voice-Id"] == "anna"
    assert r.headers["X-Rtv-Sample-Id"] == sid
    assert r.headers["X-Rtv-Model"] == "tts-alt"
    assert r.headers["X-Rtv-Profile-Id"] == "alt"
    # sha8 is the sha of the user's stored sample
    voice = tc.get("/api/voices/anna", headers=AUTH).json()
    sample = next(s for s in voice["samples"] if s["id"] == sid)
    assert r.headers["X-Rtv-Reference-Sha8"] == sample["sha256"][:8]


def test_preview_shows_sample_sha_even_when_profile_embeds_reference(env):
    """The legacy speaker-named profile carries extra.ref_audio/ref_text; the
    preview must materialise the USER's sample and report ITS sha8 — proving
    the embedded legacy reference was overridden."""
    tc, cap, _ = env
    sid = _setup_voice(tc)
    r = tc.post("/api/voices/anna/preview", headers=AUTH, json={"tts_profile": "tts-legacy", "text": "Hallo"})
    assert r.status_code == 200, r.text
    voice = tc.get("/api/voices/anna", headers=AUTH).json()
    sample = next(s for s in voice["samples"] if s["id"] == sid)
    assert r.headers["X-Rtv-Reference-Sha8"] == sample["sha256"][:8]
    # the effective legacy fingerprint is NOT the sample's; the header proves
    # which one the backend actually received
    assert r.headers["X-Rtv-Reference-Sha8"] != "LEGACYREF"[:8]
    speech = [c for c in cap.calls if c["path"].endswith("/audio/speech")][-1]
    payload = json.loads(speech["content"])
    assert base64.b64decode(payload["ref_audio"])[:4] == b"RIFF"
    assert payload["ref_text"] == "Hallo Welt"  # user's transcript, not "legacy text"


def test_preview_headers_and_engine_view_never_expose_internal_host(env):
    """P2-1: internal hostnames/ports must not leave the server in preview
    headers or in the /api/tts-engines picker. The full URL stays available in
    the Backends catalog view only (where the edit form needs it)."""
    tc, cap, _ = env
    _setup_voice(tc)
    engines = tc.get("/api/tts-engines", headers=AUTH).json()
    assert "secret-host.example" not in json.dumps(engines)
    assert all("host" not in e for e in engines["engines"])
    # every engine has the neutral backend label instead
    assert all(e.get("backend_label") for e in engines["engines"])
    r = tc.post("/api/voices/anna/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 200, r.text
    header_blob = json.dumps(dict(r.headers))
    assert "cap.test" not in header_blob and "secret-host.example" not in header_blob
    # the Backends catalog DOES carry the full base URL for its edit form
    cat = tc.get("/api/catalog", headers=AUTH).json()
    tts_profiles = {p["id"]: p for p in cat["roles"]["tts"]["profiles"]}
    assert tts_profiles["mystery"]["base_url"] == "http://secret-host.example:9000/v1"
