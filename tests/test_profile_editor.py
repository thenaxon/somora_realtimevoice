"""F3 server-side: incomplete-set test sessions & delete protections.

Regression tests for Alex's functional feedback round 1:
- POST …/test-session on an empty set once returned HTTP 500 (KeyError: '');
  it must return 409 set_incomplete with the issues.
- DELETE must be allowed for a set that is switched off, also one that was
  on before (N-2), and refused while it is on.

Synthetic audio only; capturing fake HTTP backend — never a real endpoint.
"""

import io
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
    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Hallo Welt"})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(SPEECH))
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "x"}, {"id": "tts-alt"}, {"id": "stt-alt"}, {"id": "llm-alt"}]})
        if path.endswith("/chat/completions"):
            return httpx.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"error": {"message": "nf"}})


def make_settings() -> Settings:
    s = Settings()
    for role in ("stt", "llm", "tts"):
        s.profiles[role]["alt"] = BackendProfile(id="alt", role=role, base_url="http://cap.test/v1", model=f"{role}-alt")
    s.profiles["tts"]["alt"].capabilities.voice_cloning = "reference_audio"
    s.voice_library.transcription_stt_profile = "alt"
    return s


@pytest.fixture
def env(tmp_path):
    client = httpx.AsyncClient(transport=httpx.MockTransport(Capture().handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        yield tc


def _confirmed_voice(tc, vid="anna"):
    tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": "Anna"})
    up = tc.post(
        f"/api/voices/{vid}/samples", headers=AUTH,
        files={"file": ("r.wav", wav_bytes(sine_pcm()), "audio/wav")}, data={"source": "upload"},
    )
    sid = up.json()["id"]
    tc.patch(f"/api/voices/{vid}/samples/{sid}", headers=AUTH, json={"transcript": "Hallo Welt", "transcript_confirmed": True})
    return sid


def _complete_set(tc, pid="f3", *, enabled=False):
    _confirmed_voice(tc)
    return tc.post("/api/conversation-profiles", headers=AUTH, json={
        "id": pid, "display_name": "F3", "enabled": enabled, "spec": {
            "stt_profile": "alt", "llm_profile": "alt", "tts_profile": "alt",
            "language": "de", "voice": {"kind": "library", "voice_id": "anna"},
        },
    })


def test_test_session_incomplete_set_returns_409(env):
    tc = env
    # empty set: previously raised KeyError('') -> HTTP 500
    assert tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "empty", "display_name": "E"}).status_code == 201
    r = tc.post("/api/conversation-profiles/empty/test-session", headers=AUTH, json={})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "set_incomplete"
    assert r.json().get("issues")


def test_test_session_complete_set_returns_ticket_without_switching_it_on(env):
    tc = env
    assert _complete_set(tc, "ok").status_code == 201
    r = tc.post("/api/conversation-profiles/ok/test-session", headers=AUTH, json={})
    assert r.status_code == 200, r.text
    assert r.json()["model"].startswith("profile-test:")
    assert tc.get("/api/conversation-profiles/ok", headers=AUTH).json()["enabled"] is False


def test_resolve_test_incomplete_set_has_clear_code(env):
    """A ticket created before the set was emptied must not 500 on connect."""
    from realtimevoice.config import Settings
    from realtimevoice.profiles import ProfileResolutionError, resolve_test
    from realtimevoice.config import ProfileSpec

    s = Settings()
    s.profiles["stt"]["alt"] = BackendProfile(id="alt", role="stt", base_url="http://cap.test/v1", model="m")
    s.profiles["llm"]["alt"] = BackendProfile(id="alt", role="llm", base_url="http://cap.test/v1", model="m")
    s.profiles["tts"]["alt"] = BackendProfile(id="alt", role="tts", base_url="http://cap.test/v1", model="m")
    from realtimevoice.config import ConversationProfile
    import time

    s.conversation_profiles["pp"] = ConversationProfile(
        id="pp", display_name="P", spec=ProfileSpec(), created_at=time.time(), updated_at=time.time()
    )
    with pytest.raises(ProfileResolutionError) as exc:
        resolve_test(s, "pp")
    assert exc.value.code == "set_incomplete"


def test_delete_a_set_switched_off_after_being_on_is_allowed(env):
    tc = env
    # switch a complete set on, then off: N-2 says delete is allowed
    assert _complete_set(tc, "del", enabled=True).status_code == 201
    r = tc.put("/api/conversation-profiles/del", headers=AUTH, json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False
    r = tc.delete("/api/conversation-profiles/del", headers=AUTH)
    assert r.status_code == 200, r.text
    assert tc.get("/api/conversation-profiles/del", headers=AUTH).status_code == 404


def test_delete_switched_on_set_still_blocked(env):
    tc = env
    assert _complete_set(tc, "avail", enabled=True).status_code == 201
    r = tc.delete("/api/conversation-profiles/avail", headers=AUTH)
    assert r.status_code == 409 and r.json()["code"] == "conversation_profile_active"