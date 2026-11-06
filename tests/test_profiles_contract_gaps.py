"""A6: contract §10 gaps not already covered by the module-specific suites.

- save atomicity (a refused save leaves the stored settings unchanged),
- pending route => zero STT calls (not just zero LLM/TTS),
- explicit `voice-route:<set>` resolves that set at connect,
- transcription timeout keeps audio and reports backend_timeout,
- delete protection for a voice a set speaks with; a voice kept only by an
  earlier version is not protected, and restoring that version is refused.
"""

import asyncio
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
from realtimevoice.backends.base import Backends
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.config import BackendProfile, Settings, SettingsStore, Voice
from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample

from test_profiles_realtime import Capture, connect, initial_update, recv_until, sine_pcm, _wav


def base_settings(tmp_path, *, make_profile=True):
    s = Settings()
    s.profiles["stt"]["stt-a"] = BackendProfile(id="stt-a", role="stt", base_url="http://cap.test/v1", model="stt-a")
    s.profiles["llm"]["llm-a"] = BackendProfile(id="llm-a", role="llm", base_url="http://cap.test/v1", model="llm-a")
    s.profiles["tts"]["tts-a"] = BackendProfile(id="tts-a", role="tts", base_url="http://cap.test/v1",
                                                model="chatterbox", voice="v")
    s.profiles["tts"]["tts-a"].capabilities.voice_cloning = "reference_audio"
    store = SampleStore(str(tmp_path))
    c = canonicalise_wav(_wav(sine_pcm(1500)))
    store.write("anna", c.sha256, c.data)
    smp = new_sample(c, source="upload", created_at=0.0)
    smp.transcript = "hallo"
    smp.transcript_confirmed = True
    s.voices["anna"] = Voice(id="anna", display_name="Anna", language="de", samples=[smp],
                             reference_sample_id=smp.id, created_at=0.0, updated_at=0.0)
    if make_profile:
        from realtimevoice.config import ConversationProfile, ProfileSpec, VoiceSelection

        spec = ProfileSpec(stt_profile="stt-a", llm_profile="llm-a", tts_profile="tts-a", language="de",
                           voice=VoiceSelection(kind="library", voice_id="anna"))
        s.conversation_profiles["anna-agent"] = ConversationProfile(
            id="anna-agent", display_name="Anna", spec=spec, enabled=True, created_at=0.0, updated_at=0.0,
        )
    return s


def make_client(tmp_path, settings, cap, *, fail_models=False):
    client = httpx.AsyncClient(transport=httpx.MockTransport(cap.handler))

    def factory(st, http_client=None):
        stt, llm, tts = build_backends(st.stt, st.llm, st.tts, client)
        return Backends(stt=stt, llm=llm, tts=tts)

    app = create_app(settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
                     backend_factory=factory, auth_token="tok")
    return app, client


def _stored(tc):
    return tc.app.state.settings_store._settings.model_dump()  # noqa: SLF001 (test)


def test_a_refused_save_leaves_settings_unchanged(tmp_path):
    s = base_settings(tmp_path)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        h = {"Authorization": "Bearer tok"}
        before = _stored(tc)
        # a switched-on set saved with a model that does not exist
        r = tc.put("/api/conversation-profiles/anna-agent", headers=h,
                   json={"display_name": "Renamed", "spec": {"stt_profile": "nope", "llm_profile": "llm-a",
                                                             "tts_profile": "tts-a", "language": "de"}})
        assert r.status_code == 409 and r.json()["code"] == "set_invalid"
        # a model edit that would stop the switched-on set
        r = tc.put("/api/profiles/tts/tts-a", headers=h, json={"capabilities": {"voice_cloning": "none"}})
        assert r.status_code == 409 and r.json()["code"] == "breaks_sets"
        # nothing of either request was stored, not even the display name
        assert _stored(tc) == before


def test_a_failing_probe_does_not_block_and_is_reported(tmp_path):
    s = base_settings(tmp_path)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    orig = cap.handler

    def failing(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(500, json={"error": {"message": "no"}})
        return orig(request)

    with TestClient(app) as tc:
        tc.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
        h = {"Authorization": "Bearer tok"}
        r = tc.put("/api/conversation-profiles/anna-agent", headers=h,
                   json={"spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-a",
                                  "language": "en", "voice": {"kind": "library", "voice_id": "anna"}}})
        assert r.status_code == 200, r.text
        assert r.json()["spec"]["language"] == "en"
        assert {role: p["ok"] for role, p in r.json()["probes"].items()} == {"stt": False, "llm": False, "tts": False}


def test_pending_route_makes_zero_stt_calls(tmp_path):
    s = base_settings(tmp_path, make_profile=False)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        with connect(tc, "voice-route:auto") as ws:
            ws.receive_json()
            pcm = sine_pcm(400)
            silence = b"\x00\x00" * 480 * 8
            for i in range(0, len(pcm), 9600):
                ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + 9600]).decode()})
            for _ in range(8):
                ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()})
            # drain the one pending error
            err = recv_until(ws, "error")
            assert err["error"]["code"] == "voice_route_pending"
    assert cap.of("/audio/transcriptions") == []  # zero STT calls


def test_explicit_voice_route_profile_resolves_at_connect(tmp_path):
    s = base_settings(tmp_path)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        with connect(tc, "voice-route:anna-agent") as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            assert created["session"]["x_realtimevoice"]["route"]["profile"] == "anna-agent"


def test_transcribe_timeout_keeps_audio_and_reports_timeout(tmp_path):
    s = base_settings(tmp_path)
    cap = Capture()
    # make the STT request hang beyond the profile timeout
    s.profiles["stt"]["stt-a"].request_timeout_s = 1.0

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json={"text": "late"})

    async def handler(request):
        if request.url.path.endswith("/audio/transcriptions"):
            return await slow(request)
        return cap.handler(request)

    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        h = {"Authorization": "Bearer tok"}
        # upload a sample
        up = tc.post("/api/voices/anna/samples", headers=h,
                     files={"file": ("r.wav", _wav(sine_pcm(1500)), "audio/wav")}, data={"source": "upload"})
        sid = up.json()["id"]
        r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=h, json={"stt_profile": "stt-a"})
        assert r.status_code == 200
        assert r.json()["transcription"]["state"] == "failed"
        assert r.json()["transcription"]["error_code"] == "backend_timeout"
        # audio kept
        assert tc.get(f"/api/voices/anna/samples/{sid}/audio", headers=h).status_code == 200


def test_a_voice_a_set_speaks_with_cannot_be_deleted(tmp_path):
    s = base_settings(tmp_path)
    s.conversation_profiles["anna-agent"].enabled = False
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        # also a switched-off set: it would silently lose its voice
        r = tc.delete("/api/voices/anna", headers={"Authorization": "Bearer tok"})
        assert r.status_code == 409
        assert r.json()["code"] == "voice_referenced"


def test_a_voice_kept_only_by_an_earlier_version_can_be_deleted(tmp_path):
    """Versions are not protected (no drop_history any more): the voice can
    go, and restoring the version that spoke with it is then refused for a
    switched-on set instead of breaking its calls."""
    s = base_settings(tmp_path)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        h = {"Authorization": "Bearer tok"}
        # the set now speaks with a built-in voice -> anna is history-only
        r = tc.put("/api/conversation-profiles/anna-agent", headers=h,
                   json={"spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-a",
                                  "language": "de", "voice": {"kind": "native", "native_voice": "ryan"}}})
        assert r.status_code == 200, r.text
        assert r.json()["history"][0]["spec"]["voice"]["voice_id"] == "anna"
        r = tc.delete("/api/voices/anna", headers=h)
        assert r.status_code == 200, r.text
        r = tc.post("/api/conversation-profiles/anna-agent/restore", headers=h, json={"index": 0})
        assert r.status_code == 409 and r.json()["code"] == "set_invalid"
        assert tc.get("/api/conversation-profiles/anna-agent", headers=h).json()["spec"]["voice"]["kind"] == "native"


def test_root_redirects_to_ui_and_the_old_page_is_gone(tmp_path):
    """`/` leads to the management UI; the old Talk page is gone."""
    s = base_settings(tmp_path, make_profile=False)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)
    with TestClient(app) as tc:
        r = tc.get("/", follow_redirects=False)
        assert r.status_code in (307, 308)
        assert r.headers["location"].endswith("/ui/")
        assert tc.get("/legacy.html").status_code == 404
        # the SPA itself remains reachable and is not shadowed by the redirect
        assert tc.get("/ui/index.html").status_code == 200

def _during_probes(monkeypatch, action):
    """Run `action(settings)` as a concurrent save while a set save awaits its
    model probes."""
    import realtimevoice.routes_profiles as rp

    real = rp._probe_spec

    async def probe_then_concurrent_change(request, settings, spec):
        await request.app.state.settings_store.mutate(action)
        return await real(request, settings, spec)

    monkeypatch.setattr(rp, "_probe_spec", probe_then_concurrent_change)


def test_a_rename_during_a_switch_off_does_not_switch_the_set_back_on(tmp_path, monkeypatch):
    s = base_settings(tmp_path)
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)

    def switch_off(settings):
        settings.conversation_profiles["anna-agent"].enabled = False

    _during_probes(monkeypatch, switch_off)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        r = tc.put("/api/conversation-profiles/anna-agent", headers={"Authorization": "Bearer tok"},
                   json={"display_name": "Anna 2"})
        assert r.status_code == 200, r.text
        assert r.json()["display_name"] == "Anna 2"
        assert r.json()["enabled"] is False


def test_switching_on_is_rechecked_against_changes_made_meanwhile(tmp_path, monkeypatch):
    s = base_settings(tmp_path)
    s.conversation_profiles["anna-agent"].enabled = False
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)

    def unconfirm(settings):
        settings.voices["anna"].samples[0].transcript_confirmed = False

    _during_probes(monkeypatch, unconfirm)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        r = tc.put("/api/conversation-profiles/anna-agent", headers={"Authorization": "Bearer tok"},
                   json={"enabled": True})
        assert r.status_code == 409 and r.json()["code"] == "set_invalid"
        assert tc.app.state.settings_store._settings.conversation_profiles["anna-agent"].enabled is False  # noqa: SLF001


def test_a_restore_never_applies_another_version_than_the_one_chosen(tmp_path, monkeypatch):
    from realtimevoice.config import SetVersion

    s = base_settings(tmp_path)
    p = s.conversation_profiles["anna-agent"]
    old = p.spec.model_copy(update={"language": "en"})
    p.history = [SetVersion(saved_at=1.0, spec=old)]
    cap = Capture()
    app, client = make_client(tmp_path, s, cap)

    def save_meanwhile(settings):
        cp = settings.conversation_profiles["anna-agent"]
        cp.history.insert(0, SetVersion(saved_at=2.0, spec=cp.spec))
        cp.spec = cp.spec.model_copy(update={"language": "fr"})

    _during_probes(monkeypatch, save_meanwhile)
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        r = tc.post("/api/conversation-profiles/anna-agent/restore", headers={"Authorization": "Bearer tok"},
                    json={"index": 0})
        assert r.status_code == 409 and r.json()["code"] == "history_changed"
        assert tc.app.state.settings_store._settings.conversation_profiles["anna-agent"].spec.language == "fr"  # noqa: SLF001
