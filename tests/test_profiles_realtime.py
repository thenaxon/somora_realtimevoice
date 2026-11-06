"""A4: realtime endpoint resolution — `profile:`, `profile-test:`,
`voice-route:auto`/`voice-route:<set>`, the set for other apps, pending
guards, `/v1/models`; a saved set reaches the next call while a running call
keeps its copy.

Two PARALLEL websocket sessions against capturing fake HTTP backends prove
per-session STT/LLM/TTS model + reference sha and no cross-talk (contract §6,
§7, §10). All audio synthetic.
"""

import base64
import json
import math

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.backends.base import Backends
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.config import (
    BackendProfile,
    ConversationProfile,
    ProfileSpec,
    Settings,
    SettingsStore,
    Voice,
    VoiceSelection,
    reference_fingerprint,
)
from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample

REF_A = b"RIFFsynthetic-ref-anna"
REF_B = b"RIFFsynthetic-ref-bert"


def sine_pcm(ms=1500, rate=24000):
    n = int(rate * ms / 1000)
    return (6000 * np.sin(2 * math.pi * 300 * np.arange(n) / rate)).astype("<i2").tobytes()


def make_settings(tmp_path):
    s = Settings()
    # two TTS profiles with distinct models and reference material
    s.profiles["tts"]["tts-anna"] = BackendProfile(
        id="tts-anna", role="tts", base_url="http://cap.test/v1", model="chatterbox-anna",
        voice="anna-voice",
        extra={"ref_audio": base64.b64encode(REF_A).decode(), "ref_text": "anna alt"},
    )
    s.profiles["tts"]["tts-anna"].capabilities.voice_cloning = "reference_audio"
    s.profiles["tts"]["tts-bert"] = BackendProfile(
        id="tts-bert", role="tts", base_url="http://cap.test/v1", model="chatterbox-bert",
        voice="bert-voice",
        extra={"ref_audio": base64.b64encode(REF_B).decode(), "ref_text": "bert alt"},
    )
    s.profiles["tts"]["tts-bert"].capabilities.voice_cloning = "reference_audio"
    s.profiles["stt"]["stt-a"] = BackendProfile(id="stt-a", role="stt", base_url="http://cap.test/v1", model="stt-a-model")
    s.profiles["llm"]["llm-a"] = BackendProfile(id="llm-a", role="llm", base_url="http://cap.test/v1", model="llm-a-model")
    s.profiles["stt"]["stt-b"] = BackendProfile(id="stt-b", role="stt", base_url="http://cap.test/v1", model="stt-b-model")
    s.profiles["llm"]["llm-b"] = BackendProfile(id="llm-b", role="llm", base_url="http://cap.test/v1", model="llm-b-model")

    store = SampleStore(str(tmp_path))
    for vid, ref in (("anna", REF_A), ("bert", REF_B)):
        # the stored WAV must be a real WAV for materialisation
        c = canonicalise_wav(_wav(sine_pcm(1500)))
        store.write(vid, c.sha256, c.data)
        smp = new_sample(c, source="upload", created_at=0.0)
        smp.transcript = f"{vid} text"
        smp.transcript_confirmed = True
        s.voices[vid] = Voice(id=vid, display_name=vid.capitalize(), language="de",
                              samples=[smp], reference_sample_id=smp.id, created_at=0.0, updated_at=0.0)

    for pid, v, tts, stt, llm in (
        ("anna-agent", "anna", "tts-anna", "stt-a", "llm-a"),
        ("bert-agent", "bert", "tts-bert", "stt-b", "llm-b"),
    ):
        spec = ProfileSpec(stt_profile=stt, llm_profile=llm, tts_profile=tts, language="de",
                           voice=VoiceSelection(kind="library", voice_id=v))
        s.conversation_profiles[pid] = ConversationProfile(
            id=pid, display_name=pid, spec=spec, enabled=True, created_at=0.0, updated_at=0.0,
        )
    return s


def _wav(pcm):
    import io, wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return buf.getvalue()


class Capture:
    def __init__(self):
        self.calls = []
        self._lock = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload = json.loads(request.content)
            except ValueError:
                payload = {}
        self.calls.append({"method": request.method, "path": request.url.path, "payload": payload,
                           "body": request.read()})
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "hallo"})
        if request.url.path.endswith("/chat/completions"):
            sse = (
                'data: {"choices":[{"delta":{"content":"Kurzer Satz."}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=_wav(sine_pcm(300)))
        if request.url.path.endswith("/models"):
            # P1: readiness probes require the configured model id to be listed.
            # Advertise every model the profile fixtures across the suites use.
            return httpx.Response(200, json={"data": [
                {"id": "m"},
                {"id": "stt-a-model"}, {"id": "llm-a-model"},
                {"id": "stt-b-model"}, {"id": "llm-b-model"},
                {"id": "chatterbox-anna"}, {"id": "chatterbox-bert"},
                {"id": "stt-a"}, {"id": "llm-a"}, {"id": "chatterbox"},
            ]})
        return httpx.Response(404, json={"error": {"message": "nf"}})

    def of(self, suffix):
        return [c for c in self.calls if c["path"].endswith(suffix)]


@pytest.fixture
def env(tmp_path):
    cap = Capture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(cap.handler))

    def factory(settings, http_client=None):
        stt, llm, tts = build_backends(settings.stt, settings.llm, settings.tts, client)
        return Backends(stt=stt, llm=llm, tts=tts)

    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=make_settings(tmp_path)),
        backend_factory=factory,
        auth_token="tok",
    )
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        yield tc, cap, tmp_path


def connect(client, model):
    return client.websocket_connect(f"/v1/realtime?model={model}", subprotocols=["rtv.bearer", "rtv.bearer.tok"])


def recv_until(ws, *types):
    while True:
        msg = ws.receive_json()
        if msg.get("type") in types:
            return msg


def initial_update(voice):
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "Sei kurz.",
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000},
                          "transcription": {"model": "whisper-1", "language": "de"}},
                "output": {"voice": voice},
            },
        },
    }


def run_response(ws):
    ws.send_json({"type": "response.create"})
    while True:
        msg = ws.receive_json()
        if msg["type"] == "response.done":
            return msg["response"]["status"]


# --------------------------------------------------- two parallel sessions

def _speech_turn(ws, seconds=1.5):
    """Push a speech burst followed by silence to trigger VAD STT."""
    pcm = sine_pcm(int(seconds * 1000))
    silence = b"\x00\x00" * 480 * 6
    for i in range(0, len(pcm), 9600):
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + 9600]).decode()})
    for _ in range(8):
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()})
    while True:
        msg = ws.receive_json()
        if msg["type"] == "response.done":
            return


def test_two_parallel_sessions_no_cross_talk(env):
    tc, cap, tmp = env
    with connect(tc, "voice-route:auto") as ws1, connect(tc, "profile:bert-agent") as ws2:
        ws1.receive_json()
        ws2.receive_json()
        # session 1 resolves the set named by its voice
        ws1.send_json(initial_update("anna-agent"))
        upd = recv_until(ws1, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        route = upd["session"]["x_realtimevoice"]["route"]
        assert route["profile"] == "anna-agent" and route["mode"] == "profile"
        # session 2 is pre-resolved
        assert ws2.receive_json  # noop
        _speech_turn(ws1)
        _speech_turn(ws2)
    # per-session model assertions
    stt_models = [c["body"].decode("latin1") for c in cap.of("/audio/transcriptions")]
    assert any("stt-a-model" in b for b in stt_models)
    assert any("stt-b-model" in b for b in stt_models)
    llm_models = {c["payload"]["model"] for c in cap.of("/chat/completions")}
    assert {"llm-a-model", "llm-b-model"} <= llm_models
    tts_calls = cap.of("/audio/speech")
    models = {c["payload"]["model"] for c in tts_calls}
    assert models == {"chatterbox-anna", "chatterbox-bert"}
    # each TTS request references the right stored WAV (sha of the stored bytes)
    import hashlib

    by_model = {c["payload"]["model"]: c["payload"] for c in tts_calls}
    anna_ref = base64.b64decode(by_model["chatterbox-anna"]["ref_audio"])
    bert_ref = base64.b64decode(by_model["chatterbox-bert"]["ref_audio"])
    assert hashlib.sha256(anna_ref) != hashlib.sha256(bert_ref)
    assert by_model["chatterbox-anna"]["ref_text"] == "anna text"
    assert by_model["chatterbox-bert"]["ref_text"] == "bert text"
    assert "anna alt" not in by_model["chatterbox-anna"]["ref_text"]


def test_profile_route_frozen_voice(env):
    tc, cap, _ = env
    with connect(tc, "profile:anna-agent") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        # client tries to change the voice: ignored (profile wins)
        ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "someone-else"}}}})
        msg = recv_until(ws, "session.updated", "error")
        assert msg["type"] in ("session.updated", "error")
        # drain any error then run a response and verify actual voice
        status = run_response(ws)
        assert status == "completed"
    assert cap.of("/audio/speech")[-1]["payload"]["voice"] == "anna-voice"


# --------------------------------------------------------- pending guards

def test_pending_audio_discarded_and_response_refused(env):
    tc, cap, _ = env
    with connect(tc, "voice-route:auto") as ws:
        ws.receive_json()
        # audio before the route: discarded, no STT (one pending error emitted)
        pcm = sine_pcm(300)
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        pending = recv_until(ws, "error")
        assert pending["error"]["code"] == "voice_route_pending"
        # response.create: error + no LLM, every time
        for _ in range(2):
            ws.send_json({"type": "response.create"})
            err = recv_until(ws, "error")
            assert err["error"]["code"] == "voice_route_pending"
        # now resolve the route and confirm the same connection works
        ws.send_json(initial_update("anna-agent"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        _speech_turn(ws)
    assert cap.of("/chat/completions")  # works after resolution
    assert len(cap.of("/audio/speech")) >= 1


def test_unknown_voice_route_keeps_pending(env):
    tc, cap, _ = env
    with connect(tc, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(initial_update("nobody"))
        err = recv_until(ws, "error")
        assert err["error"]["code"] == "unknown_voice_route"
        # still pending: response refused, no LLM
        ws.send_json({"type": "response.create"})
        err2 = recv_until(ws, "error")
        assert err2["error"]["code"] == "voice_route_pending"
    assert cap.of("/chat/completions") == []
    assert cap.of("/audio/speech") == []


def test_language_overridden_warning_only(env):
    tc, cap, _ = env
    with connect(tc, "voice-route:auto") as ws:
        ws.receive_json()
        body = initial_update("anna-agent")
        body["session"]["language"] = "en"
        body["session"]["audio"]["input"]["transcription"]["language"] = "en"
        ws.send_json(body)
        upd = recv_until(ws, "session.updated", "error")
        # never a non-invalid_request_error error event
        assert upd["type"] == "session.updated"
        assert "language_overridden_by_profile" in upd["session"]["x_realtimevoice"]["warnings"]
        assert upd["session"]["language"] == "de"


# ------------------------------------------------------- live session snapshot

def _bert_models_for_anna():
    return {"stt_profile": "stt-b", "llm_profile": "llm-b", "tts_profile": "tts-bert",
            "language": "de", "voice": {"kind": "library", "voice_id": "anna"}}


def test_saving_a_set_during_a_call_reaches_only_the_next_call(env):
    tc, cap, tmp = env
    h = {"Authorization": "Bearer tok"}
    with connect(tc, "profile:anna-agent") as ws:
        ws.receive_json()
        # the set is saved with other models while the call runs
        r = tc.put("/api/conversation-profiles/anna-agent", headers=h, json={"spec": _bert_models_for_anna()})
        assert r.status_code == 200, r.text
        # the running call keeps the models it started with
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-anna"
    assert "stt-a-model" in cap.of("/audio/transcriptions")[-1]["body"].decode("latin1")
    assert cap.of("/chat/completions")[-1]["payload"]["model"] == "llm-a-model"
    # the next call uses what was saved, with no further step
    with connect(tc, "profile:anna-agent") as ws:
        created = ws.receive_json()
        assert created["session"]["x_realtimevoice"]["route"]["tts_model"] == "chatterbox-bert"
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-bert"
    assert "stt-b-model" in cap.of("/audio/transcriptions")[-1]["body"].decode("latin1")


def test_a_changed_model_entry_reaches_the_next_call(env):
    tc, cap, tmp = env
    h = {"Authorization": "Bearer tok"}
    with connect(tc, "profile:anna-agent") as ws:
        ws.receive_json()
        r = tc.put("/api/profiles/tts/tts-anna", headers=h, json={"model": "chatterbox-bert"})
        assert r.status_code == 200, r.text
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-anna"
    with connect(tc, "profile:anna-agent") as ws:
        ws.receive_json()
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-bert"


def test_a_switched_off_set_is_refused_at_connect(env):
    from starlette.testclient import WebSocketDenialResponse

    tc, cap, _ = env
    r = tc.put("/api/conversation-profiles/bert-agent", headers={"Authorization": "Bearer tok"},
               json={"enabled": False})
    assert r.status_code == 200, r.text
    with pytest.raises(WebSocketDenialResponse) as denied:
        with connect(tc, "profile:bert-agent"):
            pass
    assert denied.value.status_code == 404
    assert denied.value.json()["error"]["code"] == "profile_not_available"


# ---------------------------------------------------------------- /v1/models

def test_models_listing(env):
    tc, cap, _ = env
    h = {"Authorization": "Bearer tok"}
    tc.put("/api/conversation-profiles/bert-agent", headers=h, json={"enabled": False})
    tc.put("/api/settings", headers=h, json={"default_set": "anna-agent"})
    r = tc.get("/v1/models", headers=h)
    assert r.status_code == 200
    data = r.json()["data"]
    ids = [m["id"] for m in data]
    # only switched-on sets are offered
    assert ids == ["profile:anna-agent", "voice-route:auto"]
    anna = data[0]
    assert anna["x_realtimevoice"] == {"display_name": "anna-agent", "language": "de", "default": True}
    # metadata only: no urls/keys/paths
    flat = json.dumps(data)
    assert "http" not in flat and "api_key" not in flat and "/home" not in flat
    assert tc.get("/v1/models").status_code == 401


# ------------------------------------------------------------- test tickets

def test_profile_test_ticket_and_expiry(env):
    tc, cap, tmp = env
    h = {"Authorization": "Bearer tok"}
    # a ticket without a spec tests the saved set, also a switched-off one
    assert tc.put("/api/conversation-profiles/anna-agent", headers=h, json={"enabled": False}).status_code == 200
    r = tc.post("/api/conversation-profiles/anna-agent/test-session", headers=h, json={})
    assert r.status_code == 200, r.text
    with connect(tc, r.json()["model"]) as ws:
        ws.receive_json()
        _speech_turn(ws)
    assert "stt-a-model" in cap.of("/audio/transcriptions")[-1]["body"].decode("latin1")
    # a ticket with the editor's unsaved spec tests that
    r = tc.post("/api/conversation-profiles/anna-agent/test-session", headers=h,
                json={"spec": _bert_models_for_anna()})
    assert r.status_code == 200
    model = r.json()["model"]
    assert model.startswith("profile-test:")
    with connect(tc, model) as ws:
        ws.receive_json()
        _speech_turn(ws)
    assert "stt-b-model" in cap.of("/audio/transcriptions")[-1]["body"].decode("latin1")
    # expired ticket -> refused
    ticket = model.split(":", 1)[1]
    tests = tc.app.state.profile_tests
    tests._tickets[ticket] = (tests._tickets[ticket][0], 0.0)  # noqa: SLF001 (test)
    with pytest.raises(Exception):
        with connect(tc, model) as ws:
            ws.receive_json()


def test_test_ticket_route_reports_the_unsaved_selection_over_websocket(env):
    """F10: the ticket must test the UNSAVED editor selection all the way
    through the real realtime socket, not just at ticket creation. The saved
    set uses anna/tts-anna; the ticket carries an unsaved spec that switches
    the speaking model to tts-bert. The resolved route (session.created) and
    the actual TTS request must both name the unsaved model, while the saved
    set stays untouched."""
    tc, cap, _ = env
    unsaved = {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-bert",
               "language": "de", "voice": {"kind": "library", "voice_id": "anna"}}
    r = tc.post("/api/conversation-profiles/anna-agent/test-session",
                headers={"Authorization": "Bearer tok"},
                json={"spec": unsaved})
    assert r.status_code == 200, r.text
    model = r.json()["model"]
    with connect(tc, model) as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        route = created["session"]["x_realtimevoice"]["route"]
        assert route["tts_model"] == "chatterbox-bert", route
        assert route["voice"]["kind"] == "library" and route["voice"]["voice_id"] == "anna"
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-bert"
    # nothing was persisted: the saved set still names tts-anna
    saved_after = tc.get("/api/conversation-profiles/anna-agent",
                         headers={"Authorization": "Bearer tok"}).json()["spec"]
    assert saved_after["tts_profile"] == "tts-anna"


def test_auto_route_takes_the_profiles_pacing_and_turn_detection(tmp_path):
    """A voice-route:auto session (how voice clients connect) gets the routed
    set's output pacing and turn-detection defaults, with the client's own
    turn-detection values on top — as a profile:<id> session does."""
    settings = make_settings(tmp_path)
    prof = settings.conversation_profiles["anna-agent"]
    prof.spec = prof.spec.model_copy(
        update={"output_pacing": "realtime", "turn_detection_default": {"silence_duration_ms": 700}}
    )
    cap = Capture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(cap.handler))

    def factory(s, http_client=None):
        stt, llm, tts = build_backends(s.stt, s.llm, s.tts, client)
        return Backends(stt=stt, llm=llm, tts=tts)

    app = create_app(settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings), backend_factory=factory, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        with connect(tc, "voice-route:auto") as ws:
            ws.receive_json()
            update = initial_update("anna-agent")
            update["session"]["audio"]["input"]["turn_detection"] = {"type": "server_vad", "threshold": 0.4}
            ws.send_json(update)
            upd = recv_until(ws, "session.updated", "error")
            assert upd["type"] == "session.updated", upd
            session = upd["session"]
            assert session["x_realtimevoice"]["output_pacing"] == "realtime"
            td = session["audio"]["input"]["turn_detection"]
            assert td["silence_duration_ms"] == 700
            assert td["threshold"] == 0.4


def test_refused_connections_say_why_with_an_http_status(env):
    """A refused connection is answered with an HTTP status and an error
    body, not a bare close before accept (which clients only see as 403)."""
    from starlette.testclient import WebSocketDenialResponse

    tc, _, _ = env
    with pytest.raises(WebSocketDenialResponse) as bad_token:
        with tc.websocket_connect("/v1/realtime?model=profile:anna-agent", subprotocols=["rtv.bearer", "rtv.bearer.wrong"]):
            pass
    assert bad_token.value.status_code == 401
    assert bad_token.value.json()["error"]["code"] == "invalid_token"
    with pytest.raises(WebSocketDenialResponse) as unknown:
        with connect(tc, "profile:no-such-profile"):
            pass
    assert unknown.value.status_code == 404
    assert unknown.value.json()["error"]["message"]


def test_a_client_naming_no_set_gets_the_set_for_other_apps_or_a_clear_refusal(env):
    """A model name that names no set (another app's own model name, or a
    typo) is never silently served with some backends: without a set for
    other apps it is refused with `no_set_selected`; with one, that set
    answers."""
    from starlette.testclient import WebSocketDenialResponse

    tc, cap, _ = env
    with pytest.raises(WebSocketDenialResponse) as refused:
        with connect(tc, "voice-rout:auto"):  # a typo
            pass
    assert refused.value.status_code == 404
    assert refused.value.json()["error"]["code"] == "no_set_selected"

    r = tc.put("/api/settings", headers={"Authorization": "Bearer tok"}, json={"default_set": "bert-agent"})
    assert r.status_code == 200, r.text
    with connect(tc, "gpt-realtime") as ws:
        created = ws.receive_json()
        route = created["session"]["x_realtimevoice"]["route"]
        assert route["profile"] == "bert-agent" and route["requested"] == "gpt-realtime"
        _speech_turn(ws)
    assert cap.of("/audio/speech")[-1]["payload"]["model"] == "chatterbox-bert"
