"""R2 / CP4: set language semantics and client/API precedence.

The set's language wins in ALL calls bound to a set (`voice-route:auto` after
resolution, `voice-route:<set>`, `profile:<id>`, `profile-test:<ticket>`, and a
call naming no set, which gets the default set), including repeated LATER
updates. A different client language (top-level `language`,
`input_audio_transcription.language`, `audio.input.transcription.language`) is
ignored for the session and reported with the warning
`language_overridden_by_profile` (never an error event).

Each model is sent the set's language in its own `language_format`: "code"
("de"), "name" ("German") or "none" (not sent); a region ("de-AT") is sent as
its base language.
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
from realtimevoice.backends.base import Backends
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.config import Settings, SettingsStore
from set_helpers import add_set


def sine_pcm(ms=1500, rate=24000):
    n = int(rate * ms / 1000)
    return (6000 * np.sin(2 * math.pi * 300 * np.arange(n) / rate)).astype("<i2").tobytes()


def _wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return buf.getvalue()


class Capture:
    """Records adapter requests; answers like a tiny OpenAI-compatible box."""

    def __init__(self):
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload = json.loads(request.content)
            except ValueError:
                payload = {}
        self.calls.append({"path": request.url.path, "payload": payload, "body": request.read()})
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
        return httpx.Response(404, json={"error": {"message": "nf"}})

    def of(self, suffix):
        return [c for c in self.calls if c["path"].endswith(suffix)]


def make_settings():
    s = Settings()
    s.session.language = "en"  # the server's call default; a set's language wins
    add_set(s, "anna", language="de", base_url="http://cap.test/v1", native_voice="anna-voice")
    # a default set whose models want the language by name / not at all
    add_set(
        s, "bert", language="de-AT", base_url="http://cap.test/v1", native_voice="bert-voice",
        language_formats={"stt": "name", "tts": "none"}, make_default=True,
    )
    return s


def _app(tmp_path, settings, cap):
    client = httpx.AsyncClient(transport=httpx.MockTransport(cap.handler))

    def factory(settings, http_client=None):
        stt, llm, tts = build_backends(settings.stt, settings.llm, settings.tts, client)
        return Backends(stt=stt, llm=llm, tts=tts)

    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
        backend_factory=factory,
        auth_token="tok",
    )
    return app, client


@pytest.fixture
def env(tmp_path):
    cap = Capture()
    app, client = _app(tmp_path, make_settings(), cap)
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
    return {"type": "session.update", "session": {"voice": voice}}


def speech_turn(ws, seconds=1.5):
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


def stt_languages(cap):
    """The `language` form field of every STT request, in order."""
    out = []
    for call in cap.of("/audio/transcriptions"):
        body = call["body"].decode("latin1")
        if "name=\"language\"" in body:
            out.append(body.split("name=\"language\"")[1].split("\r\n\r\n", 1)[1].split("\r\n", 1)[0])
        else:
            out.append(None)
    return out


def warnings_of(msg):
    return (msg["session"].get("x_realtimevoice") or {}).get("warnings") or []


# ------------------------------------------------ B: later update overridden

def test_later_update_language_is_overridden(env):
    """B: `voice-route:auto` resolved with de, then a LATER session.update with
    `language: en` must keep de and warn — never switch to en silently."""
    tc, cap, _ = env
    with connect(tc, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(initial_update("anna"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        # LATER update carries a different language
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" in warnings_of(upd)
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "de"


def test_repeated_later_update_stays_overridden(env):
    """B': a second later update with en must STILL report de + warning."""
    tc, cap, _ = env
    with connect(tc, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(initial_update("anna"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        for _ in range(2):
            ws.send_json({"type": "session.update", "session": {"language": "en"}})
            upd = recv_until(ws, "session.updated", "error")
            assert upd["type"] == "session.updated", upd
            assert upd["session"]["language"] == "de"
            assert "language_overridden_by_profile" in warnings_of(upd)
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "de"


# ------------------------------------------------ C: profile:<id>

def test_profile_route_update_language_is_overridden(env):
    """C: `profile:<id>` later update with top-level AND transcription language
    en must keep de + warning, and the STT request must ask for de."""
    tc, cap, _ = env
    with connect(tc, "profile:anna") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "session.update",
            "session": {
                "language": "en",
                "input_audio_transcription": {"language": "en"},
                "audio": {"input": {"transcription": {"language": "en"}}},
            },
        })
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" in warnings_of(upd)
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "de"


def test_no_client_language_keeps_profile_language(env):
    """D: no client language -> de and NO warning."""
    tc, cap, _ = env
    with connect(tc, "profile:anna") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"instructions": "x"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" not in warnings_of(upd)


def test_identical_client_language_no_warning(env):
    """An identical client language is a silent no-op (no warning)."""
    tc, cap, _ = env
    with connect(tc, "profile:anna") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"language": "de"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" not in warnings_of(upd)


# ------------------------------------------------ C': profile-test:<ticket>

def test_profile_test_route_update_language_is_overridden(env):
    """C': `profile-test:<ticket>` behaves exactly like `profile:<id>`."""
    tc, cap, _ = env
    r = tc.post(
        "/api/conversation-profiles/anna/test-session",
        headers={"Authorization": "Bearer tok"},
        json={},
    )
    assert r.status_code == 200, r.text
    model = r.json()["model"]
    with connect(tc, model) as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" in warnings_of(upd)
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "de"


# ------------------------------------------------ voice-route:<set>, no set named

def test_voice_route_set_name_update_language_is_overridden(env):
    tc, cap, _ = env
    with connect(tc, "voice-route:anna") as ws:
        assert ws.receive_json()["session"]["language"] == "de"
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de"
        assert "language_overridden_by_profile" in warnings_of(upd)
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "de"


@pytest.mark.parametrize("model", ["", "some-other-apps-model"])
def test_a_call_naming_no_set_keeps_the_default_sets_language(env, model):
    """A call naming no set gets the default set, and with it the set's
    language: the client language is overridden like on any set."""
    tc, cap, _ = env
    with connect(tc, model) as ws:
        created = ws.receive_json()
        assert created["session"]["realtimevoice"]["set"] == "bert"
        assert created["session"]["language"] == "de-AT"
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        upd = recv_until(ws, "session.updated", "error")
        assert upd["type"] == "session.updated", upd
        assert upd["session"]["language"] == "de-AT"
        assert "language_overridden_by_profile" in warnings_of(upd)


# ------------------------------------------------ each model in its own format

def test_stt_gets_the_sets_language_in_its_format(env):
    """bert's listening model wants the language by name; de-AT is sent as
    its base language."""
    tc, cap, _ = env
    with connect(tc, "profile:bert") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        speech_turn(ws)
    assert stt_languages(cap)[-1] == "German"


def test_tts_with_format_none_is_sent_no_language(env):
    tc, cap, _ = env
    with connect(tc, "profile:bert") as ws:
        ws.receive_json()
        ws.send_json({"type": "response.create"})
        recv_until(ws, "response.done")
    payload = cap.of("/audio/speech")[-1]["payload"]
    assert payload["model"] == "tts-model"
    assert "language" not in payload


def test_tts_with_format_code_gets_the_code(env):
    tc, cap, _ = env
    with connect(tc, "profile:anna") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        ws.send_json({"type": "response.create"})
        recv_until(ws, "response.done")
    assert cap.of("/audio/speech")[-1]["payload"]["language"] == "de"
