"""Voice routing (`voice-route:` model namespace) regressions.

The requested voice names a set; the set's own models and voice are what
the backends get. Proves end to end:

- `voice-route:auto` resolves at the initial session.update (the EXACT GA
  form Somora's adapter sends: type/model/instructions/output_modalities/
  audio.input.{format,transcription,turn_detection}/audio.output.voice/
  tools/tool_choice) BEFORE any input audio / first response; the payload
  carries the set's ACTUAL backend voice and recording, never the requested
  name,
- an unknown or switched-off set fails closed with machine-readable codes
  (no wrong-voice or default fallback), unrouted TTS is refused,
- after the first resolution the voice is frozen: identical updates are
  no-ops, any different voice afterwards is rejected — no hot changes; a new
  connection per voice,
- explicit `voice-route:<name>` resolves at connect (fail closed before
  accept on an unknown or switched-off set),
- instructions + tools of the initial update are preserved (LLM payload
  proof, PCM 24k input format accepted), tool roundtrip and cancel work on
  a routed session.

Fixtures are synthetic; no private runtime data.
"""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.app import create_app
from realtimevoice.backends.base import Backends
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.config import (
    BackendProfile,
    Settings,
    SettingsStore,
    VoiceSelection,
    reference_fingerprint,
)
from realtimevoice.voices import SampleStore
from test_catalog_api import add_library_voice, add_set

AUTH = {"Authorization": "Bearer tok"}


def wav_bytes(pcm: bytes) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return buf.getvalue()


class RecordingTransport(httpx.MockTransport):
    def __init__(self, requests: list):
        self.requests = requests
        super().__init__(handler=self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, payload))
        if request.url.path.endswith("/chat/completions"):
            sse = (
                'data: {"choices":[{"delta":{"content":"Kurzer Satz."}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(b"\x01\x02" * 600))
        return httpx.Response(200, json={"data": []})


def make_settings(data_dir: str) -> tuple[Settings, str]:
    """Sets `clara` and `jonas` (on), `parked` (off); one speaking model per
    set, the same model id, a library recording each. Returns the settings
    and the fingerprint of clara's recording."""
    s = Settings()
    for role in ("stt", "llm"):
        s.profiles[role][f"{role}-a"] = BackendProfile(
            id=f"{role}-a", role=role, base_url="http://backend.test/v1", model=f"{role}-model",
        )
    for pid, voice in (("clone-clara", "clara-reference"), ("clone-jonas", "jonas-reference")):
        s.profiles["tts"][pid] = BackendProfile(
            id=pid, role="tts", base_url="http://backend.test/v1", model="chatterbox-fp16",
            voice=voice, capabilities={"voice_cloning": "reference_audio"},
        )
    clara = add_library_voice(s, data_dir, "clara-voice", ms=2000)
    add_library_voice(s, data_dir, "jonas-voice", ms=2500)
    add_set(s, "clara", enabled=True, tts="clone-clara",
            voice=VoiceSelection(kind="library", voice_id="clara-voice"))
    add_set(s, "jonas", enabled=True, tts="clone-jonas",
            voice=VoiceSelection(kind="library", voice_id="jonas-voice"))
    add_set(s, "parked", enabled=False, tts="clone-clara",
            voice=VoiceSelection(kind="library", voice_id="clara-voice"))
    raw = SampleStore(data_dir).read("clara-voice", clara.samples[0].sha256)
    return s, reference_fingerprint({"ref_audio": base64.b64encode(raw).decode()})


def somora_initial(model: str, voice: str, instructions: str = "Du bist das Sprach-Ich von Agent A.") -> dict:
    """The EXACT initial session.update structure Somora's adapter sends
    (openai-provider.ts configure()): GA form, per-agent voice."""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": model,
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "whisper-1", "language": "de"},
                    "turn_detection": {
                        "type": "server_vad", "threshold": 0.4,
                        "prefix_padding_ms": 200, "silence_duration_ms": 420,
                        "interrupt_response": True,
                    },
                },
                "output": {"voice": voice},
            },
            "tools": [{
                "type": "function", "name": "get_weather",
                "description": "Provides the current demo weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            }],
            "tool_choice": "auto",
        },
    }


@pytest.fixture
def routed_env(tmp_path):
    """Recording-transport env (payload-level proof)."""
    requests: list = []

    def factory(settings, client=None):
        stt, llm, tts = build_backends(
            settings.stt, settings.llm, settings.tts,
            httpx.AsyncClient(transport=RecordingTransport(requests), base_url="http://backend.test"),
        )
        return Backends(stt=stt, llm=llm, tts=tts)

    settings, fp = make_settings(str(tmp_path))
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
        backend_factory=factory,
        auth_token="tok",
    )
    with TestClient(app) as client:
        client.fp_clara = fp
        yield client, requests


def connect(client, model):
    return client.websocket_connect(
        f"/v1/realtime?model={model}", subprotocols=["rtv.bearer", "rtv.bearer.tok"]
    )


def recv_until(ws, *types):
    while True:
        msg = ws.receive_json()
        if msg.get("type") in types:
            return msg


def run_response(ws, instructions="Sag kurz Hallo."):
    """response.create -> returns (status, errors_seen_in_order). A None
    instructions sends a PLAIN create (system prompt = session
    instructions, no one-off override)."""
    body = (
        {"type": "response.create", "response": {"instructions": instructions}}
        if instructions is not None
        else {"type": "response.create"}
    )
    ws.send_json(body)
    errors = []
    while True:
        msg = ws.receive_json()
        if msg["type"] == "error":
            errors.append(msg["error"])
        elif msg["type"] == "response.done":
            return msg["response"]["status"], errors


def speech_payloads(requests):
    return [p for _, path, p in requests if path.endswith("/audio/speech")]


def test_auto_route_initial_update_resolves_actual_voice(routed_env):
    """voice-route:auto + Somora-shaped initial update naming set 'clara':
    the first TTS carries the set's ACTUAL voice, model and recording — the
    requested name is never substituted into the payload; instructions +
    tools from the initial update are preserved."""
    client, requests = routed_env
    with connect(client, "voice-route:auto") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        ws.send_json(somora_initial("voice-route:auto", "clara"))
        updated = recv_until(ws, "session.updated", "error")
        assert updated["type"] == "session.updated", updated
        rv = updated["session"]["realtimevoice"]
        assert rv["set"] == "clara"
        assert rv["tts"]["profile"] == "clone-clara"
        assert rv["tts"]["voice"] == "clara-reference"  # ACTUAL voice
        assert updated["session"]["voice"] == "clara-reference"
        assert updated["session"]["instructions"] == "Du bist das Sprach-Ich von Agent A."
        assert [t["name"] for t in updated["session"]["tools"]] == ["get_weather"]

        status, errors = run_response(ws, instructions=None)  # plain create: session instructions reach the LLM
        assert status == "completed"
        assert errors == []  # identical-name no-op must not break the voice
        # send the SAME name a second time: the TTS must STILL use the
        # actual backend voice (the name never overwrites cfg.voice)
        ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "clara"}}}})
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        status2, errors2 = run_response(ws, instructions=None)
        assert status2 == "completed" and errors2 == []
    payload = speech_payloads(requests)[-1]
    assert payload["voice"] == "clara-reference"  # NOT the requested name 'clara'
    assert payload["model"] == "chatterbox-fp16"
    import hashlib

    assert hashlib.sha256(str(payload["ref_audio"]).encode()).hexdigest() == client.fp_clara
    llm_payloads = [p for _, path, p in requests if path.endswith("/chat/completions")]
    assert any("Sprach-Ich von Agent A" in json.dumps(p) for p in llm_payloads)
    assert any("get_weather" in json.dumps(p) for p in llm_payloads)


def test_unknown_voice_fails_closed_no_fallback(routed_env):
    client, requests = routed_env
    with connect(client, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(somora_initial("voice-route:auto", "nobody"))
        msg = recv_until(ws, "error", "session.updated")
        assert msg["type"] == "error"
        assert msg["error"]["code"] == "unknown_voice_route"
        # a response while unrouted is refused with the pending error and
        # makes ZERO LLM calls (contract §6; no response is created)
        ws.send_json({"type": "response.create", "response": {"instructions": "Hallo?"}})
        err = recv_until(ws, "error")
        assert err["error"].get("code") == "voice_route_pending"
    assert speech_payloads(requests) == []  # never synthesized with a fallback
    assert [p for _, path, p in requests if path.endswith("/chat/completions")] == []


def test_unrouted_response_refused_before_initial_update(routed_env):
    client, requests = routed_env
    with connect(client, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json({"type": "response.create", "response": {"instructions": "x"}})
        # repeated response.create: an error EVERY time, zero LLM calls
        seen = set()
        for _ in range(2):
            msg = recv_until(ws, "error")
            seen.add(msg["error"].get("code"))
            ws.send_json({"type": "response.create", "response": {"instructions": "x"}})
        assert "voice_route_pending" in seen
        assert not [p for _, path, p in requests if path.endswith("/chat/completions")]
    assert speech_payloads(requests) == []


def test_identical_voice_after_freeze_is_noop(routed_env):
    client, _ = routed_env
    with connect(client, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(somora_initial("voice-route:auto", "clara"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        # identical requested voice: accepted, NOT voice_frozen
        ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "clara"}}}})
        msg = recv_until(ws, "session.updated", "error")
        assert msg["type"] == "session.updated"
        assert msg["session"]["voice"] == "clara-reference"


def test_different_voice_after_freeze_rejected(routed_env):
    client, requests = routed_env
    with connect(client, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(somora_initial("voice-route:auto", "clara"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        # another set's name: rejected (frozen), the rest of the update applies
        ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "jonas"}}}})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "error":
                assert msg["error"]["code"] == "voice_route_frozen"
                break
        # the frozen configuration still synthesizes correctly
        status, errors = run_response(ws)
        assert status == "completed" and errors == []
    payload = speech_payloads(requests)[-1]
    assert payload["voice"] == "clara-reference"


def test_switched_off_set_fails_closed_on_auto_route(routed_env):
    client, requests = routed_env
    with connect(client, "voice-route:auto") as ws:
        ws.receive_json()
        ws.send_json(somora_initial("voice-route:auto", "parked"))
        msg = recv_until(ws, "error", "session.updated")
        assert msg["type"] == "error"
        assert msg["error"]["code"] == "profile_not_available"
        ws.send_json({"type": "response.create", "response": {"instructions": "Hallo?"}})
        assert recv_until(ws, "error")["error"].get("code") == "voice_route_pending"
    assert speech_payloads(requests) == []


def test_explicit_route_resolves_at_connect(routed_env):
    client, requests = routed_env
    with connect(client, "voice-route:clara") as ws:
        created = ws.receive_json()
        rv = created["session"]["realtimevoice"]
        assert rv["set"] == "clara"
        assert rv["tts"]["profile"] == "clone-clara"
        assert rv["tts"]["voice"] == "clara-reference"
        # identical initial update is an accepted no-op (the name never
        # lands in cfg.voice)
        ws.send_json(somora_initial("voice-route:clara", "clara"))
        assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
        status, errors = run_response(ws)
        assert status == "completed" and errors == []
    payload = speech_payloads(requests)[-1]
    assert payload["voice"] == "clara-reference"


@pytest.mark.parametrize(
    "name,code", [("nobody", "unknown_voice_route"), ("parked", "profile_not_available")]
)
def test_explicit_route_unknown_or_off_fails_closed_before_accept(routed_env, name, code):
    from starlette.testclient import WebSocketDenialResponse

    client, _ = routed_env
    with pytest.raises(WebSocketDenialResponse) as refused:
        with connect(client, f"voice-route:{name}") as ws:
            ws.receive_json()
    assert refused.value.json()["error"]["code"] == code


def test_tool_roundtrip_and_cancel_on_routed_session(tmp_path):
    """Routed session with the Somora-shaped initial form: instructions +
    tools preserved, tool call answered and the follow-up response
    completes; a mid-response cancel stops cleanly (PCM 24k in/out)."""
    llm = FakeLLM(script=[{"tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Wien\"}"}]},
                          {"text": "In Wien ist es sonnig."}])
    tts = FakeTTS()
    stt = FakeSTT()
    settings, _fp = make_settings(str(tmp_path))
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
        backend_factory=lambda s, client=None: make_backends(stt=stt, llm=llm, tts=tts),
        auth_token="tok",
    )
    with TestClient(app) as client:
        with connect(client, "voice-route:auto") as ws:
            ws.receive_json()
            ws.send_json(somora_initial("voice-route:auto", "clara"))
            assert recv_until(ws, "session.updated", "error")["type"] == "session.updated"
            # response 1: tool call
            ws.send_json({"type": "response.create", "response": {"instructions": "Wetter?"}})
            saw_call = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "response.output_item.done" and msg["item"]["type"] == "function_call":
                    saw_call = msg["item"]
                if msg["type"] == "response.done":
                    break
            assert saw_call is not None and saw_call["name"] == "get_weather"
            # tool output -> follow-up response completes
            ws.send_json({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": saw_call["call_id"],
                         "output": json.dumps({"temperature_c": 21, "condition": "sunny"})},
            })
            ws.send_json({"type": "response.create"})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "response.done":
                    assert msg["response"]["status"] == "completed"
                    break
            # routed TTS got the ACTUAL voice (never the requested name), PCM 24k
            assert tts.calls and tts.calls[-1]["voice"] == "clara-reference"
            # cancel mid-response: gate the LLM stream so the response is
            # still active when response.cancel arrives, then release
            import asyncio

            llm.gate = asyncio.Event()
            llm.script.append({"text": "x" * 400})
            ws.send_json({"type": "response.create", "response": {"instructions": "lang"}})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "response.created":
                    break
            ws.send_json({"type": "response.cancel"})
            llm.gate.set()
            while True:
                msg = ws.receive_json()
                if msg["type"] == "response.done":
                    assert msg["response"]["status"] == "cancelled"
                    break
    # LLM saw the initial instructions and the tool schema
    assert any("Sprach-Ich von Agent A" in json.dumps(c["messages"]) for c in llm.calls)
    assert any(c["tools"] for c in llm.calls)
