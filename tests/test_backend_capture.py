"""fakeHTTP request captures: chosen backend model/voice actually sent.

The public WS resolves a set to a frozen snapshot; these tests prove the
RESOLUTION reaches the real adapter requests by capturing the exact HTTP
bodies (model, voice, language) through an httpx.MockTransport-backed
client — the same code path used against live backends.
"""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_backends import tone_pcm
from realtimevoice.app import create_app
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.audio import pcm16_to_wav
from realtimevoice.config import Settings, SettingsStore, VoiceSelection
from set_helpers import add_set

SPEECH_WAV = pcm16_to_wav(tone_pcm(200), 24000)


class CaptureHTTP:
    """Records adapter requests; answers like a tiny OpenAI-compatible box."""

    def __init__(self):
        self.calls: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append({"host": request.url.host, "path": path, "content": request.read()})
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Hallo Welt, wie geht es dir heute?"})
        if path.endswith("/chat/completions"):
            sse = (
                'data: {"choices":[{"delta":{"content":"Guten Tag, das ist ein Test."}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=SPEECH_WAV, headers={"content-type": "audio/wav"})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    def of(self, suffix: str) -> list[dict]:
        return [c for c in self.calls if c["path"].endswith(suffix)]

    def json_of(self, suffix: str) -> dict:
        calls = self.of(suffix)
        assert calls, f"no request to {suffix} was captured"
        return json.loads(calls[-1]["content"].decode("utf-8"))


def make_settings() -> Settings:
    settings = Settings()
    settings.session.voice = "ryan"   # the server's call default, never a set's
    settings.session.language = "de"
    # the set for clients that name none
    add_set(
        settings, "base", base_url="http://capture.test/v1", language="de",
        models={"stt": "whisper-base", "llm": "llm-base", "tts": "tts-base"},
        make_default=True,
    )
    settings.profiles["tts"]["base-tts"].voice = "vivian"
    # a set on other models; its speaking model wants the language by name
    add_set(
        settings, "alt", base_url="http://alt-capture.test/v1", language="de-AT",
        models={"stt": "alt-stt-model", "llm": "alt-llm-model", "tts": "alt-tts-model"},
        language_formats={"tts": "name"},
    )
    settings.profiles["tts"]["alt-tts"].voice = "serena"
    settings.profiles["tts"]["alt-tts"].capabilities.voices = ["serena", "ryan"]
    # the same models, with a built-in voice picked in the set
    settings.conversation_profiles["alt-voiced"] = settings.conversation_profiles["alt"].model_copy(
        deep=True, update={"id": "alt-voiced"}
    )
    settings.conversation_profiles["alt-voiced"].spec.voice = VoiceSelection(kind="native", native_voice="ryan")
    return settings


@pytest.fixture
def harness(tmp_path):
    capture = CaptureHTTP()
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=make_settings()),
        backend_factory=lambda settings, http_client=None: Backends_of(settings, client),
        auth_token="ws-token",
    )
    return TestClient(app), capture


def Backends_of(settings, client):
    from realtimevoice.backends.base import Backends

    stt, llm, tts = build_backends(settings.stt, settings.llm, settings.tts, client)
    return Backends(stt=stt, llm=llm, tts=tts)


def run_speech_turn(test_client, model=""):
    with test_client.websocket_connect(
        f"/v1/realtime?model={model}", subprotocols=["rtv.bearer", "rtv.bearer.ws-token"]
    ) as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        speech = tone_pcm(300)
        silence = b"\x00\x00" * 480 * 5
        for i in range(0, len(speech), 9600):
            ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(speech[i : i + 9600]).decode()})
        for _ in range(6):
            ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()})
        got = []
        while "response.done" not in {g["type"] for g in got}:
            got.append(ws.receive_json())
        done = [g for g in got if g["type"] == "response.done"][-1]
        assert done["response"]["status"] == "completed"
        return created


def _form_field(content: bytes, name: str) -> str | None:
    """A field of a captured multipart form body (None = not sent)."""
    marker = f'name="{name}"'.encode()
    if marker not in content:
        return None
    rest = content.split(marker, 1)[1].split(b"\r\n\r\n", 1)[1]
    return rest.split(b"\r\n", 1)[0].decode()


def test_set_models_reach_actual_backend_requests(harness):
    test_client, capture = harness
    run_speech_turn(test_client, "profile:alt")
    # STT: multipart form carries the model and the set's language as a code
    stt_calls = capture.of("/audio/transcriptions")
    assert stt_calls, "STT request missing"
    assert _form_field(stt_calls[-1]["content"], "model") == "alt-stt-model"
    assert _form_field(stt_calls[-1]["content"], "language") == "de"
    # LLM: JSON body carries the model
    llm_body = capture.json_of("/chat/completions")
    assert llm_body["model"] == "alt-llm-model"
    assert llm_body["stream"] is True
    # TTS: JSON body carries the model, the voice and the language by name
    tts_body = capture.json_of("/audio/speech")
    assert tts_body["model"] == "alt-tts-model"
    assert tts_body["voice"] == "serena"
    assert tts_body["language"] == "German"
    assert tts_body["input"] == "Guten Tag, das ist ein Test."
    # every request went to the set's models, none to another set's
    assert {c["host"] for c in capture.calls} == {"alt-capture.test"}
    assert not any(b"whisper-base" in c["content"] or b"tts-base" in c["content"] for c in capture.calls)


def test_the_speaking_models_voice_beats_the_server_default_voice(harness):
    """The set picks no voice: its speaking model's voice 'serena' wins over
    the server's call default 'ryan'."""
    test_client, capture = harness
    run_speech_turn(test_client, "profile:alt")
    tts_body = capture.json_of("/audio/speech")
    assert tts_body["voice"] == "serena"
    assert tts_body["model"] == "alt-tts-model"


def test_the_sets_built_in_voice_wins_over_the_models_voice(harness):
    test_client, capture = harness
    run_speech_turn(test_client, "profile:alt-voiced")
    tts_body = capture.json_of("/audio/speech")
    assert tts_body["voice"] == "ryan"
    assert tts_body["model"] == "alt-tts-model"


@pytest.mark.parametrize("model", ["", "some-other-apps-model"])
def test_a_call_naming_no_set_uses_the_default_sets_models(harness, model):
    test_client, capture = harness
    run_speech_turn(test_client, model)
    stt_calls = capture.of("/audio/transcriptions")
    assert stt_calls, "STT request missing"
    assert _form_field(stt_calls[-1]["content"], "model") == "whisper-base"
    assert capture.json_of("/chat/completions")["model"] == "llm-base"
    tts_body = capture.json_of("/audio/speech")
    assert tts_body["model"] == "tts-base"
    assert tts_body["voice"] == "vivian"
    assert tts_body["language"] == "de"


def test_captured_requests_leak_no_credentials(harness):
    test_client, capture = harness
    run_speech_turn(test_client, "profile:alt")
    for call in capture.calls:
        # the adapters send Authorization headers, but never keys in bodies
        assert b"sk-" not in call["content"]
