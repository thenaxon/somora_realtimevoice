"""WebSocket integration over the real /v1/realtime endpoint."""

import base64
import threading

import pytest
from fastapi.testclient import TestClient

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends, tone_pcm
from realtimevoice.app import create_app
from realtimevoice.config import Settings, SettingsStore
from set_helpers import add_set, settings_with_default_set


class BlockingSTT(FakeSTT):
    """Blocks on a threading.Event via a thread (loop stays live)."""

    def __init__(self):
        super().__init__()
        self._gate = threading.Event()

    async def transcribe(self, wav, *, language=None):
        self.calls.append(wav)
        import asyncio

        await asyncio.get_running_loop().run_in_executor(None, self._gate.wait)
        return self.text


@pytest.fixture
def app_client(tmp_path):
    """Normal flow fixture: plain FakeSTT (no gate); a call naming no set
    gets the server's default set."""
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings_with_default_set()),
        backend_factory=lambda settings, client=None: make_backends(FakeSTT(), FakeLLM(), FakeTTS()),
        auth_token="ws-token",
    )
    return TestClient(app)


@pytest.fixture
def blocked_app_client(tmp_path):
    """Fixture with a gated STT; the test must release the gate in finally."""
    stt = BlockingSTT()
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings_with_default_set()),
        backend_factory=lambda settings, client=None: make_backends(stt, FakeLLM(), FakeTTS()),
        auth_token="ws-token",
    )
    return TestClient(app), stt


def b64(pcm):
    return base64.b64encode(pcm).decode()


def test_ws_subprotocol_token_auth(app_client):
    """Browser path: token offered as second subprotocol, never in the URL."""
    test_client = app_client
    with test_client.websocket_connect(
        "/v1/realtime?model=",
        subprotocols=["rtv.bearer", "rtv.bearer.ws-token"],
    ) as ws:
        accepted = ws.accepted_subprotocol
        assert accepted == "rtv.bearer"
        assert ws.receive_json()["type"] == "session.created"


def test_ws_subprotocol_wrong_token_rejected(app_client):
    test_client = app_client
    with pytest.raises(Exception):
        with test_client.websocket_connect(
            "/v1/realtime",
            subprotocols=["rtv.bearer", "rtv.bearer.wrong-token"],
        ) as ws:
            ws.receive_json()


def test_ws_requires_token(app_client):
    test_client = app_client
    with pytest.raises(Exception):
        with test_client.websocket_connect("/v1/realtime"):
            pass
    # wrong token
    with pytest.raises(Exception):
        with test_client.websocket_connect("/v1/realtime?token=wrong"):
            pass


def test_ws_flow_over_public_protocol(app_client):
    test_client = app_client
    with test_client.websocket_connect("/v1/realtime?token=ws-token") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}

        ws.send_json({"type": "session.update", "session": {"instructions": "Kurz antworten."}})
        types = set()
        while "session.updated" not in types:
            types.add(ws.receive_json()["type"])

        # stream a speech turn in chunks; reader must stay live throughout
        speech = tone_pcm(300)
        silence = b"\x00\x00" * 480 * 5
        for i in range(0, len(speech), 9600):
            ws.send_json({"type": "input_audio_buffer.append", "audio": b64(speech[i : i + 9600])})
        for _ in range(6):
            ws.send_json({"type": "input_audio_buffer.append", "audio": b64(silence)})

        got = []
        while "response.done" not in {g["type"] for g in got}:
            got.append(ws.receive_json())
        types_list = [g["type"] for g in got]
        assert "input_audio_buffer.speech_started" in types_list
        assert "input_audio_buffer.speech_stopped" in types_list
        assert "conversation.item.input_audio_transcription.completed" in types_list
        assert "response.created" in types_list
        assert "response.output_audio.delta" in types_list
        done = [g for g in got if g["type"] == "response.done"][-1]
        assert done["response"]["status"] == "completed"


def test_ws_reader_live_while_stt_blocked(blocked_app_client):
    test_client, stt = blocked_app_client
    with test_client.websocket_connect("/v1/realtime?token=ws-token") as ws:
        try:
            assert ws.receive_json()["type"] == "session.created"
            speech = tone_pcm(300)
            for i in range(0, len(speech), 9600):
                ws.send_json({"type": "input_audio_buffer.append", "audio": b64(speech[i : i + 9600])})
            for _ in range(7):
                ws.send_json({"type": "input_audio_buffer.append", "audio": b64(b"\x00\x00" * 480 * 5)})
            seen = set()
            while "input_audio_buffer.speech_stopped" not in seen:
                seen.add(ws.receive_json()["type"])
            # STT is blocked now; the reader must still echo control events
            ws.send_json({"type": "session.update", "session": {"instructions": "immer noch live"}})
            while "session.updated" not in seen:
                seen.add(ws.receive_json()["type"])
            assert "session.updated" in seen
        finally:
            # release the gate BEFORE the websocket context exits, otherwise
            # the executor thread blocks the context __exit__ on shutdown
            stt._gate.set()


def test_ws_session_isolation(app_client):
    test_client = app_client
    with test_client.websocket_connect("/v1/realtime?token=ws-token") as ws1, test_client.websocket_connect(
        "/v1/realtime?token=ws-token"
    ) as ws2:
        first1 = ws1.receive_json()
        first2 = ws2.receive_json()
        assert first1["type"] == "session.created" and first2["type"] == "session.created"
        assert first1["session"]["id"] != first2["session"]["id"]
        ws1.send_json({"type": "session.update", "session": {"instructions": "Nur Eins."}})
        ws2.send_json({"type": "session.update", "session": {"instructions": "Nur Zwei."}})
        updated = []
        while len([u for u in updated if u["session"]["instructions"] == "Nur Zwei."]) < 1 or len(
            [u for u in updated if u["session"]["instructions"] == "Nur Eins."]
        ) < 1:
            updated.append(ws1.receive_json())
            updated.append(ws2.receive_json())
        upd1 = [u for u in updated if u["session"]["instructions"] == "Nur Eins."][-1]
        upd2 = [u for u in updated if u["session"]["instructions"] == "Nur Zwei."][-1]
        assert upd1["session"]["id"] == first1["session"]["id"]
        assert upd2["session"]["id"] == first2["session"]["id"]


def test_ws_binary_frame_rejected(app_client):
    test_client = app_client
    with test_client.websocket_connect("/v1/realtime?token=ws-token") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"\x00\x01")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["error"]["type"] == "binary_frame_not_supported"


def test_ws_bad_json_rejected(app_client):
    test_client = app_client
    with test_client.websocket_connect("/v1/realtime?token=ws-token") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_text("this is not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["error"]["type"] == "invalid_json"


# ------------------------------------------------------------ set selection

AUTH_SUBPROTOCOLS = ["rtv.bearer", "rtv.bearer.ws-token"]


def _app(tmp_path, settings: Settings) -> TestClient:
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
        backend_factory=lambda settings, client=None: make_backends(FakeSTT(), FakeLLM(), FakeTTS()),
        auth_token="ws-token",
    )
    return TestClient(app)


def _refusal(tc: TestClient, model: str):
    from starlette.testclient import WebSocketDenialResponse

    with pytest.raises(WebSocketDenialResponse) as refused:
        with tc.websocket_connect(f"/v1/realtime?model={model}", subprotocols=AUTH_SUBPROTOCOLS):
            pass
    return refused.value


@pytest.mark.parametrize("model", ["", "gpt-realtime"])
def test_a_call_naming_no_set_is_refused_without_a_default_set(tmp_path, model):
    """No default set chosen: a client naming no set (model "" or its own
    model name) is refused instead of silently getting some models."""
    settings = Settings()
    add_set(settings, "anna")  # enabled, but not the default
    refused = _refusal(_app(tmp_path, settings), model)
    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "no_set_selected"


@pytest.mark.parametrize("model", ["", "gpt-realtime"])
def test_a_call_naming_no_set_gets_the_default_set(tmp_path, model):
    settings = Settings()
    add_set(settings, "anna", language="de")
    add_set(settings, "bert", language="en", make_default=True)
    with _app(tmp_path, settings).websocket_connect(
        f"/v1/realtime?model={model}", subprotocols=AUTH_SUBPROTOCOLS
    ) as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        info = created["session"]["realtimevoice"]
        assert info["set"] == "bert"
        assert info["tts"]["profile"] == "bert-tts"
        assert info["language"] == "en"
        assert created["session"]["x_realtimevoice"]["route"]["requested"] == model


def test_a_default_set_that_is_switched_off_refuses_the_call(tmp_path):
    settings = Settings()
    add_set(settings, "anna", enabled=False, make_default=True)
    refused = _refusal(_app(tmp_path, settings), "")
    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "profile_not_available"


def test_profile_model_resolves_an_enabled_set_and_refuses_a_disabled_one(tmp_path):
    settings = Settings()
    add_set(settings, "anna")
    add_set(settings, "bert", enabled=False)
    tc = _app(tmp_path, settings)
    with tc.websocket_connect("/v1/realtime?model=profile:anna", subprotocols=AUTH_SUBPROTOCOLS) as ws:
        assert ws.receive_json()["session"]["realtimevoice"]["set"] == "anna"
    off = _refusal(tc, "profile:bert")
    assert off.status_code == 404 and off.json()["error"]["code"] == "profile_not_available"
    unknown = _refusal(tc, "profile:nobody")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "unknown_conversation_profile"


def test_voice_route_with_a_set_name_resolves_that_set(tmp_path):
    settings = Settings()
    add_set(settings, "anna", language="de")
    add_set(settings, "bert", language="en", make_default=True)
    tc = _app(tmp_path, settings)
    with tc.websocket_connect("/v1/realtime?model=voice-route:anna", subprotocols=AUTH_SUBPROTOCOLS) as ws:
        session = ws.receive_json()["session"]
        assert session["realtimevoice"]["set"] == "anna"
        assert session["realtimevoice"]["language"] == "de"
        assert session["x_realtimevoice"]["route"]["requested"] == "anna"
    unknown = _refusal(tc, "voice-route:nobody")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_voice_route"


def test_voice_route_auto_binds_the_set_named_by_the_first_voice(tmp_path):
    settings = Settings()
    add_set(settings, "anna", language="de")
    with _app(tmp_path, settings).websocket_connect(
        "/v1/realtime?model=voice-route:auto", subprotocols=AUTH_SUBPROTOCOLS
    ) as ws:
        created = ws.receive_json()
        assert "realtimevoice" not in created["session"]  # pending: no set yet
        # an unknown voice names no set: error, the route stays pending
        ws.send_json({"type": "session.update", "session": {"voice": "nobody"}})
        msg = ws.receive_json()
        while msg["type"] != "error":
            msg = ws.receive_json()
        assert msg["error"]["code"] == "unknown_voice_route"
        msg = ws.receive_json()  # the rest of that update still applies
        while msg["type"] != "session.updated":
            msg = ws.receive_json()
        assert "realtimevoice" not in msg["session"]
        ws.send_json({"type": "session.update", "session": {"voice": "anna"}})
        msg = ws.receive_json()
        while msg["type"] != "session.updated":
            msg = ws.receive_json()
        assert msg["session"]["realtimevoice"]["set"] == "anna"


def test_models_lists_enabled_sets_and_the_auto_route(tmp_path):
    settings = Settings()
    add_set(settings, "anna", display_name="Anna", language="de", make_default=True)
    add_set(settings, "bert", display_name="Bert", language="en")
    add_set(settings, "carl", enabled=False)
    tc = _app(tmp_path, settings)
    assert tc.get("/v1/models").status_code == 401
    body = tc.get("/v1/models", headers={"Authorization": "Bearer ws-token"}).json()
    by_id = {entry["id"]: entry for entry in body["data"]}
    assert list(by_id) == ["profile:anna", "profile:bert", "voice-route:auto"]
    assert by_id["profile:anna"]["x_realtimevoice"] == {"display_name": "Anna", "language": "de", "default": True}
    assert by_id["profile:bert"]["x_realtimevoice"] == {"display_name": "Bert", "language": "en", "default": False}
    assert by_id["voice-route:auto"]["x_realtimevoice"]["default"] is False
