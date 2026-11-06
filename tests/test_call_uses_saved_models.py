"""A call speaks with the saved models and recording; a running call keeps
what it started with.

Proven at the EFFECTIVE REQUEST level (real adapter + recording transport,
payload bytes captured):

- a session-only settings save (as the Settings page sends it) changes no
  model, and a following call on a set carries the set's model, voice and
  REFERENCE fingerprint in the actual TTS payload,
- a model edit while a call is running: the running call keeps its frozen
  configuration, the next call on the same set uses the edited model at once
  (a saved set is live, nothing to re-apply),
- two simultaneous calls on different sets run different frozen snapshots,
- the editor-shaped masked-key update keeps the stored secret and the
  reference extras (no clobber by omission/mask).

Fixtures are synthetic (no private runtime data).
"""

import asyncio
import base64
import hashlib
import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_backends import tone_pcm
from realtimevoice.app import create_app
from realtimevoice.backends.base import Backends
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.catalog import mask_profile
from realtimevoice.config import (
    SECRET_MASK,
    BackendProfile,
    Settings,
    SettingsStore,
    VoiceSelection,
    reference_fingerprint,
)
from test_catalog_api import add_library_voice, add_set

AUTH = {"Authorization": "Bearer tok"}
EMBEDDED_REF = base64.b64encode(b"RIFFsynthetic-embedded-reference").decode()


def wav_bytes(pcm: bytes) -> bytes:
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
            return httpx.Response(200, content=wav_bytes(tone_pcm(120)))
        return httpx.Response(200, json={"data": []})


def make_settings(data_dir: str) -> tuple[Settings, str]:
    s = Settings()
    for role in ("stt", "llm"):
        s.profiles[role][f"{role}-a"] = BackendProfile(
            id=f"{role}-a", role=role, base_url="http://backend.test/v1", model=f"{role}-model",
        )
    s.profiles["tts"]["clone"] = BackendProfile(
        id="clone", role="tts", display_name="Chatterbox",
        base_url="http://backend.test/v1",
        model="chatterbox-fp16",
        voice="clara-reference",
        # an old embedded reference: a set's library voice always replaces it
        extra={"ref_audio": EMBEDDED_REF, "ref_text": "embedded transcript"},
        capabilities={"voice_cloning": "reference_audio"},
    )
    s.profiles["tts"]["qwen"] = BackendProfile(
        id="qwen", role="tts", base_url="http://backend.test/v1", model="qwen-tts",
        capabilities={"voices": ["serena", "ryan"]},
    )
    voice = add_library_voice(s, data_dir, "clara", ms=2000)
    add_set(s, "clara", enabled=True, voice=VoiceSelection(kind="library", voice_id="clara"))
    add_set(s, "serena", enabled=True, tts="qwen", voice=VoiceSelection(kind="native", native_voice="serena"))
    sample = voice.samples[0]
    from realtimevoice.voices import SampleStore

    ref = base64.b64encode(SampleStore(data_dir).read("clara", sample.sha256)).decode()
    return s, reference_fingerprint({"ref_audio": ref})


@pytest.fixture
def env(tmp_path):
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
        yield client, requests, fp


def tts_payloads(requests):
    return [p for _, path, p in requests if path.endswith("/audio/speech")]


def fingerprint_of(payload: dict) -> str:
    raw = payload.get("ref_audio")
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest() if raw else ""


def run_response(ws):
    ws.send_json({"type": "response.create", "response": {"instructions": "Sag kurz Hallo."}})
    while True:
        msg = ws.receive_json()
        if msg["type"] == "response.done":
            return msg["response"]["status"]


def connect(client, model):
    return client.websocket_connect(
        f"/v1/realtime?model={model}", subprotocols=["rtv.bearer", "rtv.bearer.tok"]
    )


def test_session_only_save_changes_no_model_then_set_call_carries_reference(env):
    client, requests, fp = env
    catalog_before = client.get("/api/catalog", headers=AUTH).json()
    resp = client.put(
        "/api/settings",
        json={"session": {"instructions": "Antworte besonders kurz.",
                          "turn_detection": {"prefix_padding_ms": 0}}},
        headers=AUTH,
    )
    assert resp.status_code == 200
    after = resp.json()
    assert after["session"]["instructions"] == "Antworte besonders kurz."
    assert after["session"]["turn_detection"]["prefix_padding_ms"] == 0  # explicit 0 preserved
    assert client.get("/api/catalog", headers=AUTH).json() == catalog_before

    with connect(client, "profile:clara") as ws:
        created = ws.receive_json()
        rv = created["session"]["realtimevoice"]
        assert rv["set"] == "clara"
        assert rv["tts"]["profile"] == "clone"
        assert rv["tts"]["ref_sha256"] == fp
        assert "ref_audio" not in json.dumps(created)
        assert run_response(ws) == "completed"
    payload = tts_payloads(requests)[-1]
    assert payload["model"] == "chatterbox-fp16"
    assert payload["voice"] == "clara-reference"
    # the voice's recording, never the entry's embedded reference
    assert fingerprint_of(payload) == fp
    assert payload["ref_audio"] != EMBEDDED_REF
    assert payload["ref_text"] == "Hallo Welt"


def test_model_edit_running_call_frozen_next_call_uses_it(env):
    client, requests, fp = env
    with connect(client, "profile:clara") as ws:
        created = ws.receive_json()
        assert created["session"]["realtimevoice"]["tts"]["model"] == "chatterbox-fp16"
        assert run_response(ws) == "completed"
        resp = client.put(
            "/api/profiles/tts/clone",
            json={"display_name": "Clara EDITED", "voice": "clara-reference-v2",
                  "model": "chatterbox-fp16-v2"},
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        # frozen: a response on the STILL-OPEN socket uses the old config
        assert run_response(ws) == "completed"
        old_payload = tts_payloads(requests)[-1]
        assert old_payload["model"] == "chatterbox-fp16"
        assert old_payload["voice"] == "clara-reference"
        assert fingerprint_of(old_payload) == fp

    # the next call on the SAME set: the edited model, no re-apply needed
    with connect(client, "profile:clara") as ws:
        rv = ws.receive_json()["session"]["realtimevoice"]
        assert rv["tts"]["display_name"] == "Clara EDITED"
        assert rv["tts"]["model"] == "chatterbox-fp16-v2"
        assert rv["tts"]["voice"] == "clara-reference-v2"
        assert rv["tts"]["ref_sha256"] == fp  # the recording is unchanged
        assert run_response(ws) == "completed"
    new_payload = tts_payloads(requests)[-1]
    assert new_payload["model"] == "chatterbox-fp16-v2"
    assert new_payload["voice"] == "clara-reference-v2"
    assert fingerprint_of(new_payload) == fp


def test_two_calls_on_different_sets_run_different_frozen_snapshots(env):
    client, requests, fp = env
    with connect(client, "profile:clara") as ws_a, connect(client, "profile:serena") as ws_b:
        info_a = ws_a.receive_json()["session"]["realtimevoice"]
        info_b = ws_b.receive_json()["session"]["realtimevoice"]
        assert info_a["tts"]["model"] == "chatterbox-fp16"
        assert info_b["tts"]["model"] == "qwen-tts"
        assert info_b["tts"]["voice"] == "serena"
        assert "ref_sha256" not in info_b["tts"]
        # a model edit during both calls changes neither
        assert client.put(
            "/api/profiles/tts/qwen", json={"model": "qwen-tts-v2"}, headers=AUTH
        ).status_code == 200
        assert run_response(ws_a) == "completed"
        payload_a = tts_payloads(requests)[-1]
        assert run_response(ws_b) == "completed"
        payload_b = tts_payloads(requests)[-1]
    assert payload_a["model"] == "chatterbox-fp16" and fingerprint_of(payload_a) == fp
    assert payload_b["model"] == "qwen-tts" and payload_b["voice"] == "serena"
    # a native voice never carries reference material
    assert "ref_audio" not in payload_b


def test_masked_key_and_extras_preserved_via_editor_update(env):
    """The editor PUT (masked api_key, no extra) keeps the stored secret and
    the reference material; the catalog view shows the mask only."""
    client, _, _ = env
    resp = client.post(
        "/api/profiles/llm",
        json={"id": "llm-evidence", "role": "llm", "model": "llm-a",
              "api_key": "sk-evidence-secret", "base_url": "http://backend.test/v1"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.json()["api_key"] == SECRET_MASK

    resp = client.put(
        "/api/profiles/llm/llm-evidence",
        json={"model": "llm-b", "api_key": SECRET_MASK},  # masked = unchanged
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.json()["api_key"] == SECRET_MASK

    catalog = client.get("/api/catalog", headers=AUTH).json()
    profile_view = next(p for p in catalog["roles"]["llm"]["profiles"] if p["id"] == "llm-evidence")
    assert profile_view["api_key"] == SECRET_MASK
    assert "sk-evidence-secret" not in json.dumps(catalog)
    settings = asyncio.run(client.app.state.settings_store.get())
    assert settings.profiles["llm"]["llm-evidence"].api_key == "sk-evidence-secret"

    # the reference extras survive an editor PUT that omits `extra`
    resp = client.put("/api/profiles/tts/clone", json={"display_name": "Clara EDITED-2"}, headers=AUTH)
    assert resp.status_code == 200
    settings = asyncio.run(client.app.state.settings_store.get())
    row = settings.profiles["tts"]["clone"]
    assert row.extra["ref_audio"] == EMBEDDED_REF
    assert row.extra["ref_text"] == "embedded transcript"
    assert mask_profile(row)["extra"]["ref_audio"] == SECRET_MASK  # masked in views
