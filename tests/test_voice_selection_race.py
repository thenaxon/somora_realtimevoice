"""Voice switch race (bug1d795917): same-model reference proof.

Field report: switching the voice, then Stop -> Talk, the FIRST output used
the old voice; a second Stop/Talk was correct. Both voices speak through ONE
speaking model (chatterbox-fp16) and differ ONLY in the reference recording
(ref_audio/ref_text) — so model-level proof is insufficient.

These regressions prove, at the EFFECTIVE REQUEST level (real adapter +
recording HTTP transport, payload bytes captured):
- after the set is saved with the new voice while a call runs, Stop, Talk:
  the first TTS request of the new call carries the NEW recording's
  fingerprint, once and twice (repeated Stop/Talk), while the running call
  kept the old one,
- the fingerprint convention is SHA256 over the LITERAL extra["ref_audio"]
  string (fixed 2026-09-12),
- session.created (realtimevoice block) and the tts_started lifecycle log
  carry that fingerprint — same-model calls are distinguishable — and never
  the reference data itself,
- an `extra` key that overrides a payload field is logged honestly
  (session_voice vs payload_voice) and the fingerprint still matches what
  the backend received.

Recordings below are synthetic fixtures (no private runtime data).
"""

import base64
import hashlib
import io
import json
import logging
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_backends import tone_pcm
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
    """Records every backend request (method, path, JSON payload)."""

    def __init__(self, requests: list):
        self.requests = requests
        super().__init__(handler=self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, payload))
        if request.url.path.endswith("/chat/completions"):
            sse = (
                'data: {"choices":[{"delta":{"content":"Guten Tag, kurzer Satz."}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(tone_pcm(120)))
        return httpx.Response(200, json={"data": []})


def fingerprint_of(payload: dict) -> str:
    """SHA256 of the LITERAL ref_audio string the backend actually received."""
    raw = payload.get("ref_audio")
    if not raw:
        return ""
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


def _fp(data_dir, voice) -> str:
    sample = voice.samples[0]
    raw = SampleStore(data_dir).read(voice.id, sample.sha256)
    return reference_fingerprint({"ref_audio": base64.b64encode(raw).decode()})


@pytest.fixture
def env(tmp_path):
    data_dir = str(tmp_path)
    s = Settings()
    for role in ("stt", "llm"):
        s.profiles[role][f"{role}-a"] = BackendProfile(
            id=f"{role}-a", role=role, base_url="http://backend.test/v1", model=f"{role}-model",
        )
    s.profiles["tts"]["clone"] = BackendProfile(
        id="clone", role="tts", base_url="http://backend.test/v1",
        model="chatterbox-fp16",  # ONE model for both voices — the field trap
        voice="cloned", extra={"ref_audio": EMBEDDED_REF, "temperature": 0.8},
        capabilities={"voice_cloning": "reference_audio"},
    )
    jonas = add_library_voice(s, data_dir, "jonas", ms=2000, transcript="Ich bin Jonas")
    clara = add_library_voice(s, data_dir, "clara", ms=2500, transcript="Ich bin Clara")
    fps = {"jonas": _fp(data_dir, jonas), "clara": _fp(data_dir, clara), "none": ""}
    assert fps["jonas"] != fps["clara"]  # same model, different recordings must differ
    add_set(s, "agent", enabled=True, voice=VoiceSelection(kind="library", voice_id="jonas"))
    add_set(s, "jonas", enabled=True, voice=VoiceSelection(kind="library", voice_id="jonas"))
    add_set(s, "clara", enabled=True, voice=VoiceSelection(kind="library", voice_id="clara"))

    requests: list = []

    def factory(settings, client=None):
        stt, llm, tts = build_backends(
            settings.stt, settings.llm, settings.tts,
            httpx.AsyncClient(transport=RecordingTransport(requests), base_url="http://backend.test"),
        )
        return Backends(stt=stt, llm=llm, tts=tts)

    app = create_app(
        settings_store=SettingsStore(data_dir=data_dir, settings=s),
        backend_factory=factory,
        auth_token="tok",
    )
    with TestClient(app) as client:
        yield client, requests, fps


def tts_payloads(requests):
    return [p for _, path, p in requests if path.endswith("/audio/speech")]


def run_response(ws):
    ws.send_json({"type": "response.create", "response": {"instructions": "Sag kurz Hallo."}})
    while True:
        msg = ws.receive_json()
        if msg["type"] == "response.done":
            return msg["response"]["status"]


def save_voice(client, set_id, voice):
    resp = client.put(
        f"/api/conversation-profiles/{set_id}",
        json={"spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "clone",
                       "language": "de", "voice": voice}},
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    # the set view carries ids only, never reference material
    assert "ref_audio" not in resp.text and "ref_text" not in resp.text


def connect(client, model):
    return client.websocket_connect(
        f"/v1/realtime?model={model}", subprotocols=["rtv.bearer", "rtv.bearer.tok"]
    )


@pytest.mark.parametrize(
    "first_voice,first_key",
    [({"kind": "none"}, "none"), ({"kind": "library", "voice_id": "jonas"}, "jonas")],
    ids=["no-voice-to-clara", "jonas-to-clara"],
)
def test_save_stop_talk_first_tts_uses_new_reference_once_and_twice(env, first_voice, first_key):
    """The field case at protocol level: talk on the old voice, save the set
    with the new voice while the call is live, Stop, Talk — the FIRST TTS of
    the new call must carry the Clara recording; a second Stop/Talk must stay
    on it (no silent revert), despite the identical model id."""
    client, requests, fps = env
    save_voice(client, "agent", first_voice)
    with connect(client, "profile:agent") as ws:
        info = ws.receive_json()["session"]["realtimevoice"]
        assert info["tts"].get("ref_sha256", "") == fps[first_key]
        assert run_response(ws) == "completed"
        save_voice(client, "agent", {"kind": "library", "voice_id": "clara"})
        # the running call keeps its voice
        assert run_response(ws) == "completed"
    for _ in range(2):
        with connect(client, "profile:agent") as ws:
            created = ws.receive_json()
            info = created["session"]["realtimevoice"]
            assert info["tts"]["profile"] == "clone"
            assert info["tts"]["model"] == "chatterbox-fp16"
            assert info["tts"]["ref_sha256"] == fps["clara"]
            assert "ref_audio" not in json.dumps(created)
            assert run_response(ws) == "completed"

    payloads = tts_payloads(requests)
    assert len(payloads) == 4
    assert fingerprint_of(payloads[0]) == fps[first_key]
    assert fingerprint_of(payloads[1]) == fps[first_key]
    for payload in payloads[2:]:
        assert payload["model"] == "chatterbox-fp16"
        assert fingerprint_of(payload) == fps["clara"]
        assert payload["ref_text"] == "Ich bin Clara"
    # the entry's own embedded reference never reaches the backend
    assert all(p.get("ref_audio") != EMBEDDED_REF for p in payloads)


def _capture_lifecycle():
    records: list[str] = []
    logging.getLogger().setLevel(logging.INFO)
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logging.getLogger("realtimevoice.lifecycle").addHandler(handler)
    return records, handler


def test_tts_started_lifecycle_carries_profile_voice_model_fingerprint(env):
    """Metadata diagnostics: the first TTS of a call logs WHICH model entry,
    effective voice, model and reference it uses (allowlist fields
    tts_profile/session_voice/payload_voice/model/ref_sha256) — never the
    reference data."""
    client, _, fps = env
    records, handler = _capture_lifecycle()
    try:
        for set_id in ("jonas", "clara"):
            with connect(client, f"profile:{set_id}") as ws:
                ws.receive_json()
                assert run_response(ws) == "completed"
    finally:
        logging.getLogger("realtimevoice.lifecycle").removeHandler(handler)

    started = [json.loads(r) for r in records if '"action":"tts_started"' in r]
    assert len(started) == 2
    first, second = started
    for entry, key in ((first, "jonas"), (second, "clara")):
        assert entry["tts_profile"] == "clone"
        assert entry["session_voice"] == "cloned"
        assert entry["payload_voice"] == "cloned"
        assert entry["model"] == "chatterbox-fp16"
        assert entry["ref_sha256"] == fps[key]
        assert "voice" not in entry  # no ambiguous field
    blob = " ".join(records)
    assert EMBEDDED_REF not in blob
    assert "Ich bin" not in blob  # reference transcript never reaches logs


def test_fingerprint_equals_actual_payload_after_extra_overrides(env):
    """The fingerprint must match the ref_audio the backend ACTUALLY received:
    after the entry's extra gains a key that overrides a base payload field
    (and a new embedded reference that a library voice ignores),
    session.created, the tts_started log and the HTTP payload agree."""
    client, requests, fps = env
    resp = client.put(
        "/api/profiles/tts/clone",
        json={"extra": {"ref_audio": base64.b64encode(b"RIFF-other").decode(), "voice": "payload-override"}},
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    records, handler = _capture_lifecycle()
    try:
        with connect(client, "profile:clara") as ws:
            created = ws.receive_json()
            assert created["session"]["realtimevoice"]["tts"]["ref_sha256"] == fps["clara"]
            assert run_response(ws) == "completed"
    finally:
        logging.getLogger("realtimevoice.lifecycle").removeHandler(handler)

    payload = tts_payloads(requests)[-1]
    assert payload["voice"] == "payload-override"
    assert fingerprint_of(payload) == fps["clara"]
    started = [json.loads(r) for r in records if '"action":"tts_started"' in r]
    assert started and started[-1]["ref_sha256"] == fps["clara"]
    # honest voice semantics: session_voice is what the adapter was asked
    # with, payload_voice what the request carried after the extra override
    assert started[-1]["session_voice"] == "cloned"
    assert started[-1]["payload_voice"] == "payload-override"


def test_fingerprint_convention_is_literal_string_not_decoded_audio():
    """Fingerprint convention: SHA256 over the literal extra["ref_audio"] string."""
    value = base64.b64encode(b"RIFFsynthetic-clara-reference").decode()
    assert reference_fingerprint({"ref_audio": value}) == hashlib.sha256(value.encode("utf-8")).hexdigest()
    assert reference_fingerprint({}) == ""
    assert reference_fingerprint(None) == ""
    assert reference_fingerprint({"ref_audio": ""}) == ""
