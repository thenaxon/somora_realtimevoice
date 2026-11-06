"""Server-side /api/tts-synthesis + voice-reference (ref_audio/ref_text) masking.

The controlled TTS comparison synthesizes DIRECTLY from one profile without a
WS session. Credentials and voice-reference material must stay server-side:
they are masked in all browser-visible views, preserved on masked updates,
and still sent for real to the backend request.
"""

import base64
import json
import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.audio import pcm16_to_wav
from realtimevoice.backends.openai_http import build_backends
from realtimevoice.backends.base import Backends
from realtimevoice.config import SECRET_MASK, BackendProfile, Settings, SettingsStore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from tts_comparison import parse_profile_spec, redact  # noqa: E402

REF_AUDIO_B64 = base64.b64encode(pcm16_to_wav(b"\x01\x00" * 2000, 22050)).decode()
SPEECH_WAV = pcm16_to_wav(bytes([0, 100, 55, 200] * 600), 24000)


class TTSCapture:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def handler(self, request: httpx.Request):
        self.calls.append({"path": request.url.path, "content": request.read()})
        if self.fail:
            return httpx.Response(500, json={"error": {"message": "boom with sk-testkey-123456 inside"}})
        return httpx.Response(200, content=SPEECH_WAV, headers={"content-type": "audio/wav"})

    def last_json(self):
        assert self.calls, "no request captured"
        return json.loads(self.calls[-1]["content"].decode("utf-8"))


@pytest.fixture
def harness(tmp_path):
    capture = TTSCapture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    settings = Settings()
    settings.session.voice = "ryan"
    settings.profiles["tts"]["tts-alt"] = BackendProfile(
        id="tts-alt",
        role="tts",
        base_url="http://alt-capture.test/v1",
        model="alt-tts-model",
        voice="serena",
        capabilities={"voices": ["serena", "ryan"]},
    )
    settings.profiles["tts"]["tts-ref"] = BackendProfile(
        id="tts-ref",
        role="tts",
        base_url="http://ref-capture.test/v1",
        model="chatterbox-fp16",
        language_format="name",
        extra={
            "ref_audio": REF_AUDIO_B64,
            "ref_text": "Dies ist das Referenztranskript der Jonas-Stimme.",
            "top_k": 0,
        },
        capabilities={"voices": [], "languages": ["de"]},
    )
    app = create_app(
        settings_store=SettingsStore(data_dir=str(tmp_path), settings=settings),
        backend_factory=lambda s, http_client=None: Backends(
            *build_backends(s.stt, s.llm, s.tts, client)
        ),
        auth_token="tok",
    )
    with TestClient(app) as test_client:
        # the synthesis route uses app.state.http_client (created at startup);
        # point it at the capturing transport client
        app.state.http_client = client
        yield test_client, capture, tmp_path


def auth():
    return {"Authorization": "Bearer tok"}


def test_tts_synthesis_success_proves_model_voice(harness):
    test_client, capture, _ = harness
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Hallo Welt, das ist ein Test.", "text_id": "short"},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.content[:4] == b"RIFF"
    assert resp.headers["X-Rtv-Model"] == "alt-tts-model"
    assert resp.headers["X-Rtv-Voice"] == "serena"  # profile voice
    assert resp.headers["X-Rtv-Text-Id"] == "short"
    body = capture.last_json()
    assert body["model"] == "alt-tts-model"
    assert body["voice"] == "serena"
    assert body["input"] == "Hallo Welt, das ist ein Test."
    assert body["stream"] is True


def test_tts_synthesis_voice_override_and_validation(harness):
    test_client, _, _ = harness
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Kurzer Text.", "voice": "ryan"},
        headers=auth(),
    )
    assert resp.status_code == 200
    assert resp.headers["X-Rtv-Voice"] == "ryan"
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Kurzer Text.", "voice": "whisper-guy"},
        headers=auth(),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "voice_mismatch"


def test_tts_synthesis_language_sent_in_the_models_format(harness):
    """The optional language is sent as each model wants it: a code for a
    "code" model, the language name for a "name" model; none when absent."""
    test_client, capture, _ = harness
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Hallo.", "language": "de-AT"},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert capture.last_json()["language"] == "de"
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-ref", "text": "Hallo.", "language": "de"},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert capture.last_json()["language"] == "German"
    test_client.post("/api/tts-synthesis", json={"profile_id": "tts-alt", "text": "Hallo."}, headers=auth())
    assert "language" not in capture.last_json()


def test_tts_synthesis_input_validation(harness):
    test_client, _, _ = harness
    assert (
        test_client.post(
            "/api/tts-synthesis", json={"profile_id": "nope", "text": "x"}, headers=auth()
        ).status_code
        == 404
    )
    assert (
        test_client.post(
            "/api/tts-synthesis", json={"profile_id": "tts-alt", "text": "  "}, headers=auth()
        ).status_code
        == 422
    )
    assert (
        test_client.post(
            "/api/tts-synthesis",
            json={"profile_id": "tts-alt", "text": "x" * 1001},
            headers=auth(),
        ).status_code
        == 422
    )


def test_tts_synthesis_backend_error_redacted(harness):
    test_client, capture, _ = harness
    capture.fail = True
    resp = test_client.post(
        "/api/tts-synthesis", json={"profile_id": "tts-alt", "text": "Hallo."}, headers=auth()
    )
    assert resp.status_code == 502
    assert resp.json()["code"] == "tts_backend_error"
    assert "sk-testkey-123456" not in resp.text


def test_ref_audio_ref_text_masked_in_views_but_sent_to_backend(harness):
    test_client, capture, _ = harness
    # synthesis: real values must reach the backend request
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-ref", "text": "Guten Tag.", "text_id": "medium"},
        headers=auth(),
    )
    assert resp.status_code == 200
    body = capture.last_json()
    assert body["ref_audio"] == REF_AUDIO_B64
    assert body["ref_text"].startswith("Dies ist das Referenztranskript")
    # catalog view: masked, never the base64 blob or the transcript
    catalog = test_client.get("/api/catalog", headers=auth()).json()
    profile = next(p for p in catalog["roles"]["tts"]["profiles"] if p["id"] == "tts-ref")
    assert profile["extra"]["ref_audio"] == SECRET_MASK
    assert profile["extra"]["ref_text"] == SECRET_MASK
    assert REF_AUDIO_B64[:20] not in json.dumps(catalog)
    assert "Referenztranskript" not in json.dumps(catalog)
    # masked update keeps the stored reference material
    resp = test_client.put(
        "/api/profiles/tts/tts-ref",
        json={"extra": {"ref_audio": SECRET_MASK, "ref_text": SECRET_MASK, "top_k": 1}},
        headers=auth(),
    )
    assert resp.status_code == 200
    stored = json.loads((harness[2] / "settings.json").read_text())
    stored_extra = stored["profiles"]["tts"]["tts-ref"]["extra"]
    assert stored_extra["ref_audio"] == REF_AUDIO_B64
    assert stored_extra["ref_text"].startswith("Dies ist das Referenztranskript")
    assert stored_extra["top_k"] == 1
    # next synthesis still sends the preserved reference material
    test_client.post(
        "/api/tts-synthesis", json={"profile_id": "tts-ref", "text": "Nochmal."}, headers=auth()
    )
    body = capture.last_json()
    assert body["ref_audio"] == REF_AUDIO_B64


def test_runner_helpers():
    assert parse_profile_spec("tts-qwen") == ("tts-qwen", "")
    assert parse_profile_spec("tts-qwen:ryan") == ("tts-qwen", "ryan")
    text = "error Bearer abc.def.ghi and sk-abcdefghijklmnop and ref_audio: base64data"
    out = redact(text, "instance-token")
    assert "abc.def.ghi" not in out
    assert "sk-abcdefghijklmnop" not in out
    assert "base64data" not in out
    assert "instance-token" not in out


# ------------------------------------------------- endpoint review hardening


def test_tts_synthesis_text_id_bounded_safe_ascii(harness):
    test_client, _, _ = harness
    ok = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Hallo.", "text_id": "short.run-01"},
        headers=auth(),
    )
    assert ok.status_code == 200
    assert ok.headers["X-Rtv-Text-Id"] == "short.run-01"
    for bad in ("a\r\nX-Injected: 1", "pfad/ü", "with space", "x" * 65, ""):
        if bad == "":
            continue  # empty text_id is allowed (optional field)
        resp = test_client.post(
            "/api/tts-synthesis",
            json={"profile_id": "tts-alt", "text": "Hallo.", "text_id": bad},
            headers=auth(),
        )
        assert resp.status_code == 422, bad
        assert resp.status_code == 422


def test_tts_synthesis_header_metadata_never_breaks_response(harness, tmp_path):
    test_client, capture, _ = harness
    # profile with unicode/CRLF model + voice straight from (hypothetical)
    # stored settings: header emission must sanitize, never 500
    from realtimevoice.config import BackendProfile

    test_client.app.state.settings_store._settings.profiles["tts"]["tts-weird"] = BackendProfile(
        id="tts-weird",
        role="tts",
        base_url="http://weird-capture.test/v1",
        model="modél\r\nX-Injected: yes",
        voice="stïmme\ndefault",
    )
    resp = test_client.post(
        "/api/tts-synthesis", json={"profile_id": "tts-weird", "text": "Hallo."}, headers=auth()
    )
    assert resp.status_code == 200, resp.text
    assert "\r" not in resp.headers["X-Rtv-Model"] and "\n" not in resp.headers["X-Rtv-Model"]
    # CRLF (the actual injection vector) is neutralized to '?'; the remaining
    # text stays a single header VALUE and cannot split the header block
    assert resp.headers["X-Rtv-Model"] == "mod?l??X-Injected: yes"
    assert resp.headers["X-Rtv-Voice"] == "st?mme?default"


def test_tts_synthesis_first_pcm_is_backend_ttfa_not_buffer_total(harness, monkeypatch):
    """Empty chunks must not count; the timestamp is taken at the first
    nonempty adapter PCM (PCM decoder boundary), not after full buffering."""
    import asyncio

    import realtimevoice.routes_catalog as routes_catalog

    class SlowChunkedTTS:
        def __init__(self, settings, client):
            pass

        async def stream_speech(self, text, *, voice=None, language=None):
            yield b""  # empty adapter chunk: not PCM, not TTFA
            await asyncio.sleep(0.08)
            yield b"\x00\x01" * 2400
            await asyncio.sleep(0.08)
            yield b"\x00\x02" * 2400

    monkeypatch.setattr(routes_catalog, "OpenAIHTTPTTS", SlowChunkedTTS)
    test_client, _, _ = harness
    resp = test_client.post(
        "/api/tts-synthesis",
        json={"profile_id": "tts-alt", "text": "Hallo Welt.", "text_id": "medium"},
        headers=auth(),
    )
    assert resp.status_code == 200
    first_pcm = int(resp.headers["X-Rtv-First-Pcm-Ms"])
    synth = int(resp.headers["X-Rtv-Synth-Ms"])
    assert first_pcm >= 50, "first nonempty PCM must be measured, not the buffered end"
    assert first_pcm < synth, "TTFA must precede total synthesis time"


def test_tts_synthesis_unexpected_exception_is_sanitized(harness, monkeypatch):
    import realtimevoice.routes_catalog as routes_catalog

    class LeakyTTS:
        def __init__(self, settings, client):
            pass

        async def stream_speech(self, text, *, voice=None, language=None):
            raise RuntimeError(
                "connect to http://db.internal:9999/v1 failed with "
                "api_key=sk-supersecret123456 bearer REALTOKEN base64 "
                "QUJDREVGR0hJSktMTU5PUUJDREVGR0hJSktMTU5PUQ=="
            )
            yield b""  # pragma: no cover

    monkeypatch.setattr(routes_catalog, "OpenAIHTTPTTS", LeakyTTS)
    test_client, _, _ = harness
    resp = test_client.post(
        "/api/tts-synthesis", json={"profile_id": "tts-alt", "text": "Hallo."}, headers=auth()
    )
    assert resp.status_code == 502
    body = resp.json()
    assert body["code"] == "tts_synthesis_failed"
    assert "sk-supersecret123456" not in body["detail"]
    assert "REALTOKEN" not in body["detail"]
    assert "db.internal" not in body["detail"]
    assert "QUJDREVGR0hJSktMTU5PUQ" not in body["detail"]
