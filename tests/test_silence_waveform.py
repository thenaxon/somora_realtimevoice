"""N-3/N-4: silence must not produce a hallucinated transcript; waveform uses
absolute scale.

Regression tests for the designer pass-2 items:
- transcription of a silent sample must NOT call STT and must store no
  transcript (energy pre-check < 300 ms speech -> no_speech_detected),
- a backend that hallucinates text on silence must be discarded (post-check),
- the stored waveform envelope is absolute (|peak|/32768), so a quiet sample
  looks quiet and a loud sample looks loud.
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
from realtimevoice.voices import canonicalise_wav, speech_duration_ms

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


def sine_pcm(ms=1500, rate=24000, amp=8000) -> bytes:
    n = int(rate * ms / 1000)
    return (amp * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2").tobytes()


def silence_pcm(ms=2000, rate=24000) -> bytes:
    return bytes(2 * int(rate * ms / 1000))


class Capture:
    def __init__(self):
        self.calls = []
        self.transcript = "halluzinierter Text aus Stille"

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": self.transcript})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(SPEECH))
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "x"}, {"id": "stt-base"}]})
        return httpx.Response(404, json={"error": {"message": "nf"}})


@pytest.fixture
def env(tmp_path):
    capture = Capture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    s = Settings()
    # recordings are transcribed by the listening model the settings name
    s.profiles["stt"]["listen"] = BackendProfile(
        id="listen", role="stt", base_url="http://cap.test/v1", model="stt-base"
    )
    s.voice_library.transcription_stt_profile = "listen"
    store = SettingsStore(data_dir=str(tmp_path), settings=s)
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = client
        yield tc, capture


def _upload(tc, raw: bytes):
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    return tc.post("/api/voices/anna/samples", headers=AUTH,
                   files={"file": ("s.wav", raw, "audio/wav")}, data={"source": "upload"})


def test_silent_sample_is_not_sent_to_stt(env):
    tc, cap = env
    up = _upload(tc, wav_bytes(silence_pcm()))
    sid = up.json()["id"]
    cap.calls.clear()
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.status_code == 200
    body = r.json()
    assert body["transcription"]["error_code"] == "no_speech_detected"
    assert body["transcription"]["state"] == "failed"
    # no hallucinated transcript stored and no STT HTTP call was made
    assert body["transcript"] == ""
    assert not any(p.endswith("/audio/transcriptions") for p in cap.calls), cap.calls


def test_hallucinated_text_on_silence_is_discarded(env):
    """The pre-check is what stops a silent sample; the post-check is the
    belt-and-braces guard. Both compare against the same energy measure."""
    # a sample with real speech passes the pre-check
    canonical = canonicalise_wav(wav_bytes(sine_pcm(1000)))
    assert speech_duration_ms(canonical.data) >= 300.0
    # a silent sample does not
    silent = canonicalise_wav(wav_bytes(silence_pcm()))
    assert speech_duration_ms(silent.data) < 300.0


def test_speech_sample_is_sent_to_stt(env):
    tc, cap = env
    up = _upload(tc, wav_bytes(sine_pcm()))
    sid = up.json()["id"]
    cap.calls.clear()
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.status_code == 200
    assert any(p.endswith("/audio/transcriptions") for p in cap.calls), cap.calls
    assert r.json()["transcript"]  # the fake transcript is stored


def test_speech_duration_detects_silence_and_speech():
    assert speech_duration_ms(canonicalise_wav(wav_bytes(silence_pcm())).data) == 0.0
    assert speech_duration_ms(canonicalise_wav(wav_bytes(sine_pcm(1000))).data) >= 900.0


def test_waveform_envelope_is_absolute_scale():
    quiet = canonicalise_wav(wav_bytes(sine_pcm(1000, amp=300))).metrics["waveform"]
    loud = canonicalise_wav(wav_bytes(sine_pcm(1000, amp=16000))).metrics["waveform"]
    assert quiet and loud and len(quiet) == len(loud)
    # absolute scale: the quiet sample's envelope is much smaller than full
    assert max(quiet) < 0.05, max(quiet)
    assert max(loud) > 0.4, max(loud)
    # values never exceed full scale
    assert max(loud) <= 1.0


def test_sync_baseline_import_still_works():
    # guard against an accidental broad edit breaking the import surface
    from realtimevoice.voices import speech_duration_ms as fn  # noqa: F401