"""A2: set & voice REST API (CRUD, save-is-live with versions and restore,
the switched-on-set validity rule, samples, transcription, preview,
voice-library settings).

Synthetic audio only; capturing fake HTTP backends — never a real endpoint.
"""

import base64
import io
import json
import math
import time
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import (
    SECRET_MASK,
    BackendProfile,
    Settings,
    SettingsStore,
    Voice,
)
from realtimevoice.profiles import Base64LRU
from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample

AUTH = {"Authorization": "Bearer tok"}
SPEECH = b"\x01\x02" * 600


def wav_bytes(pcm: bytes, rate=24000, channels=1, sampwidth=2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def sine_pcm(ms=1500, rate=24000) -> bytes:
    n = int(rate * ms / 1000)
    return (8000 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2").tobytes()


class Capture:
    def __init__(self):
        self.calls = []
        self.transcript = "Hallo Welt"
        self.transcribe_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append({"path": path, "content": request.read()})
        if path.endswith("/audio/transcriptions"):
            if self.transcribe_status != 200:
                return httpx.Response(self.transcribe_status, json={"error": {"message": "boom"}})
            return httpx.Response(200, json={"text": self.transcript})
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=wav_bytes(SPEECH))
        if path.endswith("/models"):
            # P1: the readiness probe verifies the configured model id is
            # listed; the fake advertises every model this fixture uses
            return httpx.Response(200, json={"data": [
                {"id": "x"},
                {"id": "stt-alt"}, {"id": "llm-alt"}, {"id": "tts-alt"},
                {"id": "stt-base"}, {"id": "llm-base"}, {"id": "tts-base"},
            ]})
        if path.endswith("/chat/completions"):
            return httpx.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"error": {"message": "nf"}})


def make_settings() -> Settings:
    s = Settings()
    for role in ("stt", "llm", "tts"):
        s.profiles[role]["base"] = BackendProfile(
            id="base", role=role, base_url="http://cap.test/v1", model=f"{role}-base"
        )
        s.profiles[role]["alt"] = BackendProfile(
            id="alt", role=role, base_url="http://cap.test/v1", model=f"{role}-alt"
        )
    # recordings are transcribed with the base listening model
    s.voice_library.transcription_stt_profile = "base"
    s.profiles["tts"]["alt"].capabilities.voice_cloning = "reference_audio"
    s.profiles["tts"]["alt"].voice = "alt-voice"
    s.profiles["tts"]["alt"].extra = {"ref_audio": "OLDSECRETREF", "ref_text": "old", "temperature": 0.5}
    return s


@pytest.fixture
def env(tmp_path):
    capture = Capture()
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(
        settings_store=store,
        backend_factory=lambda s, c=None: None,
        auth_token="tok",
    )
    with TestClient(app) as tc:
        # startup creates its own client; swap in the capturing one
        tc.app.state.http_client = client
        yield tc, capture, tmp_path


def _upload(tc, voice_id, pcm=None):
    pcm = pcm or sine_pcm()
    return tc.post(
        f"/api/voices/{voice_id}/samples",
        headers=AUTH,
        files={"file": ("ref.wav", wav_bytes(pcm), "audio/wav")},
        data={"source": "upload"},
    )


def _setup_voice(tc, vid="anna", confirmed=True):
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": "Anna"}).status_code == 201
    up = _upload(tc, vid)
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    # transcribe (fake returns text)
    tr = tc.post(f"/api/voices/{vid}/samples/{sid}/transcribe", headers=AUTH, json={})
    assert tr.status_code == 200, tr.text
    if confirmed:
        patched = tc.patch(
            f"/api/voices/{vid}/samples/{sid}",
            headers=AUTH,
            json={"transcript": "Hallo Welt", "transcript_confirmed": True},
        )
        assert patched.status_code == 200
    return sid


def _spec(voice_id="anna", **over):
    spec = {
        "stt_profile": "alt",
        "llm_profile": "alt",
        "tts_profile": "alt",
        "language": "de",
        "voice": {"kind": "library", "voice_id": voice_id},
    }
    spec.update(over)
    return spec


def _make_profile(tc, pid="anna-agent", check=True, *, enabled=False, spec=None):
    body = {"id": pid, "display_name": "Anna", "spec": _spec() if spec is None else spec, "enabled": enabled}
    r = tc.post("/api/conversation-profiles", headers=AUTH, json=body)
    if check:
        assert r.status_code == 201, r.text
    return r.json() if check else r


def _view(tc, pid="anna-agent"):
    r = tc.get(f"/api/conversation-profiles/{pid}", headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()


def _failing_models(cap, body=None, status=500):
    """A transport whose /models answers `status` (or a list without the
    configured models); everything else goes to the capture."""
    def handler(request):
        if request.url.path.endswith("/models"):
            if body is not None:
                return httpx.Response(200, json=body)
            return httpx.Response(status, json={"error": {"message": "nope"}})
        return cap.handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- CRUD

def test_set_crud_and_conflicts(env):
    tc, cap, _ = env
    # bad id
    assert tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "tmp_x"}).status_code == 422
    assert tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "Anna"}).status_code == 422
    v = _make_profile(tc)
    assert v["id"] == "anna-agent" and v["enabled"] is False
    assert v["history"] == [] and v["is_default"] is False
    assert v["spec"]["tts_profile"] == "alt"
    # the view says what would stop the set from working (no voice yet)
    assert any(i["code"] == "missing_reference" for i in v["issues"])
    assert "compatibility_warnings" in v
    # none of the removed fields come back
    for gone in ("draft", "active", "available_to_clients", "backend_changes"):
        assert gone not in v
    # duplicate
    r = _make_profile(tc, check=False)
    assert r.status_code == 409 and r.json()["code"] == "conversation_profile_exists"
    # update
    r = tc.put(
        "/api/conversation-profiles/anna-agent",
        headers=AUTH,
        json={"display_name": "Anna 2", "description": "d", "spec": {"stt_profile": "alt", "language": "de"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["display_name"] == "Anna 2" and body["description"] == "d"
    assert body["spec"]["tts_profile"] == ""
    assert "issues" in body and "compatibility_warnings" in body
    # delete a switched-off set
    assert tc.delete("/api/conversation-profiles/anna-agent", headers=AUTH).status_code == 200
    assert tc.get("/api/conversation-profiles/anna-agent", headers=AUTH).status_code == 404


def test_a_switched_off_set_can_be_saved_incomplete(env):
    tc, cap, _ = env
    assert _make_profile(tc, spec={})["enabled"] is False
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"spec": {"language": "de"}})
    assert r.status_code == 200, r.text
    codes = {i["code"] for i in r.json()["issues"]}
    assert "missing" in codes  # reported, not refused
    assert _view(tc)["spec"]["language"] == "de"


def test_a_switched_on_set_must_be_valid(env):
    tc, cap, _ = env
    # creating it switched on with an incomplete spec is refused
    r = _make_profile(tc, check=False, enabled=True, spec={"language": "de"})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "set_invalid" and r.json()["issues"]
    assert tc.get("/api/conversation-profiles/anna-agent", headers=AUTH).status_code == 404

    # an incomplete switched-off set cannot be switched on
    _make_profile(tc, spec={"language": "de"})
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"enabled": True})
    assert r.status_code == 409 and r.json()["code"] == "set_invalid"
    assert _view(tc)["enabled"] is False

    # a valid set is switched on; saving it broken while on is refused and
    # leaves what calls use untouched
    _setup_voice(tc)
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"spec": _spec(), "enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True and r.json()["issues"] == []
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH,
               json={"spec": _spec(tts_profile="")})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "set_invalid"
    assert any(i["path"] == "tts_profile" for i in r.json()["issues"])
    view = _view(tc)
    assert view["spec"]["tts_profile"] == "alt" and view["enabled"] is True
    # switching it off and saving the same incomplete spec is fine
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH,
               json={"spec": _spec(tts_profile=""), "enabled": False})
    assert r.status_code == 200, r.text


def test_switching_on_with_a_failing_probe_still_saves_and_reports_it(env):
    """Probes are reported, never blocking: an offline model server must not
    prevent saving a set."""
    tc, cap, _ = env
    _setup_voice(tc)
    _make_profile(tc)
    tc.app.state.http_client = _failing_models(cap)
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is True
    assert body["probes"]["tts"]["ok"] is False
    assert _view(tc)["enabled"] is True


def test_model_not_listed_is_reported_not_blocking(env):
    """P1: /models answers with a list that omits the configured model id ->
    the probe fails with model_not_listed, in validate and on save alike."""
    tc, cap, _ = env
    _setup_voice(tc)
    _make_profile(tc)
    tc.app.state.http_client = _failing_models(cap, body={"data": [{"id": "some-other-model"}]})
    val = tc.post("/api/conversation-profiles/anna-agent/validate", headers=AUTH).json()
    assert val["ok"] is True
    assert val["probes"]["tts"]["ok"] is False
    assert val["probes"]["tts"]["error_code"] == "model_not_listed"
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["probes"]["tts"]["error_code"] == "model_not_listed"


def test_switching_on_reports_passing_probes(env):
    """P1 control: the fixture /models lists the configured models -> every
    probe passes (guards against the probe becoming over-strict)."""
    tc, cap, _ = env
    _setup_voice(tc)
    r = _make_profile(tc, enabled=True)
    assert r["enabled"] is True
    assert all(p["ok"] for p in r["probes"].values()), r["probes"]


def test_validate_checks_an_unsaved_spec(env):
    tc, cap, _ = env
    _setup_voice(tc)
    _make_profile(tc, spec={})
    assert tc.post("/api/conversation-profiles/anna-agent/validate", headers=AUTH).json()["ok"] is False
    r = tc.post("/api/conversation-profiles/anna-agent/validate", headers=AUTH, json={"spec": _spec()})
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["issues"] == []
    assert _view(tc)["spec"]["tts_profile"] == "", "validate never saves"


def test_save_keeps_earlier_versions_and_restore_brings_one_back(env):
    tc, cap, _ = env
    _setup_voice(tc)
    _make_profile(tc, enabled=True)
    # saving without a spec change adds no version
    tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"display_name": "Anna"})
    assert _view(tc)["history"] == []
    r = tc.put("/api/conversation-profiles/anna-agent", headers=AUTH, json={"spec": _spec(language="en")})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["spec"]["language"] == "en"
    assert [h["spec"]["language"] for h in view["history"]] == ["de"]
    assert "saved_at" in view["history"][0]

    roll = tc.post("/api/conversation-profiles/anna-agent/restore", headers=AUTH, json={"index": 0})
    assert roll.status_code == 200, roll.text
    assert roll.json()["spec"]["language"] == "de"
    # the replaced spec is a version itself: restore is reversible
    assert [h["spec"]["language"] for h in roll.json()["history"]] == ["en"]
    assert roll.json()["enabled"] is True

    r = tc.post("/api/conversation-profiles/anna-agent/restore", headers=AUTH, json={"index": 5})
    assert r.status_code == 409 and r.json()["code"] == "no_history"

    # at most 10 versions, newest first
    for i in range(12):
        tc.put("/api/conversation-profiles/anna-agent", headers=AUTH,
               json={"spec": _spec(instructions_default=f"v{i}")})
    history = _view(tc)["history"]
    assert len(history) == 10
    assert history[0]["spec"]["instructions_default"] == "v10"


def test_delete_enabled_set_and_voice_reference_protections(env):
    tc, cap, _ = env
    sid = _setup_voice(tc)
    _make_profile(tc, enabled=True)
    r = tc.delete("/api/conversation-profiles/anna-agent", headers=AUTH)
    assert r.status_code == 409 and r.json()["code"] == "conversation_profile_active"
    r = tc.delete("/api/voices/anna", headers=AUTH)
    assert r.status_code == 409 and r.json()["code"] == "voice_referenced"
    # the recording a set speaks with cannot be deleted
    r = tc.delete(f"/api/voices/anna/samples/{sid}", headers=AUTH)
    assert r.status_code == 409 and r.json()["code"] == "sample_is_reference"


def test_voice_view_names_the_sets_that_use_it(env):
    tc, cap, _ = env
    _setup_voice(tc)
    _make_profile(tc, enabled=True)
    _make_profile(tc, "anna-off")
    used = tc.get("/api/voices/anna", headers=AUTH).json()["used_by"]
    assert used == [
        {"profile": "anna-agent", "display_name": "Anna", "enabled": True},
        {"profile": "anna-off", "display_name": "Anna", "enabled": False},
    ]
    listed = tc.get("/api/voices", headers=AUTH).json()["voices"][0]
    assert listed["used_by"] == used
    for gone in ("live_sets",):
        assert gone not in listed


def test_set_view_marks_the_set_for_other_apps(env):
    tc, cap, _ = env
    _make_profile(tc, spec={})
    assert tc.put("/api/settings", headers=AUTH, json={"default_set": "anna-agent"}).status_code == 200
    assert _view(tc)["is_default"] is True
    listed = tc.get("/api/conversation-profiles", headers=AUTH).json()["profiles"]
    assert [p["is_default"] for p in listed] == [True]


# ---------------------------------------------------------------- samples

def test_sample_upload_validation_and_audio_read(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    # bad format -> 415
    r = tc.post("/api/voices/anna/samples", headers=AUTH,
                files={"file": ("x.wav", b"not a wav", "audio/wav")}, data={"source": "upload"})
    assert r.status_code == 415 and r.json()["code"] == "unsupported_format"
    # too short -> 400 sample_too_short
    r = tc.post("/api/voices/anna/samples", headers=AUTH,
                files={"file": ("x.wav", wav_bytes(sine_pcm(200)), "audio/wav")}, data={"source": "upload"})
    assert r.status_code == 400 and r.json()["code"] == "sample_too_short"
    # ok
    up = _upload(tc, "anna")
    assert up.status_code == 200
    sid = up.json()["id"]
    assert up.json()["transcription"] == {}  # state none
    audio = tc.get(f"/api/voices/anna/samples/{sid}/audio", headers=AUTH)
    assert audio.status_code == 200
    assert audio.headers["cache-control"] == "no-store"
    assert audio.content[:4] == b"RIFF"
    # auth required
    assert tc.get(f"/api/voices/anna/samples/{sid}/audio").status_code == 401


def test_manual_patch_resets_confirmation_unless_same_request(env):
    tc, cap, _ = env
    sid = _setup_voice(tc, confirmed=False)
    r = tc.patch(f"/api/voices/anna/samples/{sid}", headers=AUTH, json={"transcript": "neu"})
    assert r.json()["transcript"] == "neu"
    assert r.json()["transcript_source"] == "manual"
    assert r.json()["transcript_confirmed"] is False
    r = tc.patch(f"/api/voices/anna/samples/{sid}", headers=AUTH, json={"transcript": "neu2", "transcript_confirmed": True})
    assert r.json()["transcript_confirmed"] is True


def test_reference_selection(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    a = _upload(tc, "anna").json()["id"]
    b = _upload(tc, "anna", pcm=sine_pcm(1600)).json()["id"]
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": b})
    assert r.status_code == 200
    assert r.json()["reference_sample_id"] == b


# ------------------------------------------------------------ transcription

def test_transcribe_success_sets_source_and_unconfirmed(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    cap.transcript = "  language German<asr_text>Hallo Welt  "
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    body = r.json()
    assert r.status_code == 200
    assert body["transcript"] == "Hallo Welt"  # ASR prefix stripped
    assert body["transcript_source"] == "stt" and body["transcript_confirmed"] is False
    assert body["transcription"]["state"] == "done"


def test_transcribe_empty_reports_no_speech(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    cap.transcript = ""
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.status_code == 200
    assert r.json()["transcription"]["state"] == "failed"
    assert r.json()["transcription"]["error_code"] == "no_speech_detected"
    assert r.json()["transcript"] == ""
    # audio kept on failure
    assert tc.get(f"/api/voices/anna/samples/{sid}/audio", headers=AUTH).status_code == 200


def test_transcribe_backend_error_keeps_state(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    cap.transcribe_status = 500
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.status_code == 200
    assert r.json()["transcription"]["state"] == "failed"
    assert r.json()["transcription"]["error_code"] in ("stt_failed", "backend_unreachable", "backend_timeout")


def test_transcribe_needs_a_chosen_transcription_model(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    assert tc.put("/api/voice-library/settings", headers=AUTH,
                  json={"transcription_stt_profile": ""}).status_code == 200
    before = len(cap.calls)
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.status_code == 409 and r.json()["code"] == "no_transcription_model"
    assert cap.calls[before:] == []


def test_transcribe_uses_the_chosen_transcription_model(env):
    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    r = tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert r.json()["transcription"]["stt_profile"] == "base"
    assert r.json()["transcription"]["model"] == "stt-base"
    stt = [c for c in cap.calls if c["path"].endswith("/audio/transcriptions")][-1]
    assert b"stt-base" in stt["content"]


def test_transcribe_logs_no_transcript_text(env, caplog):
    import logging

    tc, cap, _ = env
    tc.post("/api/voices", headers=AUTH, json={"id": "anna", "display_name": "Anna"})
    sid = _upload(tc, "anna").json()["id"]
    cap.transcript = "SECRET_SPOKEN_TEXT"
    with caplog.at_level(logging.DEBUG):
        tc.post(f"/api/voices/anna/samples/{sid}/transcribe", headers=AUTH, json={})
    assert "SECRET_SPOKEN_TEXT" not in caplog.text


# ---------------------------------------------------------------- preview

def test_preview_materialises_and_never_mutates_settings(env):
    tc, cap, _ = env
    _setup_voice(tc)
    before = tc.get("/api/settings", headers=AUTH).json()
    r = tc.post("/api/voices/anna/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Rtv-First-Pcm-Ms"] != ""
    assert "X-Rtv-Total-Ms" in r.headers and "X-Rtv-Audio-Ms" in r.headers
    after = tc.get("/api/settings", headers=AUTH).json()
    assert before == after
    # captured TTS request carries the materialised reference, not the old one
    speech = [c for c in cap.calls if c["path"].endswith("/audio/speech")][-1]
    payload = json.loads(speech["content"])
    assert payload["ref_text"] == "Hallo Welt"
    assert base64.b64decode(payload["ref_audio"])[:4] == b"RIFF"


def test_preview_rejects_unconfirmed_transcript(env):
    tc, cap, _ = env
    sid = _setup_voice(tc, confirmed=False)
    r = tc.post("/api/voices/anna/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 409 and r.json()["code"] == "transcript_not_confirmed"


# ------------------------------------------------------------- settings

def test_voice_library_settings_validated(env):
    tc, cap, _ = env
    assert tc.get("/api/voice-library/settings", headers=AUTH).json() == {"transcription_stt_profile": "base"}
    r = tc.put("/api/voice-library/settings", headers=AUTH, json={"transcription_stt_profile": "alt"})
    assert r.status_code == 200 and r.json()["transcription_stt_profile"] == "alt"
    r = tc.put("/api/voice-library/settings", headers=AUTH, json={"transcription_stt_profile": "nope"})
    assert r.status_code == 400 and r.json()["code"] == "unknown_stt_profile"


def test_all_routes_require_auth(env):
    tc, cap, _ = env
    assert tc.get("/api/conversation-profiles").status_code == 401
    assert tc.get("/api/voices").status_code == 401
    assert tc.get("/api/voice-library/settings").status_code == 401