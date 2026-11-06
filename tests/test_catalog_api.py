"""Models API: model entries, generated ids, the sets that use them.

Regression focus: id generation and validation, masking, deletion rules
(a model a set uses or that transcribes recordings cannot go), a model edit
that would stop a switched-on set from working is refused (409 breaks_sets)
while the same edit with no switched-on set affected goes through, honest
probe metadata, and the removed preset/selection routes.
"""

import asyncio
import io
import json
import math
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fake_backends import make_backends
from realtimevoice.app import create_app
from realtimevoice.config import (
    SECRET_MASK,
    BackendProfile,
    ConversationProfile,
    ProfileSpec,
    Settings,
    SettingsStore,
    Voice,
    VoiceSelection,
)
from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample


def _wav(ms: int, rate: int = 24000) -> bytes:
    n = int(rate * ms / 1000)
    pcm = (8000 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def add_library_voice(
    settings: Settings, data_dir: str, voice_id: str, *, ms: int = 3000,
    transcript: str = "Hallo Welt", wav: bytes | None = None,
) -> Voice:
    """A library voice with one confirmed reference recording, its WAV stored
    where the app's sample store reads it. Shared by other test modules."""
    canonical = canonicalise_wav(wav or _wav(ms))
    sample = new_sample(canonical, source="upload")
    sample.transcript = transcript
    sample.transcript_confirmed = True
    sample.transcript_source = "manual"
    SampleStore(data_dir).write(voice_id, canonical.sha256, canonical.data)
    now = time.time()
    voice = Voice(
        id=voice_id, display_name=voice_id.capitalize(), samples=[sample],
        reference_sample_id=sample.id, created_at=now, updated_at=now,
    )
    settings.voices[voice_id] = voice
    return voice


def add_set(settings: Settings, set_id: str, *, enabled: bool, voice: VoiceSelection, **models) -> None:
    now = time.time()
    spec = ProfileSpec(
        stt_profile=models.get("stt", "stt-a"), llm_profile=models.get("llm", "llm-a"),
        tts_profile=models.get("tts", "clone"), language="de", voice=voice,
    )
    settings.conversation_profiles[set_id] = ConversationProfile(
        id=set_id, display_name=set_id.capitalize(), spec=spec, enabled=enabled,
        created_at=now, updated_at=now,
    )


def make_settings(data_dir: str) -> Settings:
    settings = Settings()
    for role in ("stt", "llm"):
        settings.profiles[role][f"{role}-a"] = BackendProfile(
            id=f"{role}-a", role=role, base_url="http://backend.test/v1", model=f"{role}-model-a",
        )
    settings.profiles["stt"]["stt-b"] = BackendProfile(
        id="stt-b", role="stt", base_url="http://backend.test/v1", model="stt-model-b",
    )
    settings.profiles["tts"]["clone"] = BackendProfile(
        id="clone", role="tts", base_url="http://backend.test/v1", model="clone-model",
    )
    settings.profiles["tts"]["clone"].capabilities.voice_cloning = "reference_audio"
    settings.profiles["tts"]["clone"].capabilities.max_reference_ms = 10_000
    settings.voice_library.transcription_stt_profile = "stt-b"
    add_library_voice(settings, data_dir, "anna", ms=3000)
    library = VoiceSelection(kind="library", voice_id="anna")
    add_set(settings, "anna-on", enabled=True, voice=library)
    return settings


@pytest.fixture
def client(tmp_path):
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings(str(tmp_path)))
    app = create_app(
        settings_store=store,
        backend_factory=lambda s, client=None: make_backends(),
        auth_token="tok",
    )
    with TestClient(app) as test_client:
        yield test_client, tmp_path, app


def auth():
    return {"Authorization": "Bearer tok"}


def create_profile(client, role, pid=None, **extra):
    payload = {"base_url": "http://backend.test/v1", "model": f"{pid or 'gen'}-model", **extra}
    if pid is not None:
        payload["id"] = pid
    return client.post(f"/api/profiles/{role}", json=payload, headers=auth())


def stored(app) -> Settings:
    return asyncio.run(app.state.settings_store.get())


# ---------------------------------------------------------------- entries


def test_profile_crud_roundtrip(client):
    test_client, _, _ = client
    resp = create_profile(test_client, "tts", "tts-alt", voice="serena", capabilities={"voices": ["serena"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "tts-alt"
    assert body["model"] == "tts-alt-model"
    assert body["language_format"] == "code"
    assert "language" not in body
    assert body["capabilities"]["voices"] == ["serena"]
    assert body["capabilities"]["streaming"] == "unknown"
    assert body["capabilities"]["evidence"] == "unverified"
    assert body["used_by"] == []

    # a given id that already exists -> 409 with code
    resp = create_profile(test_client, "tts", "tts-alt")
    assert resp.status_code == 409
    assert resp.json()["code"] == "profile_exists"

    # update partial, masked api_key keeps stored secret
    create_profile(test_client, "llm", "llm-alt", api_key="sk-real-llm-key")
    resp = test_client.put(
        "/api/profiles/llm/llm-alt",
        json={"api_key": SECRET_MASK, "model": "llm-new", "language_format": "name"},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "llm-new"
    assert resp.json()["language_format"] == "name"
    assert resp.json()["api_key"] == SECRET_MASK
    catalog = test_client.get("/api/catalog", headers=auth()).json()
    llm_alt = next(p for p in catalog["roles"]["llm"]["profiles"] if p["id"] == "llm-alt")
    assert llm_alt["api_key"] == SECRET_MASK

    # delete unused works; unknown is 404
    assert test_client.delete("/api/profiles/tts/tts-alt", headers=auth()).status_code == 200
    assert test_client.delete("/api/profiles/llm/never-existed", headers=auth()).status_code == 404


def test_language_format_is_validated(client):
    test_client, _, _ = client
    assert create_profile(test_client, "tts", "t1", language_format="German").status_code == 422
    assert create_profile(test_client, "tts", "t2", language_format="none").json()["language_format"] == "none"


def test_profile_invalid_role_and_bad_ids(client):
    test_client, _, _ = client
    assert test_client.post("/api/profiles/vox", json={"id": "x"}, headers=auth()).status_code == 422
    assert create_profile(test_client, "stt", "UPPER").status_code == 422
    assert create_profile(test_client, "stt", "-dash").status_code == 422


def test_id_generated_from_model_name_with_collision_suffix(client):
    test_client, _, app = client
    payload = {"base_url": "http://backend.test/v1", "model": "Org/Speech Model 2"}
    first = test_client.post("/api/profiles/stt", json=payload, headers=auth())
    second = test_client.post("/api/profiles/stt", json=payload, headers=auth())
    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    assert first.json()["id"] == "org-speech-model-2"
    assert second.json()["id"] == "org-speech-model-2-2"
    # no model name: from the display name
    third = test_client.post("/api/profiles/stt", json={"display_name": "My STT"}, headers=auth())
    assert third.json()["id"] == "my-stt"
    rows = stored(app).profiles["stt"]
    assert {"org-speech-model-2", "org-speech-model-2-2", "my-stt"} <= set(rows)
    assert rows["org-speech-model-2"].role == "stt"


def test_catalog_shape_lists_models_and_the_sets_using_them(client):
    test_client, _, _ = client
    catalog = test_client.get("/api/catalog", headers=auth()).json()
    assert set(catalog) == {"roles"}
    assert set(catalog["roles"]) == {"stt", "llm", "tts"}
    for role in ("stt", "llm", "tts"):
        assert set(catalog["roles"][role]) == {"profiles", "engine_groups"}
    by_id = {p["id"]: p for p in catalog["roles"]["tts"]["profiles"]}
    assert by_id["clone"]["used_by"] == ["anna-on"]
    stt = {p["id"]: p for p in catalog["roles"]["stt"]["profiles"]}
    assert stt["stt-a"]["used_by"] == ["anna-on"]
    assert stt["stt-b"]["used_by"] == []


def test_nested_extra_secrets_masked(client):
    test_client, tmp_path, _ = client
    create_profile(test_client, "stt", "stt-sec", extra={"api_secret": "super-secret", "notes": "fine"})
    catalog = test_client.get("/api/catalog", headers=auth()).json()
    profile = next(p for p in catalog["roles"]["stt"]["profiles"] if p["id"] == "stt-sec")
    assert profile["extra"]["api_secret"] == SECRET_MASK
    assert profile["extra"]["notes"] == "fine"
    resp = test_client.put(
        "/api/profiles/stt/stt-sec",
        json={"extra": {"api_secret": SECRET_MASK, "notes": "changed"}},
        headers=auth(),
    )
    assert resp.status_code == 200
    raw = json.loads((tmp_path / "settings.json").read_text())
    assert raw["profiles"]["stt"]["stt-sec"]["extra"]["api_secret"] == "super-secret"
    assert resp.json()["extra"]["notes"] == "changed"
    assert resp.json()["extra"]["api_secret"] == SECRET_MASK


# ------------------------------------------------------------ delete rules


def test_delete_refused_while_a_set_uses_the_model(client):
    test_client, _, app = client
    resp = test_client.delete("/api/profiles/tts/clone", headers=auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "model_in_use"
    assert "Anna-on" in resp.json()["detail"]
    assert "clone" in stored(app).profiles["tts"]
    # a switched-off set protects its models just the same
    asyncio.run(app.state.settings_store.mutate(
        lambda s: setattr(s.conversation_profiles["anna-on"], "enabled", False)
    ))
    assert test_client.delete("/api/profiles/llm/llm-a", headers=auth()).json()["code"] == "model_in_use"


def test_delete_refused_for_the_transcription_model(client):
    test_client, _, app = client
    resp = test_client.delete("/api/profiles/stt/stt-b", headers=auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "model_in_use"
    assert "stt-b" in stored(app).profiles["stt"]
    # after choosing another transcription model it can go
    assert test_client.put(
        "/api/settings", json={"voice_library": {"transcription_stt_profile": "stt-a"}}, headers=auth()
    ).status_code == 200
    assert test_client.delete("/api/profiles/stt/stt-b", headers=auth()).status_code == 200


# ------------------------------------------------------------ breaks_sets


def test_update_that_breaks_a_switched_on_set_is_refused(client):
    test_client, _, app = client
    before = stored(app).profiles["tts"]["clone"].model_dump()
    resp = test_client.put(
        "/api/profiles/tts/clone",
        json={"capabilities": {"voice_cloning": "none"}},
        headers=auth(),
    )
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "breaks_sets"
    assert body["issues"][0]["profile"] == "anna-on"
    assert body["issues"][0]["display_name"] == "Anna-on"
    assert [i["code"] for i in body["issues"][0]["issues"]] == ["voice_incompatible"]
    assert stored(app).profiles["tts"]["clone"].model_dump() == before


def test_lowering_max_reference_below_the_recording_is_refused(client):
    test_client, _, app = client
    resp = test_client.put(
        "/api/profiles/tts/clone",
        json={"capabilities": {"max_reference_ms": 1000}},  # recording is 3 s
        headers=auth(),
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "breaks_sets"
    assert resp.json()["issues"][0]["issues"][0]["code"] == "reference_too_long"
    assert stored(app).profiles["tts"]["clone"].capabilities.max_reference_ms == 10_000
    # a limit the recording still fits is fine
    resp = test_client.put(
        "/api/profiles/tts/clone", json={"capabilities": {"max_reference_ms": 5000}}, headers=auth()
    )
    assert resp.status_code == 200, resp.text


def test_same_update_goes_through_when_no_switched_on_set_is_affected(client):
    test_client, _, app = client
    asyncio.run(app.state.settings_store.mutate(
        lambda s: setattr(s.conversation_profiles["anna-on"], "enabled", False)
    ))
    resp = test_client.put(
        "/api/profiles/tts/clone",
        json={"capabilities": {"voice_cloning": "none"}},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert stored(app).profiles["tts"]["clone"].capabilities.voice_cloning == "none"


def test_harmless_update_of_a_model_an_on_set_uses_goes_through(client):
    test_client, _, app = client
    resp = test_client.put(
        "/api/profiles/tts/clone", json={"display_name": "Cloner", "model": "clone-model-2"}, headers=auth()
    )
    assert resp.status_code == 200, resp.text
    # the set speaks with the edited model at once (no frozen copy)
    assert stored(app).profiles["tts"]["clone"].model == "clone-model-2"
    assert resp.json()["used_by"] == ["anna-on"]


# ------------------------------------------------------------------ probe


def test_probe_profile_honest_note_and_unknown(client):
    test_client, _, _ = client
    create_profile(test_client, "stt", "stt-alt")
    # unreachable endpoint: the probe must report failure honestly, never
    # claim capability evidence, and always carry the honest note
    resp = test_client.post("/api/probe-profile", json={"role": "stt", "id": "stt-alt"}, headers=auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["detail"]
    assert "not verified" in body["note"]
    resp = test_client.post("/api/probe-profile", json={"role": "stt", "id": "nope"}, headers=auth())
    assert resp.status_code == 404
    resp = test_client.post("/api/probe-profile", json={"role": "nope", "id": "x"}, headers=auth())
    assert resp.status_code == 422
    # no base_url configured: explicit, not a crash
    create_profile(test_client, "tts", "tts-nourl", base_url="")
    resp = test_client.post("/api/probe-profile", json={"role": "tts", "id": "tts-nourl"}, headers=auth())
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert "base_url" in resp.json()["detail"]


# ---------------------------------------------------------- removed routes


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/presets"),
        ("post", "/api/presets"),
        ("post", "/api/session-selection"),
        ("get", "/api/session-config"),
        ("post", "/api/tts-engines/cleanup"),
    ],
)
def test_preset_and_selection_routes_are_gone(client, method, path):
    test_client, _, _ = client
    kwargs = {"json": {}} if method == "post" else {}
    resp = getattr(test_client, method)(path, headers=auth(), **kwargs)
    assert resp.status_code in (404, 405)


def test_catalog_requires_token(client):
    test_client, _, _ = client
    assert test_client.get("/api/catalog").status_code == 401
    assert test_client.post("/api/profiles/stt", json={}).status_code == 401
    assert test_client.delete("/api/profiles/stt/stt-a").status_code == 401
