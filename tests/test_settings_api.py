"""Settings API: auth, the server's own keys only, validation errors,
persistence, file privacy."""

import asyncio
import json
import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import (
    BackendProfile,
    ConversationProfile,
    Settings,
    SettingsStore,
)


def make_settings() -> Settings:
    settings = Settings()
    for role in ("stt", "llm", "tts"):
        settings.profiles[role]["base"] = BackendProfile(
            id="base", role=role, base_url="http://backend.test/v1",
            model=f"{role}-model", api_key=f"sk-stored-{role}-key-123456",
        )
    settings.profiles["stt"]["other"] = BackendProfile(
        id="other", role="stt", base_url="http://backend.test/v1", model="stt-other",
    )
    settings.voice_library.transcription_stt_profile = "base"
    now = time.time()
    settings.conversation_profiles["anna"] = ConversationProfile(
        id="anna", display_name="Anna", created_at=now, updated_at=now,
    )
    return settings


@pytest.fixture
def client(tmp_path):
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(
        settings_store=store,
        backend_factory=lambda settings, client=None: None,
        auth_token="test-token",
    )
    with TestClient(app) as test_client:
        yield test_client, tmp_path


def auth(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


def test_requires_token(client):
    test_client, _ = client
    assert test_client.get("/api/settings").status_code == 401
    assert test_client.get("/api/settings", headers=auth("wrong")).status_code == 401
    assert test_client.put("/api/settings", json={}).status_code == 401


def test_get_settings_shows_only_server_keys_and_no_model_secret(client):
    test_client, _ = client
    data = test_client.get("/api/settings", headers=auth()).json()
    assert set(data) == {"session", "default_set", "voice_library"}
    assert data["default_set"] == ""
    assert data["voice_library"]["transcription_stt_profile"] == "base"
    assert "sk-stored" not in json.dumps(data)


@pytest.mark.parametrize(
    "key",
    ["stt", "llm", "tts", "profiles", "voices", "conversation_profiles", "presets", "voice_routing"],
)
def test_put_refuses_keys_that_are_not_server_settings(client, key):
    test_client, tmp_path = client
    test_client.put("/api/settings", json={"session": {"language": "de"}}, headers=auth())
    before = (tmp_path / "settings.json").read_text()
    resp = test_client.put(
        "/api/settings",
        json={key: {}, "session": {"language": "en"}},
        headers=auth(),
    )
    assert resp.status_code == 422
    assert key in resp.json()["detail"]
    # refused as a whole: nothing of the update was stored
    assert (tmp_path / "settings.json").read_text() == before


def test_default_set_must_name_an_existing_set(client):
    test_client, _ = client
    resp = test_client.put("/api/settings", json={"default_set": "nobody"}, headers=auth())
    assert resp.status_code == 422
    assert "nobody" in resp.json()["detail"]
    assert test_client.get("/api/settings", headers=auth()).json()["default_set"] == ""
    resp = test_client.put("/api/settings", json={"default_set": "anna"}, headers=auth())
    assert resp.status_code == 200, resp.text
    assert resp.json()["default_set"] == "anna"
    # "" clears it again (calls naming no set are then refused)
    resp = test_client.put("/api/settings", json={"default_set": ""}, headers=auth())
    assert resp.status_code == 200 and resp.json()["default_set"] == ""


def test_transcription_model_must_exist(client):
    test_client, _ = client
    resp = test_client.put(
        "/api/settings",
        json={"voice_library": {"transcription_stt_profile": "missing"}},
        headers=auth(),
    )
    assert resp.status_code == 422
    assert "missing" in resp.json()["detail"]
    resp = test_client.put(
        "/api/settings",
        json={"voice_library": {"transcription_stt_profile": "other"}},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["voice_library"]["transcription_stt_profile"] == "other"


def test_settings_persist_across_restart_without_touching_models(client):
    test_client, tmp_path = client
    resp = test_client.put(
        "/api/settings",
        json={"session": {"language": "en", "voice": "serena"}, "default_set": "anna"},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    settings = asyncio.run(SettingsStore(data_dir=str(tmp_path)).get())
    assert settings.session.language == "en"
    assert settings.session.voice == "serena"
    assert settings.default_set == "anna"
    # model entries (and their keys) are untouched by a settings save
    assert settings.profiles["stt"]["base"].api_key == "sk-stored-stt-key-123456"


def test_runtime_backend_triple_is_never_stored(client):
    """Settings.stt/llm/tts are filled per call only: never in the file and
    never in a dump, even when set on the in-memory model."""
    test_client, tmp_path = client
    test_client.put("/api/settings", json={"session": {"language": "en"}}, headers=auth())
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["version"] == 3
    assert not {"stt", "llm", "tts", "presets", "default_preset", "voice_routing"} & set(stored)
    s = Settings()
    s.stt.api_key = "sk-runtime-only"
    assert "sk-runtime-only" not in json.dumps(s.model_dump())


def test_settings_validation_error_is_explicit(client):
    test_client, _ = client
    resp = test_client.put(
        "/api/settings",
        json={"session": {"output_pacing": "warp"}},
        headers=auth(),
    )
    assert resp.status_code == 422
    assert "output_pacing" in resp.json()["detail"]


def test_settings_invalid_turn_detection_rejected(client):
    test_client, _ = client
    resp = test_client.put(
        "/api/settings",
        json={"session": {"turn_detection": {"silence_duration_ms": -5}}},
        headers=auth(),
    )
    assert resp.status_code == 422


def test_probe_route_is_gone(client):
    test_client, _ = client
    resp = test_client.post("/api/probe", json={"target": "llm"}, headers=auth())
    assert resp.status_code in (404, 405)


def test_settings_file_is_private(client):
    test_client, tmp_path = client
    test_client.put("/api/settings", json={"session": {"language": "en"}}, headers=auth())
    mode = stat.S_IMODE(os.stat(tmp_path / "settings.json").st_mode)
    assert mode & 0o077 == 0, "settings file must not be group/world readable"


def test_health_public(client):
    test_client, _ = client
    assert test_client.get("/api/health").json() == {"status": "ok"}
