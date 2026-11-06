"""F5 server-side: creating backends, immutable ids, write-only api keys.

Regression tests for operator feedback round 1 (F5): the operator
needs an explicit way to ADD a STT/LLM/TTS endpoint, editing must never
overwrite a different id, and a masked/absent key must keep the stored secret.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import SECRET_MASK, Settings, SettingsStore

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def client(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "nf"}})

    store = SettingsStore(data_dir=str(tmp_path), settings=Settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        tc.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        yield tc


def test_create_backend_appears_in_catalog(client):
    tc = client
    r = tc.post("/api/profiles/tts", headers=AUTH, json={
        "id": "tts-new", "role": "tts", "display_name": "New TTS",
        "base_url": "http://cap.test/v1", "model": "clone-model",
        "capabilities": {"voice_cloning": "reference_audio"},
    })
    assert r.status_code == 200, r.text
    cat = tc.get("/api/catalog", headers=AUTH).json()
    ids = [p["id"] for p in cat["roles"]["tts"]["profiles"]]
    assert "tts-new" in ids
    # and in the F1 engine discovery used by the voice editor
    engines = {e["id"]: e for e in tc.get("/api/tts-engines", headers=AUTH).json()["engines"]}
    assert engines["tts-new"]["voice_cloning"] == "reference_audio"


def test_id_is_immutable_on_update(client):
    tc = client
    tc.post("/api/profiles/llm", headers=AUTH, json={"id": "llm-a", "role": "llm", "model": "m-a"})
    # a PUT whose body claims a different id must not create/rename another profile
    r = tc.put("/api/profiles/llm/llm-a", headers=AUTH, json={"id": "llm-b", "model": "m-b"})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "llm-a"
    cat = tc.get("/api/catalog", headers=AUTH).json()
    ids = [p["id"] for p in cat["roles"]["llm"]["profiles"]]
    assert "llm-a" in ids and "llm-b" not in ids


def test_api_key_write_only_and_masked_round_trip(client):
    tc = client
    tc.post("/api/profiles/llm", headers=AUTH, json={"id": "llm-key", "role": "llm", "api_key": "topsecret"})
    # view never returns the raw secret
    cat = tc.get("/api/catalog", headers=AUTH).json()
    prof = next(p for p in cat["roles"]["llm"]["profiles"] if p["id"] == "llm-key")
    assert prof["api_key"] == SECRET_MASK
    # a masked/omitted key keeps the stored secret
    tc.put("/api/profiles/llm/llm-key", headers=AUTH, json={"api_key": SECRET_MASK, "model": "m2"})
    tc.put("/api/profiles/llm/llm-key", headers=AUTH, json={"model": "m3"})
    import json
    from pathlib import Path

    raw = json.loads((Path(tc.app.state.settings_store.data_dir) / "settings.json").read_text())
    assert raw["profiles"]["llm"]["llm-key"]["api_key"] == "topsecret"


def test_voice_cloning_unknown_is_accepted(client):
    tc = client
    r = tc.post("/api/profiles/tts", headers=AUTH, json={
        "id": "tts-unknown", "role": "tts", "model": "m",
        "capabilities": {"voice_cloning": "unknown"},
    })
    assert r.status_code == 200, r.text
    assert r.json()["capabilities"]["voice_cloning"] == "unknown"


def test_clone_copies_secret_server_side_and_appears_in_catalog(client):
    import json
    from pathlib import Path

    tc = client
    r = tc.post("/api/profiles/tts", headers=AUTH, json={
        "id": "tts-src", "role": "tts", "display_name": "Source",
        "base_url": "http://cap.test/v1", "api_key": "topsecret", "model": "clone-model",
        "capabilities": {"voice_cloning": "reference_audio"},
    })
    assert r.status_code == 200, r.text
    # a client-supplied api_key must be ignored: the source's secret is copied
    r = tc.post("/api/profiles/tts/tts-src/clone", headers=AUTH, json={
        "id": "e2e-tts", "display_name": "E2E TTS", "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "api_key": "attacker-supplied",
    })
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "e2e-tts"
    assert r.json()["api_key"] == SECRET_MASK
    cat = tc.get("/api/catalog", headers=AUTH).json()
    ids = [p["id"] for p in cat["roles"]["tts"]["profiles"]]
    assert "tts-src" in ids and "e2e-tts" in ids
    raw = json.loads((Path(tc.app.state.settings_store.data_dir) / "settings.json").read_text())
    assert raw["profiles"]["tts"]["e2e-tts"]["api_key"] == "topsecret"
    assert raw["profiles"]["tts"]["e2e-tts"]["model"] == "Qwen3-TTS-12Hz-1.7B-Base-8bit"


def test_clone_unknown_source_duplicate_and_invalid_role(client):
    tc = client
    assert tc.post("/api/profiles/llm/missing/clone", headers=AUTH, json={"id": "x"}).status_code == 404
    tc.post("/api/profiles/llm", headers=AUTH, json={"id": "llm-a", "role": "llm", "model": "m"})
    dup = tc.post("/api/profiles/llm/llm-a/clone", headers=AUTH, json={"id": "llm-a"})
    assert dup.status_code == 409 and dup.json()["code"] == "profile_exists", dup.text
    bad_role = tc.post("/api/profiles/bogus/llm-a/clone", headers=AUTH, json={"id": "x"})
    assert bad_role.status_code == 422, bad_role.text
    bad_id = tc.post("/api/profiles/llm/llm-a/clone", headers=AUTH, json={"id": "Bad Id"})
    assert bad_id.status_code == 422, bad_id.text


def test_clone_without_id_generates_one_from_the_model_name(client):
    tc = client
    tc.post("/api/profiles/llm", headers=AUTH, json={"id": "llm-a", "role": "llm", "model": "Org/Chat 7B"})
    first = tc.post("/api/profiles/llm/llm-a/clone", headers=AUTH, json={"display_name": "Copy"})
    assert first.status_code == 201, first.text
    assert first.json()["id"] == "org-chat-7b"
    second = tc.post("/api/profiles/llm/llm-a/clone", headers=AUTH, json={})
    assert second.status_code == 201, second.text
    assert second.json()["id"] == "org-chat-7b-2"
    # a model override names the clone
    third = tc.post("/api/profiles/llm/llm-a/clone", headers=AUTH, json={"model": "other-model"})
    assert third.json()["id"] == "other-model" and third.json()["model"] == "other-model"