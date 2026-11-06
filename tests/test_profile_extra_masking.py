"""Model entries: secret and reference-material handling in `extra`.

- voice-reference material: keep (masked placeholder), replace (new value),
  explicit clear (JSON null) — raw values never appear in any response,
- error bodies never echo submitted credentials/reference material
  (pydantic input values are stripped),
- masked views saved back keep stored secrets at any depth (lists and
  nested objects included); a mask never becomes stored data.

Fixtures are synthetic; no production settings are touched.
"""

import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient

from fake_backends import make_backends
from realtimevoice.app import create_app
from realtimevoice.config import (
    SECRET_MASK,
    BackendProfile,
    Settings,
    SettingsStore,
    reference_fingerprint,
)

AUTH = {"Authorization": "Bearer tok"}
REF_A = base64.b64encode(b"RIFFsynthetic-reference-A").decode()
REF_B = base64.b64encode(b"RIFFsynthetic-reference-B").decode()
FP_A = reference_fingerprint({"ref_audio": REF_A})
FP_B = reference_fingerprint({"ref_audio": REF_B})


@pytest.fixture
def env(tmp_path):
    settings = Settings()
    settings.profiles["tts"]["base"] = BackendProfile(
        id="base", role="tts", base_url="http://backend.test/v1", model="tts-model",
    )
    store = SettingsStore(data_dir=str(tmp_path), settings=settings)
    app = create_app(
        settings_store=store,
        backend_factory=lambda s, client=None: make_backends(),
        auth_token="tok",
    )
    with TestClient(app) as client:
        yield client, tmp_path, app


def auth():
    return dict(AUTH)


def stored_settings(app):
    return asyncio.run(app.state.settings_store.get())


def test_reference_keep_replace_clear_via_catalog_put(env):
    """ref material semantics on the catalog PUT: masked = keep, new value =
    replace, JSON null = explicit remove; omitted = untouched; raw values
    never leave the server in any view."""
    client, _, app = env
    resp = client.post(
        "/api/profiles/tts",
        json={"id": "tts-refs", "role": "tts", "base_url": "http://backend.test/v1",
              "model": "chatterbox", "voice": "ref-voice",
              "extra": {"ref_audio": REF_A, "ref_text": "reference transcript A"}},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    view = resp.json()
    assert view["extra"]["ref_audio"] == SECRET_MASK
    assert view["ref_sha256"] == FP_A
    assert view["sensitive_extra_keys"] == ["ref_audio", "ref_text"]
    assert "RIFFsynthetic-reference-A" not in resp.text

    # KEEP: masked placeholders never become stored data
    resp = client.put(
        "/api/profiles/tts/tts-refs",
        json={"extra": {"ref_audio": SECRET_MASK, "ref_text": SECRET_MASK, "notes": "kept"}},
        headers=auth(),
    )
    assert resp.status_code == 200
    stored = stored_settings(app).profiles["tts"]["tts-refs"]
    assert stored.extra["ref_audio"] == REF_A
    assert stored.extra["ref_text"] == "reference transcript A"
    assert stored.extra["notes"] == "kept"

    # REPLACE: a real new value updates; the other ref field stays
    resp = client.put(
        "/api/profiles/tts/tts-refs",
        json={"extra": {"ref_audio": REF_B}},
        headers=auth(),
    )
    assert resp.status_code == 200
    assert resp.json()["ref_sha256"] == FP_B
    stored = stored_settings(app).profiles["tts"]["tts-refs"]
    assert stored.extra["ref_audio"] == REF_B
    assert stored.extra["ref_text"] == "reference transcript A"
    assert "RIFFsynthetic-reference-B" not in resp.text

    # explicit CLEAR: JSON null removes exactly the submitted keys
    resp = client.put(
        "/api/profiles/tts/tts-refs",
        json={"extra": {"ref_audio": None, "ref_text": None}},
        headers=auth(),
    )
    assert resp.status_code == 200
    assert "ref_sha256" not in resp.json()
    stored = stored_settings(app).profiles["tts"]["tts-refs"]
    assert "ref_audio" not in stored.extra
    assert "ref_text" not in stored.extra
    assert stored.extra["notes"] == "kept"  # untouched extras survive

    # masked keep with nothing stored adds nothing (never stores the mask)
    resp = client.put(
        "/api/profiles/tts/tts-refs",
        json={"extra": {"ref_audio": SECRET_MASK}},
        headers=auth(),
    )
    assert resp.status_code == 200
    stored = stored_settings(app).profiles["tts"]["tts-refs"]
    assert "ref_audio" not in stored.extra
    assert SECRET_MASK not in json.dumps(stored.extra)


def test_error_body_never_echoes_submitted_secret_or_reference(env):
    """A validation failure must not echo pydantic input values: submitted
    credentials/reference material stay out of the error body."""
    client, _, _ = env
    resp = client.post(
        "/api/profiles/tts",
        json={"id": "tts-bad", "role": "tts", "base_url": "ftp://not-http",
              "api_key": "sk-very-secret-98765",
              "extra": "ref_audio=RIFFsynthetic-reference-LEAK"},
        headers=auth(),
    )
    assert resp.status_code == 422
    detail = json.dumps(resp.json())
    assert "sk-very-secret-98765" not in detail
    assert "RIFFsynthetic-reference-LEAK" not in detail
    assert "base_url" in detail  # still an explicit, useful error

    # same class of failure through the catalog mutation path
    resp = client.put(
        "/api/profiles/tts/base",
        json={"extra": ["ref_audio", "RIFFsynthetic-reference-LEAK"]},
        headers=auth(),
    )
    assert resp.status_code == 422
    assert "RIFFsynthetic-reference-LEAK" not in json.dumps(resp.json())


def test_merge_extra_values_masked_list_roundtrip():
    """Pure-function counterexample: the masked VIEW of a
    list-carried secret must merge back to the stored value unchanged —
    mask placeholders inside list items are keep-markers, never data."""
    from realtimevoice.config import mask_sensitive_value, merge_extra_values

    old = {"options": [{"api_key": "synthetic-test-secret", "x": 1}]}
    assert merge_extra_values(old, mask_sensitive_value(old)) == old


def test_merge_extra_values_masked_nested_list_roundtrip():
    """Lists nested in lists merge positionally too: a mask inside a
    list-of-lists keeps its stored counterpart (no wholesale replace)."""
    from realtimevoice.config import mask_sensitive_value, merge_extra_values

    old = {"options": [[{"api_key": "synthetic-test-secret"}]]}
    assert merge_extra_values(old, mask_sensitive_value(old)) == old

    deeper = {"a": [[{"api_key": "s1", "n": [[{"token": "s2"}]]}]]}
    assert merge_extra_values(deeper, mask_sensitive_value(deeper)) == deeper


def test_merge_extra_values_new_nested_mask_never_stored():
    """A nested masked path with NOTHING stored must never store the literal
    mask (masked leaves of genuinely new values are stripped/dropped)."""
    from realtimevoice.config import merge_extra_values

    result = merge_extra_values({}, {"options": [{"api_key": "********", "x": 1}]})
    assert result == {"options": [{"x": 1}]}
    assert "********" not in json.dumps(result)

    result = merge_extra_values({}, {"auth": {"password": "********", "user": "u"}})
    assert result == {"auth": {"user": "u"}}
    assert "********" not in json.dumps(result)

    # a mask AS the whole new value drops out entirely
    assert merge_extra_values({}, {"notes": "********"}) == {}


def test_masked_view_save_roundtrip_through_api(env):
    """HTTP-level: saving the masked view back must preserve stored secrets
    and list-carried secrets byte-for-byte; the response stays masked."""
    client, _, app = env
    client.put("/api/settings", json={"session": {"language": "de"}}, headers=auth())
    secret = "synthetic-list-secret-42"
    resp = client.post(
        "/api/profiles/llm",
        json={"id": "llm-list", "role": "llm", "base_url": "http://backend.test/v1",
              "model": "ml",
              "extra": {"options": [{"api_key": secret, "x": 1}], "api_secret": "top-secret-7"}},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    view = resp.json()
    assert view["extra"]["api_secret"] == SECRET_MASK
    assert view["extra"]["options"][0]["api_key"] == SECRET_MASK
    assert secret not in resp.text and "top-secret-7" not in resp.text

    # save the masked view back unchanged (the editor's exact behavior)
    resp = client.put("/api/profiles/llm/llm-list", json={"extra": view["extra"]}, headers=auth())
    assert resp.status_code == 200
    assert secret not in resp.text
    stored = stored_settings(app).profiles["llm"]["llm-list"]
    assert stored.extra["options"][0]["api_key"] == secret  # roundtrip preserved
    assert stored.extra["options"][0]["x"] == 1
    assert stored.extra["api_secret"] == "top-secret-7"
    assert SECRET_MASK not in json.dumps(stored.extra)  # no literal mask stored


def test_nested_extras_preserved_and_merged(env):
    """Backward-compatible nested extras: stop arrays / response_format
    objects keep working through every write path; unchanged nested Saves
    (including masked sensitive leaves) preserve stored values recursively;
    JSON null removes at its path and is never resurrected by the deep
    merge; views mask nested secrets recursively."""
    client, _, app = env
    client.put("/api/settings", json={"session": {"language": "de"}}, headers=auth())
    resp = client.post(
        "/api/profiles/llm",
        json={"id": "llm-nested", "role": "llm", "base_url": "http://backend.test/v1",
              "model": "mn",
              "extra": {"stop": ["A", "B"], "response_format": {"type": "json_object"},
                        "options": {"api_key": "sk-nested-1", "notes": "n"}}},
        headers=auth(),
    )
    assert resp.status_code == 200, resp.text
    stored = stored_settings(app).profiles["llm"]["llm-nested"]
    assert stored.extra["stop"] == ["A", "B"]
    assert stored.extra["response_format"] == {"type": "json_object"}

    # unchanged nested Save with a nested masked leaf: nothing changes
    resp = client.put(
        "/api/profiles/llm/llm-nested",
        json={"extra": {"stop": ["A", "B"], "response_format": {"type": "json_object"},
                        "options": {"api_key": SECRET_MASK, "notes": "n"}}},
        headers=auth(),
    )
    assert resp.status_code == 200
    stored = stored_settings(app).profiles["llm"]["llm-nested"]
    assert stored.extra["options"]["api_key"] == "sk-nested-1"  # nested mask kept stored
    assert stored.extra["stop"] == ["A", "B"]

    # nested dicts MERGE (partial response_format update)
    client.put("/api/profiles/llm/llm-nested", json={"extra": {"response_format": {"language": "de"}}}, headers=auth())
    stored = stored_settings(app).profiles["llm"]["llm-nested"]
    assert stored.extra["response_format"] == {"type": "json_object", "language": "de"}

    # top-level null removes exactly that key; siblings survive
    client.put("/api/profiles/llm/llm-nested", json={"extra": {"stop": None}}, headers=auth())
    stored = stored_settings(app).profiles["llm"]["llm-nested"]
    assert "stop" not in stored.extra
    assert stored.extra["response_format"] == {"type": "json_object", "language": "de"}

    # views mask nested secrets recursively; raw value absent everywhere
    catalog = client.get("/api/catalog", headers=auth()).json()
    view = next(p for p in catalog["roles"]["llm"]["profiles"] if p["id"] == "llm-nested")
    assert view["extra"]["options"]["api_key"] == SECRET_MASK
    assert "sk-nested-1" not in json.dumps(catalog)
    assert "sk-nested-1" not in json.dumps(client.get("/api/settings", headers=auth()).json())


def test_nested_stored_secret_masked_recursively_in_views(env):
    """A manually seeded nested secret (file-level, bypassing the write
    guard) is masked recursively in every outward view — never returned raw."""
    client, _, app = env

    def seed():
        async def _seed():
            def mutate(settings):
                settings.profiles.setdefault("llm", {})["llm-deep"] = __import__(
                    "realtimevoice.config", fromlist=["BackendProfile"]
                ).BackendProfile(
                    id="llm-deep", role="llm", base_url="http://backend.test/v1",
                    extra={"options": {"api_key": "sk-deep-secret"}, "notes": "ok"},
                )
            await app.state.settings_store.mutate(mutate)
        asyncio.run(_seed())

    seed()
    catalog_text = json.dumps(client.get("/api/catalog", headers=auth()).json())
    assert "sk-deep-secret" not in catalog_text
    assert "sk-deep-secret" not in json.dumps(client.get("/api/settings", headers=auth()).json())
