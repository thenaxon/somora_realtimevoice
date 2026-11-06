"""R2 / C3 (final corrections queue): a saved backend must not silently keep
sending its stored key after its endpoint changed.

`PUT /api/profiles/{role}/{id}` with a changed `base_url` and a blank/masked
`api_key` used to keep the stored key (`merge_profile_masked`). The next probe
or conversation then sent that secret to the new endpoint — the same
exfiltration class as CP3 C1, via save instead of the unsaved probe.

Fix: when the normalised endpoint (scheme, host, port, path — the C1 helper)
changes and no new key is sent, refuse with `409 key_required_for_changed_endpoint`
unless the request explicitly confirms `keep_api_key_for_new_endpoint: true`.
The edit form then asks.

Synthetic data only; the secret is only ever asserted ABSENT from responses.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import BackendProfile, Settings, SettingsStore

AUTH = {"Authorization": "Bearer tok"}
V1 = "http://cap.test/v1"
V2 = "http://new.test/v1"
STORED_KEY = "sk-stored-profile-secret"


def make_settings() -> Settings:
    s = Settings()
    s.profiles["stt"]["stt-stored"] = BackendProfile(
        id="stt-stored",
        role="stt",
        display_name="Stored STT",
        base_url=V1,
        api_key=STORED_KEY,
        model="stt-base",
    )
    return s


@pytest.fixture
def env(tmp_path):
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    with TestClient(app) as tc:
        yield tc, store, tmp_path


def _payload(base_url, **extra):
    body = {
        "id": "stt-stored",
        "role": "stt",
        "display_name": "Stored STT",
        "base_url": base_url,
        "model": "stt-base",
    }
    body.update(extra)
    return body


def _stored_key(tc, store):
    import asyncio

    settings = asyncio.run(store.get())
    return settings.profiles["stt"]["stt-stored"].api_key


def _stored_base_url(tc, store):
    import asyncio

    settings = asyncio.run(store.get())
    return settings.profiles["stt"]["stt-stored"].base_url


# ---------------------------------------------------------------- changed host

def test_put_changed_host_blank_key_is_refused(env):
    """Changed host + blank key -> 409 `key_required_for_changed_endpoint` and
    the stored row is byte-identical (no key follows the new endpoint)."""
    tc, store, _ = env
    r = tc.put("/api/profiles/stt/stt-stored", json=_payload(V2, api_key=""), headers=AUTH)
    assert r.status_code == 409, r.text
    out = r.json()
    assert out.get("code") == "key_required_for_changed_endpoint", out
    assert STORED_KEY not in r.text
    assert _stored_key(tc, store) == STORED_KEY
    assert _stored_base_url(tc, store) == V1, "the row must not move to the new endpoint"


def test_put_changed_host_masked_key_is_refused(env):
    """The masked placeholder (as a view would echo it) is not a new key."""
    tc, store, _ = env
    r = tc.put("/api/profiles/stt/stt-stored", json=_payload(V2, api_key="********"), headers=AUTH)
    assert r.status_code == 409, r.text
    assert r.json().get("code") == "key_required_for_changed_endpoint"
    assert STORED_KEY not in r.text
    assert _stored_key(tc, store) == STORED_KEY
    assert _stored_base_url(tc, store) == V1


def test_put_changed_host_typed_key_is_saved(env):
    """An explicitly typed key is the operator's decision: save it for the new
    endpoint."""
    tc, store, _ = env
    typed = "sk-typed-on-purpose"
    r = tc.put("/api/profiles/stt/stt-stored", json=_payload(V2, api_key=typed), headers=AUTH)
    assert r.status_code == 200, r.text
    assert typed not in r.text, "a key must never be echoed"
    assert _stored_key(tc, store) == typed
    assert _stored_base_url(tc, store) == V2


def test_put_changed_host_with_explicit_confirm_keeps_key(env):
    """`keep_api_key_for_new_endpoint: true` is the explicit confirmation: the
    endpoint changes and the stored key is kept."""
    tc, store, _ = env
    r = tc.put(
        "/api/profiles/stt/stt-stored",
        json=_payload(V2, api_key="", keep_api_key_for_new_endpoint=True),
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    assert STORED_KEY not in r.text
    assert _stored_key(tc, store) == STORED_KEY
    assert _stored_base_url(tc, store) == V2


def test_put_confirm_flag_is_not_a_silent_blanket_keep(env):
    """The confirmation only matters for a changed endpoint; it never blocks a
    typed key from being stored."""
    tc, store, _ = env
    typed = "sk-typed-and-confirmed"
    r = tc.put(
        "/api/profiles/stt/stt-stored",
        json=_payload(V2, api_key=typed, keep_api_key_for_new_endpoint=True),
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    assert _stored_key(tc, store) == typed


# ------------------------------------------------------------- unchanged host

def test_put_unchanged_endpoint_blank_key_keeps_stored_key(env):
    """Unchanged endpoint + blank key keeps the stored key (today's behaviour)."""
    tc, store, _ = env
    r = tc.put("/api/profiles/stt/stt-stored", json=_payload(V1, api_key="", display_name="Renamed"), headers=AUTH)
    assert r.status_code == 200, r.text
    assert STORED_KEY not in r.text
    assert _stored_key(tc, store) == STORED_KEY
    assert _stored_base_url(tc, store) == V1


def test_put_same_endpoint_trailing_slash_and_host_case_is_unchanged(env):
    """Trailing `/` and host case differences count as the SAME endpoint: no
    confirmation is needed and the stored key stays."""
    tc, store, _ = env
    r = tc.put(
        "/api/profiles/stt/stt-stored",
        json=_payload("http://CAP.TEST/v1/", api_key=""),
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    assert _stored_key(tc, store) == STORED_KEY


def test_put_changed_path_is_refused(env):
    """A changed PATH on the same host is a different endpoint too."""
    tc, store, _ = env
    r = tc.put("/api/profiles/stt/stt-stored", json=_payload("http://cap.test/v2", api_key=""), headers=AUTH)
    assert r.status_code == 409, r.text
    assert r.json().get("code") == "key_required_for_changed_endpoint"
    assert _stored_base_url(tc, store) == V1


# ------------------------------------------------------------------- clone

def test_clone_cannot_change_base_url_and_copies_the_key(env):
    """A clone always keeps the source's endpoint (body cannot override it), so
    the copied key is only ever used for that same endpoint."""
    tc, store, _ = env
    r = tc.post(
        "/api/profiles/stt/stt-stored/clone",
        json={"id": "stt-copy", "base_url": V2, "api_key": "sk-should-be-ignored"},
        headers=AUTH,
    )
    assert r.status_code == 201, r.text
    assert STORED_KEY not in r.text
    assert "sk-should-be-ignored" not in r.text

    import asyncio

    settings = asyncio.run(store.get())
    copy = settings.profiles["stt"]["stt-copy"]
    assert copy.base_url == V1, "clone must not change the endpoint"
    assert copy.api_key == STORED_KEY, "clone keeps the stored secret server-side"