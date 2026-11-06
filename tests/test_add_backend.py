"""Round 2 / CP3: adding a backend must be understandable (server surface).

Covers:
- POST /api/probe-profile with an UNSAVED `profile` object: reachable / model
  listed / model not listed / unreachable via a capturing fake backend;
- the stored key is used server-side when editing and the key is blank;
- the api_key is NEVER echoed in the response body;
- create refuses an existing id (and never falls back to PUT);
- the native-voice trust field: migration of the legacy hard-coded model list
  (old list -> "honoured", every other existing row -> "unknown"), the field
  default and engine grouping staying identical (CP1 probe).

Synthetic audio only; capturing fake HTTP backend — never a real endpoint.
"""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.config import BackendProfile, Settings, SettingsStore

AUTH = {"Authorization": "Bearer tok"}
V1 = "http://cap.test/v1"


class Capture:
    """A capturing OpenAI-compatible fake.

    `/models` answers with `models`; every other endpoint answers 200 so a
    reachability probe succeeds. Secrets are recorded verbatim so a test can
    prove the stored key was sent, without ever receiving it back.
    """

    def __init__(self, models=None, fail_models=False):
        self.models = models if models is not None else []
        self.fail_models = fail_models
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "auth": request.headers.get("authorization", ""),
            }
        )
        if self.fail_models:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        return httpx.Response(404, json={"error": {"message": "nf"}})

    def auth_for(self, path):
        for call in self.calls:
            if call["path"].endswith(path):
                return call["auth"]
        return None


def make_settings() -> Settings:
    s = Settings()
    s.profiles["stt"]["stt-stored"] = BackendProfile(
        id="stt-stored",
        role="stt",
        display_name="Stored STT",
        base_url=V1,
        api_key="sk-stored-profile-secret",
        model="stt-base",
    )
    return s


def _env(tmp_path, models=None, fail_models=False):
    capture = Capture(models=models, fail_models=fail_models)
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    tc = TestClient(app)
    tc = tc.__enter__()
    tc.app.state.http_client = client
    return tc, store, capture


@pytest.fixture
def env(tmp_path):
    tc, store, capture = _env(tmp_path, models=["stt-base", "stt-alt"])
    yield tc, store, capture
    tc.__exit__(None, None, None)


# ------------------------------------------------- unsaved profile probe


def test_probe_unsaved_profile_reachable_and_model_listed(env):
    tc, _, capture = env
    body = {
        "role": "stt",
        "profile": {"role": "stt", "base_url": V1, "model": "stt-alt"},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True
    assert "stt-alt" in out["models"]
    assert "stt-base" in out["models"]


def test_probe_unsaved_profile_model_not_listed(env):
    tc, _, _ = env
    body = {
        "role": "stt",
        "profile": {"role": "stt", "base_url": V1, "model": "not-a-model"},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is False
    assert out["error_code"] == "model_not_listed"
    assert out["models"] == ["stt-base", "stt-alt"]


def test_probe_unsaved_profile_unreachable(tmp_path):
    tc, _, _ = _env(tmp_path, fail_models=True)
    body = {"role": "llm", "profile": {"role": "llm", "base_url": V1, "model": "m"}}
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is False
    tc.__exit__(None, None, None)


def test_probe_unsaved_profile_never_echoes_key(env):
    tc, _, _ = env
    secret = "sk-super-secret-typed-key"
    body = {
        "role": "stt",
        "profile": {"role": "stt", "base_url": V1, "model": "stt-base", "api_key": secret},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert secret not in r.text
    # the typed key DID reach the backend (proves it was used, not dropped)
    assert env[2].auth_for("/models") == f"Bearer {secret}"


def test_probe_edit_blank_key_uses_stored_profile_key(env):
    tc, _, capture = env
    # editing an existing saved row: key field left blank -> the STORED key is
    # used server-side, never a blank/absent Authorization header
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": V1, "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert capture.auth_for("/models") == "Bearer sk-stored-profile-secret"
    assert "sk-stored-profile-secret" not in r.text


def test_probe_unsaved_blank_key_without_stored_row_sends_no_auth(env):
    tc, _, capture = env
    body = {
        "role": "stt",
        "profile": {"role": "stt", "base_url": V1, "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert capture.auth_for("/models") in (None, "")


def test_probe_unsaved_missing_base_url_returns_not_ok(env):
    tc, _, _ = env
    body = {"role": "stt", "profile": {"role": "stt", "base_url": "", "model": "m"}}
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is False


def test_probe_existing_id_still_works(env):
    tc, _, _ = env
    r = tc.post("/api/probe-profile", json={"role": "stt", "id": "stt-stored"}, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


# ----------------------------------------------- create never falls back


def test_create_refuses_existing_id_and_leaves_row_byte_identical(env):
    tc, store, _ = env
    before = tc.get("/api/catalog", headers=AUTH).json()
    row_before = next(
        p for p in before["roles"]["stt"]["profiles"] if p["id"] == "stt-stored"
    )
    r = tc.post(
        "/api/profiles/stt",
        json={"id": "stt-stored", "base_url": V1, "model": "changed"},
        headers=AUTH,
    )
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "profile_exists"
    after = tc.get("/api/catalog", headers=AUTH).json()
    row_after = next(p for p in after["roles"]["stt"]["profiles"] if p["id"] == "stt-stored")
    assert row_after == row_before, "a refused create must not mutate the existing row"


def test_create_does_not_fall_back_to_put(env):
    """A create of a brand-new id must be a POST only: no PUT is attempted and
    the id appears exactly once in the catalog."""
    tc, _, _ = env
    r = tc.post(
        "/api/profiles/llm",
        json={"id": "llm-new", "base_url": V1, "model": "llm-new-model"},
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    cat = tc.get("/api/catalog", headers=AUTH).json()
    ids = [p["id"] for p in cat["roles"]["llm"]["profiles"]]
    assert ids.count("llm-new") == 1


# ------------------------------------------------ native-voice trust field


def test_capabilities_default_native_voice_labels_unknown():
    caps = BackendProfile(id="x", role="tts", model="m").capabilities
    assert caps.native_voice_labels == "unknown"


def test_capabilities_native_voice_labels_literal():
    for value in ("honoured", "ignored", "unknown"):
        caps = BackendProfile(
            id="x", role="tts", model="m", capabilities={"native_voice_labels": value}
        ).capabilities
        assert caps.native_voice_labels == value


def test_native_voice_labels_rejects_other_values():
    with pytest.raises(Exception):
        BackendProfile(id="x", role="tts", model="m", capabilities={"native_voice_labels": "maybe"})


def test_engine_group_uses_row_data_not_hardcoded_model_list():
    from realtimevoice.engine_groups import group_engines, group_engine_view

    # a model that was never in the old hard-coded list, explicitly honoured
    honoured = BackendProfile(
        id="tts-new",
        role="tts",
        base_url=V1,
        model="brand-new-engine",
        capabilities={"voice_cloning": "none", "voices": ["a", "b"], "native_voice_labels": "honoured"},
    )
    ignored = BackendProfile(
        id="tts-ignore",
        role="tts",
        base_url=V1,
        model="some-engine",
        capabilities={"voice_cloning": "none", "voices": ["a"], "native_voice_labels": "ignored"},
    )
    unknown = BackendProfile(
        id="tts-unknown",
        role="tts",
        base_url=V1,
        model="another-engine",
        capabilities={"voice_cloning": "none", "voices": ["a"]},
    )
    view = {}
    for row in (honoured, ignored, unknown):
        group = group_engines([row])[0]
        view[row.id] = group_engine_view(group, role="tts")
    assert view["tts-new"]["native_voices"] == ["a", "b"]
    assert view["tts-ignore"]["native_voices"] == []
    assert view["tts-unknown"]["native_voices"] == []


def test_engine_groups_preserves_legacy_behaviour_on_migrated_rows():
    """The two models from the old hard-coded list keep offering their native
    voices after migration; a legacy Fish-style row does not."""
    from realtimevoice.config import migrate_native_voice_labels
    from realtimevoice.engine_groups import group_engines, group_engine_view

    rows = [
        BackendProfile(
            id="qwen-cv",
            role="tts",
            base_url=V1,
            model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            capabilities={"voices": ["ryan", "serena"]},
        ),
        BackendProfile(
            id="voxtral",
            role="tts",
            base_url=V1,
            model="mlx-community--Voxtral-4B-TTS-2603-mlx-4bit",
            capabilities={"voices": ["de_male", "de_female"]},
        ),
        BackendProfile(
            id="fish",
            role="tts",
            base_url=V1,
            model="fish-audio-s2-pro-8bit",
            capabilities={"voices": ["native"]},
        ),
    ]
    data = {"profiles": {"tts": {r.id: r.model_dump() for r in rows}}}
    # the migration is idempotent: the rows are already at their final state
    assert migrate_native_voice_labels(data) is False
    migrated = {
        pid: BackendProfile.model_validate(row)
        for pid, row in data["profiles"]["tts"].items()
    }
    view = {}
    for row in migrated.values():
        group = group_engines([row])[0]
        view[row.id] = group_engine_view(group, role="tts")
    assert view["qwen-cv"]["native_voices"] == ["ryan", "serena"]
    assert view["voxtral"]["native_voices"] == ["de_male", "de_female"]
    assert view["fish"]["native_voices"] == []


def test_migration_marks_only_old_list_honoured_and_is_idempotent():
    from realtimevoice.config import migrate_native_voice_labels

    data = {
        "profiles": {
            "tts": {
                "a": {"id": "a", "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "capabilities": {}},
                "b": {"id": "b", "model": "mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", "capabilities": {}},
                "c": {"id": "c", "model": "fish-audio-s2-pro-8bit", "capabilities": {}},
                "d": {"id": "d", "model": "unrelated", "capabilities": {"native_voice_labels": "ignored"}},
            }
        }
    }
    assert migrate_native_voice_labels(data) is True
    tts = data["profiles"]["tts"]
    assert tts["a"]["capabilities"]["native_voice_labels"] == "honoured"
    assert tts["b"]["capabilities"]["native_voice_labels"] == "honoured"
    assert tts["c"]["capabilities"]["native_voice_labels"] == "unknown"
    # an explicit non-default value is preserved, never overwritten
    assert tts["d"]["capabilities"]["native_voice_labels"] == "ignored"
    # ids/references untouched
    assert sorted(tts) == ["a", "b", "c", "d"]
    # second run is a no-op
    assert migrate_native_voice_labels(data) is False


def test_migration_leaves_other_existing_rows_unchanged():
    from realtimevoice.config import migrate_native_voice_labels

    row = {
        "id": "keep",
        "role": "tts",
        "model": "some-engine",
        "voice": "v",
        "extra": {"temperature": 0.7},
        "capabilities": {"voices": ["x"], "voice_cloning": "none"},
    }
    before = json.loads(json.dumps(row))
    data = {"profiles": {"tts": {"keep": row}}}
    migrate_native_voice_labels(data)
    assert data["profiles"]["tts"]["keep"] == {
        **before,
        "capabilities": {**before["capabilities"], "native_voice_labels": "unknown"},
    }


def test_native_voice_trust_is_not_engine_identity():
    """A changed trust field must not split an engine group (it is voice
    metadata, like the other capability keys)."""
    from realtimevoice.engine_groups import engine_key

    a = BackendProfile(id="a", role="tts", base_url=V1, model="m", capabilities={"native_voice_labels": "honoured"})
    b = BackendProfile(id="b", role="tts", base_url=V1, model="m", capabilities={"native_voice_labels": "unknown"})
    assert engine_key(a) == engine_key(b)


# --------------------------------- C1: stored key must not follow a changed endpoint
# Supervisor review of 703a730: the unsaved probe attached the stored key of an
# existing row to whatever endpoint the form contained. Anyone with the UI token
# could point an existing row at an arbitrary host and make the server send that
# row's secret there. The stored key may only be used for the row's OWN endpoint.


class ExfilCapture:
    """Records the target host AND the Authorization header per request.

    Any `/models` request answers 200 with the configured models, except when
    `deny_hosts` matches the request host: those answer 401, so a test can see
    the exact detail the API returns when a stored key was deliberately withheld.
    """

    def __init__(self, models=None, deny_hosts=()):
        self.models = models if models is not None else []
        self.deny_hosts = set(deny_hosts)
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.calls.append(
            {
                "method": request.method,
                "host": host,
                "path": request.url.path,
                "auth": request.headers.get("authorization", ""),
            }
        )
        if host in self.deny_hosts:
            return httpx.Response(401, json={"error": {"message": "unauthorized"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        return httpx.Response(404, json={"error": {"message": "nf"}})

    def auth_to(self, host):
        for call in self.calls:
            if call["host"] == host:
                return call["auth"]
        return None

    @property
    def hosts(self):
        return [c["host"] for c in self.calls]


def _exfil_env(tmp_path, models=None, deny_hosts=()):
    capture = ExfilCapture(models=models, deny_hosts=deny_hosts)
    client = httpx.AsyncClient(transport=httpx.MockTransport(capture.handler))
    store = SettingsStore(data_dir=str(tmp_path), settings=make_settings())
    app = create_app(settings_store=store, backend_factory=lambda s, c=None: None, auth_token="tok")
    tc = TestClient(app)
    tc = tc.__enter__()
    tc.app.state.http_client = client
    return tc, store, capture


def test_c1_edit_unchanged_endpoint_uses_stored_key(tmp_path):
    """Unchanged endpoint: the stored key is still sent (unchanged behaviour)."""
    tc, _, capture = _exfil_env(tmp_path, models=["stt-base"])
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": V1, "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert capture.auth_to("cap.test") == "Bearer sk-stored-profile-secret"
    assert "sk-stored-profile-secret" not in r.text
    tc.__exit__(None, None, None)


def test_c1_edit_changed_endpoint_never_receives_stored_key(tmp_path):
    """Changed endpoint: the stored key must NOT be sent to the new host."""
    tc, _, capture = _exfil_env(tmp_path, models=["stt-base"])
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": "http://evil.test/v1", "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert capture.auth_to("evil.test") in (None, ""), "stored key must not follow a changed endpoint"
    assert "sk-stored-profile-secret" not in r.text
    tc.__exit__(None, None, None)


def test_c1_edit_changed_endpoint_401_explains_without_secret(tmp_path):
    """A withheld key against a denying backend surfaces a clear, secret-free
    detail and the machine-readable code."""
    tc, _, capture = _exfil_env(tmp_path, deny_hosts={"evil.test"})
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": "http://evil.test/v1", "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is False
    assert out.get("error_code") == "key_required_for_changed_endpoint", out
    assert "sk-stored-profile-secret" not in r.text
    assert "API key" in out.get("detail", "") or "key" in out.get("detail", "").lower(), out
    tc.__exit__(None, None, None)


def test_c1_new_backend_with_colliding_id_never_receives_stored_key(tmp_path):
    """A NEW backend (no top-level id) never gets a stored key, even when its
    derived id equals an existing row."""
    tc, _, capture = _exfil_env(tmp_path, models=["stt-base"])
    body = {
        "role": "stt",
        "profile": {"id": "stt-stored", "role": "stt", "base_url": "http://evil.test/v1", "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert capture.auth_to("evil.test") in (None, ""), "a new backend must never receive a stored key"
    assert "sk-stored-profile-secret" not in r.text
    tc.__exit__(None, None, None)


def test_c1_explicit_key_is_used_for_changed_endpoint(tmp_path):
    """An explicitly typed key is the operator's decision: use it anywhere."""
    tc, _, capture = _exfil_env(tmp_path, models=["stt-base"])
    typed = "sk-typed-on-purpose"
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": "http://evil.test/v1", "model": "stt-base", "api_key": typed},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert capture.auth_to("evil.test") == f"Bearer {typed}"
    assert typed not in r.text
    tc.__exit__(None, None, None)


def test_c1_endpoint_trailing_slash_and_host_case_count_as_unchanged(tmp_path):
    """Scheme+host+port+path with a trailing `/` ignored and a case-insensitive
    host is the SAME endpoint -> the stored key stays in play."""
    tc, _, capture = _exfil_env(tmp_path, models=["stt-base"])
    body = {
        "role": "stt",
        "id": "stt-stored",
        "profile": {"role": "stt", "base_url": "http://CAP.TEST/v1/", "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert capture.auth_to("cap.test") == "Bearer sk-stored-profile-secret"
    tc.__exit__(None, None, None)


def test_c1_new_backend_with_changed_endpoint_401_reports_key_required(tmp_path):
    """A new backend with no stored key reaching a 401 also explains clearly."""
    tc, _, _ = _exfil_env(tmp_path, deny_hosts={"evil.test"})
    body = {
        "role": "stt",
        "profile": {"role": "stt", "base_url": "http://evil.test/v1", "model": "stt-base", "api_key": ""},
    }
    r = tc.post("/api/probe-profile", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is False
    assert out.get("error_code") == "key_required_for_changed_endpoint", out
    tc.__exit__(None, None, None)


# ------------------- C2: migration must not override a later explicit "unknown"
# The per-model validator ran on EVERY materialisation, so an operator who
# deliberately set "unknown" on a legacy-trusted model could never save it (it
# flipped back to "honoured"). Migration may only fill a field that is ABSENT.

LEGACY_MODEL = "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"


def test_c2_explicit_unknown_on_legacy_model_survives_api_and_reload(tmp_path):
    """Set explicit `unknown` on a legacy-trusted model, save, and reload from
    disk: it must stay `unknown`, not flip back to `honoured`."""
    tc, store, _ = _env(tmp_path, models=["tts-base"])

    def create_tts(pid, value):
        payload = {
            "id": pid,
            "role": "tts",
            "base_url": V1,
            "model": LEGACY_MODEL,
            "capabilities": {"voices": ["ryan"], "native_voice_labels": value},
        }
        r = tc.post("/api/profiles/tts", json=payload, headers=AUTH)
        assert r.status_code == 200, r.text
        return r.json()

    row = create_tts("cv-unknown", "unknown")
    assert row["capabilities"]["native_voice_labels"] == "unknown", row["capabilities"]

    # the stored settings JSON on disk must also carry `unknown`
    cat = tc.get("/api/catalog", headers=AUTH).json()
    again = next(p for p in cat["roles"]["tts"]["profiles"] if p["id"] == "cv-unknown")
    assert again["capabilities"]["native_voice_labels"] == "unknown"

    # force a fresh load from disk with a NEW store over the same data dir
    import asyncio

    fresh = SettingsStore(data_dir=str(tmp_path))
    settings = asyncio.run(fresh.get())
    assert settings.profiles["tts"]["cv-unknown"].capabilities.native_voice_labels == "unknown"
    tc.__exit__(None, None, None)


def test_c2_explicit_honoured_and_ignored_are_preserved_on_legacy_model(tmp_path):
    tc, _, _ = _env(tmp_path)
    for pid, value in (("cv-h", "honoured"), ("cv-i", "ignored")):
        r = tc.post(
            "/api/profiles/tts",
            json={
                "id": pid,
                "role": "tts",
                "base_url": V1,
                "model": LEGACY_MODEL,
                "capabilities": {"native_voice_labels": value},
            },
            headers=AUTH,
        )
        assert r.status_code == 200, r.text
        assert r.json()["capabilities"]["native_voice_labels"] == value, r.json()["capabilities"]
    tc.__exit__(None, None, None)


def test_c2_legacy_row_without_the_key_is_honoured():
    """Legacy data on disk / a payload without the key -> migrated to honoured."""
    p = BackendProfile(
        id="x", role="tts", base_url=V1, model=LEGACY_MODEL, capabilities={"voices": ["ryan"]}
    )
    assert p.capabilities.native_voice_labels == "honoured"


def test_c2_non_legacy_row_without_the_key_is_unknown():
    p = BackendProfile(id="x", role="tts", base_url=V1, model="some-engine", capabilities={"voices": ["a"]})
    assert p.capabilities.native_voice_labels == "unknown"


def test_c2_dict_migration_does_not_touch_an_explicit_unknown():
    """The load-time dict migration must only fill an ABSENT field."""
    from realtimevoice.config import migrate_native_voice_labels

    data = {
        "profiles": {
            "tts": {
                "explicit": {
                    "id": "explicit",
                    "model": LEGACY_MODEL,
                    "capabilities": {"native_voice_labels": "unknown"},
                },
                "absent": {"id": "absent", "model": LEGACY_MODEL, "capabilities": {}},
            }
        }
    }
    migrate_native_voice_labels(data)
    assert data["profiles"]["tts"]["explicit"]["capabilities"]["native_voice_labels"] == "unknown"
    assert data["profiles"]["tts"]["absent"]["capabilities"]["native_voice_labels"] == "honoured"


def test_c2_dict_migration_is_idempotent_with_explicit_unknown():
    from realtimevoice.config import migrate_native_voice_labels

    data = {
        "profiles": {
            "tts": {
                "explicit": {
                    "id": "explicit",
                    "model": LEGACY_MODEL,
                    "capabilities": {"native_voice_labels": "unknown"},
                }
            }
        }
    }
    assert migrate_native_voice_labels(data) is False
    assert data["profiles"]["tts"]["explicit"]["capabilities"]["native_voice_labels"] == "unknown"


def test_c2_cp1_grouping_unchanged_for_rows_without_the_key():
    """Rows without the key keep the old grouping behaviour after migration."""
    from realtimevoice.engine_groups import group_engines, group_engine_view

    rows = [
        BackendProfile(id="qwen-cv", role="tts", base_url=V1, model=LEGACY_MODEL,
                       capabilities={"voices": ["ryan", "serena"]}),
        BackendProfile(id="fish", role="tts", base_url=V1, model="fish-audio-s2-pro-8bit",
                       capabilities={"voices": ["native"]}),
    ]
    view = {r.id: group_engine_view(group_engines([r])[0], role="tts") for r in rows}
    assert view["qwen-cv"]["native_voices"] == ["ryan", "serena"]
    assert view["fish"]["native_voices"] == []