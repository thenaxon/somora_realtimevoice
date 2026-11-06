"""CP0: web assets must force revalidation and the build identity must be visible.

Two guarantees are covered here:

* every web asset served by the static mount carries
  ``Cache-Control: no-cache`` so a normally reloading browser always
  revalidates (ETag/304 keep it cheap) instead of silently reusing a stale
  ES module after a deploy;
* ``GET /api/version`` is authenticated like the other management APIs and
  returns the server's startup revision (never fails startup).
"""

import pytest
from fastapi.testclient import TestClient

from fake_backends import make_backends
from realtimevoice.app import create_app
from realtimevoice.config import Settings, SettingsStore


ASSETS = (
    "/ui/index.html",
    "/ui/",
    "/ui/main.js",
    "/ui/audio.js",
    "/ui/shell.css",
    "/worklet.js",
)


@pytest.fixture
def client(tmp_path):
    store = SettingsStore(data_dir=str(tmp_path), settings=Settings())
    app = create_app(
        settings_store=store,
        backend_factory=lambda settings, client=None: make_backends(),
        auth_token="test-token",
    )
    with TestClient(app) as test_client:
        yield test_client


def test_web_assets_force_revalidation(client):
    for asset in ASSETS:
        resp = client.get(asset)
        assert resp.status_code == 200, asset
        cache_control = resp.headers.get("cache-control", "")
        assert cache_control == "no-cache", f"{asset}: {cache_control!r}"


def test_revalidation_keeps_not_modified_cheap(client):
    first = client.get("/ui/audio.js")
    etag = first.headers["etag"]
    second = client.get("/ui/audio.js", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.headers.get("cache-control") == "no-cache"


def test_api_version_requires_auth(client):
    assert client.get("/api/version").status_code == 401
    assert client.get("/api/version", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_api_version_returns_version_and_revision(client):
    from realtimevoice import __version__

    resp = client.get("/api/version", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"version", "revision"}
    assert body["version"] == __version__
    assert body["revision"]