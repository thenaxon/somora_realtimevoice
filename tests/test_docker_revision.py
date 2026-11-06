"""Container build identity: ``REALTIMEVOICE_REVISION`` env is authoritative.

The image has no ``.git`` (`.dockerignore` excludes it), so ``git rev-parse``
cannot report the revision of a containerised build. ``detect_revision`` must
first honour the env var set by the Dockerfile build arg (trimmed, restricted
to a safe character class), and only fall back to git when it is absent or
invalid. No environment leakage: tests pass an explicit mapping.
"""

from __future__ import annotations

from pathlib import Path

from realtimevoice.app import detect_revision

VALID = "abc1234deadbeef"
INVALID = (
    "",
    "   ",
    "with space",
    "semi;colon",
    "../escape",
    "a" * 65,
    "quote'",
    "dollar$sign",
)


def test_env_revision_is_honoured(monkeypatch):
    monkeypatch.setenv("REALTIMEVOICE_REVISION", VALID)
    # a repo_root that would fail git is irrelevant when the env wins
    assert detect_revision(Path("/nonexistent/checkout")) == VALID


def test_env_revision_is_trimmed(monkeypatch):
    monkeypatch.setenv("REALTIMEVOICE_REVISION", f"  {VALID}  ")
    assert detect_revision(Path("/nonexistent/checkout")) == VALID


def test_invalid_env_revision_is_ignored(monkeypatch):
    for value in INVALID:
        monkeypatch.setenv("REALTIMEVOICE_REVISION", value)
        # an invalid env value must never be reported verbatim; it falls back
        # to git, which returns "unknown" for a nonexistent checkout
        assert detect_revision(Path("/nonexistent/checkout")) == "unknown", value


def test_git_fallback_when_env_absent(monkeypatch):
    monkeypatch.delenv("REALTIMEVOICE_REVISION", raising=False)
    repo = Path(__file__).resolve().parent.parent
    revision = detect_revision(repo)
    assert revision != "unknown"
    assert set(revision) <= set("0123456789abcdef")


def test_created_app_reports_env_revision(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from fake_backends import make_backends
    from realtimevoice.app import create_app
    from realtimevoice.config import Settings, SettingsStore

    monkeypatch.setenv("REALTIMEVOICE_REVISION", VALID)
    store = SettingsStore(data_dir=str(tmp_path), settings=Settings())
    app = create_app(
        settings_store=store,
        backend_factory=lambda settings, client=None: make_backends(),
        auth_token="test-token",
    )
    with TestClient(app) as client:
        resp = client.get("/api/version", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert resp.json()["revision"] == VALID
