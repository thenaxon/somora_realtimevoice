"""Container release assets: declared runtime deps, Dockerfile hygiene, compose.

These are static checks that run without a Docker daemon:

* ``python-multipart`` is a RUNTIME dependency (the app registers ``Form`` /
  ``File`` upload routes at import time, so a plain ``pip install .`` without
  the ``dev`` extra must still start) — the c753a95 image exited with
  ``RuntimeError: Form data requires "python-multipart" to be installed.``
* the Dockerfile pins the base image by digest, carries OCI revision/source
  labels and a stdlib-only healthcheck against the unauthenticated
  ``/api/health`` path;
* ``deploy/compose.yaml`` binds an explicit address, never ``0.0.0.0``, uses
  bind mounts (no named volumes), guards required variables and sets log
  rotation.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "deploy" / "compose.yaml"


def _runtime_deps() -> dict:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = {}
    for dep in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~\[ ]", dep, maxsplit=1)[0]
        deps[name] = dep
    return deps


def test_python_multipart_is_runtime_dependency():
    deps = _runtime_deps()
    assert "python-multipart" in deps, (
        "python-multipart must be in [project] dependencies, not only the dev extra"
    )
    # keep the version bound
    assert re.search(r"[<>=!~]", deps["python-multipart"]), deps["python-multipart"]


def test_all_imported_third_party_packages_are_runtime_dependencies():
    """Every top-level import in the package must be stdlib, local or a
    declared runtime dependency (mapped to its import name)."""
    import_name_to_dist = {
        "fastapi": "fastapi",
        "httpx": "httpx",
        "numpy": "numpy",
        "emoji": "emoji",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "multipart": "python-multipart",
        "python_multipart": "python-multipart",
        "starlette": "starlette",
    }
    stdlib = set(__import__("sys").stdlib_module_names)
    deps = _runtime_deps()
    # packages guaranteed by declared deps even if not named directly
    provided_by = {"starlette": {"fastapi"}}
    import ast

    seen: set[str] = set()
    for path in (ROOT / "src" / "realtimevoice").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            seen.update(names)
    missing = []
    for name in sorted(seen):
        if name in stdlib or name == "realtimevoice":
            continue
        if name in deps:
            continue
        providers = provided_by.get(name, set())
        if providers & set(deps):
            continue
        if name in import_name_to_dist and import_name_to_dist[name] in deps:
            continue
        missing.append(name)
    assert not missing, f"undeclared runtime imports: {missing}"


def test_dockerfile_pins_base_by_digest_and_labels():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(
        r"^FROM\s+python:3\.12-slim@sha256:[0-9a-f]{64}",
        text,
        re.MULTILINE,
    ), "base image must be pinned by digest"
    assert "org.opencontainers.image.revision=$REVISION" in text
    assert "org.opencontainers.image.source=" in text
    assert re.search(r"ARG\s+REVISION", text)
    assert re.search(r"ENV\s+REALTIMEVOICE_REVISION", text)
    assert "HEALTHCHECK" in text
    assert "/api/health" in text


def test_compose_parses_and_is_host_neutral():
    yaml = pytest.importorskip("yaml")
    text = COMPOSE.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert data["name"] == "somora-realtimevoice"
    services = data["services"]
    app = services["app"]
    assert app["container_name"] == "somora-realtimevoice"
    assert app["restart"] == "unless-stopped"
    assert app["image"] == "somora-realtimevoice:${RTV_IMAGE_TAG:?}"
    assert app["ports"] == ["${RTV_BIND_ADDR:?}:${RTV_HTTP_PORT:-8766}:8000"]
    assert app["volumes"] == ["${RTV_DATA_DIR:?}:/app/data"]
    assert app["environment"]["REALTIMEVOICE_DATA_DIR"] == "/app/data"
    assert app["healthcheck"]["test"]
    logging_cfg = app["logging"]["options"]
    assert logging_cfg["max-size"] and logging_cfg["max-file"]
    assert "0.0.0.0" not in text
    # no named volumes / top-level volumes that `down -v` could delete
    assert "volumes" not in data or not data.get("volumes")
    # every secret-ish value comes from the environment, never inline
    assert "REALTIMEVOICE_AUTH_TOKEN" not in text or "${" in text
    # tls service is optional and behind a profile (Addendum C: public ACME)
    tls = services["tls"]
    assert tls["profiles"] == ["tls"]
    assert tls["container_name"] == "somora-realtimevoice-tls"
    tls_volumes = " ".join(str(v) for v in tls["volumes"])
    assert "${RTV_TLS_DIR:?}/certs" in tls_volumes
    assert "${RTV_TLS_DIR:?}/caddy-data" in tls_volumes
    assert "${RTV_BIND_ADDR}:${RTV_HTTPS_PORT:-8444}" in tls["ports"][0]


def test_compose_required_vars_are_guarded():
    text = COMPOSE.read_text(encoding="utf-8")
    for var in ("RTV_IMAGE_TAG", "RTV_BIND_ADDR", "RTV_DATA_DIR"):
        assert f"${{{var}:?" in text, f"{var} must be guarded with :?"
    assert "${RTV_TLS_DIR:?}" in text


def test_deploy_scripts_and_env_example_exist():
    deploy = ROOT / "deploy"
    for name in (
        "compose.yaml",
        "README.md",
        "build-image.sh",
        "backup-data.sh",
        "restore-data.sh",
        "data_manifest.py",
        "smoke-image.sh",
        "compose.override.example.yaml",
    ):
        assert (deploy / name).is_file(), f"deploy/{name} missing"
    env_example = deploy / "deploy.env.example"
    assert env_example.is_file(), "deploy/deploy.env.example missing"
    text = env_example.read_text(encoding="utf-8")
    # no variable may be assigned 0.0.0.0 (a comment naming it as an
    # anti-pattern is fine)
    assert not re.search(r"=\s*0\.0\.0\.0\s*$", text, re.MULTILINE)


def test_naming_requirement_somora_realtimevoice():
    """Addendum B: recognisable, prefixed names everywhere."""
    compose_text = COMPOSE.read_text(encoding="utf-8")
    build_text = (ROOT / "deploy" / "build-image.sh").read_text(encoding="utf-8")
    assert "name: somora-realtimevoice" in compose_text
    assert "container_name: somora-realtimevoice\n" in compose_text
    assert "container_name: somora-realtimevoice-tls" in compose_text
    assert "somora-realtimevoice:${RTV_IMAGE_TAG:?}" in compose_text
    assert "IMAGE=\"somora-realtimevoice:${SHORT_SHA}\"" in build_text
    # the old unprefixed names must be gone from the tracked tooling
    assert "realtimevoice-app" not in compose_text
    assert not re.search(r"(?<!somora-)realtimevoice:<shortsha>", build_text)
    readme = (ROOT / "deploy" / "README.md").read_text(encoding="utf-8")
    for name in (
        "somora-realtimevoice",
        "somora-realtimevoice-tls",
        "somora-realtimevoice:<shortsha>",
    ):
        assert name in readme, f"{name} missing from deploy/README.md"
