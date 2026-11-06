"""Addendum C: publicly trusted HTTPS via Let's Encrypt DNS-01 (no internal CA).

The optional `tls` profile terminates HTTPS for a configurable LAN hostname
resolved by internal DNS; certificates come from Let's Encrypt via DNS-01 only.
Two operating modes exist, each with its own Caddyfile:

* `files` — operator provisions/renews the certificate elsewhere (DNS-01) and
  drops `fullchain.pem` + `privkey.pem` into `${RTV_TLS_DIR}/certs` (read-only);
* `dns-cloudflare` — automatic renewal via the Cloudflare DNS plugin, with the
  API token supplied only through an env_file outside the repo.

There must be NO internal CA anywhere: no `tls internal`, no root-trust
instructions in any tracked file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "deploy" / "compose.yaml"
CADDY_FILES = ROOT / "deploy" / "tls" / "Caddyfile.files"
CADDY_DNS = ROOT / "deploy" / "tls" / "Caddyfile.dns-cloudflare"
TLS_DOCKERFILE = ROOT / "deploy" / "tls" / "Dockerfile.dns-cloudflare"
BUILD_TLS = ROOT / "deploy" / "build-tls-image.sh"
CHECK_CERT = ROOT / "deploy" / "check-cert.sh"
README = ROOT / "deploy" / "README.md"
ENV_EXAMPLE = ROOT / "deploy" / "deploy.env.example"


def _tracked_files() -> list[Path]:
    """All tracked text assets that must not mention an internal CA."""
    return [
        COMPOSE,
        CADDY_FILES,
        CADDY_DNS,
        TLS_DOCKERFILE,
        BUILD_TLS,
        CHECK_CERT,
        README,
        ENV_EXAMPLE,
        ROOT / "deploy" / "deploy.env.example",
    ]


def test_no_internal_ca_or_root_trust_anywhere():
    for path in _tracked_files():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "tls internal" not in text, f"{path.name}: internal CA is forbidden"
        assert not re.search(r"trust (the )?root", text, re.IGNORECASE), (
            f"{path.name}: root-trust wording is forbidden"
        )
        # a negated mention ("no internal CA") is documentation, not a config
        lowered = text.lower()
        for match in re.finditer(r"internal ca", lowered):
            window = lowered[max(0, match.start() - 5) : match.start()]
            assert "no " in window, f"{path.name}: internal CA is forbidden"


def test_files_caddyfile_uses_provided_pem_pair():
    assert CADDY_FILES.is_file(), "deploy/tls/Caddyfile.files missing"
    text = CADDY_FILES.read_text(encoding="utf-8")
    assert "{$RTV_TLS_HOSTNAME}:8444" in text
    assert "tls /certs/fullchain.pem /certs/privkey.pem" in text
    assert "reverse_proxy app:8000" in text
    assert "tls internal" not in text


def test_dns_cloudflare_caddyfile_uses_acme_dns01_only():
    assert CADDY_DNS.is_file(), "deploy/tls/Caddyfile.dns-cloudflare missing"
    text = CADDY_DNS.read_text(encoding="utf-8")
    assert "{$RTV_TLS_HOSTNAME}:8444" in text
    assert "issuer acme" in text
    assert "dns cloudflare {env.CF_API_TOKEN}" in text
    assert "disable_http_challenge" in text
    assert "disable_tlsalpn_challenge" in text
    assert "RTV_ACME_EMAIL" in text
    assert "reverse_proxy app:8000" in text
    assert "tls internal" not in text


@pytest.mark.parametrize("path", [CADDY_FILES, CADDY_DNS])
def test_caddyfiles_are_format_clean(path):
    """Structurally `caddy fmt`-clean: tab indentation, no trailing spaces,
    no blank line between the comment header and the opening block."""
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        assert "\t" in line or not line.startswith(" "), (
            f"{path.name}:{lineno}: indentation must be tabs, not spaces"
        )
        assert line == line.rstrip(), f"{path.name}:{lineno}: trailing whitespace"
    assert "\n\n\n" not in text, f"{path.name}: consecutive blank lines"
    # no space-indented continuation lines (caddy fmt uses tabs)
    assert not re.search(r"^ +", text, re.MULTILINE), f"{path.name}: space indentation"


@pytest.mark.skipif(shutil.which("caddy") is None, reason="caddy not on PATH")
@pytest.mark.parametrize("path", [CADDY_FILES, CADDY_DNS])
def test_caddy_fmt_reports_no_diff(path):
    proc = subprocess.run(
        ["caddy", "fmt", "--diff", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{path.name} is not caddy-fmt clean:\n{proc.stdout}{proc.stderr}"


def test_compose_tls_profile_public_and_mode_selectable():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    tls = data["services"]["tls"]
    assert tls["profiles"] == ["tls"]
    assert tls["container_name"] == "somora-realtimevoice-tls"
    text = COMPOSE.read_text(encoding="utf-8")
    # site address hostname is a required var
    assert "${RTV_TLS_HOSTNAME:?}" in text
    # mode selects one of the two Caddyfiles
    assert "Caddyfile.${RTV_TLS_MODE:?}" in text
    assert "${RTV_TLS_MODE:?}" in text
    # cert/state bind mounts, never named volumes
    volumes = " ".join(str(v) for v in tls["volumes"])
    assert "${RTV_TLS_DIR:?}/certs" in volumes
    assert "${RTV_TLS_DIR:?}/caddy-data" in volumes
    # cloudflare token only via env_file outside the repo
    text_joined = " ".join(str(v) for v in tls.get("env_file", []))
    assert "cloudflare.env" in text_joined
    assert "CF_API_TOKEN" not in text, "token must not be inlined in compose"


def test_dns_cloudflare_image_is_pinned_and_uses_xcaddy():
    assert TLS_DOCKERFILE.is_file(), "deploy/tls/Dockerfile.dns-cloudflare missing"
    text = TLS_DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"FROM\s+caddy:[0-9.]+-builder-alpine@sha256:[0-9a-f]{64}", text)
    assert "xcaddy" in text
    assert "github.com/caddy-dns/cloudflare@v" in text
    assert re.search(r"FROM\s+caddy:[0-9.]+-alpine@sha256:[0-9a-f]{64}", text)


def test_build_tls_image_script_exists_and_is_commit_safe():
    assert BUILD_TLS.is_file(), "deploy/build-tls-image.sh missing"
    text = BUILD_TLS.read_text(encoding="utf-8")
    assert "somora-realtimevoice-tls:" in text
    assert "Dockerfile.dns-cloudflare" in text


def test_check_cert_script_prints_metadata_only():
    assert CHECK_CERT.is_file(), "deploy/check-cert.sh missing"
    text = CHECK_CERT.read_text(encoding="utf-8")
    assert "subject" in text
    assert "SAN" in text or "-ext subjectAltName" in text
    assert "notAfter" in text or "enddate" in text
    assert "RTV_CERT_MIN_DAYS" in text
    # must never print key material
    assert "privkey" not in text.lower()
    assert "BEGIN PRIVATE KEY" not in text


def test_env_example_documents_tls_vars():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for var in ("RTV_TLS_HOSTNAME", "RTV_TLS_MODE", "RTV_TLS_DIR", "RTV_HTTPS_PORT"):
        assert var in text, f"{var} missing from deploy.env.example"
