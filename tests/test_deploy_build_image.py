"""``deploy/build-image.sh`` argument safety (runnable without a Docker daemon).

The builder must archive a COMMIT (never the working tree), so a dirty checkout
or an unknown revision is refused before any Docker call. We exercise only the
validation path; the actual ``docker build`` is out of scope here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "deploy" / "build-image.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BUILD), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )


def test_build_requires_a_commit_argument():
    proc = _run()
    assert proc.returncode != 0
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_build_refuses_unknown_commit():
    proc = _run("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    assert proc.returncode != 0
    assert "unknown" in (proc.stdout + proc.stderr).lower()


def test_build_refuses_dirty_working_tree(tmp_path):
    """A tracked file modified but uncommitted must make the builder refuse
    (it builds from git archive, so a dirty tree would silently drop changes)."""
    probe = ROOT / "README.md"
    original = probe.read_bytes()
    try:
        probe.write_bytes(original + b"\n<!-- dirty probe -->\n")
        proc = _run("HEAD")
        assert proc.returncode != 0
        assert "dirty" in (proc.stdout + proc.stderr).lower()
    finally:
        probe.write_bytes(original)
