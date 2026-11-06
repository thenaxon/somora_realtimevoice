"""``deploy/backup-data.sh`` and ``deploy/restore-data.sh`` behavioural checks.

Runnable without a Docker daemon: the scripts operate on directories only. A
backup is a ``tar.gz`` (0600) plus its ``.sha256`` and a manifest JSON; restore
verifies the checksum, refuses a non-empty target and re-runs the manifest
check against the stored manifest. No secrets ever appear in the archive's
manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "deploy" / "backup-data.sh"
RESTORE = ROOT / "deploy" / "restore-data.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="bash/tar required",
)


def _make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "settings.json").write_text(
        json.dumps({"version": 2, "stt": {"api_key": "sk-SECRET"}, "voices": {}})
    )
    (data_dir / "server.json").write_text(json.dumps({"auth_token": "SECRET-TOKEN"}))
    (data_dir / "voices" / "voice-a").mkdir(parents=True)
    (data_dir / "voices" / "voice-a" / "x.wav").write_bytes(b"RIFFsample-bytes")
    return data_dir


def _run_backup(data_dir: Path, backup_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BACKUP), str(data_dir), str(backup_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )


def test_backup_creates_archive_checksum_and_manifest(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    proc = _run_backup(data_dir, backup_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    archives = list(backup_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    archive = archives[0]
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    assert checksum.is_file()
    # 0600 archive and checksum
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert stat.S_IMODE(checksum.stat().st_mode) == 0o600
    # a manifest was produced and contains no secrets
    manifests = list(backup_dir.glob("*.manifest.json"))
    assert len(manifests) == 1
    manifest_text = manifests[0].read_text()
    assert "SECRET-TOKEN" not in manifest_text
    assert "sk-SECRET" not in manifest_text
    body = json.loads(manifest_text)
    assert body["counts"]["voices"] == 0

    # checksum actually matches the archive
    recorded = checksum.read_text().split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert recorded == actual


def test_backup_does_not_delete_recent_and_prunes_only_own_pattern(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    foreign = backup_dir / "keep-me.txt"
    foreign.write_text("unrelated")
    assert _run_backup(data_dir, backup_dir).returncode == 0
    assert foreign.is_file(), "backup must not touch unrelated files"


def test_backup_without_stop_is_self_contained(tmp_path):
    """A plain backup needs neither docker nor a compose project dir and must
    still produce a valid, secret-free archive."""
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    proc = _run_backup(data_dir, backup_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list(backup_dir.glob("*.tar.gz"))


def test_restore_refuses_non_empty_target(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    assert _run_backup(data_dir, backup_dir).returncode == 0
    archive = next(backup_dir.glob("*.tar.gz"))

    target = tmp_path / "target"
    target.mkdir()
    (target / "unexpected.txt").write_text("data")
    proc = subprocess.run(
        ["bash", str(RESTORE), str(archive), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert proc.returncode != 0
    assert "non-empty" in (proc.stdout + proc.stderr).lower()


def test_restore_refuses_bad_checksum(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    assert _run_backup(data_dir, backup_dir).returncode == 0
    archive = next(backup_dir.glob("*.tar.gz"))
    # corrupt the archive so its sha256 no longer matches
    archive.write_bytes(archive.read_bytes() + b"corruption")

    target = tmp_path / "target"
    target.mkdir()
    proc = subprocess.run(
        ["bash", str(RESTORE), str(archive), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert proc.returncode != 0
    assert "checksum" in (proc.stdout + proc.stderr).lower() or "sha256" in (
        proc.stdout + proc.stderr
    ).lower()


def test_restore_roundtrip_into_empty_dir(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    assert _run_backup(data_dir, backup_dir).returncode == 0
    archive = next(backup_dir.glob("*.tar.gz"))

    target = tmp_path / "restored"
    target.mkdir()
    proc = subprocess.run(
        ["bash", str(RESTORE), str(archive), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # data content restored
    assert (target / "settings.json").is_file()
    assert (target / "server.json").is_file()
    assert (target / "voices" / "voice-a" / "x.wav").read_bytes() == b"RIFFsample-bytes"
