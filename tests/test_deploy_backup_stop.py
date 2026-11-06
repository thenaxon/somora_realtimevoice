"""``deploy/backup-data.sh --stop-container`` semantics (docker-free).

A fake ``docker`` shim on PATH records and controls stop/start/ps/inspect so the
stop-before-archive contract can be tested without a daemon:

* ``--stop-container <name>`` (repeatable) stops the named containers with
  ``docker stop --time 20`` and starts EXACTLY the ones that were running again
  (trap on EXIT/ERR), verifying they run afterwards;
* if a requested stop fails the script aborts non-zero and creates NO archive;
* ``--allow-running`` opts into archiving a running data dir and records
  ``"consistent": false`` (``true`` when the stop succeeded);
* the legacy ``--stop <dir>`` fails with a clear pointer, no silent fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "deploy" / "backup-data.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="bash/tar required",
)

SHIM = r"""#!/usr/bin/env bash
# fake docker: log every call, control exit codes and running state via files
LOG="$FAKE_DOCKER_LOG"
echo "docker $*" >> "$LOG"
cmd="$1"; shift || true
case "$cmd" in
  stop)
    # `docker stop --time 20 <name> ...`
    while [ $# -gt 0 ]; do
      case "$1" in
        --time) shift; [ $# -gt 0 ] && shift;;
        *) name="$1"; shift
           if [ -f "$FAKE_DOCKER_STOP_FAIL/$name" ]; then exit 1; fi
           rm -f "$FAKE_DOCKER_RUNNING/$name" ;;
      esac
    done
    exit 0
    ;;
  start)
    for name in "$@"; do
      if [ -f "$FAKE_DOCKER_START_FAIL/$name" ]; then exit 1; fi
      : > "$FAKE_DOCKER_RUNNING/$name"
    done
    exit 0
    ;;
  inspect)
    # `docker inspect -f {{.State.Running}} <name>`
    name="${@: -1}"
    if [ -f "$FAKE_DOCKER_RUNNING/$name" ]; then echo true; else echo false; fi
    exit 0
    ;;
  ps)
    ls "$FAKE_DOCKER_RUNNING" 2>/dev/null
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
"""


@pytest.fixture
def fake_docker(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "docker"
    shim.write_text(SHIM)
    shim.chmod(0o755)

    state = {
        "log": tmp_path / "docker.log",
        "running": tmp_path / "running",
        "stop_fail": tmp_path / "stop_fail",
        "start_fail": tmp_path / "start_fail",
    }
    for key in ("running", "stop_fail", "start_fail"):
        state[key].mkdir()
    state["log"].write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(state["log"])
    env["FAKE_DOCKER_RUNNING"] = str(state["running"])
    env["FAKE_DOCKER_STOP_FAIL"] = str(state["stop_fail"])
    env["FAKE_DOCKER_START_FAIL"] = str(state["start_fail"])
    state["env"] = env
    state["bindir"] = bindir
    return state


def _make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "settings.json").write_text(json.dumps({"version": 2, "voices": {}}))
    (data_dir / "server.json").write_text(json.dumps({"auth_token": "x"}))
    return data_dir


def _run(data_dir, backup_dir, env, *args):
    return subprocess.run(
        ["bash", str(BACKUP), str(data_dir), str(backup_dir), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
        env=env,
    )


def _mark_running(state, *names):
    for name in names:
        (state["running"] / name).write_text("")


def _calls(state) -> list[str]:
    return [ln for ln in state["log"].read_text().splitlines() if ln]


def test_stop_container_archives_and_records_consistent_true(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    _mark_running(fake_docker, "somora-realtimevoice")
    proc = _run(
        data_dir, backup_dir, fake_docker["env"],
        "--stop-container", "somora-realtimevoice",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    archives = list(backup_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    manifest = json.loads(next(backup_dir.glob("*.manifest.json")).read_text())
    assert manifest["consistent"] is True
    calls = _calls(fake_docker)
    assert any("stop --time 20 somora-realtimevoice" in c for c in calls)
    assert any(c.startswith("docker start somora-realtimevoice") for c in calls)
    # started again and running
    assert (fake_docker["running"] / "somora-realtimevoice").is_file()


def test_stop_failure_aborts_without_archive(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    _mark_running(fake_docker, "somora-realtimevoice")
    (fake_docker["stop_fail"] / "somora-realtimevoice").write_text("")
    proc = _run(
        data_dir, backup_dir, fake_docker["env"],
        "--stop-container", "somora-realtimevoice",
    )
    assert proc.returncode != 0
    assert not list(backup_dir.glob("*.tar.gz")), "no archive on failed stop"
    assert "stop" in (proc.stdout + proc.stderr).lower()
    # the container that could not be stopped is not 'started'
    assert (fake_docker["running"] / "somora-realtimevoice").is_file()


def test_allow_running_archives_inconsistent(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    _mark_running(fake_docker, "somora-realtimevoice")
    (fake_docker["stop_fail"] / "somora-realtimevoice").write_text("")
    proc = _run(
        data_dir, backup_dir, fake_docker["env"],
        "--stop-container", "somora-realtimevoice", "--allow-running",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads(next(backup_dir.glob("*.manifest.json")).read_text())
    assert manifest["consistent"] is False
    assert list(backup_dir.glob("*.tar.gz"))


def test_multiple_containers_started_even_when_tar_fails(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    _mark_running(fake_docker, "somora-realtimevoice", "somora-realtimevoice-tls")
    # make tar fail by injecting a failing tar into a wrapping PATH dir
    wrap = tmp_path / "wrap"
    wrap.mkdir()
    (wrap / "tar").write_text("#!/usr/bin/env bash\nexit 1\n")
    (wrap / "tar").chmod(0o755)
    env = dict(fake_docker["env"])
    env["PATH"] = f"{wrap}:{env['PATH']}"
    proc = _run(
        data_dir, backup_dir, env,
        "--stop-container", "somora-realtimevoice",
        "--stop-container", "somora-realtimevoice-tls",
    )
    assert proc.returncode != 0
    calls = _calls(fake_docker)
    # both were stopped and both must be started again despite the tar failure
    assert any("stop --time 20 somora-realtimevoice" in c for c in calls)
    assert any("stop --time 20 somora-realtimevoice-tls" in c for c in calls)
    started = [c for c in calls if c.startswith("docker start ")]
    assert started, "containers must be restarted even when archiving fails"
    restarted = " ".join(started)
    assert "somora-realtimevoice" in restarted
    assert "somora-realtimevoice-tls" in restarted
    assert (fake_docker["running"] / "somora-realtimevoice").is_file()
    assert (fake_docker["running"] / "somora-realtimevoice-tls").is_file()
    assert not list(backup_dir.glob("*.tar.gz"))


def test_stopped_container_is_not_started_again(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    # container exists but is NOT running: must not be started
    proc = _run(
        data_dir, backup_dir, fake_docker["env"],
        "--stop-container", "somora-realtimevoice",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = _calls(fake_docker)
    assert not any(c.startswith("docker start ") for c in calls), calls
    manifest = json.loads(next(backup_dir.glob("*.manifest.json")).read_text())
    # nothing was stopped -> a running-dir archive is trivially consistent
    assert manifest["consistent"] is True


def test_legacy_stop_flag_is_a_clear_error(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    project = tmp_path / "compose"
    project.mkdir()
    proc = _run(data_dir, backup_dir, fake_docker["env"], "--stop", str(project))
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "--stop-container" in combined
    assert not list(backup_dir.glob("*.tar.gz"))


def test_plain_backup_is_consistent_true(tmp_path, fake_docker):
    data_dir = _make_data_dir(tmp_path)
    backup_dir = tmp_path / "backups"
    proc = _run(data_dir, backup_dir, fake_docker["env"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads(next(backup_dir.glob("*.manifest.json")).read_text())
    assert manifest["consistent"] is True


def test_usage_documents_stop_container(tmp_path, fake_docker):
    proc = subprocess.run(
        ["bash", str(BACKUP)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
        env=fake_docker["env"],
    )
    assert proc.returncode != 0
    assert "--stop-container" in (proc.stdout + proc.stderr)


def test_compose_override_example_is_host_neutral():
    override = ROOT / "deploy" / "compose.override.example.yaml"
    assert override.is_file(), "deploy/compose.override.example.yaml missing"
    text = override.read_text(encoding="utf-8")
    # placeholder documentation subnet only; no real host values
    assert "192.0.2.0/24" in text
    assert "compose.override.yaml" in text
    # must not be an internal-CA style secret or a real private range
    assert not any(x in text for x in ("10.0.", "172.16.", "192.168.1."))


def test_gitignore_ignores_local_override_but_not_example():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "deploy/compose.override.yaml" in gitignore
    # the example must NOT be ignored
    proc = subprocess.run(
        ["git", "check-ignore", "deploy/compose.override.example.yaml"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode != 0, "the example override must stay tracked"
