#!/usr/bin/env python3
"""Browser-scratch isolation for the UI harnesses.

WHY THIS EXISTS
---------------
The two UI harnesses drive a real Chromium renderer. On CI/dev containers the
default temporary directory (often a small tmpfs on ``/tmp``) can be completely
full. Chromium then fails page loads with ``net::ERR_INSUFFICIENT_RESOURCES``
and the renderer process aborts mid-flow — the harness sees
``Page.goto: Target crashed`` for every subsequent step. When the scratch path
is instead moved to a roomy filesystem, Chromium aborts at startup with
``FATAL:process_singleton_posix.cc Socket path too long`` because its Unix
socket lives under ``$TMPDIR`` and must stay under ~104 bytes.

This module gives both harnesses a scratch directory that is (a) on the same
filesystem as the repository (roomy) and (b) reachable through a SHORT
``/tmp``-rooted symlink so the Chromium singleton socket path stays short.
``prepare_scratch`` returns the short path; callers put it into ``TMPDIR`` /
``TEMP`` / ``TMP`` before launching Playwright and pass ``scratch_chromium_args``
to the browser. ``write_crash_report`` records a renderer crash's reason and the
browser stderr tail into the run dir so a future failure is diagnosable.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _install_cleanup(link: Path, backing: Path) -> None:
    """Remove an auto-created scratch dir (and its short symlink) at exit.

    The run dir (``--out-dir``) is separate and never touched. Set
    RTV_KEEP_SCRATCH=1 to keep the scratch for debugging.
    """
    if os.environ.get("RTV_KEEP_SCRATCH"):
        return

    def _cleanup() -> None:
        try:
            if link.is_symlink():
                link.unlink()
        except Exception:
            pass
        try:
            shutil.rmtree(backing, ignore_errors=True)
        except Exception:
            pass

    atexit.register(_cleanup)


def prepare_scratch(prefix: str = "rtv-e2e", explicit: str = "") -> Path:
    """Return a SHORT path usable as TMPDIR whose backing store has space.

    Backing store: ``<explicit>`` if given (kept as-is and NOT cleaned up), else
    a temp dir on the repository filesystem (removed at process exit). The
    returned path is a symlink rooted at the system temp dir (short, for
    Chromium's socket limit) pointing at the backing store. If the short symlink
    cannot be created the backing store itself is returned.
    """
    auto = not explicit
    if explicit:
        backing = Path(explicit).expanduser()
        backing.mkdir(parents=True, exist_ok=True)
    else:
        base = REPO / ".scratch-browser"
        base.mkdir(parents=True, exist_ok=True)
        backing = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(base)))
    # a /tmp-rooted symlink keeps the path short regardless of where the repo
    # lives; if the system temp dir is itself the backing store that's fine.
    system_tmp = tempfile.gettempdir()
    try:
        if Path(system_tmp).resolve() == backing.resolve():
            if auto:
                _install_cleanup(backing, backing)
            return backing
    except Exception:
        pass
    link = Path(system_tmp) / f"rtv-scratch-{os.getpid()}"
    try:
        if link.is_symlink() or link.exists():
            try:
                if link.resolve() != backing.resolve():
                    link.unlink()
                    link.symlink_to(backing)
            except Exception:
                pass
        else:
            link.symlink_to(backing)
        if auto:
            _install_cleanup(link, backing)
        # Chromium adds "/org.chromium.Chromium.XXXX/SingletonSocket"; keep the
        # resulting path under the ~104 byte sun_path limit.
        if len(str(link)) + 40 < 100:
            return link
        return backing
    except Exception:
        if auto:
            _install_cleanup(backing, backing)
        return backing


def apply_env(scratch: Path) -> None:
    """Point Python and Chromium temporary files at ``scratch``.

    ``XDG_CACHE_HOME`` is deliberately NOT touched: Playwright resolves its
    bundled browser executables relative to that variable.
    """
    value = str(scratch)
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = value
    # Python caches the first resolved temp dir; reset it so later
    # tempfile.mkdtemp() calls land in the scratch store too.
    tempfile.tempdir = value


def scratch_chromium_args(scratch: Path, log_file: Path | None = None) -> list[str]:
    """Chromium flags routing cache/crashpad off the default (possibly full) tmp."""
    args = [
        f"--disk-cache-dir={scratch / 'chromium-cache'}",
        f"--crash-dumps-dir={scratch / 'chromium-crash'}",
        "--disable-crash-reporter",
    ]
    if log_file is not None:
        args += ["--enable-logging", "--v=1", f"--log-file={log_file}"]
    return args


def install_crash_logging(page, run_dir: Path, browser_stderr: Path | None = None) -> dict:
    """Attach crash/console logging; return a mutable state dict.

    A renderer crash is recorded to ``run_dir/browser-crash.txt`` together with
    the tail of the captured browser stderr (Playwright's own stderr is terse,
    but Chromium writes its FATAL reason there).
    """
    state: dict = {"crashed": False, "reason": ""}

    def on_crash(_page) -> None:
        state["crashed"] = True
        state["reason"] = "renderer process crashed (page.on('crash'))"
        _dump(run_dir, browser_stderr, state)

    page.on("crash", on_crash)
    return state


def record_crash(run_dir: Path, note: str, browser_stderr: Path | None = None) -> None:
    _dump(run_dir, browser_stderr, {"crashed": True, "reason": note})


def _dump(run_dir: Path, browser_stderr: Path | None, state: dict) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"renderer crash: {state.get('reason', 'unknown')}"]
        if browser_stderr and browser_stderr.exists():
            tail = browser_stderr.read_text(errors="replace").splitlines()[-80:]
            lines.append("--- browser stderr tail ---")
            lines.extend(tail)
        (run_dir / "browser-crash.txt").write_text("\n".join(lines) + "\n")
    except Exception:
        pass


def panic_guard() -> None:
    """Print a one-line hint if the scratch setup itself could not help."""
    print(
        f"[browser_scratch] TMPDIR={os.environ.get('TMPDIR')}",
        file=sys.stderr,
    )