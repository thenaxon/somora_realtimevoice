"""Node-based unit test of the shared sample playback controller (F6).

Runs tests/sample_player.mjs against the real shipped sample-player.js:
play -> pause -> resume keeps position and the same element; switching samples
stops the first; rapid clicks never create overlapping audio; ended/error
reset; stop() tears down.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "sample_player.mjs"
MODULE = ROOT / "src" / "realtimevoice" / "web" / "ui" / "sample-player.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_sample_player_state_machine():
    assert MODULE.exists()
    result = subprocess.run(
        ["node", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"sample player test failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL SAMPLE PLAYER TESTS PASSED" in result.stdout