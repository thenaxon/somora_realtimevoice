"""Node-based unit test of the browser worklets (play ring + capture).

Runs tests/worklet_ring.mjs with Node against the real shipped worklet.js
source: 15 s of audio must fully drain the 60 s ring in order without
overflow, clear() must not leak, overflow must be counted explicitly, and
capture quantization must preserve negative extremes (the old char-code
encoding bug corrupted exactly those).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "worklet_ring.mjs"
WORKLET = ROOT / "src" / "realtimevoice" / "web" / "worklet.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_worklet_ring_preserves_samples():
    assert WORKLET.exists()
    result = subprocess.run(
        ["node", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"worklet test failed:\n{result.stdout}\n{result.stderr}"
    assert "WORKLET TESTS PASSED" in result.stdout
