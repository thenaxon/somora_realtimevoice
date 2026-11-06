"""Node-based unit test of the microphone selection + resampler helpers (F2/F4).

Runs tests/mic_resample.mjs with Node against the real shipped UI modules:
deviceId constraint building, remembered-device resolution (gone -> system
default reported), and the streaming 48k/44.1k -> 24k PCM16 resampler.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "mic_resample.mjs"
MIC = ROOT / "src" / "realtimevoice" / "web" / "ui" / "mic-devices.js"
AUDIO = ROOT / "src" / "realtimevoice" / "web" / "ui" / "audio.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_mic_and_resampler_helpers():
    assert MIC.exists() and AUDIO.exists()
    result = subprocess.run(
        ["node", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"mic/resampler test failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL RESAMPLER/MIC TESTS PASSED" in result.stdout