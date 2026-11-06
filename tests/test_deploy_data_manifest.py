"""``deploy/data_manifest.py``: secret-free inventory of a runtime data dir.

The manifest is the safety net for backup/restore: it counts catalog objects
and verifies that every stored voice sample file is present and
content-addressed (sha256 in the filename == sha256 of the bytes). It must
NEVER emit credential, token, transcript or URL values — only counts, ids and
hashes. Non-zero exit on missing/mismatching samples or an explicit compare
difference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "data_manifest.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )


def _write_wav(path: Path, payload: bytes) -> str:
    """Write bytes whose filename is their content sha256 (content-addressed)."""
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest


def _sample_path(data_dir: Path, voice_id: str, digest: str) -> Path:
    return data_dir / "voices" / voice_id / f"{digest}.wav"


def _settings(voice_id: str, digest: str) -> dict:
    """A v3 settings dict carrying every secret-like field the manifest must
    never leak, plus catalog objects to count."""
    return {
        "version": 3,
        "session": {"instructions": "SECRET INSTRUCTIONS TEXT", "language": "de"},
        "profiles": {
            "stt": {"stt-a": {"id": "stt-a", "role": "stt", "base_url": "http://a.invalid/v1", "api_key": "sk-ROW-A"}},
            "llm": {"llm-a": {"id": "llm-a", "role": "llm", "base_url": "http://b.invalid/v1", "api_key": "sk-ROW-B"}},
            "tts": {"tts-a": {"id": "tts-a", "role": "tts", "base_url": "http://c.invalid/v1", "api_key": "sk-ROW-C"}},
        },
        "default_set": "prof-1",
        "voice_library": {"transcription_stt_profile": "stt-a"},
        "voices": {
            voice_id: {
                "id": voice_id,
                "display_name": "Voice",
                "language": "de",
                "samples": [
                    {
                        "id": "smp_" + digest[:16],
                        "sha256": digest,
                        "source": "upload",
                        "duration_ms": 1000,
                        "sample_rate": 16000,
                        "channels": 1,
                        "size_bytes": 1234,
                        "transcript": "SECRET TRANSCRIPT TEXT",
                        "created_at": 1.0,
                    }
                ],
                "reference_sample_id": "smp_" + digest[:16],
                "created_at": 1.0,
                "updated_at": 1.0,
            }
        },
        "conversation_profiles": {
            "prof-1": {
                "id": "prof-1",
                "display_name": "Profile One",
                "description": "",
                "spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-a"},
                "enabled": True,
                "history": [
                    {"saved_at": 1.0, "spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-a"}},
                    {"saved_at": 2.0, "spec": {"stt_profile": "stt-a", "llm_profile": "llm-a", "tts_profile": "tts-a"}},
                ],
                "created_at": 1.0,
                "updated_at": 2.0,
            },
            "prof-2": {
                "id": "prof-2",
                "display_name": "Profile Two",
                "spec": {},
                "enabled": False,
                "created_at": 1.0,
                "updated_at": 1.0,
            },
        },
        "server_hint_url": "http://internal.invalid:9999/secret",
        "auth_token": "SECRET-INSTANCE-TOKEN",
    }


def _make_data_dir(tmp_path: Path) -> tuple[Path, str]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = b"RIFF....WAVE fake sample bytes for manifest test"
    digest = _write_wav(_sample_path(data_dir, "voice-a", hashlib.sha256(payload).hexdigest()), payload)
    (data_dir / "settings.json").write_text(json.dumps(_settings("voice-a", digest)))
    (data_dir / "server.json").write_text(json.dumps({"auth_token": "SECRET-INSTANCE-TOKEN"}))
    return data_dir, digest


def test_manifest_is_valid_json_and_counts(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path)
    proc = _run(str(data_dir))
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["settings_version"] == 3
    assert body["default_set"] == "prof-1"
    assert body["transcription_stt_profile"] == "stt-a"
    assert body["counts"]["conversation_profiles"] == 2
    assert body["counts"]["enabled_sets"] == ["prof-1"]
    assert body["counts"]["set_history"] == {"prof-1": 2, "prof-2": 0}
    assert body["counts"]["backend_rows"] == {"stt": 1, "llm": 1, "tts": 1}
    assert "presets" not in body["counts"]
    assert body["counts"]["voices"] == 1
    assert body["counts"]["samples"] == 1
    assert len(body["samples"]) == 1
    sample = body["samples"][0]
    assert sample["voice_id"] == "voice-a"
    assert sample["present"] is True
    assert sample["sha256_matches_filename"] is True
    assert sample["bytes"] > 0
    assert body["settings_sha256"] and len(body["settings_sha256"]) == 64
    assert body["total_bytes"] > 0


def test_manifest_never_leaks_secrets(tmp_path):
    data_dir, digest = _make_data_dir(tmp_path)
    proc = _run(str(data_dir))
    assert proc.returncode == 0, proc.stderr
    for secret in (
        "SECRET INSTRUCTIONS TEXT",
        "sk-ROW-A",
        "sk-ROW-B",
        "sk-ROW-C",
        "SECRET TRANSCRIPT TEXT",
        "SECRET-INSTANCE-TOKEN",
        "http://internal.invalid:9999/secret",
        "http://a.invalid/v1",
        "http://c.invalid/v1",
    ):
        assert secret not in proc.stdout, f"manifest leaked {secret!r}"
    # the digest is allowed (it is a hash), transcripts are not
    assert digest in proc.stdout


def test_manifest_missing_sample_fails(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path)
    for wav in (data_dir / "voices").rglob("*.wav"):
        wav.unlink()
    proc = _run(str(data_dir))
    assert proc.returncode != 0
    assert "missing" in (proc.stdout + proc.stderr).lower()


def test_manifest_hash_mismatch_fails(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path)
    for wav in (data_dir / "voices").rglob("*.wav"):
        wav.write_bytes(wav.read_bytes() + b"tampered")
    proc = _run(str(data_dir))
    assert proc.returncode != 0
    assert "mismatch" in (proc.stdout + proc.stderr).lower()


def test_manifest_compare_reports_differences(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path)
    good = _run(str(data_dir))
    assert good.returncode == 0, good.stderr
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(good.stdout)

    # unchanged data dir: compare must succeed
    same = _run(str(data_dir), "--compare", str(manifest_path))
    assert same.returncode == 0, same.stdout + same.stderr

    # change a sample file: compare must fail
    for wav in (data_dir / "voices").rglob("*.wav"):
        wav.write_bytes(b"different bytes")
    changed = _run(str(data_dir), "--compare", str(manifest_path))
    assert changed.returncode != 0


def test_manifest_compare_reports_a_switched_off_set(tmp_path):
    data_dir, digest = _make_data_dir(tmp_path)
    good = _run(str(data_dir))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(good.stdout)
    settings = _settings("voice-a", digest)
    settings["conversation_profiles"]["prof-1"]["enabled"] = False
    (data_dir / "settings.json").write_text(json.dumps(settings))
    changed = _run(str(data_dir), "--compare", str(manifest_path))
    assert changed.returncode == 2
    assert "enabled_sets" in changed.stderr


def test_manifest_compare_missing_file_fails(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path)
    missing = tmp_path / "nope.json"
    proc = _run(str(data_dir), "--compare", str(missing))
    assert proc.returncode != 0
