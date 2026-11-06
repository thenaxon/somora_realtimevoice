"""B4: two_profile_live.py — arg handling + summary building (offline).

The live harness is never run by pytest; these tests cover the pure pieces:
argument validation, URL building, PCM resampling and summary construction
with fake session results.
"""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from two_profile_live import (  # noqa: E402
    HarnessArgs,
    SessionState,
    _apply_event,
    build_summary,
    chunk_pcm,
    collect_response,
    create_out_dir,
    http_url_for,
    parse_args,
    resample_to_pcm16,
    sha8,
    validate_args,
    wav_to_pcm16_24k,
    ws_url_for,
)


def base_args(**overrides) -> HarnessArgs:
    values = dict(
        base_url="https://voice.example:8443",
        profile_a="clara-reference",
        profile_b="anna-agent",
        tts_preview_profile="tts-anna",
        out_dir="/tmp/m1-live-test",
    )
    values.update(overrides)
    return HarnessArgs(**values)


def test_validate_args_accepts_complete():
    assert validate_args(base_args()) == []


def test_validate_args_rejects_missing_and_bad():
    assert validate_args(base_args(base_url="ftp://x"))
    assert validate_args(base_args(profile_a=""))
    assert validate_args(base_args(profile_b=""))
    assert validate_args(base_args(tts_preview_profile=""))
    assert validate_args(base_args(out_dir=""))
    assert validate_args(base_args(profile_a="same", profile_b="same"))
    assert validate_args(base_args(chunk_ms=0))
    assert validate_args(base_args(silence_ms=-1))


def test_parse_args_and_dry_run(capsys):
    from two_profile_live import main

    rc = main(
        [
            "--base-url", "http://127.0.0.1:8765",
            "--profile-a", "a",
            "--profile-b", "b",
            "--tts-preview-profile", "tts",
            "--out-dir", "/tmp/nope-unique-dir",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run OK" in out
    assert "voice-route%3Aauto" in out
    assert "profile%3Ab" in out


def test_ws_and_http_url_building():
    assert ws_url_for("https://h:8443", "voice-route:auto") == "wss://h:8443/v1/realtime?model=voice-route%3Aauto"
    assert ws_url_for("http://h:8765/", "profile:b") == "ws://h:8765/v1/realtime?model=profile%3Ab"
    assert http_url_for("https://h:8443/", "/api/tts-synthesis") == "https://h:8443/api/tts-synthesis"


def test_sha8_and_transcript_only_hash():
    assert sha8("Wien") == sha8("Wien")
    assert len(sha8("Wie spät ist es gerade in Wien?")) == 8


def test_resample_changes_length():
    # 48 kHz mono 48 samples -> 24 kHz 24 samples
    pcm = struct.pack("<48h", *range(48))
    out = resample_to_pcm16(pcm, 48000, 1, 24000)
    assert len(out) == 24 * 2


def test_wav_to_pcm16_24k_roundtrip():
    import io
    import wave

    n = 2400
    pcm = struct.pack(f"<{n}h", *(i % 100 for i in range(n)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm)
    out = wav_to_pcm16_24k(buf.getvalue())
    assert len(out) == n  # 48k -> 24k halves the sample count


def test_wav_rejects_non_pcm16():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(24000)
        w.writeframes(b"\x00" * 240)
    with pytest.raises(ValueError):
        wav_to_pcm16_24k(buf.getvalue())


def test_chunk_pcm_20ms():
    pcm = bytes(2 * 24000 // 10)  # 100 ms
    chunks = list(chunk_pcm(pcm, 20))
    assert len(chunks) == 5
    assert all(len(c) == 960 for c in chunks)  # 20 ms * 24 kHz * 2 bytes


def fake_session(has_route=True, has_transcript=True, has_audio=True, route=None):
    result = {
        "session_id": "sess_x",
        "route": route or {"llm_model": "llm-a", "mode": "profile"},
        "transcript_sha8": ["abcd1234"],
        "transcript_chars": [12],
        "response_status": "completed",
        "audio_bytes": 48000,
        "timings": {"speech_to_transcript_ms": 100.0, "transcript_to_first_audio_ms": 200.0, "first_to_last_audio_ms": 800.0},
        "errors": [],
        "has_route": has_route,
        "has_transcript": has_transcript,
        "has_audio": has_audio,
    }
    return result


def test_build_summary_passes_when_routes_differ():
    a = fake_session(route={"llm_model": "llm-anna", "mode": "profile"})
    b = fake_session(route={"llm_model": "llm-bert", "mode": "profile"})
    summary = build_summary(session_a=a, session_b=b, profile_a="anna", profile_b="bert", utterance_chars=30)
    assert summary["routes_differ"] is True
    assert summary["passed"] is True
    assert summary["route_models"] == {"a": "llm-anna", "b": "llm-bert"}
    assert summary["sessions"]["A"]["audio_bytes"] == 48000


def test_build_summary_fails_when_routes_same():
    a = fake_session(route={"llm_model": "same"})
    b = fake_session(route={"llm_model": "same"})
    summary = build_summary(session_a=a, session_b=b, profile_a="a", profile_b="b", utterance_chars=30)
    assert summary["routes_differ"] is False
    assert summary["passed"] is False


def test_build_summary_fails_when_audio_or_transcript_missing():
    a = fake_session(route={"llm_model": "a"})
    b = fake_session(has_transcript=False, has_audio=False, route={"llm_model": "b"})
    summary = build_summary(session_a=a, session_b=b, profile_a="a", profile_b="b", utterance_chars=30)
    assert summary["passed"] is False


def test_summary_carries_no_transcript_text():
    a = fake_session(route={"llm_model": "a"})
    b = fake_session(route={"llm_model": "b"})
    summary = build_summary(session_a=a, session_b=b, profile_a="a", profile_b="b", utterance_chars=30)
    import json

    payload = json.dumps(summary)
    assert "Wie spät" not in payload
    assert "transcript" not in payload or '"transcript_sha8"' in payload


def test_apply_event_reduces_to_metadata_only():
    state = SessionState(label="A")
    _apply_event(
        state,
        {
            "type": "session.updated",
            "session": {"id": "s1", "x_realtimevoice": {"route": {"mode": "profile", "llm_model": "m"}}},
        },
    )
    _apply_event(state, {"type": "input_audio_buffer.speech_stopped"})
    _apply_event(
        state,
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Wie spät ist es?"},
    )
    import base64

    delta = base64.b64encode(bytes(2 * 2400)).decode()  # 100 ms @ 24k
    _apply_event(state, {"type": "response.output_audio.delta", "delta": delta})
    _apply_event(state, {"type": "response.done", "response": {"status": "completed"}})

    result = state.result()
    # transcript is reduced to sha8 + length; text never kept in the result
    assert result["transcript_sha8"] == [sha8("Wie spät ist es?")]
    assert result["transcript_chars"] == [len("Wie spät ist es?")]
    assert "Wie spät" not in json_dumps(result)
    assert result["has_route"] and result["has_transcript"] and result["has_audio"]
    assert result["audio_ms"] == 100.0
    assert result["response_status"] == "completed"
    assert "speech_to_transcript_ms" in result["timings"]


def json_dumps(value):
    import json

    return json.dumps(value, ensure_ascii=False)


def test_apply_event_errors_recorded():
    state = SessionState(label="B")
    _apply_event(state, {"type": "error", "error": {"code": "backend_unreachable"}})
    assert state.result()["errors"] == ["backend_unreachable"]


class _FakeClock:
    """Monotonic clock that jumps 6 s (past the 5 s poll window) on its first
    reading, so the harness observes a silence longer than the poll window."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _SilentThenDoneWS:
    """Fake WS: the first recv() raises TimeoutError (the 5 s poll window
    elapses silently), then audio + response.done arrive. A buggy collector
    that treats any recv timeout as terminal would stop before them."""

    def __init__(self):
        self.recv_calls = 0

    async def recv(self):
        import base64
        import json

        self.recv_calls += 1
        if self.recv_calls == 1:
            raise __import__("asyncio").TimeoutError
        if self.recv_calls == 2:
            return json.dumps(
                {"type": "response.output_audio.delta", "delta": base64.b64encode(bytes(2 * 2400)).decode()}
            )
        return json.dumps({"type": "response.done", "response": {"status": "completed"}})


async def test_collect_response_survives_silence_longer_than_poll():
    """A >5 s silent stretch must not end collection; keep waiting until the
    audio and the terminal response.done arrive."""
    ws = _SilentThenDoneWS()
    state = SessionState(label="B")
    # first clock() reads 6 s: more than the 5 s poll window has elapsed
    clock = _FakeClock(start=6.0)

    outcome = await collect_response(ws, state, deadline=90.0, clock=clock)

    assert outcome == "done", "collection must reach response.done, not stop on silence"
    assert state.response_done is True
    assert state.response_status == "completed"
    assert state.audio_bytes > 0, "the audio that arrived after the silence is kept"
    assert ws.recv_calls == 3


async def test_collect_response_stops_on_closed_connection():
    import time as _time

    class _ClosedWS:
        async def recv(self):
            raise ConnectionError("peer closed")

    state = SessionState(label="B")
    outcome = await collect_response(_ClosedWS(), state, deadline=_time.monotonic() + 90.0)
    assert outcome == "closed"
    assert state.response_done is False


async def test_collect_response_stops_at_deadline_when_silent():
    import time as _time

    class _SilentWS:
        async def recv(self):
            await __import__("asyncio").sleep(3600)

    state = SessionState(label="B")
    outcome = await collect_response(_SilentWS(), state, deadline=_time.monotonic() + 0.05, poll_s=0.01)
    assert outcome == "deadline"


def test_create_out_dir_fails_if_exists(tmp_path):
    target = tmp_path / "runs" / "x"
    assert create_out_dir(str(target)) == target
    with pytest.raises(FileExistsError):
        create_out_dir(str(target))
    # 0700 on the created dir
    assert (target.stat().st_mode & 0o777) == 0o700