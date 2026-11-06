"""F11 R2 — controlled acoustic feedback (synthetic echo) against a throwaway
in-process session with fake backends.

What this proves (and what it does NOT):
  * It reproduces the field sequence: a COMPLETED response is followed by a
    `speech_started` during `estimated_playback_tail_ms` (exactly the F11
    lifecycle log: 198-275 ms after response.done, during the tail).
  * It shows the SERVER reaction: the response is already done, so there is no
    cancel and no truncation. The client's own flush is therefore the only
    thing that can stop the buffered tail.
  * It then replays the observed server event stream through the REAL shipped
    client (`ui/audio.js`) and asserts the BASELINE behaviour: with
    interrupt_response on, a `speech_started` flushes the buffered tail
    immediately (03cb679 app.js:320-338). Echo protection comes from the
    microphone's AEC (R1), NOT from ignoring `speech_started`.

  * It does NOT prove echo-suppression quality: real acoustic echo cancellation
    happens in the browser/OS and cannot be proven by a synthetic stream. This
    test proves the mechanism and the client/server reaction only.

Synthetic audio only; no real data. Metadata-only assertions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.config import Settings

TESTS_DIR = Path(__file__).resolve().parent


async def settle(ms=60):
    await asyncio.sleep(ms / 1000.0)


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.getMessage()) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def _attenuate(pcm: bytes, factor: float = 0.25) -> bytes:
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return struct.pack(f"<{n}h", *[int(s * factor) for s in samples])


def _client_reaction(events: list[dict]) -> dict:
    """Replay the server event stream through the real shipped client."""
    payload = json.dumps(events)
    proc = subprocess.run(
        ["node", str(TESTS_DIR / "client_reaction.mjs"), "/dev/stdin"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"client harness failed: {proc.stderr}"
    return json.loads(proc.stdout)


async def test_r2_echo_during_playback_tail_flushes_per_baseline(make_session, lifecycle):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH: F11 client-reaction harness requires Node")
    settings = Settings()
    settings.session.turn_detection.silence_duration_ms = 100
    settings.session.turn_detection.barge_in_confirm_ms = 250
    settings.session.output_pacing = "off"
    tts = FakeTTS(chunks=8, chunk_ms=100)  # 800 ms of answer audio
    session, driver = make_session(
        settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts)
    )
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"

        done_log = [r for r in lifecycle() if r["action"] == "response_done"][-1]
        tail_ms = done_log["estimated_playback_tail_ms"]
        assert tail_ms is not None and tail_ms > 0, "no estimated playback tail to test against"

        deltas = driver.of_type("response.output_audio.delta")
        assert deltas, "no answer audio was emitted"
        answer = b"".join(base64.b64decode(d["delta"]) for d in deltas)

        # The synthetic echo: the assistant's own output, attenuated and fed
        # back as microphone input while the answer is still playing. This is
        # what a RAW microphone (no AEC) produces.
        echo = _attenuate(answer)
        started_before = driver.count("input_audio_buffer.speech_started")
        for off in range(0, len(echo), 4800):
            session.handle_event(
                {"type": "input_audio_buffer.append", "audio": _b64(echo[off : off + 4800])}
            )
            await settle(10)
        await settle(300)

        # SERVER reaction: a start was detected during the playback tail, but the
        # response was already completed, so the server neither cancels nor
        # truncates. The cut therefore came from the client.
        starts = driver.of_type("input_audio_buffer.speech_started")
        assert len(starts) > started_before, "the synthetic echo was not detected as speech"
        start_log = [r for r in lifecycle() if r["action"] == "speech_started"][-1]
        assert start_log["during_estimated_playback_tail"] is True
        cancels = [r for r in lifecycle() if r["action"] == "response_cancelled"]
        assert not cancels, "a post-done echo must not cancel the completed response"
        assert not driver.of_type("conversation.item.truncated")

        # CLIENT reaction (real shipped ui/audio.js) on that exact stream. This
        # is the BASELINE behaviour (03cb679 app.js:320-338): with
        # interrupt_response on, a speech_started flushes the buffered tail —
        # echo protection is the microphone's AEC (R1), not ignoring the event.
        stream = [e for e in driver.events if e.get("type") in {
            "response.created", "response.output_item.added",
            "response.output_audio.delta", "response.done",
            "input_audio_buffer.speech_started",
        }]
        reaction = _client_reaction(stream)
        assert reaction["cleared"] is True, (
            "a genuine speech_started (interrupt_response on) must flush the "
            f"buffered answer, baseline semantics: {reaction}"
        )
        assert reaction["state"] == "listening", reaction
    finally:
        await driver.stop()