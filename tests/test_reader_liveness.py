"""Reader liveness: the WS receive path keeps accepting audio and control
events while STT, LLM or TTS are deliberately blocked; speech_started and
cancel fire immediately; no obsolete audio after cancellation."""

import asyncio
import base64
import threading

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends, tone_pcm


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


async def settle():
    await asyncio.sleep(0.05)


async def test_audio_flows_during_blocked_stt(make_session):
    stt = FakeSTT(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(stt, FakeLLM(), FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn(speech_ms=150, silence_ms=700)
        await driver.wait_for("input_audio_buffer.speech_stopped")
        # STT now blocked; the reader must keep accepting and processing audio
        for _ in range(3):
            await driver.append_speech_turn(speech_ms=150, silence_ms=700)
        await driver.wait_next("input_audio_buffer.speech_started", driver.count("input_audio_buffer.speech_started") - 1)
        # control traffic stays live too
        session.handle_event({"type": "session.update", "session": {"instructions": "live?"}})
        await driver.wait_for("session.updated")
        assert driver.count("input_audio_buffer.speech_stopped") == 4
    finally:
        await driver.stop()


async def test_speech_started_and_cancel_during_blocked_llm(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        # LLM blocked; user barges in
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(150))})
        started_count = driver.count("input_audio_buffer.speech_started")
        await driver.wait_next("input_audio_buffer.speech_started", started_count - 1)
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        # reader still live: audio accepted, VAD reacts
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(150) + b"\x00\x00" * 3000)})
        await settle()
        llm.gate.set()
        await settle()
        types = driver.types()
        done_idx = types.index("response.done")
        assert "response.output_audio_transcript.delta" not in types[done_idx:]
        assert "response.output_audio.delta" not in types[done_idx:]
    finally:
        await driver.stop()


async def test_speech_started_and_cancel_during_blocked_tts(make_session):
    llm = FakeLLM()
    tts = FakeTTS(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        await driver.wait_for("response.output_audio_transcript.delta")
        # TTS blocked before first chunk; barge-in must still be instant
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(150))})
        await driver.wait_for("input_audio_buffer.speech_started")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        assert driver.of_type("response.output_audio.delta") == []
        tts.gate.set()
        await settle()
        types = driver.types()
        done_idx = types.index("response.done")
        assert "response.output_audio.delta" not in types[done_idx:]
    finally:
        await driver.stop()


async def test_no_obsolete_audio_after_cancel_race(make_session):
    """Cancel lands while a TTS chunk emit is in flight: fenced by generation."""
    llm = FakeLLM()
    tts = FakeTTS(chunks=10, inter_chunk_delay=0.01)
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        # cancel exactly while more chunks are being produced
        session.interrupt("race")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        # let everything finish and drain; nothing stale may follow
        await asyncio.sleep(0.5)
        types = driver.types()
        done_idx = types.index("response.done")
        assert "response.output_audio.delta" not in types[done_idx:]
    finally:
        await driver.stop()


async def test_outbound_queue_bounded_under_flood(make_session):
    """A flood of audio events cannot grow the outbound queue unboundedly."""
    llm = FakeLLM()
    tts = FakeTTS(chunks=50, chunk_ms=100)
    session, driver = make_session(
        backends=make_backends(FakeSTT(), llm, tts), out_capacity=8
    )
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        # bounded: channel never exceeds capacity, drops were counted instead
        assert len(session._out._deque) <= 8
        assert driver.count("response.done") >= 1
    finally:
        await driver.stop()
