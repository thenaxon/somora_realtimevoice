"""Cancellation races: barge-in, response.cancel, stale-generation fencing."""

import asyncio

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends


async def settle():
    await asyncio.sleep(0.05)


async def make_gated(make_session, tts_delay=0.0):
    llm = FakeLLM(gate=asyncio.Event())
    stt = FakeSTT()
    tts = FakeTTS(inter_chunk_delay=tts_delay)
    session, driver = make_session(backends=make_backends(stt, llm, tts))
    return session, driver, llm, stt, tts


async def test_barge_in_during_llm_blocks_cancels_response(make_session):
    session, driver, llm, stt, tts = await make_gated(make_session)
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # user starts speaking while the LLM is blocked -> barge-in
        session.handle_event({"type": "input_audio_buffer.append", "audio": _tone_b64(200)})
        await driver.wait_for("input_audio_buffer.speech_started")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        assert driver.of_type("response.output_audio.delta") == []
        # unblock the LLM: no late transcript/audio may appear
        llm.gate.set()
        await settle()
        types_after = [e["type"] for e in driver.events]
        done_idx = types_after.index("response.done")
        for stale in ("response.output_audio_transcript.delta", "response.output_audio.delta"):
            assert stale not in types_after[done_idx:]
    finally:
        await driver.stop()


def _tone_b64(ms):
    import base64

    from fake_backends import tone_pcm

    return base64.b64encode(tone_pcm(ms)).decode()


async def test_no_stale_audio_after_cancel_mid_tts(make_session):
    llm = FakeLLM()
    tts = FakeTTS(chunks=6, inter_chunk_delay=0.02)
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "response.cancel"})
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        done_idx = driver.types().index("response.done")
        # drain the world: nothing of this response may be appended afterwards
        await asyncio.sleep(0.3)
        new_types = [e["type"] for e in driver.events[done_idx + 1 :]]
        assert "response.output_audio.delta" not in new_types
        assert "response.output_audio_transcript.delta" not in new_types
        audio_before = [e for e in driver.events[:done_idx] if e["type"] == "response.output_audio.delta"]
        assert audio_before, "audio before cancel expected"
    finally:
        await driver.stop()


async def test_interrupt_emits_truncation_when_audio_was_sent(make_session):
    from realtimevoice.config import Settings

    settings = Settings()
    # immediate barge-in (0): this test pins the truncation mechanism itself,
    # disjoint from the H2 soft-barge-in confirmation window
    settings.session.turn_detection.barge_in_confirm_ms = 0
    llm = FakeLLM()
    tts = FakeTTS(chunks=4, inter_chunk_delay=0.01)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "input_audio_buffer.append", "audio": _tone_b64(150)})
        await driver.wait_for("input_audio_buffer.speech_started")
        truncated = await driver.wait_for("conversation.item.truncated")
        assert truncated["audio_end_ms"] > 0
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
    finally:
        await driver.stop()


async def test_cancel_without_active_response_is_noop(make_session):
    session, driver = make_session()
    await settle()
    session.handle_event({"type": "response.cancel"})
    await settle()
    assert driver.of_type("error") == []
    assert driver.of_type("response.done") == []


async def test_response_after_cancel_works(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver, llm, stt, tts = await make_gated(make_session)
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        session.handle_event({"type": "response.cancel"})
        await driver.wait_for("response.done")
        dones_after_cancel = driver.count("response.done")
        llm.gate.set()
        session.handle_event({"type": "response.create"})
        done = await driver.wait_next("response.done", dones_after_cancel)
        assert done["response"]["status"] == "completed"
        assert len(llm.calls) == 2
    finally:
        await driver.stop()


async def test_two_interrupts_in_a_row(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver, llm, stt, tts = await make_gated(make_session)
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        session.interrupt("test1")
        await driver.wait_for("response.done")
        session.interrupt("test2")
        await settle()
        dones = driver.of_type("response.done")
        assert len(dones) == 1
    finally:
        await driver.stop()


async def test_stt_survives_barge_in(make_session):
    stt = FakeSTT(text="Spaeter gefragt.")
    stt.gate = asyncio.Event()
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn(speech_ms=200, silence_ms=700)
        await driver.wait_for("input_audio_buffer.speech_stopped")
        # STT blocked now; user barges in during a (nonexistent) response is a
        # no-op for STT. Unblock and expect the transcript still completes.
        stt.gate.set()
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await driver.wait_for("response.done")
    finally:
        await driver.stop()


async def test_turn_queue_bounded_drops_oldest(make_session):
    stt = FakeSTT(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(stt, FakeLLM(), FakeTTS()))
    try:
        await settle()
        # four turns back to back while STT is blocked: queue (2) + in-flight
        for _ in range(4):
            await driver.append_speech_turn(speech_ms=150, silence_ms=700)
        await settle()
        stt.gate.set()
        deadline = asyncio.get_running_loop().time() + 3.0
        while (
            driver.count("conversation.item.input_audio_transcription.completed") < 3
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.005)
        overflow_errors = [
            e for e in driver.of_type("error") if e["error"]["code"] == "turn_queue_overflow"
        ]
        assert overflow_errors, "bounded turn queue must signal drops"
        transcripts = driver.of_type("conversation.item.input_audio_transcription.completed")
        # bounded pipeline: 1 in-flight + 2 queued; the oldest queued turn was
        # dropped instead of growing the queue
        assert len(transcripts) == 3
    finally:
        await driver.stop()


async def test_tts_error_does_not_kill_session(make_session):
    llm = FakeLLM()
    tts = FakeTTS(error="boom")
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "incomplete"
        errors = [e for e in driver.of_type("error") if "boom" in e["error"]["message"]]
        assert errors
        # session still usable
        session.handle_event({"type": "response.create"})
        done2 = await driver.wait_for("response.done")
        assert done2["response"]["status"] == "incomplete"
    finally:
        await driver.stop()
