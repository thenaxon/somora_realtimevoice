"""M2 lifecycle fixes: soft barge-in (H2), output pacing (H1), history
ordering (H5a) and H4 duplicate-request visibility.

All synthetic; metadata-only assertions. No transcript/instruction text is
inspected beyond the fake fixtures that already exist.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends, silence_pcm, tone_pcm
from realtimevoice.config import Settings
from realtimevoice.session import _role_sequence


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


# ------------------------------------------------------- H2 soft barge-in

async def test_soft_barge_in_short_burst_response_survives(make_session, lifecycle):
    """A short burst (< confirm) is rejected: the response completes with all
    its audio and no cancel (H2 fix)."""
    settings = Settings()
    settings.session.turn_detection.silence_duration_ms = 100
    settings.session.turn_detection.barge_in_confirm_ms = 250
    tts = FakeTTS(chunks=4, chunk_ms=100)
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(60))})
        await driver.wait_for("input_audio_buffer.speech_started")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(300))})
        await settle(200)
        assert driver.count("response.done") == 0, "response must not be cancelled"
        rejected = [r for r in lifecycle() if r["action"] == "barge_in_rejected"]
        assert rejected and rejected[-1]["speech_ms"] < 250
        llm.gate.set()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        # all four 100 ms chunks were emitted
        deltas = driver.of_type("response.output_audio.delta")
        assert len(deltas) >= 4
    finally:
        await driver.stop()


async def test_soft_barge_in_long_speech_cancels(make_session, lifecycle):
    """Speech that continues for the whole confirm window cancels the response
    with reason barge_in."""
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 60
    tts = FakeTTS(chunks=20, chunk_ms=100, inter_chunk_delay=0.02)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        # 400 ms of speech: the detector keeps speaking past the 60 ms window
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(400))})
        await driver.wait_for("input_audio_buffer.speech_started")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        cancels = [r for r in lifecycle() if r["action"] == "response_cancelled"]
        assert cancels and cancels[-1]["reason"] == "barge_in"
    finally:
        await driver.stop()


async def test_soft_barge_in_race_response_finishes_during_pending(make_session, lifecycle):
    """The answer is fully synthesized while a burst is still unconfirmed: the
    burst is decided first (here: noise), the held tail goes out, and only then
    the done events — no audio after output_audio.done, response completed."""
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 300
    tts = FakeTTS(chunks=4, chunk_ms=100)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(60))})
        await driver.wait_for("input_audio_buffer.speech_started")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(600))})
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        # the tail synthesized while pending was still flushed ...
        assert len(driver.of_type("response.output_audio.delta")) >= 4
        # ... and before the response said its audio was done
        types = driver.types()
        audio_done = types.index("response.output_audio.done")
        assert "response.output_audio.delta" not in types[audio_done + 1 :]
    finally:
        await driver.stop()


async def test_soft_barge_in_no_audio_after_confirmed_cancel(make_session):
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 40
    tts = FakeTTS(chunks=30, chunk_ms=100, inter_chunk_delay=0.02)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(400))})
        await driver.wait_for("input_audio_buffer.speech_started")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        done_idx = driver.types().index("response.done")
        await asyncio.sleep(0.2)
        after = driver.types()[done_idx + 1:]
        assert "response.output_audio.delta" not in after
    finally:
        await driver.stop()


async def test_soft_barge_in_config_invalid_rejected(make_session):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event(
            {"type": "session.update", "session": {"turn_detection": {"barge_in_confirm_ms": -1}}}
        )
        await settle(40)
        errs = driver.of_type("error")
        assert errs, "a negative confirm must be rejected"
    finally:
        await driver.stop()


async def test_session_update_exposes_barge_in_confirm_ms(make_session):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event(
            {"type": "session.update", "session": {"turn_detection": {"barge_in_confirm_ms": 120}}}
        )
        updated = await driver.wait_for("session.updated")
        assert updated["session"]["turn_detection"]["barge_in_confirm_ms"] == 120
    finally:
        await driver.stop()


# ------------------------------------------------------ H5a history ordering

async def test_h5a_single_assistant_message_with_content_and_tool_calls(make_session):
    llm = FakeLLM(
        script=[
            {"text": "Ich schaue nach.", "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": "{}"}]},
            {"text": "Fertig."},
        ]
    )
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            }
        )
        session.handle_event({"type": "response.create"})
        await driver.wait_next("response.done", 1)
        await settle(60)
        messages = llm.calls[-1]["messages"]
        assistants = [m for m in messages if m.get("role") == "assistant"]
        # exactly one assistant message: content + tool_calls together
        assert len(assistants) == 1, [m.get("role") for m in messages]
        assert assistants[0].get("content")
        assert assistants[0].get("tool_calls")
        assert assistants[0]["tool_calls"][0]["id"] == "call_1"
        # the tool result follows that assistant message
        idx_a = messages.index(assistants[0])
        assert messages[idx_a + 1]["role"] == "tool"
        # role sequence shows one a[tc] then t, never a[tc],a,t
        seq = _role_sequence(messages)
        assert seq.endswith("a[tc],t")
        assert "a[tc],a" not in seq
    finally:
        await driver.stop()


# -------------------------------------------------------------- H1 pacing

async def test_output_pacing_defers_done_by_lead(make_session):
    """With pacing on, response.done lands ~ (audio duration - lead) after the
    response starts, not as soon as synthesis finishes."""
    settings = Settings()
    settings.session.output_pacing = "realtime"
    settings.session.output_pacing_lead_ms = 300
    # 10 chunks x 100 ms = 1000 ms of audio, synthesized almost instantly;
    # a single-sentence default keeps it one segment / one TTS call
    tts = FakeTTS(chunks=10, chunk_ms=100)
    session, driver = make_session(
        settings=settings, backends=make_backends(FakeSTT(), FakeLLM(default_text="Hallo"), tts)
    )
    try:
        await settle()
        started = asyncio.get_running_loop().time()
        session.handle_event({"type": "response.create"})
        done = await driver.wait_for("response.done", timeout=5.0)
        elapsed = asyncio.get_running_loop().time() - started
        assert done["response"]["status"] == "completed"
        # ~ (1000 - 300) ms; allow scheduling slack
        assert 0.5 <= elapsed <= 1.6, elapsed
    finally:
        await driver.stop()


async def test_output_pacing_off_is_immediate(make_session):
    settings = Settings()
    settings.session.output_pacing = "off"
    tts = FakeTTS(chunks=10, chunk_ms=100)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        started = asyncio.get_running_loop().time()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done", timeout=5.0)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.5, elapsed
    finally:
        await driver.stop()


async def test_output_pacing_cancel_drops_queued_audio(make_session, lifecycle):
    settings = Settings()
    settings.session.output_pacing = "realtime"
    settings.session.output_pacing_lead_ms = 0
    # 30 x 100 ms = 3 s of audio; cancel early
    tts = FakeTTS(chunks=30, chunk_ms=100)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "response.cancel"})
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        dropped = [r for r in lifecycle() if r["action"] == "pacing_dropped_audio_ms"]
        assert dropped and dropped[-1]["audio_ms"] > 0
        done_idx = driver.types().index("response.done")
        await asyncio.sleep(0.2)
        assert "response.output_audio.delta" not in driver.types()[done_idx + 1:]
    finally:
        await driver.stop()


async def test_output_pacing_extension_key_accepted(make_session):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event(
            {"type": "session.update", "session": {"x_realtimevoice": {"output_pacing": "realtime"}}}
        )
        updated = await driver.wait_for("session.updated")
        assert updated["session"]["x_realtimevoice"]["output_pacing"] == "realtime"
        session.handle_event(
            {"type": "session.update", "session": {"x_realtimevoice": {"output_pacing": "bogus"}}}
        )
        await settle(40)
        assert driver.of_type("error"), "invalid pacing value must be rejected"
    finally:
        await driver.stop()


# --------------------------------------------------- H4 duplicate visibility

async def test_h4_duplicate_request_suspect_flag(make_session, lifecycle):
    llm = FakeLLM(script=[{"text": "Eins."}, {"text": "Zwei."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create", "event_id": "evt_1"})
        await driver.wait_for("response.done")
        await settle(40)
        session.handle_event({"type": "response.create", "event_id": "evt_2"})
        await driver.wait_next("response.done", 1)
        await settle(40)
        reserved = [r for r in lifecycle() if r["action"] == "response_reserved"]
        assert len(reserved) == 2
        # first create: no prior done -> not a suspect
        assert reserved[0].get("duplicate_request_suspect") in (False, None)
        # second create: client, nothing new, no instructions -> suspect
        assert reserved[1]["duplicate_request_suspect"] is True
    finally:
        await driver.stop()


async def test_h4_flag_false_with_one_off_instructions(make_session, lifecycle):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        await settle(40)
        session.handle_event({"type": "response.create", "response": {"instructions": "sag etwas"}})
        await driver.wait_next("response.done", 1)
        await settle(40)
        reserved = [r for r in lifecycle() if r["action"] == "response_reserved"]
        assert reserved[-1]["has_instructions"] is True
        assert reserved[-1]["duplicate_request_suspect"] is False
    finally:
        await driver.stop()

async def test_confirmed_barge_in_keeps_only_the_heard_audio_in_history(make_session):
    """Audio held during the confirm window was never sent: when the barge-in
    is confirmed, the truncation counts only what the client received."""
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 150
    tts = FakeTTS(chunks=30, chunk_ms=100, inter_chunk_delay=0.01)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(400))})
        await driver.wait_for("input_audio_buffer.speech_started")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(400))})
        truncated = await driver.wait_for("conversation.item.truncated")
        sent_ms = sum(len(base64.b64decode(e["delta"])) for e in driver.of_type("response.output_audio.delta")) / 48
        assert truncated["audio_end_ms"] <= sent_ms + 1, (truncated["audio_end_ms"], sent_ms)
    finally:
        await driver.stop()


async def test_a_short_noise_does_not_confirm_a_barge_in(make_session):
    """Seen live: a 20 ms noise in the room cancelled the answer, because the
    confirm window asked only whether the burst was still open (it stays open
    for the whole trailing-silence window), not whether anyone spoke."""
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 250
    tts = FakeTTS(chunks=20, chunk_ms=100, inter_chunk_delay=0.02)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), FakeLLM(), tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        # as a microphone delivers it: 20 ms at a time, in real time
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(40))})
        for _ in range(35):
            session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(20))})
            await asyncio.sleep(0.02)
        done = await driver.wait_for("response.done", timeout=5)
        assert done["response"]["status"] == "completed"
    finally:
        await driver.stop()


async def test_an_answer_cut_by_what_turns_out_to_be_noise_is_given_again(make_session):
    """Safety net (immediate barge-in, confirm window 0): noise cancels the
    answer at once; when the burst then turns out too short to be speech, the
    question is answered again instead of being left without an answer."""
    settings = Settings()
    settings.session.turn_detection.barge_in_confirm_ms = 0
    tts = FakeTTS(chunks=20, chunk_ms=100, inter_chunk_delay=0.02)
    llm = FakeLLM()
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Wie spaet?"}]}})
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.output_audio.delta")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(40) + silence_pcm(700))})
        first = await driver.wait_for("response.done", timeout=5)
        assert first["response"]["status"] == "cancelled"
        second = await driver.wait_next("response.done", 1, timeout=5)
        assert second["response"]["status"] == "completed"
        assert len(llm.calls) == 2
    finally:
        await driver.stop()
