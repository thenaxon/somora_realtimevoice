"""A turn must not get lost, and a response must not hang.

Each test here failed before its fix:

- a short noise burst (a cough, a click) while the user's sentence is still
  being transcribed swallowed the sentence: no answer at all;
- a text-to-speech worker that died (an unexpected error, no TTS backend)
  left the response open for good, refusing every later one;
- a tool call the client had not answered yet made the next LLM request
  invalid (an assistant tool call with no tool message after it).
"""

from __future__ import annotations

import asyncio

from fake_backends import Backends, FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.vad import BurstStats


async def _done_within(driver, seconds: float) -> bool:
    try:
        await asyncio.wait_for(driver.wait_for("response.done"), seconds)
        return True
    except asyncio.TimeoutError:
        return False


class GatedSTT(FakeSTT):
    """Transcribes only once the test opens the gate."""

    def __init__(self, text: str) -> None:
        super().__init__(text=text)
        self.gate = asyncio.Event()

    async def transcribe(self, wav, *, language=None):
        await self.gate.wait()
        return await super().transcribe(wav, language=language)


def _speak(session) -> None:
    session._on_speech_started(None)
    session._on_speech_stopped(b"\x00\x01" * 24000, BurstStats(speech_ms=900))


def _noise(session) -> None:
    session._on_speech_started(None)
    session._on_burst_discarded(BurstStats(speech_ms=40, outcome="discarded_short"))


async def test_noise_while_the_sentence_is_transcribed_does_not_swallow_it(make_session):
    stt = GatedSTT("Wie spaet ist es?")
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await asyncio.sleep(0)
        _speak(session)
        await asyncio.sleep(0.01)
        _noise(session)
        stt.gate.set()
        assert await _done_within(driver, 2), "the sentence got no answer"
        assert len(llm.calls) == 1
    finally:
        await driver.stop()


async def test_noise_right_after_the_sentence_was_transcribed_does_not_swallow_it(make_session):
    stt = GatedSTT("Wie spaet ist es?")
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await asyncio.sleep(0)
        _speak(session)
        await asyncio.sleep(0.01)
        # the noise starts, the transcript arrives while it is still open,
        # then the noise turns out too short to be speech
        session._on_speech_started(None)
        stt.gate.set()
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await asyncio.sleep(0.01)
        session._on_burst_discarded(BurstStats(speech_ms=40, outcome="discarded_short"))
        assert await _done_within(driver, 2), "the sentence got no answer"
        assert len(llm.calls) == 1
    finally:
        await driver.stop()


async def test_a_real_second_sentence_still_supersedes_the_first(make_session):
    """Kept behaviour: two sentences in a row get ONE answer, to both."""
    stt = GatedSTT("Erster Satz.")
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await asyncio.sleep(0)
        _speak(session)
        await asyncio.sleep(0.01)
        _speak(session)
        stt.gate.set()
        assert await _done_within(driver, 2)
        await asyncio.sleep(0.2)
        assert len(llm.calls) == 1
        users = [m for m in llm.calls[0]["messages"] if m["role"] == "user"]
        assert len(users) == 2
    finally:
        await driver.stop()


class BoomTTS(FakeTTS):
    async def stream_speech(self, text, *, voice=None, language=None):
        raise RuntimeError("boom")
        yield b""  # pragma: no cover - makes this an async generator


LONG = " ".join(f"Das ist Satz Nummer {i} hier." for i in range(40))


async def test_a_dead_tts_worker_ends_the_response_instead_of_hanging(make_session):
    llm = FakeLLM(script=[{"text": LONG}, {"text": "Kurz."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, BoomTTS()))
    try:
        await asyncio.sleep(0)
        session.handle_event({"type": "response.create"})
        assert await _done_within(driver, 3), "the response never ended"
        assert session._active_response is None
        # and the next one is not refused as "already active"
        driver.events.clear()
        session.handle_event({"type": "response.create"})
        assert await _done_within(driver, 3)
        assert not [e for e in driver.of_type("error") if "active response" in e["error"]["message"]]
    finally:
        await driver.stop()


async def test_no_tts_backend_ends_the_response_instead_of_hanging(make_session):
    llm = FakeLLM(script=[{"text": LONG}])
    session, driver = make_session(backends=Backends(stt=FakeSTT(), llm=llm, tts=None))
    try:
        await asyncio.sleep(0)
        session.handle_event({"type": "response.create"})
        assert await _done_within(driver, 3), "the response never ended"
    finally:
        await driver.stop()


async def test_an_unanswered_tool_call_does_not_break_the_next_request(make_session):
    llm = FakeLLM(script=[{"text": "", "tool_calls": [{"id": "c1", "name": "get_time", "arguments": "{}"}]}, {"text": "Gut."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio.sleep(0)
        session.handle_event({"type": "session.update", "session": {"tools": [{"type": "function", "name": "get_time"}]}})
        session.handle_event({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Zeit?"}]}})
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        # the tool-only response lists only what it announced
        done = driver.of_type("response.done")[-1]
        added = {e["item"]["id"] for e in driver.of_type("response.output_item.added")}
        assert {o["id"] for o in done["response"]["output"]} <= added
        # the user speaks again before the client answered the call
        driver.events.clear()
        session.handle_event({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hallo?"}]}})
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        messages = llm.calls[1]["messages"]
        for i, m in enumerate(messages):
            for call in m.get("tool_calls") or []:
                answered = [x for x in messages[i + 1 :] if x["role"] == "tool" and x.get("tool_call_id") == call["id"]]
                assert answered, f"tool call {call['id']} has no tool message"
                assert messages[i + 1]["role"] == "tool", "a tool message must follow its call directly"
    finally:
        await driver.stop()


async def test_manual_mode_commit_reaches_the_model(make_session):
    import base64

    from fake_backends import silence_pcm, tone_pcm
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.type = "none"
    llm = FakeLLM()
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(text="Manuell gesprochen."), llm, FakeTTS()))
    try:
        await asyncio.sleep(0)
        audio = base64.b64encode(tone_pcm(400) + silence_pcm(100)).decode()
        session.handle_event({"type": "input_audio_buffer.append", "audio": audio})
        session.handle_event({"type": "input_audio_buffer.commit"})
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        users = [m for m in llm.calls[0]["messages"] if m["role"] == "user"]
        assert users and users[0]["content"] == "Manuell gesprochen."
    finally:
        await driver.stop()


async def test_malformed_events_are_answered_with_an_error_not_a_dropped_connection(make_session):
    session, driver = make_session()
    try:
        await asyncio.sleep(0)
        for event in (
            {"type": "conversation.item.truncate", "item_id": "x", "content_index": "abc", "audio_end_ms": 0},
            {"type": "conversation.item.truncate", "item_id": "x", "content_index": 0, "audio_end_ms": None},
            {"type": "conversation.item.create", "item": {"id": {"not": "a string"}, "type": "message", "role": "user", "content": []}},
        ):
            session.handle_event(event)  # must not raise
        await asyncio.sleep(0.05)
        assert len(driver.of_type("error")) == 3
    finally:
        await driver.stop()


def test_the_openai_threshold_means_the_same_as_the_server_default():
    """0.5 (OpenAI's default) is the server's default energy threshold; a
    client asking for more sensitivity (lower) gets a lower threshold."""
    from realtimevoice.config import TurnDetectionConfig
    from realtimevoice.session import ga_threshold_to_rms, rms_to_ga

    assert ga_threshold_to_rms(0.5) == TurnDetectionConfig().threshold_rms
    assert ga_threshold_to_rms(0.4) < TurnDetectionConfig().threshold_rms
    assert ga_threshold_to_rms(0.0) == 150.0 and ga_threshold_to_rms(1.0) == 1050.0
    for t in (0.0, 0.2, 0.4, 0.5, 0.7, 1.0):
        assert abs(rms_to_ga(ga_threshold_to_rms(t)) - t) < 1e-9
