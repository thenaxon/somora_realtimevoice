"""A speaking model that needs time to start must not be heard as gaps.

In a call an answer is spoken sentence by sentence, each its own request.
The next sentence is asked for while the current one is spoken, and a model
can have a head start (start_buffer_ms) before a response begins speaking.
"""

import asyncio

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.config import Settings


class SlowStartTTS(FakeTTS):
    """Each sentence takes `startup` seconds before its first audio; records
    when each request starts and ends."""

    def __init__(self, startup=0.2, **kw):
        super().__init__(**kw)
        self.startup = startup
        self.started: list[tuple[str, float]] = []
        self.finished: list[tuple[str, float]] = []

    async def stream_speech(self, text, *, voice=None, language=None):
        loop = asyncio.get_running_loop()
        self.started.append((text, loop.time()))
        await asyncio.sleep(self.startup)
        async for pcm in super().stream_speech(text, voice=voice, language=language):
            yield pcm
        self.finished.append((text, loop.time()))


TEXT = "Erster Satz ist hier. Zweiter Satz folgt jetzt. Dritter Satz zum Schluss."


async def _answer(make_session, tts, settings=None):
    session, driver = make_session(
        settings=settings, backends=make_backends(FakeSTT(), FakeLLM(script=[{"text": TEXT}]), tts)
    )
    await asyncio.sleep(0)
    session.handle_event({"type": "response.create"})
    await asyncio.wait_for(driver.wait_for("response.done"), 10)
    return session, driver


async def test_the_next_sentence_is_asked_for_while_the_current_one_is_spoken(make_session):
    tts = SlowStartTTS(startup=0.2, chunks=3, inter_chunk_delay=0.05)
    session, driver = await _answer(make_session, tts)
    try:
        assert len(tts.started) == 3
        # sentence 2 started before sentence 1 was done
        assert tts.started[1][1] < tts.finished[0][1]
        # and every sentence was spoken, in order, completely
        done = driver.of_type("response.done")[-1]
        assert done["response"]["status"] == "completed"
        assert [c["text"] for c in tts.calls] == [t for t, _ in tts.started]
    finally:
        await driver.stop()


class CountingTTS(SlowStartTTS):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.yielded = 0

    async def stream_speech(self, text, *, voice=None, language=None):
        async for pcm in super().stream_speech(text, voice=voice, language=language):
            self.yielded += 1
            yield pcm


async def test_a_head_start_holds_the_first_audio_back(make_session, monkeypatch):
    from realtimevoice.session import RealtimeSession

    settings = Settings()
    settings.tts.start_buffer_ms = 250
    tts = CountingTTS(startup=0.0, chunks=5, inter_chunk_delay=0.02)  # 100 ms per chunk
    seen: list[int] = []
    real = RealtimeSession._deliver_audio_delta

    def spy(self, resp, chunk):
        seen.append(tts.yielded)
        return real(self, resp, chunk)

    monkeypatch.setattr(RealtimeSession, "_deliver_audio_delta", spy)
    session, driver = await _answer(make_session, tts, settings)
    try:
        # the first audio left only once 300 ms (3 pieces) were ready ...
        assert seen and seen[0] >= 3
        # ... and nothing was lost
        assert len(seen) == 15
    finally:
        await driver.stop()


async def test_without_a_head_start_audio_leaves_at_once(make_session, monkeypatch):
    from realtimevoice.session import RealtimeSession

    tts = CountingTTS(startup=0.0, chunks=5, inter_chunk_delay=0.02)
    seen: list[int] = []
    real = RealtimeSession._deliver_audio_delta

    def spy(self, resp, chunk):
        seen.append(tts.yielded)
        return real(self, resp, chunk)

    monkeypatch.setattr(RealtimeSession, "_deliver_audio_delta", spy)
    session, driver = await _answer(make_session, tts)
    try:
        assert seen[0] == 1
    finally:
        await driver.stop()


async def test_a_short_sentence_held_for_its_head_start_still_leaves_in_pieces(make_session, monkeypatch):
    from realtimevoice.session import RealtimeSession

    settings = Settings()
    settings.tts.start_buffer_ms = 5000  # longer than any sentence here
    tts = CountingTTS(startup=0.0, chunks=5)
    sizes: list[int] = []
    real = RealtimeSession._deliver_audio_delta

    def spy(self, resp, chunk):
        sizes.append(len(chunk))
        return real(self, resp, chunk)

    monkeypatch.setattr(RealtimeSession, "_deliver_audio_delta", spy)
    session, driver = await _answer(make_session, tts, settings)
    try:
        assert sizes and max(sizes) <= 4800
    finally:
        await driver.stop()


async def test_the_current_sentence_is_asked_for_before_the_next(make_session):
    tts = SlowStartTTS(startup=0.05, chunks=2)
    session, driver = await _answer(make_session, tts)
    try:
        texts = [t for t, _ in tts.started]
        assert texts[0].startswith("Erster")
    finally:
        await driver.stop()


async def test_closing_a_call_mid_answer_ends_every_speech_request(make_session):
    tts = SlowStartTTS(startup=0.0, chunks=50, inter_chunk_delay=0.05)
    session, driver = make_session(
        backends=make_backends(FakeSTT(), FakeLLM(script=[{"text": TEXT}]), tts)
    )
    await asyncio.sleep(0)
    session.handle_event({"type": "response.create"})
    await driver.wait_for("response.output_audio.delta")
    await asyncio.sleep(0.1)
    before = {t for t in asyncio.all_tasks() if "produce" in repr(t)}
    assert before, "a sentence was being fetched"
    await session.aclose()
    assert all(t.done() for t in before)
    await driver.stop()
