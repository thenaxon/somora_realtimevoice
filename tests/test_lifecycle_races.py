"""Deterministic lifecycle race regressions.

Covers the authorized fixes:
- atomic response reservation (double create, cancel before task start)
- turn epoch: superseded STT results must not start a stale response
- internal auto-response coalescing instead of busy-error spam
- tool output accepted during an active response
- configuration variants: interrupt_response / create_response off
"""

import asyncio
import base64

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends, tone_pcm


async def settle(ms=50):
    await asyncio.sleep(ms / 1000.0)


def tone_b64(ms):
    return base64.b64encode(tone_pcm(ms)).decode()


async def test_cancel_before_task_starts_emits_coherent_lifecycle(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        # reservation is made synchronously; the task is scheduled but has
        # not executed its first slice yet
        session.handle_event({"type": "response.create"})
        session.handle_event({"type": "response.cancel"})
        await settle()
        created = driver.of_type("response.created")
        dones = driver.of_type("response.done")
        assert len(created) == 1
        assert len(dones) == 1
        assert dones[0]["response"]["status"] == "cancelled"
        assert created[0]["response"]["id"] == dones[0]["response"]["id"]
        assert llm.calls == [], "cancelled reservation must never reach the LLM"
        # the session stays fully usable
        llm.gate.set()
        session.handle_event({"type": "response.create"})
        done = await driver.wait_next("response.done", 1)
        assert done["response"]["status"] == "completed"
        assert len(llm.calls) == 1
    finally:
        await driver.stop()


async def test_two_rapid_response_creates_reserve_once(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        # back-to-back, before the first task has run: the second must be
        # rejected against the synchronous reservation
        session.handle_event({"type": "response.create"})
        error = await driver.wait_for("error")
        assert error["error"]["code"] == "already_active_response"
        llm.gate.set()
        await driver.wait_for("response.done")
        await settle(150)
        assert driver.count("response.created") == 1
        assert driver.count("response.done") == 1
        assert len(llm.calls) == 1
    finally:
        await driver.stop()


async def test_superseded_stt_does_not_start_stale_response(make_session):
    # STT A pending -> speech B starts (new turn epoch) -> STT A completes:
    # transcript A must be preserved, but only B may auto-start a response
    class QueuedTextSTT(FakeSTT):
        async def transcribe(self, wav, *, language):
            self.calls.append(wav)
            self.language_seen = language
            if self.gate is not None:
                await self.gate.wait()
            return self.texts.pop(0) if getattr(self, "texts", None) else self.text

    stt = QueuedTextSTT()
    stt.texts = ["Erste Aeusserung.", "Zweite Aeusserung."]
    stt.gate = asyncio.Event()
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        # turn A: speech starts/ends, STT A is queued and blocked on the gate
        session._on_speech_started()
        session._on_speech_stopped(b"turn-a")
        await settle(80)
        assert driver.count("response.created") == 0
        # turn B: new speech starts -> epoch advances while STT A is pending
        session._on_speech_started()
        session._on_speech_stopped(b"turn-b")
        await settle(80)
        assert driver.count("response.created") == 0
        # both STT results arrive now, A first
        stt.gate.set()
        await driver.wait_for("response.done")
        await settle(150)
        transcripts = driver.of_type("conversation.item.input_audio_transcription.completed")
        assert [t["transcript"] for t in transcripts] == ["Erste Aeusserung.", "Zweite Aeusserung."]
        # exactly one response: for turn B only
        assert driver.count("response.created") == 1
        assert driver.count("response.done") == 1
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        user_contents = [m["content"] for m in messages if m.get("role") == "user"]
        assert "Erste Aeusserung." in user_contents, "legitimate content must not be lost"
        busy = [e for e in driver.of_type("error") if e["error"]["code"] == "already_active_response"]
        assert busy == []
    finally:
        await driver.stop()


async def test_internal_auto_coalesces_while_response_active(make_session):
    settings_managed = dict()
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.interrupt_response = False
    settings_managed["s"] = settings
    llm = FakeLLM(gate=asyncio.Event())
    stt = FakeSTT(text="Waehrend der Antwort gesprochen.")
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # speech completes while the response is active and must not cancel
        # it (interrupt_response=false); its STT result queues an auto request
        await driver.append_speech_turn(speech_ms=200, silence_ms=600)
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await settle(100)
        busy = [e for e in driver.of_type("error") if e["error"]["code"] == "already_active_response"]
        assert busy == [], "internal automation must coalesce, not spam busy errors"
        assert driver.count("response.created") == 1
        llm.gate.set()
        first_done = await driver.wait_for("response.done")
        assert first_done["response"]["status"] == "completed"
        # the coalesced auto request flushes automatically after the active
        # response finishes
        second_done = await driver.wait_next("response.done", 1)
        assert second_done["response"]["status"] == "completed"
        assert driver.count("response.created") == 2
        assert len(llm.calls) == 2
    finally:
        await driver.stop()


async def test_queued_auto_discarded_when_newer_speech_started(make_session):
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.interrupt_response = False
    llm = FakeLLM(gate=asyncio.Event())
    stt = FakeSTT(text="Veralteter Turn.")
    stt.gate = asyncio.Event()
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # turn A speech completes; its STT result queues an auto request
        # (epoch 1) while response R1 stays active
        session._on_speech_started()
        session._on_speech_stopped(b"turn-a")
        stt.gate.set()
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await settle(100)
        busy = [e for e in driver.of_type("error") if e["error"]["code"] == "already_active_response"]
        assert busy == []
        assert driver.count("response.created") == 1
        # newer speech STARTS (epoch 2) but does not cancel R1
        session._on_speech_started()
        llm.gate.set()
        first_done = await driver.wait_for("response.done")
        assert first_done["response"]["status"] == "completed"
        await settle(200)
        # the queued request belongs to the superseded epoch: it must be
        # discarded, not flushed as a stale second answer
        assert driver.count("response.done") == 1, driver.types()
        assert driver.count("response.created") == 1
        assert len(llm.calls) == 1
    finally:
        await driver.stop()


async def test_cancel_with_queued_auto_no_orphan_response(make_session):
    # active A + queued internal auto B + explicit client cancel: the cancel
    # must not orphan a reservation or produce a double response, and the
    # queued auto is dropped (cancel semantics), then an explicit create works
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.interrupt_response = False
    llm = FakeLLM(gate=asyncio.Event())
    stt = FakeSTT(text="Wartende Folge.")
    stt.gate = asyncio.Event()
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        rid_a = driver.of_type("response.created")[-1]["response"]["id"]
        # speech A completes -> auto request coalesces while A is active
        session._on_speech_started()
        session._on_speech_stopped(b"turn-a")
        stt.gate.set()
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await settle(100)
        assert session._auto_queued_epoch is not None
        # explicit cancel of A
        session.handle_event({"type": "response.cancel"})
        await driver.wait_for("response.done")
        await settle(150)
        assert driver.count("response.done") == 1, driver.types()
        assert driver.of_type("response.done")[0]["response"]["status"] == "cancelled"
        assert session._active_response is None
        assert session._response_task is None or session._response_task.done()
        # the queued auto was dropped with the cancel, no orphan response
        assert session._auto_queued_epoch is None
        await settle(150)
        assert driver.count("response.created") == 1
        assert driver.count("response.done") == 1
        # a subsequent explicit create works and completes cleanly
        llm.gate.set()
        session.handle_event({"type": "response.create"})
        done = await driver.wait_next("response.done", 1)
        assert done["response"]["status"] == "completed"
        assert driver.count("response.done") == 2
        assert len(llm.calls) == 2
        busy = [e for e in driver.of_type("error") if e["error"]["code"] == "already_active_response"]
        assert busy == []
    finally:
        await driver.stop()


async def test_tool_output_accepted_during_active_response(make_session):
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # tool results must be taken in immediately, even mid-response
        # (after their call: a result with no call before it is not valid
        # chat history and is left out of the request)
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            }
        )
        await driver.wait_for("conversation.item.created")
        driver.events.clear()
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'},
            }
        )
        created = await driver.wait_for("conversation.item.created")
        assert created["item"]["call_id"] == "call_1"
        # while the explicit-create rejection stays correct
        session.handle_event({"type": "response.create"})
        error = await driver.wait_for("error")
        assert error["error"]["code"] == "already_active_response"
        llm.gate.set()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        # and the tool output survives in the context for the next call
        session.handle_event({"type": "response.create"})
        await driver.wait_next("response.done", 1)
        messages = llm.calls[-1]["messages"]
        assert {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'} in messages
    finally:
        await driver.stop()


async def test_create_response_false_disables_auto_and_queued_flush(make_session):
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.create_response = False
    llm = FakeLLM()
    stt = FakeSTT(text="Nur transkribieren.")
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        await settle(150)
        assert driver.count("response.created") == 0, "auto responses must stay off"
        assert driver.of_type("error") == []
        # client-driven responses still work
        session.handle_event({"type": "response.create"})
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        await settle(150)
        assert driver.count("response.done") == 1
    finally:
        await driver.stop()


async def test_interrupt_response_false_keeps_audio_playing(make_session):
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.interrupt_response = False
    settings.session.turn_detection.create_response = False
    llm = FakeLLM(gate=asyncio.Event())
    tts = FakeTTS(chunks=4, inter_chunk_delay=0.01)
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # user speech starts mid-response: with interrupt_response=false the
        # active response must neither be cancelled nor truncated
        session._on_speech_started()
        await settle(100)
        assert driver.of_type("response.done") == []
        assert driver.of_type("conversation.item.truncated") == []
        llm.gate.set()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        assert driver.count("response.output_audio.delta") >= 4, "audio must not be cut"
        assert driver.count("response.created") == 1
    finally:
        await driver.stop()


async def test_cancel_before_start_then_bargein_epoch_usable(make_session):
    # cancel of a not-yet-started reservation, then a full VAD turn on the
    # same session: the epoch counter must be unaffected by the early cancel
    llm = FakeLLM(gate=asyncio.Event())
    llm2_gate = llm.gate
    stt = FakeSTT(text="Nach dem Cancel gesprochen.")
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        session.handle_event({"type": "response.cancel"})
        await settle()
        assert driver.count("response.done") == 1
        llm2_gate.set()
        await driver.append_speech_turn()
        done = await driver.wait_next("response.done", 1)
        assert done["response"]["status"] == "completed"
        assert len(llm.calls) == 1
        assert stt.calls, "turn after early cancel must still be transcribed"
    finally:
        await driver.stop()
