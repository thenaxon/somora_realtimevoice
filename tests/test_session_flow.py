"""Protocol flow, ordering, session.update merging, truncation."""

import base64

import pytest

from realtimevoice.config import Settings
from realtimevoice.session import SentenceBuffer

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends


async def run_turn(make_session, llm_script=None, stt_text="Hallo Welt. Wie geht es dir?"):
    llm = FakeLLM(script=llm_script or [])
    stt = FakeSTT(text=stt_text)
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(stt, llm, tts))
    await driver.append_speech_turn()
    await driver.wait_for("response.done")
    return session, driver, llm, stt, tts


async def test_session_created_first(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    assert driver.events[0]["type"] == "session.created"
    assert driver.events[0]["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}


async def asyncio_sleep():
    import asyncio

    await asyncio.sleep(0.05)


async def test_full_turn_event_order(make_session):
    session, driver, llm, stt, tts = await run_turn(make_session)
    types = driver.types()
    expected_prefix = [
        "session.created",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert types[: len(expected_prefix)] == expected_prefix
    expected_suffix = [
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    assert types[-len(expected_suffix) :] == expected_suffix
    # transcript deltas appear between content_part.added and done
    assert "response.output_audio_transcript.delta" in types
    # audio deltas exist and every one precedes response.done
    audio_deltas = [i for i, t in enumerate(types) if t == "response.output_audio.delta"]
    assert audio_deltas
    assert max(audio_deltas) < len(types) - 1


async def test_transcript_and_audio_content(make_session):
    session, driver, llm, stt, tts = await run_turn(make_session)
    done = driver.of_type("response.output_audio_transcript.done")[-1]
    assert done["transcript"] == "Guten Tag! Wie kann ich Ihnen helfen?"
    deltas = [e["delta"] for e in driver.of_type("response.output_audio_transcript.delta")]
    assert "".join(deltas) == done["transcript"]
    audio_deltas = driver.of_type("response.output_audio.delta")
    assert audio_deltas
    assert all(e["response_id"] == driver.of_type("response.created")[-1]["response"]["id"] for e in audio_deltas)


async def test_llm_receives_conversation_context(make_session):
    session, driver, llm, stt, tts = await run_turn(make_session, stt_text="Wie ist das Wetter?")
    messages = llm.calls[-1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Wie ist das Wetter?"}


async def test_conversation_history_accumulates(make_session):
    llm = FakeLLM()
    stt = FakeSTT(text="Erste Frage.")
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        stt.text = "Zweite Frage."
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        assert len(llm.calls) == 2
        second_messages = llm.calls[1]["messages"]
        user_contents = [m["content"] for m in second_messages if m.get("role") == "user"]
        assert user_contents == ["Erste Frage.", "Zweite Frage."]
    finally:
        await driver.stop()


async def test_stt_language_passed(make_session):
    session, driver, llm, stt, tts = await run_turn(make_session)
    assert stt.language_seen == "de"


async def test_session_update_partial_merge(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    voice_before = driver.events[0]["session"]["voice"]
    session.handle_event({"type": "session.update", "session": {"instructions": "Nur kurz antworten."}})
    updated = (await driver.wait_for("session.updated"))["session"]
    assert updated["instructions"] == "Nur kurz antworten."
    assert updated["voice"] == voice_before
    assert updated["audio"]["input"]["turn_detection"]["type"] == "server_vad"


async def test_session_update_nested_audio_and_turn_detection(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "transcription": {"language": "en"},
                        "turn_detection": {"silence_duration_ms": 700},
                    },
                    "output": {"voice": "serena"},
                }
            },
        }
    )
    updated = (await driver.wait_for("session.updated"))["session"]
    assert updated["language"] == "en"
    assert updated["audio"]["input"]["transcription"]["language"] == "en"
    assert updated["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 700
    assert updated["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert updated["audio"]["output"]["voice"] == "serena"
    assert updated["voice"] == "serena"


async def test_session_update_model_rejected(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event({"type": "session.update", "session": {"model": "other-model"}})
    error = await driver.wait_for("error")
    assert "model" in error["error"]["message"]
    # no session.updated emitted
    assert driver.of_type("session.updated") == []


async def test_session_update_invalid_format_rejected(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event(
        {
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/pcmu", "rate": 8000}}}},
        }
    )
    error = await driver.wait_for("error")
    assert "24000" in error["error"]["message"]


async def test_session_update_tools(make_session):
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio_sleep()
        session.handle_event(
            {
                "type": "session.update",
                "session": {
                    "tools": [
                        {"type": "function", "name": "get_weather", "description": "w", "parameters": {"type": "object"}}
                    ]
                },
            }
        )
        await driver.wait_for("session.updated")
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        assert llm.calls[-1]["tools"][0]["function"]["name"] == "get_weather"
    finally:
        await driver.stop()


async def test_response_create_with_oneoff_instructions(make_session):
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio_sleep()
        session.handle_event(
            {"type": "response.create", "response": {"instructions": "Sprich nur ein Wort."}}
        )
        await driver.wait_for("response.done")
        messages = llm.calls[-1]["messages"]
        assert messages[0]["content"] == "Sprich nur ein Wort."
    finally:
        await driver.stop()


async def test_response_create_while_active_errors(make_session):
    llm = FakeLLM(gate=__import__("asyncio").Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio_sleep()
        session.handle_event({"type": "response.create"})
        created = await driver.wait_for("response.created")
        session.handle_event({"type": "response.create"})
        error = await driver.wait_for("error")
        assert "already" in error["error"]["message"].lower()
        assert len(llm.calls) <= 1
    finally:
        await driver.stop()


async def test_unknown_event_type_errors(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event({"type": "totally.bogus"})
    error = await driver.wait_for("error")
    assert error["error"]["type"] == "unknown_event_type"


async def test_invalid_audio_append_errors(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event({"type": "input_audio_buffer.append", "audio": "!!!not-base64!!!"})
    error = await driver.wait_for("error")
    assert error["error"]["type"] == "invalid_audio_data"


async def test_manual_mode_commit(make_session):
    settings = Settings()
    settings.session.turn_detection.type = "none"
    llm = FakeLLM()
    stt = FakeSTT(text="Manuell gesprochen.")
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        from fake_backends import silence_pcm, tone_pcm

        await asyncio_sleep()
        audio = base64.b64encode(tone_pcm(400) + silence_pcm(100)).decode()
        session.handle_event({"type": "input_audio_buffer.append", "audio": audio})
        session.handle_event({"type": "input_audio_buffer.commit"})
        await driver.wait_for("conversation.item.input_audio_transcription.completed")
        # manual mode: the client drives responses
        assert driver.of_type("response.created") == []
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        completed = driver.of_type("conversation.item.input_audio_transcription.completed")
        assert stt.text in [e["transcript"] for e in completed]
    finally:
        await driver.stop()


async def test_commit_rejected_with_vad(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event({"type": "input_audio_buffer.commit"})
    error = await driver.wait_for("error")
    assert error["error"]["type"] == "buffer_commit_not_allowed"


async def test_truncate_cuts_transcript(make_session):
    session, driver, llm, stt, tts = await run_turn(
        make_session, llm_script=[{"text": "Erster Satz hier. Zweiter Satz folgt jetzt."}]
    )
    item_id = None
    for event in driver.of_type("response.output_item.done"):
        if event["item"]["type"] == "message":
            item_id = event["item"]["id"]
    assert item_id
    session.handle_event(
        {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 100}
    )
    truncated = await driver.wait_for("conversation.item.truncated")
    assert truncated["item_id"] == item_id
    assert truncated["audio_end_ms"] == 100


async def test_truncate_unknown_item_errors(make_session):
    session, driver = make_session()
    await asyncio_sleep()
    session.handle_event({"type": "conversation.item.truncate", "item_id": "nope", "audio_end_ms": 10})
    error = await driver.wait_for("error")
    assert error["error"]["type"] == "item_not_found"


async def test_item_create_function_call_output_roundtrip(make_session):
    llm = FakeLLM(
        script=[
            {"text": "", "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": '{"city": "Wien"}'}]},
            {"text": "In Wien ist es sonnig."},
        ]
    )
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio_sleep()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temp": 21}'},
            }
        )
        created = await driver.wait_for("conversation.item.created")
        # find the function_call_output item among created events
        outputs = [e for e in driver.of_type("conversation.item.created") if e["item"]["type"] == "function_call_output"]
        assert outputs and outputs[-1]["item"]["call_id"] == "call_1"
        session.handle_event({"type": "response.create"})
        deadline = __import__("asyncio").get_running_loop().time() + 3.0
        while len(llm.calls) < 2 and __import__("asyncio").get_running_loop().time() < deadline:
            await __import__("asyncio").sleep(0.005)
        await driver.wait_for("response.done")
        assert len(llm.calls) == 2
        messages = llm.calls[-1]["messages"]
        assert {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'} in messages
        assert messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'}
    finally:
        await driver.stop()


async def test_session_isolation(make_session):
    llm_a, llm_b = FakeLLM(), FakeLLM()
    s1, d1 = make_session(backends=make_backends(FakeSTT(), llm_a, FakeTTS()))
    s2, d2 = make_session(backends=make_backends(FakeSTT(), llm_b, FakeTTS()))
    try:
        await asyncio_sleep()
        assert s1.id != s2.id
        s1.handle_event({"type": "session.update", "session": {"instructions": "Nur A."}})
        await d1.wait_for("session.updated")
        s2.handle_event({"type": "session.update", "session": {"instructions": "Nur B."}})
        await d2.wait_for("session.updated")
        await d1.append_speech_turn()
        await d1.wait_for("response.done")
        await d2.append_speech_turn()
        await d2.wait_for("response.done")
        for m in llm_b.calls[-1]["messages"]:
            assert m.get("content") != "Nur A."
        assert llm_a.calls[-1]["messages"][0]["content"] == "Nur A."
        assert llm_b.calls[-1]["messages"][0]["content"] == "Nur B."
    finally:
        await d1.stop()
        await d2.stop()


def test_sentence_buffer_boundaries():
    buf = SentenceBuffer(min_chars=5, max_chars=100)
    segs = buf.push("Hallo! Wie geht")
    assert [s.text for s in segs] == ["Hallo!"]
    segs = buf.push(" es dir? Mir geht es gut.")
    assert [s.text for s in segs] == ["Wie geht es dir?"]
    segs = buf.flush()
    assert [s.text for s in segs] == ["Mir geht es gut."]


def test_sentence_buffer_force_split():
    import re

    buf = SentenceBuffer(min_chars=5, max_chars=30)
    long_text = "wort " * 20
    segs = buf.push(long_text) + buf.flush()
    assert segs
    assert all(len(s.text) <= 35 for s in segs)
    assert re.sub(r"\s+", "", "".join(s.text for s in segs)) == re.sub(r"\s+", "", long_text)
