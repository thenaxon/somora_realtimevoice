"""Lifecycle log tests: correlation metadata present, content never leaks.

The lifecycle logger must emit structured allowlisted metadata (session/turn/
response ids, actions, codes, durations, queue depths) and must NEVER contain
transcripts, tool arguments/results, audio/base64, tokens or raw backend
exception text. Client-facing error messages are sanitized too.
"""

import asyncio
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.errors import BackendError


class LeakySTT(FakeSTT):
    """Pretends a backend echoed request context into its error message."""

    async def transcribe(self, wav, *, language):
        self.calls.append(wav)
        self.language_seen = language
        raise BackendError(
            "upstream 500 at https://stt.internal.example:9443/v1/transcribe"
            " Authorization=Bearer sk-totallysecret123456 trace=AAAAaaaaaaaaBBBBccccccccDDDDdddddddd",
            code="stt_http_error",
        )


class ExplodingTTS(FakeTTS):
    async def stream_speech(self, text, *, voice=None, language=None):
        self.calls.append({"text": text, "voice": voice, "language": language})
        raise RuntimeError(
            "connection to http://10.42.0.7:8080/speak failed; api_key=hunter2xyz token=abcdef0123456789"
        )
        yield b""  # pragma: no cover — makes this an async generator like the real backend


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


async def test_full_turn_lifecycle_correlation(make_session, lifecycle):
    llm = FakeLLM()
    stt = FakeSTT(text="Korrelationsfrage.")
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        await settle(100)
        records = lifecycle()
        actions = [r["action"] for r in records]
        # backend + lifecycle coverage required by the diagnostic contract
        for action in (
            "session_open",
            "speech_started",
            "speech_stopped",
            "stt_started",
            "stt_completed",
            "response_reserved",
            "response_started",
            "tts_started",
            "tts_completed",
            "response_done",
        ):
            assert action in actions, f"missing lifecycle action {action}: {actions}"
        # every record carries the session id
        assert all(r["session_id"] == session.id for r in records)
        # open mapping includes the configured backend models
        opened = next(r for r in records if r["action"] == "session_open")
        assert opened["model"] == session.model
        assert {"stt_model", "llm_model", "tts_model"} <= set(opened)
        # stable turn attribution: response events pin the epoch they were
        # reserved in, stt events the epoch of their job
        reserved = [r for r in records if r["action"] == "response_reserved"]
        assert reserved and reserved[0]["turn_epoch"] == records[-1]["turn_epoch"]
        stt_started = next(r for r in records if r["action"] == "stt_started")
        stt_completed = next(r for r in records if r["action"] == "stt_completed")
        assert stt_started["turn_epoch"] == stt_completed["turn_epoch"]
        assert stt_started["item_id"] == stt_completed["item_id"]
        # response ids correlate across reserve/start/done
        response_ids = {
            r["response_id"]
            for r in records
            if r["action"] in ("response_reserved", "response_started", "response_done")
        }
        assert response_ids == {done["response"]["id"]}
        # durations and queue depths are present where specified
        done_rec = next(r for r in records if r["action"] == "response_done")
        assert isinstance(done_rec["duration_ms"], (int, float)) and done_rec["duration_ms"] >= 0
        assert isinstance(reserved[0]["out_queue"], int)
        assert done_rec["audio_ms"] > 0
        assert stt_completed["duration_ms"] >= 0
        # backend/model mapping on backend diagnostics
        assert stt_completed["backend"] == "stt" and "model" in stt_completed
        tts_done = next(r for r in records if r["action"] == "tts_completed")
        assert tts_done["backend"] == "tts" and "model" in tts_done
        # client request correlation via event_id
        session.handle_event({"type": "response.create", "event_id": "evt_client_1"})
        await driver.wait_next("response.done", 1)
        await settle(50)
        records = lifecycle()
        rejected_free = [r for r in records if r["action"] == "response_reserved"]
        assert any(r.get("event_id") == "evt_client_1" for r in rejected_free)
    finally:
        await driver.stop()


async def test_lifecycle_never_leaks_content(make_session, lifecycle):
    secret_args = '{"city": "Wien", "secret_payload": "TOPSECRET42"}'
    llm = FakeLLM(
        script=[
            {"text": "", "tool_calls": [{"id": "call_9", "name": "get_weather", "arguments": secret_args}]},
            {"text": "Erledigt."},
        ]
    )
    stt = FakeSTT(text="GEHEIMERUSERINHALT123")
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_9", "output": '{"result": "TOPSECRETRESULT"}'},
            }
        )
        session.handle_event({"type": "response.create"})
        await driver.wait_next("response.done", 1)
        await settle(100)
        raw = "\n".join(json.dumps(r) for r in lifecycle())
        for leak in ("GEHEIMERUSERINHALT123", "TOPSECRET42", "TOPSECRETRESULT", "Wien"):
            assert leak not in raw
        # tool correlation uses call_id, not content
        tool_rec = next(r for r in lifecycle() if r["action"] == "tool_call")
        assert tool_rec["call_id"] == "call_9"
        result_rec = next(r for r in lifecycle() if r["action"] == "tool_result_received")
        assert result_rec["call_id"] == "call_9"
        # transcripts reach the client but never the lifecycle log
        assert any(e.get("transcript") == "GEHEIMERUSERINHALT123" for e in driver.events)
    finally:
        await driver.stop()


class _StrippingLLM(FakeLLM):
    """Emits a speakable TextDelta plus a ReasoningStripped metadata cue, as
    the HTTP adapter does after filtering a reasoning block."""

    async def stream_chat(self, messages, tools=None):
        from realtimevoice.backends.base import Finish, ReasoningStripped, TextDelta

        self.calls.append({"messages": messages, "tools": tools})
        yield TextDelta("Die Antwort ist 42.")
        yield ReasoningStripped(321)
        yield Finish("stop")


async def test_reasoning_chars_stripped_is_logged_metadata_only(make_session, lifecycle):
    llm = _StrippingLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(text="Frage?"), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(80)
        finish = next(r for r in lifecycle() if r["action"] == "llm_finish")
        assert finish["reasoning_chars_stripped"] == 321
        # only the speakable text reaches the client transcript
        transcripts = "".join(
            e.get("delta", "") for e in driver.events if e.get("type") == "response.output_audio_transcript.delta"
        )
        assert transcripts == "Die Antwort ist 42."
        # the metadata value is a number, never any reasoning text
        assert isinstance(finish["reasoning_chars_stripped"], int)
    finally:
        await driver.stop()


async def test_backend_error_sanitized_for_client_and_log(make_session, lifecycle):
    session, driver = make_session(backends=make_backends(LeakySTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        error = await driver.wait_for("error")
        assert error["error"]["code"] == "stt_http_error"
        message = error["error"]["message"]
        # no URLs, hosts, bearer/sk secrets or long blobs survive
        for leak in ("https://", "stt.internal.example", "9443", "Bearer", "sk-totallysecret123456", "trace="):
            assert leak not in message, message
        raw = "\n".join(json.dumps(r) for r in lifecycle())
        for leak in ("https://", "stt.internal.example", "sk-totallysecret123456", "hunter2xyz"):
            assert leak not in raw
        # error code and correlation remain
        assert any(r["action"] == "stt_failed" and r["code"] == "stt_http_error" for r in lifecycle())
        assert any(r["action"] == "error_emitted" and r["code"] == "stt_http_error" for r in lifecycle())
    finally:
        await driver.stop()


async def test_unexpected_tts_exception_sanitized(make_session, lifecycle):
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, ExplodingTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "incomplete"
        error = driver.of_type("error")[-1]
        assert error["error"]["code"] == "tts_error"
        message = error["error"]["message"]
        for leak in ("http://", "10.42.0.7", "hunter2xyz", "abcdef0123456789", "token="):
            assert leak not in message, message
        raw = "\n".join(json.dumps(r) for r in lifecycle())
        for leak in ("10.42.0.7", "hunter2xyz", "abcdef0123456789"):
            assert leak not in raw
        records = lifecycle()
        failed = [r for r in records if r["action"] == "tts_failed"]
        assert failed and failed[0]["code"] == "tts_error"
        assert failed[0]["backend"] == "tts"
        # the exception class name is safe diagnostic metadata
        assert "RuntimeError" in json.dumps(failed[0])
    finally:
        await driver.stop()


async def test_cancel_and_supersede_lifecycle_actions(make_session, lifecycle):
    from realtimevoice.config import Settings

    settings = Settings()
    settings.session.turn_detection.interrupt_response = False
    llm = FakeLLM(gate=asyncio.Event())
    stt = FakeSTT(text="Abgebrochenes Turn.")
    stt.gate = asyncio.Event()
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create", "event_id": "evt_create_7"})
        await driver.wait_for("response.created")
        session.handle_event({"type": "response.cancel", "event_id": "evt_cancel_8"})
        await driver.wait_for("response.done")
        records = lifecycle()
        actions = [r["action"] for r in records]
        assert "response_cancel_requested" in actions
        assert "response_cancelled" in actions
        cancelled = next(r for r in records if r["action"] == "response_cancelled")
        assert cancelled["reason"] == "client_cancel"
        # client event_ids correlated on the reserved/cancel records
        reserved = next(r for r in records if r["action"] == "response_reserved")
        assert reserved.get("event_id") == "evt_create_7"
        # the cancelled done carries the epoch the response was reserved in
        done_rec = next(r for r in records if r["action"] == "response_done")
        assert done_rec["status"] == "cancelled"
        assert done_rec["turn_epoch"] == reserved["turn_epoch"]
    finally:
        await driver.stop()
