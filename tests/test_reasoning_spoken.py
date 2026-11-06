"""R2 / F13 — Qwen3 reasoning must never be spoken (adapter + session + option).

A user report: with LLM Qwen3 14B the voice reads the model's thinking aloud;
Gemma does not. Measured on the real oMLX backend: when the token budget is hit
*while still thinking* the final ``content`` delta is a byte-identical copy of
the accumulated ``reasoning_content`` — with no tags. The tag stripper does not
recognise it, so it reached ``TextDelta``, TTS and history.

These tests replay the measured SSE sequences against the real adapter and
drive the real session; all assertions are metadata-only (lengths, flags, event
types) — reasoning text is only ever compared *by value within the test* and
never printed or logged.

The ``enable_thinking`` row option tests are guards over the existing
"extra is copied into the payload" behaviour plus the load/save round trip.
"""

from __future__ import annotations

import asyncio
import json
import struct

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from fake_backends import (
    FakeLLM,
    FakeSTT,
    FakeTTS,
    Driver,
    make_backends,
    tone_pcm,
)
from realtimevoice.backends.base import Finish, ReasoningMeta, TextDelta, ToolCallDelta
from realtimevoice.backends.openai_http import OpenAIHTTPLLM
from realtimevoice.config import BackendSettings, Settings
from realtimevoice.session import RealtimeSession


def _b64(pcm: bytes) -> str:
    import base64

    return base64.b64encode(pcm).decode("ascii")


# ----------------------------------------------------------- SSE fake backend


def sse_app(steps):
    """Fake OpenAI-compatible chat backend.

    ``steps`` is a list of ``{"delta": {...}, "finish_reason": ...}`` dicts.
    Exactly one SSE ``data:`` frame is emitted per step (so a single step with
    a big ``content`` replays the "whole duplicate in one delta" case)."""

    async def chat(request):
        async def gen():
            for step in steps:
                payload = {
                    "choices": [
                        {
                            "delta": step.get("delta", {}),
                            "finish_reason": step.get("finish_reason"),
                        }
                    ]
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def collect_events(app, *, model="llm"):
    settings = BackendSettings(base_url="http://backend.test/v1", model=model)
    events = []

    async def main():
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://backend.test"
        )
        try:
            llm = OpenAIHTTPLLM(settings, client)
            async for ev in llm.stream_chat([{"role": "user", "content": "hi"}]):
                events.append(ev)
        finally:
            await client.aclose()

    asyncio.run(main())
    return events


def _spoken(events):
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def _tag_stripped(events):
    from realtimevoice.backends.base import ReasoningStripped

    return sum(e.chars for e in events if isinstance(e, ReasoningStripped))


def _meta(events):
    metas = [e for e in events if isinstance(e, ReasoningMeta)]
    assert metas, "adapter must emit ReasoningMeta metadata at stream end"
    out = metas[-1]
    return {"reasoning_chars": out.reasoning_chars, "duplicate": out.duplicate_suppressed_chars}


# --------------------------------------------------------- adapter: duplicate


def test_duplicate_in_one_delta_with_length_is_not_spoken():
    reasoning = "Der Nutzer fragt nach dem Wetter, ich sollte kurz antworten."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": reasoning}, "finish_reason": "length"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == ""
    assert _meta(events) == {"reasoning_chars": len(reasoning), "duplicate": len(reasoning)}
    reasons = [e.reason for e in events if isinstance(e, Finish)]
    assert "length" in reasons


def test_duplicate_split_over_several_deltas_is_not_spoken():
    reasoning = "Ich überlege erst, dann spreche ich."
    mid = len(reasoning) // 2
    steps = [
        {"delta": {"reasoning_content": reasoning[:mid]}},
        {"delta": {"reasoning_content": reasoning[mid:]}},
        {"delta": {"content": reasoning[:mid]}},
        {"delta": {"content": reasoning[mid:]}, "finish_reason": "length"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == ""
    assert _meta(events)["duplicate"] == len(reasoning)
    assert _meta(events)["reasoning_chars"] == len(reasoning)


def test_reasoning_then_real_answer_stop_speaks_only_the_answer():
    reasoning = "Ich prüfe die Frage im Kopf."
    answer = "Es sind 42."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": answer[:5]}},
        {"delta": {"content": answer[5:]}, "finish_reason": "stop"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == answer
    assert _meta(events)["duplicate"] == 0
    assert _meta(events)["reasoning_chars"] == len(reasoning)


def test_reasoning_field_variant_duplicate_is_suppressed():
    reasoning = "Anderer Feldname, gleicher Effekt."
    steps = [
        {"delta": {"reasoning": reasoning}},
        {"delta": {"content": reasoning}, "finish_reason": "length"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == ""
    assert _meta(events)["duplicate"] == len(reasoning)


def test_complete_answer_that_is_a_strict_prefix_of_reasoning_with_stop_is_spoken():
    # C1: the end-of-stream drop must not swallow a COMPLETE genuine answer that
    # happens to be a strict prefix of the reasoning when the model finished
    # normally. The measured oMLX replay is always the whole reasoning with
    # finish_reason=length.
    reasoning = "Sieben. Danach folgen weitere Ausführungen, die niemand hören soll."
    answer = "Sieben."
    assert reasoning.startswith(answer) and answer != reasoning
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": answer}, "finish_reason": "stop"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == answer
    assert _meta(events)["duplicate"] == 0
    assert _meta(events)["reasoning_chars"] == len(reasoning)


def test_full_exact_replay_with_stop_is_still_suppressed():
    # An exact full replay is the replay by definition, whatever the finish.
    reasoning = "Die komplette Gedankenkette, wortgleich wiederholt."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": reasoning}, "finish_reason": "stop"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == ""
    assert _meta(events)["duplicate"] == len(reasoning)


def test_short_answer_that_starts_like_reasoning_is_spoken_once_diverged():
    reasoning = "Zuerst denke ich lange nach und dann antworte ich."
    # the answer starts as a prefix of the reasoning, then diverges
    first = "Zuerst denke ich lange nach"
    answer = first + " und sage 42."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": first}},
        {"delta": {"content": " und sage 42."}, "finish_reason": "stop"},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == answer
    assert _meta(events)["duplicate"] == 0


def test_gemma_style_content_only_stop_and_length_unchanged():
    stop_steps = [
        {"delta": {"content": "Hallo "}},
        {"delta": {"content": "Welt."}, "finish_reason": "stop"},
    ]
    assert _spoken(collect_events(sse_app(stop_steps))) == "Hallo Welt."
    assert _meta(collect_events(sse_app(stop_steps)))["reasoning_chars"] == 0

    length_steps = [
        {"delta": {"content": "Hallo "}},
        {"delta": {"content": "Welt, mehr"}, "finish_reason": "length"},
    ]
    assert _spoken(collect_events(sse_app(length_steps))) == "Hallo Welt, mehr"


def test_reasoning_then_tool_call_is_intact_with_no_text():
    reasoning = "Ich rufe das Werkzeug auf."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {
            "delta": {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "get_time", "arguments": "{}"}}
                ]
            }
        },
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == ""
    tools = [e for e in events if isinstance(e, ToolCallDelta)]
    assert tools and tools[0].name == "get_time"


def test_split_reasoning_tags_still_stripped():
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    steps = [
        {"delta": {"content": "Antwort: "}},
        {"delta": {"content": "<thi"}},
        {"delta": {"content": "nk>secret" + close_tag + " 42."}},
    ]
    events = collect_events(sse_app(steps))
    assert _spoken(events) == "Antwort:  42."
    assert _tag_stripped(events) > 0
    assert _meta(events)["duplicate"] == 0


# -------------------------------------------------- session: never spoken


class AdapterLLM:
    """Drive the REAL adapter inside a session (loop-bound client)."""

    def __init__(self, app, *, model="llm"):
        self._app = app
        self._model = model
        self.calls = 0

    async def stream_chat(self, messages, tools=None):
        self.calls += 1
        settings = BackendSettings(base_url="http://backend.test/v1", model=self._model)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self._app), base_url="http://backend.test"
        )
        try:
            llm = OpenAIHTTPLLM(settings, client)
            async for ev in llm.stream_chat(messages, tools):
                yield ev
        finally:
            await client.aclose()

    async def probe(self):
        from realtimevoice.backends.base import ProbeResult

        return ProbeResult(ok=True, detail="adapter llm", models=["llm"])


async def _drive_one_response(session, driver):
    session.handle_event({"type": "response.create"})
    return await driver.wait_for("response.done")


@pytest.fixture
def lifecycle(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [
        json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"
    ]


async def test_session_duplicate_length_speaks_nothing(make_session, lifecycle):
    reasoning = "Ich denke nach und komme nicht zum Ende, weil das Budget endet."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": reasoning}, "finish_reason": "length"},
    ]
    tts = FakeTTS()
    llm = AdapterLLM(sse_app(steps))
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await asyncio.sleep(0.05)
        await _drive_one_response(session, driver)
        deltas = [e["delta"] for e in driver.of_type("response.output_audio_transcript.delta")]
        assert deltas == []
        assert all(reasoning not in (c.get("text") or "") for c in tts.calls)
        assert tts.calls == []
        # no assistant item was created from reasoning
        created = driver.of_type("conversation.item.created")
        assert not any(e["item"].get("type") == "message" for e in created)
        finish_logs = [r for r in lifecycle() if r["action"] == "llm_finish"]
        assert finish_logs
        last = finish_logs[-1]
        assert last["no_answer"] is True
        assert last["output_chars"] == 0
        assert last["reasoning_chars"] == len(reasoning)
    finally:
        await driver.stop()


async def test_session_reasoning_then_answer_speaks_answer(make_session):
    reasoning = "Kurz überlegen."
    answer = "Antwort: 42."
    steps = [
        {"delta": {"reasoning_content": reasoning}},
        {"delta": {"content": answer}, "finish_reason": "stop"},
    ]
    tts = FakeTTS()
    llm = AdapterLLM(sse_app(steps))
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await asyncio.sleep(0.05)
        await _drive_one_response(session, driver)
        done = driver.of_type("response.output_audio_transcript.done")[-1]
        assert done["transcript"] == answer
        assert any(c["text"] == answer for c in tts.calls)
    finally:
        await driver.stop()


async def test_session_tool_round_trip_still_works(make_session):
    # reasoning, then a tool call, then a plain answer after the tool result
    reasoning = "Erst das Werkzeug."
    llm = FakeLLM(
        script=[
            {
                "text": "",
                "reasoning": reasoning,
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": '{"city": "Wien"}'}],
            },
            {"text": "In Wien ist es sonnig."},
        ]
    )
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await asyncio.sleep(0.05)
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temp": 21}'},
            }
        )
        await driver.wait_for("conversation.item.created")
        session.handle_event({"type": "response.create"})
        deadline = asyncio.get_running_loop().time() + 3.0
        while len(llm.calls) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        await driver.wait_for("response.done")
        assert len(llm.calls) == 2
        messages = llm.calls[-1]["messages"]
        assert {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'} in messages
    finally:
        await driver.stop()


# ------------------------------------------------------- thinking row option


def payload_capturing_app(captured):
    async def chat(request):
        captured.update(json.loads((await request.body()).decode()))

        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def _payload_for_extra(extra):
    captured = {}
    app = payload_capturing_app(captured)
    settings = BackendSettings(base_url="http://backend.test/v1", model="llm", extra=extra)

    async def main():
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://backend.test"
        )
        try:
            llm = OpenAIHTTPLLM(settings, client)
            async for _ in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass
        finally:
            await client.aclose()

    asyncio.run(main())
    return captured


def test_enable_thinking_off_off_on_reach_the_payload():
    off = _payload_for_extra({"chat_template_kwargs": {"enable_thinking": False}})
    assert off["chat_template_kwargs"]["enable_thinking"] is False
    on = _payload_for_extra({"chat_template_kwargs": {"enable_thinking": True}})
    assert on["chat_template_kwargs"]["enable_thinking"] is True


def test_enable_thinking_default_omits_the_key():
    payload = _payload_for_extra({"max_tokens": 200})
    assert "chat_template_kwargs" not in payload
    payload2 = _payload_for_extra({})
    assert "chat_template_kwargs" not in payload2


@pytest.fixture
def catalog_client(tmp_path):
    from fastapi.testclient import TestClient

    from fake_backends import make_backends
    from realtimevoice.app import create_app
    from realtimevoice.config import SettingsStore

    settings = Settings()
    for role in ("stt", "llm", "tts"):
        getattr(settings, role).base_url = "http://backend.test/v1"
    store = SettingsStore(data_dir=str(tmp_path), settings=settings)
    app = create_app(
        settings_store=store,
        backend_factory=lambda s, client=None: make_backends(),
        auth_token="tok",
    )
    with TestClient(app) as test_client:
        yield test_client


def _auth():
    return {"Authorization": "Bearer tok"}


def test_row_option_load_save_round_trip(catalog_client):
    r = catalog_client.post(
        "/api/profiles/llm",
        json={
            "id": "llm-qwen-think-off",
            "base_url": "http://backend.test/v1",
            "model": "qwen3-14b-4bit",
            "extra": {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.2},
        },
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    assert r.json()["extra"]["chat_template_kwargs"]["enable_thinking"] is False

    # a partial editor PUT that omits `extra` keeps the stored option
    r = catalog_client.put(
        "/api/profiles/llm/llm-qwen-think-off",
        json={"display_name": "Qwen3 14B (thinking off)"},
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    assert r.json()["extra"]["chat_template_kwargs"]["enable_thinking"] is False
    assert r.json()["extra"]["temperature"] == 0.2

    # switching the option to default removes the key again
    r = catalog_client.put(
        "/api/profiles/llm/llm-qwen-think-off",
        json={"extra": {"chat_template_kwargs": {"enable_thinking": None}}},
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    assert r.json()["extra"]["chat_template_kwargs"] == {}


def test_default_row_has_no_thinking_key_after_save(catalog_client):
    r = catalog_client.post(
        "/api/profiles/llm",
        json={
            "id": "llm-gemma-default",
            "base_url": "http://backend.test/v1",
            "model": "gemma-4-26b-a4b-it-4bit",
            "extra": {"max_tokens": 200},
        },
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    assert "chat_template_kwargs" not in r.json()["extra"]

    # a plain save (as the UI sends for "backend default") never adds the key
    r = catalog_client.put(
        "/api/profiles/llm/llm-gemma-default",
        json={"extra": {"max_tokens": 200}},
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    assert "chat_template_kwargs" not in r.json()["extra"]