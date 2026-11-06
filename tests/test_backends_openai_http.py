"""OpenAI-compatible HTTP adapter tests with a fake oMLX-style backend."""

import asyncio
import json
import struct

import httpx
import pytest

from realtimevoice.backends.openai_http import _redact, build_backends
from realtimevoice.backends.base import Backends
from realtimevoice.config import BackendSettings
from realtimevoice.errors import BackendError


def wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1) -> bytes:
    byte_rate = rate * channels * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, channels, rate, byte_rate, channels * 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def fake_app(transcript_text="Erkannt.", llm_chunks=None, tts_rate=24000):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def models(request):
        return JSONResponse({"data": [{"id": "m1"}, {"id": "m2"}]})

    async def transcriptions(request):
        form = await request.form()
        assert "file" in form
        assert form["model"]
        return JSONResponse({"text": transcript_text})

    async def chat(request):
        async def gen():
            for chunk in llm_chunks or ["Guten ", "Tag!"]:
                payload = {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def speech(request):
        pcm = struct.pack("<8h", 1, 2, 3, 4, 5, 6, 7, 8) * 20
        wav = wav_bytes(pcm, rate=tts_rate)

        async def gen():
            # deliberately split the header across network chunks
            yield wav[:5]
            yield wav[5:44]
            yield wav[44:100]
            yield wav[100:]

        return StreamingResponse(gen(), media_type="audio/wav")

    return Starlette(
        routes=[
            Route("/v1/models", models),
            Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/audio/speech", speech, methods=["POST"]),
        ]
    )


def starlette_app(routes):
    from starlette.applications import Starlette
    from starlette.routing import Route

    return Starlette(routes=[Route(path, endpoint, methods=methods) for path, endpoint, methods in routes])


def run_with_backends(app, coro_fn, stt_settings=None, llm_settings=None, tts_settings=None):
    """Create loop-bound backends + client, run coro_fn(backends), clean up."""

    async def main():
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://backend.test"
        )
        stt, llm, tts = build_backends(
            stt_settings or BackendSettings(),
            llm_settings or BackendSettings(),
            tts_settings or BackendSettings(),
            client,
        )
        backends = Backends(stt=stt, llm=llm, tts=tts, owned=False)
        try:
            return await coro_fn(backends)
        finally:
            await client.aclose()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()


V1 = "http://backend.test/v1"


def test_stt_transcription_json():
    async def body(backends):
        return await backends.stt.transcribe(wav_bytes(b"\x00\x00" * 100), language="de")

    assert run_with_backends(fake_app(), body, stt_settings=BackendSettings(base_url=V1, model="asr", api_key="sk-secret")) == "Erkannt."


def test_llm_streaming_text():
    async def body(backends):
        out = []
        async for ev in backends.llm.stream_chat([{"role": "user", "content": "hi"}]):
            out.append(ev)
        return out

    events = run_with_backends(fake_app(), body, llm_settings=BackendSettings(base_url=V1, model="llm"))
    texts = [e.text for e in events if type(e).__name__ == "TextDelta"]
    assert texts == ["Guten ", "Tag!"]
    assert any(type(e).__name__ == "Finish" for e in events)


def streaming_delta_app(delta_dicts):
    """Fake OpenAI-compatible chat backend yielding the given delta bodies."""
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    async def chat(request):
        async def gen():
            for delta in delta_dicts:
                payload = {"choices": [{"delta": delta, "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def collect(backends):
    async def body(b):
        out = []
        async for ev in b.llm.stream_chat([{"role": "user", "content": "hi"}]):
            out.append(ev)
        return out

    return run_with_backends(backends, body, llm_settings=BackendSettings(base_url=V1, model="llm"))


def _texts_and_stripped(events):
    from realtimevoice.backends.base import ReasoningStripped

    texts = [e.text for e in events if type(e).__name__ == "TextDelta"]
    stripped = sum(e.chars for e in events if isinstance(e, ReasoningStripped))
    return "".join(texts), stripped


def test_llm_reasoning_block_split_across_deltas_is_not_spoken():
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    deltas = [
        {"content": "Antwort: "},
        {"content": "<thi"},
        {"content": "nk>secret reasoning"},
        {"content": " continues</thin"},
        {"content": "k> 42."},
    ]
    events = collect(streaming_delta_app(deltas))
    text, stripped = _texts_and_stripped(events)
    assert text == "Antwort:  42."
    assert stripped > 0
    assert "secret reasoning" not in text


def test_llm_unterminated_reasoning_block_is_not_spoken():
    open_tag = "<" + "think" + ">"
    deltas = [{"content": "Kurz: "}, {"content": open_tag + "still thinking"}]
    events = collect(streaming_delta_app(deltas))
    text, stripped = _texts_and_stripped(events)
    assert text == "Kurz: "
    assert stripped == len(open_tag) + len("still thinking")


def test_llm_normal_text_unaffected_by_stripper():
    deltas = [{"content": "Hallo "}, {"content": "Welt, 2 < 3."}]
    events = collect(streaming_delta_app(deltas))
    text, stripped = _texts_and_stripped(events)
    assert text == "Hallo Welt, 2 < 3."
    assert stripped == 0


def test_llm_reasoning_content_field_is_ignored():
    deltas = [
        {"reasoning_content": "internal chain of thought"},
        {"content": "sichtbare Antwort"},
        {"reasoning": "more hidden"},
    ]
    events = collect(streaming_delta_app(deltas))
    text, _ = _texts_and_stripped(events)
    assert text == "sichtbare Antwort"
    assert "internal chain of thought" not in text
    assert "more hidden" not in text


def test_llm_tool_calls_unaffected_by_stripper():
    from realtimevoice.backends.base import ToolCallDelta

    deltas = [
        {"content": "<" + "think" + ">plan the call</" + "think" + ">"},
        {
            "tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "get_time", "arguments": "{}"}}
            ]
        },
    ]
    events = collect(streaming_delta_app(deltas))
    text, _ = _texts_and_stripped(events)
    assert text == ""
    tool = [e for e in events if isinstance(e, ToolCallDelta)]
    assert tool and tool[0].name == "get_time"


def test_tts_streaming_split_wav_and_resample():
    async def body(backends):
        out = []
        async for pcm in backends.tts.stream_speech("Hallo", voice="serena", language="de"):
            out.append(pcm)
        return out

    chunks = run_with_backends(
        fake_app(tts_rate=48000), body,
        tts_settings=BackendSettings(base_url=V1, model="tts", language="de", language_format="name", voice="serena"),
    )
    assert chunks
    joined = b"".join(chunks)
    # 160 mono samples at 48 kHz -> 80 samples at 24 kHz -> 160 bytes
    assert len(joined) == 2 * 80


LANGUAGE_CASES = [
    # (set language, model's language_format, session language, sent)
    ("de", "code", "", "de"),
    ("de", "name", "", "German"),
    ("de-AT", "code", "", "de"),
    ("de-AT", "name", "", "German"),
    ("de", "none", "", None),
    # the set's language wins over the session's
    ("en", "code", "de", "en"),
    ("en", "name", "de", "English"),
    # no set language: the session's, in the model's format
    ("", "code", "de", "de"),
    ("", "name", "de", "German"),
    ("", "none", "de", None),
    ("", "code", "auto", None),
    ("", "code", "", None),
]


def _speech_capture_app(captured):
    from starlette.responses import Response

    async def speech(request):
        captured.update(json.loads((await request.body()).decode()))
        return Response(content=wav_bytes(struct.pack("<2h", 1, 2)), media_type="audio/wav")

    return starlette_app([("/v1/audio/speech", speech, ["POST"])])


def _transcription_capture_app(captured):
    from starlette.responses import JSONResponse

    async def transcriptions(request):
        form = await request.form()
        captured["sent"] = "language" in form
        captured["language"] = form.get("language")
        return JSONResponse({"text": "ok"})

    return starlette_app([("/v1/audio/transcriptions", transcriptions, ["POST"])])


@pytest.mark.parametrize("set_language,language_format,session_language,sent", LANGUAGE_CASES)
def test_tts_payload_language_in_the_models_format(set_language, language_format, session_language, sent):
    captured = {}

    async def body(backends):
        async for _ in backends.tts.stream_speech("x", voice=None, language=session_language or None):
            pass

    run_with_backends(
        _speech_capture_app(captured), body,
        tts_settings=BackendSettings(base_url=V1, model="tts", language=set_language, language_format=language_format),
    )
    assert captured.get("language") == sent
    assert ("language" in captured) is (sent is not None)
    assert captured["voice"] == "default"


@pytest.mark.parametrize("set_language,language_format,session_language,sent", LANGUAGE_CASES)
def test_stt_form_language_in_the_models_format(set_language, language_format, session_language, sent):
    captured = {}

    async def body(backends):
        await backends.stt.transcribe(wav_bytes(b"\x00\x00"), language=session_language or None)

    run_with_backends(
        _transcription_capture_app(captured), body,
        stt_settings=BackendSettings(base_url=V1, model="asr", language=set_language, language_format=language_format),
    )
    assert captured["language"] == sent
    assert captured["sent"] is (sent is not None)


def test_stt_http_error_surfaced_without_raw_body_leak():
    from starlette.responses import JSONResponse

    async def transcriptions(request):
        return JSONResponse(
            {"error": {"message": "model not found; context Bearer sk-toplevel-secret-9999"}},
            status_code=500,
        )

    app = starlette_app([("/v1/audio/transcriptions", transcriptions, ["POST"])])
    async def body(backends):
        await backends.stt.transcribe(wav_bytes(b"\x00\x00"), language="de")

    with pytest.raises(BackendError) as excinfo:
        run_with_backends(app, body, stt_settings=BackendSettings(base_url=V1, model="asr"))
    assert "model not found" in str(excinfo.value)
    assert "sk-toplevel-secret-9999" not in str(excinfo.value)


def test_llm_unreachable_raises_backend_error():
    async def main():
        async with httpx.AsyncClient() as client:
            stt, llm, tts = build_backends(
                BackendSettings(),
                BackendSettings(base_url="http://127.0.0.1:9/v1", model="x"),
                BackendSettings(),
                client,
            )
            async for _ in llm.stream_chat([]):
                pass

    with pytest.raises(BackendError):
        asyncio.run(main())


def test_redact_strips_secrets():
    text = "Bearer sk-abcdef12345678 failed api_key=supersecret Authorization: Bearer xyz"
    out = _redact(text)
    assert "sk-abcdef12345678" not in out
    assert "supersecret" not in out
    assert "[redacted]" in out


def test_probe_lists_models():
    async def body(backends):
        return await backends.stt.probe()

    result = run_with_backends(fake_app(), body, stt_settings=BackendSettings(base_url=V1))
    assert result.ok
    assert result.models == ["m1", "m2"]


def test_probe_model_listed_ok():
    """P1: configured model id present in a non-empty /models list -> ok."""

    async def body(backends):
        return await backends.stt.probe()

    result = run_with_backends(
        fake_app(), body, stt_settings=BackendSettings(base_url=V1, model="m1")
    )
    assert result.ok is True
    assert result.error_code == ""
    assert result.models == ["m1", "m2"]


def test_probe_model_not_listed_fails():
    """P1: configured model id absent from a non-empty /models list -> not ok
    with error_code=model_not_listed."""

    async def body(backends):
        return await backends.stt.probe()

    for role_settings in (dict(stt_settings=BackendSettings(base_url=V1, model="ghost")),):
        result = run_with_backends(fake_app(), body, **role_settings)
        assert result.ok is False
        assert result.error_code == "model_not_listed"
        assert "ghost" in result.detail


def test_probe_no_model_list_keeps_reachability():
    """P1: an empty /models list keeps the previous reachability-only behaviour."""

    async def body(backends):
        return await backends.stt.probe()

    from realtimevoice.backends.openai_http import _probe

    async def main():
        async def empty_models(request):
            from starlette.responses import JSONResponse

            return JSONResponse({"data": []})

        app = starlette_app([("/v1/models", empty_models, ["GET"])])
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend.test")
        try:
            result = await _probe(V1, {}, client, "any-model")()
        finally:
            await client.aclose()
        return result

    result = asyncio.run(main())
    assert result.ok is True
    assert result.error_code == ""
    assert result.models == []


def test_probe_non_json_models_keeps_reachability():
    """P1: a non-JSON /models body (no list) keeps reachability-only."""

    async def main():
        from realtimevoice.backends.openai_http import _probe
        from starlette.responses import PlainTextResponse

        async def text_models(request):
            return PlainTextResponse("not json")

        app = starlette_app([("/v1/models", text_models, ["GET"])])
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend.test")
        try:
            result = await _probe(V1, {}, client, "any-model")()
        finally:
            await client.aclose()
        return result

    result = asyncio.run(main())
    assert result.ok is True
    assert result.error_code == ""


def test_backends_owned_close():
    async def main():
        stt, llm, tts = build_backends(BackendSettings(), BackendSettings(), BackendSettings(), None)
        backends = Backends(stt=stt, llm=llm, tts=tts, owned=True)
        await backends.aclose()
        return stt._client.is_closed

    assert asyncio.run(main())


def test_backends_not_owned_noop_close():
    async def main():
        backends = Backends(owned=False)
        await backends.aclose()

    asyncio.run(main())
