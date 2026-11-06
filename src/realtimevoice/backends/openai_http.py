"""Generic OpenAI-compatible HTTP adapters for oMLX-style servers.

All timeouts, connection errors and non-200 answers surface as BackendError;
nothing here blocks the caller's event loop beyond streaming reads that httpx
schedules asynchronously.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
from typing import Any, AsyncIterator

import httpx

from ..audio import Mp3Decoder, PcmParser, WavParser, resample_pcm16, to_mono
from ..config import BackendSettings, language_for_model, reference_fingerprint
from ..errors import BackendError
from ..reasoning import ReasoningStripper
from .base import (
    Finish,
    LLMEvent,
    ProbeResult,
    ReasoningMeta,
    ReasoningStripped,
    TextDelta,
    ToolCallDelta,
)

DEFAULT_TIMEOUT = 60.0

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|authorization)\s*[=:]\s*\S+"),
)


def _redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def _timeout(settings: BackendSettings) -> httpx.Timeout:
    read = max(1.0, float(settings.request_timeout_s or DEFAULT_TIMEOUT))
    return httpx.Timeout(connect=10.0, read=read, write=read, pool=read)


def _headers(settings: BackendSettings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}


def _probe(base_url: str, headers: dict[str, str], client: httpx.AsyncClient, model: str = "") -> Any:
    async def probe() -> ProbeResult:
        url = base_url.rstrip("/") + "/models"
        try:
            resp = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            return ProbeResult(ok=False, detail=f"connection failed: {exc}")
        if resp.status_code in (404, 405):
            # the server answered (and took the key), it just lists no models
            # at this address, e.g. a router's special route for one model
            return ProbeResult(ok=True, detail="reachable (no model list at this address)")
        if resp.status_code >= 400:
            return ProbeResult(ok=False, detail=f"HTTP {resp.status_code} from {url}")
        try:
            body = resp.json()
        except ValueError:
            return ProbeResult(ok=True, detail="reachable (non-JSON /models)")
        models = []
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            models = [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]
        # P1: when the backend returns a model list, the configured model id must
        # be in it; an absent id is a readiness failure the operator must see.
        # An empty/absent list keeps the reachability-only behaviour.
        if models and model and model not in models:
            return ProbeResult(
                ok=False,
                detail=f"model {model!r} is not listed by the backend ({len(models)} models offered)",
                models=models,
                error_code="model_not_listed",
            )
        return ProbeResult(ok=True, detail="reachable", models=models)

    return probe


def build_backends(
    settings_stt: BackendSettings,
    settings_llm: BackendSettings,
    settings_tts: BackendSettings,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, Any, Any]:
    owned = client is None
    http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    stt = OpenAIHTTPSTT(settings_stt, http, owned)
    llm = OpenAIHTTPLLM(settings_llm, http, owned)
    tts = OpenAIHTTPTTS(settings_tts, http, owned)
    return stt, llm, tts


class OpenAIHTTPSTT:
    def __init__(self, settings: BackendSettings, client: httpx.AsyncClient, owned: bool = False):
        self._settings = settings
        self._client = client
        self._owned = owned

    async def transcribe(self, wav: bytes, *, language: str | None) -> str:
        base = self._settings.base_url.rstrip("/")
        if not base:
            raise BackendError("STT base_url not configured", code="stt_not_configured")
        data: dict[str, str] = {"model": self._settings.model}
        # the set's language wins over the session's, in the model's format
        lang = language_for_model(self._settings.language or language or "", self._settings.language_format)
        if lang:
            data["language"] = lang
        for key, value in (self._settings.extra or {}).items():
            data[str(key)] = str(value)
        try:
            resp = await self._client.post(
                f"{base}/audio/transcriptions",
                headers=_headers(self._settings),
                files={"file": ("audio.wav", wav, "audio/wav")},
                data=data,
                timeout=_timeout(self._settings),
            )
        except httpx.TimeoutException as exc:
            raise BackendError(f"STT timed out: {exc}", code="stt_timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"STT request failed: {_redact(str(exc))}", code="stt_unreachable") from exc
        if resp.status_code >= 400:
            raise BackendError(
                f"STT backend returned HTTP {resp.status_code}: {_body_message(resp)}",
                code="stt_http_error",
            )
        return _parse_transcription(resp)

    async def probe(self) -> ProbeResult:
        return await _probe(
            self._settings.base_url, _headers(self._settings), self._client, self._settings.model
        )()

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()


class OpenAIHTTPLLM:
    def __init__(self, settings: BackendSettings, client: httpx.AsyncClient, owned: bool = False):
        self._settings = settings
        self._client = client
        self._owned = owned

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        base = self._settings.base_url.rstrip("/")
        if not base:
            raise BackendError("LLM base_url not configured", code="llm_not_configured")
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        for key, value in (self._settings.extra or {}).items():
            payload[str(key)] = value
        try:
            async with self._client.stream(
                "POST",
                f"{base}/chat/completions",
                headers={**_headers(self._settings), "Content-Type": "application/json"},
                json=payload,
                timeout=_timeout(self._settings),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise BackendError(
                        f"LLM backend returned HTTP {resp.status_code}: {_text_message(body)}",
                        code="llm_http_error",
                    )
                finished = False
                stripper = ReasoningStripper()
                # Reasoning-aware duplicate guard. Some reasoning backends (e.g.
                # oMLX Qwen3 when the token budget ends while it is still
                # thinking) replay the accumulated chain of thought as the
                # FINAL ordinary `content` delta — byte-identical and without
                # tags, so the tag stripper cannot see it. We accumulate the
                # reasoning fields and HOLD `content` while it is still a
                # prefix of that reasoning; held text is released the moment it
                # diverges (a genuine answer) and dropped at stream end when it
                # only ever reproduced the reasoning.
                reasoning_parts: list[str] = []
                reasoning_chars = 0
                duplicate_suppressed = 0
                held = ""
                content_released = False

                def _speak(text: str) -> list[LLMEvent]:
                    """Feed released content through the tag stripper."""
                    out: list[LLMEvent] = []
                    before = stripper.stripped_chars
                    speakable = stripper.feed(text)
                    if speakable:
                        out.append(TextDelta(speakable))
                    if stripper.stripped_chars != before:
                        out.append(ReasoningStripped(stripper.stripped_chars - before))
                    return out

                byte_stream = resp.aiter_bytes()
                async with contextlib.aclosing(byte_stream):
                    async with contextlib.aclosing(_iter_sse(byte_stream)) as sse_events:
                        async for event in sse_events:
                            kind = event.get("type")
                            if kind == "error":
                                raise BackendError(
                                    f"LLM backend error: {event.get('message', 'unknown')}",
                                    code="llm_error",
                                )
                            if kind is None:
                                choices = event.get("choices") or []
                                delta = (choices[0].get("delta") or {}) if choices else {}
                                finish = choices[0].get("finish_reason") if choices else None
                                # reasoning_content/reasoning deltas are never
                                # speech: accumulate their length only (the
                                # text itself is never logged or carried)
                                for field in ("reasoning_content", "reasoning"):
                                    value = delta.get(field)
                                    if isinstance(value, str) and value:
                                        reasoning_parts.append(value)
                                        reasoning_chars += len(value)
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    if content_released or not reasoning_parts:
                                        # no reasoning seen (yet): normal text
                                        content_released = True
                                        for ev in _speak(content):
                                            yield ev
                                    else:
                                        reasoning_text = "".join(reasoning_parts)
                                        held += content
                                        if reasoning_text.startswith(held):
                                            # still could be a verbatim replay
                                            pass
                                        else:
                                            # diverged: a genuine answer
                                            content_released = True
                                            released, held = held, ""
                                            for ev in _speak(released):
                                                yield ev
                                for tc in delta.get("tool_calls") or []:
                                    if not isinstance(tc, dict):
                                        continue
                                    fn = tc.get("function") or {}
                                    yield ToolCallDelta(
                                        index=int(tc.get("index", 0)),
                                        id=tc.get("id"),
                                        name=fn.get("name"),
                                        arguments_delta=fn.get("arguments") or "",
                                    )
                                if finish and not finished:
                                    finished = True
                                    yield Finish(str(finish))
                            elif kind == "transcript.text.delta":
                                pass
                # end of stream: a still-held `content` is dropped as the
                # duplicated-thinking replay ONLY when it is provably that
                # replay — an EXACT full copy of the reasoning (the measured
                # oMLX case, any finish reason) or a prefix of it when the
                # budget cut the stream (`length`). A complete genuine answer
                # that merely happens to be a strict prefix of the reasoning and
                # finished normally is released through the tag stripper.
                if held:
                    reasoning_text = "".join(reasoning_parts)
                    exact_replay = bool(reasoning_text) and held == reasoning_text
                    length_cut = finish == "length" and reasoning_text.startswith(held)
                    if exact_replay or length_cut:
                        duplicate_suppressed += len(held)
                    else:
                        for ev in _speak(held):
                            yield ev
                    held = ""
                # release any buffered non-tag tail so normal text that merely
                # ended on a '<' is not lost; flush() may also drop an
                # unterminated reasoning block, count it too
                flushed_before = stripper.stripped_chars
                tail = stripper.flush()
                if tail:
                    yield TextDelta(tail)
                if stripper.stripped_chars != flushed_before:
                    yield ReasoningStripped(stripper.stripped_chars - flushed_before)
                yield ReasoningMeta(
                    reasoning_chars=reasoning_chars,
                    duplicate_suppressed_chars=duplicate_suppressed,
                )
        except httpx.TimeoutException as exc:
            raise BackendError(f"LLM timed out: {exc}", code="llm_timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"LLM request failed: {_redact(str(exc))}", code="llm_unreachable") from exc
        yield Finish(None)

    async def probe(self) -> ProbeResult:
        return await _probe(
            self._settings.base_url, _headers(self._settings), self._client, self._settings.model
        )()

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()


def _tts_payload(settings: BackendSettings, text: str, voice: str | None, language: str | None) -> dict[str, Any]:
    """The exact JSON body the TTS adapter sends: base fields first, then the
    profile's extra keys merged OVER them (extra can override base fields,
    e.g. voice). Both stream_speech and the effective-voice diagnostic use
    this one assembly so they can never diverge."""
    payload: dict[str, Any] = {
        "model": settings.model,
        "input": text,
        "voice": voice or settings.voice or "default",
        "response_format": settings.audio_format,
        "stream": True,
    }
    # the set's language wins over the session's, in the model's format
    # (e.g. Qwen TTS wants "German" for "de")
    lang = language_for_model(settings.language or language or "", settings.language_format)
    if lang:
        payload["language"] = lang
    for key, value in (settings.extra or {}).items():
        payload[str(key)] = value
    return payload


async def _sse_audio(resp: httpx.Response):
    """Audio from an OpenAI-style speech event stream: `speech.audio.delta`
    events carry base64 audio (in the requested format) and may name its
    sample rate. Yields (sample_rate or 0, bytes); an error event raises."""
    buf = ""
    async for text in resp.aiter_text():
        buf += text
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error" or isinstance(event.get("error"), dict):
                err = event.get("error") if isinstance(event.get("error"), dict) else event
                raise BackendError(
                    f"TTS backend reported an error: {_redact(str(err.get('message') or err))[:300]}",
                    code="tts_http_error",
                )
            audio = event.get("audio") or event.get("delta")
            if isinstance(audio, str) and audio:
                try:
                    chunk = base64.b64decode(audio)
                except ValueError:
                    continue
                yield int(event.get("sample_rate") or 0), chunk


def _audio_decoder(settings: BackendSettings, response_format: str, rate: int | None = None):
    """The parser for what this speaking model answers with: the format the
    request actually asked for (an `extra` field may override it)."""
    if response_format == "pcm":
        return PcmParser(rate or settings.pcm_sample_rate)
    if response_format == "mp3":
        return Mp3Decoder()
    return WavParser()


class OpenAIHTTPTTS:
    def __init__(self, settings: BackendSettings, client: httpx.AsyncClient, owned: bool = False):
        self._settings = settings
        self._client = client
        self._owned = owned
        self._ref_fingerprint: str | None = None

    def reference_fingerprint(self) -> str:
        """Safe metadata: SHA256 of the EFFECTIVE per-request voice reference
        this adapter actually sends (literal extra["ref_audio"] string, same
        convention as the catalog views). "" = no reference material. Cached:
        the frozen settings cannot change during a session."""
        if self._ref_fingerprint is None:
            self._ref_fingerprint = reference_fingerprint(
                self._settings.extra, self._settings.reference_audio_field
            )
        return self._ref_fingerprint

    def effective_request_voice(self, voice: str | None) -> str:
        """The voice this adapter would ACTUALLY put into the HTTP payload for
        a request asked with `voice` — after extra overrides. The session
        voice is only the base; extra keys merge over the base fields, so the
        effective voice can differ (diagnostic honesty: both are logged)."""
        return str(_tts_payload(self._settings, "", voice, None)["voice"])

    async def stream_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield mono PCM16 at 24 kHz regardless of backend output format."""
        base = self._settings.base_url.rstrip("/")
        if not base:
            raise BackendError("TTS base_url not configured", code="tts_not_configured")
        payload = _tts_payload(self._settings, text, voice, language)
        try:
            async with self._client.stream(
                "POST",
                f"{base}/audio/speech",
                headers={**_headers(self._settings), "Content-Type": "application/json"},
                json=payload,
                timeout=_timeout(self._settings),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise BackendError(
                        f"TTS backend returned HTTP {resp.status_code}: {_text_message(body)}",
                        code="tts_http_error",
                    )
                response_format = str(payload.get("response_format") or "wav")
                sse = "text/event-stream" in resp.headers.get("content-type", "")
                parser = _audio_decoder(self._settings, response_format, None)
                chunks = _sse_audio(resp) if sse else resp.aiter_bytes()
                async for chunk in chunks:
                    if isinstance(chunk, tuple):
                        # an event stream names its audio's rate: raw PCM is
                        # read at that rate instead of the configured one
                        rate_hint, chunk = chunk
                        if rate_hint and response_format == "pcm" and parser.rate != rate_hint:
                            parser = _audio_decoder(self._settings, response_format, rate_hint)
                    for rate, channels, pcm in parser.feed(chunk):
                        pcm = to_mono(pcm, channels)
                        if rate != 24_000:
                            pcm = resample_pcm16(pcm, rate, 24_000)
                        if pcm:
                            yield pcm
                try:
                    tail = await asyncio.to_thread(parser.flush) if response_format == "mp3" else parser.flush()
                except ValueError as exc:
                    raise BackendError(str(exc), code="tts_bad_audio") from exc
                for rate, channels, pcm in tail:
                    pcm = to_mono(pcm, channels)
                    if rate != 24_000:
                        pcm = resample_pcm16(pcm, rate, 24_000)
                    if pcm:
                        yield pcm
        except httpx.TimeoutException as exc:
            raise BackendError(f"TTS timed out: {exc}", code="tts_timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"TTS request failed: {_redact(str(exc))}", code="tts_unreachable") from exc

    async def probe(self) -> ProbeResult:
        return await _probe(
            self._settings.base_url, _headers(self._settings), self._client, self._settings.model
        )()

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()


def _body_message(resp: httpx.Response) -> str:
    return _text_message(resp.content)


def _text_message(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return "empty body"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return _redact(str(parsed["error"].get("message") or text))[:300]
    except ValueError:
        pass
    return _redact(text)[:300]


def _parse_transcription(resp: httpx.Response) -> str:
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        text_parts: list[str] = []
        for raw_line in resp.text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if event.get("type") == "transcript.text.delta":
                text_parts.append(str(event.get("delta") or ""))
            elif isinstance(event.get("text"), str):
                text_parts.append(event["text"])
        return "".join(text_parts).strip()
    try:
        body = resp.json()
    except ValueError:
        return resp.text.strip()
    if isinstance(body, dict):
        text = body.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


async def _iter_sse(aiter: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    buffer = b""
    done = False
    async for chunk in aiter:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = _split_line(buffer)
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            payload = text[5:].strip()
            if payload == "[DONE]":
                # keep consuming until the server closes the stream so the
                # generator chain unwinds naturally (no orphaned aclose)
                done = True
                continue
            if done:
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def _split_line(buffer: bytes) -> tuple[bytes, bytes]:
    if b"\r\n" in buffer:
        line, rest = buffer.split(b"\r\n", 1)
        return line, rest
    line, rest = buffer.split(b"\n", 1)
    return line.rstrip(b"\r"), rest
