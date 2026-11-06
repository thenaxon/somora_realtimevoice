"""Shared fixtures: fake STT/LLM/TTS backends with controllable gates."""

from __future__ import annotations

import asyncio
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import pytest

from realtimevoice.backends.base import (
    Backends,
    Finish,
    ProbeResult,
    TextDelta,
    ToolCallDelta,
)
from realtimevoice.config import Settings
from realtimevoice.errors import BackendError
from realtimevoice.session import RealtimeSession

SAMPLE_RATE = 24000


# ---------------------------------------------------------------- audio utils

def tone_pcm(duration_ms: int, amplitude: float = 3000.0, freq: float = 440.0) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)),
    )


def silence_pcm(duration_ms: int) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return bytes(2 * n)


# ---------------------------------------------------------------- fake backends

@dataclass
class FakeSTT:
    text: str = "Hallo, wie geht es dir?"
    error: str | None = None
    gate: asyncio.Event | None = None
    calls: list = field(default_factory=list)
    language_seen: str | None = None

    async def transcribe(self, wav: bytes, *, language: str | None) -> str:
        self.calls.append(wav)
        self.language_seen = language
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise BackendError(self.error, code="stt_http_error")
        return self.text

    async def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, detail="fake stt", models=["fake-stt"])


@dataclass
class FakeLLM:
    """script entries: {"text": str, "tool_calls": [{id,name,arguments}], "error": str}"""

    script: list = field(default_factory=list)
    gate: asyncio.Event | None = None
    calls: list = field(default_factory=list)
    default_text: str = "Guten Tag! Wie kann ich Ihnen helfen?"

    async def stream_chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self.gate is not None:
            await self.gate.wait()
        behavior = self.script.pop(0) if self.script else {"text": self.default_text}
        if behavior.get("error"):
            raise BackendError(behavior["error"], code="llm_http_error")
        text = behavior.get("text", "")
        for i in range(0, len(text), 8):
            yield TextDelta(text[i : i + 8])
            await asyncio.sleep(0)
        for tc in behavior.get("tool_calls", []):
            args = tc.get("arguments", "{}")
            half = max(1, len(args) // 2)
            yield ToolCallDelta(index=tc.get("index", 0), id=tc.get("id"), name=tc.get("name"))
            await asyncio.sleep(0)
            yield ToolCallDelta(index=tc.get("index", 0), arguments_delta=args[:half])
            await asyncio.sleep(0)
            yield ToolCallDelta(index=tc.get("index", 0), arguments_delta=args[half:])
            await asyncio.sleep(0)
        yield Finish("stop")

    async def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, detail="fake llm", models=["fake-llm"])


@dataclass
class FakeTTS:
    chunk_ms: int = 100
    chunks: int = 2
    gate: asyncio.Event | None = None
    error: str | None = None
    calls: list = field(default_factory=list)
    inter_chunk_delay: float = 0.0

    async def stream_speech(self, text, *, voice=None, language=None):
        self.calls.append({"text": text, "voice": voice, "language": language})
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise BackendError(self.error, code="tts_http_error")
        n = int(SAMPLE_RATE * self.chunk_ms / 1000)
        for i in range(self.chunks):
            if self.inter_chunk_delay:
                await asyncio.sleep(self.inter_chunk_delay)
            yield struct.pack(f"<{n}h", *(int(2000 * math.sin(2 * math.pi * 440 * j / SAMPLE_RATE)) for j in range(n)))
            await asyncio.sleep(0)

    async def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, detail="fake tts", models=["fake-tts"])


def make_backends(stt=None, llm=None, tts=None) -> Backends:
    return Backends(stt=stt or FakeSTT(), llm=llm or FakeLLM(), tts=tts or FakeTTS())


# ---------------------------------------------------------------- session driver

class Driver:
    def __init__(self, session: RealtimeSession):
        self.session = session
        self.events: list[dict[str, Any]] = []
        self.task = asyncio.get_running_loop().create_task(self._collect())

    async def _collect(self):
        async for event in self.session.events():
            self.events.append(event)

    def of_type(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == etype]

    async def wait_for(self, etype: str, timeout: float = 3.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            found = self.of_type(etype)
            if found:
                return found[-1]
            await asyncio.sleep(0.005)
        raise AssertionError(f"timed out waiting for {etype}; got {[e.get('type') for e in self.events]}")

    async def wait_next(self, etype: str, previous_count: int, timeout: float = 3.0) -> dict[str, Any]:
        """Wait until more than `previous_count` events of this type exist."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            found = self.of_type(etype)
            if len(found) > previous_count:
                return found[-1]
            await asyncio.sleep(0.005)
        raise AssertionError(
            f"timed out waiting for a new {etype} beyond {previous_count}; "
            f"got {[e.get('type') for e in self.events]}"
        )

    def count(self, etype: str) -> int:
        return len(self.of_type(etype))

    def types(self) -> list[str]:
        return [e.get("type") for e in self.events]

    async def stop(self):
        self.session.close()
        self.task.cancel()

    async def append_speech_turn(self, speech_ms=300, silence_ms=600, chunk_ms=100):
        """Push a speech burst followed by silence, chunked like a live mic."""
        total = speech_ms + silence_ms
        for t in range(0, total, chunk_ms):
            if t < speech_ms:
                pcm = tone_pcm(min(chunk_ms, speech_ms - t))
            else:
                pcm = silence_pcm(min(chunk_ms, total - t))
            self.session.handle_event(
                {"type": "input_audio_buffer.append", "audio": _b64(pcm)}
            )
            await asyncio.sleep(0.001)


def _b64(pcm: bytes) -> str:
    import base64

    return base64.b64encode(pcm).decode("ascii")
