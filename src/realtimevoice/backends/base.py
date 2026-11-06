"""Backend adapters: generic, configurable STT/LLM/TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from ..errors import BackendError


@dataclass
class ProbeResult:
    ok: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)
    # machine-readable failure reason for the readiness/activation probe
    # ("" = none); e.g. "model_not_listed" when /models answers but the
    # configured model id is absent from a non-empty list
    error_code: str = ""


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class TextDelta:
    text: str


@dataclass
class Finish:
    reason: str | None = None


@dataclass
class ReasoningStripped:
    """Metadata-only signal: ``chars`` characters of a reasoning block were
    suppressed before they could reach the transcript/history/TTS. The text
    itself is never carried (and never logged)."""

    chars: int = 0


@dataclass
class ReasoningMeta:
    """Metadata-only signal emitted once at end of stream: how many characters
    of model reasoning the backend reported (``reasoning_content`` /
    ``reasoning`` fields) and how many characters of ``content`` were dropped
    because they merely reproduced that reasoning. Never carries text."""

    reasoning_chars: int = 0
    duplicate_suppressed_chars: int = 0


LLMEvent = TextDelta | ToolCallDelta | Finish | ReasoningStripped | ReasoningMeta


class STTBackend(Protocol):
    async def transcribe(self, wav: bytes, *, language: str | None) -> str: ...

    async def probe(self) -> ProbeResult: ...


class LLMBackend(Protocol):
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]: ...

    async def probe(self) -> ProbeResult: ...


class TTSBackend(Protocol):
    def stream_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def probe(self) -> ProbeResult: ...


@dataclass
class Backends:
    stt: STTBackend | None = None
    llm: LLMBackend | None = None
    tts: TTSBackend | None = None
    owned: bool = False

    async def aclose(self) -> None:
        if not self.owned:
            return
        for backend in (self.stt, self.llm, self.tts):
            if backend is not None and hasattr(backend, "aclose"):
                try:
                    await backend.aclose()
                except Exception:
                    pass


def require(backend: Any, role: str) -> Any:
    if backend is None:
        raise BackendError(f"no {role} backend configured", code=f"{role}_not_configured")
    return backend
