"""One realtime conversation: public protocol in, orchestrated backends out.

Design invariants:

- handle_event() never awaits backend work; the WebSocket receive loop stays
  live at all times (microphone audio keeps flowing during STT/LLM/TTS).
- All buffers are bounded; overflow drops stale audio instead of growing.
- Every response carries a generation number; cancellation bumps it once, so
  no stale transcript/audio delta is ever emitted afterwards.
- VAD/audio CPU work runs in a dedicated thread pool, not on the event loop.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import difflib
import hashlib
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .audio import TARGET_SAMPLE_RATE, decode_base64_pcm16, encode_base64_pcm16, frame_ms, pcm16_to_wav
from .backends.base import (
    Backends,
    Finish,
    LLMEvent,
    ReasoningMeta,
    ReasoningStripped,
    TextDelta,
    ToolCallDelta,
)
from .config import Settings, TurnDetectionConfig
from .errors import BackendError, ProtocolError
from .logfilter import sanitize_error_text
from .segment_capture import SegmentCapture
from .symbol_filter import strip_emoji_for_speech
from .vad import (
    BufferCleared,
    BurstDiscarded,
    SpeechStarted,
    SpeechStopped,
    TurnDetector,
    VadEvent,
)

AUDIO_RATE = TARGET_SAMPLE_RATE
BYTES_PER_MS = AUDIO_RATE * 2 / 1000.0
DELTA_CHUNK_MS = 100
# end marker of a sentence's audio fetched ahead
_FETCH_DONE = object()
MAX_APPEND_BYTES = 1 << 20
MAX_MANUAL_BUFFER_MS = 60_000
# H2: bound the unconfirmed-barge-in audio hold; oldest held chunks are dropped
# if a stalled confirm somehow lets it grow past this
MAX_HELD_AUDIO_MS = 10_000.0
# H1: bound the paced audio queue (chunks of ~100 ms each)
PACING_QUEUE_MAX = 512
AUDIO_EVENT_TYPES = frozenset(
    {"response.output_audio.delta", "response.output_audio_transcript.delta"}
)
GA_TOOL_CALL_ITEM = "function_call"

# Lifecycle logs are metadata-only by allowlist: any field name outside this
# set is dropped, so content (transcripts, tool args, audio, tokens) can never
# leak through a careless call site.
LIFECYCLE_FIELDS = frozenset(
    {
        "session_id",
        "action",
        "source",
        "event_id",
        "response_id",
        "item_id",
        "call_id",
        "generation",
        "turn_epoch",
        "turn_epoch_restored",
        "task",
        "out_events_dropped",
        "out_audio_dropped",
        "mic_audio_dropped",
        "reason",
        "status",
        "code",
        "backend",
        "model",
        "stt_model",
        "llm_model",
        "tts_model",
        "tts_profile",
        "session_voice",
        "payload_voice",
        "ref_sha256",
        "route_mode",
        "requested_voice",
        "profile",
        "revision",
        "requested_language",
        "profile_language",
        "duration_ms",
        "audio_ms",
        "out_queue",
        "turn_queue",
        "audio_queue",
        "count",
        "stale",
        "segment_index",
        "seg_start_ms",
        "chars",
        "pcm_bytes",
        "emoji_removed",
        # B1 diagnostics (metadata only): VAD burst stats, response provenance,
        # LLM request shape, similarity ratios and audio timing. All values are
        # numbers, booleans, ids, sha8 prefixes or fixed enum-like strings —
        # never transcript/instruction text.
        "speech_ms",
        "silence_ms",
        "peak_rms",
        "mean_rms",
        "outcome",
        "gap_ms_since_prev_stop",
        "response_cancelled_by_burst",
        "has_instructions",
        "instructions_sha8",
        "new_items_since_last_response",
        "new_items_by_type",
        "last_item_type",
        "client_event_id",
        "ms_since_last_response_done",
        "role_sequence",
        "message_count",
        "tools_count",
        "max_tokens",
        "finish_reason",
        "output_chars",
        "tool_calls_count",
        "reasoning_chars_stripped",
        "reasoning_chars",
        "reasoning_duplicate_suppressed_chars",
        "no_answer",
        "assistant_repeat_ratio",
        "stt_echo_ratio",
        "first_audio_delta_at",
        "last_audio_delta_at",
        "estimated_playback_tail_ms",
        "during_estimated_playback_tail",
        "speaking_before",
        # M2 lifecycle fixes: soft barge-in / output pacing / H4 visibility.
        "barge_in_pending",
        "held_audio_ms",
        "confirm_ms",
        "output_pacing",
        "pacing_dropped_audio_ms",
        "duplicate_request_suspect",
    }
)

# Runtime spoken-output boundary: appended by the server to EVERY LLM prompt
# (after the operator's instructions, before the conversation) — it never
# replaces or rewrites the stored instructions, user content or tools. The
# emoji strip filter at the TTS boundary is independent of LLM compliance.
SPEECH_OUTPUT_DIRECTIVE = (
    "Voice-output boundary (server runtime): speak plain natural language "
    "only — no emojis or emoticons, no markdown formatting (asterisks, "
    "hashes, lists, code fences), no stage directions or roleplay "
    "annotations. This boundary is added by the voice server and does not "
    "replace the operator's instructions."
)

lifecycle_logger = logging.getLogger("realtimevoice.lifecycle")

_MIN_SEGMENT_CHARS = 8
_MAX_SEGMENT_CHARS = 220
_BOUNDARY = re.compile(r"(?<=[.!?…。？！])\s+")
_PAUSE = re.compile(r"[,;:—–]\s+")


# The OpenAI threshold (0..1, default 0.5, lower = reacts to quieter speech)
# onto this server's energy threshold, anchored so that both defaults mean the
# same: 0.5 is RMS 350. Below it 0..0.5 spans RMS 150..350, above it
# 0.5..1 spans 350..1050. Before, 0.5 was RMS 600, so a client asking for a
# MORE sensitive 0.4 got RMS 510, less sensitive than the server's own default.
_RMS_AT_DEFAULT = 350.0


def _speech_classifier(td: TurnDetectionConfig):
    """The speech classifier for a detector, or None (loudness only, by
    choice or because the model is unavailable)."""
    if td.speech_check != "model":
        return None
    from . import speech_model

    if not speech_model.available():
        return None
    return speech_model.SpeechClassifier(AUDIO_RATE)


def ga_threshold_to_rms(threshold: float) -> float:
    t = max(0.0, min(1.0, float(threshold)))
    if t <= 0.5:
        return 150.0 + (_RMS_AT_DEFAULT - 150.0) * (t / 0.5)
    return _RMS_AT_DEFAULT + (1050.0 - _RMS_AT_DEFAULT) * ((t - 0.5) / 0.5)


def rms_to_ga(threshold_rms: float) -> float:
    r = float(threshold_rms)
    if r <= _RMS_AT_DEFAULT:
        t = 0.5 * (r - 150.0) / (_RMS_AT_DEFAULT - 150.0)
    else:
        t = 0.5 + 0.5 * (r - _RMS_AT_DEFAULT) / (1050.0 - _RMS_AT_DEFAULT)
    return max(0.0, min(1.0, t))


@dataclass
class Segment:
    text: str
    start_char: int
    text_len: int = 0


class SentenceBuffer:
    """Splits streamed LLM text into TTS-sized sentences with hard bounds."""

    def __init__(self, min_chars: int = _MIN_SEGMENT_CHARS, max_chars: int = _MAX_SEGMENT_CHARS):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._consumed = 0

    def push(self, delta: str) -> list[Segment]:
        self._buf += delta
        return self._drain(final=False)

    def flush(self) -> list[Segment]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[Segment]:
        segments: list[Segment] = []
        while True:
            match = _BOUNDARY.search(self._buf)
            if match and (match.end() >= self.min_chars or final):
                raw = self._buf[: match.end()]
                # offset of the stripped text within the full LLM stream must
                # be captured BEFORE _advance (it is the base of the
                # truncation milestone mapping)
                start = self._consumed + (len(raw) - len(raw.lstrip()))
                text = raw.strip()
                self._advance(match.end())
                if text:
                    segments.append(Segment(text, start))
                continue
            if final and self._buf.strip():
                raw = self._buf
                start = self._consumed + (len(raw) - len(raw.lstrip()))
                segments.append(Segment(raw.strip(), start))
                self._advance(len(self._buf))
                continue
            if len(self._buf) >= self.max_chars:
                text, rest = self._force_split()
                start = self._consumed + (len(text) - len(text.lstrip()))
                self._advance(len(self._buf) - len(rest))
                if text.strip():
                    segments.append(Segment(text.strip(), start))
                continue
            return segments

    def _force_split(self) -> tuple[str, str]:
        region = self._buf[: self.max_chars]
        best = 0
        for match in _PAUSE.finditer(region):
            best = match.end()
        if best < self.min_chars:
            best = region.rfind(" ")
            if best < self.min_chars // 2:
                best = len(region)
        return self._buf[:best], self._buf[best:]

    def _advance(self, chars: int) -> None:
        self._consumed += len(self._buf[:chars])
        self._buf = self._buf[chars:]


class EventChannel:
    """Bounded, never-blocking outbound queue; audio deltas may be evicted."""

    def __init__(self, capacity: int = 256):
        self._deque: collections.deque[dict[str, Any]] = collections.deque()
        self._capacity = capacity
        self._wake = asyncio.Event()
        self._closed = False
        self.dropped_audio = 0
        self.dropped_events = 0

    def emit(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type", ""))
        if len(self._deque) >= self._capacity:
            if etype in AUDIO_EVENT_TYPES:
                self.dropped_audio += 1
                return
            for idx, queued in enumerate(self._deque):
                if str(queued.get("type")) in AUDIO_EVENT_TYPES:
                    del self._deque[idx]
                    self.dropped_audio += 1
                    break
            else:
                self.dropped_events += 1
                return
        self._deque.append(event)
        self._wake.set()

    def qsize(self) -> int:
        return len(self._deque)

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def recv(self) -> dict[str, Any] | None:
        while True:
            if self._deque:
                return self._deque.popleft()
            if self._closed:
                return None
            await self._wake.wait()
            self._wake.clear()


@dataclass
class ToolCallAcc:
    item_id: str
    call_id: str
    name: str = ""
    arguments: str = ""
    added: bool = False


@dataclass
class ResponseContext:
    id: str
    gen: int
    instructions: str | None
    tts_queue: asyncio.Queue
    item: dict[str, Any]
    output_index: int = 0
    text_parts: list[str] = field(default_factory=list)
    milestones: list[list[float]] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    tool_calls: dict[int, ToolCallAcc] = field(default_factory=dict)
    audio_ms: float = 0.0
    tts_error: str | None = None
    cancelled: bool = False
    cancel_reason: str | None = None
    finished: bool = False
    created_sent: bool = False
    tts_closed: bool = False
    # H2 soft barge-in: while a speech_started burst is unconfirmed, new audio
    # deltas are held here (never emitted) instead of being sent; a confirmed
    # barge-in drops them, a rejected burst flushes them in order.
    barge_in_pending: bool = False
    held_audio: list[bytes] = field(default_factory=list)
    held_audio_ms: float = 0.0
    # H1 realtime output pacing (None/False when pacing is off)
    pacing: bool = False
    pacing_queue: asyncio.Queue | None = None
    pacing_task: asyncio.Task | None = None
    pacing_started_at: float = 0.0
    pacing_delivered_ms: float = 0.0
    # total audio delivered to the client (for cancel truncation under pacing)
    delivered_audio_ms: float = 0.0
    tts_task: asyncio.Task | None = None
    tts_dropped_segments: int = 0
    # held barge-in audio dropped by the MAX_HELD_AUDIO_MS cap: never sent
    held_overflow_ms: float = 0.0
    response_task: asyncio.Task | None = None
    usage: dict[str, Any] | None = None
    reserved_at: float = 0.0
    # audio timing (monotonic, relative to reserved_at), filled by the TTS
    # path; first/last delta arrival times plus the H1 playback-tail estimate
    first_audio_delta_at: float | None = None
    last_audio_delta_at: float | None = None
    # LLM request shape (H5): role sequence etc., recorded at llm_request
    llm_request_shape: dict[str, Any] | None = None
    llm_finish_reason: str | None = None
    # turn attribution frozen at reservation time (the live counter moves on)
    turn_epoch: int = 0
    # test-only per-segment capture (inert unless the env gate is set)
    seg_capture: SegmentCapture | None = None
    # a sentence the TTS loop has taken from the queue to fetch ahead; still
    # "queued" for a cancel's bookkeeping
    tts_ahead: list = field(default_factory=list)


@dataclass
class SttJob:
    item_id: str
    pcm: bytes
    epoch: int = 0


class RealtimeSession:
    def __init__(
        self,
        *,
        session_id: str | None = None,
        backends: Backends,
        settings: Settings,
        model: str = "",
        voice_frozen: bool = False,
        resolved_info: dict[str, Any] | None = None,
        voice_route: dict[str, Any] | None = None,
        route_pending: bool = False,
        route_resolver=None,
        full_backend_factory=None,
        out_capacity: int = 256,
        audio_queue_max: int = 400,
        turn_queue_max: int = 2,
        tts_queue_max: int = 16,
        closed_cb=None,
        warnings: list[str] | None = None,
    ):
        self.id = session_id or f"sess_{uuid.uuid4().hex[:24]}"
        self.backends = backends
        self.model = model
        self.settings = settings
        # voice frozen: a set's session speaks with the set's voice;
        # session.update voice changes are then rejected (non-fatal) while
        # every other field still applies
        self._voice_frozen = voice_frozen
        # voice routing (`voice-route:` model namespace): the requested voice
        # names a set; the set's own voice is what the backend gets. `resolved`
        # marks a pre-resolved route; an auto route resolves at the first
        # voice-carrying session.update (which must precede any input audio /
        # first response) and freezes the voice for the session lifetime:
        # identical updates are no-ops, any different voice afterwards is
        # rejected (no hot swap, no fallback).
        self._voice_route = voice_route
        self._voice_route_resolved = bool(voice_route and voice_route.get("resolved"))
        self._voice_route_frozen_voice = (
            voice_route.get("requested") if self._voice_route_resolved else None
        )
        # full-profile late binding (`voice-route:auto`): while pending, no
        # STT/LLM/TTS backend is bound; the first voice-carrying session.update
        # resolves all three at once via route_resolver + full_backend_factory.
        self._route_pending = bool(route_pending)
        self._route_resolver = route_resolver
        self._full_backend_factory = full_backend_factory
        # a pre-resolved conversation profile route carries its x_realtimevoice
        # metadata block from session.created on
        self._profile_route: dict[str, Any] | None = (
            voice_route if (voice_route and voice_route.get("mode") == "profile") else None
        )
        # CP4: the language a profile-bound session keeps for its whole life.
        # Captured at bind time so a later client session.update can never
        # change it (the client value is only reported as a warning).
        self._profile_language: str = (
            settings.session.language if self._profile_route is not None else ""
        )
        # reported in session.updated x_realtimevoice.warnings
        self._route_warnings: list[str] = list(warnings or [])
        self._resolved_info = resolved_info
        self._cfg: dict[str, Any] = settings.session.model_dump()
        self._out = EventChannel(out_capacity)
        self._audio_in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=audio_queue_max)
        self._turns: asyncio.Queue[SttJob] = asyncio.Queue(maxsize=turn_queue_max)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtv-audio")
        self._loop = asyncio.get_running_loop()
        self._detector = self._make_detector()
        self._conversation: list[dict[str, Any]] = []
        self._items_by_id: dict[str, dict[str, Any]] = {}
        self._gen = 0
        self._active_response: ResponseContext | None = None
        self._response_task: asyncio.Task | None = None
        # turn epoch: bumped on every speech_started; superseded STT results
        # carry an old epoch and must not auto-start a stale response
        self._turn_epoch = 0
        # The epoch before the burst that is open now: a burst the VAD then
        # discards as noise gives it back, so a cough while a sentence is
        # being transcribed does not make that sentence stale.
        self._epoch_before_burst: int | None = None
        # A transcript that arrived while a newer burst was open and was
        # therefore not answered; answered when that burst turns out to be
        # noise or transcribes to nothing.
        self._superseded_unanswered = False
        self._mic_dropped = 0
        # coalesced internal auto-response request (epoch of the requesting
        # turn, or None); flushed once the active response finishes
        self._auto_queued_epoch: int | None = None
        # ---- B1 diagnostics: metadata-only provenance/timing state ----
        # items appended to the conversation since the last response was
        # reserved, keyed by item type (counts only, never content)
        self._items_since_last_response: collections.Counter[str] = collections.Counter()
        self._last_item_type: str = ""
        # transcript-level similarity inputs; SMELL: transcripts live in
        # process memory only long enough to compute a ratio — never logged
        self._last_assistant_transcript: str = ""
        self._last_response_done_at: float | None = None
        # estimated ms of audio the client may still be playing when the server
        # declared response.done (H1 playback tail); None until first response
        self._last_estimated_tail_ms: float | None = None
        # last stopped burst outcome, by turn epoch: lets a later burst know
        # whether IT cancelled a response (barge-in attribution)
        self._burst_cancelled_response: bool = False
        self._manual_buffer = bytearray()
        self._in_ms = 0.0
        self._closed = False
        self._closed_cb = closed_cb
        self._tasks: set[asyncio.Task] = set()
        self._id_counter = 0
        self._current_event_id: str | None = None
        # H2 soft barge-in: a timer that confirms an unconfirmed speech burst
        # after barge_in_confirm_ms while the detector keeps speaking
        self._barge_in_task: asyncio.Task | None = None
        self._ingest_task = self._spawn(self._ingest_loop())
        self._stt_task = self._spawn(self._stt_loop())
        self._emit(self.session_created_event())
        self._log(
            "session_open",
            model=self.model,
            stt_model=self.settings.stt.model,
            llm_model=self.settings.llm.model,
            tts_model=self.settings.tts.model,
        )

    # ------------------------------------------------------------------ ids

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}_{self.id[-8:]}{self._id_counter:06d}"

    def _next_event_id(self) -> str:
        self._id_counter += 1
        return f"event_{self.id[-8:]}{self._id_counter:06d}"

    # ------------------------------------------------------------- lifecycle

    def _spawn(self, coro) -> asyncio.Task:
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._log_task_failure)
        return task

    def _log_task_failure(self, task: asyncio.Task) -> None:
        """A background task that died on an exception said nothing before:
        the session could go deaf (the audio ingest loop) without a trace."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._log("task_failed", level=logging.ERROR, reason=type(exc).__name__, task=task.get_coro().__qualname__)

    def _detector_signature_for(self, td: TurnDetectionConfig) -> tuple[Any, ...]:
        """Identity of the parameters that actually shape the TurnDetector.
        Only these may rebuild it (H7): instruction/voice/tool updates must
        not drop an utterance that is mid-flight."""
        return (td.threshold_rms, td.prefix_padding_ms, td.silence_duration_ms, td.speech_check)

    def _make_detector(self, *, reason: str = "init") -> TurnDetector:
        td = self._turn_detection_cfg()
        prev_signature = getattr(self, "_detector_signature", None)
        new_signature = self._detector_signature_for(td)
        # H7 fix: rebuild ONLY when turn-detection parameters changed. The
        # caller decides whether to emit the vad_reset diagnostic; a rebuild
        # that loses an in-progress utterance is exactly what H7 is about.
        if prev_signature is not None and prev_signature != new_signature:
            speaking_before = bool(getattr(self, "_detector", None) and self._detector.speaking)
            self._log(
                "vad_reset",
                speaking_before=speaking_before,
                reason=reason,
            )
        self._detector_signature = new_signature
        return TurnDetector(
            threshold_rms=td.threshold_rms,
            prefix_padding_ms=td.prefix_padding_ms,
            silence_duration_ms=td.silence_duration_ms,
            classifier=_speech_classifier(td),
        )

    def _turn_detection_cfg(self) -> TurnDetectionConfig:
        return TurnDetectionConfig.model_validate(self._cfg["turn_detection"])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # what was lost on the way, so a choppy call can be told apart from a
        # slow backend: outbound audio/events the client did not take in time,
        # microphone audio the server could not keep up with
        self._log(
            "session_closed",
            turn_queue=self._turns.qsize(),
            audio_queue=self._audio_in.qsize(),
            out_audio_dropped=self._out.dropped_audio or None,
            out_events_dropped=self._out.dropped_events or None,
            mic_audio_dropped=self._mic_dropped or None,
        )
        self.interrupt("session_closed")
        for task in list(self._tasks):
            task.cancel()
        self._out.close()
        self._pool.shutdown(wait=False)
        if self._closed_cb is not None:
            self._closed_cb(self.id)

    async def aclose(self) -> None:
        """close() plus awaiting the cancelled tasks so nothing is destroyed
        while pending (httpx streams, executors, generators)."""
        self.close()
        tasks = [t for t in self._tasks if t is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def closed(self) -> bool:
        return self._closed

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._out.recv()
            if event is None:
                return
            yield event

    def _emit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        self._out.emit({"event_id": self._next_event_id(), **event})

    def _emit_error(self, message: str, *, code: str = "invalid_request_error", fatal: bool = False) -> None:
        self._emit(
            {
                "type": "error",
                "error": {"type": code, "code": code, "message": sanitize_error_text(message), "fatal": fatal},
            }
        )
        self._log("error_emitted", code=code)

    def _log(self, action: str, *, level: int = logging.INFO, **fields: Any) -> None:
        """Structured lifecycle log. Metadata allowlist only: unknown fields
        are dropped, values are bounded primitives — never content."""
        payload: dict[str, Any] = {
            "session_id": self.id,
            "action": action,
            "generation": self._gen,
            "turn_epoch": self._turn_epoch,
        }
        if self._current_event_id:
            payload["event_id"] = self._current_event_id
        for key, value in fields.items():
            if key not in LIFECYCLE_FIELDS or value is None:
                continue
            if isinstance(value, float):
                value = round(value, 1)
            elif not isinstance(value, (int, str, bool)):
                value = str(value)
            if isinstance(value, str) and len(value) > 120:
                value = value[:117] + "..."
            payload[key] = value
        lifecycle_logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _is_stale(self, resp: ResponseContext) -> bool:
        return resp.cancelled or resp.finished or resp.gen != self._gen

    # ------------------------------------------------------------ inbound

    def handle_event(self, event: dict[str, Any]) -> None:
        """Non-blocking dispatch; safe to call from the WS receive loop."""
        if self._closed:
            return
        etype = event.get("type")
        incoming_event_id = event.get("event_id")
        if isinstance(incoming_event_id, str) and incoming_event_id:
            # correlates lifecycle logs (create/cancel/errors) with the
            # client's request; cleared after dispatch so async paths stay clean
            self._current_event_id = incoming_event_id
        try:
            if etype == "session.update":
                self._apply_session_update(event.get("session") or {})
            elif etype == "input_audio_buffer.append":
                self._handle_audio_append(event.get("audio"))
            elif etype == "input_audio_buffer.commit":
                self._handle_commit()
            elif etype == "input_audio_buffer.clear":
                self._handle_buffer_clear()
            elif etype == "conversation.item.create":
                self._handle_item_create(event.get("item") or {})
            elif etype == "conversation.item.truncate":
                self._handle_item_truncate(event)
            elif etype == "conversation.item.delete":
                self._handle_item_delete(event.get("item_id"))
            elif etype == "response.create":
                self._request_response(event.get("response") or {}, source="client")
            elif etype == "response.cancel":
                if self._active_response is None:
                    return
                self._log(
                    "response_cancel_requested",
                    response_id=self._active_response.id,
                    turn_epoch=self._active_response.turn_epoch,
                )
                self.interrupt("client_cancel")
            elif etype == "output_audio_buffer.clear":
                self._emit({"type": "output_audio_buffer.cleared"})
            elif etype == "close":
                return
            else:
                self._emit_error(f"unknown event type: {etype!r}", code="unknown_event_type")
        except ProtocolError as exc:
            self._emit_error(exc.message, code=exc.code, fatal=exc.fatal)
        except Exception as exc:  # noqa: BLE001 - a bad event must not end the connection
            # A malformed event the checks above did not foresee: the client
            # is told, the session goes on. Never the raw exception text.
            self._log("event_failed", level=logging.WARNING, reason=type(exc).__name__)
            self._emit_error(f"could not handle {etype!r}: {type(exc).__name__}", code="invalid_event")
        finally:
            self._current_event_id = None

    def _handle_audio_append(self, audio: Any) -> None:
        if not isinstance(audio, str) or not audio:
            raise ProtocolError("input_audio_buffer.append requires base64 'audio' string")
        if self._voice_route_pending():
            return  # fail closed: unrouted input audio is discarded (no STT)
        try:
            pcm = decode_base64_pcm16(audio)
        except ValueError as exc:
            raise ProtocolError(str(exc), code="invalid_audio_data") from exc
        if len(pcm) > MAX_APPEND_BYTES:
            raise ProtocolError("audio chunk too large", code="invalid_audio_data")
        try:
            self._audio_in.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._audio_in.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._audio_in.put_nowait(pcm)
            self._mic_dropped += 1
            if self._mic_dropped == 1 or self._mic_dropped % 100 == 0:
                # the oldest microphone audio gave way: the server is behind
                self._log("mic_audio_dropped", level=logging.WARNING, count=self._mic_dropped)
        td = self._turn_detection_cfg()
        if td.type == "none":
            self._manual_buffer.extend(pcm)
            max_bytes = int(MAX_MANUAL_BUFFER_MS * BYTES_PER_MS)
            if len(self._manual_buffer) > max_bytes:
                del self._manual_buffer[: len(self._manual_buffer) - max_bytes]

    def _handle_commit(self) -> None:
        if self._voice_route_pending():
            return  # fail closed: unrouted input audio is discarded (no STT)
        td = self._turn_detection_cfg()
        if td.type == "server_vad":
            self._emit_error(
                "input_audio_buffer.commit is not allowed while server VAD is active",
                code="buffer_commit_not_allowed",
            )
            return
        pcm = bytes(self._manual_buffer)
        self._manual_buffer.clear()
        if len(pcm) < 2 * AUDIO_RATE * 1 // 100:
            self._emit_error("input audio buffer is empty", code="input_audio_buffer_commit_empty")
            return
        item = self._create_user_audio_item()
        # in the conversation, so its transcript reaches the model
        self._register_item(item)
        self._emit(
            {
                "type": "conversation.item.created",
                "item": _public_item(item),
                "previous_item_id": self._conversation[-2]["id"] if len(self._conversation) > 1 else None,
            }
        )
        self._queue_stt_job(item["id"], pcm)

    def _handle_buffer_clear(self) -> None:
        # on the audio thread, after any process() already queued there: the
        # detector is never cleared while it is working on a frame
        self._pool.submit(self._detector.clear)
        self._manual_buffer.clear()
        while True:
            try:
                self._audio_in.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._emit({"type": "input_audio_buffer.cleared"})

    def _handle_item_create(self, raw_item: dict[str, Any]) -> None:
        item, error = self._validate_client_item(raw_item)
        if error:
            self._emit_error(error)
            return
        if item["type"] == "function_call_output":
            # metadata only: call_id, never the output content
            self._log("tool_result_received", call_id=item["call_id"], item_id=item["id"])
        self._register_item(item)
        self._emit(
            {
                "type": "conversation.item.created",
                "item": _public_item(item),
                "previous_item_id": self._conversation[-2]["id"] if len(self._conversation) > 1 else None,
            }
        )

    def _validate_client_item(self, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(raw, dict):
            return None, "conversation.item.create requires an 'item' object"
        itype = raw.get("type", "message")
        if raw.get("id") is not None and not isinstance(raw.get("id"), str):
            return None, "item 'id' must be a string"
        item_id = raw.get("id") or self._next_id("item")
        if itype == "function_call_output":
            call_id = raw.get("call_id")
            output = raw.get("output", "")
            if not call_id or not isinstance(call_id, str):
                return None, "function_call_output requires a call_id"
            if not isinstance(output, str):
                return None, "function_call_output 'output' must be a string"
            item = {
                "id": item_id,
                "type": "function_call_output",
                "status": "completed",
                "call_id": call_id,
                "output": output,
            }
            return item, None
        if itype == "function_call":
            if not raw.get("call_id") or not raw.get("name"):
                return None, "function_call item requires call_id and name"
            item = {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": raw["call_id"],
                "name": raw["name"],
                "arguments": raw.get("arguments", "{}"),
            }
            return item, None
        if itype != "message":
            return None, f"unsupported item type: {itype!r}"
        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system"):
            return None, f"invalid message role: {role!r}"
        content = raw.get("content", [])
        if not isinstance(content, list):
            return None, "message 'content' must be a list"
        normalized: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                return None, "message content parts must be objects"
            ptype = part.get("type")
            if ptype in ("input_text", "text", "output_text"):
                normalized.append({"type": ptype, "text": str(part.get("text", ""))})
            elif ptype == "input_audio":
                normalized.append({"type": "input_audio", "transcript": str(part.get("transcript", ""))})
            else:
                return None, f"unsupported content part type: {ptype!r}"
        return {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": role,
            "content": normalized,
        }, None

    def _handle_item_truncate(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        try:
            content_index = int(event.get("content_index", 0))
            audio_end_ms = float(event.get("audio_end_ms", 0))
        except (TypeError, ValueError):
            self._emit_error("content_index and audio_end_ms must be numbers", code="invalid_value")
            return
        if content_index < 0 or audio_end_ms < 0:
            self._emit_error("content_index and audio_end_ms must not be negative", code="invalid_value")
            return
        item = self._items_by_id.get(item_id) if isinstance(item_id, str) else None
        if item is None or item.get("type") != "message" or item.get("role") != "assistant":
            self._emit_error(f"item {item_id!r} not found or not truncatable", code="item_not_found")
            return
        cut_chars = self._cut_chars_for_audio(item, audio_end_ms)
        part = item["content"][content_index] if content_index < len(item["content"]) else None
        if part is None:
            self._emit_error(f"content_index {content_index} out of range", code="invalid_content_index")
            return
        if "transcript" not in part:
            self._emit_error(f"content_index {content_index} holds no audio transcript", code="invalid_content_index")
            return
        part["transcript"] = part["transcript"][:cut_chars]
        item["truncated_audio_ms"] = audio_end_ms
        self._emit(
            {
                "type": "conversation.item.truncated",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": int(audio_end_ms),
            }
        )

    def _handle_item_delete(self, item_id: Any) -> None:
        item = self._items_by_id.get(item_id) if isinstance(item_id, str) else None
        if item is None:
            self._emit_error(f"item {item_id!r} not found", code="item_not_found")
            return
        self._conversation = [i for i in self._conversation if i["id"] != item_id]
        self._items_by_id.pop(item_id, None)
        self._emit({"type": "conversation.item.deleted", "item_id": item_id})

    # ------------------------------------------------------- session.update

    def _apply_profile_language(self, update: dict[str, Any]) -> None:
        """CP4: enforce the profile language on every session.update of a
        profile-bound session (already resolved `voice-route`, `profile:<id>`,
        `profile-test:<ticket>`). A different client language is removed from
        the update and reported once as `language_overridden_by_profile` — the
        session keeps the profile language; identical/no language is a no-op.
        Legacy sessions without a profile are untouched."""
        if self._profile_route is None:
            return
        client_language = update.get("language")
        profile_language = self._profile_language
        if not client_language or not profile_language:
            return
        if str(client_language) == str(profile_language):
            update.pop("language", None)
            return
        update.pop("language", None)
        if "language_overridden_by_profile" not in self._route_warnings:
            self._route_warnings.append("language_overridden_by_profile")
        self._log(
            "language_overridden_by_profile",
            requested_language=str(client_language),
            profile_language=str(profile_language),
        )

    def _route_voice_update(self, update: dict[str, Any]) -> str | None:
        """Voice-routing hook for session.update. Returns a violation code
        for a non-fatal error, or None. A routed session freezes the ACTUAL
        voice after its first resolution: an identical requested voice is a
        no-op, ANY different voice afterwards is rejected."""
        if self._route_pending:
            if "voice" not in update:
                return None
            requested = update["voice"]
            return self._resolve_pending_route(requested, update)
        if self._voice_route is None or "voice" not in update:
            return None
        requested = update["voice"]
        if self._voice_route_resolved:
            if requested == self._voice_route_frozen_voice:
                # identical no-op: REMOVE the voice from the update so the
                # stored ACTUAL voice (profile.voice) is never overwritten
                # by the client alias
                del update["voice"]
                return None
            del update["voice"]
            self._log(
                "voice_route_frozen",
                requested_voice=requested,
                frozen_voice=self._voice_route_frozen_voice,
                tts_profile=(self._voice_route or {}).get("profile_id", ""),
            )
            return "voice_route_frozen"
        return None

    def _resolve_pending_route(self, requested: str, update: dict[str, Any]) -> str | None:
        """First voice-carrying update on a pending full-profile route:
        resolve all three backends + voice in ONE step, apply profile
        language/defaults, emit x_realtimevoice.route + warnings and the
        route_resolved lifecycle log. Fail closed on unknown routes (route
        stays pending)."""
        if self._route_resolver is None:
            del update["voice"]
            self._emit_error("voice routing is not available", code="unknown_voice_route")
            return None
        try:
            selection, route = self._route_resolver(requested)
        except Exception as exc:
            code = getattr(exc, "code", "unknown_voice_route")
            del update["voice"]
            self._log("voice_route_unknown", requested_voice=requested, code=code)
            self._emit_error(str(exc), code=code)
            return None
        # one-step bind: settings + all three backends from the resolved snapshot
        self.settings = selection.settings
        if self._full_backend_factory is not None:
            self.backends = self._full_backend_factory(selection.settings)
        self._cfg["voice"] = selection.settings.session.voice or self._cfg["voice"]
        self._cfg["language"] = selection.settings.session.language
        self._cfg["instructions"] = selection.settings.session.instructions
        # the profile's pacing and turn-detection defaults, as a profile:<id>
        # session gets them at connect; the client's own turn-detection values
        # in this update are merged on top afterwards
        self._cfg["output_pacing"] = selection.settings.session.output_pacing
        self._cfg["turn_detection"] = selection.settings.session.turn_detection.model_dump()
        profile_language = selection.settings.session.language
        self._profile_route = route
        self._profile_language = profile_language or ""
        self._route_pending = False
        # keep the frozen-voice machinery for later updates: identical voice
        # is a no-op, any different voice is rejected (voice_route_frozen)
        resolved_route = dict(route)
        resolved_route["resolved"] = True
        resolved_route["profile_id"] = route.get("profile", "")
        resolved_route["actual_voice"] = selection.settings.session.voice
        self._voice_route = resolved_route
        self._voice_route_resolved = True
        self._voice_route_frozen_voice = requested
        # profile language wins; a different client language is a warning only
        warnings: list[str] = []
        client_language = update.get("language")
        if client_language and profile_language and str(client_language) != str(profile_language):
            warnings.append("language_overridden_by_profile")
            self._log(
                "language_overridden_by_profile",
                requested_language=str(client_language),
                profile_language=str(profile_language),
            )
            update.pop("language", None)
        self._route_warnings = warnings
        self._resolved_info = selection.info()
        route_modes = dict(route)
        self._log(
            "backend_bound",
            route_mode=route_modes.get("mode", "profile"),
            stt_model=selection.settings.stt.model,
            llm_model=selection.settings.llm.model,
            tts_model=selection.settings.tts.model,
        )
        self._log(
            "route_resolved",
            route_mode=route_modes.get("mode", "profile"),
            requested_voice=requested,
            profile=route_modes.get("profile", ""),
            revision=route_modes.get("revision", 0),
            stt_model=route_modes.get("stt_model", ""),
            llm_model=route_modes.get("llm_model", ""),
            tts_model=route_modes.get("tts_model", ""),
            session_voice=selection.settings.session.voice or "",
            language=profile_language or "",
            ref_sha256=(route_modes.get("voice") or {}).get("ref_sha256") or None,
        )
        # never let the alias land in cfg.voice
        del update["voice"]
        return None

    def _voice_route_pending(self) -> bool:
        """True while a voice-route session has not resolved its route yet.
        Emits the pending error ONCE (per session) — input audio is then
        dropped (no STT, no TTS): fail closed, never a fallback voice."""
        if self._route_pending:
            if not getattr(self, "_profile_route_pending_warned", False):
                self._profile_route_pending_warned = True
                self._emit_error(
                    "voice routing is pending: the initial session.update with "
                    "audio.output.voice must arrive before any input audio or "
                    "response; input audio is discarded and no fallback voice "
                    "is used",
                    code="voice_route_pending",
                )
            return True
        if self._voice_route is None or self._voice_route_resolved:
            return False
        if not getattr(self, "_voice_route_pending_warned", False):
            self._voice_route_pending_warned = True
            self._emit_error(
                "voice routing is pending: the initial session.update with "
                "audio.output.voice must arrive before any input audio or "
                "response; input audio is discarded and no fallback voice "
                "is used",
                code="voice_route_pending",
            )
        return True

    def _apply_session_update(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ProtocolError("session.update requires a 'session' object")
        update: dict[str, Any] = {}
        errors: list[str] = []
        voice_frozen_violation = False
        if "model" in payload and payload["model"] != self.model:
            errors.append("model cannot be changed after session creation")
        if "instructions" in payload:
            if not isinstance(payload["instructions"], str):
                errors.append("instructions must be a string")
            else:
                update["instructions"] = payload["instructions"]
        if "voice" in payload:
            if not isinstance(payload["voice"], str):
                errors.append("voice must be a string")
            else:
                update["voice"] = payload["voice"]
        if "language" in payload:
            if not isinstance(payload["language"], str):
                errors.append("language must be a string")
            else:
                update["language"] = payload["language"]
        if "modalities" in payload:
            modalities = payload["modalities"]
            if not isinstance(modalities, list) or "audio" not in modalities:
                errors.append("modalities must include 'audio'; text-only output is not supported")
        if "tools" in payload:
            tools, tool_error = self._validate_tools(payload["tools"])
            if tool_error:
                errors.append(tool_error)
            else:
                update["tools"] = tools
        audio = payload.get("audio")
        if isinstance(audio, dict):
            audio_update = self._translate_audio(audio, errors)
            if audio_update:
                update.update(audio_update)
        if "input_audio_transcription" in payload:
            iat = payload["input_audio_transcription"]
            if isinstance(iat, dict) and "language" in iat:
                update.setdefault("language", str(iat["language"]))
        # additive extension key `x_realtimevoice.output_pacing`: the only
        # vendor extension the server accepts on session.update (H1 opt-in)
        ext = payload.get("x_realtimevoice")
        if isinstance(ext, dict) and "output_pacing" in ext:
            pacing = ext["output_pacing"]
            if pacing not in ("off", "realtime"):
                errors.append("x_realtimevoice.output_pacing must be 'off' or 'realtime'")
            else:
                update["output_pacing"] = pacing
        for td_key in ("turn_detection",):
            if td_key in payload:
                td_update = self._translate_turn_detection(payload[td_key], errors)
                if td_update is not None:
                    # only what the client set: merged into the session (or
                    # into a profile resolved in this same update) below
                    update["turn_detection"] = td_update
        if errors:
            self._emit_error("; ".join(errors))
            return
        if self._voice_frozen and "voice" in update:
            # the set pinned the voice for this session; the rest of the
            # update still applies (partial-update semantics preserved). The
            # name the call was routed by, or the set's own voice, is a no-op.
            requested = update.pop("voice")
            voice_frozen_violation = requested not in (self._voice_route_frozen_voice, self._cfg.get("voice"))
        route_violation = self._route_voice_update(update)
        # CP4: a profile-bound session keeps the profile language for its whole
        # life. A different client language (top-level or transcription) is
        # ignored for the session and reported as a warning, never an error.
        self._apply_profile_language(update)
        self._cfg = _deep_merge(self._cfg, update)
        # H7: only a turn-detection parameter change may rebuild the detector;
        # instructions/voice/tools updates must NOT reset an utterance that is
        # in progress (the detector keeps speaking=True and its buffers)
        new_signature = self._detector_signature_for(self._turn_detection_cfg())
        if new_signature != self._detector_signature:
            self._detector = self._make_detector(reason="session_update")
        self._emit({"type": "session.updated", "session": self.public_session()})
        if voice_frozen_violation:
            self._emit_error(
                "the voice is the set's for this call; the change is ignored "
                "(choose another set for another voice)",
                code="voice_frozen",
            )
        if route_violation == "voice_route_frozen":
            self._emit_error(
                "voice is frozen by voice routing for this session "
                f"(resolved voice: {self._voice_route_frozen_voice!r}); "
                "open a new connection for a different voice",
                code="voice_route_frozen",
            )

    def _translate_audio(self, audio: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        update: dict[str, Any] = {}
        ai = audio.get("input") or {}
        if isinstance(ai, dict):
            fmt = ai.get("format")
            if fmt is not None and not _is_pcm16_24k(fmt):
                errors.append("only audio/pcm at 24000 Hz is supported for input")
            trans = ai.get("transcription")
            if isinstance(trans, dict) and "language" in trans:
                update["language"] = str(trans["language"])
            if "turn_detection" in ai:
                td_update = self._translate_turn_detection(ai["turn_detection"], errors)
                if td_update is not None:
                    # only what the client set: merged into the session (or
                    # into a profile resolved in this same update) below
                    update["turn_detection"] = td_update
        ao = audio.get("output") or {}
        if isinstance(ao, dict):
            fmt = ao.get("format")
            if fmt is not None and not _is_pcm16_24k(fmt):
                errors.append("only audio/pcm at 24000 Hz is supported for output")
            if "voice" in ao:
                update["voice"] = str(ao["voice"])
            if "speed" in ao:
                errors.append("audio.output.speed is not supported")
        return update

    def _translate_turn_detection(self, td: Any, errors: list[str]) -> dict[str, Any] | None:
        if td is None:
            return {"type": "none"}
        if not isinstance(td, dict):
            errors.append("turn_detection must be an object or null")
            return None
        update: dict[str, Any] = {}
        if "type" in td:
            if td["type"] not in ("server_vad", "none"):
                errors.append("turn_detection.type must be 'server_vad' or 'none'")
            else:
                update["type"] = td["type"]
        if "threshold" in td and isinstance(td["threshold"], (int, float)):
            if not 0.0 <= float(td["threshold"]) <= 1.0:
                errors.append("turn_detection.threshold must be between 0 and 1")
            else:
                update["threshold"] = float(td["threshold"])
                update["threshold_rms"] = ga_threshold_to_rms(td["threshold"])
        if "threshold_rms" in td and isinstance(td["threshold_rms"], (int, float)):
            update["threshold_rms"] = float(td["threshold_rms"])
        for key in ("prefix_padding_ms", "silence_duration_ms", "barge_in_confirm_ms"):
            if key in td:
                value = td[key]
                if not isinstance(value, int) or value < 0:
                    errors.append(f"turn_detection.{key} must be a non-negative integer")
                else:
                    update[key] = value
        for key in ("interrupt_response", "create_response"):
            if key in td:
                update[key] = bool(td[key])
        if "speech_check" in td:
            if td["speech_check"] in ("model", "loudness"):
                update["speech_check"] = td["speech_check"]
            else:
                errors.append("turn_detection.speech_check must be 'model' or 'loudness'")
        return update

    def _validate_tools(self, tools: Any) -> tuple[list[dict[str, Any]], str | None]:
        if not isinstance(tools, list):
            return [], "tools must be a list"
        out: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                return [], "each tool must be an object"
            ttype = tool.get("type", "function")
            if ttype != "function":
                return [], f"unsupported tool type: {ttype!r}"
            name = tool.get("name")
            if not name or not isinstance(name, str):
                return [], "function tool requires a name"
            out.append(
                {
                    "type": "function",
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return out, None

    def public_session(self) -> dict[str, Any]:
        td = self._turn_detection_cfg()
        ga_td = {
            "type": td.type,
            "threshold": round(rms_to_ga(td.threshold_rms), 4),
            "prefix_padding_ms": td.prefix_padding_ms,
            "silence_duration_ms": td.silence_duration_ms,
            "interrupt_response": td.interrupt_response,
            "create_response": td.create_response,
            "barge_in_confirm_ms": td.barge_in_confirm_ms,
            "speech_check": td.speech_check,
        }
        session: dict[str, Any] = {
            "type": "session",
            "id": self.id,
            "model": self.model,
            "instructions": self._cfg["instructions"],
            "modalities": ["audio", "text"],
            "voice": self._cfg["voice"],
            "language": self._cfg["language"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": self.settings.stt.model, "language": self._cfg["language"]},
            "turn_detection": ga_td,
            "tools": self._cfg["tools"],
            "tool_choice": "auto",
            "temperature": 0.8,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": AUDIO_RATE},
                    "transcription": {"model": self.settings.stt.model, "language": self._cfg["language"]},
                    "turn_detection": ga_td,
                },
                "output": {
                    "voice": self._cfg["voice"],
                    "format": {"type": "audio/pcm", "rate": AUDIO_RATE},
                },
            },
        }
        # additive, vendor-neutral extension: safe runtime metadata for the
        # bundled test page; unknown clients simply ignore it (no secrets)
        if self._resolved_info is not None:
            session["realtimevoice"] = self._resolved_info
        ext_block: dict[str, Any] = {
            "output_pacing": self._cfg.get("output_pacing", "off"),
        }
        if self._profile_route is not None or getattr(self, "_route_warnings", None):
            ext_block["route"] = self._profile_route or {}
            ext_block["warnings"] = list(getattr(self, "_route_warnings", []) or [])
        session["x_realtimevoice"] = ext_block
        return session

    def session_created_event(self) -> dict[str, Any]:
        return {"event_id": self._next_event_id(), "type": "session.created", "session": self.public_session()}

    # ------------------------------------------------------------- audio in

    async def _ingest_loop(self) -> None:
        while True:
            pcm = await self._audio_in.get()
            td = self._turn_detection_cfg()
            if td.type != "server_vad":
                continue
            self._in_ms += frame_ms(len(pcm), AUDIO_RATE)
            events: list[VadEvent] = await self._loop.run_in_executor(self._pool, self._detector.process, pcm)
            for event in events:
                if isinstance(event, SpeechStarted):
                    self._on_speech_started(event.stats)
                elif isinstance(event, SpeechStopped):
                    self._on_speech_stopped(event.utterance, event.stats)
                elif isinstance(event, BurstDiscarded):
                    self._on_burst_discarded(event.stats)
                elif isinstance(event, BufferCleared):
                    self._emit({"type": "input_audio_buffer.cleared"})

    def _near_playback_tail(self) -> tuple[bool, float | None]:
        """True when now is still inside the estimated playback tail of the
        last response (server finished streaming before the client could have
        finished playing it). Returns (during, ms_since_done)."""
        last_done = self._last_response_done_at
        if last_done is None:
            return False, None
        ms_since = (self._loop.time() - last_done) * 1000.0
        tail = self._last_estimated_tail_ms
        return (tail is not None and ms_since < tail), ms_since

    def _on_speech_started(self, stats=None) -> None:
        # every new speech burst opens a new turn: superseded STT results
        # from earlier turns may no longer auto-start a response
        self._epoch_before_burst = self._turn_epoch
        self._turn_epoch += 1
        during_tail, ms_since_done = self._near_playback_tail()
        self._log(
            "speech_started",
            audio_ms=int(self._in_ms),
            during_estimated_playback_tail=during_tail,
            ms_since_last_response_done=ms_since_done,
        )
        self._emit(
            {
                "type": "input_audio_buffer.speech_started",
                "audio_start_ms": int(self._in_ms),
            }
        )
        td = self._turn_detection_cfg()
        self._burst_cancelled_response = False
        if self._active_response is not None and td.interrupt_response:
            if td.barge_in_confirm_ms > 0:
                self._begin_barge_in_pending(self._active_response, td.barge_in_confirm_ms)
            else:
                self.interrupt("barge_in")
                self._burst_cancelled_response = True

    def _begin_barge_in_pending(self, resp: ResponseContext, confirm_ms: int) -> None:
        """H2: a speech_started arrived during an active response, but the burst
        is not yet confirmed. Hold new audio instead of cancelling immediately;
        the confirm timer / speech_stopped / discarded-burst paths resolve it."""
        if resp.barge_in_pending:
            # a second start on the same response: keep the ORIGINAL window
            return
        resp.barge_in_pending = True
        self._log(
            "barge_in_pending",
            response_id=resp.id,
            turn_epoch=resp.turn_epoch,
            confirm_ms=confirm_ms,
        )
        self._barge_in_task = self._spawn(self._confirm_barge_in_after(resp, confirm_ms))

    async def _confirm_barge_in_after(self, resp: ResponseContext, confirm_ms: int) -> None:
        await asyncio.sleep(confirm_ms / 1000.0)
        if not resp.barge_in_pending or self._is_stale(resp):
            return
        if self._detector.speaking and self._detector.voiced_ms >= self._detector.min_speech_ms:
            # someone has been speaking (not only a click whose burst is still
            # open in its trailing silence): a real barge-in
            self._confirm_barge_in(resp)
        elif self._detector.speaking:
            # the burst is still open but almost nothing was voiced yet: decide
            # when it ends — kept speech confirms, a discarded burst rejects
            self._barge_in_task = self._spawn(self._confirm_barge_in_after(resp, 50))
        else:
            # speech ended without reaching the confirm window and was not
            # discarded (e.g. a kept short utterance after speech_stopped):
            # resolve as rejected so no turn is lost
            self._reject_barge_in(resp, speech_ms=None)

    def _confirm_barge_in(self, resp: ResponseContext) -> None:
        if not resp.barge_in_pending or self._is_stale(resp):
            return
        resp.barge_in_pending = False
        self._cancel_barge_in_task()
        self.interrupt("barge_in")
        self._burst_cancelled_response = True

    def _reject_barge_in(self, resp: ResponseContext, *, speech_ms: float | None) -> None:
        """The pending burst did not confirm: resume the response by flushing
        the held audio in order and logging the rejection (H2)."""
        if not resp.barge_in_pending:
            return
        resp.barge_in_pending = False
        self._cancel_barge_in_task()
        held = resp.held_audio
        resp.held_audio = []
        held_ms = resp.held_audio_ms
        resp.held_audio_ms = 0.0
        self._log(
            "barge_in_rejected",
            response_id=resp.id,
            turn_epoch=resp.turn_epoch,
            speech_ms=speech_ms,
            held_audio_ms=held_ms,
        )
        for chunk in held:
            if self._is_stale(resp):
                return
            if resp.pacing:
                self._enqueue_paced_audio(resp, chunk, frame_ms(len(chunk), AUDIO_RATE))
            else:
                self._deliver_audio_delta(resp, chunk)

    async def _settle_barge_in(self, resp: ResponseContext) -> None:
        """Wait for a pending barge-in on this response to be decided (its
        confirm window), then reject it if nothing decided it."""
        task = self._barge_in_task
        if resp.barge_in_pending and task is not None and not task.done():
            await asyncio.wait({task})
        if resp.barge_in_pending:
            self._reject_barge_in(resp, speech_ms=None)

    def _cancel_barge_in_task(self) -> None:
        task = self._barge_in_task
        self._barge_in_task = None
        if task is not None and not task.done():
            task.cancel()

    def _on_speech_stopped(self, utterance: bytes, stats=None) -> None:
        # the burst was speech: its turn stays the current one
        self._epoch_before_burst = None
        td = self._turn_detection_cfg()
        if td.type != "server_vad":
            return
        if self._active_response is not None and self._active_response.barge_in_pending:
            speech_ms = stats.speech_ms if stats is not None else None
            if speech_ms is not None and speech_ms >= td.barge_in_confirm_ms:
                self._confirm_barge_in(self._active_response)
            else:
                self._reject_barge_in(self._active_response, speech_ms=speech_ms)
        self._log(
            "speech_stopped",
            audio_ms=int(self._in_ms),
            **self._burst_fields(stats),
        )
        self._log("vad_burst", **self._burst_fields(stats))
        item = self._create_user_audio_item()
        self._register_item(item)
        self._emit(
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": item["id"],
                "audio_end_ms": int(self._in_ms),
            }
        )
        self._emit(
            {
                "type": "conversation.item.created",
                "item": _public_item(item),
                "previous_item_id": self._conversation[-2]["id"] if len(self._conversation) > 1 else None,
            }
        )
        if utterance:
            self._queue_stt_job(item["id"], utterance, epoch=self._turn_epoch)

    @staticmethod
    def _burst_fields(stats) -> dict[str, Any]:
        """Map VAD BurstStats to allowlisted metadata fields (numbers only)."""
        if stats is None:
            return {}
        fields: dict[str, Any] = {
            "speech_ms": stats.speech_ms,
            "silence_ms": stats.silence_ms,
            "peak_rms": stats.peak_rms,
            "mean_rms": stats.mean_rms,
            "outcome": stats.outcome,
        }
        gap = getattr(stats, "gap_ms_since_prev_stop", -1.0)
        if gap is not None and gap >= 0:
            fields["gap_ms_since_prev_stop"] = gap
        return fields

    def _register_item(self, item: dict[str, Any]) -> None:
        """Append an item to the live conversation and record its type for the
        next response's provenance (counts only; never content)."""
        self._conversation.append(item)
        self._items_by_id[item["id"]] = item
        itype = str(item.get("type", "unknown"))
        if itype == "message":
            itype = f"message:{item.get('role', '')}"
        self._items_since_last_response[itype] += 1
        self._last_item_type = itype

    def _on_burst_discarded(self, stats) -> None:
        """A speech_started burst that never produced a kept utterance (too
        short): log it with the response impact. With soft barge-in (H2) an
        unconfirmed burst is REJECTED here — the response resumes with its held
        audio, so no response is killed by noise. Without the option the burst
        already cancelled the response and that is recorded."""
        if self._active_response is not None and self._active_response.barge_in_pending:
            speech_ms = stats.speech_ms if stats is not None else None
            self._reject_barge_in(self._active_response, speech_ms=speech_ms)
        # Noise is not a turn: the turn before it is current again, so its
        # transcript (still in STT, or already back) gets its answer.
        restored = self._epoch_before_burst is not None
        if restored:
            self._turn_epoch = self._epoch_before_burst
            self._epoch_before_burst = None
        self._log(
            "vad_burst_discarded",
            response_cancelled_by_burst=self._burst_cancelled_response,
            turn_epoch_restored=restored,
            **self._burst_fields(stats),
        )
        cut_by_noise = self._burst_cancelled_response
        self._burst_cancelled_response = False
        if restored and (self._superseded_unanswered or cut_by_noise):
            # the answer was cancelled by what turned out to be noise (or a
            # transcript waited for it): the question gets its answer again
            self._answer_superseded("answer_cut_by_noise" if cut_by_noise else "burst_discarded")

    def _create_user_audio_item(self) -> dict[str, Any]:
        return {
            "id": self._next_id("item"),
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": "", "audio": None}],
        }

    def _queue_stt_job(self, item_id: str, pcm: bytes, *, epoch: int | None = None) -> None:
        job = SttJob(item_id=item_id, pcm=pcm, epoch=self._turn_epoch if epoch is None else epoch)
        try:
            self._turns.put_nowait(job)
        except asyncio.QueueFull:
            dropped: SttJob | None = None
            try:
                dropped = self._turns.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._turns.put_nowait(job)
            except asyncio.QueueFull:
                self._log("turn_queue_overflow", turn_queue=self._turns.qsize(), item_id=item_id)
                self._emit_error(
                    "turn dropped: STT pipeline is saturated", code="turn_queue_overflow"
                )
                return
            if dropped is not None:
                self._log(
                    "turn_dropped",
                    turn_queue=self._turns.qsize(),
                    item_id=dropped.item_id,
                    count=1,
                )
                self._emit_error(
                    "queued turn dropped to keep the pipeline bounded", code="turn_queue_overflow"
                )

    async def _stt_loop(self) -> None:
        if self.backends.stt is None and not self._route_pending:
            self._emit_error("no STT backend configured", code="stt_not_configured")
            return
        while True:
            job = await self._turns.get()
            # read the STT backend at JOB time: a pending full-profile route
            # may bind it only after the route resolves (no early capture)
            stt = self.backends.stt
            if stt is None:
                self._emit_error("no STT backend configured", code="stt_not_configured")
                continue
            started = self._loop.time()
            self._log(
                "stt_started",
                item_id=job.item_id,
                turn_epoch=job.epoch,
                backend="stt",
                model=self.settings.stt.model,
            )
            try:
                wav = await self._loop.run_in_executor(self._pool, pcm16_to_wav, job.pcm)
                language = self._cfg["language"] or None
                text = await asyncio.wait_for(
                    stt.transcribe(wav, language=language),
                    timeout=self.settings.stt.request_timeout_s,
                )
            except asyncio.TimeoutError:
                self._log(
                    "stt_timeout",
                    item_id=job.item_id,
                    turn_epoch=job.epoch,
                    backend="stt",
                    model=self.settings.stt.model,
                    duration_ms=(self._loop.time() - started) * 1000,
                )
                self._emit_error("STT timed out", code="stt_timeout")
                continue
            except BackendError as exc:
                self._log(
                    "stt_failed",
                    item_id=job.item_id,
                    turn_epoch=job.epoch,
                    backend="stt",
                    model=self.settings.stt.model,
                    code=exc.code,
                )
                self._emit_error(f"STT failed: {exc.message}", code=exc.code)
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(
                    "stt_failed",
                    item_id=job.item_id,
                    turn_epoch=job.epoch,
                    backend="stt",
                    model=self.settings.stt.model,
                    code="stt_error",
                    reason=type(exc).__name__,
                )
                self._emit_error(f"STT failed unexpectedly: {type(exc).__name__}", code="stt_error")
                continue
            if self._closed:
                return
            item = self._items_by_id.get(job.item_id)
            if item is not None:
                item["content"][0]["transcript"] = text
            self._emit(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": job.item_id,
                    "content_index": 0,
                    "transcript": text,
                }
            )
            self._log(
                "stt_completed",
                item_id=job.item_id,
                turn_epoch=job.epoch,
                backend="stt",
                model=self.settings.stt.model,
                duration_ms=(self._loop.time() - started) * 1000,
                # H3: how much the recognized user speech resembles what the
                # assistant just said (echo of TTS back into the mic)
                stt_echo_ratio=self._similarity_ratio(text, self._last_assistant_transcript),
            )
            td = self._turn_detection_cfg()
            if td.type == "server_vad" and td.create_response:
                if job.epoch == self._turn_epoch:
                    if text:
                        self._superseded_unanswered = False
                        self._request_response({}, source="auto", turn_epoch=job.epoch)
                    elif self._superseded_unanswered:
                        # the newer burst said nothing: answer the sentence before it
                        self._answer_superseded("empty_transcript")
                elif text:
                    # speech for a newer turn already started: this result
                    # waits for it — answered together with it, or alone if
                    # the newer burst is noise or transcribes to nothing
                    self._superseded_unanswered = True
                    self._log(
                        "stt_result_superseded",
                        item_id=job.item_id,
                        turn_epoch=job.epoch,
                        stale=True,
                        turn_queue=self._turns.qsize(),
                    )

    # ------------------------------------------------------------ responses

    def _request_response(self, req: dict[str, Any], *, source: str = "client", turn_epoch: int | None = None) -> None:
        if not isinstance(req, dict):
            raise ProtocolError("response.create requires a 'response' object")
        # fail closed while a full-profile route is pending: EVERY
        # response.create is rejected and NO LLM call is made
        if self._route_pending:
            self._log("response_create_rejected", source=source, reason="voice_route_pending")
            self._emit_error(
                "voice routing is pending: the initial session.update with "
                "audio.output.voice must arrive before any response",
                code="voice_route_pending",
            )
            return
        instructions = req.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ProtocolError("response.instructions must be a string")
        if self._active_response is not None:
            if source == "client":
                # public API: an explicit response.create during an active
                # response stays a hard error
                self._log(
                    "response_create_rejected",
                    response_id=self._active_response.id,
                    source=source,
                    reason="already_active",
                )
                self._emit_error(
                    "Conversation already has an active response in progress",
                    code="already_active_response",
                )
            else:
                # internal VAD automation coalesces instead of spamming busy
                # errors; flushed when the active response finishes
                queued = self._turn_epoch if turn_epoch is None else turn_epoch
                self._auto_queued_epoch = queued
                self._log(
                    "auto_response_queued",
                    response_id=self._active_response.id,
                    source=source,
                    turn_epoch=queued,
                    turn_queue=self._turns.qsize(),
                )
            return
        resp = ResponseContext(
            id=self._next_id("resp"),
            gen=self._gen,
            instructions=instructions,
            tts_queue=asyncio.Queue(maxsize=16),
            item={
                "id": self._next_id("item"),
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [{"type": "audio", "transcript": ""}],
                "segments": [],
            },
            reserved_at=self._loop.time(),
            turn_epoch=self._turn_epoch,
            pacing=self._pacing_on(),
        )
        # atomic reservation: the slot is taken synchronously, before the
        # task is spawned — back-to-back create events can never double-book
        self._active_response = resp
        ms_since_done = (
            (self._loop.time() - self._last_response_done_at) * 1000.0
            if self._last_response_done_at is not None
            else None
        )
        by_type = dict(self._items_since_last_response)
        new_items_total = sum(by_type.values())
        # the assistant message a previous response itself appended is not
        # "new input": H4 is about a create with no new USER/TOOL item
        new_input_items = new_items_total - int(by_type.get("message:assistant", 0))
        # H4 visibility (no dedupe): a client response.create with nothing new
        # since the last completed response and no one-off instructions is the
        # duplicate-answer shape — flag it so operators can see it in reports.
        duplicate_suspect = (
            source == "client"
            and new_input_items <= 0
            and instructions is None
            and self._last_response_done_at is not None
        )
        self._log(
            "response_reserved",
            response_id=resp.id,
            source=source,
            turn_epoch=resp.turn_epoch,
            out_queue=self._out.qsize(),
            turn_queue=self._turns.qsize(),
            audio_queue=self._audio_in.qsize(),
            # provenance (H4/H5): what is the LLM actually being asked, and did
            # anything new arrive since the previous response?
            has_instructions=instructions is not None,
            duplicate_request_suspect=duplicate_suspect,
            instructions_sha8=self._instructions_sha8(instructions),
            new_items_since_last_response=sum(by_type.values()),
            new_items_by_type=json.dumps(by_type, sort_keys=True, separators=(",", ":")) if by_type else "",
            last_item_type=self._last_item_type or None,
            client_event_id=self._current_event_id,
            ms_since_last_response_done=ms_since_done,
        )
        # the reservation consumes the accumulated new-item bookkeeping
        self._items_since_last_response.clear()
        self._response_task = self._spawn(self._run_response(resp))

    @staticmethod
    def _similarity_ratio(a: str, b: str) -> float | None:
        """difflib ratio of two texts in [0,1], rounded; None if either is
        empty (no meaningful base). Used to spot echo (H3) and duplicated
        answers (H4/H6) without ever logging the text."""
        if not a or not b:
            return None
        return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)

    @staticmethod
    def _instructions_sha8(instructions: str | None) -> str | None:
        """SHA256 prefix of one-off instructions (never the text itself)."""
        if not instructions:
            return None
        return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]

    def _answer_superseded(self, reason: str) -> None:
        """Answer a transcript that was held back for a newer burst which
        then turned out to be no speech. Only when nothing else is pending:
        a response running or an STT job queued answers it anyway."""
        self._superseded_unanswered = False
        if self._closed:
            return
        self._log("superseded_turn_answered", reason=reason, turn_epoch=self._turn_epoch)
        self._request_response({}, source="auto", turn_epoch=self._turn_epoch)

    def _maybe_start_queued_auto(self) -> None:
        epoch = self._auto_queued_epoch
        if epoch is None or self._closed:
            return
        self._auto_queued_epoch = None
        td = self._turn_detection_cfg()
        if not (td.type == "server_vad" and td.create_response):
            return
        if epoch != self._turn_epoch:
            # speech for a newer turn started meanwhile: the queued request
            # is stale and must not answer the old turn on its own — it waits
            # for the newer burst (answered with it, or alone if it is noise)
            self._superseded_unanswered = True
            self._log(
                "auto_response_discarded",
                reason="turn_epoch_advanced",
                turn_epoch=epoch,
                turn_queue=self._turns.qsize(),
                stale=True,
            )
            return
        self._log("auto_response_flushed", turn_epoch=epoch)
        self._request_response({}, source="auto", turn_epoch=epoch)

    def interrupt(self, reason: str) -> None:
        resp = self._active_response
        if resp is None:
            return
        # a confirmed barge-in (or any cancel) ends the pending hold: the
        # not-yet-emitted held audio is dropped, not flushed
        resp.barge_in_pending = False
        held_ms = resp.held_audio_ms + resp.held_overflow_ms
        resp.held_audio = []
        resp.held_audio_ms = 0.0
        resp.held_overflow_ms = 0.0
        self._cancel_barge_in_task()
        # H1: dropping the paced-but-not-yet-sent audio makes the cancel
        # effective; the untruncated audio must not count as spoken either
        dropped_pacing_ms = self._drop_pacing_queue(resp)
        if resp.pacing_task is not None:
            resp.pacing_task.cancel()
            resp.pacing_task = None
        total_dropped = held_ms + dropped_pacing_ms
        if total_dropped > 0:
            self._log(
                "pacing_dropped_audio_ms",
                response_id=resp.id,
                turn_epoch=resp.turn_epoch,
                reason=reason,
                audio_ms=int(total_dropped),
            )
        if total_dropped > 0:
            # only the delivered part was actually heard: neither the audio
            # held for an unconfirmed barge-in nor the paced-not-sent audio
            resp.audio_ms = max(0.0, resp.audio_ms - total_dropped)
        self._gen += 1
        resp.cancelled = True
        resp.cancel_reason = reason
        self._log("response_cancelled", response_id=resp.id, turn_epoch=resp.turn_epoch, reason=reason)
        # segments still waiting in the TTS queue never reach synthesis; they
        # must be visible in metadata (count + reason), otherwise the
        # missing-rest queue stays invisible after a cancel
        discarded = self._drain_tts_queue(resp)
        if discarded:
            self._log(
                "tts_queue_discarded",
                response_id=resp.id,
                turn_epoch=resp.turn_epoch,
                count=len(discarded),
                reason=reason,
            )
            if resp.seg_capture is not None:
                for queued_index, seg in enumerate(discarded):
                    resp.seg_capture.record_queued(queued_index=queued_index, text=seg.text, reason=reason)
        task = self._response_task
        if task is not None:
            # also covers the not-yet-started reservation: a task cancelled
            # before its first slice never executes, so the synchronous
            # finish below is the only lifecycle emission for that response
            task.cancel()
        if resp.tts_task is not None:
            resp.tts_task.cancel()
        if resp.audio_ms > 0:
            self._truncate_response_item(resp)
        elif "".join(resp.text_parts) and resp.item["id"] not in self._items_by_id:
            # spoken nothing audible but said something: keep it out of context
            pass
        self._finish_response(resp, status="cancelled")
        # identity-checked cleanup: _finish_response may have deferred work
        # scheduled, but it must never clobber a newer reservation
        if self._active_response is resp:
            self._active_response = None
        if task is not None and self._response_task is task:
            self._response_task = None
    def _truncate_response_item(self, resp: ResponseContext) -> None:
        full_text = "".join(resp.text_parts)
        # interrupt may strike while the LLM stream is still open or while the
        # TTS worker is mid-segment: snapshot the LIVE milestones here, the
        # item snapshot can be empty/partial (history metadata only)
        resp.item["segments"] = [list(m) for m in resp.milestones]
        cut = self._cut_chars_for_audio(resp.item, resp.audio_ms)
        truncated_text = full_text[:cut]
        resp.item["content"][0]["transcript"] = truncated_text
        resp.item["truncated_audio_ms"] = resp.audio_ms
        resp.item["status"] = "completed"
        if resp.item["id"] not in self._items_by_id:
            self._register_item(resp.item)
        self._emit(
            {
                "type": "conversation.item.truncated",
                "item_id": resp.item["id"],
                "content_index": 0,
                "audio_end_ms": int(resp.audio_ms),
            }
        )

    def _drain_tts_queue(self, resp: ResponseContext) -> list[Segment]:
        # a sentence fetched ahead was taken from the queue but never spoken
        discarded: list[Segment] = [s for s in resp.tts_ahead if isinstance(s, Segment)]
        resp.tts_ahead.clear()
        while True:
            try:
                item = resp.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            if isinstance(item, Segment):
                discarded.append(item)

    def _finish_response(self, resp: ResponseContext, *, status: str) -> None:
        if resp.finished:
            return
        # a response that completes while a barge-in is still unconfirmed must
        # not lose its held tail: flush it before declaring done
        if resp.barge_in_pending and status == "completed":
            self._reject_barge_in(resp, speech_ms=None)
        elif resp.barge_in_pending:
            resp.barge_in_pending = False
            resp.held_audio = []
            resp.held_audio_ms = 0.0
            self._cancel_barge_in_task()
        if not resp.created_sent:
            self._emit_response_created(resp)
        resp.finished = True
        if self._active_response is resp:
            self._active_response = None
        response_obj = {
            "id": resp.id,
            "object": "realtime.response",
            "status": status,
            "output": self._response_output(resp),
            "usage": resp.usage,
        }
        self._emit({"type": "response.done", "response": response_obj})
        now = self._loop.time()
        # H1 playback tail: how much audio the client may still be playing
        # when the SERVER already declared the response done. Only meaningful
        # for a completed response; a cancel truncates the emitted stream.
        tail_ms: float | None = None
        if (
            status == "completed"
            and resp.first_audio_delta_at is not None
            and resp.audio_ms > 0
        ):
            since_first = now - resp.reserved_at - resp.first_audio_delta_at / 1000.0
            tail_ms = max(0.0, resp.audio_ms - since_first * 1000.0)
        transcript = "".join(resp.text_parts)
        self._log(
            "response_done",
            response_id=resp.id,
            turn_epoch=resp.turn_epoch,
            status=status,
            duration_ms=(now - resp.reserved_at) * 1000 if resp.reserved_at else None,
            audio_ms=int(resp.audio_ms),
            first_audio_delta_at=resp.first_audio_delta_at,
            last_audio_delta_at=resp.last_audio_delta_at,
            estimated_playback_tail_ms=tail_ms,
            assistant_repeat_ratio=self._similarity_ratio(
                transcript, self._last_assistant_transcript
            ),
        )
        # remember for the next response (similarity base) and for
        # speech_started playback-tail attribution
        if transcript:
            self._last_assistant_transcript = transcript
        self._last_response_done_at = now
        self._last_estimated_tail_ms = tail_ms
        self._schedule_queued_auto(status)

    def _schedule_queued_auto(self, status: str) -> None:
        """Cancel semantics: a queued internal auto request is dropped when a
        response ends cancelled/failed/incomplete — it is only flushed after a
        cleanly completed response, and never reentrantly during interrupt()
        (scheduled onto the loop so state is coherent first)."""
        epoch = self._auto_queued_epoch
        if epoch is None:
            return
        if status != "completed":
            self._auto_queued_epoch = None
            self._log(
                "auto_response_dropped",
                reason=f"response_{status}",
                turn_epoch=epoch,
                turn_queue=self._turns.qsize(),
            )
            return
        self._spawn(self._flush_queued_auto())

    async def _flush_queued_auto(self) -> None:
        # yield once so any in-progress synchronous cleanup (e.g. interrupt()
        # epilogue) finishes before a new response is reserved
        await asyncio.sleep(0)
        self._maybe_start_queued_auto()

    def _emit_response_created(self, resp: ResponseContext) -> None:
        if resp.created_sent:
            return
        resp.created_sent = True
        self._emit(
            {
                "type": "response.created",
                "response": {
                    "id": resp.id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                    "instructions": resp.instructions,
                    "voice": self._cfg["voice"],
                },
            }
        )

    def _response_output(self, resp: ResponseContext) -> list[dict[str, Any]]:
        outputs = []
        for acc in resp.tool_calls.values():
            if acc.added:
                outputs.append(
                    {
                        "id": acc.item_id,
                        "type": GA_TOOL_CALL_ITEM,
                        "status": "completed",
                        "call_id": acc.call_id,
                        "name": acc.name,
                        "arguments": acc.arguments,
                    }
                )
        # the assistant message only when it was announced (a tool-only
        # response never added one, so it must not appear in response.done)
        if resp.item["id"] in resp.output_ids:
            outputs.append(_public_item(resp.item))
        return outputs

    async def _run_response(self, resp: ResponseContext) -> None:
        response_task = asyncio.current_task()
        assert response_task is not None
        resp.response_task = response_task
        if self._is_stale(resp):
            # the reservation was cancelled before this task got its first
            # slice; interrupt() already emitted the coherent created/done
            return
        if self.backends.llm is None:
            self._emit_error("no LLM backend configured", code="llm_not_configured")
            self._finish_response(resp, status="failed")
            return
        llm = self.backends.llm
        self._log(
            "response_started",
            response_id=resp.id,
            turn_epoch=resp.turn_epoch,
            backend="llm",
            model=self.settings.llm.model,
            out_queue=self._out.qsize(),
        )
        resp.seg_capture = SegmentCapture(resp.id, resp.turn_epoch)
        try:
            resp.tts_task = self._spawn(self._tts_loop(resp))
            self._emit_response_created(resp)
            messages = self._build_messages(resp.instructions)
            tools = self._chat_tools()
            # H5 request shape: how the context is laid out (roles in order,
            # counts, tools) — never the text itself
            self._log(
                "llm_request",
                response_id=resp.id,
                turn_epoch=resp.turn_epoch,
                backend="llm",
                model=self.settings.llm.model,
                role_sequence=_role_sequence(messages),
                message_count=len(messages),
                tools_count=len(tools or []),
                max_tokens=self.settings.llm.extra.get("max_tokens") if isinstance(self.settings.llm.extra, dict) else None,
            )
            status = "completed"
            buffer = SentenceBuffer()
            item_created = False
            tool_calls_count = 0
            reasoning_chars_stripped = 0
            reasoning_chars = 0
            reasoning_duplicate_suppressed = 0
            finish_reason: str | None = None
            try:
                draining = False
                async with contextlib.aclosing(llm.stream_chat(messages, tools)) as stream:
                    async for event in stream:
                        if self._is_stale(resp):
                            # keep consuming so the backend generator chain
                            # unwinds naturally instead of being orphaned
                            draining = True
                        # ReasoningMeta is pure metadata and arrives AFTER the
                        # Finish event (the adapter resolves duplicate reasoning
                        # at stream end), so it must be collected even while
                        # draining. It never carries text.
                        if isinstance(event, ReasoningMeta):
                            reasoning_chars += event.reasoning_chars
                            reasoning_duplicate_suppressed += event.duplicate_suppressed_chars
                            continue
                        if draining:
                            continue
                        if isinstance(event, TextDelta):
                            if not item_created:
                                item_created = True
                                resp.output_ids.append(resp.item["id"])
                                self._emit(
                                    {
                                        "type": "response.output_item.added",
                                        "response_id": resp.id,
                                        "output_index": 0,
                                        "item": _public_item(resp.item),
                                    }
                                )
                                self._emit(
                                    {
                                        "type": "response.content_part.added",
                                        "response_id": resp.id,
                                        "item_id": resp.item["id"],
                                        "output_index": 0,
                                        "content_index": 0,
                                        "part": {"type": "audio", "transcript": ""},
                                    }
                                )
                            resp.text_parts.append(event.text)
                            self._emit(
                                {
                                    "type": "response.output_audio_transcript.delta",
                                    "response_id": resp.id,
                                    "item_id": resp.item["id"],
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": event.text,
                                }
                            )
                            for segment in buffer.push(event.text):
                                segment.text_len = len(segment.text)
                                await self._queue_tts(resp, segment)
                        elif isinstance(event, ToolCallDelta):
                            self._handle_tool_delta(resp, event)
                        elif isinstance(event, ReasoningStripped):
                            # metadata only: count suppressed reasoning chars;
                            # the text never becomes a transcript or speech
                            reasoning_chars_stripped += event.chars
                        elif isinstance(event, Finish):
                            finish_reason = event.reason or "stop"
                            draining = True
                if self._is_stale(resp):
                    return
                tool_calls_count = sum(1 for acc in resp.tool_calls.values() if acc.name)
                self._log(
                    "llm_finish",
                    response_id=resp.id,
                    turn_epoch=resp.turn_epoch,
                    backend="llm",
                    model=self.settings.llm.model,
                    finish_reason=finish_reason,
                    output_chars=len("".join(resp.text_parts)),
                    tool_calls_count=tool_calls_count,
                    reasoning_chars_stripped=reasoning_chars_stripped or None,
                    reasoning_chars=reasoning_chars or None,
                    reasoning_duplicate_suppressed_chars=reasoning_duplicate_suppressed or None,
                    # F13: a `length` finish with no speakable text means the
                    # token budget ran out (often while the model was still
                    # thinking). Nothing was spoken or stored — flag it so the
                    # cause is visible without ever logging reasoning text.
                    no_answer=True if (finish_reason == "length" and not resp.text_parts) else None,
                )
                for segment in buffer.flush():
                    segment.text_len = len(segment.text)
                    await self._queue_tts(resp, segment)
                self._finalize_tool_calls(resp)
                full_text = "".join(resp.text_parts)
                if full_text or resp.audio_ms > 0:
                    resp.item["content"][0]["transcript"] = full_text
                    resp.item["segments"] = [list(m) for m in resp.milestones]
                    self._register_item(resp.item)
                    await self._close_tts_worker(resp)
                    await self._drain_tts(resp)
                    # H2: a barge-in still unconfirmed when the answer is fully
                    # synthesized is decided first — confirmed it cancels, as
                    # it would have mid-answer; rejected its held audio goes
                    # out BEFORE the done events and through the pacing still open
                    await self._settle_barge_in(resp)
                    if self._is_stale(resp):
                        return
                    # H1: with realtime pacing, wait out the schedule so
                    # response.done only lands after the last delta is due
                    await self._close_pacing(resp)
                    if self._is_stale(resp):
                        return
                    # the pre-drain snapshot missed milestones of segments the
                    # still-running TTS worker emitted during the drain; the
                    # completed item must carry the final mapping (history
                    # metadata only — a later client truncate depends on it)
                    resp.item["segments"] = [list(m) for m in resp.milestones]
                    self._emit(
                        {
                            "type": "response.output_audio_transcript.done",
                            "response_id": resp.id,
                            "item_id": resp.item["id"],
                            "output_index": 0,
                            "content_index": 0,
                            "transcript": full_text,
                        }
                    )
                    self._emit(
                        {
                            "type": "response.output_audio.done",
                            "response_id": resp.id,
                            "item_id": resp.item["id"],
                            "output_index": 0,
                            "content_index": 0,
                        }
                    )
                    self._emit(
                        {
                            "type": "response.content_part.done",
                            "response_id": resp.id,
                            "item_id": resp.item["id"],
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "audio", "transcript": full_text},
                        }
                    )
                    resp.item["status"] = "completed"
                    self._emit(
                        {
                            "type": "response.output_item.done",
                            "response_id": resp.id,
                            "output_index": 0,
                            "item": _public_item(resp.item),
                        }
                    )
                if resp.tts_error:
                    status = "incomplete"
                    self._emit_error(f"TTS failed: {resp.tts_error}", code="tts_error")
            except BackendError as exc:
                status = "incomplete"
                self._emit_error(f"{exc.message}", code=exc.code)
            finally:
                await self._close_tts_worker(resp)
            self._finish_response(resp, status=status)
        except asyncio.CancelledError:
            self._finish_response(resp, status="cancelled")
            raise
        except Exception as exc:
            # never leak raw backend exception text to the client
            self._log("response_failed", response_id=resp.id, code="response_error", reason=type(exc).__name__)
            self._emit_error(f"response failed unexpectedly: {type(exc).__name__}", code="response_error")
            self._finish_response(resp, status="failed")
        finally:
            if self._active_response is resp:
                self._active_response = None
            if self._response_task is response_task:
                self._response_task = None

    def _handle_tool_delta(self, resp: ResponseContext, delta: ToolCallDelta) -> None:
        acc = resp.tool_calls.get(delta.index)
        if acc is None:
            acc = ToolCallAcc(
                item_id=self._next_id("item"),
                call_id=delta.id or self._next_id("call"),
            )
            resp.tool_calls[delta.index] = acc
        if delta.name and not acc.name:
            acc.name = delta.name
        if delta.id:
            acc.call_id = delta.id
        if delta.arguments_delta:
            acc.arguments += delta.arguments_delta
        if acc.name and not acc.added:
            acc.added = True
            self._emit(
                {
                    "type": "response.output_item.added",
                    "response_id": resp.id,
                    "output_index": self._tool_output_index(resp, acc),
                    "item": {
                        "id": acc.item_id,
                        "type": GA_TOOL_CALL_ITEM,
                        "status": "in_progress",
                        "call_id": acc.call_id,
                        "name": acc.name,
                        "arguments": "",
                    },
                }
            )
        if delta.arguments_delta:
            self._emit(
                {
                    "type": "response.function_call_arguments.delta",
                    "response_id": resp.id,
                    "item_id": acc.item_id,
                    "output_index": self._tool_output_index(resp, acc),
                    "call_id": acc.call_id,
                    "delta": delta.arguments_delta,
                }
            )

    def _tool_output_index(self, resp: ResponseContext, acc: ToolCallAcc) -> int:
        if acc.item_id not in resp.output_ids:
            resp.output_ids.append(acc.item_id)
        return resp.output_ids.index(acc.item_id)

    def _finalize_tool_calls(self, resp: ResponseContext) -> None:
        for acc in resp.tool_calls.values():
            if not acc.name:
                continue
            item = {
                "id": acc.item_id,
                "type": GA_TOOL_CALL_ITEM,
                "status": "completed",
                "call_id": acc.call_id,
                "name": acc.name,
                "arguments": acc.arguments or "{}",
            }
            self._register_item(item)
            self._log("tool_call", call_id=item["call_id"], item_id=item["id"])
            self._emit(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": resp.id,
                    "item_id": acc.item_id,
                    "output_index": self._tool_output_index(resp, acc),
                    "call_id": acc.call_id,
                    "name": acc.name,
                    "arguments": item["arguments"],
                }
            )
            self._emit(
                {
                    "type": "response.output_item.done",
                    "response_id": resp.id,
                    "output_index": self._tool_output_index(resp, acc),
                    "item": item,
                }
            )

    async def _queue_tts(self, resp: ResponseContext, segment) -> None:
        """Hand a sentence to the TTS worker. A worker that has ended (TTS
        failed, no TTS backend, an unrouted voice) reads nothing more, so the
        sentence is dropped instead of blocking the response on a full queue
        for good; the response then ends with the TTS error."""
        task = resp.tts_task
        if task is None or task.done():
            resp.tts_dropped_segments += 1
            return
        put = asyncio.ensure_future(resp.tts_queue.put(segment))
        done, _ = await asyncio.wait({put, task}, return_when=asyncio.FIRST_COMPLETED)
        if put not in done:
            put.cancel()
            resp.tts_dropped_segments += 1

    async def _close_tts_worker(self, resp: ResponseContext) -> None:
        """Signal the TTS worker to exit after draining pending segments."""
        task = resp.tts_task
        if task is None or task.done() or resp.tts_closed:
            return
        resp.tts_closed = True
        await self._queue_tts(resp, None)

    async def _drain_tts(self, resp: ResponseContext) -> None:
        task = resp.tts_task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            raise

    async def _tts_loop(self, resp: ResponseContext) -> None:
        if self._voice_route_pending():
            # fail closed: an unrouted voice-route session never synthesizes
            # with a fallback voice (the one-time pending error was already
            # emitted by the shared guard)
            resp.tts_error = "voice routing pending"
            return
        if self.backends.tts is None:
            resp.tts_error = "no TTS backend configured"
            self._emit_error("no TTS backend configured", code="tts_not_configured")
            return
        tts = self.backends.tts
        voice = self._cfg["voice"]
        language = self._cfg["language"] or None
        pending = bytearray()
        chunk_bytes = int(DELTA_CHUNK_MS * BYTES_PER_MS)
        # a model that delivers its first pieces barely in real time gets a
        # head start: a sentence begins speaking once this much of it is
        # ready (or it is complete), so the listener never runs dry mid-word
        start_buffer_bytes = int(getattr(self.settings.tts, "start_buffer_ms", 0) * BYTES_PER_MS)
        speaking = start_buffer_bytes <= 0
        tts_started_at: float | None = None
        # per-segment metadata (id/count/status/duration/reason, never content)
        capture = resp.seg_capture if resp.seg_capture is not None else SegmentCapture(resp.id, resp.turn_epoch)
        segment_index = 0
        emitted_bytes_total = 0
        skipped_symbols = 0

        def log_segment(
            status: str,
            seg_start_ms: float,
            seg_emitted: int,
            reason: str | None,
            emoji_removed: int = 0,
        ) -> None:
            self._log(
                "tts_segment",
                response_id=resp.id,
                turn_epoch=resp.turn_epoch,
                segment_index=segment_index,
                seg_start_ms=seg_start_ms,
                audio_ms=resp.audio_ms - seg_start_ms,
                chars=len(getattr(segment, "text", "")),
                pcm_bytes=seg_emitted,
                status=status,
                reason=reason,
                emoji_removed=emoji_removed or None,
            )
            capture.record(
                segment_index=segment_index,
                text=segment.text,
                spoken_text=cleaned.text if removed_here else None,
                backend_pcm=seg_backend_pcm or b"",
                emitted_pcm=seg_emitted_pcm,
                seg_start_ms=seg_start_ms,
                audio_ms=resp.audio_ms - seg_start_ms,
                start_byte=emitted_bytes_total,
                end_byte=emitted_bytes_total + seg_emitted,
                status=status,
                reason=reason,
            )

        # The next sentence is asked for while the current one is spoken, so
        # a model's start-up time (up to a second for some) is not heard as
        # a gap between sentences. One sentence ahead; its audio is held
        # until its turn.
        ahead: list[Any] = resp.tts_ahead  # [segment] taken from the queue (None = end)
        fetches: dict[int, tuple[asyncio.Queue, asyncio.Task]] = {}
        producers: list[asyncio.Task] = []  # every request started, awaited at the end

        def speakable(seg: Any) -> bool:
            return seg is not None and bool(strip_emoji_for_speech(seg.text).text.strip())

        def start_fetch(seg: Any) -> None:
            if id(seg) in fetches:
                return
            q: asyncio.Queue = asyncio.Queue()
            text = strip_emoji_for_speech(seg.text).text

            async def produce() -> None:
                try:
                    async for pcm in tts.stream_speech(text, voice=voice, language=language):
                        q.put_nowait(pcm)
                    q.put_nowait(_FETCH_DONE)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # handed to the consumer in order
                    q.put_nowait(exc)

            task = asyncio.ensure_future(produce())
            producers.append(task)
            fetches[id(seg)] = (q, task)

        def look_ahead() -> None:
            if not ahead:
                try:
                    ahead.append(resp.tts_queue.get_nowait())
                except asyncio.QueueEmpty:
                    return
            if speakable(ahead[0]) and not self._is_stale(resp):
                start_fetch(ahead[0])

        async def fetched(seg: Any):
            start_fetch(seg)
            q, _task = fetches[id(seg)]
            try:
                while True:
                    item = await q.get()
                    if item is _FETCH_DONE:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    look_ahead()
                    yield item
            finally:
                entry = fetches.pop(id(seg), None)
                if entry is not None and not entry[1].done():
                    entry[1].cancel()

        try:
            while True:
                segment = ahead.pop(0) if ahead else await resp.tts_queue.get()
                if segment is None:
                    break
                if self._is_stale(resp):
                    break
                # TTS-input hygiene: strip emoji sequences (ZWJ families, skin
                # tones, flags, VS16, keycaps->digit) from the SPOKEN text only;
                # the display/transcript text and the original-offset/truncation
                # mapping keep the original. A segment that is empty after the
                # strip is skipped entirely (metadata only, never content).
                cleaned = strip_emoji_for_speech(segment.text)
                removed_here = cleaned.removed
                if not cleaned.text.strip():
                    # purely decorative symbol/emoji segment: no TTS request,
                    # no audio milestone — metadata-only record (never content)
                    skipped_symbols += 1
                    self._log(
                        "tts_segment",
                        response_id=resp.id,
                        turn_epoch=resp.turn_epoch,
                        segment_index=segment_index,
                        seg_start_ms=resp.audio_ms,
                        audio_ms=0,
                        chars=len(segment.text),
                        pcm_bytes=0,
                        status="skipped",
                        reason="non_speakable_symbol",
                        count=skipped_symbols,
                        emoji_removed=removed_here or None,
                    )
                    capture.record(
                        segment_index=segment_index,
                        text=segment.text,
                        backend_pcm=b"",
                        emitted_pcm=b"",
                        seg_start_ms=resp.audio_ms,
                        audio_ms=0,
                        start_byte=emitted_bytes_total,
                        end_byte=emitted_bytes_total,
                        status="skipped",
                        reason="non_speakable_symbol",
                    )
                    segment_index += 1
                    continue
                if tts_started_at is None:
                    tts_started_at = self._loop.time()
                    ref_fp = (
                        tts.reference_fingerprint()
                        if hasattr(tts, "reference_fingerprint")
                        else ""
                    )
                    # effective per-first-TTS identity, honestly separated:
                    # session_voice = what the session ASKS the adapter with;
                    # payload_voice = what the adapter ACTUALLY puts into the
                    # HTTP request (extra keys can override it). Plus the
                    # resolved profile, model and reference fingerprint — all
                    # safe metadata, so same-model reference switches are
                    # provable in the field (Clara/Jonas chatterbox)
                    self._log(
                        "tts_started",
                        response_id=resp.id,
                        turn_epoch=resp.turn_epoch,
                        backend="tts",
                        model=self.settings.tts.model,
                        tts_profile=(self._resolved_info or {}).get("tts", {}).get("profile", "") or None,
                        session_voice=self._cfg["voice"] or None,
                        payload_voice=(
                            tts.effective_request_voice(voice)
                            if hasattr(tts, "effective_request_voice")
                            else None
                        ) or None,
                        ref_sha256=ref_fp or None,
                    )
                seg_start_ms = resp.audio_ms
                seg_start_char = segment.start_char
                seg_text_len = segment.text_len or len(segment.text)
                # the SPOKEN text is the cleaned one; the milestone keeps the
                # original offsets and carries the cleaned->original position
                # map so a mid-segment truncate cuts the ORIGINAL transcript
                seg_spoken_text = cleaned.text
                seg_offsets = cleaned.offsets if removed_here else None
                seg_emitted_pcm = bytearray()
                # raw backend stream of this segment; accumulated only when a
                # test capture is active (never in production)
                seg_backend_pcm = bytearray() if capture.enabled else None
                try:
                    # this sentence is asked for first, then the next one: a
                    # server that works one request at a time keeps the order
                    start_fetch(segment)
                    look_ahead()
                    # each sentence starts speaking once it has its head start
                    # (a sentence fetched ahead usually has it already)
                    speaking = start_buffer_bytes <= 0
                    async for pcm in fetched(segment):
                        if seg_backend_pcm is not None:
                            seg_backend_pcm.extend(pcm)
                        if self._is_stale(resp):
                            continue
                        pending.extend(pcm)
                        if not speaking and len(pending) >= start_buffer_bytes:
                            speaking = True
                        while speaking and len(pending) >= chunk_bytes:
                            if self._is_stale(resp):
                                continue
                            chunk = bytes(pending[:chunk_bytes])
                            del pending[:chunk_bytes]
                            self._emit_audio_delta(resp, chunk, seg_start_ms, seg_start_char, seg_text_len, seg_offsets)
                            if not self._is_stale(resp):
                                seg_emitted_pcm.extend(chunk)
                    speaking = True
                    # a sentence that ended before its head start was full:
                    # still out in the usual pieces (pacing relies on them)
                    while len(pending) >= chunk_bytes and not self._is_stale(resp):
                        chunk = bytes(pending[:chunk_bytes])
                        del pending[:chunk_bytes]
                        self._emit_audio_delta(resp, chunk, seg_start_ms, seg_start_char, seg_text_len, seg_offsets)
                        if not self._is_stale(resp):
                            seg_emitted_pcm.extend(chunk)
                    if pending:
                        chunk = bytes(pending)
                        pending.clear()
                        self._emit_audio_delta(resp, chunk, seg_start_ms, seg_start_char, seg_text_len, seg_offsets)
                        if not self._is_stale(resp):
                            seg_emitted_pcm.extend(chunk)
                except BackendError as exc:
                    resp.tts_error = exc.message
                    self._log(
                        "tts_failed",
                        response_id=resp.id,
                        turn_epoch=resp.turn_epoch,
                        backend="tts",
                        model=self.settings.tts.model,
                        code=exc.code,
                        duration_ms=(self._loop.time() - tts_started_at) * 1000 if tts_started_at else None,
                    )
                    log_segment("failed", seg_start_ms, len(seg_emitted_pcm), exc.code, removed_here)
                    # Stop at the first failed sentence: the rest would each
                    # ask a backend that just failed and report it again. The
                    # response ends incomplete with ONE tts error.
                    return
                except asyncio.CancelledError:
                    log_segment("cancelled", seg_start_ms, len(seg_emitted_pcm), resp.cancel_reason, removed_here)
                    raise
                else:
                    if self._is_stale(resp):
                        log_segment("cancelled", seg_start_ms, len(seg_emitted_pcm), resp.cancel_reason, removed_here)
                    elif not seg_emitted_pcm:
                        log_segment("zero_audio", seg_start_ms, 0, None, removed_here)
                    else:
                        log_segment("ok", seg_start_ms, len(seg_emitted_pcm), None, removed_here)
                    emitted_bytes_total += len(seg_emitted_pcm)
                    segment_index += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # never the raw exception text: it can carry backend details
            resp.tts_error = f"unexpected {type(exc).__name__}"
            self._log(
                "tts_failed",
                response_id=resp.id,
                turn_epoch=resp.turn_epoch,
                backend="tts",
                model=self.settings.tts.model,
                code="tts_error",
                reason=type(exc).__name__,
                duration_ms=(self._loop.time() - tts_started_at) * 1000 if tts_started_at else None,
            )
        else:
            if tts_started_at is not None and not self._is_stale(resp):
                self._log(
                    "tts_completed",
                    response_id=resp.id,
                    turn_epoch=resp.turn_epoch,
                    backend="tts",
                    model=self.settings.tts.model,
                    duration_ms=(self._loop.time() - tts_started_at) * 1000,
                    audio_ms=int(resp.audio_ms),
                )
        finally:
            # a sentence fetched ahead that will not be spoken (cancelled,
            # failed, stale) must not keep its request running, and its
            # request is closed before the backends are
            for task in producers:
                task.cancel()
            if producers:
                await asyncio.gather(*producers, return_exceptions=True)

    def _emit_audio_delta(
        self,
        resp: ResponseContext,
        chunk: bytes,
        seg_start_ms: float,
        seg_start_char: int,
        seg_text_len: int,
        seg_offsets: list[int] | None = None,
    ) -> None:
        """Synthesis-side bookkeeping + routing.

        The milestone/audio_ms timeline is built here at synthesis time (TTS
        still runs ahead); then the chunk is held (H2 unconfirmed barge-in),
        queued for paced delivery (H1 realtime pacing) or delivered now."""
        if self._is_stale(resp):
            return
        if not resp.milestones or resp.milestones[-1][1] != seg_start_char:
            milestone = [seg_start_ms, seg_start_char, 0.0, seg_text_len]
            if seg_offsets is not None:
                milestone.append(list(seg_offsets))
            resp.milestones.append(milestone)
        duration = frame_ms(len(chunk), AUDIO_RATE)
        resp.milestones[-1][2] = resp.audio_ms + duration - seg_start_ms
        resp.audio_ms += duration
        resp.item["content"][0]["transcript"] = "".join(resp.text_parts)
        if resp.barge_in_pending:
            # H2: hold the delta (never emit) until the burst is resolved; on
            # rejection the exact chunk is delivered in order via _reject
            resp.held_audio.append(chunk)
            resp.held_audio_ms += duration
            while resp.held_audio_ms > MAX_HELD_AUDIO_MS and resp.held_audio:
                dropped_chunk = resp.held_audio.pop(0)
                dropped_ms = frame_ms(len(dropped_chunk), AUDIO_RATE)
                resp.held_audio_ms -= dropped_ms
                resp.held_overflow_ms += dropped_ms
            return
        if resp.pacing:
            self._enqueue_paced_audio(resp, chunk, duration)
            return
        self._deliver_audio_delta(resp, chunk)

    def _deliver_audio_delta(self, resp: ResponseContext, chunk: bytes) -> None:
        """Emission-side only: the chunk leaves the server now."""
        if self._is_stale(resp) or not chunk:
            return
        elapsed_ms = (self._loop.time() - resp.reserved_at) * 1000.0
        if resp.first_audio_delta_at is None:
            resp.first_audio_delta_at = elapsed_ms
        resp.last_audio_delta_at = elapsed_ms
        resp.delivered_audio_ms += frame_ms(len(chunk), AUDIO_RATE)
        self._emit(
            {
                "type": "response.output_audio.delta",
                "response_id": resp.id,
                "item_id": resp.item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": encode_base64_pcm16(chunk),
            }
        )

    # ------------------------------------------------------ H1 output pacing

    def _pacing_on(self) -> bool:
        return str(self._cfg.get("output_pacing", "off")) == "realtime"

    def _enqueue_paced_audio(self, resp: ResponseContext, chunk: bytes, duration_ms: float) -> None:
        if resp.pacing_queue is None:
            resp.pacing_queue = asyncio.Queue(maxsize=PACING_QUEUE_MAX)
            resp.pacing_task = self._spawn(self._pacing_loop(resp))
        try:
            resp.pacing_queue.put_nowait((chunk, duration_ms))
        except asyncio.QueueFull:
            # bounded: dropping the oldest queued chunk keeps the live audio
            # centre of gravity while never growing without bound
            try:
                resp.pacing_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            with contextlib.suppress(asyncio.QueueFull):
                resp.pacing_queue.put_nowait((chunk, duration_ms))

    async def _pacing_loop(self, resp: ResponseContext) -> None:
        """Emit queued audio at ~1x with a lead. Audio-time t becomes due at
        wall-time (t - lead) after the first chunk is delivered, so the last
        chunk (and response.done) lands at duration - lead."""
        lead_ms = int(self._cfg.get("output_pacing_lead_ms", 600))
        emitted_ms = 0.0
        try:
            while True:
                item = await resp.pacing_queue.get()
                if item is None:
                    return
                chunk, duration_ms = item
                if self._is_stale(resp):
                    return
                if resp.pacing_started_at == 0.0:
                    resp.pacing_started_at = self._loop.time()
                # audio-time `emitted_ms` becomes due `lead_ms` early, so the
                # last chunk (and response.done) lands at duration - lead
                due = resp.pacing_started_at + (emitted_ms - lead_ms) / 1000.0
                delay = due - self._loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._is_stale(resp):
                    return
                self._deliver_audio_delta(resp, chunk)
                emitted_ms += duration_ms
                resp.pacing_delivered_ms = emitted_ms
        except asyncio.CancelledError:
            raise

    def _drop_pacing_queue(self, resp: ResponseContext) -> float:
        """Drain the paced queue at cancel time; returns the dropped ms."""
        queue = resp.pacing_queue
        if queue is None:
            return 0.0
        dropped_ms = 0.0
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None:
                dropped_ms += item[1]
        return dropped_ms

    async def _close_pacing(self, resp: ResponseContext) -> None:
        """Drain the paced queue (waiting out the real-time schedule) so
        response.done is only emitted once the last audio has been delivered."""
        task = resp.pacing_task
        if task is None:
            return
        queue = resp.pacing_queue
        if queue is not None:
            while not queue.empty():
                if self._is_stale(resp):
                    break
                await asyncio.sleep(0.005)
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await task
        resp.pacing_task = None

    def _cut_chars_for_audio(self, item: dict[str, Any], audio_end_ms: float) -> int:
        segments = item.get("segments") or []
        if not segments or audio_end_ms <= 0:
            return 0
        kept = 0
        for seg in segments:
            start_ms, start_char, duration_ms, text_len = seg[0], seg[1], seg[2], seg[3]
            offsets = seg[4] if len(seg) > 4 else None
            if start_ms >= audio_end_ms:
                break
            if start_ms + duration_ms <= audio_end_ms:
                kept = start_char + text_len
            elif offsets is not None and len(offsets) > 1:
                # the segment's SPOKEN text was shorter than the original
                # (emoji stripped): map the spoken fraction through the
                # cleaned->original position list so the cut lands on the
                # position in the ORIGINAL transcript that was actually heard
                fraction = (audio_end_ms - start_ms) / max(duration_ms, 1.0)
                idx = min(int(fraction * (len(offsets) - 1)), len(offsets) - 1)
                kept = start_char + offsets[idx]
                break
            else:
                fraction = (audio_end_ms - start_ms) / max(duration_ms, 1.0)
                kept = start_char + int(text_len * fraction)
                break
        return kept

    # ------------------------------------------------------------- context

    def _build_messages(self, one_off_instructions: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        instructions = one_off_instructions if one_off_instructions is not None else self._cfg["instructions"]
        if instructions:
            messages.append({"role": "system", "content": instructions})
        # runtime spoken-output boundary: appended to EVERY request after the
        # (user/operator) instructions — it never replaces or rewrites the
        # stored settings instructions, user content or tools; the emoji
        # strip at the TTS boundary is independent of LLM compliance
        messages.append({"role": "system", "content": SPEECH_OUTPUT_DIRECTIVE})
        max_items = self._cfg["max_history_items"]
        # H5a: when ONE response produced both text and tool calls, its history
        # must be a SINGLE assistant message carrying `content` + `tool_calls`
        # (OpenAI chat format), followed by the tool results — never a separate
        # assistant-text message plus an assistant-tool-call message. The
        # conversation stores the function_call items and the assistant message
        # as adjacent entries; merge them here.
        for item in self._conversation[-max_items:]:
            itype = item.get("type")
            if itype == "message":
                role = item["role"]
                transcript = _item_text(item)
                if role == "user":
                    content = transcript or "[audio]"
                    messages.append({"role": "user", "content": content})
                elif role == "assistant" and transcript:
                    messages.append({"role": "assistant", "content": transcript})
            elif itype == GA_TOOL_CALL_ITEM:
                call = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
                # attach to the immediately preceding assistant message when it
                # belongs to the same response (text + tool call); append the
                # call to an already-merged message at the tail otherwise
                if (
                    messages
                    and messages[-1].get("role") == "assistant"
                    and messages[-1].get("tool_calls")
                ):
                    messages[-1]["tool_calls"].append(call)
                elif (
                    messages
                    and messages[-1].get("role") == "assistant"
                    and messages[-1].get("content")
                ):
                    # assistant text directly precedes the call: this IS the
                    # same response — fold the call into that one message
                    messages[-1]["tool_calls"] = [call]
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif itype == "function_call_output":
                messages.append(
                    {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
                )
        # second pass: an assistant text message that directly follows a
        # call-only assistant message is the same response — merge them so the
        # request carries one assistant message with content + tool_calls
        normalized: list[dict[str, Any]] = []
        for message in messages:
            prev = normalized[-1] if normalized else None
            if (
                prev is not None
                and prev.get("role") == "assistant"
                and prev.get("tool_calls")
                and not prev.get("content")
                and message.get("role") == "assistant"
                and message.get("content")
                and not message.get("tool_calls")
            ):
                prev["content"] = message["content"]
                continue
            normalized.append(message)
        return _pair_tool_results(normalized)

    def _chat_tools(self) -> list[dict[str, Any]] | None:
        tools = self._cfg["tools"] or []
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]


def _is_pcm16_24k(fmt: Any) -> bool:
    if isinstance(fmt, str):
        return fmt in ("pcm16", "audio/pcm")
    if isinstance(fmt, dict):
        ftype = fmt.get("type")
        rate = fmt.get("rate", AUDIO_RATE)
        return ftype in ("audio/pcm", "pcm16") and int(rate or AUDIO_RATE) == AUDIO_RATE
    return False


def _role_sequence(messages: list[dict[str, Any]]) -> str:
    """Compact role layout of an LLM request, e.g. ``s,s,u,a[tc],a,t``.

    Diagnostics only (H5): tool-calling assistant messages are marked with
    ``[tc]`` so a function_call landing before the assistant text of the same
    response is visible. Never carries message content.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "?"))
        letter = {"system": "s", "user": "u", "assistant": "a", "tool": "t"}.get(role, "?")
        if role == "assistant" and message.get("tool_calls"):
            letter = "a[tc]"
        # run-length compress consecutive identical entries so long histories
        # stay within the log length bound without losing their structure
        if parts and parts[-1][0] == letter and parts[-1][0] != "a[tc]":
            parts[-1][1] += 1
            continue
        parts.append([letter, 1])
    return ",".join(f"{letter}*{count}" if count > 1 else letter for letter, count in parts)


def _item_text(item: dict[str, Any]) -> str:
    parts = []
    for part in item.get("content", []):
        if isinstance(part, dict) and isinstance(part.get("transcript"), str):
            parts.append(part["transcript"])
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


# What the model reads for a tool call the client has not answered yet:
# chat APIs refuse an assistant tool call that is not followed by its result.
PENDING_TOOL_RESULT = "(no result yet: the call is still running)"


def _pair_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every assistant tool call directly followed by its tool message, as
    chat APIs require: a result that arrived after later messages (the user
    spoke before the client answered) moves up to its call, a missing one is
    a placeholder, and a result whose call was cut off by the history limit
    is dropped."""
    results: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") not in results:
            results[m["tool_call_id"]] = m
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            continue
        out.append(m)
        for call in m.get("tool_calls") or []:
            out.append(results.pop(call["id"], None) or {"role": "tool", "tool_call_id": call["id"], "content": PENDING_TOOL_RESULT})
    return out


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    public = {k: v for k, v in item.items() if k not in ("segments",)}
    if item.get("type") == "message" and item.get("role") == "assistant":
        content = [dict(part) for part in item.get("content", [])]
        for part in content:
            part.pop("audio", None)
        public["content"] = content
    return public


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
