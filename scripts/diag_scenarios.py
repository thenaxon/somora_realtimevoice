#!/usr/bin/env python3
"""Deterministic synthetic scenarios against fake backends, via the WS API.

Reproduces and measures the MECHANISM behind each interruption/duplication
hypothesis (H1, H2, H4, H5a, H5b, H7) end to end through the public
`/v1/realtime` WebSocket, with fake STT/LLM/TTS backends and synthetic audio.
These are measurements of mechanisms, not claims about the real root cause.

Output: one metrics JSON object per scenario on stdout (or --out). No real
audio, no transcripts/instructions in the output beyond counts/hashes.

Usage:
  scripts/diag_scenarios.py                 # run all scenarios
  scripts/diag_scenarios.py --scenario H4 --scenario H7
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi.testclient import TestClient

from realtimevoice.app import create_app
from realtimevoice.backends.base import Backends, Finish, ProbeResult, TextDelta, ToolCallDelta
from realtimevoice.config import (
    BackendProfile,
    ConversationProfile,
    ProfileSpec,
    Settings,
    SettingsStore,
)

SAMPLE_RATE = 24000
AUTH_TOKEN = "diag-token"
DIAG_SET = "diag-set"


def with_diag_set(settings: Settings) -> Settings:
    """A copy of the settings with one enabled set as the server's default,
    so a call naming no set gets models. The set keeps the session defaults
    the scenario configured (its spec only names the models and the pacing)."""
    out = settings.model_copy(deep=True)
    ids: dict[str, str] = {}
    for role in ("stt", "llm", "tts"):
        pid = f"{DIAG_SET}-{role}"
        out.profiles.setdefault(role, {})[pid] = BackendProfile(
            id=pid, role=role, base_url="http://diag.invalid/v1", model=f"diag-{role}"
        )
        ids[role] = pid
    now = time.time()
    out.conversation_profiles[DIAG_SET] = ConversationProfile(
        id=DIAG_SET,
        display_name="Diagnostics",
        spec=ProfileSpec(
            stt_profile=ids["stt"],
            llm_profile=ids["llm"],
            tts_profile=ids["tts"],
            output_pacing=settings.session.output_pacing,
        ),
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    out.default_set = DIAG_SET
    return out


# --------------------------------------------------------------- synthetic audio

def tone_pcm(duration_ms: int, amplitude: float = 3000.0, freq: float = 440.0) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return struct.pack(
        f"<{n}h", *(int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n))
    )


def silence_pcm(duration_ms: int) -> bytes:
    return bytes(2 * int(SAMPLE_RATE * duration_ms / 1000))


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


# ------------------------------------------------------------------ fake backends

@dataclass
class ScenarioSTT:
    text: str = "Wie ist das Wetter in Wien?"
    gate: asyncio.Event | None = None
    calls: list = field(default_factory=list)

    async def transcribe(self, wav: bytes, *, language: str | None) -> str:
        self.calls.append(wav)
        if self.gate is not None:
            await self.gate.wait()
        return self.text

    async def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, detail="scenario stt")


@dataclass
class ScenarioLLM:
    """script entries: {"text", "tool_calls", "gate"}"""

    script: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    default_text: str = "Guten Tag, wie kann ich helfen?"
    gate: asyncio.Event | None = None

    async def stream_chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self.gate is not None:
            await self.gate.wait()
        behavior = self.script.pop(0) if self.script else {"text": self.default_text}
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
        return ProbeResult(ok=True, detail="scenario llm")


@dataclass
class ScenarioTTS:
    chunk_ms: int = 100
    chunks: int = 4
    gate: asyncio.Event | None = None
    calls: list = field(default_factory=list)

    async def stream_speech(self, text, *, voice=None, language=None):
        self.calls.append({"text": text, "voice": voice, "language": language})
        if self.gate is not None:
            await self.gate.wait()
        n = int(SAMPLE_RATE * self.chunk_ms / 1000)
        for _ in range(self.chunks):
            yield struct.pack(f"<{n}h", *(int(2000 * math.sin(2 * math.pi * 440 * j / SAMPLE_RATE)) for j in range(n)))
            await asyncio.sleep(0)

    async def probe(self) -> ProbeResult:
        return ProbeResult(ok=True, detail="scenario tts")


# ------------------------------------------------------------------ log capture

_ACTIVE_CLIENTS: list["ScenarioClient"] = []


class LifecycleCapture(logging.Handler):
    """Collects lifecycle JSON records in-process (metadata only)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
        except Exception:
            return
        if isinstance(payload, dict) and payload.get("action"):
            self.records.append(payload)

    def of(self, action: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("action") == action]


# --------------------------------------------------------------------- scaffold

class ScenarioClient:
    """Thin WS driver + lifecycle capture around a TestClient app."""

    def __init__(self) -> None:
        self.capture = LifecycleCapture()
        self._logger = logging.getLogger("realtimevoice.lifecycle")
        # the app logger may be at WARNING; the handler only works with the
        # level lowered for the capture (in-process diagnostics, no console)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self.capture)
        _ACTIVE_CLIENTS.append(self)
        self.backends = Backends()
        self.settings = Settings()
        self.llm: ScenarioLLM | None = None
        self.stt: ScenarioSTT | None = None
        self.tts: ScenarioTTS | None = None

    def close(self) -> None:
        self._logger.removeHandler(self.capture)
        if self in _ACTIVE_CLIENTS:
            _ACTIVE_CLIENTS.remove(self)

    def make_app(self) -> TestClient:
        def factory(settings, http_client=None):
            return self.backends

        app = create_app(
            settings_store=SettingsStore(settings=with_diag_set(self.settings)),
            backend_factory=factory,
            auth_token=AUTH_TOKEN,
            web_dir=None,
        )
        return TestClient(app)

    def configure(self, *, stt=None, llm=None, tts=None, settings=None) -> "ScenarioClient":
        self.settings = settings or Settings()
        self.stt = stt or ScenarioSTT()
        self.llm = llm or ScenarioLLM()
        self.tts = tts or ScenarioTTS()
        self.backends = Backends(stt=self.stt, llm=self.llm, tts=self.tts)
        return self


def close_all() -> None:
    """Detach every capture handler; safe to call between scenarios/tests."""
    for client in list(_ACTIVE_CLIENTS):
        client.close()


def _connect(tc: TestClient):
    return tc.websocket_connect(
        "/v1/realtime?model=", subprotocols=["rtv.bearer", f"rtv.bearer.{AUTH_TOKEN}"]
    )


def _recv_until(ws, types: set[str], timeout_events: int = 400) -> dict[str, Any]:
    for _ in range(timeout_events):
        msg = ws.receive_json()
        if msg.get("type") in types:
            return msg
    raise AssertionError(f"no {types} received")


# --------------------------------------------------------------------- scenarios

def _isolated(fn: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
    """Run a scenario and always detach its capture handler afterwards."""

    def wrapper() -> dict[str, Any]:
        try:
            return fn()
        finally:
            close_all()

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_isolated
def scenario_h4() -> dict[str, Any]:
    """H4: back-to-back response.create without new user/tool items. The
    second create is refused while a response is active; a create after
    response.done re-answers the same context."""
    gate = asyncio.Event()
    cx = ScenarioClient().configure(
        stt=ScenarioSTT(text="Frage?"),
        llm=ScenarioLLM(
            script=[{"text": "Antwort eins."}, {"text": "Antwort zwei."}],
            gate=gate,
        ),
        tts=ScenarioTTS(),
    )
    with cx.make_app() as tc:
        with _connect(tc) as ws:
            ws.receive_json()  # session.created
            # first response (gated: stays active while we fire the second)
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.created"})
            # second create while the first is active: refused (busy)
            ws.send_json({"type": "response.create"})
            busy = _recv_until(ws, {"error"})
            # release: the first completes
            gate.set()
            _recv_until(ws, {"response.done"})
            # a create after response.done, still with NO new user/tool item:
            # the model answers the same context again (multiplication)
            gate.clear()
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.created"})
            gate.set()
            _recv_until(ws, {"response.done"})

    reserved = cx.capture.of("response_reserved")
    dones = cx.capture.of("response_done")
    return {
        "scenario": "H4",
        "mechanism": "back-to-back response.create without new items",
        "responses_done": len(dones),
        "llm_calls": len(cx.llm.calls),
        "no_new_items": sum(1 for r in reserved if r.get("new_items_since_last_response") == 0),
        "new_items_by_type": [r.get("new_items_by_type", "") for r in reserved],
        # H4 visibility only: the duplicate-answer shape is flagged, not deduped
        "duplicate_request_suspect": [bool(r.get("duplicate_request_suspect")) for r in reserved],
        "busy_error_code": busy["error"]["code"],
        "duplicated_answer": len(dones) > 1,
        "assistant_repeat_ratio": next(
            (r["assistant_repeat_ratio"] for r in dones if r.get("assistant_repeat_ratio")),
            None,
        ),
    }


@_isolated
def scenario_h1() -> dict[str, Any]:
    """H1: response.done arrives while audio would still be playing; a
    barge-in then hits the estimated tail and cancels nothing server-side."""
    llm = ScenarioLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    cx = ScenarioClient().configure(llm=llm, tts=ScenarioTTS(chunks=10, chunk_ms=100))
    with cx.make_app() as tc:
        with _connect(tc) as ws:
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.done"})
            # no response is active anymore: the client barge-in (server VAD)
            # finds nothing to cancel even though the tail is still playing
            ws.send_json({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(200))})
            _recv_until(ws, {"input_audio_buffer.speech_started"})
            ws.send_json({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(600))})
            _recv_until(ws, {"input_audio_buffer.speech_stopped"})
            # the discarded/kept burst turn then produces its own response
            _recv_until(ws, {"response.done"})

    done = cx.capture.of("response_done")
    tail = [r.get("estimated_playback_tail_ms") for r in done if r.get("estimated_playback_tail_ms")]
    started = cx.capture.of("speech_started")
    return {
        "scenario": "H1",
        "mechanism": "barge-in after response.done while audio would still play",
        "response_done_status": done[-1].get("status") if done else None,
        "estimated_playback_tail_ms": tail[-1] if tail else None,
        "barge_in_during_tail": started[-1].get("during_estimated_playback_tail") if started else None,
        "cancelled_by_that_burst": any(
            r.get("response_cancelled_by_burst") for r in cx.capture.of("vad_burst_discarded")
        ),
        "server_side_cancel_after_done": any(
            r.get("action") == "response_cancelled" for r in cx.capture.records
        ),
    }


def _run_h1_pacing(pacing: str, lead_ms: int) -> dict[str, Any]:
    """One H1 pass: measure the server-side playback tail with pacing off/on."""
    settings = Settings()
    settings.session.output_pacing = pacing
    settings.session.output_pacing_lead_ms = lead_ms
    # one sentence -> one TTS call; 20 x 100 ms = 2000 ms of audio synthesized
    # far faster than real time
    llm = ScenarioLLM(script=[{"text": "Hallo"}], default_text="Hallo")
    cx = ScenarioClient().configure(settings=settings, llm=llm, tts=ScenarioTTS(chunks=20, chunk_ms=100))
    try:
        with cx.make_app() as tc:
            with _connect(tc) as ws:
                ws.receive_json()
                ws.send_json({"type": "response.create"})
                _recv_until(ws, {"response.done"})
        done = cx.capture.of("response_done")[-1]
        return {
            "pacing": pacing,
            "audio_ms": done.get("audio_ms"),
            "estimated_playback_tail_ms": done.get("estimated_playback_tail_ms"),
            "status": done.get("status"),
            "pacing_dropped_audio_ms": sum(
                r.get("audio_ms", 0) or 0 for r in cx.capture.of("pacing_dropped_audio_ms")
            ),
        }
    finally:
        close_all()


@_isolated
def scenario_h1_pacing() -> dict[str, Any]:
    """H1 fix: with output_pacing=realtime the server holds synthesized audio
    and emits it at ~1x with a lead, so response.done lands near the end of
    playback (small estimated tail) instead of at synthesis end. A barge-in is
    then still able to cancel an ACTIVE response."""
    off = _run_h1_pacing("off", 600)
    on = _run_h1_pacing("realtime", 600)
    off_tail = off.get("estimated_playback_tail_ms")
    on_tail = on.get("estimated_playback_tail_ms")
    return {
        "scenario": "H1p",
        "mechanism": "output pacing makes response.done track real playback",
        "off": off,
        "on": on,
        "tail_reduced": (
            off_tail is not None and on_tail is not None and on_tail < off_tail
        ),
        "before_after": {
            "estimated_playback_tail_ms_off": off_tail,
            "estimated_playback_tail_ms_on": on_tail,
            "audio_ms": off.get("audio_ms"),
        },
    }


def _run_h2_burst(confirm_ms: int) -> dict[str, Any]:
    """One H2 pass with a given barge_in_confirm_ms (0 = old immediate mode)."""
    settings = Settings()
    settings.session.turn_detection.silence_duration_ms = 100
    settings.session.turn_detection.barge_in_confirm_ms = confirm_ms
    # gate the TTS so the response stays active while the burst is processed
    tts = ScenarioTTS(gate=asyncio.Event())
    cx = ScenarioClient().configure(settings=settings, llm=ScenarioLLM(), tts=tts)
    try:
        with cx.make_app() as tc:
            with _connect(tc) as ws:
                ws.receive_json()
                ws.send_json({"type": "response.create"})
                _recv_until(ws, {"response.created"})
                # 60 ms tone (>= start_frames) then silence: speech_started
                # fires, but the burst is below min_speech_ms and is discarded
                ws.send_json({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(60))})
                _recv_until(ws, {"input_audio_buffer.speech_started"})
                ws.send_json({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(300))})
                # in off mode the burst cancels the response (done arrives); in
                # soft mode it is rejected. Wait for done only after releasing
                # the gated TTS; poll the captured lifecycle to know when the
                # burst was resolved before releasing.
                for _ in range(400):
                    if cx.capture.of("vad_burst_discarded"):
                        break
                    time.sleep(0.01)
                tts.gate.set()
                _recv_until(ws, {"response.done"})
        discarded = cx.capture.of("vad_burst_discarded")
        return {
            "burst_discarded": len(discarded) >= 1,
            "burst_outcome": discarded[-1].get("outcome") if discarded else None,
            "cancelled_response_by_burst": (
                discarded[-1].get("response_cancelled_by_burst") if discarded else None
            ),
            "barge_in_rejected": len(cx.capture.of("barge_in_rejected")) >= 1,
            "barge_in_pending": len(cx.capture.of("barge_in_pending")) >= 1,
            "kept_bursts": len(cx.capture.of("vad_burst")),
            # the response the burst hit (a re-answer may follow it)
            "response_done_status": (
                cx.capture.of("response_done")[0].get("status")
                if cx.capture.of("response_done")
                else None
            ),
            "answered_again": any(
                r.get("reason") == "answer_cut_by_noise" for r in cx.capture.of("superseded_turn_answered")
            ),
            "new_turn_stt_calls": len(cx.stt.calls),
        }
    finally:
        close_all()


@_isolated
def scenario_h2() -> dict[str, Any]:
    """H2: a short noise burst during an active response. Before the fix
    (barge_in_confirm_ms=0) the burst kills the response and no new turn
    replaces it; with the soft barge-in default the burst is rejected and the
    response survives with all its audio."""
    before = _run_h2_burst(0)
    after = _run_h2_burst(250)
    return {
        "scenario": "H2",
        "mechanism": "short noise burst during an active response",
        "off": before,
        "on": after,
        # legacy flat keys (off-mode values) kept for existing consumers
        **{k: before[k] for k in ("burst_discarded", "burst_outcome", "cancelled_response_by_burst", "kept_bursts", "response_done_status", "new_turn_stt_calls", "answered_again")},
        "fix_active": (
            after["response_done_status"] == "completed"
            and after["cancelled_response_by_burst"] is False
            and after["barge_in_rejected"] is True
        ),
        "before_after": {
            "response_survives_off": before["response_done_status"],
            "response_survives_on": after["response_done_status"],
        },
    }


@_isolated
def scenario_h7() -> dict[str, Any]:
    """H7: a session.update mid-utterance. Before the fix the detector was
    rebuilt (utterance dropped); after the fix it survives unless the
    turn-detection parameters actually changed."""
    cx = ScenarioClient().configure()
    detector_survived: bool | None = None
    speaking_after: bool | None = None
    reset_logged = False
    with cx.make_app() as tc:
        with _connect(tc) as ws:
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(200))})
            _recv_until(ws, {"input_audio_buffer.speech_started"})
            session = next(iter(tc.app.state.sessions.values()))
            detector_before = session._detector
            ws.send_json({"type": "session.update", "session": {"instructions": "neue Anweisung"}})
            _recv_until(ws, {"session.updated"})
            detector_survived = session._detector is detector_before
            speaking_after = session._detector.speaking
            # a real turn-detection change must rebuild + log vad_reset
            ws.send_json({"type": "session.update", "session": {"turn_detection": {"threshold": 0.8}}})
            _recv_until(ws, {"session.updated"})
            reset_logged = bool(cx.capture.of("vad_reset"))
    return {
        "scenario": "H7",
        "mechanism": "session.update mid-utterance",
        "detector_survived_instructions_update": detector_survived,
        "detector_still_speaking": speaking_after,
        "vad_reset_logged_on_td_change": reset_logged,
        "fix_active": detector_survived is True and speaking_after is True and reset_logged,
    }


@_isolated
def scenario_h5a() -> dict[str, Any]:
    """H5a: a function_call item is appended to the history BEFORE the
    assistant text of the same response — the next LLM request sees the tool
    call, then the assistant text, then the tool result."""
    llm = ScenarioLLM(
        script=[
            {"text": "Ich schaue nach.", "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": "{}"}]},
            {"text": "Fertig."},
        ]
    )
    cx = ScenarioClient().configure(llm=llm)
    with cx.make_app() as tc:
        with _connect(tc) as ws:
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.done"})
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
                }
            )
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.done"})

    seqs = [r.get("role_sequence") for r in cx.capture.of("llm_request")]
    # history order in the second request: assistant text and tool call
    msgs = cx.llm.calls[-1]["messages"] if cx.llm.calls else []
    history = []
    merged_assistant = False
    for m in msgs:
        if m.get("role") == "assistant":
            has_calls = bool(m.get("tool_calls"))
            has_text = bool(m.get("content"))
            if has_calls and has_text:
                history.append("assistant_tool_call+text")
                merged_assistant = True
            elif has_calls:
                history.append("assistant_tool_call")
            else:
                history.append("assistant_text")
        elif m.get("role") == "tool":
            history.append("tool_result")
    return {
        "scenario": "H5a",
        "mechanism": "text + tool_call ordering in history",
        "role_sequences": seqs,
        "history_shape": history,
        "tool_call_before_tool_result": (any("assistant_tool_call" in h for h in history) and "tool_result" in history),
        "assistant_text_kept_in_context": any("assistant_text" in h or "assistant_tool_call+text" in h for h in history),
        # H5a fix: one assistant message carries BOTH content and tool_calls
        "merged_assistant_message": merged_assistant,
    }


@_isolated
def scenario_h5b() -> dict[str, Any]:
    """H5b: a one-off response.instructions replaces the session instructions
    for that response only; the next response uses the session instructions
    again."""
    session_instr = "SPRICH-IRISCH"
    oneoff = "SPRICH-LATEIN"
    cx = ScenarioClient().configure()
    cx.settings.session.instructions = session_instr
    with cx.make_app() as tc:
        with _connect(tc) as ws:
            ws.receive_json()
            ws.send_json({"type": "response.create", "response": {"instructions": oneoff}})
            _recv_until(ws, {"response.done"})
            ws.send_json({"type": "response.create"})
            _recv_until(ws, {"response.done"})

    reserved = cx.capture.of("response_reserved")
    # inspect the system messages actually sent to the LLM
    first_systems = [m["content"] for m in cx.llm.calls[0]["messages"] if m.get("role") == "system"]
    second_systems = [m["content"] for m in cx.llm.calls[1]["messages"] if m.get("role") == "system"]
    return {
        "scenario": "H5b",
        "mechanism": "one-off response.instructions replace session instructions",
        "first_has_oneoff": oneoff in first_systems,
        "first_has_session_instr": session_instr in first_systems,
        "second_has_oneoff": oneoff in second_systems,
        "second_has_session_instr": session_instr in second_systems,
        "oneoff_sha8_present": any(r.get("has_instructions") for r in reserved),
        "oneoff_text_logged": any(oneoff in json.dumps(r) for r in cx.capture.records),
        "replaces_for_one_response_only": (
            (oneoff in first_systems)
            and (session_instr not in first_systems)
            and (session_instr in second_systems)
            and (oneoff not in second_systems)
        ),
    }


SCENARIOS: dict[str, Callable[[], dict[str, Any]]] = {
    "H1": scenario_h1,
    "H1p": scenario_h1_pacing,
    "H2": scenario_h2,
    "H4": scenario_h4,
    "H5a": scenario_h5a,
    "H5b": scenario_h5b,
    "H7": scenario_h7,
}


def run_scenarios(names: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    try:
        for name in names:
            scenario = SCENARIOS.get(name)
            if scenario is None:
                raise SystemExit(f"unknown scenario: {name} (choose from {sorted(SCENARIOS)})")
            results.append(scenario())
    finally:
        for client in list(_ACTIVE_CLIENTS):
            client.close()
    return {"scenarios": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="scenario id to run (repeatable); default all",
    )
    parser.add_argument("--out", default="", help="write metrics JSON here (else stdout)")
    args = parser.parse_args(argv)
    names = args.scenarios or list(SCENARIOS)
    report = run_scenarios(names)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())