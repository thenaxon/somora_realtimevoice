#!/usr/bin/env python3
"""M1 live acceptance harness: two conversation profiles in parallel.

Against a REAL running service, proves that two different `model=` routes bind
two different profile configurations and each produces a transcript + audio:

  Session A: model=voice-route:auto, then a Somora-style session.update with
             audio.output.voice=<profile-a> (the name of the set to bind).
  Session B: model=profile:<profile-b> (resolved before accept).

Both sessions receive the SAME short synthetic German utterance (synthesized
once through POST /api/tts-synthesis with --tts-preview-profile, resampled to
PCM16 24 kHz) followed by 1.5 s silence, streamed in 20 ms chunks. The harness
verifies the two sessions resolve to different route models and each completes
a transcription and an audio response.

Privacy: the token comes from the RTV_TOKEN environment variable (or
--token-env) and is NEVER printed. Transcription content is reduced to length
+ sha8 before touching disk. `summary.json` and `events-metadata.jsonl` carry
event types, ids and timings only — no transcript text, no audio, no token.

Usage:
  RTV_TOKEN=... scripts/two_profile_live.py \
      --base-url https://voice.example:8443 \
      --profile-a clara-reference --profile-b anna-agent \
      --tts-preview-profile tts-anna --out-dir /tmp/m1-live-20260915

Exit codes: 0 pass, 3 assertion failures, 2 setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTTERANCE_TEXT = "Wie spät ist es gerade in Wien?"
TARGET_RATE = 24000
CHUNK_MS = 20
TRAILING_SILENCE_MS = 1500
DEFAULT_TOKEN_ENV = "RTV_TOKEN"
DEFAULT_TIMEOUT = 90.0
# how long a single ws.recv() may stay silent before we loop again; a silent
# stretch is NOT terminal (only a closed connection or response.done is)
RECV_POLL_S = 5.0
TERMINAL_RESPONSE_STATUSES = ("completed", "cancelled", "failed", "incomplete")


# ----------------------------------------------------------------- validation

@dataclass
class HarnessArgs:
    base_url: str
    profile_a: str
    profile_b: str
    tts_preview_profile: str
    out_dir: str
    token_env: str = DEFAULT_TOKEN_ENV
    timeout: float = DEFAULT_TIMEOUT
    chunk_ms: int = CHUNK_MS
    silence_ms: int = TRAILING_SILENCE_MS
    dry_run: bool = False


def validate_args(args: HarnessArgs) -> list[str]:
    """Pure argument validation shared by --dry-run and the live run."""
    problems: list[str] = []
    if not args.base_url or not args.base_url.startswith(("http://", "https://")):
        problems.append("--base-url must start with http:// or https://")
    if not args.profile_a:
        problems.append("--profile-a is required")
    if not args.profile_b:
        problems.append("--profile-b is required")
    if not args.tts_preview_profile:
        problems.append("--tts-preview-profile is required")
    if not args.out_dir:
        problems.append("--out-dir is required")
    if args.profile_a and args.profile_b and args.profile_a == args.profile_b:
        problems.append("--profile-a and --profile-b must differ")
    if args.chunk_ms <= 0:
        problems.append("--chunk-ms must be positive")
    if args.silence_ms < 0:
        problems.append("--silence-ms must be non-negative")
    return problems


def ws_url_for(base_url: str, model: str) -> str:
    parts = urllib.parse.urlsplit(base_url.rstrip("/"))
    scheme = "wss" if parts.scheme in ("https", "wss") else "ws"
    path = parts.path.rstrip("/") + "/v1/realtime"
    query = urllib.parse.urlencode({"model": model})
    return urllib.parse.urlunsplit((scheme, parts.netloc, path, query, ""))


def http_url_for(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def resample_to_pcm16(raw: bytes, src_rate: int, channels: int, dst_rate: int = TARGET_RATE) -> bytes:
    """Minimal linear resampler + downmix to mono PCM16 (mirrors audio.py)."""
    import struct

    if channels and channels > 1:
        samples = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
        frames = [samples[i : i + channels] for i in range(0, len(samples), channels)]
        raw = struct.pack(f"<{len(frames)}h", *(sum(f) // channels for f in frames))
    if src_rate == dst_rate or not raw:
        return raw
    n_in = len(raw) // 2
    samples = struct.unpack(f"<{n_in}h", raw)
    if n_in == 0:
        return b""
    n_out = int(round(n_in * dst_rate / src_rate))
    out = []
    for i in range(n_out):
        pos = i * src_rate / dst_rate
        i0 = int(pos)
        frac = pos - i0
        a = samples[min(i0, n_in - 1)]
        b = samples[min(i0 + 1, n_in - 1)]
        out.append(int(a * (1 - frac) + b * frac))
    return struct.pack(f"<{len(out)}h", *out)


def wav_to_pcm16_24k(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, channels, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"WAV must be PCM16 (got sample width {width})")
    return resample_to_pcm16(raw, rate, channels, TARGET_RATE)


def chunk_pcm(pcm: bytes, chunk_ms: int, rate: int = TARGET_RATE):
    size = max(2, int(rate * 2 * chunk_ms / 1000))
    for i in range(0, len(pcm), size):
        yield pcm[i : i + size]


# --------------------------------------------------------------- summary build

def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def build_summary(
    *,
    session_a: dict[str, Any],
    session_b: dict[str, Any],
    profile_a: str,
    profile_b: str,
    utterance_chars: int,
) -> dict[str, Any]:
    """Pure summary construction (unit-tested with fake session results).

    Each session result is a metadata dict:
      {session_id, route, transcripts: [str], transcript_sha8: [..],
       response_status, audio_ms, timings: {...}, errors: [...],
       has_route: bool, has_transcript: bool, has_audio: bool}
    """
    sessions = {"A": session_a, "B": session_b}
    passed = all(s.get("has_route") and s.get("has_transcript") and s.get("has_audio") for s in sessions.values())
    route_a = (session_a.get("route") or {})
    route_b = (session_b.get("route") or {})
    model_a = _route_model(route_a)
    model_b = _route_model(route_b)
    routes_differ = bool(model_a and model_b and model_a != model_b)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "utterance_chars": utterance_chars,
        "profiles": {"a": profile_a, "b": profile_b},
        "sessions": sessions,
        "route_models": {"a": model_a, "b": model_b},
        "routes_differ": routes_differ,
        "passed": bool(passed and routes_differ),
    }


def _route_model(route: dict[str, Any]) -> str:
    for key in ("llm_model", "stt_model", "tts_model"):
        value = route.get(key)
        if value:
            return str(value)
    return ""


# --------------------------------------------------------------- live harness

@dataclass
class SessionState:
    label: str
    session_id: str = ""
    route: dict[str, Any] = field(default_factory=dict)
    transcripts: list[str] = field(default_factory=list)
    response_status: str = ""
    audio_ms: float = 0.0
    audio_bytes: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    event_metadata: list[dict[str, Any]] = field(default_factory=list)
    # set once a response.done event has been seen (terminal for collection)
    response_done: bool = False

    def result(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "route": self.route,
            "transcript_chars": [len(t) for t in self.transcripts],
            "transcript_sha8": [sha8(t) for t in self.transcripts],
            "response_status": self.response_status,
            "audio_ms": round(self.audio_ms, 1),
            "audio_bytes": self.audio_bytes,
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "errors": self.errors,
            "has_route": bool(self.route),
            "has_transcript": bool(self.transcripts),
            "has_audio": self.audio_bytes > 0,
        }

    def note(self, event_type: str, **meta: Any) -> None:
        entry = {"t": round(time.monotonic(), 3), "type": event_type}
        entry.update(meta)
        self.event_metadata.append(entry)


async def fetch_utterance(base_url: str, token: str, profile: str, text: str, timeout: float) -> bytes:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            http_url_for(base_url, "/api/tts-synthesis"),
            headers={"Authorization": f"Bearer {token}"},
            json={"profile_id": profile, "text": text, "text_id": "m1-two-profile"},
        )
        response.raise_for_status()
        return response.content


def somora_initial_update(voice: str, language: str = "de") -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "Du bist ein Sprachassistent. Antworte kurz.",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": TARGET_RATE},
                    "transcription": {"model": "whisper-1", "language": language},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.4,
                        "prefix_padding_ms": 200,
                        "silence_duration_ms": 420,
                        "interrupt_response": True,
                    },
                },
                "output": {"voice": voice},
            },
            "tool_choice": "auto",
        },
    }


def _apply_event(state: SessionState, msg: dict[str, Any]) -> None:
    etype = msg.get("type", "")
    now = time.monotonic()
    if etype == "session.created":
        session = msg.get("session") or {}
        state.session_id = session.get("id", "") or state.session_id
        route = (session.get("x_realtimevoice") or {}).get("route")
        if route:
            state.route = route
        state.note("session.created")
    elif etype == "session.updated":
        session = msg.get("session") or {}
        route = (session.get("x_realtimevoice") or {}).get("route")
        if route:
            state.route = route
        state.note("session.updated", mode=(route or {}).get("mode"))
    elif etype == "input_audio_buffer.speech_stopped":
        state.timings["speech_stopped"] = now
        state.note("speech_stopped")
    elif etype == "conversation.item.input_audio_transcription.completed":
        text = str(msg.get("transcript") or "")
        state.transcripts.append(text)
        if "speech_stopped" in state.timings and "speech_to_transcript_ms" not in state.timings:
            state.timings["speech_to_transcript_ms"] = (now - state.timings["speech_stopped"]) * 1000.0
        state.note("transcription.completed", chars=len(text), sha8=sha8(text))
    elif etype == "response.output_audio.delta":
        delta = msg.get("delta") or ""
        if delta:
            if "first_audio" not in state.timings:
                state.timings["first_audio"] = now
                if "transcript_at" in state.timings:
                    state.timings["transcript_to_first_audio_ms"] = (
                        now - state.timings["transcript_at"]
                    ) * 1000.0
            state.timings["last_audio"] = now
            state.audio_bytes += len(base64.b64decode(delta))
            state.audio_ms += len(base64.b64decode(delta)) / 2 / TARGET_RATE * 1000.0
        state.note("response.output_audio.delta")
    elif etype == "response.done":
        state.response_status = (msg.get("response") or {}).get("status") or state.response_status
        state.response_done = True
        state.note("response.done", status=state.response_status)
    elif etype == "error":
        code = (msg.get("error") or {}).get("code", "")
        state.errors.append(code)
        state.note("error", code=code)
    if "transcript_at" not in state.timings and state.transcripts:
        state.timings["transcript_at"] = now
    if "first_audio" in state.timings and "last_audio" in state.timings:
        state.timings["first_to_last_audio_ms"] = (
            state.timings["last_audio"] - state.timings["first_audio"]
        ) * 1000.0


async def collect_response(
    ws: Any,
    state: SessionState,
    *,
    deadline: float,
    poll_s: float = RECV_POLL_S,
    clock: Any = time.monotonic,
) -> str:
    """Collect events until the response is terminal or the overall deadline.

    A silent stretch longer than `poll_s` is NOT terminal: a slower LLM may
    think for many seconds before the first audio, so a recv timeout merely
    loops until the overall deadline. Stop only when the peer closes/errors
    (`asyncio.wait_for` raising anything but TimeoutError) or when a terminal
    response.done arrives — including a completed response whose audio already
    streamed (we never stop merely because audio has arrived).
    """
    while clock() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, min(poll_s, deadline - clock())))
        except asyncio.TimeoutError:
            continue  # silence is allowed; keep waiting until the deadline
        except Exception:
            return "closed"
        msg = json.loads(raw)
        _apply_event(state, msg)
        if state.response_done and state.response_status in TERMINAL_RESPONSE_STATUSES:
            return "done"
    return "deadline"


async def run_session(
    *,
    base_url: str,
    token: str,
    label: str,
    model: str,
    pcm: bytes,
    voice: str | None,
    chunk_ms: int,
    silence_ms: int,
    timeout: float,
) -> SessionState:
    import websockets

    state = SessionState(label=label)
    ws_url = ws_url_for(base_url, model)
    async with websockets.connect(
        ws_url,
        subprotocols=["rtv.bearer", f"rtv.bearer.{token}"],
        max_size=None,
        open_timeout=timeout,
    ) as ws:
        # initial update (A resolves its late-bound route here; B is already
        # resolved, so the update is optional and only carries the voice for A)
        if voice is not None:
            await ws.send(json.dumps(somora_initial_update(voice)))
        # stream the utterance + trailing silence in 20 ms chunks
        silence = bytes(2 * int(TARGET_RATE * silence_ms / 1000))
        started = time.monotonic()
        for chunk in chunk_pcm(pcm, chunk_ms):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))
            await asyncio.sleep(chunk_ms / 1000.0)
        for chunk in chunk_pcm(silence, chunk_ms):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))
            await asyncio.sleep(chunk_ms / 1000.0)
        # collect until response.done or the overall deadline (silence is fine)
        deadline = time.monotonic() + timeout
        outcome = await collect_response(ws, state, deadline=deadline)
        state.note("collect.finished", outcome=outcome)
        state.timings["stream_ms"] = (time.monotonic() - started) * 1000.0
    return state


async def async_main(args: HarnessArgs, token: str, out_dir: Path) -> dict[str, Any]:
    wav = await fetch_utterance(args.base_url, token, args.tts_preview_profile, UTTERANCE_TEXT, args.timeout)
    pcm = wav_to_pcm16_24k(wav)

    state_a, state_b = await asyncio.gather(
        run_session(
            base_url=args.base_url, token=token, label="A",
            model="voice-route:auto", pcm=pcm, voice=args.profile_a,
            chunk_ms=args.chunk_ms, silence_ms=args.silence_ms, timeout=args.timeout,
        ),
        run_session(
            base_url=args.base_url, token=token, label="B",
            model=f"profile:{args.profile_b}", pcm=pcm, voice=None,
            chunk_ms=args.chunk_ms, silence_ms=args.silence_ms, timeout=args.timeout,
        ),
    )
    summary = build_summary(
        session_a=state_a.result(),
        session_b=state_b.result(),
        profile_a=args.profile_a,
        profile_b=args.profile_b,
        utterance_chars=len(UTTERANCE_TEXT),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = []
    for state in (state_a, state_b):
        metadata.extend({"session": state.label, **entry} for entry in state.event_metadata)
    (out_dir / "events-metadata.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in metadata) + "\n", encoding="utf-8"
    )
    return summary


def create_out_dir(out_dir: str) -> Path:
    """A NEW unique directory; fail if it already exists (never clobber)."""
    path = Path(out_dir)
    if path.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    path.mkdir(parents=True, exist_ok=False)
    os.chmod(path, 0o700)
    return path


def parse_args(argv: list[str] | None = None) -> HarnessArgs:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="running service base URL (http(s))")
    parser.add_argument("--profile-a", required=True, help="set name for session A (voice-route:auto, bound by its voice)")
    parser.add_argument("--profile-b", required=True, help="set id for session B (profile:<id>)")
    parser.add_argument("--tts-preview-profile", required=True, help="tts profile id for /api/tts-synthesis")
    parser.add_argument("--out-dir", required=True, help="NEW unique output directory (must not exist)")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help="env var holding the bearer token")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS)
    parser.add_argument("--silence-ms", type=int, default=TRAILING_SILENCE_MS)
    parser.add_argument("--dry-run", action="store_true", help="validate args only; no network")
    parsed = parser.parse_args(argv)
    return HarnessArgs(
        base_url=parsed.base_url,
        profile_a=parsed.profile_a,
        profile_b=parsed.profile_b,
        tts_preview_profile=parsed.tts_preview_profile,
        out_dir=parsed.out_dir,
        token_env=parsed.token_env,
        timeout=parsed.timeout,
        chunk_ms=parsed.chunk_ms,
        silence_ms=parsed.silence_ms,
        dry_run=parsed.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)

    problems = validate_args(args)
    if problems:
        for problem in problems:
            print(f"SETUP ERROR: {problem}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("dry-run OK: arguments valid")
        print(f"  session A: {ws_url_for(args.base_url, 'voice-route:auto')} voice={args.profile_a!r}")
        print(f"  session B: {ws_url_for(args.base_url, f'profile:{args.profile_b}')}")
        print(f"  synthesis: {http_url_for(args.base_url, '/api/tts-synthesis')} profile={args.tts_preview_profile!r}")
        print(f"  out-dir:   {args.out_dir}")
        return 0

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        print(f"SETUP ERROR: token env {args.token_env} is empty", file=sys.stderr)
        return 2
    try:
        out_dir = create_out_dir(args.out_dir)
    except FileExistsError as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        summary = asyncio.run(async_main(args, token, out_dir))
    except Exception as exc:  # noqa: BLE001
        message = str(exc).replace(token, "[REDACTED]")
        (out_dir / "summary.json").write_text(json.dumps({"error": message, "passed": False}, indent=2), encoding="utf-8")
        print(f"HARNESS ERROR: {message}", file=sys.stderr)
        return 2

    print(json.dumps({k: summary[k] for k in ("route_models", "routes_differ", "passed")}, indent=2))
    return 0 if summary["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())