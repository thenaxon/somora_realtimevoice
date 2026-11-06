"""Opt-in live WS smoke client: drives the public /v1/realtime protocol
against a running realtimevoice instance (or any compatible server).

Reads the auth token from the instance data dir (server.json / env) or
--token; never prints it. Sends a WAV file as realtime PCM16/24 kHz chunks in
real time plus trailing silence, collects transcript/audio/response events
with timings.

Modes:
  normal: require a completed response with nonempty transcript and audio.
  tool:   register a demo tool, instruct the model to call it, answer the
          call with a nonce result, and require a second COMPLETED response
          whose transcript contains the nonce.
  cancel: after the first audio chunk, cancel the response; require status
          cancelled and no stale transcript/audio deltas for --observe-ms
          after response.done.
  bargein: full lifecycle acceptance on ONE WebSocket. Stream --wav (A) and
          wait for the first actual response audio; then stream --wav-b (B)
          WITHOUT any manual response.cancel — server VAD must interrupt.
          Requires: response A cancelled, no A-id audio/text deltas after A's
          response.done, a new-turn STT transcript, a new completed response
          with PCM, and zero already_active_response errors. Then (unless
          --skip-cancel-turn) a third turn starts, an explicit response.cancel
          is sent after its first audio, and a final turn must still complete
          on the same socket. Use --timeout 240 or larger for this mode.

Exit codes: 0 pass, 1 no response.done at all, 2 setup/transport error,
3 criteria failed (includes backend errors reported by the server).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import struct
import sys
import time
import urllib.parse
import wave

try:
    import websockets
except ImportError:  # pragma: no cover
    print("websockets is required: pip install 'uvicorn[standard]' or websockets", file=sys.stderr)
    sys.exit(2)


def load_token(data_dir: str | None) -> str | None:
    token = os.environ.get("REALTIMEVOICE_AUTH_TOKEN", "").strip()
    if token:
        return token
    if data_dir and os.path.exists(os.path.join(data_dir, "server.json")):
        with open(os.path.join(data_dir, "server.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("auth_token")
    return None


def compose_ws_url(url: str, model: str | None = None) -> str:
    """Query-safe WS URL for live_smoke runs.

    - ``--model`` given: it wins (any existing model= is replaced).
    - ``--model`` omitted: an existing ``?model=`` from the URL is preserved
      (e.g. ``profile:<set>``), otherwise the empty model is appended (the
      server's default set; refused with ``no_set_selected`` when none is
      chosen).
    - Other query parameters are never dropped; the token never goes into
      the URL (it travels via the rtv.bearer subprotocol pair instead).
    """
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    existing = next((v for k, v in query if k == "model"), None)
    chosen = model if model is not None else (existing if existing is not None else "")
    rest = [(k, v) for k, v in query if k != "model"]
    rest.append(("model", chosen))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(rest)))


def ws_subprotocols(token: str) -> list[str]:
    """Browser-style auth pair: the token never appears in the URL, access
    logs, or exceptions."""
    return ["rtv.bearer", "rtv.bearer." + token]


def read_wav_24k_mono(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise SystemExit(f"WAV must be PCM16 (got sample width {width})")
    if channels > 1:
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        frames = [samples[i : i + channels] for i in range(0, len(samples), channels)]
        raw = struct.pack(f"<{len(frames)}h", *(sum(f) // channels for f in frames))
    if rate != 24000:
        n_in = len(raw) // 2
        samples = struct.unpack(f"<{n_in}h", raw)
        n_out = int(n_in * 24000 / rate)
        out = []
        for i in range(n_out):
            pos = i * rate / 24000
            i0 = int(pos)
            frac = pos - i0
            a = samples[min(i0, n_in - 1)]
            b = samples[min(i0 + 1, n_in - 1)]
            out.append(int(a * (1 - frac) + b * frac))
        raw = struct.pack(f"<{len(out)}h", *out)
    return raw


TOOL_NAME = "get_demo_token"
TOOL_SCHEMA = {
    "type": "function",
    "name": TOOL_NAME,
    "description": "Liefert einen einmaligen Demo-Code. Rufe dieses Tool auf und nenne den Code.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class Smoke:
    def __init__(self, args):
        self.args = args
        if self.args.mode == "bargein":
            self.args.wav_b = self.args.wav_b or self.args.wav
        self.events: list[dict] = []
        self.timings: dict[str, float] = {}
        self.audio_chunks = 0
        self.audio_bytes_decoded = 0
        self.tool_answers = 0
        self.pending_tools: list[dict] = []
        self.cancel_sent_at: float | None = None
        self.cancel_target_rid: str | None = None
        self.nonce = secrets.token_hex(4)
        self.done = asyncio.Event()
        # bargein-mode state machine: 1 wait-A-audio, 2 wait-B-done,
        # 3 wait-C-audio/ cancel sent, 4 wait-D-done, 5 finished (skip flag)
        self.phase = 1
        self.created_order: list[str] = []
        self.resp_status: dict[str, str] = {}
        self.resp_audio: dict[str, int] = {}
        self.resp_text: dict[str, int] = {}
        self.barge_trigger = asyncio.Event()
        self.turn2_done = asyncio.Event()
        self.turn3_audio = asyncio.Event()
        self.turn3_cancelled = asyncio.Event()
        self.turn4_done = asyncio.Event()

    def note(self, key: str) -> None:
        if key not in self.timings:
            self.timings[key] = time.monotonic()

    async def run(self) -> int:
        url = self.args.url
        token = self.args.token or load_token(self.args.data_dir)
        if not token:
            print("no auth token found (use --token or --data-dir)", file=sys.stderr)
            return 2
        # query-safe composition: --model wins, else an existing ?model= is
        # preserved, else empty (default set); the token NEVER travels in the
        # URL — it goes via the browser-style subprotocol pair below
        ws_url = compose_ws_url(url, self.args.model)
        print(f"connecting {self.args.url} (mode={self.args.mode}, model={self.args.model!r})")
        try:
            async with websockets.connect(
                ws_url,
                max_size=16 * 1024 * 1024,
                subprotocols=ws_subprotocols(token),
            ) as ws:
                # timing origin for every mode: report offsets are relative
                # to the established socket, never raw monotonic values
                self.note("send_start")
                receiver = asyncio.create_task(self.receive_loop(ws))
                sender = asyncio.create_task(self.send_audio(ws))
                try:
                    # outer bound: even if the sender finishes, the whole run
                    # must end after the timeout
                    await asyncio.wait_for(
                        self.done.wait() if self.args.mode != "cancel" else self.cancel_run_done(),
                        timeout=self.args.timeout,
                    )
                except asyncio.TimeoutError:
                    print(f"TIMEOUT after {self.args.timeout}s")
                finally:
                    receiver.cancel()
                    sender.cancel()
                    try:
                        await receiver
                    except (asyncio.CancelledError, Exception):
                        pass
        except (OSError, websockets.exceptions.InvalidStatus) as exc:
            print(f"connection failed: {exc}", file=sys.stderr)
            return 2
        return self.report()

    async def cancel_run_done(self) -> None:
        """In cancel mode: wait for response.done, then observe stale window."""
        await self.done.wait()
        await asyncio.sleep(self.args.observe_ms / 1000.0)

    async def receive_loop(self, ws) -> None:
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                event["_recv_t"] = time.monotonic()
                self.events.append(event)
                self.handle(ws, event)
        except asyncio.CancelledError:
            pass

    def handle(self, ws, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "error":
            print(f"< error {event.get('error')}")
        elif etype == "input_audio_buffer.speech_stopped":
            self.note("speech_stopped")
        elif etype == "conversation.item.input_audio_transcription.completed":
            self.note("stt_final")
            print(f"< transcript {event.get('transcript', '')!r}")
        elif etype == "response.created":
            self.note("response_created")
            rid = (event.get("response") or {}).get("id")
            if rid:
                self.created_order.append(rid)
                self.resp_status.setdefault(rid, "in_progress")
        elif etype == "response.output_audio_transcript.delta":
            rid = event.get("response_id") or ""
            self.resp_text[rid] = self.resp_text.get(rid, 0) + 1
        elif etype == "response.output_audio_transcript.done":
            print(f"< transcript.done {event.get('transcript', '')!r}")
        elif etype == "response.output_audio.delta":
            rid = event.get("response_id") or ""
            self.resp_audio[rid] = self.resp_audio.get(rid, 0) + 1
            if self.audio_chunks == 0:
                self.note("first_audio")
            self.audio_chunks += 1
            self.audio_bytes_decoded += len(base64.b64decode(event.get("delta", "")))
            if self.args.mode == "cancel" and self.cancel_sent_at is None:
                self.cancel_sent_at = time.monotonic()
                asyncio.get_running_loop().create_task(self.send_cancel(ws))
            elif self.args.mode == "bargein" and self.phase == 1:
                self.barge_trigger.set()
                self.phase = 2
            elif self.args.mode == "bargein" and self.phase == 3:
                self.turn3_audio.set()
        elif etype == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self.note("tool_call")
                print(f"< function_call {item.get('name')} {item.get('arguments')}")
                if self.args.mode == "tool":
                    self.pending_tools.append(item)
        elif etype == "response.done":
            resp = event.get("response") or {}
            status = resp.get("status")
            rid = resp.get("id") or ""
            if rid:
                self.resp_status[rid] = status
            print(f"< response.done status={status}")
            if self.args.mode == "bargein":
                self._bargein_done(status)
            elif self.args.mode == "cancel" and status == "cancelled":
                self.note("cancelled_done")
                self.done.set()
            elif self.args.mode == "tool":
                if self.pending_tools and self.tool_answers == 0:
                    self.note("first_done")
                elif self.tool_answers > 0:
                    self.note("second_done")
                    self.done.set()
                else:
                    self.done.set()
            else:
                self.done.set()

    def _bargein_done(self, status) -> None:
        if status == "completed":
            if self.phase == 2:
                self.turn2_done.set()
                if self.args.skip_cancel_turn:
                    self.phase = 5
                    self.done.set()
                else:
                    self.phase = 3
            elif self.phase == 4:
                self.phase = 5
                self.done.set()
            # completed in other phases: leave to sender timeouts / report
        elif status == "cancelled" and self.phase == 3 and self.cancel_sent_at is not None:
            self.turn3_cancelled.set()
            self.phase = 4

    async def send_cancel(self, ws) -> None:
        # let a little audio through so there is something to interrupt
        await asyncio.sleep(self.args.cancel_after_ms / 1000.0)
        await ws.send(json.dumps({"type": "response.cancel"}))
        self.note("cancel_sent_at")
        print("> response.cancel")

    async def stream_wav(self, ws, path: str, silence_ms: int) -> None:
        pcm = read_wav_24k_mono(path)
        chunk = 24000 * 2 * self.args.chunk_ms // 1000
        total_ms = len(pcm) / 48
        print(f"streaming {total_ms:.0f} ms of audio from {path}")
        for i in range(0, len(pcm), chunk):
            if self.done.is_set():
                return
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + chunk]).decode()})
            )
            await asyncio.sleep(self.args.chunk_ms / 1000.0)
        silence = b"\x00\x00" * (24000 * 2 * self.args.chunk_ms // 1000 // 2)
        for _ in range(int(silence_ms / self.args.chunk_ms)):
            if self.done.is_set():
                return
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
            await asyncio.sleep(self.args.chunk_ms / 1000.0)

    async def _wait(self, ev: asyncio.Event, timeout: float, what: str) -> bool:
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"TIMEOUT waiting for {what}")
            return False

    async def send_audio_bargein(self, ws) -> None:
        # Phase 1: WAV A until the first real response audio comes back.
        await self.stream_wav(ws, self.args.wav, self.args.silence_ms)
        if not await self._wait(self.barge_trigger, 30, "first response audio for turn A"):
            return
        # Phase 2: spoken WAV B — no manual cancel; server VAD must interrupt A.
        self.note("bargein_send")
        print(f"> barge-in: streaming {self.args.wav_b} (no response.cancel)")
        await self.stream_wav(ws, self.args.wav_b, self.args.silence_ms)
        if not await self._wait(self.turn2_done, 45, "completed response after barge-in"):
            return
        if self.args.skip_cancel_turn:
            self.done.set()
            return
        # Phase 3: next turn; explicit response.cancel once its audio flows.
        self.note("cancel_turn_send")
        await self.stream_wav(ws, self.args.wav_b, self.args.silence_ms)
        if not await self._wait(self.turn3_audio, 30, "audio of the turn to cancel"):
            return
        self.cancel_target_rid = self.created_order[-1] if self.created_order else ""
        await ws.send(json.dumps({"type": "response.cancel"}))
        self.cancel_sent_at = time.monotonic()
        self.note("cancel_sent_at")
        print("> response.cancel (explicit)")
        if not await self._wait(self.turn3_cancelled, 15, "cancelled response.done"):
            return
        # Phase 4: the same socket must still carry a full next turn.
        self.note("post_cancel_send")
        await self.stream_wav(ws, self.args.wav_b, self.args.silence_ms)
        await self._wait(self.turn4_done, 45, "final completed response after explicit cancel")
        self.done.set()

    async def send_audio(self, ws) -> None:
        if self.args.mode == "bargein":
            await self.send_audio_bargein(ws)
            return
        pcm = read_wav_24k_mono(self.args.wav)
        chunk = 24000 * 2 * self.args.chunk_ms // 1000
        total_ms = len(pcm) / 48
        print(f"streaming {total_ms:.0f} ms of audio from {self.args.wav}")
        self.note("send_start")

        if self.args.mode == "tool":
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "tools": [TOOL_SCHEMA],
                            "instructions": (
                                f"Wenn der Nutzer spricht, rufe das Tool {TOOL_NAME} auf "
                                "und nenne anschließend den zurückgegebenen Code wörtlich."
                            ),
                        },
                    }
                )
            )
            print(f"> session.update with tool {TOOL_NAME} (nonce {self.nonce})")

        for i in range(0, len(pcm), chunk):
            if self.done.is_set():
                return
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + chunk]).decode()})
            )
            await asyncio.sleep(self.args.chunk_ms / 1000.0)
        silence = b"\x00\x00" * (24000 * 2 * self.args.chunk_ms // 1000 // 2)
        for _ in range(int(self.args.silence_ms / self.args.chunk_ms)):
            if self.done.is_set():
                return
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
            await asyncio.sleep(self.args.chunk_ms / 1000.0)

        # answer pending tool calls and ask for the second response
        for _ in range(60):
            if self.done.is_set() or self.pending_tools:
                break
            await asyncio.sleep(0.1)
        for item in list(self.pending_tools):
            result = json.dumps({"code": f"DEMO-{self.nonce}"})
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": item["call_id"], "output": result},
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))
            self.tool_answers += 1
            print(f"> tool result sent (nonce DEMO-{self.nonce}), response.create")
        self.pending_tools.clear()

    def report(self) -> int:
        t = self.timings
        print("\n--- smoke report ---")
        print(f"mode:                 {self.args.mode}")
        print(f"events received:      {len(self.events)}")
        print(f"audio chunks:         {self.audio_chunks}")
        print(f"decoded PCM bytes:    {self.audio_bytes_decoded}")
        print(f"tool answers sent:    {self.tool_answers}")
        base = t.get("send_start", 0)
        for key in (
            "speech_stopped",
            "stt_final",
            "response_created",
            "first_audio",
            "bargein_send",
            "cancel_turn_send",
            "cancelled_done",
            "cancel_sent_at",
            "post_cancel_send",
        ):
            if key in t:
                print(f"{key}: +{(t[key] - base):.2f}s")
        if "cancel_sent_at" in t and "first_audio" in t:
            print(f"cancel sent {((t['cancel_sent_at'] - t['first_audio']) * 1000):.0f} ms after first audio")

        failures: list[str] = []
        dones = [e for e in self.events if e.get("type") == "response.done"]
        statuses = [(d.get("response") or {}).get("status") for d in dones]
        print(f"response.done statuses: {statuses}")

        transcripts = [
            e.get("transcript", "")
            for e in self.events
            if e.get("type") == "response.output_audio_transcript.done"
        ]

        if self.args.mode == "normal":
            if "completed" not in statuses:
                failures.append(f"expected a completed response, got {statuses}")
            if not transcripts:
                failures.append("no transcript.done received")
            if self.audio_bytes_decoded <= 0:
                failures.append("no audio received")
        elif self.args.mode == "tool":
            if not any(e.get("type") == "response.output_item.done" for e in self.events):
                failures.append("no output items at all")
            tool_seen = any(
                (e.get("item") or {}).get("type") == "function_call" and (e.get("item") or {}).get("name") == TOOL_NAME
                for e in self.events
                if e.get("type") == "response.output_item.done"
            )
            if not tool_seen:
                failures.append(f"no {TOOL_NAME} function_call received")
            if self.tool_answers == 0:
                failures.append("tool callback was never sent")
            if "completed" not in statuses[self.tool_answers :]:
                failures.append(f"no completed second response (statuses={statuses})")
            joined = " ".join(transcripts)
            if self.nonce not in joined and f"DEMO-{self.nonce}" not in joined:
                failures.append(f"nonce {self.nonce} missing from transcripts: {joined!r}")
        elif self.args.mode == "cancel":
            if "cancelled" not in statuses:
                failures.append(f"expected a cancelled response, got {statuses}")
            done_idx = self.events.index(dones[-1])
            stale = [
                e["type"]
                for e in self.events[done_idx + 1 :]
                if e.get("type") in ("response.output_audio.delta", "response.output_audio_transcript.delta")
            ]
            if stale:
                failures.append(f"stale deltas after cancel: {stale}")
            else:
                print(f"no stale deltas for {self.args.observe_ms} ms after cancel")
            if self.audio_chunks == 0:
                failures.append("no audio at all before cancel")
        elif self.args.mode == "bargein":
            self._report_bargein(failures, statuses)

        if failures:
            print("\nFAIL:")
            for f in failures:
                print(f"  - {f}")
            return 3
        print("\nPASS")
        return 0

    def _report_bargein(self, failures: list[str], statuses: list) -> None:
        # VAD barge-in: the FIRST response (turn A) must have been cancelled
        # by speech B — no manual response.cancel involved.
        if not self.created_order:
            failures.append("no response.created at all")
            return
        rid_a = self.created_order[0]
        status_a = self.resp_status.get(rid_a)
        print(f"turn A response {rid_a}: {status_a}")
        if status_a != "cancelled":
            failures.append(
                f"expected turn A cancelled by VAD barge-in (interrupt_response), got {status_a!r}"
            )
        # no audio/text deltas of a cancelled response AFTER its response.done
        for rid, status in self.resp_status.items():
            if status != "cancelled":
                continue
            done_idx = None
            for i, e in enumerate(self.events):
                if e.get("type") == "response.done" and (e.get("response") or {}).get("id") == rid:
                    done_idx = i
            if done_idx is None:
                failures.append(f"cancelled response {rid} has no response.done")
                continue
            stale = [
                e["type"]
                for e in self.events[done_idx + 1 :]
                if e.get("type") in ("response.output_audio.delta", "response.output_audio_transcript.delta")
                and e.get("response_id") == rid
            ]
            if stale:
                failures.append(f"{len(stale)} stale deltas of {rid} after its cancellation")
            else:
                print(f"no stale deltas of {rid} after cancellation")
        # a NEW turn must have been transcribed after the barge-in started
        barge_t = self.timings.get("bargein_send")
        if barge_t is None:
            failures.append("barge-in was never started (no first response audio for turn A)")
        else:
            stt_after = [
                e
                for e in self.events
                if e.get("type") == "conversation.item.input_audio_transcription.completed"
                and e.get("_recv_t", 0) > barge_t
            ]
            if not stt_after:
                failures.append("no new-turn transcript after barge-in")
            else:
                print(f"new-turn transcripts after barge-in: {len(stt_after)}")
        # fresh PCM and a completed response must arrive after the cancellation
        cancel_done_ts = [
            e.get("_recv_t", 0)
            for e in self.events
            if e.get("type") == "response.done" and (e.get("response") or {}).get("status") == "cancelled"
        ]
        if cancel_done_ts:
            after = min(cancel_done_ts)
            audio_after = sum(
                1
                for e in self.events
                if e.get("type") == "response.output_audio.delta" and e.get("_recv_t", 0) > after
            )
            if audio_after == 0:
                failures.append("no response audio after the first cancellation (new turn produced no PCM)")
            else:
                print(f"response audio deltas after cancellation: {audio_after}")
        if statuses.count("completed") < (1 if self.args.skip_cancel_turn else 2):
            failures.append(
                f"expected {'1' if self.args.skip_cancel_turn else '2'} completed response(s) "
                f"(new turn + post-cancel turn), got {statuses}"
            )
        last_done = [
            e for e in self.events if e.get("type") == "response.done"
        ]
        last_done = last_done[-1] if last_done else None
        if not last_done or (last_done.get("response") or {}).get("status") != "completed":
            failures.append(f"final turn did not complete (last status: {statuses[-1] if statuses else None})")
        elif self.resp_audio.get((last_done.get("response") or {}).get("id") or "", 0) == 0:
            failures.append("final completed response carried no audio PCM")
        # the explicit-cancel phase must actually have cancelled something
        if self.cancel_sent_at is not None:
            rid_at_cancel = self.cancel_target_rid or (self.created_order[-1] if self.created_order else "")
            st = self.resp_status.get(rid_at_cancel)
            print(f"explicit cancel target {rid_at_cancel}: {st}")
            if st != "cancelled":
                failures.append(f"explicit response.cancel did not cancel its response (status={st!r})")
        # the busy error must never appear in a healthy lifecycle
        busy = [
            e
            for e in self.events
            if e.get("type") == "error" and (e.get("error") or {}).get("code") == "already_active_response"
        ]
        if busy:
            failures.append(f"received {len(busy)} already_active_response errors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument(
        "--model",
        default=None,
        help="WS model string (e.g. profile:<set> for a set's bargein/cancel check); default: empty (the server's default set). If omitted, an existing ?model= in --url is preserved",
    )
    parser.add_argument("--token", default=None, help="auth token (else env/data dir)")
    parser.add_argument("--data-dir", default=None, help="instance data dir holding server.json")
    parser.add_argument("--wav", required=True, help="input WAV (any rate, mono preferred)")
    parser.add_argument(
        "--wav-b",
        default=None,
        help="bargein mode: speech WAV that interrupts turn A and drives the following turns (default: --wav)",
    )
    parser.add_argument("--mode", choices=["normal", "tool", "cancel", "bargein"], default="normal")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--silence-ms", type=int, default=1200, help="trailing silence to trigger VAD")
    parser.add_argument("--timeout", type=float, default=90.0, help="outer bound for the whole run")
    parser.add_argument("--observe-ms", type=int, default=1500, help="stale-observation window (cancel mode)")
    parser.add_argument("--cancel-after-ms", type=int, default=800, help="delay between first audio and cancel")
    parser.add_argument(
        "--skip-cancel-turn",
        action="store_true",
        help="bargein mode: stop after the VAD barge-in turn (skips the explicit response.cancel phase)",
    )
    args = parser.parse_args()
    if args.mode == "bargein" and args.timeout < 240 and not args.skip_cancel_turn:
        print("hint: bargein mode streams 4 turns; consider --timeout 240 or larger", file=sys.stderr)
    return asyncio.run(Smoke(args).run())


if __name__ == "__main__":
    sys.exit(main())
