#!/usr/bin/env python3
"""Throwaway realtimevoice instance + fake OpenAI-compatible backends for the
Somora client harness (scripts/somora_client_harness.mjs).

Starts, in one process:

  * an HTTP fake backend that speaks the three OpenAI-compatible roles
    (``/audio/transcriptions``, ``/chat/completions`` SSE,
    ``/audio/speech`` streaming WAV) with a small control API so the harness
    can script the next STT transcript and LLM behaviour deterministically,
  * a realtimevoice app on a free loopback port with those fake backends,
    one enabled set (native voice) per ``--voice``, named like the voice, so
    ``voice-route:auto`` + the requested voice resolves the set of that name.

It writes a 0600 token file (for the harness ``--key-file``) and a
``<token>.config.json`` next to it holding the non-secret wiring the harness
needs (ws URL, model, profile ids, backend control URL). Then it serves until
SIGINT/SIGTERM.

Nothing here prints the token or any secret. No private paths are embedded; the
data dir is a fresh temp dir and the token is random per run.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import secrets
import signal
import socket
import stat
import struct
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

AUDIO_RATE = 24000


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tone_pcm(duration_ms: int, freq: float = 440.0, amp: int = 2600) -> bytes:
    n = int(AUDIO_RATE * duration_ms / 1000)
    return struct.pack(
        f"<{n}h",
        *(int(amp * math.sin(2 * math.pi * freq * i / AUDIO_RATE)) for i in range(n)),
    )


def pcm_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------- fake backend


class FakeBackend:
    """Control state + handler shared across requests.

    The harness drives it through ``/__control/...``:

      POST /__control/script  {"stt": "...", "llm": [{...}, ...],
                               "tts_ms": 900}
      GET  /__control/records

    An LLM script entry is ``{"text": "..."}``, ``{"tool_call": {"name": ...,
    "arguments": "..."}}`` or ``{"error": "..."}``; when the script is empty a
    fixed default text is returned. STT transcript is the same for every call
    until the script changes (fail-closed determinism).
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stt_text = "Hallo, wie geht es dir heute?"
        self.llm_script: list[dict] = []
        self.tts_ms = 900
        # S2 live fix: when set, an LLM script entry that asks for a tool call
        # only fires if the request's system instructions contain this text —
        # the fake stands in for a real model that only consults when the
        # consult-policy line is there. Empty = old always-on behaviour.
        self.require_consult_policy = ""
        self.records: list[dict] = []

    def set_script(self, payload: dict) -> None:
        with self.lock:
            if "stt" in payload:
                self.stt_text = str(payload["stt"])
            if "llm" in payload:
                self.llm_script = list(payload["llm"])
            if "tts_ms" in payload:
                self.tts_ms = int(payload["tts_ms"])
            if "require_consult_policy" in payload:
                self.require_consult_policy = str(payload["require_consult_policy"] or "")

    def consult_policy_present(self, body: dict) -> bool:
        """True when the request's instructions ask for consults (or when no
        policy gate is configured, keeping the pre-S2 behaviour)."""
        if not self.require_consult_policy:
            return True
        policy = self.require_consult_policy
        # The policy text differs per value; match the live-line fragments
        # buildVoiceInstructions emits (persona.ts). `always` is the strictest:
        # "Everything factual and every request to act goes through".
        needles = {
            "always": "Everything factual and every request to act goes through",
            "substantive": "Anything about your work",
            "auto": "whenever the answer needs your files",
        }
        needle = needles.get(policy, policy)
        for message in body.get("messages", []) or []:
            content = message.get("content")
            if isinstance(content, str) and needle in content:
                return True
        return False

    def pop_llm(self, body: dict | None = None) -> dict:
        with self.lock:
            if not self.llm_script:
                return {"text": "Guten Tag! Wie kann ich Ihnen helfen?"}
            entry = self.llm_script[0]
            if entry.get("tool_call") and body is not None and not self.consult_policy_present(body):
                # The model was not told to consult: answer directly instead.
                self.llm_script.pop(0)
                return {"text": "Ich antworte lieber direkt ohne Werkzeug."}
            self.llm_script.pop(0)
            return entry

    def pop_stt(self) -> str:
        with self.lock:
            return self.stt_text

    def get_tts_ms(self) -> int:
        with self.lock:
            return self.tts_ms

    def record(self, entry: dict) -> None:
        with self.lock:
            self.records.append(entry)


def make_handler(state: FakeBackend):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 without keep-alive: closing the connection is what ends an
        # SSE stream for the OpenAI-compatible adapter (same as the proven
        # ui_functional_e2e fake backend).
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):  # noqa: D401 - silence
            pass

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.endswith("/models"):
                ids = ["harness-stt", "harness-llm", "harness-tts"]
                self._json(200, {"data": [{"id": i} for i in ids]})
            elif self.path.startswith("/__control/records"):
                with state.lock:
                    self._json(200, {"records": list(state.records)})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            raw = self._body()
            if self.path.startswith("/__control/script"):
                try:
                    payload = json.loads(raw.decode() or "{}")
                except ValueError:
                    self._json(400, {"error": {"message": "bad json"}})
                    return
                state.set_script(payload)
                self._json(200, {"ok": True})
                return
            if self.path.endswith("/audio/transcriptions"):
                text = state.pop_stt()
                state.record({"role": "stt", "wav_ok": b"RIFF" in raw})
                self._json(200, {"text": text})
                return
            if self.path.endswith("/chat/completions"):
                try:
                    chat_body = json.loads(raw.decode() or "{}")
                except ValueError:
                    chat_body = {}
                behavior = state.pop_llm(chat_body)
                state.record(
                    {
                        "role": "llm",
                        "tool_call": bool(behavior.get("tool_call")),
                        "text_len": len(behavior.get("text", "")),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                if behavior.get("error"):
                    self.wfile.write(
                        f'data: {json.dumps({"error": {"message": behavior["error"]}})}\n\n'.encode()
                    )
                elif behavior.get("tool_call"):
                    tc = behavior["tool_call"]
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tc.get("id", "call_harness"),
                                            "type": "function",
                                            "function": {
                                                "name": tc["name"],
                                                "arguments": tc.get("arguments", "{}"),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                else:
                    text = behavior.get("text", "Guten Tag!")
                    for i in range(0, len(text), 24):
                        chunk = {
                            "choices": [
                                {"delta": {"content": text[i : i + 24]}, "finish_reason": None}
                            ]
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(
                        f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'.encode()
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if self.path.endswith("/audio/speech"):
                total_ms = state.get_tts_ms()
                chunk_ms = 100
                state.record({"role": "tts", "ms": total_ms})
                body = pcm_wav(tone_pcm(total_ms))
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                # Header first, then PCM in real-time-paced chunks so audio
                # genuinely streams while the harness injects a barge-in.
                header_len = min(44, len(body))
                self.wfile.write(body[:header_len])
                self.wfile.flush()
                offset = header_len
                while offset < len(body):
                    step = min(len(body) - offset, 4800)  # 100 ms @ 24 kHz PCM16
                    self.wfile.write(body[offset : offset + step])
                    self.wfile.flush()
                    offset += step
                    time.sleep(chunk_ms / 1000.0)
                return
            self._json(404, {"error": {"message": "not found"}})

    return Handler


# ------------------------------------------------------------------ instance


class Instance:
    def __init__(self) -> None:
        self.state = FakeBackend()
        self.server: ThreadingHTTPServer | None = None
        self.uvicorn_server = None

    def start_backend(self) -> int:
        port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.state))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return port

    def build_settings(
        self, backend_port: int, profile_ids: list[str], languages: list[str], output_pacing: str = "off"
    ):
        from realtimevoice.config import (
            BackendProfile,
            ConversationProfile,
            ProfileSpec,
            Settings,
            VoiceSelection,
        )
        settings = Settings()
        base = f"http://127.0.0.1:{backend_port}/v1"
        settings.session.language = languages[0]

        # one native-voice set per requested voice id, named like the voice
        for index, profile_id in enumerate(profile_ids):
            self._add_backend_profiles(settings, BackendProfile, profile_id, base)
            spec = ProfileSpec(
                stt_profile=f"{profile_id}-stt",
                llm_profile=f"{profile_id}-llm",
                tts_profile=f"{profile_id}-tts",
                voice=VoiceSelection(kind="native", native_voice=f"{profile_id}-voice"),
                language=languages[index] if index < len(languages) else languages[0],
                output_pacing=output_pacing,
            )
            now = time.time()
            settings.conversation_profiles[profile_id] = ConversationProfile(
                id=profile_id,
                display_name=f"Harness {profile_id}",
                spec=spec,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        return settings

    @staticmethod
    def _add_backend_profiles(settings, BackendProfile, profile_id: str, base: str) -> None:
        settings.profiles["stt"][f"{profile_id}-stt"] = BackendProfile(
            id=f"{profile_id}-stt", role="stt", display_name="Harness STT",
            base_url=base, model="harness-stt",
        )
        settings.profiles["llm"][f"{profile_id}-llm"] = BackendProfile(
            id=f"{profile_id}-llm", role="llm", display_name="Harness LLM",
            base_url=base, model="harness-llm",
        )
        settings.profiles["tts"][f"{profile_id}-tts"] = BackendProfile(
            id=f"{profile_id}-tts", role="tts", display_name="Harness TTS",
            base_url=base, model="harness-tts",
            voice=f"{profile_id}-voice",
            capabilities={"voices": [f"{profile_id}-voice"], "voice_cloning": "none"},
        )

    def start_app(self, settings, data_dir: str, token: str) -> int:
        import uvicorn

        from realtimevoice.app import create_app
        from realtimevoice.config import SettingsStore

        store = SettingsStore(data_dir=data_dir, settings=settings)
        app = create_app(settings_store=store, auth_token=token)
        port = free_port()
        self.uvicorn_server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        threading.Thread(target=self.uvicorn_server.run, daemon=True).start()
        import urllib.error
        import urllib.request

        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.5)
                break
            except urllib.error.HTTPError:
                break  # answered (401 without the token): the server is up
            except Exception:
                time.sleep(0.05)
        return port


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Throwaway instance + fake backends for the Somora harness")
    p.add_argument("--voice", action="append", required=True,
                   help="conversation profile id / voice-routing key; repeat for the reconnect scenario")
    p.add_argument("--language", action="append", default=None,
                   help="language per --voice (default de)")
    p.add_argument("--token-out", required=True, help="path to write the 0600 bearer token")
    p.add_argument("--config-out", default="", help="path for the non-secret harness config (default <token>.config.json)")
    p.add_argument("--data-dir", default="", help="instance data dir (default a fresh temp dir)")
    p.add_argument("--output-pacing", choices=("off", "realtime"), default="off",
                   help="H1 output pacing for the harness profile (default off)")
    args = p.parse_args(argv)

    import tempfile

    data_dir = args.data_dir or tempfile.mkdtemp(prefix="rtv-somora-harness-")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    languages = args.language or ["de"] * len(args.voice)

    inst = Instance()
    backend_port = inst.start_backend()
    settings = inst.build_settings(backend_port, list(args.voice), languages, args.output_pacing)
    token = secrets.token_urlsafe(32)
    port = inst.start_app(settings, data_dir, token)

    token_path = Path(args.token_out)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n")
    os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)

    config_path = Path(args.config_out) if args.config_out else Path(str(token_path) + ".config.json")
    config = {
        "ws_url": f"ws://127.0.0.1:{port}/v1/realtime",
        "http_url": f"http://127.0.0.1:{port}",
        "model": "voice-route:auto",
        "voice": args.voice[0],
        "alt_voice": args.voice[1] if len(args.voice) > 1 else "",
        "language": languages[0],
        "backend_control_url": f"http://127.0.0.1:{backend_port}/__control",
        "data_dir": data_dir,
        "output_pacing": args.output_pacing,
    }
    config_path.write_text(json.dumps(config, indent=2))
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)

    print(json.dumps(config), flush=True)
    print(f"harness instance ready on port {port} (backend {backend_port})", file=sys.stderr, flush=True)

    stop = threading.Event()

    def _handle(signum, frame):  # noqa: ANN001
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    while not stop.is_set():
        time.sleep(0.2)

    if inst.uvicorn_server is not None:
        inst.uvicorn_server.should_exit = True
    if inst.server is not None:
        inst.server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))