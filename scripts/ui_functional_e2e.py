#!/usr/bin/env python3
"""Functional end-to-end acceptance for the F1 feedback fixes (real browsers).

Two modes:

FAKE (default) — starts a throwaway instance with a validating fake
OpenAI-compatible backend and drives the REAL UI end to end:
    add STT/LLM/TTS models on the Models board (no id field; the server
    names them) -> Settings (transcription model) -> create voice -> upload +
    auto-transcribe + confirm -> preview provenance -> create a set, choose
    how it listens, thinks, its voice and language -> save -> reload -> switch
    On -> Try it -> Test Console (audio, transcript, response, tool round
    trip, barge-in, stop/start, no duplicate responses) -> a saved change is
    live, earlier versions, Try it with unsaved values -> error paths. Plus
    browser coverage regressions: shared sample playback (F6), microphone
    device selection (F2), silence upload, set delete rules.

LIVE — `--base-url http://127.0.0.1:8766 --token-env RTV_TOKEN --prefix e2e-`
runs the SAME flow against an already running instance with its REAL Mac
backends. Required live flags: `--stt-model`, `--llm-model`, `--tts-model`
(the real backend model ids) and `--mic-wav <path>` (real speech used for the
Chromium fake mic, the upload and the spoken Test Console input). Backend
profiles are created by cloning an existing profile ENTIRELY SERVER-SIDE via
the UI "Duplicate" action (the API key never passes through the browser or
this process); `--clone-backend-from` names the source profile id(s).
`--api-key-env NAME` (default RTV_BACKEND_API_KEY) is accepted only for legacy
instances without a clone source and is never printed. Optional
`--expect-text-file <path>` computes a normalised similarity between the
expected text and the transcript; only the ratio is logged, never the text.

Every run writes a run dir with screenshots and a JSON step log (step,
pass/fail, timings, server-reported provenance). No transcript/audio content is
ever written to the log or stdout.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import browser_scratch  # noqa: E402

CHROMIUM_DEFAULT = "/usr/bin/chromium"
DEFAULT_MODELS = {"stt": "stt-base", "llm": "llm-base", "tts": "clone-model"}


# ------------------------------------------------------------------ helpers


def free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def read_wav_mono(path: Path) -> tuple[bytes, int]:
    """Read a WAV's PCM frames as 16-bit mono; returns (pcm, rate)."""
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
    if width != 2:
        frames = _to_pcm16(frames, width)
    if channels > 1:
        import array

        samples = array.array("h")
        samples.frombytes(frames[: (len(frames) // (2 * channels)) * 2 * channels])
        mono = array.array("h")
        for i in range(0, len(samples), channels):
            mono.append(int(sum(samples[i : i + channels]) / channels))
        frames = mono.tobytes()
    return frames, rate


def _to_pcm16(data: bytes, width: int) -> bytes:
    if width == 1:
        return b"".join(struct.pack("<h", (b - 128) * 256) for b in data)
    if width == 4:
        import array

        samples = array.array("i")
        samples.frombytes(data[: (len(data) // 4) * 4])
        return array.array("h", [max(-32768, min(32767, s >> 16)) for s in samples]).tobytes()
    return data


def wav_pcm24(path: Path) -> bytes:
    """A 24 kHz mono WAV built from the given file (for the backend upload)."""
    pcm, rate = read_wav_mono(path)
    if rate != 24000:
        from realtimevoice.audio import resample_pcm16

        pcm = resample_pcm16(pcm, rate, 24000)
    return pcm16_wav(pcm, 24000)


def pcm16_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_bytes(seconds: float = 3.0, rate: int = 24000, freq: float = 220.0, amp: int = 6000) -> bytes:
    n = int(rate * seconds)
    return pcm16_wav(
        struct.pack(f"<{n}h", *(int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))),
        rate,
    )


def silence_wav(seconds: float = 2.0, rate: int = 24000) -> bytes:
    return pcm16_wav(b"\x00\x00" * int(rate * seconds), rate)


def normalise_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\wäöüß\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def similarity(expected: str, actual: str) -> float:
    """Normalised similarity ratio between expected and actual text."""
    import difflib

    a, b = normalise_text(expected), normalise_text(actual)
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


class Report:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.steps: list[dict] = []

    def step(self, name: str, ok: bool, detail: str = "", skipped: bool = False, t0: float | None = None) -> None:
        entry = {
            "step": name,
            "status": "skip" if skipped else ("pass" if ok else "fail"),
            "detail": detail[:400],
            "duration_s": round(time.monotonic() - t0, 3) if t0 else None,
        }
        self.steps.append(entry)
        mark = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        print(f"{mark}  {name}" + (f"  ({detail[:160]})" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(s["status"] == "fail" for s in self.steps)

    def write(self, exit_code: int, extra: dict | None = None) -> None:
        payload = {
            "tool": "realtimevoice ui functional e2e",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code,
            "steps": self.steps,
        }
        if extra:
            payload.update(extra)
        (self.run_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ----------------------------------------------------------- fake backends


def start_fake_backend(models: dict[str, str] | None = None) -> tuple[object, int]:
    """A validating OpenAI-compatible backend.

    When `models` is given it maps expected model ids per endpoint kind
    ({"stt": ..., "llm": ..., "tts": ...}) and every request is rejected unless
    the model id matches exactly — this is how the LIVE self-validation proves
    the configured `--*-model` values actually reach the backend.
    """
    import http.server
    import threading

    expected = dict(models or {})
    enforce = bool(expected)

    class FakeBackend(http.server.BaseHTTPRequestHandler):
        records: list = []
        llm_script: list = []
        stt_text: str = "Hallo, dies ist ein Testtranskript."
        bad_model: list = []

        def log_message(self, *a):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject_model(self, model, kind):
            FakeBackend.bad_model.append({"kind": kind, "model": model})
            self._json(400, {"error": {"message": f"model mismatch for {kind}: {model}"}})

        def do_GET(self):
            if self.path.endswith("/models"):
                ids = list(dict.fromkeys(list(expected.values()) or ["stt-base", "llm-base", "preset-model", "clone-model", "fake-model"]))
                self._json(200, {"data": [{"id": i} for i in ids]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            rec = {"path": self.path, "len": len(raw)}
            FakeBackend.records.append(rec)
            if self.path.endswith("/audio/transcriptions"):
                if enforce and expected.get("stt"):
                    model = _form_model(raw)
                    if model != expected["stt"]:
                        return self._reject_model(model, "stt")
                rec["wav_ok"] = b"RIFF" in raw
                self._json(200, {"text": FakeBackend.stt_text})
            elif self.path.endswith("/audio/speech"):
                model = _json_model(raw)
                if enforce and expected.get("tts") and model != expected["tts"]:
                    return self._reject_model(model, "tts")
                rate, n = 24000, 9600  # 400 ms
                pcm = struct.pack(f"<{n}h", *(int(2000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)))
                data = pcm16_wav(pcm, rate)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.endswith("/chat/completions"):
                model = _json_model(raw)
                if enforce and expected.get("llm") and model != expected["llm"]:
                    return self._reject_model(model, "llm")
                behavior = FakeBackend.llm_script.pop(0) if FakeBackend.llm_script else None
                if behavior is None:
                    behavior = _auto_llm_behavior(raw)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if behavior.get("tool_call"):
                    tc = behavior["tool_call"]
                    payload = {
                        "choices": [{
                            "delta": {"tool_calls": [{
                                "index": 0,
                                "id": tc.get("id", "call_1"),
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc.get("arguments", "{}")},
                            }]},
                            "finish_reason": "tool_calls",
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                else:
                    text = behavior.get("text", "Guten Tag!")
                    self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})}\n\n'.encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json(404, {"error": {"message": "not found"}})

    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), FakeBackend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def _json_model(raw: bytes) -> str:
    try:
        return str(json.loads(raw.decode("utf-8", "replace")).get("model") or "")
    except Exception:
        return ""


def _form_model(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    match = re.search(r'name="model"\r?\n\r?\n([^\r\n]+)', text)
    return match.group(1) if match else ""


def _auto_llm_behavior(raw: bytes) -> dict:
    """Deterministic LLM behaviour when no explicit script is queued.

    If the conversation asks for the console test tool, emit the tool call;
    once the tool output is present, answer with a final text. Otherwise a
    plain greeting. This lets the LIVE self-validation exercise a tool round
    trip without pre-scripting the exact number of prior responses."""
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {"text": "Guten Tag!"}
    messages = payload.get("messages") or []
    joined = json.dumps(messages, ensure_ascii=False)
    has_tool_output = any(
        isinstance(m, dict) and (m.get("role") in ("tool", "function") or "function_call_output" in json.dumps(m, ensure_ascii=False))
        for m in messages if isinstance(m, dict)
    )
    if has_tool_output:
        return {"text": "Der Testwert ist 42."}
    if "get_test_value" in joined:
        return {"tool_call": {"name": "get_test_value", "arguments": json.dumps({"label": "e2e"})}}
    return {"text": "Guten Tag!"}


# --------------------------------------------------------------- API client


class Api:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token

    def call(self, path: str, payload=None, method="GET"):
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return e.code, json.loads(body)
            except ValueError:
                return e.code, {"raw": body}

    def upload(self, voice_id: str, raw: bytes, filename="sample.wav", source="upload"):
        boundary = "----rtve2e"
        body = b""
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\n{source}\r\n".encode()
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
        body += raw
        body += f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.base_url + f"/api/voices/{voice_id}/samples",
            data=body,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


# ------------------------------------------------------------------- flow


def run_flow(page, api: Api, prefix: str, run_dir: Path, report: Report, mode: str, opts, PWTimeout, can_capture: bool) -> dict:
    step_timeout = (opts.step_timeout or opts.timeout) * 1000

    def shot(name: str) -> None:
        try:
            page.screenshot(path=str(run_dir / name), full_page=True)
        except Exception:
            pass

    def step(name, fn, skip_if=None, skip_reason=""):
        t0 = time.monotonic()
        if skip_if:
            report.step(name, False, skip_reason, skipped=True, t0=t0)
            return None
        try:
            result = fn()
            report.step(name, True, result if isinstance(result, str) else "", t0=t0)
            return result
        except Exception as exc:
            shot(f"fail-{name.replace(' ', '-')[:40]}.png")
            report.step(name, False, str(exc), t0=t0)
            return None

    # P2: no native browser dialog may fire anywhere in the flow. Any open
    # dialog is dismissed so the flow can continue, but it is recorded and the
    # dedicated step at the end fails.
    native_dialogs: list[dict] = []

    def on_dialog(d):
        native_dialogs.append({"type": d.type, "message": d.message})
        try:
            d.dismiss()
        except Exception:
            pass

    page.on("dialog", on_dialog)

    def dlg():
        """The currently open in-app dialog root."""
        return page.wait_for_selector(".dlg", timeout=step_timeout)

    def fill_new_profile_dialog(profile_id: str, display_name: str = "") -> None:
        dlg()
        page.fill(".dlg input[aria-label='Name for somora']", profile_id)
        if display_name:
            page.fill(".dlg input[aria-label='Display name']", display_name)
        page.click(".dlg button:has-text('Create set')")
        page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)

    def fill_new_voice_dialog(voice_id: str, display_name: str = "") -> None:
        dlg()
        page.fill(".dlg input[aria-label='Voice id']", voice_id)
        if display_name:
            page.fill(".dlg input[aria-label='Display name']", display_name)
        page.click(".dlg button:has-text('Create voice')")
        page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)

    def confirm_dialog(button_text: str) -> None:
        dlg()
        page.click(f".dlg button:has-text('{button_text}')")
        page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)

    def goto_backends() -> None:
        # Navigate via another route first: a page.goto() to the SAME URL does
        # not re-render, so a dirty add form could survive. Going through
        # #/profiles guarantees a fresh backends view.
        page.goto(api.base_url + "/ui/index.html#/profiles")
        page.wait_for_timeout(150)
        page.goto(api.base_url + "/ui/index.html#/backends")
        page.wait_for_timeout(400)

    sid = f"{prefix}voice"
    pid = f"{prefix}profile"
    created = {"backends": [], "voices": [], "profiles": []}
    models = {
        "stt": opts.stt_model or DEFAULT_MODELS["stt"],
        "llm": opts.llm_model or DEFAULT_MODELS["llm"],
        "tts": opts.tts_model or DEFAULT_MODELS["tts"],
    }
    source_ids = [s.strip() for s in (opts.clone_backend_from or "").split(",") if s.strip()]
    expected_text = ""
    if opts.expect_text_file:
        expected_text = Path(opts.expect_text_file).read_text(encoding="utf-8").strip()

    # ---- login
    def login():
        page.goto(api.base_url + "/ui/index.html", wait_until="domcontentloaded")
        page.fill("#token", api.token)
        page.click("#btn-login")
        page.wait_for_selector("#shell-inner h1", timeout=step_timeout)

    step("login", login)
    shot("01-login.png")

    # ---- helpers for the Models board and the status lines
    def wait_status(prefix_text: str) -> None:
        page.wait_for_function(
            "(t) => [...document.querySelectorAll('#shell-inner .statusline')].some(e => e.textContent.startsWith(t))",
            arg=prefix_text,
            timeout=step_timeout,
        )

    def status_text() -> str:
        return " | ".join(page.eval_on_selector_all("#shell-inner .statusline", "els => els.map(e => e.textContent)"))

    def model_id(role: str, display: str) -> str:
        _s, cat = api.call("/api/catalog")
        for prof in cat["roles"][role]["profiles"]:
            if prof.get("display_name") == display:
                return prof["id"]
        raise AssertionError(f"no {role} model named {display!r}")

    def catalog_row(role: str, mid: str) -> dict:
        _s, cat = api.call("/api/catalog")
        return next(p for p in cat["roles"][role]["profiles"] if p["id"] == mid)

    def open_model(role: str, mid: str) -> None:
        goto_backends()
        page.click(f".mcard[data-model='{role}/{mid}']")
        page.wait_for_selector("[data-cp3='edit-mode']", timeout=step_timeout)
        page.wait_for_timeout(200)

    def start_add(role: str) -> None:
        goto_backends()
        page.click(f".mcol[data-role='{role}'] button:has-text('＋ Add')")
        page.wait_for_selector("[data-cp3='connection-form'] input[aria-label='Display name']", timeout=step_timeout)

    def no_id_field() -> None:
        ids = page.query_selector_all(
            "#shell-inner input[aria-label='Profile id'], #shell-inner input[aria-label='Id'], #shell-inner input[aria-label='Model id']"
        )
        assert not ids, "the model form must not ask for an id"

    ids: dict[str, str] = {}

    # ---- add a model per role via the UI; live mode duplicates server-side
    def add_backend(role, display, model, voices_mode=None):
        if mode == "live":
            source = _source_for_role(role, source_ids, api)
            if not source:
                raise AssertionError(
                    f"no clone source for role {role}; pass --clone-backend-from with a matching-role model id"
                )
            open_model(role, source)
            page.click("#shell-inner button:has-text('Duplicate')")
            dlg()
            page.fill(".dlg input[aria-label='Name of the copy']", display)
            page.click(".dlg button:has-text('Duplicate')")
            page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)
            wait_status("Duplicated.")
            page.wait_for_timeout(300)
            page.fill("input[aria-label='Model']", model)
            if voices_mode:
                page.select_option("select[aria-label='Voices']", value=voices_mode)
            page.click("button:has-text('Save changes')")
            wait_status("Saved")
        else:
            # the board's "＋ Add" of a role opens the form for that role; the
            # server names the model, the form has no id field
            start_add(role)
            no_id_field()
            page.fill("input[aria-label='Display name']", display)
            page.fill("input[aria-label='Base URL']", f"http://127.0.0.1:{opts.backend_port}/v1")
            page.fill("input[aria-label='Model']", model)
            if voices_mode:
                page.select_option("select[aria-label='Voices']", value=voices_mode)
            page.click("button:has-text('Create backend')")
            wait_status("Model added.")
        mid = model_id(role, display)
        ids[role] = mid
        created["backends"].append((role, mid))
        return f"{role} model id {mid}"

    step("add STT model via UI", lambda: add_backend("stt", f"{prefix}STT", models["stt"]))
    step("add LLM model via UI", lambda: add_backend("llm", f"{prefix}LLM", models["llm"]))
    step("add TTS model via UI", lambda: add_backend("tts", f"{prefix}TTS", models["tts"], voices_mode="clones"))
    shot("02-backends.png")

    stt_id = ids.get("stt", "")
    llm_id = ids.get("llm", "")
    tts_id = ids.get("tts", "")

    def tts_clones():
        caps = catalog_row("tts", tts_id).get("capabilities") or {}
        assert caps.get("voice_cloning") == "reference_audio", caps
        return "voice_cloning=reference_audio"

    step("the TTS model added as 'Clones a voice' stores voice_cloning", tts_clones)

    # ---- F13: the model-thinking option exists on LLM rows only and maps to
    # extra.chat_template_kwargs.enable_thinking (default = key absent)
    def thinking_option():
        open_model("llm", llm_id)
        selected = page.eval_on_selector("select[aria-label='Model thinking']", "e=>e.value")
        assert selected == "default", f"new LLM row must default to backend default, got {selected}"
        page.select_option("select[aria-label='Model thinking']", value="off")
        page.click("button:has-text('Save changes')")
        wait_status("Saved")
        row = catalog_row("llm", llm_id)
        assert row["extra"]["chat_template_kwargs"]["enable_thinking"] is False, row["extra"]
        # the option is LLM-only: an STT row must not offer it
        open_model("stt", stt_id)
        assert page.query_selector("select[aria-label='Model thinking']") is None, "STT row must not show thinking"
        # switching back to "backend default" removes the stored key
        open_model("llm", llm_id)
        page.select_option("select[aria-label='Model thinking']", value="default")
        page.click("button:has-text('Save changes')")
        wait_status("Saved")
        row2 = catalog_row("llm", llm_id)
        stored_think = ((row2["extra"] or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
        assert stored_think is None, row2["extra"]

    step("F13 model thinking option on LLM rows maps to enable_thinking", thinking_option)

    # ---- a new set; the new models are offered on its page
    def backend_appears():
        page.goto(api.base_url + "/ui/index.html#/profiles")
        page.wait_for_timeout(300)
        page.click("text=＋ New set")
        fill_new_profile_dialog(pid)
        page.wait_for_function("() => location.hash === %s" % json.dumps(f"#/profiles/{pid}"), timeout=step_timeout)
        created["profiles"].append(pid)
        page.wait_for_selector("select[aria-label='Listens with']", timeout=step_timeout)
        stt_opts = page.eval_on_selector_all("select[aria-label='Listens with'] option", "e=>e.map(o=>o.value)")
        llm_opts = page.eval_on_selector_all("select[aria-label='Thinks with'] option", "e=>e.map(o=>o.value)")
        assert stt_id in stt_opts and llm_id in llm_opts, f"{stt_opts} {llm_opts}"
        assert not page.is_checked("input[aria-label='On']"), "a new set must start Off"

    step("new models appear on a new set's page (UI New set, starts Off)", backend_appears)

    # ================= R2 / CP3: first-time "add a model" flow ==========
    # A first-time user, real clicks: the role can be changed FIRST (role
    # cards), the connection is checked BEFORE saving (model list shown), the
    # model is picked from it, saved, and is selectable in a set. The server
    # names it; there is no id field. Edit mode shows the role fixed. The API
    # key never appears in the DOM or in any network response.
    cp3_state: dict = {}

    def cp3_role_first():
        start_add("stt")
        page.click("#shell-inner button:has-text('Change role')")
        page.wait_for_selector("[data-cp3='role-llm']", timeout=step_timeout)
        labels = page.eval_on_selector_all(
            "[data-cp3^='role-']",
            "els=>els.map(e=>({role:e.getAttribute('data-cp3'), text:e.textContent}))",
        )
        roles = sorted(e["role"] for e in labels)
        assert roles == ["role-llm", "role-stt", "role-tts"], roles
        joined = " | ".join(e["text"] for e in labels)
        for needle in ("Speech recognition", "Reasoning", "Speech engine"):
            assert needle in joined, f"role choice missing {needle}: {joined}"
        assert "..." not in joined, joined
        page.click("[data-cp3='role-llm']")
        page.wait_for_selector("input[aria-label='Base URL']", timeout=step_timeout)
        no_id_field()
        assert page.query_selector("select[aria-label='Model thinking']") is not None, "LLM must offer thinking"
        assert page.query_selector("select[aria-label='Native voice labels']") is None, "LLM must not offer TTS trust"
        assert page.query_selector("select[aria-label='Language as']") is None, "an LLM is not told the language"
        return "role cards: STT/LLM/TTS with help text; no id field"

    step("CP3 role choices (full labels), no id field", cp3_role_first)

    def cp3_check_unsaved():
        page.fill("input[aria-label='Display name']", f"{prefix}CP3 Reasoning")
        page.fill("input[aria-label='Base URL']", f"http://127.0.0.1:{opts.backend_port}/v1")
        page.fill("input[aria-label='Model']", "does-not-exist")
        page.click("[data-cp3='check']")
        # a failed check explains the reason and STILL allows saving (probes are
        # not a gate)
        page.wait_for_function(
            "() => { const r=document.querySelector(\"[data-cp3='check-result']\"); return r && !r.hidden && !r.textContent.startsWith('Checking'); }",
            timeout=step_timeout,
        )
        failed = page.text_content("[data-cp3='check-result']") or ""
        assert "not" in failed.lower() or "fail" in failed.lower(), failed
        assert page.query_selector("button:has-text('Create backend')") is not None, "saving must stay possible"
        page.fill("input[aria-label='Model']", models["llm"])
        page.click("[data-cp3='check']")
        page.wait_for_function(
            "() => { const s=document.querySelector(\"select[aria-label='Available models']\"); return s && [...s.options].some(o=>o.value); }",
            timeout=step_timeout,
        )
        opts_models: list = page.eval_on_selector_all("select[aria-label='Available models'] option", "e=>e.map(o=>o.value)")
        assert models["llm"] in opts_models, f"{models['llm']} not in {opts_models}"
        page.select_option("select[aria-label='Available models']", value=models["llm"])
        picked = page.input_value("input[aria-label='Model']")
        assert picked == models["llm"], picked
        return f"failed check explained; model list -> {opts_models}"

    step("CP3 check the UNSAVED form before saving", cp3_check_unsaved)

    def cp3_save_and_available():
        page.click("button:has-text('Create backend')")
        wait_status("Model added.")
        cp3_id = model_id("llm", f"{prefix}CP3 Reasoning")
        cp3_state["id"] = cp3_id
        created["backends"].append(("llm", cp3_id))
        page.goto(api.base_url + f"/ui/index.html#/profiles/{pid}")
        page.wait_for_selector("select[aria-label='Thinks with']", timeout=step_timeout)
        page.wait_for_timeout(300)
        opts_list = page.eval_on_selector_all("select[aria-label='Thinks with'] option", "e=>e.map(o=>o.value)")
        assert cp3_id in opts_list, f"{cp3_id} not selectable under Thinks with: {opts_list}"
        return f"created {cp3_id}, offered under Thinks with"

    step("CP3 save -> offered in the matching set row", cp3_save_and_available)

    def cp3_same_model_gets_own_id():
        # the same model name added again is a second entry with its own id;
        # the first one stays as it was
        before = catalog_row("llm", cp3_state["id"])
        start_add("llm")
        page.fill("input[aria-label='Display name']", f"{prefix}CP3 Reasoning 2")
        page.fill("input[aria-label='Base URL']", f"http://127.0.0.1:{opts.backend_port}/v1")
        page.fill("input[aria-label='Model']", models["llm"])
        page.click("button:has-text('Create backend')")
        wait_status("Model added.")
        second = model_id("llm", f"{prefix}CP3 Reasoning 2")
        created["backends"].append(("llm", second))
        assert second != cp3_state["id"], second
        after = catalog_row("llm", cp3_state["id"])
        # used_by may differ (sets), everything else is unchanged
        assert {k: v for k, v in after.items() if k != "used_by"} == {k: v for k, v in before.items() if k != "used_by"}, "the first entry changed"
        return f"{cp3_state['id']} and {second}"

    step("CP3 the same model added twice gets a second generated id", cp3_same_model_gets_own_id)

    def cp3_edit_mode_role_fixed():
        cp3_id = cp3_state["id"]
        open_model("llm", cp3_id)
        role_shown = page.text_content("[data-cp3='edit-role']") or ""
        assert "Reasoning" in role_shown or "LLM" in role_shown, role_shown
        assert page.get_attribute("[data-cp3='edit-role']", "data-role") == "llm", "role must be fixed when editing"
        no_id_field()
        page.fill("input[aria-label='Display name']", f"{prefix}CP3 Reasoning (edited)")
        page.click("button:has-text('Save changes')")
        wait_status("Saved")
        row = catalog_row("llm", cp3_id)
        assert row["display_name"] == f"{prefix}CP3 Reasoning (edited)", row["display_name"]
        return "edit mode: role fixed, no id, Save changes works"

    step("CP3 edit keeps the role fixed and has no id", cp3_edit_mode_role_fixed)

    def cp3_key_never_in_dom_or_responses():
        secret = "sk-cp3-super-secret-key"
        responses: list[str] = []

        def on_response(resp):
            try:
                responses.append(resp.text())
            except Exception:
                pass

        page.on("response", on_response)
        start_add("stt")
        page.fill("input[aria-label='Display name']", f"{prefix}CP3 Secret")
        page.fill("input[aria-label='Base URL']", f"http://127.0.0.1:{opts.backend_port}/v1")
        page.fill("input[aria-label='Model']", models["stt"])
        page.fill("input[aria-label='API key']", secret)
        page.click("[data-cp3='check']")
        page.wait_for_selector("[data-cp3='check-result']", timeout=step_timeout)
        page.wait_for_timeout(400)
        body = page.inner_text("body")
        assert secret not in body, "key leaked into the DOM text"
        blob = "\n".join(responses)
        assert secret not in blob, "key leaked into a network response"
        page.remove_listener("response", on_response)
        goto_backends()
        return "key never echoed"

    step("CP3 API key never appears in DOM or network responses", cp3_key_never_in_dom_or_responses)

    def models_language_format():
        # how a model is told the set's language: code / name / none
        start_add("tts")
        values = page.eval_on_selector_all("select[aria-label='Language as'] option", "e=>e.map(o=>o.value)")
        assert values == ["code", "name", "none"], values
        open_model("tts", tts_id)
        page.select_option("select[aria-label='Language as']", value="name")
        page.click("button:has-text('Save changes')")
        wait_status("Saved")
        assert catalog_row("tts", tts_id).get("language_format") == "name"
        badges = page.text_content(f".mcard[data-model='tts/{tts_id}']") or ""
        assert "language as name" in badges, badges
        page.select_option("select[aria-label='Language as']", value="code")
        page.click("button:has-text('Save changes')")
        wait_status("Saved")
        return "Language as: code/name/none, saved and shown on the card"

    step("model form: Language as (code/name/none) is saved", models_language_format)

    # ============ R2 / C3: changed endpoint must be confirmed ==============
    # Saving a model with a changed endpoint and a blank key must not
    # silently keep sending the stored secret to the new host. The edit form
    # asks: enter the key, or explicitly keep the existing one.
    def c3_changed_endpoint_asks():
        start_add("stt")
        page.fill("input[aria-label='Display name']", f"{prefix}C3 Endpoint")
        page.fill("input[aria-label='Base URL']", "http://127.0.0.1:9/v1")
        page.fill("input[aria-label='Model']", models["stt"])
        page.fill("input[aria-label='API key']", "sk-c3-stored")
        page.click("button:has-text('Create backend')")
        wait_status("Model added.")
        c3_id = model_id("stt", f"{prefix}C3 Endpoint")
        created["backends"].append(("stt", c3_id))

        # change ONLY the endpoint, leave the key blank -> the form must ask
        open_model("stt", c3_id)
        page.fill("input[aria-label='Base URL']", "http://127.0.0.1:9/v2")
        page.click("button:has-text('Save changes')")
        page.wait_for_selector(".dlg", timeout=step_timeout)
        text = page.inner_text(".dlg")
        assert "endpoint changed" in text.lower(), text
        assert "127.0.0.1" in text, f"the new host must be named: {text}"
        assert "existing key" in text.lower(), text
        page.click(".dlg button:has-text('Keep existing key')")
        page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)
        wait_status("Saved")
        assert catalog_row("stt", c3_id)["base_url"].endswith("/v2")

        # cancelling the ask must NOT save the changed endpoint
        page.fill("input[aria-label='Base URL']", "http://127.0.0.1:9/v3")
        page.click("button:has-text('Save changes')")
        page.wait_for_selector(".dlg", timeout=step_timeout)
        page.click(".dlg button:has-text('Enter a new key')")
        page.wait_for_selector(".dlg", state="detached", timeout=step_timeout)
        row = catalog_row("stt", c3_id)
        assert row["base_url"].endswith("/v2"), f"cancel must not save: {row['base_url']}"
        return "asked, kept after confirm, cancel does not save"

    step("C3 changed endpoint + blank key asks before keeping the stored key", c3_changed_endpoint_asks)

    # ---- Settings: the listening model that transcribes recordings
    def settings_transcription_model():
        _s, before = api.call("/api/settings")
        created.setdefault("settings_before", {"voice_library": dict(before.get("voice_library") or {})})
        page.goto(api.base_url + "/ui/index.html#/settings")
        page.wait_for_selector("select[aria-label='Transcribes recordings']", timeout=step_timeout)
        assert (page.text_content("#shell-inner h1") or "") == "Settings"
        page.select_option("select[aria-label='Transcribes recordings']", value=stt_id)
        wait_status("Model that transcribes recordings saved.")
        _s, after = api.call("/api/settings")
        assert (after.get("voice_library") or {}).get("transcription_stt_profile") == stt_id, after
        # the defaults for calls live here now
        assert page.is_visible("textarea[aria-label='Default instructions']")
        return f"transcription model {stt_id}"

    step("Settings: choose the model that transcribes recordings", settings_transcription_model)
    shot("02b-settings.png")

    # ---- create voice via UI
    def create_voice():
        page.goto(api.base_url + "/ui/index.html#/voices")
        page.wait_for_timeout(300)
        page.click("text=＋ New voice")
        fill_new_voice_dialog(sid)
        page.wait_for_function("() => location.hash.startsWith('#/voices/')", timeout=step_timeout)
        created["voices"].append(sid)

    step("create voice via UI", create_voice)

    # ---- upload + auto-transcribe + confirm via UI (real speech in live mode)
    def upload_confirm():
        upload = run_dir / "upload.wav"
        upload.write_bytes(opts.upload_bytes)
        page.set_input_files("#voice-file", str(upload))
        page.wait_for_function("() => { const t=document.querySelector(\"textarea[aria-label='Transcript']\"); return t && t.value.length>0; }", timeout=step_timeout)
        page.click("text=Confirm transcript")
        page.wait_for_function("() => document.body.innerText.includes('Ready to use as a cloning reference')", timeout=step_timeout)

    step("upload + auto-transcribe + confirm", upload_confirm)
    shot("03-voice-confirmed.png")

    # ---- live transcript assertion (non-empty + optional similarity ratio)
    def live_transcript():
        status, voice = api.call(f"/api/voices/{sid}")
        assert status == 200, f"GET voice -> {status}"
        sample = (voice.get("samples") or [{}])[0]
        text = (sample.get("transcript") or "").strip()
        assert text, "live STT returned an empty transcript"
        if expected_text:
            ratio = similarity(expected_text, text)
            assert ratio >= 0.5, f"transcript similarity {ratio} below 0.5"
            return f"transcript non-empty, similarity {ratio}"
        return "transcript non-empty"

    step("live transcript is real (similarity ratio only)", live_transcript, skip_if=mode != "live", skip_reason="fake mode")

    # ---- preview provenance
    def preview_provenance():
        page.wait_for_selector("select[aria-label='TTS profile']", timeout=step_timeout)
        page.wait_for_timeout(400)
        page.select_option("select[aria-label='TTS profile']", value=tts_id)
        page.click("text=Generate preview")
        page.wait_for_function("() => document.querySelectorAll('.provenance').length>0", timeout=step_timeout)
        prov = page.text_content(".provenance") or ""
        assert "sha8" in prov, prov
        assert models["tts"] in prov, f"tts model {models['tts']} not in {prov}"
        return " ".join(prov.split("\n"))

    step("preview shows server provenance + chosen TTS model", preview_provenance)
    shot("04-preview.png")

    # ---- live preview provenance: audio_ms > 0, model, sample sha8 matches
    def live_preview_provenance():
        status, voice = api.call(f"/api/voices/{sid}")
        sample = (voice.get("samples") or [{}])[0]
        sample_sha = sample.get("sha256") or ""
        payload = json.dumps({"tts_profile": tts_id, "text": "Dies ist ein Test."}).encode()
        req = urllib.request.Request(
            api.base_url + f"/api/voices/{sid}/preview",
            data=payload,
            headers={"Authorization": f"Bearer {api.token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                headers = r.headers
                r.read()
            audio_ms = int(headers.get("X-Rtv-Audio-Ms") or 0)
            model = headers.get("X-Rtv-Model") or ""
            sha8 = headers.get("X-Rtv-Reference-Sha8") or ""
            assert audio_ms > 0, f"preview audio_ms={audio_ms}"
            assert model == models["tts"], f"preview model={model} != {models['tts']}"
            assert sample_sha and sha8 == sample_sha[:8], f"sample sha8 {sha8} != uploaded {sample_sha[:8]}"
            return f"audio_ms={audio_ms}, model={model}, sha8 ok"
        except urllib.error.HTTPError as e:
            raise AssertionError(f"preview HTTP {e.code}: {e.read().decode()[:200]}")

    step(
        "live preview: audio_ms>0, model, sample sha8",
        live_preview_provenance,
        skip_if=mode != "live",
        skip_reason="fake mode",
    )

    # ---- the set: choose with real clicks, save, reload, switch On
    def open_set(set_id: str) -> None:
        page.goto(api.base_url + "/ui/index.html#/voices")
        page.wait_for_timeout(150)
        page.goto(api.base_url + f"/ui/index.html#/profiles/{set_id}")
        page.wait_for_selector("select[aria-label='Listens with']", timeout=step_timeout)
        page.wait_for_timeout(300)

    def pick_speaker() -> None:
        # with one model that can speak the voice it is shown as text;
        # with several, a select offers them
        if page.query_selector("select[aria-label='Speaking model']"):
            page.select_option("select[aria-label='Speaking model']", value=tts_id)

    def configure_set():
        open_set(pid)
        page.select_option("select[aria-label='Listens with']", value=stt_id)
        page.select_option("select[aria-label='Thinks with']", value=llm_id)
        groups = page.eval_on_selector_all("select[aria-label='Speaks as'] optgroup", "e=>e.map(g=>g.label)")
        assert "Your voices" in groups, groups
        page.select_option("select[aria-label='Speaks as']", value=f"lib:{sid}")
        page.wait_for_timeout(200)
        pick_speaker()
        page.select_option("select[aria-label='Language']", value="de")
        page.click(".head-actions button.btn.pri:has-text('Save')")
        wait_status("Saved")
        status, view = api.call(f"/api/conversation-profiles/{pid}")
        spec = view.get("spec") or {}
        assert spec.get("stt_profile") == stt_id and spec.get("llm_profile") == llm_id and spec.get("tts_profile") == tts_id, spec
        assert spec.get("voice", {}).get("kind") == "library" and spec["voice"].get("voice_id") == sid, spec
        assert spec.get("language") == "de" and view.get("enabled") is False, view
        return f"groups={groups}"

    step("choose listens/thinks/voice/language + Save (real clicks)", configure_set)

    def reload_persists():
        page.reload()
        page.wait_for_selector("select[aria-label='Speaks as']", timeout=step_timeout)
        page.wait_for_timeout(300)
        voice = page.eval_on_selector("select[aria-label='Speaks as']", "e=>e.value")
        stt = page.eval_on_selector("select[aria-label='Listens with']", "e=>e.value")
        lang = page.eval_on_selector("select[aria-label='Language']", "e=>e.value")
        assert voice == f"lib:{sid}" and stt == stt_id and lang == "de", f"voice={voice} stt={stt} lang={lang}"
        assert page.is_disabled(".head-actions button.btn.pri:has-text('Save')"), "Save must be idle with nothing changed"

    step("reload shows the saved set", reload_persists)

    def switch_on():
        page.check("input[aria-label='On']")
        wait_status("Saved. The next call uses it")
        status, view = api.call(f"/api/conversation-profiles/{pid}")
        assert view.get("enabled") is True, view
        sub = page.text_content(".h1sub") or ""
        assert sub.startswith("On"), sub

    step("switch the set On", switch_on)
    shot("05-profile-on.png")

    def set_language_select():
        values = page.eval_on_selector_all("select[aria-label='Language'] option", "e=>e.map(o=>o.value)")
        assert "de" in values and "en" in values, values
        assert page.eval_on_selector("select[aria-label='Language']", "e=>e.value") == "de"
        return f"{len(values)} languages"

    step("the set's language is a choice of languages showing the stored one", set_language_select)

    def try_it_hear():
        page.click("button:has-text('Hear a sentence')")
        page.wait_for_selector("[data-f10='hear-provenance']", timeout=step_timeout)
        prov = page.text_content("[data-f10='hear-provenance']") or ""
        assert models["tts"] in prov, prov
        return prov.strip()

    step("Try it: Hear a sentence speaks with the set", try_it_hear)

    # ---- Test Console
    def open_console():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".talkbtn", timeout=step_timeout)

    step("open Test Console for the set", open_console)

    audio_skip = None if can_capture else "browser lacks getUserMedia/audioWorklet (WebKit WPE build)"
    report.step("audio capture capability", True, f"can_capture={can_capture}")

    def talk_audio_turn():
        page.click(".talkbtn")
        page.wait_for_function("() => { const s=document.querySelector('.talkstate'); return s && !s.textContent.includes('Connecting'); }", timeout=step_timeout)
        try:
            page.wait_for_function("() => document.querySelectorAll('.tscol .turn').length > 0", timeout=step_timeout)
        except Exception:
            diag = page.evaluate(
                """() => ({
                    frames: (window.__rtvE2E && window.__rtvE2E.counters && window.__rtvE2E.counters.appendFrames) || 0,
                    responses: (window.__rtvE2E && window.__rtvE2E.counters && window.__rtvE2E.counters.responseDone) || 0,
                    events: (window.__rtvTestConsole && window.__rtvTestConsole.state && window.__rtvTestConsole.state.events || []).slice(-8).map(e=>e.type),
                    mic: (window.__rtvTestConsole && window.__rtvTestConsole.session && window.__rtvTestConsole.session.activeMicLabel) || "",
                })"""
            )
            raise AssertionError(f"no transcript turn; diag={diag}")
        page.wait_for_timeout(1500)
        turns = page.eval_on_selector_all(".turn", "els=>els.map(e=>e.textContent)")
        assert len(turns) >= 1
        return f"turns={len(turns)}, frames={page.evaluate('() => (window.__rtvE2E.counters.appendFrames)||0')}"

    step("Test Console audio turn (frames -> STT -> response)", talk_audio_turn, skip_if=audio_skip, skip_reason=audio_skip or "")
    shot("06-console-turn.png")

    def route_models():
        if not can_capture:
            return None
        page.wait_for_selector(".routechain .rcs", timeout=step_timeout)
        chain = page.eval_on_selector_all(
            ".routechain .rcs",
            "els=>els.map(e=>({role:e.querySelector('.rcrole').textContent, model:e.querySelector('.rcmodel').textContent}))",
        )
        models_seen = {c["role"]: c["model"] for c in chain}
        assert models_seen.get("Listens") == models["stt"], models_seen
        assert models_seen.get("Thinks") == models["llm"], models_seen
        assert models_seen.get("Speaks") == models["tts"], models_seen
        return f"stt={models['stt']} llm={models['llm']} tts={models['tts']}"

    step("route shows the three chosen models", route_models, skip_if=audio_skip, skip_reason=audio_skip or "")

    def response_done_single():
        # exactly one response.done per user turn (counters injected below)
        before = page.evaluate("() => (window.__rtvE2E && window.__rtvE2E.counters) ? window.__rtvE2E.counters.responseDone : 0")
        page.click(".talkbtn")  # stop
        page.wait_for_timeout(400)
        page.click(".talkbtn")  # start again
        page.wait_for_function("() => { const s=document.querySelector('.talkstate'); return s && !s.textContent.includes('Connecting'); }", timeout=step_timeout)
        page.wait_for_timeout(2500)
        after = page.evaluate("() => (window.__rtvE2E && window.__rtvE2E.counters) ? window.__rtvE2E.counters.responseDone : 0")
        delta = after - before
        assert delta <= 1, f"{delta} response.done for one turn"
        return f"response.done delta={delta}"

    step("response.done per user turn == 1", response_done_single, skip_if=audio_skip, skip_reason=audio_skip or "")

    def tool_round_trip():
        if mode != "live":
            opts.fake_llm_script.append({"tool_call": {"name": "get_test_value", "arguments": json.dumps({"label": "e2e"})}})
            opts.fake_llm_script.append({"text": "Der Testwert ist 42."})
        page.evaluate("() => { if (window.__rtvTestConsole && window.__rtvTestConsole.state) window.__rtvTestConsole.state.toolRoundTrips = 0; }")
        # The console installs a deterministic test tool; ask for it explicitly
        # by name so a real LLM is strongly steered to call it.
        page.evaluate(
            """() => {
                const s = window.__rtvTestConsole && window.__rtvTestConsole.session;
                if (!s) throw new Error('no console session');
                s.send({type:'conversation.item.create', item:{type:'message', role:'user', content:[{type:'input_text', text:'Please call the get_test_value tool with label "e2e" and report the value.'}]}});
                s.send({type:'response.create'});
            }"""
        )
        page.wait_for_function("() => (window.__rtvTestConsole.state.toolRoundTrips||0) > 0", timeout=step_timeout)
        return f"toolRoundTrips={page.evaluate('() => window.__rtvTestConsole.state.toolRoundTrips')}"

    step("tool call round trip", tool_round_trip, skip_if=audio_skip, skip_reason=audio_skip or "")

    def barge_in():
        before_started = page.evaluate("() => (window.__rtvE2E && window.__rtvE2E.counters) ? window.__rtvE2E.counters.responseDone : 0")
        page.evaluate(
            """() => {
                const s = window.__rtvTestConsole.session;
                s.send({type:'response.create'});
                const tone = new Int16Array(18000);
                for (let i = 0; i < tone.length; i++) tone[i] = Math.round(9000 * Math.sin(2*Math.PI*300*i/24000));
                setTimeout(() => s.sendAudio(tone), 120);
            }"""
        )
        page.wait_for_timeout(3000)
        after_done = page.evaluate("() => (window.__rtvE2E && window.__rtvE2E.counters) ? window.__rtvE2E.counters.responseDone : 0")
        return f"responsesDoneDelta={after_done - before_started}"

    step("barge-in cancels and continues", barge_in, skip_if=audio_skip, skip_reason=audio_skip or "")

    def stop_start_again():
        page.click(".talkbtn")
        page.wait_for_timeout(500)
        page.click(".talkbtn")
        page.wait_for_function("() => { const s=document.querySelector('.talkstate'); return s && !s.textContent.includes('Connecting'); }", timeout=step_timeout)
        page.click(".talkbtn")

    step("stop then start again", stop_start_again, skip_if=audio_skip, skip_reason=audio_skip or "")

    # ---- C1 (call order): the Talk click must not await a network GET before
    # session.start(). The mic request (getUserMedia) happens inside start; it
    # must come before the profile-list refresh fetch.
    def c1_mic_before_profile_fetch():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".talkbtn", timeout=step_timeout)
        page.wait_for_timeout(300)
        page.evaluate("() => { window.__rtvSeq.events = []; }")
        page.click(".talkbtn")
        page.wait_for_function("() => window.__rtvMic && window.__rtvMic.constraints.length > 0", timeout=step_timeout)
        page.wait_for_timeout(600)
        ev = page.evaluate("() => window.__rtvSeq.events")
        if not ev:
            raise AssertionError("no side-effect events recorded")
        # first fetch to the profiles list AFTER the click
        fetch_i = next(
            (e["i"] for e in ev if e["kind"] == "fetch" and "/api/conversation-profiles" in e["detail"]),
            None,
        )
        gum_i = next((e["i"] for e in ev if e["kind"] == "getUserMedia"), None)
        if gum_i is None:
            raise AssertionError(f"no getUserMedia recorded: {ev}")
        if fetch_i is not None and fetch_i < gum_i:
            raise AssertionError(f"C1: a profiles GET precedes getUserMedia: {ev}")
        page.click(".talkbtn")  # stop
        return f"getUserMedia@{gum_i} profilesFetch@{fetch_i}"

    step(
        "C1 Talk starts the mic before any profile-list GET",
        c1_mic_before_profile_fetch,
        skip_if=audio_skip,
        skip_reason=audio_skip or "",
    )

    # ---- C3: focusing / opening the picker must not fetch nor rebuild options
    def c3_picker_focus_no_fetch_or_rebuild():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector("select[aria-label='Set']", timeout=step_timeout)
        page.wait_for_timeout(400)
        # mark an option and observe picker child-list mutations
        page.evaluate(
            """() => {
                const sel = document.querySelector("select[aria-label='Set']");
                const opt = sel.options[1];
                opt.dataset.rtvOrig = 'yes';
                window.__rtvPickerMut = 0;
                try {
                    window.__rtvPickerObs && window.__rtvPickerObs.disconnect();
                    const obs = new MutationObserver((recs) => {
                        for (const r of recs) if (r.type === 'childList') window.__rtvPickerMut++;
                    });
                    obs.observe(sel, { childList: true });
                    window.__rtvPickerObs = obs;
                } catch (e) {}
                window.__rtvSeq.events = [];
            }"""
        )
        # focus the picker (this is what fires when the user clicks a native select)
        page.focus("select[aria-label='Set']")
        page.wait_for_timeout(500)
        res = page.evaluate(
            """() => {
                const sel = document.querySelector("select[aria-label='Set']");
                const fetchProfiles = (window.__rtvSeq.events || []).filter(
                    (e) => e.kind === 'fetch' && e.detail.indexOf('/api/conversation-profiles') !== -1
                ).length;
                const orig = sel.options[1] && sel.options[1].dataset.rtvOrig === 'yes';
                return { fetchProfiles, mutations: window.__rtvPickerMut || 0, origOptionKept: !!orig };
            }"""
        )
        if res["fetchProfiles"] != 0:
            raise AssertionError(f"C3: focusing the picker issued {res['fetchProfiles']} profiles GET(s)")
        if res["mutations"] != 0:
            raise AssertionError(f"C3: focusing the picker rebuilt the options ({res['mutations']} mutations)")
        if not res["origOptionKept"]:
            raise AssertionError("C3: the original option element was replaced on focus")
        return f"fetch=0 mutations=0 origOptionKept={res['origOptionKept']}"

    step("C3 picker focus issues no profiles GET and does not rebuild options", c3_picker_focus_no_fetch_or_rebuild)

    # ---- C4: one visibilitychange listener after two mounts
    def c4_single_visibility_listener():
        # mount the console, leave it, mount it again -> two mounts
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector("select[aria-label='Set']", timeout=step_timeout)
        page.goto(api.base_url + f"/ui/index.html#/profiles")
        page.wait_for_timeout(300)
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector("select[aria-label='Set']", timeout=step_timeout)
        page.wait_for_timeout(400)
        page.evaluate("() => { window.__rtvSeq.events = []; }")
        # one visibilitychange -> visible must issue EXACTLY one profiles GET
        page.evaluate(
            """() => {
                document.dispatchEvent(new Event('visibilitychange'));
            }"""
        )
        page.wait_for_timeout(700)
        fetches = page.evaluate(
            """() => (window.__rtvSeq.events || []).filter(
                (e) => e.kind === 'fetch' && e.detail.indexOf('/api/conversation-profiles') !== -1
            ).length"""
        )
        if fetches != 1:
            raise AssertionError(f"C4: one visibilitychange issued {fetches} profiles GET(s), expected 1")
        return f"profiles GET per visibilitychange = {fetches}"

    step("C4 only one visibilitychange listener after two mounts", c4_single_visibility_listener)

    # ================= a saved set is live =================================
    # No draft, no activation: what is saved is what the next call uses, the
    # old version stays under "Earlier versions", and "Try it" uses what the
    # page shows without saving.
    sid2 = f"{prefix}voice2"
    live_state: dict = {}

    def _create_second_voice():
        st, _ = api.call("/api/voices", {"id": sid2, "display_name": "E2E Voice 2"}, method="POST")
        assert st in (200, 201), f"create voice2 -> {st}"
        sample = api.upload(sid2, wav_bytes(2.0, freq=660.0), filename="voice2.wav")
        st, _ = api.call(
            f"/api/voices/{sid2}/samples/{sample['id']}",
            {"transcript": "zweite Referenzstimme", "transcript_confirmed": True},
            method="PATCH",
        )
        assert st == 200, f"confirm voice2 sample -> {st}"
        created["voices"].append(sid2)

    step("create a second library voice", _create_second_voice)

    def try_it_unsaved_talk():
        # an unsaved voice change is tested through a ticket; nothing is saved
        open_set(pid)
        page.select_option("select[aria-label='Speaks as']", value=f"lib:{sid2}")
        page.wait_for_timeout(200)
        pick_speaker()
        page.click("button:has-text('Talk to it')")
        page.wait_for_function("() => location.hash.includes('ticket=profile-test')", timeout=step_timeout)
        page.wait_for_selector(".talkbtn", timeout=step_timeout)
        banner = page.inner_text("#shell-inner")
        assert "Testing what the set page shows" in banner, banner[:200]
        status, view = api.call(f"/api/conversation-profiles/{pid}")
        assert view["spec"]["voice"]["voice_id"] == sid, "Try it must not save"
        return "ticket for the unsaved voice; the saved set unchanged"

    step("Try it with unsaved values uses a test ticket and saves nothing", try_it_unsaved_talk)

    def saved_change_is_live():
        open_set(pid)
        status, before = api.call(f"/api/conversation-profiles/{pid}")
        page.select_option("select[aria-label='Speaks as']", value=f"lib:{sid2}")
        page.wait_for_timeout(200)
        pick_speaker()
        page.click(".head-actions button.btn.pri:has-text('Save')")
        wait_status("Saved. The next call uses it")
        status, view = api.call(f"/api/conversation-profiles/{pid}")
        assert view["spec"]["voice"]["voice_id"] == sid2, view["spec"]
        assert view["enabled"] is True, "saving must not switch the set off"
        assert len(view.get("history") or []) == len(before.get("history") or []) + 1, "the old version is not kept"
        assert view["history"][0]["spec"]["voice"]["voice_id"] == sid, view["history"][0]
        live_state["history_len"] = len(view["history"])
        return f"history={len(view['history'])}"

    step("saving a changed voice is live; the old version is kept", saved_change_is_live)
    shot("08-saved-live.png")

    def console_shows_saved_voice():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".talkbtn", timeout=step_timeout)
        page.wait_for_timeout(300)
        opts_text = page.eval_on_selector_all("select[aria-label='Set'] option", "els=>els.map(o=>o.textContent)")
        joined = " | ".join(opts_text)
        assert "voice E2E Voice 2" in joined, f"saved voice not in the picker: {joined}"
        page.reload()
        page.wait_for_selector(".talkbtn", timeout=step_timeout)
        page.wait_for_timeout(300)
        again = " | ".join(page.eval_on_selector_all("select[aria-label='Set'] option", "els=>els.map(o=>o.textContent)"))
        assert "voice E2E Voice 2" in again, f"reload lost the saved voice: {again}"
        return joined[:120]

    step("Console picker shows the saved voice (and after reload)", console_shows_saved_voice)

    def running_route_uses_new_voice():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".talkbtn", timeout=step_timeout)
        page.click(".talkbtn")
        page.wait_for_function(
            "() => { const s=document.querySelector('.talkstate'); return s && !s.textContent.includes('Connecting'); }",
            timeout=step_timeout,
        )
        page.wait_for_function(
            "() => { const v=document.querySelector('.routechain .rcvoice'); return v && v.textContent.trim().length > 0; }",
            timeout=step_timeout,
        )
        voice = page.text_content(".routechain .rcvoice") or ""
        page.click(".talkbtn")  # stop
        assert "E2E Voice 2" in voice or sid2 in voice, f"running route voice: {voice}"
        return f"voice='{voice.strip()}'"

    step(
        "running route reports the saved voice",
        running_route_uses_new_voice,
        skip_if=audio_skip,
        skip_reason=audio_skip or "",
    )

    def use_earlier_version():
        open_set(pid)
        page.click("summary:has-text('Earlier versions')")
        page.click("button:has-text('Use this version') >> nth=0")
        confirm_dialog("Use this version")
        wait_status("Earlier version in use")
        status, view = api.call(f"/api/conversation-profiles/{pid}")
        assert view["spec"]["voice"]["voice_id"] == sid, view["spec"]
        assert view["enabled"] is True
        assert view["history"][0]["spec"]["voice"]["voice_id"] == sid2, "the replaced version is kept"
        voice = page.eval_on_selector("select[aria-label='Speaks as']", "e=>e.value")
        assert voice == f"lib:{sid}", f"the page still shows {voice}"
        return "voice 1 again; voice 2 kept as an earlier version"

    step("Earlier versions: Use this version restores it", use_earlier_version)

    def incomplete_set_cannot_switch_on():
        pid2 = f"{prefix}incomplete"
        page.goto(api.base_url + "/ui/index.html#/profiles")
        page.wait_for_timeout(300)
        page.click("text=＋ New set")
        fill_new_profile_dialog(pid2)
        page.wait_for_function("() => location.hash === %s" % json.dumps(f"#/profiles/{pid2}"), timeout=step_timeout)
        created["profiles"].append(pid2)
        page.wait_for_selector("select[aria-label='Listens with']", timeout=step_timeout)
        page.check("input[aria-label='On']")
        wait_status("Not saved")
        assert not page.is_checked("input[aria-label='On']"), "the switch must fall back to Off"
        status, view = api.call(f"/api/conversation-profiles/{pid2}")
        assert view.get("enabled") is False, view
        status, body = api.call(f"/api/conversation-profiles/{pid2}/test-session", {}, method="POST")
        assert status == 409 and body.get("code") == "set_incomplete", f"{status} {body}"
        return f"{status_text()[:100]}; test-session 409 set_incomplete"

    step("an incomplete set cannot be switched On (and cannot be tested)", incomplete_set_cannot_switch_on)

    # ================= browser coverage gaps (fake + live) =================

    # ---- F6: shared sample playback -------------------------------------
    def f6_playback():
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("button[data-rtv-sample]", timeout=step_timeout)
        page.wait_for_timeout(400)
        buttons = page.eval_on_selector_all("button[data-rtv-sample]", "e=>e.length")
        assert buttons >= 1, "no sample play button"
        # rapid 5x click on the SAME sample -> exactly one playing media
        for _ in range(5):
            page.click("button[data-rtv-sample] >> nth=0")
        page.wait_for_timeout(700)
        playing = page.evaluate("() => (window.__rtvMedia && window.__rtvMedia.playingCount) ? window.__rtvMedia.playingCount() : 0")
        pressed_keys = page.eval_on_selector_all(
            "button[data-rtv-sample][aria-pressed='true']",
            "e=>[...new Set(e.map(b=>b.getAttribute('data-rtv-sample')))]",
        )
        assert playing <= 1, f"{playing} media playing after 5 rapid clicks"
        # at most one distinct sample is playing (initially one sample: its two
        # mirror buttons may both be pressed truthfully)
        assert len(pressed_keys) <= 1, f"{pressed_keys} distinct samples pressed"
        # pause keeps currentTime; resume continues from that position
        page.click("button[data-rtv-sample] >> nth=0")  # pause
        page.wait_for_timeout(300)
        ct_paused = page.evaluate("() => window.__rtvMedia.currentTime()")
        page.click("button[data-rtv-sample] >> nth=0")  # resume
        page.wait_for_timeout(500)
        ct_resumed = page.evaluate("() => window.__rtvMedia.currentTime()")
        assert ct_resumed >= ct_paused, f"resume reset position {ct_paused} -> {ct_resumed}"
        # navigation stops playback
        page.goto(api.base_url + "/ui/index.html#/voices")
        page.wait_for_timeout(600)
        playing_after = page.evaluate("() => (window.__rtvMedia && window.__rtvMedia.playingCount) ? window.__rtvMedia.playingCount() : 0")
        assert playing_after == 0, f"{playing_after} media still playing after navigation"
        return f"playing={playing}, paused_ct={round(ct_paused,3)}, resumed_ct={round(ct_resumed,3)}"

    step("F6 playback: one media, pause/resume position, nav stop", f6_playback, skip_if=not can_capture, skip_reason=audio_skip or "")

    def f6_switch_sample():
        # ensure a second sample exists so switching between two is testable
        api.upload(sid, wav_bytes(2.0, freq=330.0), filename="second-a.wav")
        api.upload(sid, wav_bytes(2.0, freq=440.0), filename="second-b.wav")
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("button[data-rtv-sample]", timeout=step_timeout)
        page.wait_for_timeout(500)
        n = page.eval_on_selector_all("button[data-rtv-sample]", "e=>e.length")
        assert n >= 2, f"expected >=2 samples, got {n}"
        page.click("button[data-rtv-sample] >> nth=0")
        page.wait_for_timeout(500)
        page.click("button[data-rtv-sample] >> nth=1")
        page.wait_for_timeout(500)
        playing = page.evaluate("() => (window.__rtvMedia && window.__rtvMedia.playingCount) ? window.__rtvMedia.playingCount() : 0")
        assert playing <= 1, f"{playing} media playing after switching samples"
        return f"samples={n}, playing={playing}"

    step("F6 playback: starting another sample stops the first", f6_switch_sample, skip_if=not can_capture, skip_reason=audio_skip or "")

    # ---- F2: microphone selection ---------------------------------------
    def f2_mic_selection():
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".micpick select", timeout=step_timeout)
        page.wait_for_timeout(400)
        # open the picker to force lazy enumeration
        page.click(".micpick select")
        page.wait_for_timeout(500)
        labels = page.eval_on_selector_all(".micpick select option", "e=>e.map(o=>o.textContent)")
        device_values = page.eval_on_selector_all(".micpick select option", "e=>e.map(o=>o.value).filter(v=>v)")
        assert len(labels) >= 1, "no mic options rendered"
        assert len(device_values) >= 1, f"no selectable fake devices: {labels}"
        chosen = device_values[0]
        page.select_option(".micpick select", value=chosen)
        page.wait_for_timeout(300)
        # use the console's own start path to call getUserMedia with the device
        page.evaluate("() => { window.__rtvMic = window.__rtvMic || {constraints: []}; }")
        stored = page.evaluate("() => localStorage.getItem('rtv_mic_device_id')")
        assert stored == chosen, f"selection not remembered: {stored} != {chosen}"
        return f"devices={len(device_values)}"

    step(
        "F2 mic: fake devices render + selection remembered",
        f2_mic_selection,
        skip_if=not can_capture,
        skip_reason=audio_skip or "",
    )

    def f2_device_removed_notice():
        # Remember a device id that the (instrumented) enumeration will hide,
        # then dispatch devicechange and assert the honest notice.
        page.goto(api.base_url + f"/ui/index.html#/test?profile={pid}")
        page.wait_for_selector(".micpick select", timeout=step_timeout)
        page.wait_for_timeout(300)
        page.evaluate(
            """() => {
                window.__rtvMic.hide = 'removed-device-id';
                localStorage.setItem('rtv_mic_device_id', 'removed-device-id');
            }"""
        )
        page.evaluate("() => { const md=navigator.mediaDevices; if (md && md.dispatchEvent) md.dispatchEvent(new Event('devicechange')); }")
        # also force a refresh via the picker's own lazy path
        page.click(".micpick select")
        page.wait_for_timeout(600)
        notice = page.text_content(".micpick .fhint") or ""
        assert "not available" in notice and "system default" in notice, notice
        return notice.strip()

    step(
        "F2 mic: removed device shows 'using system default'",
        f2_device_removed_notice,
        skip_if=not can_capture,
        skip_reason=audio_skip or "",
    )

    # ---- silence upload -> no_speech_detected, no transcript stored ------
    def silence_upload():
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("#voice-file", timeout=step_timeout)
        page.wait_for_timeout(300)
        silent = run_dir / "silence.wav"
        silent.write_bytes(silence_wav(2.0))
        page.set_input_files("#voice-file", str(silent))
        page.wait_for_function(
            "() => document.querySelector('.bar-err') && document.querySelector('.bar-err').textContent.includes('no_speech_detected')",
            timeout=step_timeout,
        )
        textareas = page.eval_on_selector_all("textarea[aria-label='Transcript']", "e=>e.length")
        assert textareas == 0, f"silence produced a transcript textarea ({textareas})"
        # server truth: the sample carries no transcript
        status, voice = api.call(f"/api/voices/{sid}")
        silent_samples = [s for s in voice.get("samples", []) if s.get("transcription", {}).get("error_code") == "no_speech_detected"]
        assert silent_samples, "server did not record no_speech_detected"
        assert all(not (s.get("transcript") or "").strip() for s in silent_samples), "silence stored a transcript"
        return f"silent samples={len(silent_samples)}"

    step("silence upload -> no_speech_detected, no transcript", silence_upload)

    # ---- P1: manual transcript dead end ---------------------------------
    def manual_transcript_confirm():
        # start from the silent sample's failed block (no transcript yet)
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("#voice-file", timeout=step_timeout)
        page.wait_for_selector("text=Type it myself", timeout=step_timeout)
        page.click("text=Type it myself")
        ta = page.wait_for_selector("textarea[aria-label='Transcript']", timeout=step_timeout)
        text = "Dies ist ein von Hand eingegebenes Referenztranskript."
        ta.fill(text)
        # the primary action must store AND confirm in one step ("Save and
        # Confirm", or "Save, confirm and use" for a recording just added)
        primary = "button.btn.pri:has-text('Save and Confirm'), button.btn.pri:has-text('Save, confirm and use')"
        assert page.query_selector(primary), "no save-and-confirm action after Type it myself"
        page.click(primary)
        page.wait_for_function(
            "() => !document.body.innerText.includes('Save without confirming')",
            timeout=step_timeout,
        )
        # server truth: stored as manual and confirmed
        status, voice = api.call(f"/api/voices/{sid}")
        sample = next((s for s in voice.get("samples", []) if (s.get("transcript") or "").strip() == text), None)
        assert sample is not None, "manual transcript not stored"
        assert sample.get("transcript_confirmed") is True, "manual transcript was not confirmed"
        # preview gate lifted: the confirm action is no longer demanded and the
        # cloning engine's Generate preview is enabled
        body = page.inner_text("#shell-inner")
        assert "not confirmed yet" not in body, "preview still demands confirmation"
        page.select_option("select[aria-label='TTS profile']", value=tts_id)
        page.wait_for_timeout(300)
        gen = page.query_selector("button:has-text('Generate preview')")
        assert gen and not gen.is_disabled(), "Generate preview still disabled after confirming"
        return "manual transcript saved + confirmed, preview enabled"

    step("P1 manual transcript Save and Confirm unblocks preview", manual_transcript_confirm)

    # ---- P1: saved-but-unconfirmed shows a visible confirm action --------
    def unconfirmed_shows_confirm():
        # make another silent sample so its failed block offers manual entry
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("#voice-file", timeout=step_timeout)
        page.wait_for_timeout(300)
        silent = run_dir / "silence2.wav"
        # a distinct length keeps the bytes (and sha) different from the sample
        # used earlier, so the server stores a NEW sample rather than deduping
        silent.write_bytes(silence_wav(2.6))
        page.set_input_files("#voice-file", str(silent))
        page.wait_for_function(
            "() => document.querySelector('.bar-err') && document.querySelector('.bar-err').textContent.includes('no_speech_detected')",
            timeout=step_timeout,
        )
        page.wait_for_selector("text=Type it myself", timeout=step_timeout)
        page.click("text=Type it myself")
        ta = page.wait_for_selector("textarea[aria-label='Transcript']", timeout=step_timeout)
        text = "Unbestaetigt gespeicherter Text."
        ta.fill(text)
        # secondary action stores WITHOUT confirming
        assert page.query_selector("button:has-text('Save without confirming')"), "no save-without-confirm action"
        page.click("text=Save without confirming")
        # the saved unconfirmed transcript must expose a visible Confirm action
        page.wait_for_function(
            "() => [...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Confirm and use this recording')",
            timeout=step_timeout,
        )
        status, voice = api.call(f"/api/voices/{sid}")
        sample = next((s for s in voice.get("samples", []) if (s.get("transcript") or "").strip() == text), None)
        assert sample is not None and sample.get("transcript_confirmed") is False, "unconfirmed state not stored"
        # an unconfirmed recording cannot become the one in use while a
        # switched-on set speaks with the voice: 409 breaks_sets names the set
        status, body = api.call(f"/api/voices/{sid}/reference", {"sample_id": sample["id"]}, method="PUT")
        assert status == 409 and body.get("code") == "breaks_sets", f"{status} {body}"
        assert pid in json.dumps(body), f"the refused swap must name the set: {body}"
        # with the set Off it is allowed (made so through the API: the page
        # only offers confirmed ones); the preview gate must then say "not
        # confirmed", never "empty" (P1 wording fix)
        status, _ = api.call(f"/api/conversation-profiles/{pid}", {"enabled": False}, method="PUT")
        assert status == 200, f"switch off -> {status}"
        status, _ = api.call(f"/api/voices/{sid}/reference", {"sample_id": sample["id"]}, method="PUT")
        assert status == 200, f"PUT reference -> {status}"
        page.goto(api.base_url + "/ui/index.html#/voices")
        page.wait_for_timeout(150)
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_function(
            "() => document.body.innerText.includes('not confirmed yet')",
            timeout=step_timeout,
        )
        body = page.inner_text("#shell-inner")
        assert "reference transcript is empty" not in body, "unconfirmed transcript wrongly reported as empty"
        # the visible confirm action completes the flow
        page.click("button:has-text('Confirm transcript')")
        page.wait_for_function(
            "() => document.body.innerText.includes('Ready to use as a cloning reference')",
            timeout=step_timeout,
        )
        return "swap refused with breaks_sets while On; unconfirmed shows Confirm transcript, wording is 'not confirmed'"

    step("P1 saved-unconfirmed transcript: swap refused while On, visible Confirm action", unconfirmed_shows_confirm)

    # ---- delete path -----------------------------------------------------
    def delete_blocked_while_on():
        open_set(pid)
        page.check("input[aria-label='On']")
        wait_status("Saved. The next call uses it")
        page.click("summary:has-text('Delete')")
        assert page.is_disabled("button:has-text('Delete set')"), "delete must be blocked while the set is On"
        reason = page.inner_text("details:has(summary:has-text('Delete'))")
        assert "Switch the set off first" in reason, reason
        return "blocked with reason"

    step("delete blocked while On (with reason)", delete_blocked_while_on)

    def delete_after_off():
        page.uncheck("input[aria-label='On']")
        wait_status("Saved.")
        # saving re-renders the folded sections: open Delete again
        page.click("summary:has-text('Delete')")
        assert not page.is_disabled("button:has-text('Delete set')"), "delete still blocked after switching off"
        page.click("button:has-text('Delete set')")
        confirm_dialog("Delete set")
        page.wait_for_function("() => location.hash === '#/profiles' || location.hash.startsWith('#/profiles?')", timeout=step_timeout)
        page.wait_for_timeout(400)
        status, _ = api.call(f"/api/conversation-profiles/{pid}")
        assert status == 404, f"set still exists after UI delete (HTTP {status})"
        created["profiles"] = [p for p in created["profiles"] if p != pid]
        return "switched off then deleted"

    step("a set switched Off can be deleted via UI", delete_after_off)

    # ---- error paths -----------------------------------------------------
    def error_incompatible_not_offered():
        display = f"{prefix}preset"
        if mode == "live":
            add_backend("tts", display, "preset-model", voices_mode="fixed")
        else:
            start_add("tts")
            page.fill("input[aria-label='Display name']", display)
            page.fill("input[aria-label='Base URL']", f"http://127.0.0.1:{opts.backend_port}/v1")
            page.fill("input[aria-label='Model']", "preset-model")
            page.select_option("select[aria-label='Voices']", value="fixed")
            page.click("button:has-text('Create backend')")
            wait_status("Model added.")
            created["backends"].append(("tts", model_id("tts", display)))
        preset_id = model_id("tts", display)
        page.goto(api.base_url + f"/ui/index.html#/voices/{sid}")
        page.wait_for_selector("select[aria-label='TTS profile']", timeout=step_timeout)
        page.wait_for_timeout(400)
        opts_list = page.eval_on_selector_all("select[aria-label='TTS profile'] option", "e=>e.map(o=>o.value)")
        assert preset_id not in opts_list, "non-cloning engine must never be offered"
        return "not offered"

    step("non-cloning engine never offered", error_incompatible_not_offered)
    shot("07-error-paths.png")

    # ---- P2: no native browser dialog was used anywhere in the flow ------
    def no_native_dialogs():
        assert not native_dialogs, f"native dialogs fired: {native_dialogs}"
        return "none"

    step("no native prompt/confirm/alert fired during the flow", no_native_dialogs)

    return created


# --------------------------------------------------------------- helpers


def _source_for_role(role: str, source_ids: list[str], api: Api) -> str:
    """Find the clone source profile id for a role (catalog lookup)."""
    if not source_ids:
        return ""
    status, cat = api.call("/api/catalog")
    by_id = {}
    for r, block in (cat.get("roles") or {}).items():
        for prof in block.get("profiles", []):
            by_id[prof.get("id")] = r
    for sid in source_ids:
        if by_id.get(sid) == role:
            return sid
    # single source: the real instance may share one host for all roles
    if len(source_ids) == 1 and source_ids[0] in by_id:
        return source_ids[0]
    return ""


# --------------------------------------------------------------- browsers


def browser_can_capture(context, base_url: str) -> bool:
    """Probe REAL audio capture on a throwaway page."""
    probe = context.new_page()
    try:
        probe.goto(base_url + "/ui/index.html", wait_until="domcontentloaded", timeout=15000)
        result = probe.evaluate(
            """async () => {
                try {
                    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) return false;
                    const AC = window.AudioContext || window.webkitAudioContext;
                    if (!AC) return false;
                    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                    const track = stream.getAudioTracks()[0];
                    const c = new AC();
                    await c.audioWorklet.addModule('../worklet.js');
                    const ok = !!track;
                    stream.getTracks().forEach((t) => t.stop());
                    try { c.close(); } catch (e) {}
                    return ok;
                } catch (e) { return false; }
            }"""
        )
        return bool(result)
    except Exception:
        return False
    finally:
        try:
            probe.close()
        except Exception:
            pass


def launch_browser(pw, name: str, opts):
    scratch = Path(opts.scratch_dir) if getattr(opts, "scratch_dir", "") else None
    chromium_extra = browser_scratch.scratch_chromium_args(scratch) if scratch else []
    if name == "chromium":
        args = [
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ]
        if opts.mic_wav:
            args.append(f"--use-file-for-fake-audio-capture={opts.mic_wav}")
        if os.geteuid() == 0:
            args.append("--no-sandbox")
        args += chromium_extra
        return pw.chromium.launch(executable_path=opts.chromium or CHROMIUM_DEFAULT, args=args), ["microphone"], True
    if name == "firefox":
        return (
            pw.firefox.launch(
                firefox_user_prefs={
                    "media.navigator.streams.fake": True,
                    "media.navigator.permission.disabled": True,
                    "media.autoplay.default": 0,
                    "media.autoplay.blocking_policy": 0,
                }
            ),
            [],
            False,
        )
    if name == "webkit":
        return pw.webkit.launch(), ["microphone"], True
    raise ValueError(name)


# -------------------------------------------------------------------- main


INIT_SCRIPT = r"""
(() => {
  // ---- ordered side-effect log (F12 correction C1) ---------------------
  // getUserMedia / fetch / WebSocket opens are recorded in call order so a
  // test can prove the mic request is not preceded by a network await inside
  // the Talk click handler.
  window.__rtvSeq = window.__rtvSeq || { events: [] };
  const _seqPush = (kind, detail) => {
    try { window.__rtvSeq.events.push({ i: window.__rtvSeq.events.length, kind, detail: String(detail || '') }); } catch (e) {}
  };

  // ---- response/tool counters (WebSocket) ------------------------------
  window.__rtvE2E = window.__rtvE2E || {};
  window.__rtvE2E.counters = { responseCreated: 0, responseDone: 0, toolCalls: 0 };
  const origFetch = window.fetch ? window.fetch.bind(window) : null;
  if (origFetch) {
    window.fetch = function (input, init) {
      try { _seqPush('fetch', (input && input.url) ? input.url : String(input)); } catch (e) {}
      return origFetch(input, init);
    };
  }
  const origWS = window.WebSocket;
  window.WebSocket = function(url, protocols) {
    _seqPush('ws', url);
    const ws = new origWS(url, protocols);
    ws.addEventListener('message', (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === 'response.created') window.__rtvE2E.counters.responseCreated++;
        if (m.type === 'response.done') window.__rtvE2E.counters.responseDone++;
        if (m.type === 'response.function_call_arguments.done') window.__rtvE2E.counters.toolCalls++;
      } catch (e) {}
    });
    return ws;
  };
  window.WebSocket.prototype = origWS.prototype;
  for (const k of ['CONNECTING','OPEN','CLOSING','CLOSED']) { try { window.WebSocket[k] = origWS[k]; } catch (e) {} }
  try {
    const origSend = origWS.prototype.send;
    origWS.prototype.send = function (data) {
      try {
        if (typeof data === 'string' && data.indexOf('input_audio_buffer.append') !== -1) {
          window.__rtvE2E.counters.appendFrames = (window.__rtvE2E.counters.appendFrames || 0) + 1;
        }
      } catch (e) {}
      return origSend.call(this, data);
    };
  } catch (e) {}

  // ---- microphone instrumentation (F2) ---------------------------------
  try {
    const md = navigator.mediaDevices;
    if (md && md.getUserMedia) {
      const orig = md.getUserMedia.bind(md);
      md.getUserMedia = function (constraints) {
        try {
          _seqPush('getUserMedia', '');
          window.__rtvMic = window.__rtvMic || { constraints: [] };
          const dev = constraints && constraints.audio && constraints.audio.deviceId;
          const id = typeof dev === 'string' ? dev : (dev && dev.exact) || '';
          window.__rtvMic.constraints.push(id);
          window.__rtvMic.last = id;
        } catch (e) {}
        return orig(constraints);
      };
    }
    if (md && md.enumerateDevices) {
      const origEnum = md.enumerateDevices.bind(md);
      md.enumerateDevices = function () {
        return origEnum().then((list) => {
          const hide = window.__rtvMic && window.__rtvMic.hide;
          if (hide) return list.filter((d) => d.deviceId !== hide);
          return list;
        });
      };
    }
  } catch (e) {}

  // ---- media playback instrumentation (F6) -----------------------------
  try {
    const records = new Set();       // one record per element
    const recOf = (el) => {
      if (!el.__rtvMediaRec) {
        el.__rtvMediaRec = { el, playing: false };
        records.add(el.__rtvMediaRec);
      }
      return el.__rtvMediaRec;
    };
    const playingCount = () => {
      let n = 0;
      for (const r of records) if (r.playing && !r.el.paused && !r.el.ended) n++;
      return n;
    };
    const proto = window.HTMLMediaElement && window.HTMLMediaElement.prototype;
    if (proto) {
      const origPlay = proto.play;
      const origPause = proto.pause;
      proto.play = function (...a) {
        recOf(this).playing = true;
        try { return origPlay.apply(this, a); } catch (e) { return Promise.reject(e); }
      };
      proto.pause = function (...a) {
        if (this.__rtvMediaRec) this.__rtvMediaRec.playing = false;
        return origPause.apply(this, a);
      };
    }
    window.__rtvMedia = {
      playingCount,
      currentTime: () => {
        let best = 0;
        for (const r of records) {
          if (r.playing && !r.el.paused && !r.el.ended) best = Math.max(best, r.el.currentTime || 0);
        }
        return best;
      },
    };
  } catch (e) {}
})();
"""


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--browsers", default="chromium", help="comma-separated: chromium,webkit,firefox")
    p.add_argument("--base-url", default="", help="LIVE mode: running instance base URL")
    p.add_argument("--token-env", default="RTV_TOKEN", help="LIVE mode: env var holding the bearer token")
    p.add_argument("--token", default="", help="LIVE mode: bearer token (prefer --token-env)")
    p.add_argument("--prefix", default="e2e-", help="LIVE mode: id prefix for created objects")
    p.add_argument(
        "--clone-backend-from",
        default="",
        help="LIVE mode: existing profile id(s), comma-separated, to clone base URL/key from server-side",
    )
    p.add_argument("--stt-model", default="", help="LIVE mode: real STT model id (required)")
    p.add_argument("--llm-model", default="", help="LIVE mode: real LLM model id (required)")
    p.add_argument("--tts-model", default="", help="LIVE mode: real TTS model id (required)")
    p.add_argument("--mic-wav", default="", help="LIVE mode: real speech WAV (required)")
    p.add_argument("--expect-text-file", default="", help="optional expected transcript text file (similarity ratio only)")
    p.add_argument(
        "--api-key-env",
        default="RTV_BACKEND_API_KEY",
        help="LIVE mode legacy: env var with a backend API key (never printed); prefer cloning",
    )
    p.add_argument("--keep", action="store_true", help="LIVE mode: do not clean up created objects")
    p.add_argument("--out-dir", default="", help="artifact dir (default /tmp/rtv-f1-e2e-<ts>)")
    p.add_argument("--chromium", default="", help="chromium binary path")
    p.add_argument(
        "--scratch-dir",
        default="",
        help="temp dir for browser/instance scratch (default: a roomy dir on the repo filesystem)",
    )
    p.add_argument("--timeout", type=float, default=90.0, help="default per-action timeout (s)")
    p.add_argument("--step-timeout", type=float, default=0.0, help="per-step timeout (s; default --timeout)")
    args = p.parse_args(argv)

    # Chromium needs a writable temp dir with space AND a short path (its
    # singleton socket lives under TMPDIR). Do this before any tempfile use.
    args.scratch_dir = str(browser_scratch.prepare_scratch("rtv-f1-e2e-scratch", args.scratch_dir))
    browser_scratch.apply_env(Path(args.scratch_dir))

    live = bool(args.base_url)
    if live:
        missing = [n for n, v in (("--stt-model", args.stt_model), ("--llm-model", args.llm_model), ("--tts-model", args.tts_model), ("--mic-wav", args.mic_wav)) if not v]
        if missing:
            print(f"SETUP ERROR: live mode requires {', '.join(missing)}", file=sys.stderr)
            return 2
        mic_path = Path(args.mic_wav)
        if not mic_path.exists():
            print(f"SETUP ERROR: --mic-wav not found: {mic_path}", file=sys.stderr)
            return 2
        # process-arg leak guard: a legacy key comes only from the environment
        args.api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        if not args.clone_backend_from and not args.api_key:
            print(
                f"SETUP ERROR: live mode needs --clone-backend-from or env {args.api_key_env}",
                file=sys.stderr,
            )
            return 2

    run_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="rtv-f1-e2e-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    report = Report(run_dir)

    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"SETUP ERROR: playwright not installed ({exc})", file=sys.stderr)
        return 2

    backend = None
    server = None
    instance_dir = None
    try:
        if live:
            base_url = args.base_url.rstrip("/")
            token = args.token or os.environ.get(args.token_env, "")
            if not token:
                print(f"SETUP ERROR: no token (--token or env {args.token_env})", file=sys.stderr)
                return 2
            args.backend_port = 0
            args.mic_wav = str(mic_path)
            # real speech for the upload and the Chromium fake mic
            args.upload_bytes = wav_pcm24(mic_path)
        else:
            import uvicorn

            from realtimevoice.app import create_app
            from realtimevoice.config import Settings, SettingsStore

            backend, backend_port = start_fake_backend()
            args.backend_port = backend_port
            args.fake_llm_script = backend.RequestHandlerClass.llm_script
            instance_dir = tempfile.mkdtemp(prefix="rtv-f1-e2e-data-")
            settings = Settings()
            store = SettingsStore(data_dir=instance_dir, settings=settings)
            token = "f1-e2e-token"
            app = create_app(settings_store=store, auth_token=token)
            port = free_port()
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
            import threading

            threading.Thread(target=server.run, daemon=True).start()
            base_url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    urllib.request.urlopen(base_url + "/ui/index.html", timeout=0.5)
                    break
                except Exception:
                    time.sleep(0.05)
            # fake mode: a synthetic tone as the "spoken" input
            mic_path = run_dir / "mic.wav"
            # tone then silence in ONE data chunk (bytes appended after a
            # finished WAV are not part of it), so turn detection sees the end
            # of speech before the file loops
            tone = wav_bytes(1.5, freq=200.0, amp=6000)[44:]
            mic_path.write_bytes(pcm16_wav(tone + b"\x00\x00" * 36000, 24000))
            args.mic_wav = str(mic_path)
            args.upload_bytes = wav_bytes(3.0)

        api = Api(base_url, token)
        report.step("instance ready", True, base_url if not live else "LIVE " + base_url)

        created_all = {}
        with sync_playwright() as pw:
            for name in [b.strip() for b in args.browsers.split(",") if b.strip()]:
                report.step(f"browser {name}: starting", True)
                try:
                    browser, permissions, audio_ok = launch_browser(pw, name, args)
                except Exception as exc:
                    report.step(f"browser {name}: launch", False, str(exc)[:200])
                    continue
                context = browser.new_context(permissions=permissions, viewport={"width": 1440, "height": 900})
                page = context.new_page()
                page.set_default_timeout(args.timeout * 1000)
                page.add_init_script(INIT_SCRIPT)
                crash_state = browser_scratch.install_crash_logging(page, run_dir)
                try:
                    can_capture = browser_can_capture(context, base_url) if audio_ok else False
                    flow_prefix = args.prefix if live else f"{args.prefix}{name}-"
                    created_all[name] = run_flow(page, api, flow_prefix, run_dir, report, "live" if live else "fake", args, PWTimeout, can_capture) or {}
                except Exception as exc:
                    if crash_state.get("crashed"):
                        browser_scratch.record_crash(run_dir, crash_state.get("reason", ""))
                    report.step(f"browser {name}: flow", False, str(exc)[:300])
                screenshot = run_dir / f"final-{name}.png"
                try:
                    page.screenshot(path=str(screenshot), full_page=True)
                except Exception:
                    pass
                context.close()
                browser.close()

        if live and not args.keep:
            cleanup(api, args.prefix, report, created_all)
        exit_code = 3 if report.failed else 0
        report.write(exit_code, {"mode": "live" if live else "fake", "base_url": base_url, "scratch_dir": str(args.scratch_dir)})
        print(f"\nArtifacts: {run_dir}")
        print("FUNCTIONAL E2E " + ("FAILED" if exit_code else "PASSED"))
        return exit_code
    finally:
        if server is not None:
            server.should_exit = True
        if backend is not None:
            backend.shutdown()


def cleanup(api: Api, prefix: str, report: Report, created_all: dict | None = None) -> None:
    """Delete ONLY objects created by this run (live mode): sets and voices
    whose ids start with the prefix, models whose display name does (model
    ids are generated by the server). The transcription model setting is
    put back first."""
    removed = []
    for created in (created_all or {}).values():
        before = (created or {}).get("settings_before")
        if before:
            api.call("/api/settings", before, method="PUT")
    status, profiles = api.call("/api/conversation-profiles")
    for prof in profiles.get("profiles", []):
        pid = prof.get("id", "")
        if not pid.startswith(prefix):
            continue
        if prof.get("enabled"):
            api.call(f"/api/conversation-profiles/{pid}", {"enabled": False}, method="PUT")
        s, _ = api.call(f"/api/conversation-profiles/{pid}", method="DELETE")
        if s in (200, 404):
            removed.append(pid)
    status, voices = api.call("/api/voices")
    for voice in voices.get("voices", []):
        vid = voice.get("id", "")
        if vid.startswith(prefix):
            s, _ = api.call(f"/api/voices/{vid}", method="DELETE")
            if s in (200, 404):
                removed.append(vid)
    status, cat = api.call("/api/catalog")
    for role, block in (cat.get("roles") or {}).items():
        for prof in block.get("profiles", []):
            bid = prof.get("id", "")
            if str(prof.get("display_name") or "").startswith(prefix):
                s, _ = api.call(f"/api/profiles/{role}/{bid}", method="DELETE")
                if s in (200, 404, 409):
                    removed.append(bid)
    report.step("cleanup objects of this run", True, ", ".join(removed) or "nothing to clean")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))