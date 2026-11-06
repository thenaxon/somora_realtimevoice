#!/usr/bin/env python3
"""Headless Chromium smoke test for the realtimevoice management UI (Lane C).

Starts the app on a RANDOM free localhost port with a throwaway temp data dir
and a local fake OpenAI-HTTP backend, then drives the real UI end to end:

  login -> Models (add a model with the form, which has no Id field; Language
  as; Duplicate asks for a name) -> Settings (transcription model) -> create
  voice -> upload a synthetic WAV -> transcribe (fake STT) -> confirm
  transcript -> Voices list (own + built-in voices) -> create a set, choose
  how it listens, thinks, its voice, speaking model and language -> save ->
  switch it On -> Settings (set for other apps) -> Test Console "Running now"
  route display.

A synthetic microphone pass (--use-fake-device-for-media-stream fed from a
generated WAV) exercises the free-recording path; the deterministic upload
path is what the assertions are built on.

Evidence (screenshots + JSON report + console log) goes into a run directory
under /tmp; nothing private is written to the repository. The auth token is
generated for this instance and never printed or screenshotted.

Exit codes:
  0 = all steps passed
  3 = one or more step failures
  2 = setup error (no chromium/playwright, page unreachable, login failed)
"""

from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import re
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import browser_scratch  # noqa: E402

AUTH_TOKEN = "ui-smoke-token"
FAKE_TRANSCRIPT = "Also heute war echt ein komischer Tag und ich stand im Regen."


# ------------------------------------------------------------------- helpers

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_wav(path: Path, seconds: float = 3.0, rate: int = 24000, freq: float = 220.0) -> None:
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *(int(6000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))))


class FakeBackend(http.server.BaseHTTPRequestHandler):
    """Local OpenAI-HTTP shaped backend: STT transcribe, TTS speech, /models."""

    def log_message(self, *args):  # silence
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            # list every model id the smoke's profiles reference: the readiness
            # probe (P1) fails with model_not_listed when a configured model is
            # absent from a non-empty /models answer, so the fixture must offer
            # exactly what the smoke creates/uses.
            ids = ("fake-model", "stt-base", "llm-base", "preset-model", "cloning-model")
            self._json(200, {"data": [{"id": i} for i in ids]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(length)
        if self.path.endswith("/audio/transcriptions"):
            self._json(200, {"text": FAKE_TRANSCRIPT})
        elif self.path.endswith("/audio/speech"):
            # 200 ms of mono PCM16 WAV at 24 kHz
            rate, n = 24000, 4800
            pcm = struct.pack(f"<{n}h", *(int(2000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)))
            buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(buf.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            data = Path(buf.name).read_bytes()
            os.unlink(buf.name)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Rtv-First-Pcm-Ms", "12")
            self.send_header("X-Rtv-Total-Ms", "34")
            self.send_header("X-Rtv-Audio-Ms", "200")
            self.send_header("X-Rtv-Model", "fake-model")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.endswith("/chat/completions"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hallo!\"},\"finish_reason\":\"stop\"}]}\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._json(404, {"error": {"message": "not found"}})


def start_fake_backend() -> tuple[http.server.ThreadingHTTPServer, int]:
    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), FakeBackend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def api_get(base_url: str, path: str) -> dict:
    import urllib.request

    req = urllib.request.Request(base_url + path, headers={"Authorization": f"Bearer {AUTH_TOKEN}"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read() or b"{}")


def base_url_backend(ids: dict) -> str:
    return ids["_backend_url"]


class Report:
    def __init__(self):
        self.steps = []

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(not s["ok"] for s in self.steps)


# --------------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default=None, help="chromium binary (default: playwright's)")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out-dir", default=None, help="artifact dir (default /tmp/rtv-ui-smoke-<ts>)")
    parser.add_argument(
        "--scratch-dir",
        default="",
        help="temp dir for browser/instance scratch (default: a roomy dir on the repo filesystem)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the instance running (debug)")
    args = parser.parse_args(argv)

    # Chromium needs a writable temp dir with space AND a short path (its
    # singleton socket lives under TMPDIR). Do this before any tempfile use.
    args.scratch_dir = str(browser_scratch.prepare_scratch("rtv-ui-smoke-scratch", args.scratch_dir))
    browser_scratch.apply_env(Path(args.scratch_dir))

    run_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="rtv-ui-smoke-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    console_path = run_dir / "console.log"
    report_path = run_dir / "report.json"
    console_lines: list[str] = []

    def log_console(kind: str, text: str) -> None:
        console_lines.append(f"[{kind}] {text}")

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except Exception as exc:  # pragma: no cover
        print(f"SETUP ERROR: playwright is not installed ({exc}).", file=sys.stderr)
        return 2

    # ---- backend + app -----------------------------------------------------
    backend, backend_port = start_fake_backend()

    import uvicorn
    from realtimevoice.app import create_app
    from realtimevoice.config import Settings, SettingsStore

    data_dir = tempfile.mkdtemp(prefix="rtv-ui-smoke-data-")
    store = SettingsStore(data_dir=data_dir, settings=Settings())
    app = create_app(settings_store=store, auth_token=AUTH_TOKEN)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base_url = f"http://127.0.0.1:{port}"
    import urllib.request

    for _ in range(50):
        try:
            urllib.request.urlopen(base_url + "/ui/index.html", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)

    def api(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(req, timeout=5).read() or b"{}")

    # the models an operator would have added; ids are generated by the server
    backend_url = f"http://127.0.0.1:{backend_port}/v1"
    ids = {
        "stt": api("/api/profiles/stt", {"display_name": "STT fixture", "base_url": backend_url, "model": "stt-base"})["id"],
        "llm": api("/api/profiles/llm", {"display_name": "LLM fixture", "base_url": backend_url, "model": "llm-base"})["id"],
        "preset": api(
            "/api/profiles/tts",
            {
                "display_name": "Preset TTS",
                "base_url": backend_url,
                "model": "preset-model",
                "language_format": "name",
                "capabilities": {"voices": ["ryan", "serena"], "voice_cloning": "none", "native_voice_labels": "honoured", "evidence": "probe"},
            },
        )["id"],
        "cloning": api(
            "/api/profiles/tts",
            {
                "display_name": "Cloning TTS",
                "base_url": backend_url,
                "model": "cloning-model",
                "capabilities": {"voice_cloning": "reference_audio", "evidence": "probe"},
            },
        )["id"],
    }
    ids["_backend_url"] = backend_url

    report = Report()
    report.step("instance started", True, f"{base_url} (data {data_dir})")
    report.step("models created without an id get one from the server", all(ids.values()), json.dumps(ids))

    mic_wav = run_dir / "synthetic-mic.wav"
    write_wav(mic_wav, 3.0)
    upload_wav = run_dir / "upload.wav"
    write_wav(upload_wav, 2.0, freq=330.0)

    launch_args = [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-audio-capture={mic_wav}",
        "--autoplay-policy=no-user-gesture-required",
    ]
    if os.geteuid() == 0:
        launch_args.append("--no-sandbox")
    if args.browser:
        launch_args.append("--browser-exe")
    launch_args += browser_scratch.scratch_chromium_args(Path(args.scratch_dir), run_dir / "chromium.log")

    exit_code = 0
    try:
        with sync_playwright() as pw:
            exe = args.browser
            browser = pw.chromium.launch(executable_path=exe, args=launch_args)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.set_default_timeout(args.timeout * 1000)
            page.on("console", lambda m: log_console(m.type, m.text))
            page.on("pageerror", lambda e: log_console("pageerror", str(e)))
            page.on("requestfailed", lambda r: log_console("requestfailed", f"{r.url} {r.failure}"))
            crash_state = browser_scratch.install_crash_logging(page, run_dir, run_dir / "chromium.log")

            def shot(name: str) -> None:
                try:
                    page.screenshot(path=str(run_dir / name), full_page=True)
                except Exception:
                    pass

            try:
                run_flow(page, base_url, run_dir, report, shot, PWTimeout, ids)
            except Exception as exc:
                if crash_state.get("crashed"):
                    browser_scratch.record_crash(run_dir, crash_state.get("reason", ""), run_dir / "chromium.log")
                report.step("flow completed without exception", False, str(exc))
                shot("99-failure.png")
            shot("99-final.png")
            browser.close()
    except Exception as exc:
        print(f"SETUP ERROR: could not launch chromium: {exc}", file=sys.stderr)
        return 2
    finally:
        if not args.keep:
            server.should_exit = True
            backend.shutdown()

    console_path.write_text("\n".join(console_lines))
    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "run_dir": str(run_dir),
        "steps": report.steps,
        "console_errors": [l for l in console_lines if l.startswith("[error]") or l.startswith("[pageerror]")],
    }
    report_path.write_text(json.dumps(result, indent=2))
    print(f"\nArtifacts: {run_dir}")
    if report.failed:
        exit_code = 3
    print("UI SMOKE PASSED" if exit_code == 0 else "UI SMOKE FAILED")
    return exit_code


def run_flow(page, base_url, run_dir, report, shot, PWTimeout, ids):
    # ---- login -------------------------------------------------------------
    page.goto(base_url + "/ui/index.html", wait_until="domcontentloaded")
    page.fill("#token", AUTH_TOKEN)
    page.click("#btn-login")
    page.wait_for_selector("#shell-inner h1", timeout=15000)
    heading = page.text_content("#shell-inner h1")
    report.step("login lands on Sets", heading == "Sets", heading)
    shot("01-profiles-empty.png")

    # narrow view: the burger + section chips replace the sidebar and the
    # active chip marks the current section
    page.set_viewport_size({"width": 430, "height": 900})
    page.wait_for_timeout(250)
    try:
        burger_visible = page.is_visible(".burger")
        active_chip = page.text_content(".mchip.on") or ""
        report.step("narrow view shows burger + active section chip", burger_visible and active_chip.strip() == "Sets", active_chip.strip())
        shot("01b-profiles-narrow.png")
    except Exception as exc:
        report.step("narrow view shows burger + active section chip", False, str(exc))
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(150)

    # ---- Models: the form has no Id field; the server names the model -----
    page.click('.navi[data-nav="backends"]')
    try:
        page.wait_for_selector('.mcol[data-role="stt"]', timeout=15000)
        tabs = page.query_selector_all("#shell-inner [role=tab]")
        report.step("Models page shows the board without tabs", len(tabs) == 0, f"tabs={len(tabs)}")
        page.click('.mcol[data-role="stt"] button:has-text("＋ Add")')
        page.wait_for_selector('input[aria-label="Display name"]', timeout=10000)
        labels = page.eval_on_selector_all(
            "#shell-inner .fld > label", "els => els.filter(e => e.offsetParent !== null).map(e => e.textContent.trim())"
        )
        id_inputs = page.query_selector_all(
            "#shell-inner input[aria-label='Id'], #shell-inner input[aria-label='ID'], #shell-inner input[aria-label='Model id']"
        )
        report.step(
            "model form has no Id field",
            not id_inputs and not any(l.lower() in ("id", "model id") for l in labels),
            ", ".join(labels)[:200],
        )
        lang_opts = page.eval_on_selector_all("select[aria-label='Language as'] option", "els => els.map(e => e.value)")
        report.step("model form asks how the language is sent", lang_opts == ["code", "name", "none"], str(lang_opts))
        page.fill('input[aria-label="Display name"]', "Second listener")
        page.fill('input[aria-label="Base URL"]', base_url_backend(ids))
        page.fill('input[aria-label="Model"]', "fake-model")
        page.select_option("select[aria-label='Language as']", value="none")
        page.click('#shell-inner button:has-text("Create backend")')
        page.wait_for_function("() => document.body.innerText.includes('Model added.')", timeout=15000)
        added = [r for r in api_get(base_url, "/api/catalog")["roles"]["stt"]["profiles"] if r.get("display_name") == "Second listener"]
        report.step(
            "a model added in the form gets a generated id and its language format",
            len(added) == 1 and bool(added[0].get("id")) and added[0].get("language_format") == "none",
            json.dumps([{k: r.get(k) for k in ("id", "language_format")} for r in added]),
        )
        shot("02a-models-added.png")
        # Duplicate asks for the copy's name
        page.click('#shell-inner button:has-text("Duplicate")')
        page.wait_for_selector(".dlg input[aria-label='Name of the copy']", timeout=10000)
        page.fill(".dlg input[aria-label='Name of the copy']", "Second listener copy")
        page.click(".dlg button:has-text('Duplicate')")
        page.wait_for_function("() => document.body.innerText.includes('Duplicated.')", timeout=15000)
        names = [r.get("display_name") for r in api_get(base_url, "/api/catalog")["roles"]["stt"]["profiles"]]
        report.step("Duplicate asks for a name and stores the copy under it", "Second listener copy" in names, ", ".join(map(str, names)))
    except Exception as exc:
        report.step("Models page flow", False, str(exc)[:200])
        shot("02a-models-failure.png")

    # ---- Settings: the model that transcribes recordings -------------------
    page.click('.navi[data-nav="settings"]')
    try:
        page.wait_for_selector("select[aria-label='Transcribes recordings']", timeout=15000)
        heading = page.text_content("#shell-inner h1")
        report.step("Settings page opens", heading == "Settings", heading)
        page.select_option("select[aria-label='Transcribes recordings']", value=ids["stt"])
        page.wait_for_function("() => document.body.innerText.includes('Model that transcribes recordings saved.')", timeout=10000)
        stored = (api_get(base_url, "/api/settings").get("voice_library") or {}).get("transcription_stt_profile")
        report.step("transcription model saved from Settings", stored == ids["stt"], str(stored))
        has_defaults = page.is_visible("textarea[aria-label='Default instructions']") and page.is_visible(
            "button:has-text('Save defaults')"
        )
        report.step("Settings page holds the defaults for calls", has_defaults)
        shot("02b-settings.png")
    except Exception as exc:
        report.step("Settings page flow", False, str(exc)[:200])
        shot("02b-settings-failure.png")

    # ---- create voice ------------------------------------------------------
    page.click('.navi[data-nav="voices"]')
    page.wait_for_timeout(300)

    page.click("text=＋ New voice")
    try:
        page.wait_for_selector(".dlg", timeout=10000)
        page.fill(".dlg input[aria-label='Voice id']", "smoke-voice")
        page.fill(".dlg input[aria-label='Display name']", "Smoke voice")
        page.click(".dlg button:has-text('Create voice')")
        page.wait_for_selector(".dlg", state="detached", timeout=10000)
        page.wait_for_function("() => location.hash.startsWith('#/voices/')", timeout=10000)
        vid = page.evaluate("location.hash").split("/")[-1]
        report.step("voice editor opened", vid == "smoke-voice", vid)
    except PWTimeout:
        report.step("voice editor opened", False, page.evaluate("location.hash"))
        return
    shot("02-voice-editor.png")

    # ---- upload + transcribe + confirm ------------------------------------
    page.set_input_files("#voice-file", str(run_dir / "upload.wav"))
    # wait for the transcript textarea prefilled with the fake STT text
    try:
        page.wait_for_function(
            "() => { const t=document.querySelector('textarea'); return t && t.value.includes('komischer Tag'); }",
            timeout=20000,
        )
        report.step("uploaded sample was auto-transcribed", True, FAKE_TRANSCRIPT[:40])
    except PWTimeout:
        body = page.text_content("#shell-inner") or ""
        report.step("uploaded sample was auto-transcribed", False, body[:200])
        return
    shot("03-transcript.png")

    # confirm transcript
    page.click("text=Confirm transcript")
    page.wait_for_function("() => document.body.innerText.includes('Ready to use as a cloning reference')", timeout=15000)
    report.step("transcript confirmed", True)
    shot("04-confirmed.png")

    # the recording in use is shown on its own, without a radio to pick it
    try:
        state = page.evaluate(
            """() => ({
                radios: document.querySelectorAll('.radio, [role=radio]').length,
                others: document.querySelectorAll('.srow').length,
            })"""
        )
        report.step("the recording in use stands alone, no radio", state["radios"] == 0 and state["others"] == 0, str(state))
    except Exception as exc:
        report.step("the recording in use stands alone, no radio", False, str(exc)[:180])

    # ---- preview against a reference-audio engine -------------------------
    try:
        page.select_option(".block select[aria-label='TTS profile']", value=ids["cloning"])
        page.click("text=Generate preview")
        page.wait_for_function(
            "() => [...document.querySelectorAll('.mlab')].some(e => e.textContent === 'First PCM')",
            timeout=20000,
        )
        mvals = page.eval_on_selector_all(".mval", "els=>els.map(e=>e.textContent)")
        report.step("voice preview shows the three server-side measurements", len(mvals) >= 3, ", ".join(mvals))
        shot("04c-preview.png")
    except Exception as exc:
        report.step("voice preview shows the three server-side measurements", False, str(exc)[:180])

    # ---- microphone pass (synthetic device, raw PCM -> AudioWorklet -> WAV)
    try:
        page.click(".recbtn")
        page.wait_for_selector(".meter i", timeout=5000)
        report.step("microphone recording starts with a live level meter", True)
        page.wait_for_timeout(2500)
        page.click("text=Stop recording")
        # the new recording is transcribed and waits to be confirmed and used
        page.wait_for_selector("button:has-text('Confirm and use this recording')", timeout=20000)
        report.step("synthetic microphone recording stored and transcribed", True)
        shot("04b-mic-sample.png")
        page.click("button:has-text('Confirm and use this recording')")
        # the first recording moves to "Other recordings", the new one is in use
        page.wait_for_function(
            "() => document.querySelectorAll('.srow').length === 1 && !document.querySelector('[data-rtv-confirm]')",
            timeout=15000,
        )
        report.step("confirming the new recording makes it the one in use", True)
    except Exception as exc:
        report.step("synthetic microphone recording stored and transcribed", False, str(exc)[:180])

    # ---- Voices list: own voices and voices built into a model ------------
    page.click('.navi[data-nav="voices"]')
    try:
        page.wait_for_selector(".seclabel", timeout=10000)
        sections = page.eval_on_selector_all(".seclabel", "els => els.map(e => e.textContent.trim())")
        builtin = page.eval_on_selector_all("button[aria-label^='Hear ']", "els => els.map(e => e.getAttribute('aria-label'))")
        report.step(
            "Voices lists your voices and the built-in ones",
            sections[:2] == ["Your voices", "Built into a model"] and "Hear ryan" in builtin and "Hear serena" in builtin,
            f"{sections} {builtin}",
        )
        shot("04d-voices-list.png")
    except Exception as exc:
        report.step("Voices lists your voices and the built-in ones", False, str(exc)[:180])

    # ---- create a set ------------------------------------------------------
    page.click('.navi[data-nav="profiles"]')
    page.wait_for_timeout(300)
    page.click("text=＋ New set")
    page.wait_for_selector(".dlg", timeout=10000)
    page.fill(".dlg input[aria-label='Name for somora']", "smoke-profile")
    page.fill(".dlg input[aria-label='Display name']", "Smoke profile")
    page.click(".dlg button:has-text('Create set')")
    page.wait_for_selector(".dlg", state="detached", timeout=10000)
    page.wait_for_function("() => location.hash === '#/profiles/smoke-profile'", timeout=10000)
    page.wait_for_selector("select[aria-label='Listens with']", timeout=15000)
    report.step("set page opened", True, page.evaluate("location.hash"))
    off_first = not page.is_checked("input[aria-label='On']")
    report.step("a new set starts Off", off_first)

    try:
        page.select_option("select[aria-label='Listens with']", value=ids["stt"])
        page.select_option("select[aria-label='Thinks with']", value=ids["llm"])
        # the voice picker groups own voices, a model's own voices and
        # models with one voice
        groups = page.eval_on_selector_all("select[aria-label='Speaks as'] optgroup", "els => els.map(e => e.label)")
        report.step(
            "voice picker groups own and built-in voices",
            groups[:2] == ["Your voices", "Preset TTS: its own voices"],
            str(groups),
        )
        # an own voice: the speaking model follows (the only cloning model)
        page.select_option("select[aria-label='Speaks as']", value="lib:smoke-voice")
        page.wait_for_timeout(200)
        speaker = page.text_content(".setrow:has(.setlab:text-is('Speaking model')) .setctl") or ""
        report.step("choosing an own voice picks the cloning model", "Cloning TTS" in speaker, speaker.strip()[:80])
        # a built-in voice: the speaking model switches to the model that has it
        page.select_option("select[aria-label='Speaks as']", value=f"nat:{ids['preset']}:ryan")
        page.wait_for_timeout(200)
        speaker = page.text_content(".setrow:has(.setlab:text-is('Speaking model')) .setctl") or ""
        report.step("choosing a built-in voice picks its model", "Preset TTS" in speaker, speaker.strip()[:80])
        page.select_option("select[aria-label='Language']", value="en")
    except Exception as exc:
        report.step("set choices made", False, str(exc)[:200])
        shot("05-profile-editor.png")
        return
    report.step("set choices made (listens/thinks/voice/language)", True)
    try:
        page.wait_for_function("() => document.body.innerText.includes('All three models answered.')", timeout=15000)
        report.step("the set page checks the choice live", True)
    except PWTimeout:
        report.step("the set page checks the choice live", False, (page.text_content(".setissues") or "")[:160])
    shot("05-profile-editor.png")

    # accessibility basics: every visible form control in the editor has an
    # accessible name (label, aria-label, aria-labelledby, or wrapping label)
    unlabeled = page.eval_on_selector_all(
        "#shell-inner input:not([type=hidden]),#shell-inner select,#shell-inner textarea",
        """els => els.filter(e => {
            if (e.offsetParent === null) return false;
            if (e.getAttribute('aria-label') || e.getAttribute('aria-labelledby')) return false;
            if (e.closest('label')) return false;
            if (e.id && document.querySelector('label[for="' + e.id + '"]')) return false;
            return true;
        }).map(e => e.tagName + ':' + (e.getAttribute('aria-label') || ''))""",
    )
    report.step("set page form controls have accessible names", len(unlabeled) == 0, ", ".join(unlabeled)[:200])

    # Save (still Off), then switch it On, which saves with it
    page.click(".head-actions button.btn.pri:has-text('Save')")
    try:
        page.wait_for_function("() => document.querySelector('#shell-inner .statusline').textContent.trim() === 'Saved.'", timeout=15000)
        stored = api_get(base_url, "/api/conversation-profiles/smoke-profile")
        spec = stored.get("spec") or {}
        ok = (
            spec.get("stt_profile") == ids["stt"]
            and spec.get("llm_profile") == ids["llm"]
            and spec.get("tts_profile") == ids["preset"]
            and spec.get("voice", {}).get("native_voice") == "ryan"
            and spec.get("language") == "en"
            and stored.get("enabled") is False
        )
        report.step("Save stores what is on screen, still Off", ok, json.dumps(spec)[:200])
        report.step("Save is disabled when nothing changed", page.is_disabled(".head-actions button.btn.pri:has-text('Save')"))
    except PWTimeout:
        report.step("Save stores what is on screen, still Off", False, (page.text_content("#shell-inner .statusline") or "")[:200])

    page.check("input[aria-label='On']")
    try:
        page.wait_for_function(
            "() => document.querySelector('#shell-inner .statusline').textContent.startsWith('Saved. The next call uses it')",
            timeout=15000,
        )
        enabled = api_get(base_url, "/api/conversation-profiles/smoke-profile").get("enabled")
        sub = page.text_content(".h1sub") or ""
        report.step("switching On saves it and somora can call it", enabled is True and sub.startswith("On"), sub[:80])
    except PWTimeout:
        report.step("switching On saves it and somora can call it", False, (page.text_content("#shell-inner .statusline") or "")[:200])

    # change the language and save: the old version is kept
    page.select_option("select[aria-label='Language']", value="de")
    page.click(".head-actions button.btn.pri:has-text('Save')")
    try:
        page.wait_for_function("() => document.querySelector('#shell-inner .statusline').textContent.startsWith('Saved. The next call uses it')", timeout=15000)
        summary = page.eval_on_selector_all("details.fsection > summary", "els => els.map(e => e.textContent.trim())")
        versions = [t for t in summary if t.startswith("Earlier versions")]
        page.click("summary:has-text('Earlier versions')")
        use_btns = page.query_selector_all("button:has-text('Use this version')")
        report.step(
            "an earlier version is kept with 'Use this version'",
            bool(versions) and versions[0] != "Earlier versions (0)" and len(use_btns) >= 1,
            f"{versions} buttons={len(use_btns)}",
        )
        report.step(
            "More holds other apps and delete",
            any(t == "Other apps" for t in summary) and any(t == "Delete" for t in summary),
            str(summary),
        )
    except PWTimeout:
        report.step("an earlier version is kept with 'Use this version'", False, (page.text_content("#shell-inner .statusline") or "")[:200])

    # P1-2: in the narrow view the account footer must live INSIDE the shell
    # (the burger menu), never as a fixed element overlapping editor fields.
    page.set_viewport_size({"width": 430, "height": 900})
    page.wait_for_timeout(250)
    try:
        no_fixed = page.evaluate(
            """() => [...document.querySelectorAll('.spa-footer')]
                .every(e => getComputedStyle(e).position !== 'fixed')"""
        )
        overlap = page.evaluate(
            """() => {
                const foot = document.querySelector('.sidefoot');
                const field = document.querySelector('#shell-inner input, #shell-inner select, #shell-inner textarea');
                if (!foot || !field) return false;
                const f = foot.getBoundingClientRect(), g = field.getBoundingClientRect();
                const inter = !(f.right <= g.left || f.left >= g.right || f.bottom <= g.top || f.top >= g.bottom);
                return inter;
            }"""
        )
        report.step("account footer is in the shell and does not overlap fields", no_fixed and not overlap, f"fixed={not no_fixed} overlap={overlap}")
        shot("05b-editor-narrow.png")
    except Exception as exc:
        report.step("account footer is in the shell and does not overlap fields", False, str(exc)[:180])
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(150)
    shot("06-saved-on.png")

    # ---- Settings: the set for other apps ----------------------------------
    page.click('.navi[data-nav="settings"]')
    try:
        page.wait_for_selector("select[aria-label='Set for other apps']", timeout=10000)
        page.select_option("select[aria-label='Set for other apps']", value="smoke-profile")
        page.wait_for_function("() => document.body.innerText.includes('Set for other apps saved.')", timeout=10000)
        report.step("set for other apps saved from Settings", api_get(base_url, "/api/settings").get("default_set") == "smoke-profile")
    except Exception as exc:
        report.step("set for other apps saved from Settings", False, str(exc)[:180])

    # the set is listed under On, with its language and the other-apps badge
    page.click('.navi[data-nav="profiles"]')
    page.wait_for_timeout(400)
    try:
        grouped = page.evaluate(
            """() => {
                const find = (name) => {
                    const lab = [...document.querySelectorAll('.seclabel')]
                        .find(e => e.textContent.includes(name));
                    if (!lab) return null;
                    const out = [];
                    for (let n = lab.nextElementSibling; n && !n.classList.contains('seclabel'); n = n.nextElementSibling) {
                        if (n.classList.contains('pcard')) out.push(n.dataset.profile);
                    }
                    return out;
                };
                const card = document.querySelector('.pcard[data-profile="smoke-profile"]');
                return {on: find('On — somora can call these'), off: find('Off'),
                        text: card ? card.innerText : ''};
            }"""
        )
        on = grouped.get("on") or []
        off = grouped.get("off") or []
        text = grouped.get("text") or ""
        report.step(
            "saved set is listed under On with its voice, language and the other-apps badge",
            "smoke-profile" in on and "smoke-profile" not in off and "ryan" in text and "German" in text and "For other apps" in text,
            f"on={on} off={off} card={text[:160]!r}",
        )
        shot("06b-profiles-grouped.png")
    except Exception as exc:
        report.step("saved set is listed under On", False, str(exc)[:180])

    # ---- test console route display ---------------------------------------
    page.click('.navi[data-nav="test"]')
    page.wait_for_timeout(400)
    report.step("test console opened", (page.text_content("#shell-inner h1") or "") == "Test")
    page.select_option('select[aria-label="Set"]', value="smoke-profile")
    page.click(".talkbtn")
    # The route panel is populated ONLY from session.updated
    # (session.x_realtimevoice.route).
    try:
        page.wait_for_function(
            "() => { const s=document.querySelector('.tsnow'); return s && !s.textContent.includes('Idle'); }",
            timeout=20000,
        )
        report.step("talk connects and leaves idle", True)
    except PWTimeout:
        report.step("talk connects and leaves idle", False, (page.text_content(".talkstate") or "")[:160])

    try:
        page.wait_for_function(
            "() => { const e=document.querySelector('.routeid'); return e && e.textContent === 'smoke-profile'; }",
            timeout=20000,
        )
        report.step("Running now route rendered from session.updated", True, "smoke-profile")
    except PWTimeout:
        report.step("Running now route rendered from session.updated", False, (page.text_content(".routeid") or page.text_content(".routepend") or "")[:160])

    # the three model ids from the route must be visible, proving the panel
    # comes from the resolved route and not a degraded state
    route_models = page.eval_on_selector_all(".routechain .rcmodel", "els=>els.map(e=>e.textContent)")
    expected_models = ["stt-base", "llm-base", "preset-model"]
    report.step(
        "Running now shows the three route model ids",
        all(m in route_models for m in expected_models),
        ", ".join(route_models),
    )
    shot("07-test-console.png")
    page.click(".talkbtn")
    page.wait_for_timeout(300)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))