#!/usr/bin/env python3
"""Set-comparison runner for a running realtimevoice instance.

Reads the instance token from --data-dir/server.json (or the
REALTIMEVOICE_AUTH_TOKEN env var) and NEVER prints, logs, or writes it.

Flow (additive-only; the runner never calls PUT /api/settings and never
creates, deletes, or modifies models or sets):
  1. GET /api/catalog, /api/conversation-profiles and /api/settings -> prints
     the models (id + model name) and the sets (id, models, language, on/off,
     default) only; everything else (URLs, keys, capabilities details) is
     not printed.
  2. For a call naming no set (model="") first, then for each --set id
     (`profile:<id>`), and for each --test-set id (a test ticket from
     POST /api/conversation-profiles/<id>/test-session, `profile-test:<ticket>`;
     the set need not be on, nothing is stored): the models, language and
     voice the call must get are derived from the set and the catalog.
  3. Connects /v1/realtime?model=... authenticating with the browser-style
     WebSocket subprotocol pair ("rtv.bearer", "rtv.bearer.<token>") — the
     token NEVER appears in any URL, query string, log line, or artifact
     (query strings show up in access logs and error traces; subprotocols
     do not appear in request-line logs) — and asserts that
     session.created.session.realtimevoice proves the set and its model ids
     (stt/llm/tts) and language. Without a default set the model="" call
     must be refused (`no_set_selected`).
  4. Optionally (--wav, unless --skip-audio) streams one short synthetic
     audio turn per variant exactly like scripts/live_smoke.py and records
     the response status / audio chunk count.

Exit codes: 0 pass, 2 setup/transport error, 3 assertion failures.

Import is offline-safe: no network at import time; the only third-party
dependency is `websockets` (needed for the WS step, same as live_smoke).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

ROLES = ("stt", "llm", "tts")


def load_token(data_dir: str | None) -> str | None:
    token = os.environ.get("REALTIMEVOICE_AUTH_TOKEN", "").strip()
    if token:
        return token
    if data_dir and os.path.exists(os.path.join(data_dir, "server.json")):
        with open(os.path.join(data_dir, "server.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        token = data.get("auth_token")
        if isinstance(token, str) and token:
            return token
    return None


# ------------------------------------------------------------------- HTTP


def http_json(method: str, url: str, token: str, body: dict | None = None) -> dict:
    """Bearer-auth JSON request; raises RuntimeError with the server detail."""
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", "Bearer " + token)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        detail = payload.get("detail") or payload.get("message") or exc.reason
        code = payload.get("code") or ""
        raise RuntimeError(f"HTTP {exc.status} {url}: {detail}" + (f" (code: {code})" if code else "")) from exc


def redact(text: str, token: str) -> str:
    """Replace any occurrence of the instance token with [REDACTED]."""
    if token and token in text:
        return text.replace(token, "[REDACTED]")
    return text


WS_PATH = "/v1/realtime"


def normalize_base_url(url: str) -> str:
    """Base URL without trailing slashes (trailing '/' would double the WS
    path and route the socket into StaticFiles -> HTTP 500)."""
    url = (url or "").strip()
    while url.endswith("/"):
        url = url[:-1]
    return url


def ws_url_for(base_url: str, model: str) -> str:
    """Public WS URL for a model string. Contains NO token: authentication
    travels via the rtv.bearer.<token> WebSocket subprotocol (see
    run_ws_variant), so this URL is safe for logs — and is still never
    printed with credentials, because there are none in it.

    Normalizes the base URL (trailing slashes, optional base path) and
    guarantees the /v1/realtime path appears exactly once."""
    base = normalize_base_url(base_url)
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    path = base.split("?", 1)[0]
    if not path.endswith(WS_PATH):
        base = base + WS_PATH
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}model={urllib.parse.quote(model or '', safe='')}"


# ------------------------------------------------------------------- WAV


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
        frames = [samples[i: i + channels] for i in range(0, len(samples), channels)]
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


# --------------------------------------------------------------- reporting


def catalog_models(catalog: dict) -> dict[str, dict[str, dict]]:
    """{role: {model id: row}} from GET /api/catalog."""
    out: dict[str, dict[str, dict]] = {}
    for role in ROLES:
        rows = ((catalog.get("roles") or {}).get(role) or {}).get("profiles") or []
        out[role] = {str(r.get("id", "")): r for r in rows}
    return out


def print_catalog(catalog: dict, sets: list[dict], server: dict) -> None:
    """Print models and sets as ids + model names only (never URLs/keys)."""
    print("--- models ---")
    for role, rows in catalog_models(catalog).items():
        print(f"[{role}]")
        for pid, row in rows.items():
            print(f"  {pid}: model={row.get('model', '')!r} display={row.get('display_name', '')!r}")
    print("--- sets ---")
    for item in sets:
        spec = item.get("spec") or {}
        refs = " ".join(f"{r}={spec.get(f'{r}_profile') or '-'}" for r in ROLES)
        flags = ["on" if item.get("enabled") else "off"]
        if item.get("id") == server.get("default_set"):
            flags.append("default")
        print(f"  {item.get('id', '')}: {refs} language={spec.get('language', '')!r} ({', '.join(flags)})")
    print(f"default_set: {server.get('default_set', '')!r}")
    print()


def expected_for_set(set_item: dict, models: dict[str, dict[str, dict]], server: dict) -> dict:
    """What a call on this set must get: the set id, each role's model id and
    model name, the language and (for a built-in voice) the voice."""
    spec = set_item.get("spec") or {}
    roles = {}
    for role in ROLES:
        pid = spec.get(f"{role}_profile") or ""
        roles[role] = {"profile": pid, "model": (models.get(role, {}).get(pid) or {}).get("model", "")}
    voice_sel = spec.get("voice") or {}
    return {
        "set": set_item.get("id", ""),
        "roles": roles,
        "language": spec.get("language") or ((server.get("session") or {}).get("language") or ""),
        "voice": voice_sel.get("native_voice", "") if voice_sel.get("kind") == "native" else "",
    }


def print_expected(model: str, expected: dict) -> None:
    roles = expected.get("roles") or {}
    parts = [f"{r}={(roles.get(r) or {}).get('profile') or '-'} → {(roles.get(r) or {}).get('model') or '-'}" for r in ROLES]
    print(
        f"  expected for model={model!r}: set={expected.get('set', '')!r}; {'; '.join(parts)}; "
        f"language={expected.get('language', '')!r} voice={expected.get('voice') or '(set by the voice)'!r}"
    )


def check_realtimevoice(rv: dict, expected: dict) -> tuple[bool, str]:
    """Assert the session.created.realtimevoice block proves the set, its
    model ids and the language."""
    problems: list[str] = []
    if not rv:
        return False, "session.created carries no realtimevoice block"
    if (expected.get("set") or "") != (rv.get("set") or ""):
        problems.append(f"set: expected={expected.get('set')!r} realtimevoice={rv.get('set')!r}")
    roles = expected.get("roles") or {}
    for role in ROLES:
        expect_model = (roles.get(role) or {}).get("model") or ""
        got_model = ((rv.get(role) or {}).get("model")) or ""
        if expect_model != got_model:
            problems.append(f"{role} model: expected={expect_model!r} realtimevoice={got_model!r}")
        expect_profile = (roles.get(role) or {}).get("profile") or ""
        got_profile = ((rv.get(role) or {}).get("profile")) or ""
        if expect_profile != got_profile:
            problems.append(f"{role} profile: expected={expect_profile!r} realtimevoice={got_profile!r}")
    if (expected.get("language") or "") != (rv.get("language") or ""):
        problems.append(f"language: expected={expected.get('language')!r} realtimevoice={rv.get('language')!r}")
    got_voice = (rv.get("tts") or {}).get("voice") or ""
    if expected.get("voice") and got_voice != expected["voice"]:
        problems.append(f"voice: expected={expected.get('voice')!r} realtimevoice={got_voice!r}")
    return (not problems), ("set/models/language proven" if not problems else "; ".join(problems))


# ------------------------------------------------------------- WS variant


async def audio_turn(ws, wav_path: str, timeout: float) -> dict:
    """One short synthetic turn: stream the WAV + trailing silence, then
    wait (bounded) for the first response.done. Same shape as live_smoke."""
    pcm = read_wav_24k_mono(wav_path)
    chunk = 24000 * 2 * 100 // 1000  # 100 ms
    total_ms = len(pcm) / 48
    print(f"  streaming {total_ms:.0f} ms of audio from {wav_path}")
    for i in range(0, len(pcm), chunk):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i: i + chunk]).decode()}))
        await asyncio.sleep(0.1)
    silence = b"\x00\x00" * 2400  # 100 ms of silence
    for _ in range(20):  # 2 s trailing silence to trigger VAD
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
        await asyncio.sleep(0.1)
    deadline = time.monotonic() + timeout
    status = None
    audio_chunks = 0
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        msg = json.loads(raw)
        mtype = msg.get("type", "")
        if mtype == "response.output_audio.delta":
            audio_chunks += 1
        elif mtype == "error":
            print(f"  < error {msg.get('error')}")
        elif mtype == "response.done":
            status = (msg.get("response") or {}).get("status")
            break
    return {"status": status, "audio_chunks": audio_chunks}


async def run_ws_variant(base_url: str, model: str, expected: dict | None, token: str, wav_path: str | None, timeout: float) -> tuple[bool, str, dict | None]:
    """Connect, assert the realtimevoice block, optionally run an audio turn.
    `expected` None: the server must refuse the call (`no_set_selected`).

    Authentication uses the browser-style subprotocol pair
    ("rtv.bearer", "rtv.bearer.<token>"); the URL carries only the model.
    Every exception path is redacted before its text can be printed/written.
    """
    if websockets is None:
        return False, "websockets is not installed (pip install 'uvicorn[standard]')", None
    url = ws_url_for(base_url, model)  # no token in the URL by construction
    audio: dict | None = None
    try:
        async with websockets.connect(
            url,
            max_size=16 * 1024 * 1024,
            subprotocols=["rtv.bearer", "rtv.bearer." + token],
        ) as ws:
            created = None
            deadline = time.monotonic() + timeout
            while created is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, "no session.created within timeout", None
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = json.loads(raw)
                mtype = msg.get("type", "")
                if mtype == "session.created":
                    created = msg
                elif mtype == "error":
                    return False, f"server error: {redact(str(msg.get('error')), token)}", None
            rv = ((created.get("session") or {}).get("realtimevoice")) or {}
            if expected is None:
                return False, "connected although no set is selected (expected no_set_selected)", None
            proven, detail = check_realtimevoice(rv, expected)
            if wav_path and proven:
                audio = await audio_turn(ws, wav_path, timeout)
    except Exception as exc:
        # connect/handshake/transport/body errors: never leak the token
        code = refusal_code(exc)
        text = redact(f"websocket error: {exc}" + (f" (code: {code})" if code else ""), token)
        if expected is None and refused_for_no_set(exc):
            return True, "refused as expected (no_set_selected)", None
        return False, text, audio
    return proven, detail, audio


# ------------------------------------------------------------------ main


def refusal_code(exc: Exception) -> str:
    """The error code of a refused handshake (from the HTTP error body), or
    "" when the server gave none."""
    body = getattr(getattr(exc, "response", None), "body", None)
    if not body:
        return ""
    try:
        return str(((json.loads(bytes(body).decode("utf-8")) or {}).get("error") or {}).get("code") or "")
    except (ValueError, AttributeError):
        return ""


def refused_for_no_set(exc: Exception) -> bool:
    """A handshake refusal naming no_set_selected (HTTP body or close reason)."""
    return refusal_code(exc) == "no_set_selected" or "no set selected" in str(exc)


def build_variants(args) -> list[dict]:
    """A call naming no set (model="") ALWAYS first; then each --set; the
    --test-set tickets are added once created."""
    variants = [{"model": "", "set": "", "source": "default"}]
    for sid in args.set:
        variants.append({"model": f"profile:{sid}", "set": sid, "source": "set"})
    return variants


async def main_async(args, token: str) -> int:
    exit_code = 0
    base = normalize_base_url(args.url)
    variants = build_variants(args)
    summary: list[dict] = []

    try:
        catalog = http_json("GET", base + "/api/catalog", token)
        sets = http_json("GET", base + "/api/conversation-profiles", token).get("profiles") or []
        server = http_json("GET", base + "/api/settings", token)
    except RuntimeError as exc:
        print(f"SETUP ERROR: catalog/sets/settings fetch failed: {exc}", file=sys.stderr)
        return 2
    print_catalog(catalog, sets, server)
    models = catalog_models(catalog)
    sets_by_id = {str(item.get("id", "")): item for item in sets}
    default_set = server.get("default_set") or ""
    variants[0]["set"] = default_set

    for sid in args.test_set:
        try:
            res = http_json("POST", base + f"/api/conversation-profiles/{urllib.parse.quote(sid, safe='')}/test-session", token, {})
        except RuntimeError as exc:
            print(f"SETUP ERROR: test ticket for set {sid!r} failed: {exc}", file=sys.stderr)
            return 2
        ticket_model = res.get("model") or ""
        variants.append({"model": ticket_model, "set": sid, "source": "test_ticket"})
        print(f"test ticket created for set {sid!r} (expires_at={res.get('expires_at')}, never printed again)")
        print()

    wav_path = args.wav
    if args.skip_audio:
        wav_path = None
    if wav_path and not os.path.isfile(wav_path):
        print(f"SETUP ERROR: --wav file not found: {wav_path}", file=sys.stderr)
        return 2
    if not wav_path:
        print("audio turns skipped (no --wav or --skip-audio)")
        print()

    for variant in variants:
        model = variant["model"]
        label = model if variant["source"] != "test_ticket" else f"profile-test:<ticket for {variant['set']}>"
        label = label or '(no set named: model="")'
        print(f"=== variant {label} (source: {variant['source']}) ===")
        entry = {
            "model": model if variant["source"] != "test_ticket" else "profile-test:<ticket>",
            "set": variant["set"],
            "source": variant["source"],
            "stt_model": None,
            "llm_model": None,
            "tts_model": None,
            "voice": None,
            "language": None,
            "ws_proven": False,
        }
        expected: dict | None = None
        if variant["set"]:
            set_item = sets_by_id.get(variant["set"])
            if set_item is None:
                print(f"  FAIL: no set is called {variant['set']!r}")
                summary.append({**entry, "error": "unknown set"})
                exit_code = 3
                continue
            expected = expected_for_set(set_item, models, server)
            print_expected(label, expected)
            if variant["source"] != "test_ticket" and not set_item.get("enabled"):
                print("  note: this set is off; only a test ticket (--test-set) can call it")
            roles = expected["roles"]
            entry.update(
                stt_model=roles["stt"]["model"],
                llm_model=roles["llm"]["model"],
                tts_model=roles["tts"]["model"],
                voice=expected["voice"] or None,
                language=expected["language"],
            )
        else:
            print("  no default set: the call must be refused (no_set_selected)")
        try:
            proven, detail, audio = await run_ws_variant(
                base, model, expected, token, wav_path, args.timeout
            )
        except Exception as exc:  # transport errors are hard failures
            print(f"  FAIL: websocket: {redact(str(exc), token)}")
            summary.append({**entry, "error": redact(str(exc), token)})
            exit_code = 3
            continue
        entry["ws_proven"] = proven
        line = f"  {'PASS' if proven else 'FAIL'}: {detail}"
        if audio is not None:
            entry["audio"] = audio
            line += f"; audio turn: status={audio.get('status')!r} audio_chunks={audio.get('audio_chunks')}"
            if audio.get("status") != "completed":
                line += " (WARNING: response did not complete)"
        print(line)
        if not proven:
            exit_code = 3
        summary.append(entry)
        print()

    if args.json:
        payload = {"variants": summary}
        try:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            print(f"summary written: {args.json}")
        except OSError as exc:
            print(f"WARNING: could not write --json {args.json}: {exc}", file=sys.stderr)

    print("--- comparison summary ---")
    for entry in summary:
        print(
            f"  model={entry.get('model') or '(none)'} set={entry.get('set') or '-'}: "
            f"stt={entry.get('stt_model')} llm={entry.get('llm_model')} tts={entry.get('tts_model')} "
            f"voice={entry.get('voice')} language={entry.get('language')} ws_proven={entry.get('ws_proven')}"
        )
    if exit_code == 0:
        print("PASS")
    else:
        print("FAIL: one or more variants could not be proven")
    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--url",
        required=True,
        help="Instance base URL, e.g. http://127.0.0.1:8765 (trailing slashes and an optional base path are normalized; /v1/realtime is appended exactly once)",
    )
    parser.add_argument("--data-dir", required=True, help="Instance data dir containing server.json (auth_token; never printed)")
    parser.add_argument("--set", action="append", default=[], metavar="ID", help="Set id to compare as profile:<id> (repeatable; the call naming no set always runs first)")
    parser.add_argument("--test-set", action="append", default=[], metavar="ID", help="Set id to compare through a test ticket (profile-test:<ticket>; the set need not be on; repeatable)")
    parser.add_argument("--wav", default=None, help="Synthetic speech WAV for one short audio turn per variant (optional)")
    parser.add_argument("--json", default=None, help="Output path for the JSON summary {variants: [...]}")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio turns even when --wav is given")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-variant WS timeout in seconds (default: 60)")
    args = parser.parse_args(argv)
    if not args.url.startswith(("http://", "https://")):
        raise SystemExit("--url must be an http(s) URL")
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])
    token = load_token(args.data_dir)
    if not token:
        print("SETUP ERROR: no auth token found (use --data-dir or REALTIMEVOICE_AUTH_TOKEN)", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args, token))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(2)
