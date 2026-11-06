#!/usr/bin/env python3
"""Controlled TTS-only comparison runner (offline-safe; live run by an operator).

Compares TTS profiles DIRECTLY (no WS session, no settings/preset mutation)
against a running instance, using the server-side /api/tts-synthesis route so
credentials and voice-reference material (extra.ref_audio / extra.ref_text)
never leave the server.

Fixed German texts (ids: short, medium, numbers_names) are synthesized
`--runs` times per profile (default 3 warm runs). Cold runs are NOT claimed:
the runner cannot prove model/state warmness, so `cold` stays null unless a
future server-side signal proves it.

Per run it measures:
- request-to-first-bytes (ms): POST start -> first body byte received,
- total (ms) and output size; the raw WAV artifact is saved unchanged
  (no re-encoding) for human listening.

Profile arguments may pin a voice per variant: `id` or `id:voice`
(e.g. `tts-qwen-ryan`, `tts-qwen-serena`, `tts-chatterbox:Jonas`).
Requested voices are validated server-side against the profile's claimed
voices. The actual model/voice that served each run is proven via response
headers (X-Rtv-Model / X-Rtv-Voice) and validated against the catalog.

Usage:
  python3 scripts/tts_comparison.py --url https://host:8443 \
      --data-dir /tmp/realtimevoice-data \
      --tts-profile tts-baseline --tts-profile tts-chatterbox \
      [--runs 3] [--texts short,medium,numbers_names] \
      [--out-dir /tmp/rtv-tts-compare-<ts>] [--json out.json]

Exit codes: 0 = all runs proven, 3 = at least one failure/mismatch,
2 = setup/transport error. The instance token is read from the private
data dir and never printed; all error text is redacted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import quote

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_profiles import load_token, normalize_base_url  # noqa: E402

TEXTS = {
    "short": "Ja, gerne. Ich erledige das sofort.",
    "medium": (
        "Guten Tag! Ich bin die Testseite Ihres Sprachassistenten und lese "
        "jetzt einen längeren Satz vor, damit wir Betonung, Pausen und "
        "Satzmelodie miteinander vergleichen können."
    ),
    "numbers_names": (
        "Am 3. Oktober treffen sich Alex und Sam um 14:30 Uhr in Linz, um über "
        "die 42 Messwerte und den Backend-Server zu sprechen."
    ),
}
DEFAULT_TEXTS = ["short", "medium", "numbers_names"]
DEFAULT_RUNS = 3
MAX_TEXT_CHARS = 1000

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|authorization)\s*[=:]\s*\S+"),
    re.compile(r"(?i)ref_audio[\"']?\s*[:=]\s*\S{8,}"),
)


def redact(text: str, token: str) -> str:
    if token and token in text:
        text = text.replace(token, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def parse_profile_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return spec.strip(), ""
    pid, voice = spec.split(":", 1)
    return pid.strip(), voice.strip()


def synth_once(client: httpx.Client, base: str, token: str, profile_id: str,
               voice: str, text: str, text_id: str, timeout: float) -> dict:
    """One synthesis call with latency accounting. Never raises on HTTP/backend
    errors; returns an outcome dict instead (errors redacted)."""
    body = {"profile_id": profile_id, "text": text, "text_id": text_id}
    if voice:
        body["voice"] = voice
    headers = {"Authorization": f"Bearer {token}"}
    t0 = time.perf_counter()
    try:
        with client.stream(
            "POST",
            base + "/api/tts-synthesis",
            json=body,
            headers=headers,
            timeout=timeout,
        ) as resp:
            first_ms = None
            chunks: list[bytes] = []
            async_bound = 40 * 1024 * 1024
            received = 0
            for chunk in resp.iter_raw():
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                received += len(chunk)
                if received > async_bound:
                    return {"ok": False, "error": "output exceeded size limit"}
                chunks.append(chunk)
            total_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                detail = b"".join(chunks)[:300].decode("utf-8", "replace")
                try:
                    parsed = json.loads(detail)
                    detail = parsed.get("detail") or detail
                except ValueError:
                    pass
                return {
                    "ok": False,
                    "status": resp.status_code,
                    "error": redact(str(detail), token),
                }
            return {
                "ok": True,
                "first_bytes_ms": round(first_ms, 1) if first_ms is not None else None,
                "total_ms": round(total_ms, 1),
                "bytes": received,
                "headers": {
                    "X-Rtv-Model": resp.headers.get("X-Rtv-Model", ""),
                    "X-Rtv-Voice": resp.headers.get("X-Rtv-Voice", ""),
                    "X-Rtv-First-Pcm-Ms": resp.headers.get("X-Rtv-First-Pcm-Ms", ""),
                    "X-Rtv-Synth-Ms": resp.headers.get("X-Rtv-Synth-Ms", ""),
                },
                "wav": b"".join(chunks),
            }
    except httpx.HTTPError as exc:
        return {"ok": False, "error": redact(str(exc), token)}


def main_async(args, token: str) -> int:
    base = normalize_base_url(args.url)
    exit_code = 0
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with httpx.Client() as client:
        catalog = client.get(
            base + "/api/catalog", headers={"Authorization": f"Bearer {token}"}, timeout=30
        )
        if catalog.status_code != 200:
            print(f"SETUP ERROR: catalog HTTP {catalog.status_code}", file=sys.stderr)
            return 2
        tts_profiles = {
            p["id"]: p
            for p in (catalog.json().get("roles", {}).get("tts", {}) or {}).get("profiles", [])
        }
        if not tts_profiles:
            print("SETUP ERROR: no tts profiles in catalog", file=sys.stderr)
            return 2

        text_ids = [t.strip() for t in args.texts.split(",") if t.strip()]
        for tid in text_ids:
            if tid not in TEXTS:
                print(f"SETUP ERROR: unknown text id '{tid}' (known: {', '.join(TEXTS)})", file=sys.stderr)
                return 2

        report = {
            "url": base,
            "cold": None,
            "cold_note": "cold runs are not claimed by this runner: warmness is not provable client-side",
            "metric_note": (
                "backend_first_pcm_ms is measured at the server PCM decoder boundary "
                "(first nonempty adapter PCM after WAV decode/resample) — NOT acoustic "
                "playback; browser buffering and VAD are excluded. first_bytes_ms is HTTP "
                "TTFB; total_ms is the fully transferred WAV body."
            ),
            "runs_per_text": args.runs,
            "texts": {tid: {"id": tid, "chars": len(TEXTS[tid])} for tid in text_ids},
            "variants": [],
        }
        for spec in args.tts_profile:
            profile_id, voice_override = parse_profile_spec(spec)
            profile = tts_profiles.get(profile_id)
            if profile is None:
                print(f"SETUP ERROR: unknown tts profile '{profile_id}'", file=sys.stderr)
                return 2
            expected_model = profile.get("model", "")
            print(f"[variant] profile={profile_id} voice={voice_override or profile.get('voice') or '(profile default)'} model={expected_model}")
            variant = {"profile_id": profile_id, "voice_request": voice_override, "runs": []}
            for tid in text_ids:
                text = TEXTS[tid]
                for run_index in range(1, args.runs + 1):
                    outcome = synth_once(
                        client, base, token, profile_id, voice_override, text, tid, args.timeout
                    )
                    safe_profile = re.sub(r"[^a-z0-9._-]", "_", profile_id.lower())
                    wav_name = f"{safe_profile}--{tid}--run{run_index}.wav"
                    wav_path = os.path.join(out_dir, wav_name)
                    entry = {
                        "profile_id": profile_id,
                        "text_id": tid,
                        "run": run_index,
                        "cold": None,
                    }
                    if not outcome.get("ok"):
                        entry["ok"] = False
                        entry["error"] = outcome.get("error", "unknown error")
                        print(f"  FAIL {tid} run{run_index}: {entry['error']}")
                        exit_code = 3
                        variant["runs"].append(entry)
                        continue
                    headers = outcome["headers"]
                    model = headers["X-Rtv-Model"]
                    voice_used = headers["X-Rtv-Voice"]
                    proven = model == expected_model
                    if not proven:
                        entry["ok"] = False
                        entry["error"] = f"model proof mismatch: served '{model}' expected '{expected_model}'"
                        exit_code = 3
                        print(f"  FAIL {tid} run{run_index}: {entry['error']}")
                        variant["runs"].append(entry)
                        continue
                    with open(wav_path, "wb") as fh:
                        fh.write(outcome["wav"])
                    entry.update(
                        ok=True,
                        model=model,
                        voice=voice_used,
                        # three DISTINCT metrics:
                        # - first_bytes_ms: HTTP TTFB (first body byte)
                        # - backend_first_pcm_ms: server PCM decoder boundary
                        #   (first nonempty adapter PCM) — NOT acoustic
                        #   playback; no browser buffering/VAD included
                        # - total_ms: full WAV body transferred
                        first_bytes_ms=outcome["first_bytes_ms"],
                        backend_first_pcm_ms=(
                            int(headers["X-Rtv-First-Pcm-Ms"])
                            if headers["X-Rtv-First-Pcm-Ms"].isdigit()
                            else None
                        ),
                        total_ms=outcome["total_ms"],
                        synth_ms=int(headers["X-Rtv-Synth-Ms"] or 0),
                        bytes=outcome["bytes"],
                        wav=wav_path,
                    )
                    print(
                        f"  PASS {tid} run{run_index}: model={model} voice={voice_used} "
                        f"backend_first_pcm={entry['backend_first_pcm_ms']} ms "
                        f"http_first_bytes={entry['first_bytes_ms']} ms total={entry['total_ms']} ms "
                        f"bytes={entry['bytes']} -> {wav_name}"
                    )
                    variant["runs"].append(entry)
            report["variants"].append(variant)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"report written: {args.json}")

    print("--- tts comparison summary (no default/settings mutation) ---")
    for variant in report["variants"]:
        runs = [r for r in variant["runs"] if r.get("ok")]
        for tid in text_ids:
            run_rows = [r for r in runs if r["text_id"] == tid]
            if run_rows:
                fb = sorted(r["backend_first_pcm_ms"] for r in run_rows if r.get("backend_first_pcm_ms") is not None)
                tt = sorted(r["total_ms"] for r in run_rows)
                fb_txt = f"{fb[len(fb)//2]} ms" if fb else "n/a (single PCM chunk)"
                print(
                    f"  {variant['profile_id']} {tid}: n={len(run_rows)} "
                    f"backend_first_pcm median={fb_txt} total median={tt[len(tt)//2]} ms"
                )
            else:
                print(f"  {variant['profile_id']} {tid}: no successful runs")
    return exit_code


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="tts_comparison.py",
        description=(
            "Controlled TTS-only comparison via server-side /api/tts-synthesis. "
            "Saves raw WAV artifacts, proves served model/voice via headers, "
            "never mutates settings/presets. Token is read from --data-dir and "
            "never printed or sent in URLs."
        ),
    )
    parser.add_argument("--url", required=True, help="Instance base URL, e.g. https://host:8443 (trailing slash normalized)")
    parser.add_argument("--data-dir", required=True, help="Instance data dir containing server.json (auth_token)")
    parser.add_argument("--tts-profile", action="append", required=True, metavar="ID[:VOICE]", help="TTS profile id to compare; optional :voice override (repeatable)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Warm runs per text (default: 3)")
    parser.add_argument("--texts", default=",".join(DEFAULT_TEXTS), help=f"Comma list of text ids (default: {','.join(DEFAULT_TEXTS)})")
    parser.add_argument("--out-dir", default=f"/tmp/rtv-tts-compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}", help="WAV artifact directory")
    parser.add_argument("--json", default=None, help="Write the machine-readable report here")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    args = parser.parse_args(argv)
    if not args.url.startswith(("http://", "https://")):
        raise SystemExit("--url must be an http(s) URL")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])
    token = load_token(args.data_dir)
    if not token:
        print("SETUP ERROR: no auth token found in --data-dir/server.json", file=sys.stderr)
        return 2
    return main_async(args, token)


if __name__ == "__main__":
    sys.exit(main())
