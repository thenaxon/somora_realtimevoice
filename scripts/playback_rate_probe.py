#!/usr/bin/env python3
"""Measure the REAL browser playback path for pitch/duration fidelity.

The bug this exists for: the wire carries PCM16 at 24 kHz, but the page runs a
native AudioContext (44.1/48 kHz). If the play-worklet emits one ring sample per
output sample, playback runs at nativeRate/24000 — 2.00x at 48 kHz, 1.84x at
44.1 kHz — i.e. too fast and an octave too high ("Mickey Mouse").

Method: feed a known tone (default 440 Hz, 2.0 s, PCM16 @ 24 kHz) into the real
`play-worklet` inside an OfflineAudioContext at a chosen native rate, render it,
then measure what came out:
  - duration  = span from first to last non-silent sample
  - frequency = zero crossings over that span
A correct implementation reproduces 440 Hz / 2.0 s at ANY native rate.

Run:
  python scripts/playback_rate_probe.py [--rates 48000,44100] [--tone 440]
         [--seconds 2.0] [--worklet <path>] [--out-dir <dir>]
Exit 0 = pitch and duration preserved within tolerance at every rate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_WORKLET = REPO / "src" / "realtimevoice" / "web" / "worklet.js"

# The page builds the tone, runs the worklet offline, and measures the result.
PAGE = """
<!doctype html><meta charset="utf-8"><title>playback rate probe</title>
<script type="module">
window.__probe = async ({workletSource, nativeRate, wireRate, tone, seconds, pcmBytes}) => {
  // load the REAL worklet source through a blob URL: no server, no routing,
  // and byte-identical to what the app ships
  const workletUrl = URL.createObjectURL(new Blob([workletSource], {type: "text/javascript"}));
  let pcm;
  if (pcmBytes) {
    // real TTS audio (PCM16 @ wireRate), base64 -> Int16Array
    const bin = atob(pcmBytes);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    pcm = new Int16Array(bytes.buffer);
  } else {
    // known tone as PCM16 @ wireRate (exactly what the server sends)
    const frames = Math.round(wireRate * seconds);
    pcm = new Int16Array(frames);
    for (let i = 0; i < frames; i++) {
      pcm[i] = Math.round(Math.sin(2 * Math.PI * tone * (i / wireRate)) * 0x6000);
    }
  }
  const frames = pcm.length;
  // generous tail so nothing is cut off even if playback is SLOWER than realtime
  const renderFrames = Math.round(nativeRate * (seconds * 3 + 1));
  const ctx = new OfflineAudioContext(1, renderFrames, nativeRate);
  await ctx.audioWorklet.addModule(workletUrl);
  const node = new AudioWorkletNode(ctx, "play-worklet", {outputChannelCount: [1]});
  node.connect(ctx.destination);
  node.port.postMessage({cmd: "feed", samples: pcm.buffer}, [pcm.buffer]);
  // the port message must reach the worklet BEFORE offline rendering runs,
  // otherwise the ring is still empty and we render pure silence
  await new Promise((r) => setTimeout(r, 150));
  const rendered = await ctx.startRendering();
  const ch = rendered.getChannelData(0);

  // non-silent span
  const EPS = 1e-4;
  let first = -1, last = -1;
  for (let i = 0; i < ch.length; i++) {
    if (Math.abs(ch[i]) > EPS) { if (first < 0) first = i; last = i; }
  }
  if (first < 0) return {error: "silence rendered"};
  const span = last - first + 1;
  const durationS = span / nativeRate;

  // frequency via zero crossings inside the span (ignore the very edges)
  const a = first + Math.floor(span * 0.05), b = last - Math.floor(span * 0.05);
  let crossings = 0;
  for (let i = a + 1; i <= b; i++) {
    if ((ch[i - 1] < 0 && ch[i] >= 0) || (ch[i - 1] >= 0 && ch[i] < 0)) crossings++;
  }
  const measuredS = (b - a) / nativeRate;
  const freq = crossings / 2 / measuredS;
  return {
    nativeRate, wireRate, durationS, freq,
    fedSamples: frames,
    renderedNonSilent: span,
  };
};
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="48000,44100")
    ap.add_argument("--tone", type=float, default=440.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--wire-rate", type=int, default=24000)
    ap.add_argument("--worklet", default=str(DEFAULT_WORKLET))
    ap.add_argument("--wav", default="",
                    help="real PCM16 mono WAV at --wire-rate; checks DURATION only "
                         "(pitch measurement is meaningless for speech)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tol-freq", type=float, default=0.03, help="relative tolerance")
    ap.add_argument("--tol-dur", type=float, default=0.05)
    args = ap.parse_args()

    worklet = pathlib.Path(args.worklet).resolve()
    if not worklet.is_file():
        print(f"worklet not found: {worklet}", file=sys.stderr)
        return 2
    rates = [int(r) for r in args.rates.split(",") if r.strip()]

    source = worklet.read_text()

    # AudioWorklet needs a secure context — about:blank (opaque origin) has no
    # ctx.audioWorklet. Serve page + worklet from 127.0.0.1, which counts as one.
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/worklet.js"):
                body, ctype = source.encode(), "text/javascript"
            else:
                body, ctype = PAGE.encode(), "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # silence
            return

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    results, failures = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/")

        pcm_b64 = ""
        expected_seconds = args.seconds
        if args.wav:
            import base64
            import wave
            with wave.open(args.wav) as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    print(f"--wav must be PCM16 mono, got ch={w.getnchannels()} width={w.getsampwidth()}",
                          file=sys.stderr)
                    return 2
                if w.getframerate() != args.wire_rate:
                    print(f"--wav rate {w.getframerate()} != wire rate {args.wire_rate}", file=sys.stderr)
                    return 2
                raw = w.readframes(w.getnframes())
                expected_seconds = w.getnframes() / w.getframerate()
            pcm_b64 = base64.b64encode(raw).decode()
            print(f"real audio: {args.wav} — {expected_seconds:.3f} s @ {args.wire_rate} Hz")

        for rate in rates:
            res = page.evaluate(
                "args => window.__probe(args)",
                {"workletSource": source, "nativeRate": rate,
                 "wireRate": args.wire_rate, "tone": args.tone,
                 "seconds": expected_seconds, "pcmBytes": pcm_b64},
            )
            if res.get("error"):
                failures.append(f"{rate} Hz: {res['error']}")
                results.append({"nativeRate": rate, "error": res["error"]})
                continue
            f_err = abs(res["freq"] - args.tone) / args.tone
            d_err = abs(res["durationS"] - expected_seconds) / expected_seconds
            speed = expected_seconds / res["durationS"] if res["durationS"] else float("inf")
            ok = d_err <= args.tol_dur and (bool(args.wav) or f_err <= args.tol_freq)
            res.update({"expectedFreq": None if args.wav else args.tone,
                        "expectedDurationS": expected_seconds,
                        "freqErrRel": f_err, "durErrRel": d_err,
                        "playbackSpeedFactor": speed, "ok": ok})
            results.append(res)
            status = "PASS" if ok else "FAIL"
            print(f"{status}  native={rate} Hz  "
                  + (f"dur={res['durationS']:.3f} s (want {expected_seconds:.3f})  "
                     if args.wav else
                     f"freq={res['freq']:.1f} Hz (want {args.tone:.0f})  "
                     f"dur={res['durationS']:.3f} s (want {expected_seconds:.2f})  ")
                  + 
                  f"speed={speed:.3f}x")
            if not ok:
                failures.append(
                    f"{rate} Hz: freq {res['freq']:.1f} vs {args.tone:.0f}, "
                    f"dur {res['durationS']:.3f}s vs {expected_seconds:.3f}s, speed {speed:.3f}x")
        browser.close()
    httpd.shutdown()

    if args.out_dir:
        d = pathlib.Path(args.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "playback-rate.json").write_text(json.dumps(
            {"worklet": str(worklet), "tone": args.tone, "seconds": args.seconds,
             "wireRate": args.wire_rate, "results": results,
             "passed": not failures}, indent=1))
        print(f"\nreport: {d / 'playback-rate.json'}")

    if failures:
        print("\nPLAYBACK RATE PROBE FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPLAYBACK RATE PROBE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
