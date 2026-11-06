// Node unit test for the shared microphone/resampler helpers (F2/F4).
//
// The shipped UI modules are plain ESM; this harness strips their import/
// export syntax, concatenates them (mic-devices first, since audio.js imports
// it), evaluates the result and exercises the pure helpers with Node. This
// keeps the test against the REAL shipped code without a bundler.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = join(here, "..", "src", "realtimevoice", "web", "ui");

function loadPlain(file) {
  let src = readFileSync(file, "utf8");
  // drop import declarations (parser-free: they are single-line here)
  src = src.replace(/^import\s+.*?;\s*$/gm, "");
  // strip export keywords
  src = src.replace(/\bexport\s+(async\s+function|function|class|const|let|var)\b/g, "$1");
  return src;
}

const micSrc = loadPlain(join(uiDir, "mic-devices.js"));
const audioSrc = loadPlain(join(uiDir, "audio.js"));

// Minimal browser globals for module-level references.
try {
  if (!globalThis.navigator) globalThis.navigator = { mediaDevices: undefined };
} catch {
  /* Node 22 exposes navigator as a getter-only global */
}
globalThis.localStorage = {
  _m: new Map(),
  getItem(k) { return this._m.has(k) ? this._m.get(k) : null; },
  setItem(k, v) { this._m.set(k, String(v)); },
  removeItem(k) { this._m.delete(k); },
};

const factory = new Function(
  `${micSrc}\n${audioSrc}\nreturn { makeResampler, micConstraints, resolveRememberedDevice, rememberedDeviceId, rememberDeviceId };`,
);
const { makeResampler, micConstraints, resolveRememberedDevice, rememberedDeviceId, rememberDeviceId } = factory();

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`PASS  ${name}`);
  else {
    failures++;
    console.log(`FAIL  ${name}${detail ? "  (" + detail + ")" : ""}`);
  }
}

// ---- micConstraints -------------------------------------------------------
const c = micConstraints("dev-1", { echoCancellation: false });
check("micConstraints uses exact deviceId", c.audio.deviceId && c.audio.deviceId.exact === "dev-1", JSON.stringify(c));
check("micConstraints preserves extra flags", c.audio.echoCancellation === false);
check("micConstraints system default has no deviceId", !micConstraints("").audio.deviceId);

// ---- resolveRememberedDevice ---------------------------------------------
const devices = [
  { kind: "audioinput", deviceId: "a", label: "Mic A" },
  { kind: "videoinput", deviceId: "v", label: "Cam" },
  { kind: "audioinput", deviceId: "b", label: "Mic B" },
];
check("remembered device present resolves", resolveRememberedDevice("b", devices).deviceId === "b");
check("remembered device present not flagged missing", resolveRememberedDevice("b", devices).missing === false);
const gone = resolveRememberedDevice("zzz", devices);
check("remembered device gone -> system default + missing", gone.deviceId === "" && gone.missing === true);
check("no remembered -> system default, not missing", resolveRememberedDevice("", devices).deviceId === "" && resolveRememberedDevice("", devices).missing === false);

// ---- localStorage round trip ---------------------------------------------
rememberDeviceId("mic-9");
check("rememberDeviceId round trips", rememberedDeviceId() === "mic-9");
rememberDeviceId("");
check("rememberDeviceId empty clears", rememberedDeviceId() === "");

// ---- resampler ------------------------------------------------------------
const ident = makeResampler(24000, 24000);
const src = Int16Array.from([1, 2, 3, 4, 5]);
check("resampler identity when rates match", ident(src) === src);

const r = makeResampler(48000, 24000);
const sine = new Int16Array(4800);
for (let i = 0; i < sine.length; i++) sine[i] = Math.round(20000 * Math.sin((2 * Math.PI * 440 * i) / 48000));
let totalOut = 0;
let maxAbs = 0;
for (let off = 0; off < sine.length; off += 480) {
  const out = r(sine.subarray(off, off + 480));
  totalOut += out.length;
  for (const v of out) maxAbs = Math.max(maxAbs, Math.abs(v));
}
check("resampler 48k->24k approximately halves samples", Math.abs(totalOut - sine.length / 2) <= 2, `out=${totalOut} in=${sine.length}`);
check("resampler keeps amplitude", maxAbs > 15000 && maxAbs <= 20001, `maxAbs=${maxAbs}`);

const r2 = makeResampler(44100, 24000);
let out2 = 0;
const chunk = new Int16Array(441);
for (let i = 0; i < 100; i++) {
  for (let j = 0; j < chunk.length; j++) chunk[j] = j % 2 ? 1000 : -1000;
  out2 += r2(chunk).length;
}
check("resampler 44.1k->24k output length sane", out2 > 20000 && out2 < 26000, `out=${out2}`);
check("resampler handles empty input", makeResampler(48000, 24000)(new Int16Array(0)).length === 0);

console.log(failures === 0 ? "ALL RESAMPLER/MIC TESTS PASSED" : `${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);