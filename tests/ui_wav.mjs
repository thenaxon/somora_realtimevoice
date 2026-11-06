/* Node unit tests for the raw-PCM WAV encoder (contract §8a: the browser
 * uploads a mono PCM16 WAV at the capture sample rate; no server-side
 * WebM/Opus decoding). Run: node tests/ui_wav.mjs */

import assert from "node:assert/strict";
import {
  WAV_HEADER_BYTES,
  downmixToMono,
  floatToPcm16,
  encodeWav,
  encodeMonoWavFromFloat,
  inspectWav,
} from "../src/realtimevoice/web/ui/wav.js";

let passed = 0;
function ok(name) {
  passed++;
  console.log("ok - " + name);
}

function readString(view, off, len) {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
  return s;
}

/* ---- header correctness ------------------------------------------------ */
{
  const pcm = new Int16Array([0, 1, -1, 32767, -32768]);
  const buf = encodeWav(pcm, 24000);
  const view = new DataView(buf);
  assert.equal(buf.byteLength, WAV_HEADER_BYTES + pcm.length * 2, "buffer length");
  assert.equal(readString(view, 0, 4), "RIFF", "RIFF tag");
  assert.equal(view.getUint32(4, true), 36 + pcm.length * 2, "ChunkSize");
  assert.equal(readString(view, 8, 4), "WAVE", "WAVE tag");
  assert.equal(readString(view, 12, 4), "fmt ", "fmt tag");
  assert.equal(view.getUint32(16, true), 16, "fmt size");
  assert.equal(view.getUint16(20, true), 1, "PCM format");
  assert.equal(view.getUint16(22, true), 1, "mono");
  assert.equal(view.getUint32(24, true), 24000, "sample rate");
  assert.equal(view.getUint32(28, true), 24000 * 2, "byte rate");
  assert.equal(view.getUint16(32, true), 2, "block align");
  assert.equal(view.getUint16(34, true), 16, "bits per sample");
  assert.equal(readString(view, 36, 4), "data", "data tag");
  assert.equal(view.getUint32(40, true), pcm.length * 2, "data size");
  ok("44-byte canonical RIFF/WAVE header with correct PCM16 mono fields");
}

/* ---- sample rate is preserved ----------------------------------------- */
{
  for (const rate of [16000, 22050, 24000, 44100, 48000]) {
    const buf = encodeWav(new Int16Array(8), rate);
    assert.equal(new DataView(buf).getUint32(24, true), rate, "rate " + rate);
    assert.equal(new DataView(buf).getUint32(28, true), rate * 2, "byte rate " + rate);
  }
  ok("capture sample rate is preserved verbatim (no resample)");
}

/* ---- sample bytes round-trip ------------------------------------------ */
{
  const pcm = new Int16Array([-32768, -12345, -1, 0, 1, 12345, 32767]);
  const buf = encodeWav(pcm, 24000);
  const out = new Int16Array(buf, WAV_HEADER_BYTES, pcm.length);
  assert.deepEqual(Array.from(out), Array.from(pcm), "samples preserved");
  ok("PCM16 samples preserved byte-for-byte after the header");
}

/* ---- mono downmix ------------------------------------------------------ */
{
  const left = new Float32Array([1, -1, 0.5, 0]);
  const right = new Float32Array([-1, 1, -0.5, 0]);
  const mono = downmixToMono([left, right]);
  assert.deepEqual(Array.from(mono).map((v) => Math.round(v * 100)), [0, 0, 0, 0], "averaged");
  const single = downmixToMono([left]);
  assert.deepEqual(Array.from(single), Array.from(left), "single channel unchanged");
  ok("stereo is downmixed to mono by averaging channels");
}

/* ---- float -> int16 conversion and clipping --------------------------- */
{
  const f = new Float32Array([-2, -1, -0.5, 0, 0.5, 1, 2]);
  const pcm = floatToPcm16(f);
  assert.equal(pcm[0], -32768, "-2 clipped to -32768");
  assert.equal(pcm[1], -32768, "-1 maps to -32768");
  assert.equal(pcm[5], 32767, "+1 maps to 32767");
  assert.equal(pcm[6], 32767, "+2 clipped to 32767");
  assert.equal(pcm[3], 0, "0 maps to 0");
  ok("float32 -> int16 uses symmetric bounds and clips out-of-range input");
}

/* ---- full capture encoding -------------------------------------------- */
{
  const rate = 44100;
  const frames = rate; // 1 s
  const ch = new Float32Array(frames);
  for (let i = 0; i < frames; i++) ch[i] = Math.sin((2 * Math.PI * 440 * i) / rate) * 0.5;
  const buf = encodeMonoWavFromFloat([ch], rate);
  const info = inspectWav(buf);
  assert.equal(info.sampleRate, rate, "capture rate");
  assert.equal(info.channels, 1, "mono");
  assert.equal(info.bits, 16, "16-bit");
  assert.equal(info.frames, frames, "frames preserved");
  ok("1 s of float capture encodes to a 1 s mono PCM16 WAV at 44100 Hz");
}

/* ---- inspect rejects non-WAV ------------------------------------------ */
{
  assert.equal(inspectWav(new ArrayBuffer(10)), null, "too short");
  const junk = new ArrayBuffer(44);
  new DataView(junk).setUint8(0, 0x00);
  assert.equal(inspectWav(junk), null, "bad tags");
  ok("inspectWav rejects short/non-RIFF input");
}

console.log(`\nWAV ENCODER TESTS PASSED (${passed})`);