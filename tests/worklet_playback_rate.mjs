// PlayWorklet rate conversion: the ring holds 24 kHz wire audio, the output
// runs at the context's NATIVE rate. Regression cover for the "Mickey Mouse"
// bug (2026-09-16), where one ring sample was emitted per output frame and a
// 24 kHz answer played back 2.00x too fast at 48 kHz / 1.84x at 44.1 kHz.
//
// Checked here, per native rate:
//   - a full drain takes native/24000 as many output frames as ring samples
//     (i.e. duration is preserved),
//   - a fed sine keeps its FREQUENCY after conversion,
//   - accounting (playedSamples / ledger heard) stays in 24 kHz samples, which
//     is what the server truncates conversation history by,
//   - clear() resets the sub-sample offset so it cannot skew the next turn.
//
// The browser-level equivalent (real AudioWorklet, real OfflineAudioContext)
// is scripts/playback_rate_probe.py.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "src", "realtimevoice", "web", "worklet.js"), "utf8");

const WIRE = 24000;

function loadPlayWorklet(nativeRate) {
  const registry = new Map();
  const AudioWorkletProcessor = class {
    constructor() {
      this.port = {
        onmessage: null,
        messages: [],
        postMessage(data) { this.messages.push(data); },
      };
    }
  };
  const registerProcessor = (name, cls) => registry.set(name, cls);
  new Function("sampleRate", "AudioWorkletProcessor", "registerProcessor", source)(
    nativeRate, AudioWorkletProcessor, registerProcessor,
  );
  const Play = registry.get("play-worklet");
  if (!Play) throw new Error("play-worklet not registered");
  return Play;
}

function fail(message) {
  console.error("FAIL:", message);
  process.exit(1);
}

function feed(node, pcm, extra = {}) {
  node.port.onmessage({ data: { cmd: "feed", samples: pcm.buffer, ...extra } });
}

function drainAll(node, frames = 128, maxBlocks = 100000) {
  const out = [];
  let blocks = 0;
  while (node.filled > 0 && blocks < maxBlocks) {
    const ch = new Float32Array(frames);
    node.process([], [[ch]]);
    out.push(...ch);
    blocks += 1;
  }
  return out;
}

// ---------------------------------------------------------------- duration
function durationTest(nativeRate) {
  const Play = loadPlayWorklet(nativeRate);
  const node = new Play();
  const seconds = 1.5;
  const ringSamples = Math.round(WIRE * seconds);
  const pcm = new Int16Array(ringSamples);
  for (let i = 0; i < ringSamples; i++) pcm[i] = 8000; // constant, non-zero

  feed(node, pcm);
  const rendered = drainAll(node);
  // trailing zeros of the last partially filled block are not playback
  let last = rendered.length - 1;
  while (last >= 0 && rendered[last] === 0) last--;
  const outFrames = last + 1;
  const expected = ringSamples * (nativeRate / WIRE);
  const err = Math.abs(outFrames - expected) / expected;
  if (err > 0.01) {
    fail(`duration @${nativeRate}: rendered ${outFrames} frames, expected ~${expected} (${(err * 100).toFixed(2)}% off)`);
  }
  if (node.filled !== 0) fail(`duration @${nativeRate}: ring not fully drained (${node.filled} left)`);
}

// --------------------------------------------------------------- frequency
function frequencyTest(nativeRate) {
  const Play = loadPlayWorklet(nativeRate);
  const node = new Play();
  const tone = 440;
  const seconds = 1.0;
  const ringSamples = Math.round(WIRE * seconds);
  const pcm = new Int16Array(ringSamples);
  for (let i = 0; i < ringSamples; i++) {
    pcm[i] = Math.round(Math.sin(2 * Math.PI * tone * (i / WIRE)) * 0x6000);
  }
  feed(node, pcm);
  const rendered = drainAll(node);
  let last = rendered.length - 1;
  while (last >= 0 && rendered[last] === 0) last--;
  // ignore the edges, count zero crossings
  const a = Math.floor(last * 0.05);
  const b = Math.floor(last * 0.95);
  let crossings = 0;
  for (let i = a + 1; i <= b; i++) {
    if ((rendered[i - 1] < 0 && rendered[i] >= 0) || (rendered[i - 1] >= 0 && rendered[i] < 0)) crossings++;
  }
  const measuredSeconds = (b - a) / nativeRate;
  const freq = crossings / 2 / measuredSeconds;
  const err = Math.abs(freq - tone) / tone;
  if (err > 0.03) fail(`frequency @${nativeRate}: measured ${freq.toFixed(1)} Hz, expected ${tone} Hz`);
}

// -------------------------------------------------------------- accounting
// The server truncates history by HEARD 24 kHz samples. Counting output frames
// instead would over-report heard audio by native/24000 and truncate wrongly.
function accountingTest(nativeRate) {
  const Play = loadPlayWorklet(nativeRate);
  const node = new Play();
  const ringSamples = 24000; // 1 s of wire audio
  const pcm = new Int16Array(ringSamples).fill(4000);
  feed(node, pcm, { itemId: "item-1", gen: 3 });
  drainAll(node);
  if (node.playedSamples !== ringSamples) {
    fail(`accounting @${nativeRate}: playedSamples=${node.playedSamples}, expected ${ringSamples} (wire samples, not output frames)`);
  }
  const heard = node.heardMap.get("3:item-1");
  if (heard !== ringSamples) {
    fail(`accounting @${nativeRate}: heard=${heard}, expected ${ringSamples}`);
  }
}

// -------------------------------------------------- partial heard on clear
function partialHeardTest(nativeRate) {
  const Play = loadPlayWorklet(nativeRate);
  const node = new Play();
  const ringSamples = 24000;
  const pcm = new Int16Array(ringSamples).fill(4000);
  feed(node, pcm, { itemId: "item-x", gen: 1 });
  // play roughly a quarter of a second of NATIVE output
  const frames = Math.round(nativeRate * 0.25);
  const ch = new Float32Array(frames);
  node.process([], [[ch]]);
  const heardWire = node.playedSamples;
  const expected = WIRE * 0.25;
  if (Math.abs(heardWire - expected) / expected > 0.02) {
    fail(`partial heard @${nativeRate}: ${heardWire} wire samples after 0.25 s output, expected ~${expected}`);
  }
  node.port.onmessage({ data: { cmd: "clear", reason: "barge_in" } });
  const cleared = node.port.messages.find((m) => m && m.cmd === "cleared");
  if (!cleared) fail(`partial heard @${nativeRate}: no cleared message`);
  const item = (cleared.items || []).find((i) => i.id === "item-x");
  if (!item) fail(`partial heard @${nativeRate}: item not reported in clear`);
  if (Math.abs(item.heardSamples - expected) / expected > 0.02) {
    fail(`partial heard @${nativeRate}: reported heardSamples=${item.heardSamples}, expected ~${expected}`);
  }
  if (item.heardSamples + item.discardedSamples !== ringSamples) {
    fail(`partial heard @${nativeRate}: heard+discarded=${item.heardSamples + item.discardedSamples}, expected ${ringSamples}`);
  }
  if (node.frac !== 0) fail(`partial heard @${nativeRate}: clear() left frac=${node.frac}, must reset`);
}

for (const rate of [48000, 44100, 24000, 16000]) {
  durationTest(rate);
  frequencyTest(rate);
  accountingTest(rate);
  partialHeardTest(rate);
}

console.log("worklet playback rate: OK (48k/44.1k/24k/16k — duration, pitch, wire-sample accounting, clear)");
