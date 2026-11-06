// Worklet unit test run under Node: verifies sample preservation through the
// exact PlayWorklet ring buffer and CaptureWorklet quantization code paths.
//
//  - 15 s of audio fed faster than realtime must fully drain, in order,
//    without overflow (60 s ring).
//  - Overflow beyond capacity must be explicit (counted, message posted).
//  - clear() must drop pending audio without leaking into the next feed.
//  - CaptureWorklet Float32 -> Int16 quantization must preserve sample values
//    within the rounding envelope, including negative extremes.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "src", "realtimevoice", "web", "worklet.js"), "utf8");

// Minimal browser globals for the worklet module.
const registry = new Map();
globalThis.sampleRate = 24000;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = {
      onmessage: null,
      messages: [],
      postMessage(data) {
        this.messages.push(data);
      },
    };
  }
};
globalThis.registerProcessor = (name, cls) => registry.set(name, cls);

new Function("sampleRate", "AudioWorkletProcessor", "registerProcessor", source)(
  globalThis.sampleRate,
  globalThis.AudioWorkletProcessor,
  globalThis.registerProcessor,
);

const PlayWorklet = registry.get("play-worklet");
const CaptureWorklet = registry.get("capture-worklet");
if (!PlayWorklet || !CaptureWorklet) throw new Error("worklet processors not registered");

function fail(message) {
  console.error("FAIL:", message);
  process.exit(1);
}

// ---------------------------------------------------------------- play ring
function drain(node, frames = 128) {
  const channel = new Float32Array(frames);
  node.process([], [[channel]]);
  return channel;
}

function playRingTest() {
  const node = new PlayWorklet();
  const seconds = 15;
  const total = 24000 * seconds;
  const orig = new Int16Array(total);
  for (let i = 0; i < total; i++) orig[i] = ((i * 1103515245) % 65536) - 32768; // deterministic full-range noise

  // feed in 100 ms chunks, faster than realtime (worst case: all up front)
  const chunk = 2400;
  for (let off = 0; off < total; off += chunk) {
    const slice = orig.subarray(off, Math.min(off + chunk, total));
    node.port.onmessage({ data: { cmd: "feed", samples: slice.slice().buffer } });
  }

  const overflowMessages = node.port.messages.filter((m) => m.cmd === "overflow");
  if (overflowMessages.length > 0) fail(`overflow on normal 15 s audio: ${JSON.stringify(overflowMessages)}`);

  // drain until empty
  const drained = [];
  let guard = total / 128 + 1000;
  while (node.filled > 0 && guard-- > 0) {
    const frame = drain(node);
    for (let i = 0; i < frame.length && node.filled >= 0; i++) {
      if (node.filled === 0 && frame[i] === 0) break;
      drained.push(frame[i]);
    }
  }
  if (node.filled !== 0) fail("ring did not fully drain");
  const expected = total;
  if (drained.length !== expected) fail(`drained ${drained.length} samples, expected ${expected}`);

  // exact sample preservation, in order
  for (let i = 0; i < expected; i++) {
    const value = Math.round(drained[i] * 32768);
    if (value !== orig[i]) fail(`sample ${i}: fed ${orig[i]}, drained ${value}`);
  }

  // clear semantics: pending audio must not leak into the next feed
  node.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([1, 2, 3, 4]).buffer } });
  node.port.onmessage({ data: { cmd: "clear" } });
  node.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([5, 6]).buffer } });
  const filledBefore = node.filled;
  const frame = drain(node, 4);
  const cleared = Array.from(frame.slice(0, filledBefore));
  if (filledBefore !== 2 || cleared[0] !== 5 / 32768 || cleared[1] !== 6 / 32768) {
    fail(`clear leaked old audio: filled=${filledBefore} cleared=${cleared}`);
  }

  // metadata telemetry: clear replies with the buffered amount it discarded,
  // stats replies with fed/played/nonzero counters (sample counts only)
  const t = new PlayWorklet();
  t.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([100, -200, 300, 0]).buffer } });
  t.port.onmessage({ data: { cmd: "clear" } });
  const clearedMsg = t.port.messages.find((m) => m.cmd === "cleared");
  if (!clearedMsg) fail("clear did not reply with a cleared telemetry message");
  if (clearedMsg.bufferedSamples !== 4) fail(`cleared.bufferedSamples=${clearedMsg.bufferedSamples}, expected 4`);
  t.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([100, -200, 300, 0]).buffer } });
  drain(t, 128); // one 128-sample frame consumes all four buffered samples
  const statsMsg = (() => { t.port.onmessage({ data: { cmd: "stats" } }); return t.port.messages.find((m) => m.cmd === "stats"); })();
  if (!statsMsg) fail("stats query did not reply");
  if (statsMsg.fedSamples !== 8 || statsMsg.playedSamples !== 4 || statsMsg.nonZeroPlayed !== 3 || statsMsg.filled !== 0) {
    fail(`stats counters wrong: ${JSON.stringify(statsMsg)}`);
  }

  // explicit overflow accounting on a tiny ring
  const tiny = new PlayWorklet();
  tiny.capacity = 4;
  tiny.ring = new Float32Array(4);
  tiny.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([1, 2, 3, 4, 5, 6]).buffer } });
  const overflow = tiny.port.messages.filter((m) => m.cmd === "overflow");
  if (overflow.length !== 1 || overflow[0].dropped !== 2) fail(`overflow not accounted: ${JSON.stringify(tiny.port.messages)}`);
  const kept = [];
  while (tiny.filled > 0) kept.push(...drain(tiny, 4));
  if (kept.length !== 4) fail(`kept ${kept.length} samples after overflow, expected 4`);

  console.log("play-worklet ring OK: 15 s drained without overflow, samples preserved in order, clear + overflow + telemetry accounted");
}

// ---------------------------------------------------------------- capture
function captureTest() {
  const node = new CaptureWorklet();
  // the worklet flushes only when its 200 ms (4800 samples) buffer is full
  const orig = new Float32Array(4800);
  orig[0] = -1.0; orig[1] = -0.5; orig[2] = -0.001; orig[3] = 0.0; orig[4] = 0.001;
  orig[5] = 0.5; orig[6] = 1.0; orig[7] = 0.25; orig[8] = -0.25;
  node.process([[orig]]);
  const posted = node.port.messages;
  if (posted.length === 0) fail("capture posted nothing after a full 200 ms block");
  const bytes = new Int16Array(posted[0]);
  if (bytes.length !== 4800) fail(`capture posted ${bytes.length} samples, expected 4800`);
  const expected = [-32768, -16384, -32, 0, 32, 16383, 32767, 8191, -8192];
  for (let i = 0; i < 9; i++) {
    if (bytes[i] !== expected[i]) fail(`capture sample ${i}: ${orig[i]} -> ${bytes[i]}, expected ${expected[i]}`);
  }
  // negative extremes must survive (the old char-code bug corrupted these)
  if (bytes[0] !== -32768) fail("negative extreme lost");
  console.log("capture-worklet quantization OK: negatives and extremes preserved across a full 200 ms block");
}

// ------------------------------------------------------- per-item ledger
// The FIFO playhead ledger lets a clear report how much of each item was
// actually HEARD (played) vs discarded, per item id + generation, bounded.
function itemLedgerTest() {
  const n = new PlayWorklet();

  // two items queued: item A partially heard, item B fully buffered
  n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(100).fill(7), itemId: "item_A", gen: 1 } });
  n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(50).fill(9), itemId: "item_B", gen: 1 } });
  for (let i = 0; i < 40; i++) drain(n, 1); // hear 40 of A's 100 samples
  n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
  const ack = n.port.messages.find((m) => m.cmd === "cleared");
  if (!ack || ack.reason !== "speech_started") fail(`clear ack missing reason: ${JSON.stringify(ack)}`);
  const a = (ack.items || []).find((x) => x.id === "item_A");
  const b = (ack.items || []).find((x) => x.id === "item_B");
  if (!a || a.heardSamples !== 40 || a.discardedSamples !== 60) fail(`item A ledger wrong: ${JSON.stringify(a)}`);
  if (!b || b.heardSamples !== 0 || b.discardedSamples !== 50) fail(`item B ledger wrong: ${JSON.stringify(b)}`);
  if (ack.bufferedSamples !== 110) fail(`ack bufferedSamples=${ack.bufferedSamples}, expected 110`);

  // fully played entries are removed and must not appear in the ack
  const n2 = new PlayWorklet();
  n2.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(10).fill(1), itemId: "item_C", gen: 2 } });
  for (let i = 0; i < 10; i++) drain(n2, 1); // fully heard
  n2.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(6).fill(2), itemId: "item_D", gen: 2 } });
  n2.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
  const ack2 = n2.port.messages.filter((m) => m.cmd === "cleared").pop();
  if ((ack2.items || []).some((x) => x.id === "item_C")) fail("fully played item C must not be in clear ack");
  const d = (ack2.items || []).find((x) => x.id === "item_D");
  if (!d || d.discardedSamples !== 6) fail(`item D ledger wrong: ${JSON.stringify(d)}`);

  // FIFO across same id re-fed after full playback: heard is CUMULATIVE
  // across chunks (5 + 3 = 8), the corrected post-fix expectation
  const n3 = new PlayWorklet();
  n3.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(5).fill(1), itemId: "item_E", gen: 3 } });
  for (let i = 0; i < 5; i++) drain(n3, 1);
  n3.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(8).fill(1), itemId: "item_E", gen: 3 } });
  for (let i = 0; i < 3; i++) drain(n3, 1);
  n3.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
  const ack3 = n3.port.messages.filter((m) => m.cmd === "cleared").pop();
  const e = (ack3.items || []).find((x) => x.id === "item_E");
  if (!e || e.heardSamples !== 8 || e.discardedSamples !== 5) fail(`item E re-feed ledger wrong: ${JSON.stringify(e)}`);

  // bounded memory (no-eviction design): every live chunk is RETAINED —
  // 11 buffered items all appear with full attribution, ledgerDropped stays 0
  const n4 = new PlayWorklet();
  for (let k = 0; k < 11; k++) {
    n4.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(4).fill(1), itemId: `item_${k}`, gen: 4 } });
  }
  n4.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
  const ack4 = n4.port.messages.filter((m) => m.cmd === "cleared").pop();
  if (ack4.ledgerDropped !== 0) fail(`ledgerDropped=${ack4.ledgerDropped}, expected 0 (no eviction of pending chunks)`);
  const ack4items = ack4.items || [];
  if (ack4items.length !== 11) fail(`ack items=${ack4items.length}, expected 11 retained live chunks`);
  for (let k = 0; k < 11; k++) {
    const it = ack4items.find((x) => x.id === `item_${k}`);
    if (!it || it.heardSamples !== 0 || it.discardedSamples !== 4) fail(`item_${k} wrong: ${JSON.stringify(it)}`);
  }

  // backward compatible: untagged feeds still work, reported with id null
  const n5 = new PlayWorklet();
  n5.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(4).fill(1) } });
  n5.port.onmessage({ data: { cmd: "clear" } });
  const ack5 = n5.port.messages.filter((m) => m.cmd === "cleared").pop();
  if (ack5.reason !== "" || (ack5.items || []).length !== 1 || ack5.items[0].id !== null) {
    fail(`untagged feed ack wrong: ${JSON.stringify(ack5)}`);
  }

  // stop-reason clear resets everything (fresh talk)
  const n6 = new PlayWorklet();
  n6.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(4).fill(1), itemId: "item_F", gen: 6 } });
  n6.port.onmessage({ data: { cmd: "clear", reason: "stop" } });
  n6.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(4).fill(1), itemId: "item_G", gen: 7 } });
  const stats6 = (() => { n6.port.onmessage({ data: { cmd: "stats" } }); return n6.port.messages.filter((m) => m.cmd === "stats").pop(); })();
  if (stats6.fedSamples !== 8 || stats6.filled !== 4) fail(`stop clear did not reset ledger/ring: ${JSON.stringify(stats6)}`);

  console.log("play-worklet item ledger OK: per-item heard/discarded on clear, FIFO advance, no eviction of pending chunks, cumulative heard, reason echoed, untagged feeds compatible");
}

  // REGRESSION (review): same item fed/drained/fed — the heard prefix must
  // SUM across chunks; a clear mid-item must not report only the latest chunk
  function ledgerCumulativeHeardTest() {
    const n = new PlayWorklet();
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(100).fill(1), itemId: "item_A", gen: 1 } });
    for (let i = 0; i < 100; i++) drain(n, 1); // chunk 1 fully heard
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(50).fill(1), itemId: "item_A", gen: 1 } });
    for (let i = 0; i < 20; i++) drain(n, 1); // chunk 2 partially heard
    n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
    const ack = n.port.messages.filter((m) => m.cmd === "cleared").pop();
    const a = (ack.items || []).find((x) => x.id === "item_A");
    if (!a || a.heardSamples !== 120 || a.discardedSamples !== 30) {
      fail(`cumulative heard wrong: ${JSON.stringify(a)} (expected heard=120 discarded=30)`);
    }
  }

  // REGRESSION (review): interleaved chunks keep ring/FIFO order — samples
  // drained after item B was fed belong to B, not to item A's new chunk
  function ledgerFifoInterleaveTest() {
    const n = new PlayWorklet();
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(100).fill(1), itemId: "item_A", gen: 1 } });
    for (let i = 0; i < 100; i++) drain(n, 1); // A chunk 1 fully heard
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(40).fill(2), itemId: "item_B", gen: 1 } });
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(50).fill(1), itemId: "item_A", gen: 1 } });
    for (let i = 0; i < 10; i++) drain(n, 1); // these 10 are B's (ring order)
    n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
    const ack = n.port.messages.filter((m) => m.cmd === "cleared").pop();
    const a = (ack.items || []).find((x) => x.id === "item_A");
    const b = (ack.items || []).find((x) => x.id === "item_B");
    if (!a || a.heardSamples !== 100 || a.discardedSamples !== 50) {
      fail(`interleave item A wrong: ${JSON.stringify(a)} (expected heard=100 discarded=50)`);
    }
    if (!b || b.heardSamples !== 10 || b.discardedSamples !== 30) {
      fail(`interleave item B wrong: ${JSON.stringify(b)} (expected heard=10 discarded=30)`);
    }
  }

  // REGRESSION (review): >8 buffered items — NO eviction of pending chunks;
  // FIFO attribution follows actual ring order exactly. 45 drains play items
  // 0-3 fully and 5 of item_4; every live chunk stays attributed.
  function ledgerCapFifoAttributionTest() {
    const n = new PlayWorklet();
    for (let k = 0; k < 9; k++) {
      n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(10).fill(1), itemId: `item_${k}`, gen: 1 } });
    }
    for (let i = 0; i < 45; i++) drain(n, 1);
    n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
    const ack = n.port.messages.filter((m) => m.cmd === "cleared").pop();
    if (ack.ledgerDropped !== 0) fail(`ledgerDropped=${ack.ledgerDropped}, expected 0 (no eviction of pending chunks)`);
    const items = ack.items || [];
    const byId = Object.fromEntries(items.map((x) => [x.id, x]));
    if (items.some((x) => ["item_0", "item_1", "item_2", "item_3"].includes(x.id))) {
      fail(`fully heard items must be absent from ack: ${JSON.stringify(items)}`);
    }
    const i4 = byId["item_4"], i5 = byId["item_5"], i6 = byId["item_6"], i7 = byId["item_7"], i8 = byId["item_8"];
    if (!i4 || i4.heardSamples !== 5 || i4.discardedSamples !== 5) fail(`item_4 wrong: ${JSON.stringify(i4)}`);
    for (const [name, it] of [["item_5", i5], ["item_6", i6], ["item_7", i7], ["item_8", i8]]) {
      if (!it || it.heardSamples !== 0 || it.discardedSamples !== 10) fail(`${name} wrong: ${JSON.stringify(it)}`);
    }
  }

  // REGRESSION (review): ring overflow rejects samples BEFORE the ledger —
  // dropped samples may never count as fed/heard/discarded of an item. The
  // fully heard item has no discarded tail and may be ABSENT from the ack
  // list; the proof is in the stats/overflow counters instead.
  function ledgerOverflowNotHeardTest() {
    const n = new PlayWorklet();
    n.capacity = 4;
    n.ring = new Float32Array(4);
    n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array([1, 2, 3, 4, 5, 6]).buffer, itemId: "item_A", gen: 1 } });
    const overflow = n.port.messages.filter((m) => m.cmd === "overflow");
    if (overflow.length !== 1 || overflow[0].dropped !== 2) fail(`overflow accounting broken: ${JSON.stringify(overflow)}`);
    for (let i = 0; i < 4; i++) drain(n, 1); // the 4 buffered samples fully heard
    const stats = (() => { n.port.onmessage({ data: { cmd: "stats" } }); return n.port.messages.filter((m) => m.cmd === "stats").pop(); })();
    // global stats: accepted = played + dropped must equal attempted fed
    if (stats.playedSamples + stats.droppedTotal !== stats.fedSamples || stats.playedSamples !== 4 || stats.droppedTotal !== 2) {
      fail(`overflow stats wrong: ${JSON.stringify(stats)} (played 4 + dropped 2 == fed 6)`);
    }
    n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
    const ack = n.port.messages.filter((m) => m.cmd === "cleared").pop();
    if ((ack.items || []).some((x) => x.id === "item_A")) {
      fail(`fully heard item_A must not be in ack: ${JSON.stringify(ack.items)}`);
    }
    if (ack.droppedTotal !== 2) fail(`ack droppedTotal=${ack.droppedTotal}, expected 2 (overflow, not unheard)`);
  }

  // REGRESSION (review): >16 interleaved LIVE items — the heardMap cap
  // applies to COMPLETED (idle) history only; every live key must remain and
  // be counted. FIFO-correct arrangement: one sample per 18 distinct IDs,
  // then a second sample per ID (36 live chunk entries), then drain 18 —
  // the first round is consumed exactly, so EVERY ID ends heard=1/discarded=1.
  function ledgerManyLiveKeysTest() {
    const n = new PlayWorklet();
    for (let k = 0; k < 18; k++) {
      n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(1).fill(1), itemId: `item_${k}`, gen: 1 } });
    }
    for (let k = 0; k < 18; k++) {
      n.port.onmessage({ data: { cmd: "feed", samples: new Int16Array(1).fill(2), itemId: `item_${k}`, gen: 1 } });
    }
    for (let i = 0; i < 18; i++) drain(n, 1); // consumes the first round exactly
    n.port.onmessage({ data: { cmd: "clear", reason: "speech_started" } });
    const ack = n.port.messages.filter((m) => m.cmd === "cleared").pop();
    const items = ack.items || [];
    if (items.length !== 18) fail(`expected 18 live items in ack, got ${items.length}: ${JSON.stringify(items)}`);
    for (let k = 0; k < 18; k++) {
      const it = items.find((x) => x.id === `item_${k}`);
      if (!it || it.heardSamples !== 1 || it.discardedSamples !== 1) {
        fail(`many-live item_${k} wrong: ${JSON.stringify(it)} (expected heard=1 discarded=1)`);
      }
    }
  }

  playRingTest();
  captureTest();
  itemLedgerTest();
  ledgerCumulativeHeardTest();
  ledgerFifoInterleaveTest();
  ledgerCapFifoAttributionTest();
  ledgerOverflowNotHeardTest();
  ledgerManyLiveKeysTest();
  console.log("WORKLET TESTS PASSED");
