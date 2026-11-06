// Node unit test for the shared sample playback controller state machine (F6).
//
// Loads the REAL shipped sample-player.js (stripped import/export) and drives
// it with a fake audio element: play -> pause -> resume keeps position; a
// second sample stops the first; rapid clicks never create overlapping audio;
// ended/error reset to Play; stop() tears down.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "src", "realtimevoice", "web", "ui", "sample-player.js"), "utf8")
  .replace(/^import\s+.*?;\s*$/gm, "")
  .replace(/\bexport\s+(async\s+function|function|class|const|let|var)\b/g, "$1");

globalThis.URL = { createObjectURL: () => "blob:fake", revokeObjectURL: () => {} };

const factory = new Function(`${src}\nreturn { SamplePlayer, PLAYER_STATES };`);
const { SamplePlayer } = factory();

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`PASS  ${name}`);
  else {
    failures++;
    console.log(`FAIL  ${name}${detail ? "  (" + detail + ")" : ""}`);
  }
}

// Fake audio element registry: tracks every created element.
const created = [];
function fakeAudioFactory() {
  const a = {
    src: "",
    paused: true,
    currentTime: 0,
    listeners: {},
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    emit(ev) { (this.listeners[ev] || []).forEach((fn) => fn()); },
    async play() { this.paused = false; return undefined; },
    pause() { this.paused = true; },
    removeAttribute() { this.src = ""; },
    load() {},
  };
  created.push(a);
  return a;
}

function newPlayer(sources) {
  return new SamplePlayer({
    load: async (key) => {
      if (!sources[key]) throw new Error("missing");
      return sources[key];
    },
    createAudio: fakeAudioFactory,
  });
}

async function testTogglesSameElementKeepsPosition() {
  const p = newPlayer({ A: "blob:a" });
  await p.toggle("A");
  check("play -> playing", p.stateFor("A") === "playing", p.stateFor("A"));
  const el = p.audio;
  el.currentTime = 4.2;
  await p.toggle("A"); // pause
  check("pause -> paused", p.stateFor("A") === "paused", p.stateFor("A"));
  check("pause keeps the same element", p.audio === el);
  check("pause keeps position", el.currentTime === 4.2, String(el.currentTime));
  await p.toggle("A"); // resume
  check("resume -> playing", p.stateFor("A") === "playing", p.stateFor("A"));
  check("resume reuses the element", p.audio === el);
  check("resume keeps position", el.currentTime === 4.2, String(el.currentTime));
}

async function testSwitchingStopsPrevious() {
  created.length = 0;
  const p = newPlayer({ A: "blob:a", B: "blob:b" });
  await p.toggle("A");
  const first = p.audio;
  await p.toggle("B");
  check("switching stops the first element", first.paused === true);
  check("switching creates exactly two elements", created.length === 2, String(created.length));
  check("B is playing", p.stateFor("B") === "playing");
  check("A is idle after switch", p.stateFor("A") === "idle");
}

async function testRapidClicksNeverOverlap() {
  created.length = 0;
  let resolvers = [];
  const p = new SamplePlayer({
    load: (key) => new Promise((res) => resolvers.push(() => res("blob:" + key))),
    createAudio: fakeAudioFactory,
  });
  // 4 rapid clicks on the same key while loading
  p.toggle("A");
  p.toggle("A");
  p.toggle("A");
  p.toggle("A");
  resolvers.forEach((r) => r());
  await new Promise((r) => setTimeout(r, 0));
  const playing = created.filter((a) => !a.paused).length;
  check("rapid clicks create at most one playing element", playing <= 1, String(playing));
  check("rapid clicks create at most one element", created.length <= 1, String(created.length));
}

async function testEndedResets() {
  const p = newPlayer({ A: "blob:a" });
  await p.toggle("A");
  const el = p.audio;
  el.emit("ended");
  check("ended resets to idle", p.stateFor("A") === "idle", p.stateFor("A"));
}

async function testErrorResets() {
  const p = newPlayer({});
  await p.toggle("MISSING");
  check("missing source -> error", p.stateFor("MISSING") === "error", p.stateFor("MISSING"));
}

async function testStopTearsDown() {
  const p = newPlayer({ A: "blob:a" });
  await p.toggle("A");
  const el = p.audio;
  p.stop();
  check("stop pauses the element", el.paused === true);
  check("stop resets state", p.stateFor("A") === "idle");
  check("stop clears the active key", p.key === null);
}

async function testPlayFailure() {
  const p = new SamplePlayer({
    load: async () => "blob:a",
    createAudio: () => {
      const a = fakeAudioFactory();
      a.play = async () => { throw new Error("blocked"); };
      return a;
    },
  });
  await p.toggle("A");
  check("play rejection -> error", p.stateFor("A") === "error", p.stateFor("A"));
}

await testTogglesSameElementKeepsPosition();
await testSwitchingStopsPrevious();
await testRapidClicksNeverOverlap();
await testEndedResets();
await testErrorResets();
await testStopTearsDown();
await testPlayFailure();

console.log(failures === 0 ? "ALL SAMPLE PLAYER TESTS PASSED" : `${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);