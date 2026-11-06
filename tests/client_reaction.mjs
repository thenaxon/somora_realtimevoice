// Harness used by tests/test_conversation_feedback.py.
//
// Reads a JSON array of server events from argv[2], replays them through the
// REAL shipped `ui/mic-devices.js` + `ui/audio.js` client, and prints one JSON
// object describing the client's reaction: how many playback `clear` commands
// were issued and the final talk state. This is how the R2 integration test
// proves the client consequence of a synthetic echo, without a real speaker.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = join(here, "..", "src", "realtimevoice", "web", "ui");

function loadPlain(file) {
  let src = readFileSync(file, "utf8");
  src = src.replace(/^import\s+.*?;\s*$/gm, "");
  src = src.replace(/\bexport\s+(async\s+function|function|class|const|let|var)\b/g, "$1");
  return src;
}

const micSrc = loadPlain(join(uiDir, "mic-devices.js"));
const audioSrc = loadPlain(join(uiDir, "audio.js"));

class FakeAudioWorkletNode {
  constructor(ctx, name) {
    this.name = name;
    this.port = { onmessage: null, messages: [], postMessage(d) { this.messages.push(d); } };
    nodes.push(this);
  }
  connect() {}
}
class FakeAudioContext {
  constructor() { this.sampleRate = 48000; this.state = "running"; this.destination = {}; this.audioWorklet = { addModule: () => Promise.resolve() }; }
  createMediaStreamSource() { return { connect() {} }; }
  close() { return Promise.resolve(); }
}
class FakeWS { constructor() { this.readyState = 1; this.sent = []; } send() {} close() {} }

const nodes = [];
const fakeNavigator = {
  mediaDevices: {
    getUserMedia: () => Promise.resolve({
      tracks: [{ stop() {}, label: "Fake", getSettings: () => ({}) }],
      getTracks() { return this.tracks; },
      getAudioTracks() { return this.tracks; },
    }),
  },
};

const mod = new Function(
  "AudioContext", "AudioWorkletNode", "WebSocket", "navigator", "window", "location", "localStorage",
  `${micSrc}\n${audioSrc}\nreturn { RealtimeSession };`,
)(FakeAudioContext, FakeAudioWorkletNode, FakeWS, fakeNavigator,
  { addEventListener() {}, removeEventListener() {} },
  { protocol: "https:", host: "x" },
  { getItem: () => null, setItem() {}, removeItem() {} });

const events = JSON.parse(readFileSync(process.argv[2], "utf8"));

const session = new mod.RealtimeSession({ token: "t", onEvent() {}, onState() {}, onError() {} });
session.connect = async (model) => {
  const sock = new FakeWS();
  session.ws = sock;
  session.wsModel = model || "";
  return sock;
};

await session.start("profile:p1");
const playNode = nodes.find((n) => n.name === "play-worklet");
for (const ev of events) session.handle(ev);
await new Promise((r) => setTimeout(r, 20));

const clearCount = playNode.port.messages.filter((m) => m.cmd === "clear").length;
console.log(JSON.stringify({ cleared: clearCount > 0, clearCount, state: session.state }));
process.exit(0);