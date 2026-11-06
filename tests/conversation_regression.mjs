// Node regression test for the Test Console conversation pipeline (F11).
//
// Loads the SHIPPED `ui/mic-devices.js` + `ui/audio.js` sources (import/export
// syntax stripped, exactly like mic_resample.mjs) into a sandbox with fake
// browser globals, then drives the REAL `RealtimeSession.start()` / `handle()`
// code paths.
//
//  R1  the CONVERSATION path must request echo cancellation, noise suppression
//      and auto gain like the proven baseline (`app.js` 03cb679:783-784), while
//      the voice-SAMPLE recorder (`openMic`) must stay raw (contract §8a).
//  R3  a `speech_started` that the server does NOT confirm as a barge-in
//      (echo-like, during the playback tail) must NOT discard the answer: no
//      `clear` may be sent and the UI must not claim `listening` while audio is
//      still queued. A confirmed cancel (`response.done status=cancelled`) must
//      still flush playback (barge-in stays a feature).

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

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`PASS  ${name}`);
  else {
    failures++;
    console.log(`FAIL  ${name}${detail ? "  (" + detail + ")" : ""}`);
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/* --------------------------------------------------------- fake globals */

class FakeAudioWorkletNode {
  constructor(ctx, name, opts) {
    this.name = name;
    this.opts = opts;
    this.port = { onmessage: null, messages: [], postMessage(d) { this.messages.push(d); } };
    env.nodes.push(this);
  }
  connect() {}
}

class FakeAudioContext {
  constructor(opts) {
    this.sampleRate = (opts && opts.sampleRate) || 48000;
    this.state = "running";
    this.destination = {};
    this.audioWorklet = { addModule: () => Promise.resolve() };
    env.contexts.push(this);
  }
  createMediaStreamSource() { return { connect() {} }; }
  close() { return Promise.resolve(); }
}

class FakeWS {
  constructor(url, protocols) {
    this.url = url;
    this.protocols = protocols;
    this.readyState = 1;
    this.sent = [];
    this.onopen = null; this.onclose = null; this.onerror = null; this.onmessage = null;
    env.sockets.push(this);
  }
  send(data) { this.sent.push(data); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; if (this.onopen) this.onopen(); }
  message(obj) { if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) }); }
}
FakeWS.OPEN = 1;

const env = { contexts: [], nodes: [], sockets: [], gumCalls: [], gumConstraints: [] };

const fakeNavigator = {
  mediaDevices: {
    getUserMedia(constraints) {
      env.gumCalls.push(constraints);
      env.gumConstraints.push(constraints);
      const stream = {
        tracks: [{ stop() {}, label: "Fake Mic", getSettings: () => ({ deviceId: "dev-1" }) }],
        getTracks() { return this.tracks; },
        getAudioTracks() { return this.tracks; },
      };
      return Promise.resolve(stream);
    },
  },
};

const fakeWindow = { addEventListener() {}, removeEventListener() {} };

const exported = new Function(
  "AudioContext", "AudioWorkletNode", "WebSocket", "navigator", "window", "location", "localStorage",
  `${micSrc}\n${audioSrc}\nreturn { RealtimeSession, micConstraints };`,
);
const mod = exported(
  FakeAudioContext, FakeAudioWorkletNode, FakeWS, fakeNavigator, fakeWindow,
  { protocol: "https:", host: "test.example" },
  { getItem: () => null, setItem() {}, removeItem() {} },
);

/* ------------------------------------------------------------------ R1 */

{
  const session = new mod.RealtimeSession({
    token: "t", onEvent() {}, onState() {}, onError() {},
  });
  session.connect = async (model) => {
    const sock = new FakeWS("ws://x", []);
    sock.readyState = 1;
    session.ws = sock;
    session.wsModel = model || "";
    return sock;
  };
  try {
    await session.start("profile:p1");
  } catch (e) {
    // start may reject on a missing piece; the constraint is captured already
  }
  const constraints = env.gumConstraints[env.gumConstraints.length - 1];
  const audio = constraints && constraints.audio;
  check(
    "R1 conversation path requests echoCancellation:true",
    !!audio && audio.echoCancellation === true,
    JSON.stringify(audio),
  );
  check(
    "R1 conversation path requests noiseSuppression:true",
    !!audio && audio.noiseSuppression === true,
    JSON.stringify(audio),
  );
  check(
    "R1 conversation path requests autoGainControl:true",
    !!audio && audio.autoGainControl === true,
    JSON.stringify(audio),
  );

  // the voice-SAMPLE recorder must stay raw (contract §8a) — it is a separate
  // helper (`openMic`) and must NOT be affected by the conversation fix.
  const openMicSrc = loadPlain(join(uiDir, "mic-devices.js"));
  const recorder = new Function(
    "navigator", "localStorage",
    `${openMicSrc}\nreturn { openMic };`,
  )(fakeNavigator, { getItem: () => null, setItem() {}, removeItem() {} });
  await recorder.openMic("dev-1");
  const rawAudio = env.gumConstraints[env.gumConstraints.length - 1].audio;
  check(
    "R1 voice-sample recorder stays raw (echoCancellation:false)",
    rawAudio.echoCancellation === false && rawAudio.noiseSuppression === false && rawAudio.autoGainControl === false,
    JSON.stringify(rawAudio),
  );
}

/* ------------------------------------------------------------------ R3 */

{
  env.contexts.length = 0;
  env.nodes.length = 0;
  env.sockets.length = 0;
  env.gumConstraints.length = 0;

  const states = [];
  const session = new mod.RealtimeSession({
    token: "t",
    onEvent() {},
    onState: (s) => states.push(s),
    onError() {},
  });
  session.connect = async (model) => {
    const sock = new FakeWS("ws://x", []);
    sock.readyState = 1;
    session.ws = sock;
    session.wsModel = model || "";
    return sock;
  };
  await session.start("profile:p1");
  if (!session.recording) {
    // start wiring completed (fakes resolve synchronously)
    await sleep(5);
  }
  const playNode = env.nodes.find((n) => n.name === "play-worklet");
  check("R3 harness: play-worklet created", !!playNode, `${env.nodes.map((n) => n.name)}`);
  const clearCount = () => playNode.port.messages.filter((m) => m.cmd === "clear").length;
  const feeds = () => playNode.port.messages.filter((m) => m.cmd === "feed").length;

  // ---- R4: drain-based `listening` (no barge-in) ---------------------------
  // A completed answer whose buffered tail is still playing. `response.done`
  // arrives before the browser has finished the audio (output pacing is off).
  // The UI must NOT report `listening` until the worklet ring really drained,
  // and `response.done(completed)` itself must not clear the ring.
  session.handle({ type: "session.created", session: { id: "sess_ir", turn_detection: { interrupt_response: true } } });
  session.handle({ type: "response.created", response: { id: "resp_1" } });
  session.handle({ type: "response.output_item.added", response_id: "resp_1", item: { id: "item_1", type: "message" } });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_1", delta: btoa("\x00\x00".repeat(240)) });
  const beforeDoneClear = clearCount();
  session.handle({ type: "response.done", response: { id: "resp_1", status: "completed" } });
  check(
    "R4 completed response.done does not clear the buffered tail",
    clearCount() === beforeDoneClear,
    `clear count ${beforeDoneClear} -> ${clearCount()}`,
  );
  check(
    "R4 UI does not report listening while audio is still queued",
    session.state !== "listening",
    `state=${session.state} states=${JSON.stringify(states.slice(-4))}`,
  );
  playNode.port.onmessage({ data: { cmd: "idle" } });
  check(
    "R4 playback drain settles the UI to listening",
    session.state === "listening",
    `state=${session.state}`,
  );

  // ---- R5: genuine barge-in in the playback tail (interrupt_response on) ---
  // Baseline 03cb679 app.js:320-338 clears on EVERY speech_started when the
  // session flag is on, explicitly including audio buffered after a finished
  // response. With output_pacing off this is the common case: the user speaks
  // into the tail, the server has no active response to cancel, and the client
  // itself must stop the assistant.
  const beforeTailClear = clearCount();
  session.handle({ type: "input_audio_buffer.speech_started", audio_start_ms: 100 });
  check(
    "R5 tail barge-in (interrupt_response on) flushes the buffered answer",
    clearCount() > beforeTailClear,
    `clear count ${beforeTailClear} -> ${clearCount()}`,
  );
  check(
    "R5 tail barge-in reports listening at once",
    session.state === "listening",
    `state=${session.state}`,
  );
  // After response.done there is no active response to fence (baseline fences
  // only the ACTIVE id); the flush above is the baseline's tail behaviour.

  // ---- R5b: interrupt_response=false keeps the audio (baseline 172/323) ----
  session.handle({ type: "session.created", session: { id: "sess_nir", turn_detection: { interrupt_response: false } } });
  session.handle({ type: "response.created", response: { id: "resp_3" } });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_3", delta: btoa("\x00\x00".repeat(240)) });
  session.handle({ type: "response.done", response: { id: "resp_3", status: "completed" } });
  const beforeNoIr = clearCount();
  session.handle({ type: "input_audio_buffer.speech_started", audio_start_ms: 300 });
  check(
    "R5b interrupt_response=false does not flush the buffered answer",
    clearCount() === beforeNoIr,
    `clear count ${beforeNoIr} -> ${clearCount()}`,
  );
  // deltas must still be accepted while the tail plays
  const noIrFeeds = feeds();
  session.handle({ type: "response.output_audio.delta", response_id: "resp_3", delta: btoa("\x00\x00".repeat(24)) });
  check(
    "R5b interrupt_response=false keeps accepting answer audio",
    feeds() === noIrFeeds + 1,
    "feed not queued with interrupt_response=false",
  );
  playNode.port.onmessage({ data: { cmd: "idle" } });

  // ---- active-response barge-in still fences + flushes ---------------------
  session.handle({ type: "session.created", session: { id: "sess_ir2", turn_detection: { interrupt_response: true } } });
  session.handle({ type: "response.created", response: { id: "resp_4" } });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_4", delta: btoa("\x00\x00".repeat(240)) });
  const beforeActiveClear = clearCount();
  session.handle({ type: "input_audio_buffer.speech_started", audio_start_ms: 400 });
  check(
    "R3 active-response barge-in flushes playback immediately",
    clearCount() > beforeActiveClear,
    `clear count ${beforeActiveClear} -> ${clearCount()}`,
  );
  const activeFeeds = feeds();
  session.handle({ type: "response.output_audio.delta", response_id: "resp_4", delta: btoa("\x00\x00".repeat(24)) });
  check(
    "R3 active-response barge-in fences further deltas",
    feeds() === activeFeeds,
    `feed count ${activeFeeds} -> ${feeds()}`,
  );
  // the server's late cancelled done is still processed
  session.handle({ type: "response.done", response: { id: "resp_4", status: "cancelled" } });
  check(
    "R3 confirmed cancel after a fence still clears and resets",
    clearCount() > beforeActiveClear && session.activeResponseId === null,
    `clears=${clearCount()} active=${session.activeResponseId}`,
  );
}

console.log(failures === 0 ? "ALL CONVERSATION REGRESSION TESTS PASSED" : `${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);