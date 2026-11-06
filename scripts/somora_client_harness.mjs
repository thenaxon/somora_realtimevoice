// Somora real-Somora-client compatibility harness.
//
// Drives Somora's OWN realtime adapter (OpenAiRealtimeProvider /
// OpenAiRealtimeSession, imported from --somora-root) against a realtimevoice
// instance that speaks the OpenAI GA realtime WebSocket protocol.
//
// Run with Somora's tsx (so the .ts imports resolve):
//
//   <somora-root>/node_modules/.bin/tsx scripts/somora_client_harness.mjs \
//     --somora-root <path> --url ws://127.0.0.1:<port>/v1/realtime \
//     --key-file <file> --model voice-route:auto --voice <profile-id> \
//     --wav <pcm16 24k mono wav> [--wav-barge <wav>] [--wav-tool <wav>] \
//     [--consult-policy always|auto|substantive] [--agent-name Ada] \
//     [--tool-attempts 2] [--tool-wait-ms 20000] [--barge-delay-ms 800] \
//     --out-dir <new dir>
//
// It builds the RealtimeSessionRequest exactly as Somora's call path does
// (buildVoiceInstructions + consultToolSpec/statusToolSpec/resultToolSpec),
// with the same consult policy a real agent session gets (default `always`),
// and records event kinds + timings only: transcripts are reduced to
// length + sha8, audio to byte counts. No token, transcript text or audio is
// ever written.
//
// Privacy: the bearer token is read from --key-file and never logged; SOMORA_HOME
// is pinned to a throwaway dir so importing Somora's logger never touches the
// operator's ~/.somora.

import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);

// ───────────────────────────────────────────────────────────── arguments

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const name = key.slice(2);
    const value = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
    out[name] = value;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const required = ['somora-root', 'url', 'key-file', 'model', 'voice', 'wav', 'out-dir'];
for (const name of required) {
  if (!args[name]) {
    process.stderr.write(`missing required --${name}\n`);
    process.exit(2);
  }
}

const SOMORA_ROOT = resolve(args['somora-root']);
const OUT_DIR = resolve(args['out-dir']);
const KEY_FILE = resolve(args['key-file']);
const WAV_PATH = resolve(args.wav);
const WAV_BARGE_PATH = args['wav-barge'] ? resolve(args['wav-barge']) : '';

// ── consult policy / persona name (S2 live fix #1) ───────────────────
// Real Somora voice sessions for agents build their instructions with the
// agent's effective consult policy (persona.voice.consultPolicy ??
// realtimeVoice.consultPolicy), and a live run showed a model only calls
// the consult tool when the policy text asks for it. Default to the same
// `always` real voice agents get, so the harness talks to the model the way
// Somora's call path does.
const CONSULT_POLICY = ['always', 'auto', 'substantive'].includes(args['consult-policy'])
  ? args['consult-policy']
  : 'always';
const AGENT_NAME = args['agent-name'] || 'Ada';
const TOOL_ATTEMPTS = Math.max(1, Number.parseInt(args['tool-attempts'] || '2', 10) || 2);
const TOOL_WAIT_MS = Math.max(1000, Number.parseInt(args['tool-wait-ms'] || '20000', 10) || 20000);
const BARGE_DELAY_MS = Math.max(0, Number.parseInt(args['barge-delay-ms'] || '800', 10) || 0);
const WAV_TOOL_PATH = args['wav-tool'] ? resolve(args['wav-tool']) : WAV_PATH;
if (!existsSync(join(SOMORA_ROOT, 'src', 'voice', 'realtime', 'openai-provider.ts'))) {
  process.stderr.write(`--somora-root does not look like a somora checkout: ${SOMORA_ROOT}\n`);
  process.exit(2);
}

// Resolve the non-secret wiring (backend control URL, alt voice) written next
// to the key file by scripts/somora_harness_fake_instance.py, unless given.
const configFile = args['config-file']
  ? resolve(args['config-file'])
  : `${KEY_FILE}.config.json`;
let instanceConfig = {};
if (existsSync(configFile)) {
  try {
    instanceConfig = JSON.parse(readFileSync(configFile, 'utf8'));
  } catch {
    instanceConfig = {};
  }
}
const BACKEND_CONTROL_URL = args['backend-control-url'] || instanceConfig.backend_control_url || '';
const VOICE_B = args['voice-b'] || instanceConfig.alt_voice || '';
const KEY = readFileSync(KEY_FILE, 'utf8').trim();

// Pin Somora's home + quiet its logger BEFORE importing anything from somora.
process.env.SOMORA_HOME = process.env.SOMORA_HOME || mkdtempSync(join(tmpdir(), 'somora-harness-home-'));
process.env.SOMORA_LOG_LEVEL = process.env.SOMORA_LOG_LEVEL || 'silent';

mkdirSync(OUT_DIR, { recursive: true });

// ───────────────────────────────────────────────────────────── helpers

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => Date.now();
const sha8 = (text) => createHash('sha256').update(text).digest('hex').slice(0, 8);

function wavToPcm(path) {
  const buf = readFileSync(path);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`not a RIFF/WAVE file: ${path}`);
  }
  let offset = 12;
  let fmt = null;
  let data = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ') {
      fmt = {
        format: buf.readUInt16LE(start),
        channels: buf.readUInt16LE(start + 2),
        sampleRate: buf.readUInt32LE(start + 4),
        bits: buf.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buf.subarray(start, start + size);
    }
    offset = start + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`WAV missing fmt/data chunk: ${path}`);
  if (fmt.format !== 1 || fmt.bits !== 16 || fmt.channels !== 1 || fmt.sampleRate !== 24000) {
    throw new Error(
      `WAV must be PCM16 mono 24 kHz (got format=${fmt.format} bits=${fmt.bits} ch=${fmt.channels} rate=${fmt.sampleRate})`,
    );
  }
  return data;
}

async function controlScript(payload) {
  if (!BACKEND_CONTROL_URL) return;
  await fetch(`${BACKEND_CONTROL_URL}/script`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// ─────────────────────────────────────────────────────── loaded somora bits

const providerUrl = pathToFileURL(join(SOMORA_ROOT, 'src', 'voice', 'realtime', 'openai-provider.ts')).href;
const personaUrl = pathToFileURL(join(SOMORA_ROOT, 'src', 'voice', 'realtime', 'persona.ts')).href;
const consultUrl = pathToFileURL(join(SOMORA_ROOT, 'src', 'voice', 'realtime', 'consult.ts')).href;

const { OpenAiRealtimeProvider } = await import(providerUrl);
const { buildVoiceInstructions } = await import(personaUrl);
const {
  CONSULT_TOOL_NAME,
  consultToolSpec,
  statusToolSpec,
  resultToolSpec,
} = await import(consultUrl);

const WebSocket = require(join(SOMORA_ROOT, 'node_modules', 'ws'));

// A Persona shaped like somora's loader produces (derived/neutral persona).
// The name is the persona name only (like an agent name); the voice-route
// voice is separate and still comes from --voice.
const PERSONA = {
  name: AGENT_NAME,
  description: 'A neutral test persona used by the compatibility harness.',
  systemPrompt: 'You are a neutral voice agent used by a protocol compatibility harness.',
  voice: { enabled: true, voice: args.voice, language: instanceConfig.language || 'de', maxSpokenSentences: 4 },
};

function buildRequest(voice, language) {
  // Same call shape as somora's VoiceCall.start(): persona + consultPolicy +
  // sessionSlug into buildVoiceInstructions, policy passed through untouched.
  const instructions = buildVoiceInstructions({
    persona: PERSONA,
    consultPolicy: CONSULT_POLICY,
    language,
    consultToolName: CONSULT_TOOL_NAME,
    sessionSlug: 'main',
  });
  return {
    model: args.model,
    voice,
    instructions: instructions.text,
    language,
    turnDetection: { threshold: 0.4, prefixPaddingMs: 200, silenceDurationMs: 420 },
    tools: [consultToolSpec(PERSONA.name), statusToolSpec(), resultToolSpec()],
    _instructionChars: instructions.chars,
  };
}

// ─────────────────────────────────────────────────────── capture plumbing

class Run {
  constructor() {
    this.raw = []; // every inbound frame {type, ts, message}
    this.outbound = []; // every outbound frame {type, ts, text}
    this.events = []; // normalized somora events {kind, ts, ...}
    this.sockets = [];
  }

  wrapSocket(url, headers) {
    const ws = new WebSocket(url, { headers });
    const run = this;
    ws.on('message', (data) => {
      let parsed = null;
      try {
        parsed = JSON.parse(String(data));
      } catch {
        return;
      }
      run.raw.push({ type: parsed.type || '', ts: now(), message: parsed });
    });
    run.sockets.push(ws);
    return {
      on(event, cb) {
        ws.on(event, cb);
      },
      send(data) {
        let parsed = null;
        try {
          parsed = JSON.parse(String(data));
        } catch {
          parsed = { type: 'unparseable' };
        }
        run.outbound.push({ type: parsed.type || '', ts: now(), message: parsed });
        ws.send(data);
      },
      close() {
        ws.close();
      },
    };
  }

  consume(session, label) {
    const run = this;
    run._consumers = run._consumers || [];
    let stopped = false;
    const task = (async () => {
      try {
        for await (const ev of session.events()) {
          run.events.push({ ...ev, scenario: label });
          if (stopped) break;
        }
      } catch {
        /* closed */
      }
    })();
    run._consumers.push({ task, stop: () => (stopped = true) });
    return task;
  }

  countRaw(type) {
    return this.raw.filter((r) => r.type === type).length;
  }
  countOutbound(type) {
    return this.outbound.filter((r) => r.type === type).length;
  }
  countEvent(kind, predicate) {
    return this.events.filter((e) => e.kind === kind && (!predicate || predicate(e))).length;
  }
  rawTypes() {
    return this.raw.map((r) => r.type);
  }
  routes() {
    const seen = [];
    for (const r of this.raw) {
      const route = r.message?.session?.x_realtimevoice?.route || r.message?.session?.realtimevoice;
      if (route) seen.push({ type: r.type, ts: r.ts, route });
    }
    return seen;
  }
}

async function waitFor(fn, timeoutMs, label) {
  const deadline = now() + timeoutMs;
  while (now() < deadline) {
    if (fn()) return true;
    await sleep(20);
  }
  throw new Error(`timeout waiting for ${label}`);
}

// ─────────────────────────────────────────────────────── audio streaming

async function streamWav(session, pcm, { chunkMs = 20, silenceMs = 1500, pace = true } = {}) {
  const chunkBytes = Math.round((24000 * chunkMs) / 1000) * 2;
  const send = async (buf) => {
    await session.sendAudio({ base64: buf.toString('base64'), rateHz: 24000 });
    if (pace) await sleep(chunkMs);
  };
  for (let off = 0; off < pcm.length; off += chunkBytes) {
    await send(pcm.subarray(off, Math.min(off + chunkBytes, pcm.length)));
  }
  const silence = Buffer.alloc(chunkBytes);
  const silenceChunks = Math.floor(silenceMs / chunkMs);
  for (let i = 0; i < silenceChunks; i += 1) await send(silence);
  return { chunks: Math.ceil(pcm.length / chunkBytes) + silenceChunks };
}

// ─────────────────────────────────────────────────────────── scenarios

const results = [];
const summary = {
  tool: 'realtimevoice somora client harness',
  started_at: new Date().toISOString(),
  somora_root_basename: SOMORA_ROOT.split('/').pop(),
  model: args.model,
  primary_voice: args.voice,
  alt_voice: VOICE_B,
  consult_policy: CONSULT_POLICY,
  agent_name: AGENT_NAME,
  scenarios: results,
  counts: {},
  timings: {},
  observations: {},
  limits: [],
};

function record(name, pass, detail, extra = {}, options = {}) {
  // 'not_exercised' means the transport never got a chance (the model decided
  // not to act), which is not a harness failure — it is the model's choice.
  const status = options.notExercised ? 'NOT EXERCISED' : pass ? 'PASS' : 'FAIL';
  results.push({ scenario: name, status, detail: String(detail).slice(0, 400), ...extra });
  process.stdout.write(`${status}  ${name}  ${String(detail).slice(0, 200)}\n`);
}

function makeProvider() {
  const run = new Run();
  const provider = new OpenAiRealtimeProvider({
    apiKey: KEY,
    url: args.url,
    connect: (url, headers) => run.wrapSocket(url, headers),
  });
  return { run, provider };
}

// Scenario 1 ── open + configure + route resolved
async function scenarioOpen() {
  const { run, provider } = makeProvider();
  const request = buildRequest(args.voice, instanceConfig.language || 'de');
  const session = await provider.open(request);
  run.consume(session, 'open');
  await waitFor(() => run.countEvent('ready') > 0, 5000, 'ready');
  const update = run.outbound.find((o) => o.type === 'session.update');
  const routeSeen = run.routes();
  const route = routeSeen.map((r) => r.route).find((r) => r && (r.profile || r.profile_id)) || {};
  const configured =
    update &&
    update.message.session?.audio?.input?.turn_detection?.type === 'server_vad' &&
    Array.isArray(update.message.session?.tools) &&
    update.message.session.tools.length === request.tools.length;
  const pass =
    run.countEvent('ready') > 0 &&
    Boolean(configured) &&
    Boolean(route.profile_id || route.profile) &&
    route.resolved !== false;
  record('open+configure+route', pass, `ready=${run.countEvent('ready')} tools=${update?.message.session?.tools?.length} route_profile=${route.profile_id || route.profile || ''} actual_voice=${route.actual_voice || ''}`, {
    route_profile: route.profile_id || route.profile || '',
    actual_voice: route.actual_voice || '',
    instruction_chars: request._instructionChars,
    session_update_types: update ? 'ok' : 'missing',
  });
  summary.timings.open_ready_ms = run.events.find((e) => e.kind === 'ready')?.ts
    ? run.events.find((e) => e.kind === 'ready').ts - run.outbound[0]?.ts
    : null;
  await session.close('scenario-1-done');
  return { run, session };
}

// Scenario 2 ── stream wav, expect transcript + model audio + response done
async function scenarioAudio() {
  await controlScript({ stt: 'Hallo, wie geht es dir?', llm: [{ text: 'Mir geht es gut. Danke der Nachfrage.' }], tts_ms: 600 });
  const { run, provider } = makeProvider();
  const request = buildRequest(args.voice, instanceConfig.language || 'de');
  const session = await provider.open(request);
  run.consume(session, 'audio');
  await waitFor(() => run.countEvent('ready') > 0, 5000, 'ready');
  const pcm = wavToPcm(WAV_PATH);
  const t0 = now();
  await streamWav(session, pcm, { chunkMs: 20, silenceMs: 1500 });
  const ok = await waitFor(
    () => run.countEvent('user_transcript') > 0 && run.countEvent('audio') > 0 && run.countRaw('response.done') > 0,
    20000,
    'transcript+audio+done',
  ).then(() => true).catch(() => false);
  const transcriptEvent = run.events.find((e) => e.kind === 'user_transcript');
  const audioDeltas = run.countRaw('response.output_audio.delta');
  const pass = ok && transcriptEvent && audioDeltas > 0 && run.countEvent('model_transcript') > 0;
  record('stream-audio', pass, `transcript_len=${transcriptEvent?.text?.length || 0} audio_deltas=${audioDeltas} model_transcript_events=${run.countEvent('model_transcript')} response_done=${run.countRaw('response.done')}`, {
    transcript_sha8: transcriptEvent ? sha8(transcriptEvent.text) : '',
    ms_to_first_audio: (() => {
      const first = run.raw.find((r) => r.type === 'response.output_audio.delta');
      return first ? first.ts - t0 : null;
    })(),
  });
  summary.counts.audio_scenario_response_done = run.countRaw('response.done');
  await session.close('scenario-2-done');
  return { run, session };
}

// Scenario 3 ── consult tool round trip + speak filler + deliver:true
//
// S2 live fix #2: whether the model chooses to call the tool is a MODEL
// decision and is reported separately (tool_call_observed) from whether the
// transport path works when it does. Up to --tool-attempts fresh sessions
// re-send the utterance; if no call happens the scenario is NOT EXERCISED
// rather than FAIL (exit code 4 if that is the only shortfall).
async function scenarioTool() {
  const toolArguments = JSON.stringify({ question: 'Wie ist das Wetter in Berlin?' });
  const pcm = wavToPcm(WAV_TOOL_PATH);
  const attempts = [];

  let gotCall = false;
  let run;
  let session;
  let call;
  let fillerDropped = false;
  let spoke = false;
  let responseDoneByStatus = {};
  let responseCreateSent = 0;
  let duplicateSuspected = 0;
  let modelTranscripts = [];
  let audioDeltas = 0;

  for (let attempt = 1; attempt <= TOOL_ATTEMPTS && !gotCall; attempt += 1) {
    // The fake LLM is scripted to call the tool ONLY when the consult policy
    // text is present (the live run's real behaviour); the policy default is
    // `always`, so a real backend sees the same instruction shape.
    await controlScript({
      stt: 'Wie ist das Wetter in Berlin?',
      llm: [
        { tool_call: { name: CONSULT_TOOL_NAME, arguments: toolArguments, id: `call_harness_${attempt}` } },
        { text: 'Laut Auskunft ist es in Berlin sonnig.' },
      ],
      tts_ms: 500,
      require_consult_policy: CONSULT_POLICY,
    });
    const made = makeProvider();
    run = made.run;
    session = await made.provider.open(buildRequest(args.voice, instanceConfig.language || 'de'));
    run.consume(session, 'tool');
    await waitFor(() => run.countEvent('ready') > 0, 5000, 'ready');

    await streamWav(session, pcm, { chunkMs: 20, silenceMs: 1500 });
    const t0 = now();
    gotCall = await waitFor(() => run.countEvent('tool_call') > 0, TOOL_WAIT_MS, 'tool_call')
      .then(() => true)
      .catch(() => false);
    call = run.events.find((e) => e.kind === 'tool_call');
    attempts.push({ attempt, tool_call_observed: gotCall, wait_ms: now() - t0 });
    if (!gotCall) {
      await session.close(`scenario-3-attempt-${attempt}-no-call`);
      continue;
    }

    // Filler vs delivery, with a response genuinely active: sendToolResult
    // immediately asks for the follow-up response (Somora sets responseActive
    // the moment it asks), so a speak() issued right after must be DROPPED
    // (a filler never talks over) while a deliver:true line must be KEPT and
    // spoken once the model falls silent.
    await session.sendToolResult(call.callId, 'Synthetisches Testergebnis (kein echter Agent): In Berlin ist es sonnig.');
    const afterTool = run.countOutbound('response.create');
    await session.speak('Einen kurzen Moment, ich schaue nach.'); // must be dropped
    const afterFiller = run.countOutbound('response.create');
    fillerDropped = afterFiller === afterTool;
    await session.speak('Das Ergebnis liegt jetzt vor, ich trage es kurz vor.', { deliver: true });

    spoke = await waitFor(
      () => run.countRaw('response.done') >= 2,
      30000,
      'tool follow-up + delivery responses done',
    ).then(() => true).catch(() => false);
    modelTranscripts = run.events.filter((e) => e.kind === 'model_transcript' && e.final);
    audioDeltas = run.countRaw('response.output_audio.delta');
    responseDoneByStatus = {};
    for (const r of run.raw.filter((r) => r.type === 'response.done')) {
      const status = r.message.response?.status || 'unknown';
      responseDoneByStatus[status] = (responseDoneByStatus[status] || 0) + 1;
    }
    responseCreateSent = run.countOutbound('response.create');
    // A well-behaved client asks exactly once per trigger; more asks with no
    // new item/instructions are the duplicate-response smell Somora guards
    // against. The tool result and the kept delivery are two triggers.
    const expectedTriggers = 2;
    duplicateSuspected = Math.max(0, responseCreateSent - expectedTriggers);
    await session.close('scenario-3-done');
  }

  const transportPass =
    gotCall && spoke && audioDeltas > 0 && modelTranscripts.length >= 1 && fillerDropped && duplicateSuspected === 0;
  const detail = gotCall
    ? `tool_call_observed=true attempts=${attempts.length} call_id=${call?.callId || ''} model_transcripts=${modelTranscripts.length} filler_dropped=${fillerDropped} response_create_sent=${responseCreateSent} duplicates_suspected=${duplicateSuspected} responses_done=${JSON.stringify(responseDoneByStatus)}`
    : `tool_call_observed=false attempts=${attempts.length} — model did not call the tool`;
  record(
    'tool-roundtrip+speak',
    transportPass,
    detail,
    {
      tool_call_observed: gotCall,
      attempts,
      tool_name: call?.name || '',
      args_parse_ok: (() => {
        try {
          return Boolean(JSON.parse(call?.args || '{}').question);
        } catch {
          return false;
        }
      })(),
      transport_pass: transportPass,
      filler_dropped: fillerDropped,
      response_create_sent: responseCreateSent,
      duplicates_suspected: duplicateSuspected,
      response_done_by_status: responseDoneByStatus,
    },
    { notExercised: !gotCall },
  );
  summary.counts.tool_scenario_tool_call_observed = gotCall;
  summary.counts.tool_scenario_attempts = attempts.length;
  summary.counts.tool_scenario_response_done_by_status = responseDoneByStatus;
  summary.counts.tool_scenario_response_create_sent = responseCreateSent;
  summary.counts.tool_scenario_duplicates_suspected = duplicateSuspected;
  return { run, session };
}

// Measure the cancelled-response audio that arrives after that response's own
// response.done. Attribution is by response_id (the server stamps it on every
// audio delta), so the next turn's audio is never miscounted.
function countDeltasAfterCancelledDone(run) {
  const cancelledDone = run.raw.filter(
    (r) => r.type === 'response.done' && r.message.response?.status === 'cancelled',
  );
  if (cancelledDone.length === 0) return 0;
  const cancelledId = cancelledDone[0].message.response?.id;
  const doneAt = cancelledDone[0].ts;
  return run.raw.filter(
    (r) =>
      r.type === 'response.output_audio.delta' &&
      r.ts > doneAt &&
      r.message.response_id === cancelledId,
  ).length;
}

// Scenario 4 ── barge-in DURING playback; no audio after cancelled done.
//
// S2 live fix #3: the first live run streamed the barge WAV before any model
// audio, so `speaking` was never true and the scenario tested nothing. Now the
// harness streams the question, waits for the first model audio
// (Somora `model_speech start`), waits --barge-delay-ms, and only THEN streams
// the barge WAV. If the response is already done before the barge (the H1
// playback-tail case with output pacing off) that is recorded as its own
// observation with timings instead of a FAIL.
async function scenarioBarge() {
  await controlScript({ stt: 'Erzähl mir bitte etwas über den Hafen.', llm: [{ text: 'Der Hafen ist heute ruhig und die Kräne stehen still.' }], tts_ms: 8000 });
  const { run, provider } = makeProvider();
  const request = buildRequest(args.voice, instanceConfig.language || 'de');
  const session = await provider.open(request);
  run.consume(session, 'barge');
  await waitFor(() => run.countEvent('ready') > 0, 5000, 'ready');

  const bargePcm = WAV_BARGE_PATH ? wavToPcm(WAV_BARGE_PATH) : wavToPcm(WAV_PATH);
  const questionPcm = wavToPcm(WAV_PATH);
  await streamWav(session, questionPcm, { chunkMs: 20, silenceMs: 1500 });

  // Wait for the model to actually start speaking (not merely prepare).
  const speechStart = await waitFor(
    () => run.countEvent('model_speech', (e) => e.phase === 'start') > 0,
    25000,
    'model_speech start',
  ).then(() => true).catch(() => false);
  const speechStartTs = run.events.find((e) => e.kind === 'model_speech' && e.phase === 'start')?.ts || 0;

  const barge = { speech_start: speechStart, response_done_before_barge: false, interrupted: false, cancel_sent: false, deltas_after_cancelled_done: 0, next_answered: false };
  if (speechStart) {
    // A response can finish before the barge if pacing is off (the server
    // sends done as soon as synthesis ends while the client is still playing).
    const alreadyDone = run.raw.some((r) => r.type === 'response.done');
    if (alreadyDone) {
      barge.response_done_before_barge = true;
      barge.done_before_barge_ms = (run.raw.find((r) => r.type === 'response.done')?.ts || 0) - speechStartTs;
      record(
        'barge-in',
        true,
        `response already done before barge (playback tail, output_pacing=${summary.output_pacing}) done_before_barge_ms=${barge.done_before_barge_ms} — scenario not applicable in this profile`,
        barge,
        { notExercised: true },
      );
      summary.observations.barge_response_done_before_barge = true;
      summary.observations.barge_done_before_barge_ms = barge.done_before_barge_ms;
      summary.counts.barge_response_done_before_barge = 1;
      await session.close('scenario-4-done-early');
      return { run, session };
    }
    // Let the playback run into the barge window, then talk over the model.
    await sleep(BARGE_DELAY_MS);
    const bargeStart = now();
    const bargeTask = streamWav(session, bargePcm, { chunkMs: 20, silenceMs: 800 });
    barge.interrupted = await waitFor(() => run.countEvent('interrupted') > 0, 15000, 'interrupted').then(() => true).catch(() => false);
    barge.cancel_sent = run.outbound.some((o) => o.type === 'response.cancel');
    await bargeTask;
    barge.ms_to_interrupt = barge.interrupted
      ? (run.events.find((e) => e.kind === 'interrupted')?.ts || 0) - bargeStart
      : null;
    barge.deltas_after_cancelled_done = countDeltasAfterCancelledDone(run);
  }

  // The next turn must be answered: script a plain answer and stream again.
  await controlScript({ stt: 'Und wie ist es morgen?', llm: [{ text: 'Morgen wird es etwas windiger.' }], tts_ms: 500 });
  const doneBefore = run.countRaw('response.done');
  await streamWav(session, wavToPcm(WAV_PATH), { chunkMs: 20, silenceMs: 1500 });
  barge.next_answered = await waitFor(
    () => run.countRaw('response.done') > doneBefore && run.countRaw('response.output_audio.delta') > 0,
    20000,
    'next turn answered',
  ).then(() => true).catch(() => false);

  // If Somora never reported speaking, the server never had a chance to
  // cancel: that is not a transport failure, it is an untested scenario.
  const exercised = speechStart && !barge.response_done_before_barge;
  const pass = exercised && barge.interrupted && barge.cancel_sent && barge.deltas_after_cancelled_done === 0 && barge.next_answered;
  record(
    'barge-in',
    pass,
    `speech_start=${speechStart} done_before_barge=${barge.response_done_before_barge} interrupted=${barge.interrupted} cancel_sent=${barge.cancel_sent} deltas_after_cancelled_done=${barge.deltas_after_cancelled_done} next_answered=${barge.next_answered} ms_to_interrupt=${barge.ms_to_interrupt ?? ''}`,
    { ...barge, output_pacing: summary.output_pacing },
    { notExercised: !exercised },
  );
  summary.observations.barge = barge;
  summary.counts.barge_deltas_after_cancelled_done = barge.deltas_after_cancelled_done;
  await session.close('scenario-4-done');
  return { run, session };
}

// Scenario 5 ── explicit interrupt() + reconnect with another voice
async function scenarioInterruptAndReconnect() {
  await controlScript({ stt: 'Zähle bitte langsam bis zwanzig.', llm: [{ text: 'Eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn.' }], tts_ms: 3000 });
  const { run, provider } = makeProvider();
  const request = buildRequest(args.voice, instanceConfig.language || 'de');
  const session = await provider.open(request);
  run.consume(session, 'interrupt');
  await waitFor(() => run.countEvent('ready') > 0, 5000, 'ready');
  await streamWav(session, wavToPcm(WAV_PATH), { chunkMs: 20, silenceMs: 1200 });
  const started = await waitFor(() => run.countRaw('response.output_audio.delta') > 0, 20000, 'audio').then(() => true).catch(() => false);
  let explicitInterruptOk = false;
  let deltasAfterCancelledDone = 0;
  if (started) {
    await session.interrupt();
    explicitInterruptOk = await waitFor(
      () => run.raw.some((r) => r.type === 'response.done' && r.message.response?.status === 'cancelled'),
      15000,
      'explicit interrupt cancelled',
    ).then(() => true).catch(() => false);
    const cancelled = run.raw.find((r) => r.type === 'response.done' && r.message.response?.status === 'cancelled');
    if (cancelled) {
      const t = cancelled.ts;
      const cancelledId = cancelled.message.response?.id;
      await sleep(400);
      deltasAfterCancelledDone = run.raw.filter(
        (r) =>
          r.type === 'response.output_audio.delta' &&
          r.ts > t &&
          r.message.response_id === cancelledId,
      ).length;
    }
  }
  await session.close('scenario-5-interrupted');

  // Reconnect with a different voice => a different route.
  let routeDiffers = false;
  let routeB = {};
  if (VOICE_B) {
    await controlScript({ stt: 'Hallo!', llm: [{ text: 'Hallo, schön dich zu hören.' }], tts_ms: 400 });
    const second = makeProvider();
    const requestB = buildRequest(VOICE_B, instanceConfig.language || 'de');
    const sessionB = await second.provider.open(requestB);
    second.run.consume(sessionB, 'reconnect');
    await waitFor(() => second.run.countEvent('ready') > 0, 5000, 'ready-b');
    const routeASeen = run.routes().map((r) => r.route).find((r) => r && (r.profile_id || r.profile)) || {};
    routeB = second.run.routes().map((r) => r.route).find((r) => r && (r.profile_id || r.profile)) || {};
    routeDiffers =
      Boolean(routeB.profile_id || routeB.profile) &&
      (routeB.profile_id || routeB.profile) !== (routeASeen.profile_id || routeASeen.profile);
    await sessionB.close('scenario-5-reconnect-done');
  }

  const pass = explicitInterruptOk && deltasAfterCancelledDone === 0 && (VOICE_B ? routeDiffers : true);
  record('interrupt+reconnect', pass, `explicit_interrupt=${explicitInterruptOk} deltas_after_cancelled_done=${deltasAfterCancelledDone} alt_voice=${VOICE_B || '(none)'} route_differs=${routeDiffers}`, {
    route_a: run.routes().map((r) => r.route?.profile_id || r.route?.profile).filter(Boolean)[0] || '',
    route_b: routeB.profile_id || routeB.profile || '',
    deltas_after_cancelled_done: deltasAfterCancelledDone,
  });
  summary.counts.interrupt_deltas_after_cancelled_done = deltasAfterCancelledDone;
  return { run, session };
}

// ─────────────────────────────────────────────────────────────── main

async function main() {
  // Profile output pacing is a property of the instance the harness talks to,
  // not a harness choice; read it from the config the fake instance writes.
  summary.output_pacing = instanceConfig.output_pacing || 'unknown';
  const runs = {};
  const scenarios = [
    ['open+configure+route', scenarioOpen],
    ['stream-audio', scenarioAudio],
    ['tool-roundtrip+speak', scenarioTool],
    ['barge-in', scenarioBarge],
    ['interrupt+reconnect', scenarioInterruptAndReconnect],
  ];
  for (const [name, fn] of scenarios) {
    const t0 = now();
    try {
      const result = await fn();
      if (result?.run) runs[name] = result.run;
    } catch (err) {
      record(name, false, `exception: ${err.message}`);
    }
    summary.timings[`${name}_ms`] = now() - t0;
  }

  // Event log: kinds + timings only. Transcripts are reduced to length +
  // sha8, audio to delta counts; never transcript text or audio bytes.
  const logLines = [];
  for (const [name, run] of Object.entries(runs)) {
    for (const ev of run.events) {
      const line = { scenario: name, kind: ev.kind, ts: ev.ts };
      if (typeof ev.text === 'string') {
        line.text_len = ev.text.length;
        line.text_sha8 = sha8(ev.text);
      }
      if (typeof ev.phase === 'string') line.phase = ev.phase;
      if (typeof ev.fatal === 'boolean') line.fatal = ev.fatal;
      if (typeof ev.callId === 'string') line.call_id = ev.callId;
      if (typeof ev.name === 'string') line.tool_name = ev.name;
      if (typeof ev.inputTokens === 'number') line.input_tokens = ev.inputTokens;
      if (typeof ev.outputTokens === 'number') line.output_tokens = ev.outputTokens;
      logLines.push(JSON.stringify(line));
    }
  }
  writeFileSync(join(OUT_DIR, 'events.jsonl'), `${logLines.join('\n')}\n`);

  summary.counts.audio_deltas_total = Object.values(runs).reduce(
    (sum, run) => sum + run.countRaw('response.output_audio.delta'),
    0,
  );
  summary.counts.interrupted_events = Object.values(runs).reduce(
    (sum, run) => sum + run.countEvent('interrupted'),
    0,
  );

  if (!VOICE_B) {
    summary.limits.push('no alternate voice supplied (--voice-b / config alt_voice): reconnect route-differs check skipped');
  }
  if (!WAV_BARGE_PATH) {
    summary.limits.push('no --wav-barge supplied: the barge-in scenario reused --wav');
  }
  summary.limits.push('tool-call utterances/instructions are synthetic; whether the model calls the tool is a model decision reported separately');
  summary.limits.push('transcripts are recorded as length + sha8 only; no text or audio is written');
  summary.limits.push(`consult policy sent: ${CONSULT_POLICY} (default always, as real agent voice sessions get)`);

  const failed = results.filter((r) => r.status === 'FAIL');
  const notExercised = results.filter((r) => r.status === 'NOT EXERCISED');
  const exitCode = failed.length ? 3 : notExercised.length ? 4 : 0;
  summary.finished_at = new Date().toISOString();
  summary.exit_code = exitCode;
  summary.pass = failed.length === 0 && notExercised.length === 0;
  summary.not_exercised = notExercised.map((r) => r.scenario);
  writeFileSync(join(OUT_DIR, 'summary.json'), JSON.stringify(summary, null, 2));
  process.stdout.write(`\nsummary: ${join(OUT_DIR, 'summary.json')}\n`);
  if (failed.length) {
    process.stdout.write(`SOMORA CLIENT HARNESS FAILED (${failed.length})\n`);
  } else if (notExercised.length) {
    process.stdout.write(`SOMORA CLIENT HARNESS NOT FULLY EXERCISED (${notExercised.length}: ${notExercised.map((r) => r.scenario).join(', ')})\n`);
  } else {
    process.stdout.write('SOMORA CLIENT HARNESS PASSED\n');
  }
  process.exit(exitCode);
}

await main();