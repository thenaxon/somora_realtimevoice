/* Node unit tests for the pure presentation mappings used by the views:
 * transcript state machine, route display, error-code -> text mapping,
 * model form helpers and the set editor helpers (voice picker, language,
 * issues). Run: node tests/ui_format.mjs */

import assert from "node:assert/strict";
import {
  fmtDuration,
  fmtMs,
  sampleMetaLine,
  transcriptChips,
  transcriptionState,
  transcriptionErrorText,
  uploadErrorText,
  probeErrorText,
  routeFromSession,
  routeWarningText,
  voiceAbility,
  endpointHost,
  changedEndpointKeyPrompt,
  changedEndpointKeyDetail,
  activeVoiceLabel,
  testConsoleOptionLabel,
  testConsoleRouteKind,
  LANGUAGES,
  languageLabel,
  LANGUAGE_FORMAT_CHOICES,
  voiceChoices,
  voiceKeyFromSpec,
  specVoiceFromKey,
  findVoiceChoice,
  setIssueText,
  MODEL_FIELD_HELP,
} from "../src/realtimevoice/web/ui/format.js";

let passed = 0;
function ok(name) {
  passed++;
  console.log("ok - " + name);
}

/* ---- transcript state machine (voice-sample-flow v1.2 §4/§5) ---------- */
{
  const fresh = { transcript: "", transcript_confirmed: false, transcript_source: "", transcription: {} };
  assert.deepEqual(transcriptChips(fresh), ["not_confirmed"], "empty sample is just not_confirmed");
  assert.equal(transcriptionState(fresh), "none", "no transcription");

  const auto = { transcript: "Hallo Welt", transcript_confirmed: false, transcript_source: "stt", transcription: { state: "done" } };
  assert.deepEqual(transcriptChips(auto), ["auto", "not_confirmed"], "auto-transcribed + not confirmed is the normal state");
  assert.equal(transcriptionState(auto), "done");

  const edited = { transcript: "Hallo Wörld", transcript_confirmed: false, transcript_source: "manual" };
  assert.deepEqual(transcriptChips(edited), ["edit", "not_confirmed"], "edited resets confirmation");

  const confirmed = { transcript: "Hallo Welt", transcript_confirmed: true, transcript_source: "stt" };
  assert.deepEqual(transcriptChips(confirmed), ["auto", "confirmed"]);

  const failed = { transcript: "", transcript_confirmed: false, transcription: { state: "failed", error_code: "no_speech_detected" } };
  assert.equal(transcriptionState(failed), "failed", "failed preserves empty transcript");
  assert.deepEqual(transcriptChips(failed), ["not_confirmed"], "no invented text on failure");

  // P1: a hand-typed transcript after a failed attempt is editable/confirmable
  const typedAfterFail = {
    transcript: "Von Hand eingegeben.",
    transcript_confirmed: false,
    transcript_source: "manual",
    transcription: { state: "failed", error_code: "no_speech_detected" },
  };
  assert.equal(transcriptionState(typedAfterFail), "done", "a non-empty transcript is never a dead end");
  assert.deepEqual(transcriptChips(typedAfterFail), ["edit", "not_confirmed"]);
  ok("transcript chips state machine matches the spec (auto/edit + confirmed/unconfirmed)");
}

/* ---- transcription error text mapping --------------------------------- */
{
  assert.match(transcriptionErrorText("backend_unreachable"), /did not answer/);
  assert.match(transcriptionErrorText("backend_timeout"), /too long/);
  assert.match(transcriptionErrorText("stt_failed"), /could not process/);
  assert.match(transcriptionErrorText("no_speech_detected"), /No speech was detected/);
  assert.match(transcriptionErrorText("unknown_code"), /could not process/, "fallback");
  ok("transcription error_code -> operator text mapping");
}

/* ---- upload error text mapping (delta v1.1 §6) ------------------------ */
{
  assert.match(uploadErrorText("sample_too_short"), /at least 1 second/);
  assert.match(uploadErrorText("sample_too_long"), /at most 30 seconds/);
  assert.match(uploadErrorText("unsupported_format"), /Only WAV files/);
  assert.match(uploadErrorText("too_large"), /10 MB/);
  ok("upload rejection error_code -> text mapping");
}

/* ---- probe error text uses the response timeout_ms ---- */
{
  assert.match(probeErrorText("backend_timeout", { timeout_ms: 10000 }), /within 10 seconds/);
  assert.match(probeErrorText("backend_timeout", { timeout_ms: 5000 }), /within 5 seconds/);
  assert.match(probeErrorText("backend_unreachable", {}), /did not answer/);
  ok("probe timeout text is derived from the per-role timeout_ms, never hard-coded");
}

/* ---- P1 model_not_listed is explained with the configured model id ----- */
{
  const text = probeErrorText("model_not_listed", { model: "gemma-4-26b" });
  assert.match(text, /model gemma-4-26b is not in its model list/);
  assert.match(probeErrorText("model_not_listed", {}), /the configured one/);
  assert.match(probeErrorText("model_not_listed", {}), /not in its model list/);
  ok("model_not_listed names the configured model and stays readable without it");
}

/* ---- what a speaking model does with voices ---------------------------- */
{
  assert.equal(voiceAbility({ voice_cloning: "reference_audio", native_voices: [] }, null), "clones");
  assert.equal(voiceAbility({ voice_cloning: "none", native_voices: ["ryan"] }, null), "builtin");
  assert.equal(voiceAbility(null, { voice_cloning: "none", voices: [] }), "fixed");
  assert.equal(voiceAbility(null, { voice_cloning: "reference_audio" }), "clones");
  ok("a speaking model clones, has built-in voices, or speaks with one voice");
}

/* ---- route display comes ONLY from x_realtimevoice.route ------------- */
{
  const session = {
    id: "sess_1",
    x_realtimevoice: {
      route: {
        mode: "profile",
        profile: "max-de",
        revision: 8,
        stt_model: "Qwen3-ASR 0.6B",
        llm_model: "Gemma-4 26B",
        tts_model: "Chatterbox fp16",
        voice: { kind: "library", voice_id: "max", ref_sha256: "abc" },
        language: "de",
      },
      warnings: [{ code: "language_overridden_by_profile", requested_language: "en" }],
    },
  };
  const r = routeFromSession(session);
  assert.equal(r.route.profile, "max-de");
  assert.equal(r.route.voice.voice_id, "max");
  assert.equal(r.warnings.length, 1);
  assert.equal(routeFromSession({ id: "x" }), null, "no extension => no route");
  assert.equal(routeFromSession({ x_realtimevoice: {} }), null, "no route block => null");
  assert.equal(routeFromSession(null), null);
  ok("route display resolves only from session.updated x_realtimevoice.route");

  // the backend in this worktree emits session.realtimevoice with legacy
  // voice-route fields (profile_id/actual_voice); normalise without inventing
  const legacy = {
    realtimevoice: {
      route: { mode: "explicit", resolved: true, profile_id: "max-de", actual_voice: "max", tts_model: "Chatterbox fp16" },
    },
  };
  const lr = routeFromSession(legacy);
  assert.equal(lr.route.profile, "max-de", "profile_id normalised to profile");
  assert.equal(lr.route.voice.voice_id, "max", "actual_voice normalised to a native voice");
  assert.equal(lr.route.mode, "explicit");
  ok("legacy session.realtimevoice route is normalised (documented backend gap)");
}

/* ---- route warnings ---------------------------------------------------- */
{
  const text = routeWarningText({ code: "language_overridden_by_profile", requested_language: "English" });
  assert.match(text, /asked for English/);
  assert.match(text, /wins/);
  assert.equal(routeWarningText("language_overridden_by_profile"), routeWarningText({ code: "language_overridden_by_profile" }));
  assert.equal(routeWarningText("some_other_warning"), "some_other_warning");
  ok("route warnings map to yellow-notice sentences (language override)");
}

/* ---- formatting -------------------------------------------------------- */
{
  assert.equal(fmtDuration(14000), "0:14");
  assert.equal(fmtDuration(65000), "1:05");
  assert.equal(fmtDuration(null), "—");
  assert.equal(fmtMs(120), "120 ms");
  assert.equal(fmtMs(1310), "1.31 s");
  const meta = sampleMetaLine({ duration_ms: 14000, sample_rate: 24000, metrics: { peak_dbfs: -18 } });
  assert.match(meta, /0:14 · 24 kHz · -18 dBFS peak/);
  ok("duration / latency / sample-meta formatting");
}

/* ------------------------------------------- F8: dB display scale for levels */
{
  const { levelToDisplay, amplitudeToDbfs } = await import("../src/realtimevoice/web/ui/format.js");

  // digital silence must stay silent - the floor clamps, it never lifts audio
  assert.equal(levelToDisplay(0), 0, "silence must render as 0");
  assert.equal(levelToDisplay(-1), 0, "negative amplitude is not a level");
  assert.equal(levelToDisplay(NaN), 0, "NaN is not a level");
  assert.equal(levelToDisplay(0.0001), 0, "-80 dBFS is below the -60 floor");

  // full scale
  assert.equal(levelToDisplay(1), 1, "full scale is 1");

  // the actual complaint: normal speech must be clearly visible.
  // Alex's recording peaked at -16.95 dBFS; linear that is 0.142 -> 14 %.
  const alexPeak = 10 ** (-16.95 / 20);
  const alexDisplay = levelToDisplay(alexPeak);
  assert.ok(alexDisplay > 0.6,
    `normal speech (-16.95 dBFS) must be clearly visible, got ${alexDisplay.toFixed(3)}`);
  assert.ok(alexPeak < 0.15, "…and it is only 0.14 on the linear scale that caused the bug");

  // -20 dBFS -> ~0.67, -40 -> ~0.33 (monotonic, evenly spaced in dB)
  assert.ok(Math.abs(levelToDisplay(10 ** (-20 / 20)) - 2 / 3) < 0.01, "-20 dBFS ~ 0.67");
  assert.ok(Math.abs(levelToDisplay(10 ** (-40 / 20)) - 1 / 3) < 0.01, "-40 dBFS ~ 0.33");

  // monotonic
  let prev = -1;
  for (const db of [-60, -50, -40, -30, -20, -10, 0]) {
    const v = levelToDisplay(10 ** (db / 20));
    assert.ok(v >= prev, `display level must be monotonic in dB (at ${db})`);
    prev = v;
  }

  assert.equal(amplitudeToDbfs(0), null, "silence has no dBFS");
  assert.ok(Math.abs(amplitudeToDbfs(1)) < 1e-9, "full scale is 0 dBFS");
  console.log("format: F8 dB level scale OK");
}

/* ------------------------------------------ F10: reference length helpers */
{
  const { referenceFit, referenceTooLongText, excerptRoute, referenceTooLongIssue, fmtSeconds } =
    await import("../src/realtimevoice/web/ui/format.js");

  // Alex's case, exactly
  const f = referenceFit(12200, 10000);
  assert.equal(f.state, "too_long");
  assert.equal(f.overMs, 2200);
  assert.equal(referenceFit(4700, 10000).state, "fits");
  assert.equal(referenceFit(10000, 10000).state, "fits", "equal to the limit fits (server uses >)");
  // no limit advertised (e.g. Qwen3-TTS Base) -> never blocks
  assert.equal(referenceFit(25000, 0).state, "no_limit");
  // unknown client-side duration is NOT reported as fitting
  assert.equal(referenceFit(0, 10000).state, "unknown");

  const text = referenceTooLongText(12200, 10000);
  assert.ok(text.includes("12.2 s") && text.includes("10.0 s"), text);
  assert.ok(/unchanged/.test(text), "must reassure the original is kept");
  assert.ok(!/reference_length|max_reference_ms|ms exceeds/.test(text), "no internal codes in user text");

  const route = excerptRoute("alex", "smp_b8de09fee4473a09", 10000, "alex");
  assert.equal(route, "#/voices/alex?excerpt=smp_b8de09fee4473a09&max=10000&from=alex");

  const issue = { code: "reference_too_long", sample_ms: 12200, max_ms: 10000 };
  assert.equal(referenceTooLongIssue({ issues: [{ code: "missing" }, issue] }), issue);
  assert.equal(referenceTooLongIssue([issue]), issue);
  assert.equal(referenceTooLongIssue(new Error("x")), null);

  assert.equal(fmtSeconds(12200), "12.2 s");
  assert.equal(fmtSeconds(0), "—");
  console.log("format: F10 reference length helpers OK");
}

/* ---------------------------------------- F10: excerpts are recognisable */
{
  const { sampleTitle } = await import("../src/realtimevoice/web/ui/format.js");
  const mic = { source: "mic", created_at: 1789543798 };
  const ex = { source: "mic", created_at: 1789543900, original_filename: "excerpt 1.0-11.0 s of recording.wav",
               derived_from: { sample_id: "smp_a", start_ms: 1000, end_ms: 11000 } };
  assert.ok(sampleTitle(mic).startsWith("Recorded "), sampleTitle(mic));
  assert.equal(sampleTitle(ex), "Excerpt 1.0–11.0 s");
  assert.notEqual(sampleTitle(ex), sampleTitle({ ...ex, derived_from: {} }), "an excerpt must not look like a plain take");
  console.log("format: F10 excerpt title OK");
}

/* ------------------------------- CP1: clone engine constraint + route voice */
{
  const { engineCloneConstraint, cloneEngineLabel, routeFromSession } =
    await import("../src/realtimevoice/web/ui/format.js");

  // Chatterbox: 10 s cap, silent cut
  assert.match(engineCloneConstraint({ max_reference_ms: 10000 }), /up to 10\.0 s/);
  assert.match(engineCloneConstraint({ max_reference_ms: 10000 }), /silently ignored/);
  // Qwen3-TTS Base: whole reference + transcript required
  assert.match(engineCloneConstraint({ reference_text_required: true, max_reference_ms: 0 }),
    /whole reference; a matching transcript is required/);
  // engine without a limit and without a transcript requirement
  assert.equal(engineCloneConstraint({ max_reference_ms: 0 }), "uses the whole reference");

  const grouped = cloneEngineLabel({
    id: "chatterbox-clara", display_name: "Chatterbox", model: "chatterbox-fp16",
    max_reference_ms: 10000, variants: ["chatterbox-clara", "chatterbox-jonas"],
    embeds_reference: true, role: "tts",
  });
  assert.match(grouped, /^Chatterbox \(chatterbox-fp16\)/);
  assert.match(grouped, /up to 10\.0 s/);
  assert.match(grouped, /2 installed variants/);
  assert.match(grouped, /built-in reference/);
  assert.match(grouped, /stored as chatterbox-clara/);
  assert.ok(!/host|:9000|http/.test(grouped), "no endpoint may appear in the label");

  // C3: the route display normalises display_voice; library voice_id still
  // available, and a plain actual_voice falls back cleanly
  const r = routeFromSession({ x_realtimevoice: { route: {
    mode: "profile", profile: "alex-test", tts_model: "chatterbox-fp16",
    voice: { kind: "library", voice_id: "alex-1" }, display_voice: "Alex",
  } } });
  assert.equal(r.route.voice.voice_id, "alex-1");
  assert.equal(r.route.display_voice, "Alex");
  const legacy = routeFromSession({ realtimevoice: { route: {
    mode: "explicit", resolved: true, profile_id: "p", actual_voice: "clara-reference",
  } } });
  assert.equal(legacy.route.display_voice, "clara-reference", "falls back to actual_voice");
  console.log("format: CP1 clone constraint + route display_voice OK");
}

/* ------------------- CP1 correction: engine options named after the MODEL */
{
  const { engineModelLabel, cloneEngineLabel } =
    await import("../src/realtimevoice/web/ui/format.js");

  // The exact model ids from the brief must read as a human-readable model
  // name, never as a stored row's display_name.
  const cases = [
    ["chatterbox-fp16", "tts", "Chatterbox (chatterbox-fp16)"],
    ["Qwen3-ASR-0.6B-8bit", "stt", "Qwen3-ASR 0.6B 8bit"],
    ["Qwen3-ASR-1.7B-8bit", "stt", "Qwen3-ASR 1.7B 8bit"],
    ["Qwen3-TTS-12Hz-1.7B-Base-8bit", "tts", "Qwen3-TTS 1.7B Base 8bit"],
    ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "tts", "Qwen3-TTS 1.7B CustomVoice 8bit"],
    ["mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", "tts", "Voxtral 4B TTS"],
    ["whisper-large-v3-turbo", "stt", "Whisper large-v3-turbo"],
    ["gemma-4-26b-a4b-it-4bit", "llm", "Gemma 4 26B"],
    ["qwen3-14b-4bit", "llm", "Qwen3 14B"],
  ];
  for (const [model, role, want] of cases) {
    assert.equal(engineModelLabel(model, role), want, `label for ${model}`);
  }

  // Never a speaker reference, never "Baseline"/"snapshot", never a host:port.
  const rows = [
    { id: "chatterbox-clara", display_name: "Chatterbox / Clara", model: "chatterbox-fp16" },
    { id: "chatterbox-jonas", display_name: "Chatterbox / Jonas", model: "chatterbox-fp16" },
    { id: "stt-baseline", display_name: "Baseline STT", model: "Qwen3-ASR-0.6B-8bit" },
    { id: "stt-snapshot", display_name: "STT snapshot", model: "Qwen3-ASR-0.6B-8bit" },
    { id: "llm-baseline", display_name: "Baseline LLM", model: "gemma-4-26b-a4b-it-4bit" },
  ];
  for (const r of rows) {
    const label = engineModelLabel(r.model, "tts");
    assert.ok(!/clara|jonas|baseline|snapshot/i.test(label), `row name leaked into ${label}`);
  }
  // a model id that a row happens to embed no host/port must never reach a label
  const hosty = engineModelLabel("http://mac.local:9000/v1", "tts");
  assert.ok(!/mac\.local|9000|:\/\//.test(hosty), `host/port leaked into ${hosty}`);

  // cloneEngineLabel must be model-first too, with the row id only secondary
  const clone = cloneEngineLabel({
    id: "chatterbox-clara", display_name: "Chatterbox / Clara", model: "chatterbox-fp16",
    max_reference_ms: 10000, variants: ["chatterbox-jonas", "chatterbox-clara"],
    embeds_reference: true, role: "tts",
  });
  assert.ok(clone.startsWith("Chatterbox (chatterbox-fp16)"), clone);
  assert.ok(!/clara|baseline|snapshot/i.test(clone.split("stored as")[0]), clone);
  assert.match(clone, /up to 10\.0 s/);
  assert.ok(!/mac\.local|9000|http:\/\//.test(clone), clone);
  console.log("format: CP1 engine model labels OK");
}

/* ------------------------------- F13: model-thinking row option (LLM only) */
{
  const { thinkingModeFromExtra, extraWithThinking, THINKING_MODES } =
    await import("../src/realtimevoice/web/ui/format.js");

  assert.deepEqual(THINKING_MODES, ["default", "off", "on"]);
  assert.equal(thinkingModeFromExtra({}), "default", "absent => backend default");
  assert.equal(thinkingModeFromExtra({ chat_template_kwargs: {} }), "default");
  assert.equal(thinkingModeFromExtra({ chat_template_kwargs: { enable_thinking: false } }), "off");
  assert.equal(thinkingModeFromExtra({ chat_template_kwargs: { enable_thinking: true } }), "on");
  // unrelated extra is never mistaken for the option
  assert.equal(thinkingModeFromExtra({ max_tokens: 200 }), "default");

  // default REMOVES the key via an explicit JSON null when it was present
  // (omitting it would let the server's recursive merge resurrect the stored
  // value), and leaves an extra that never had it untouched.
  const base = { max_tokens: 200, chat_template_kwargs: { top_k: 5 } };
  const d = extraWithThinking(base, "default");
  assert.deepEqual(d, base, "a row without the key is unchanged by default");
  assert.equal("enable_thinking" in (d.chat_template_kwargs || {}), false);
  assert.equal("enable_thinking" in base.chat_template_kwargs, false, "input not mutated");

  const off = extraWithThinking(base, "off");
  assert.equal(off.chat_template_kwargs.enable_thinking, false);
  const on = extraWithThinking(off, "on");
  assert.equal(on.chat_template_kwargs.enable_thinking, true);
  // round trip back to default: an explicit null signals removal to the merge
  const back = extraWithThinking(on, "default");
  assert.equal(back.chat_template_kwargs.enable_thinking, null, "null removes the stored key");
  assert.equal(back.chat_template_kwargs.top_k, 5, "sibling kwargs preserved");

  // an empty extra with default stays empty ({}, no chat_template_kwargs)
  assert.deepEqual(extraWithThinking(undefined, "default"), {});
  console.log("format: F13 thinking option mapping OK");
}

/* --------------- R2 / CP3: add-backend flow helpers (role fields) */
{
  const {
    backendRoleFields,
    BACKEND_ROLE_CHOICES,
    NATIVE_VOICE_LABEL_MODES,
  } = await import("../src/realtimevoice/web/ui/format.js");

  // step 1: three role choices with full labels and one-line help
  assert.equal(BACKEND_ROLE_CHOICES.length, 3);
  for (const c of BACKEND_ROLE_CHOICES) {
    assert.ok(["stt", "llm", "tts"].includes(c.role), c.role);
    assert.ok(c.label.length > 8 && c.help.length > 10, JSON.stringify(c));
    assert.ok(!/\.\.\./.test(c.label), "role label must not be truncated");
  }
  assert.equal(backendRoleFields("stt").includes("language"), true);
  assert.equal(backendRoleFields("llm").includes("thinking"), true);
  assert.equal(backendRoleFields("llm").includes("voice"), false);
  assert.equal(backendRoleFields("stt").includes("thinking"), false);
  assert.equal(backendRoleFields("tts").includes("voice"), true);
  assert.equal(backendRoleFields("tts").includes("native_trust"), true);
  assert.equal(backendRoleFields("stt").includes("native_trust"), false);

  // native-voice trust option exists with the three honest values
  assert.deepEqual(NATIVE_VOICE_LABEL_MODES, ["unknown", "honoured", "ignored"]);
  console.log("format: CP3 add-backend helpers OK");
}

/* --------------- R2 / C3: changed-endpoint key confirmation ------------- */
{
  // the operator must see the host a kept key would be sent to
  assert.equal(endpointHost("http://evil.test/v1"), "evil.test");
  assert.equal(endpointHost("https://Evil.Test:8443/v1/"), "evil.test:8443");
  assert.equal(endpointHost(""), "");
  assert.equal(endpointHost("not a url"), "not a url");

  const prompt = changedEndpointKeyPrompt("http://evil.test/v1");
  assert.match(prompt, /endpoint changed/i);
  assert.match(prompt, /API key/i);
  assert.match(prompt, /evil\.test/, "names the new host");
  assert.match(prompt, /keep sending the existing key/i);
  // no host -> a safe, non-empty fallback
  assert.match(changedEndpointKeyPrompt(""), /new endpoint/i);

  const detail = changedEndpointKeyDetail();
  assert.match(detail, /secret|key/i);
  assert.match(detail, /new endpoint/i);
  console.log("format: C3 changed-endpoint key helpers OK");
}

/* --------------- model form: what it does with voices ------------------- */
{
  const { voiceModeFromCaps, capsForVoiceMode, MODEL_FIELD_HELP, VOICE_MODE_CHOICES, VOICE_MODE_TRUST } = await import("../src/realtimevoice/web/ui/format.js");
  assert.equal(voiceModeFromCaps({ voice_cloning: "reference_audio" }), "clones");
  assert.equal(voiceModeFromCaps({ voice_cloning: "none", voices: ["ryan", "serena"] }), "builtin");
  assert.equal(voiceModeFromCaps({ voice_cloning: "none", voices: [] }), "fixed");
  assert.equal(voiceModeFromCaps({ voice_cloning: "unknown" }), "unknown");
  assert.equal(voiceModeFromCaps({}), "unknown");
  // not verified stays not verified: saving it unchanged keeps "unknown"
  const unverified = { voice_cloning: "unknown", voices: ["a", "b"] };
  assert.equal(voiceModeFromCaps(unverified), "unknown");
  assert.equal(capsForVoiceMode(voiceModeFromCaps(unverified), unverified).voice_cloning, "unknown");
  assert.deepEqual(capsForVoiceMode("unknown", unverified).voices, ["a", "b"]);

  // the trust field is the person's: never overridden on save
  for (const t of ["unknown", "ignored", "honoured"]) {
    assert.equal(capsForVoiceMode("builtin", { voices: ["a"], native_voice_labels: t }).native_voice_labels, t);
  }
  assert.equal(VOICE_MODE_TRUST.builtin, "honoured");
  assert.equal(VOICE_MODE_TRUST.fixed, "ignored");
  assert.deepEqual(capsForVoiceMode("fixed", { voices: ["a"] }).voices, []);
  // a recording limit belongs to a cloning model only
  assert.equal(capsForVoiceMode("clones", { max_reference_ms: 10000 }).max_reference_ms, 10000);
  assert.equal(capsForVoiceMode("clones", { max_reference_ms: 10000 }).voice_cloning, "reference_audio");
  assert.equal(capsForVoiceMode("builtin", { max_reference_ms: 10000, voices: ["a"] }).max_reference_ms, 0);

  // every answer has a help text for its voice field
  for (const c of VOICE_MODE_CHOICES) assert.ok(MODEL_FIELD_HELP.voice[c.value], c.value);
  for (const role of ["stt", "llm", "tts"]) assert.ok(MODEL_FIELD_HELP.extra[role], role);
  console.log("format: model form voice answers OK");
}

/* --------------- sets: the voice picker groups -------------------------- */
{
  const tts = [
    { id: "qwen-custom", display_name: "Qwen3-TTS CustomVoice", capabilities: { voice_cloning: "none", voices: ["ryan", "serena"], native_voice_labels: "honoured" } },
    // voice names the server may ignore are not offered as separate voices
    { id: "unsure-tts", display_name: "Unsure TTS", capabilities: { voice_cloning: "none", voices: ["a", "b"], native_voice_labels: "unknown" } },
    { id: "ignoring-tts", display_name: "Ignoring TTS", capabilities: { voice_cloning: "none", voices: ["c"], native_voice_labels: "ignored" } },
    { id: "chatterbox", display_name: "Chatterbox", capabilities: { voice_cloning: "reference_audio", voices: [] } },
    { id: "plain-tts", display_name: "Plain TTS", capabilities: { voice_cloning: "none", voices: [] } },
    { id: "base-tts", display_name: "Base TTS", capabilities: { voice_cloning: "reference_audio" } },
  ];
  const voices = [
    { id: "zed", display_name: "Zed", reference_sample_id: "s1", samples: [{ id: "s1", transcript_confirmed: true }] },
    { id: "anna", display_name: "Anna", reference_sample_id: "s2", samples: [{ id: "s2", transcript_confirmed: false }] },
    { id: "bert", display_name: "Bert", reference_sample_id: "", samples: [] },
  ];
  const groups = voiceChoices(tts, voices);
  assert.deepEqual(groups.map((g) => g.label),
    ["Your voices", "Qwen3-TTS CustomVoice: its own voices", "Models with one voice"]);

  const yours = groups[0].items;
  assert.deepEqual(yours.map((i) => i.label), ["Anna", "Bert", "Zed"], "own voices sorted by name");
  for (const it of yours) {
    assert.equal(it.kind, "library");
    assert.deepEqual(it.modelIds, ["base-tts", "chatterbox"], "spoken by every cloning model (sorted by name)");
  }
  assert.equal(yours.find((i) => i.voiceId === "zed").note, "", "a confirmed recording needs no note");
  assert.match(yours.find((i) => i.voiceId === "anna").note, /not confirmed/);
  assert.match(yours.find((i) => i.voiceId === "bert").note, /no recording/);

  const builtIn = groups[1].items;
  assert.deepEqual(builtIn.map((i) => [i.key, i.kind, i.nativeVoice, i.modelIds]), [
    ["nat:qwen-custom:ryan", "native", "ryan", ["qwen-custom"]],
    ["nat:qwen-custom:serena", "native", "serena", ["qwen-custom"]],
  ]);
  assert.deepEqual(groups[2].items.map((i) => [i.key, i.kind, i.label]), [
    ["own:ignoring-tts", "none", "Ignoring TTS"],
    ["own:plain-tts", "none", "Plain TTS"],
    ["own:unsure-tts", "none", "Unsure TTS"],
  ], "voice names that are not honoured: the model counts as one voice");
  assert.equal(findVoiceChoice(groups, "nat:unsure-tts:a"), null);

  // no cloning model: every own voice says why it cannot be used
  const noClone = voiceChoices(tts.filter((r) => r.capabilities.voice_cloning !== "reference_audio"), voices);
  for (const it of noClone[0].items) {
    assert.match(it.note, /no speaking model clones/);
    assert.deepEqual(it.modelIds, []);
  }
  // no voices at all: no empty "Your voices" group
  assert.equal(voiceChoices(tts, []).some((g) => g.label === "Your voices"), false);
  assert.deepEqual(voiceChoices(null, null), []);

  assert.equal(findVoiceChoice(groups, "nat:qwen-custom:serena").nativeVoice, "serena");
  assert.equal(findVoiceChoice(groups, "lib:nobody"), null);
  ok("voiceChoices groups own voices, honoured built-in voices and one-voice models, with notes");
}

/* --------------- sets: voice key <-> spec round trip -------------------- */
{
  const specs = [
    { tts_profile: "chatterbox", voice: { kind: "library", voice_id: "anna", native_voice: "" } },
    { tts_profile: "qwen-custom", voice: { kind: "native", voice_id: "", native_voice: "ryan" } },
    { tts_profile: "plain-tts", voice: { kind: "none", voice_id: "", native_voice: "" } },
  ];
  assert.equal(voiceKeyFromSpec(specs[0]), "lib:anna");
  assert.equal(voiceKeyFromSpec(specs[1]), "nat:qwen-custom:ryan");
  assert.equal(voiceKeyFromSpec(specs[2]), "own:plain-tts");
  for (const spec of specs) {
    assert.deepEqual(specVoiceFromKey(voiceKeyFromSpec(spec)), spec.voice, `round trip ${spec.voice.kind}`);
  }
  // a native voice name containing a colon survives (split on the first one)
  assert.equal(specVoiceFromKey("nat:m:a:b").native_voice, "a:b");
  assert.equal(voiceKeyFromSpec({}), "own:");
  assert.deepEqual(specVoiceFromKey(""), { kind: "none", voice_id: "", native_voice: "" });
  ok("voice picker keys round-trip with the set's spec.voice");
}

/* --------------- sets: language ----------------------------------------- */
{
  assert.equal(languageLabel("de"), "German");
  assert.equal(languageLabel("DE"), "German", "case-insensitive");
  assert.equal(languageLabel("de-AT"), "German (de-AT)", "a region is shown with the code");
  assert.equal(languageLabel("xx"), "xx", "an unknown code shows as itself");
  assert.equal(languageLabel(""), "—");
  assert.equal(languageLabel(null), "—");
  assert.ok(LANGUAGES.every(([c, n]) => /^[a-z]{2}$/.test(c) && n.length > 2));
  assert.equal(new Set(LANGUAGES.map(([c]) => c)).size, LANGUAGES.length, "codes are unique");
  assert.deepEqual(LANGUAGE_FORMAT_CHOICES.map((c) => c.value), ["code", "name", "none"]);
  assert.ok(MODEL_FIELD_HELP.language_format);
  assert.equal("language" in MODEL_FIELD_HELP, false, "models no longer have a language");
  assert.equal("id" in MODEL_FIELD_HELP, false, "model ids are not shown");
  ok("language labels (incl. region) and the model's language format choices");
}

/* --------------- sets: what is wrong, in one sentence ------------------- */
{
  assert.equal(setIssueText([]), "");
  assert.equal(setIssueText(null), "");
  assert.match(setIssueText([{ code: "missing" }]), /not chosen/);
  assert.match(setIssueText([{ code: "transcript_unconfirmed" }, { code: "missing" }]), /not confirmed/, "first issue wins");
  assert.match(setIssueText([{ code: "unknown_profile" }]), /no longer exists/);
  assert.match(setIssueText([{ code: "voice_incompatible" }]), /cannot clone/);
  assert.equal(setIssueText([{ code: "new_code", message: "Server says why" }]), "Server says why", "unknown code: server message");
  assert.equal(setIssueText([{ code: "new_code" }]), "new_code", "no message: the code");
  for (const text of [setIssueText([{ code: "missing_reference" }]), setIssueText([{ code: "reference_too_long" }])]) {
    assert.ok(text && !/_/.test(text), `plain words: ${text}`);
  }
  ok("setIssueText names the first blocking issue in plain words");
}

/* --------------- sets: saved voice label from spec ---------------------- */
{
  const find = (id) => (id === "anna" ? { id: "anna", display_name: "Anna" } : null);
  assert.equal(activeVoiceLabel({ spec: { voice: { kind: "library", voice_id: "anna" } } }, find), "Anna");
  assert.equal(activeVoiceLabel({ spec: { voice: { kind: "library", voice_id: "gone" } } }, find), "gone",
    "a voice not in the library falls back to its id");
  assert.equal(activeVoiceLabel({ spec: { voice: { kind: "library", voice_id: "anna" } } }), "anna", "no lookup");
  assert.equal(activeVoiceLabel({ spec: { voice: { kind: "native", native_voice: "ryan" } } }, find), "ryan");
  assert.equal(activeVoiceLabel({ spec: { voice: { kind: "none" } } }, find), "the model's own voice");
  assert.equal(activeVoiceLabel({}, find), "the model's own voice");
  // the old draft/active fields are not read any more
  assert.equal(activeVoiceLabel({ active: { spec: { voice: { kind: "native", native_voice: "x" } } } }, find),
    "the model's own voice");
  assert.equal(testConsoleOptionLabel({ id: "s", display_name: "Studio" }, "Anna"), "Studio · voice Anna");
  assert.equal(testConsoleOptionLabel({ id: "s" }, ""), "s");
  // a "Try it" ticket (unsaved values) vs the saved set must stay distinguishable
  assert.equal(testConsoleRouteKind({ ticket: "profile-test:abc" }), "draft-test");
  assert.equal(testConsoleRouteKind({ ticket: "", profileId: "studio" }), "active");
  assert.equal(testConsoleRouteKind({ ticket: "", profileId: "" }), "none");
  assert.equal(testConsoleRouteKind(null), "none");
  ok("activeVoiceLabel reads the set's spec; test console option label and route kind");
}

console.log(`\nFORMAT TESTS PASSED (${passed})`);
