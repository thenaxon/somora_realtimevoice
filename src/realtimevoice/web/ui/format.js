/* Pure presentation mappings shared by the views. Kept free of DOM so
 * tests/ui_format.mjs can exercise them under node. */

/* ---------------------------------------------------------- time helpers */

export function fmtDuration(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  const total = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function fmtMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtClock(epochSeconds) {
  if (!epochSeconds) return "";
  const d = new Date(epochSeconds * 1000);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/* -------------------------------------------------------- sample metrics */

export function sampleMetaLine(sample) {
  const parts = [];
  parts.push(fmtDuration(sample.duration_ms));
  if (sample.sample_rate) parts.push(`${Math.round(sample.sample_rate / 1000)} kHz`);
  const peak = sample.metrics && sample.metrics.peak_dbfs;
  if (peak != null && Number.isFinite(peak)) parts.push(`${Math.round(peak)} dBFS peak`);
  if (sample.source === "upload") parts.push("uploaded");
  return parts.join(" · ");
}

export function sampleTitle(sample) {
  // F10: an excerpt must be recognisable next to its original. Its `source`
  // is inherited from the original (e.g. "mic"), so without this it would be
  // titled "Recorded HH:MM" like any other take.
  const d = sample && sample.derived_from;
  if (d && d.sample_id && Number.isFinite(Number(d.start_ms)) && Number.isFinite(Number(d.end_ms))) {
    return `Excerpt ${(Number(d.start_ms) / 1000).toFixed(1)}–${(Number(d.end_ms) / 1000).toFixed(1)} s`;
  }
  if (sample.source === "upload" && sample.original_filename) return sample.original_filename;
  return `Recorded ${new Date((sample.created_at || 0) * 1000).toTimeString().slice(0, 5)}`;
}

export function transcriptQuote(sample) {
  const text = (sample.transcript || "").trim();
  if (!text) return "";
  const first = text.length > 72 ? text.slice(0, 72).trimEnd() + "…" : text;
  return `“${first}”`;
}

export function clippingPct(sample) {
  const v = sample.metrics && sample.metrics.clipping_pct;
  return Number.isFinite(v) ? v : 0;
}

/* --------------------------------------------- transcript state machine */

/** Chips for a sample's transcript block (voice-sample-flow v1.2 §4/§5). */
export function transcriptChips(sample) {
  const chips = [];
  const confirmed = !!sample.transcript_confirmed;
  const hasText = !!(sample.transcript || "").trim();
  if (hasText && sample.transcript_source === "stt") chips.push("auto");
  if (hasText && sample.transcript_source === "manual") chips.push("edit");
  chips.push(confirmed ? "confirmed" : "not_confirmed");
  return chips;
}

export const TRANSCRIPTION_STATES = ["none", "running", "done", "failed"];

/** Derive the visible step state from a stored sample.
 *
 * A non-empty transcript always counts as "done" — even when an earlier STT
 * attempt failed (e.g. no_speech_detected). Otherwise a user who typed the
 * transcript by hand would be sent back to the failure block and could never
 * reach the Confirm step (P1-1 dead end). */
export function transcriptionState(sample) {
  const t = sample.transcription || {};
  if ((sample.transcript || "").trim()) return "done";
  if (t.state === "failed") return "failed";
  return "none";
}

/* ------------------------------------------------------- API error text */

/* Contract §8a error_code → sentence. Every failure keeps "Your audio is
   saved" and two ways out; the error bar composes the full pattern. */
export const TRANSCRIPTION_ERROR_TEXT = {
  backend_unreachable: "The speech recognition backend did not answer.",
  backend_timeout: "Speech recognition took too long to answer.",
  stt_failed: "Speech recognition could not process this audio.",
  no_speech_detected: "No speech was detected in this recording. Record again, a bit closer to the microphone.",
};

export function transcriptionErrorText(code) {
  return TRANSCRIPTION_ERROR_TEXT[code] || "Speech recognition could not process this audio.";
}

export const UPLOAD_ERROR_TEXT = {
  sample_too_short: "Samples need at least 1 second.",
  sample_too_long: "Samples can be at most 30 seconds.",
  unsupported_format: "Only WAV files work here. Convert the file and try again.",
  too_large: "The limit is 10 MB.",
};

export function uploadErrorText(code) {
  return UPLOAD_ERROR_TEXT[code] || "This file could not be used.";
}

/* Activation probe error_code → operator sentence. timeout_ms comes from
   the activate response per role (never hardcode it). */
export const PROBE_ERROR_TEXT = {
  backend_timeout: (p) => `The backend did not answer within ${Math.round((p.timeout_ms || 0) / 1000)} seconds.`,
  backend_unreachable: () => "The backend did not answer.",
  probe_failed: () => "The backend answered with an error.",
  unknown_profile: () => "The referenced backend profile no longer exists.",
  no_base_url: () => "The referenced backend profile has no base URL.",
  model_not_listed: (p) =>
    `The backend answered, but model ${p && p.model ? p.model : "the configured one"} is not in its model list.`,
};

export function probeErrorText(code, probe) {
  const fn = PROBE_ERROR_TEXT[code];
  return fn ? fn(probe || {}) : "The readiness check failed.";
}

/* ------------------------------------------------------------- status */

/* ------------------------------------------ activation state (R2 / F12) */

/* A user report: on an ALREADY ACTIVE profile, "Save draft" (a plain PUT)
 * saves the draft but does NOT create a new revision. Conversations keep using
 * the active revision until an explicit activation. The editor must therefore
 * distinguish THREE states that the old single "unsaved changes" wording
 * blurred:
 *   - "dirty"               on-screen edits are not saved at all;
 *   - "saved-not-activated" the SAVED draft differs from the active revision
 *                           (a PUT does not raise the revision);
 *   - "in-sync"             the saved draft equals the active revision.
 * A never-activated profile is its own state. */

/**
 * The voice of the ACTIVE revision, from the frozen profile view.
 * `findVoice(id)` resolves a library voice's display name; may return null.
 * Returns "" when there is no active revision. `none` reads "engine default".
 */
export function activeVoiceLabel(profile, findVoice) {
  const spec = (profile && profile.spec) || {};
  const voice = spec.voice || {};
  if (voice.kind === "library") {
    const v = typeof findVoice === "function" ? findVoice(voice.voice_id) : null;
    return (v && (v.display_name || v.id)) || voice.voice_id || "";
  }
  if (voice.kind === "native") return voice.native_voice || "";
  return "the model's own voice";
}

/* ------------------------------------------------------- Test page labels */

/** Picker option: name + the saved set's voice. */
export function testConsoleOptionLabel(profile, voiceLabel) {
  const name = profile.display_name || profile.id;
  return voiceLabel ? `${name} · voice ${voiceLabel}` : name;
}

/** "active" | "draft-test" | "none" — what the Console is currently set to. */
export function testConsoleRouteKind(state) {
  const s = state || {};
  if (s.ticket) return "draft-test";
  if (s.profileId) return "active";
  return "none";
}

/* -------------------------------------------------- redacted drift diff */

/* -------------------------------------------- test console route mapping */

/**
 * Resolve the "Running now" panel EXCLUSIVELY from
 * session.updated.session.x_realtimevoice.route (v1.1 §7, README §3).
 * Returns null while pending.
 */
export function routeFromSession(session) {
  if (!session) return null;
  // contract §6 specifies session.x_realtimevoice; the backend in this
  // worktree still emits session.realtimevoice (API gap, see lane report).
  // Accept both so the console works against the contract and the current
  // backend; prefer the contract field.
  const ext = session.x_realtimevoice || session.realtimevoice;
  if (!ext || !ext.route) return null;
  const r = ext.route;
  // legacy voice-route state uses profile_id/actual_voice; the contract uses
  // profile/voice. Normalise without inventing values.
  const route = {
    mode: r.mode || "profile",
    requested: r.requested || "",
    profile: r.profile || r.profile_id || "",
    revision: r.revision != null ? r.revision : null,
    stt_model: r.stt_model || "",
    llm_model: r.llm_model || "",
    tts_model: r.tts_model || "",
    voice: r.voice || (r.actual_voice ? { kind: "native", voice_id: r.actual_voice } : { kind: "none" }),
    // C3: the honest display label (additive; falls back to actual_voice)
    display_voice: r.display_voice || r.actual_voice || "",
    language: r.language || "",
  };
  return { route, warnings: ext.warnings || [] };
}

export const ROUTE_WARNING_TEXT = {
  language_overridden_by_profile: (w) => {
    const asked = w && w.requested_language ? w.requested_language : "another language";
    return `The app asked for ${asked}; the set has a different language and wins. The conversation continues in the set's language.`;
  },
};

export function routeWarningText(warning) {
  if (typeof warning === "string") {
    const fn = ROUTE_WARNING_TEXT[warning];
    return fn ? fn(null) : warning;
  }
  if (warning && warning.code) {
    const fn = ROUTE_WARNING_TEXT[warning.code];
    if (fn) return fn(warning);
    return warning.code;
  }
  return String(warning);
}

/* --------------------------------------- client copy block (contract §8) */

/* ------------------------------------------------------- misc formatting */

export function modelTitle(profile) {
  return (profile && (profile.display_name || profile.model || profile.id)) || "—";
}

export function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ------------------------------------------- model thinking option (F13) */

/* Reasoning models may spend part of the token budget "thinking" before the
 * answer. For oMLX/vLLM-style backends that is switched with
 * `chat_template_kwargs.enable_thinking`; the adapter copies `extra` straight
 * into the request payload. The UI presents this as a first-class option but
 * stores ONLY the mapping inside `extra` — "backend default" means the key is
 * absent, so an existing row is never changed and no model is switched. The
 * voice never speaks reasoning either way; the adapter enforces that. */

export const THINKING_MODES = ["default", "off", "on"];

/** Read the stored option from a backend row's `extra`. */
export function thinkingModeFromExtra(extra) {
  const ct = extra && extra.chat_template_kwargs;
  const value = ct && ct.enable_thinking;
  if (value === false) return "off";
  if (value === true) return "on";
  return "default";
}

/** Apply the option to an `extra` object (returns a new object).
 *
 *  "default" must REMOVE a stored `enable_thinking`. The server merges `extra`
 *  recursively and only a JSON null removes a key — an omitted key keeps the
 *  stored value — so when the key is present we write an explicit `null`; an
 *  `extra` that never had it is returned unchanged (no empty object invented).
 *  Sibling `chat_template_kwargs` (e.g. top_k) are preserved. */
export function extraWithThinking(extra, mode) {
  const out = { ...(extra || {}) };
  const ct = { ...(out.chat_template_kwargs || {}) };
  if (mode === "off") ct.enable_thinking = false;
  else if (mode === "on") ct.enable_thinking = true;
  else if ("enable_thinking" in ct) ct.enable_thinking = null;
  else return out;
  out.chat_template_kwargs = ct;
  return out;
}

/* ------------------------------------------------------------ id validation */

/** Stable id pattern shared by the voice/profile/backend create dialogs. */
export const ID_PATTERN = /^[a-z0-9][a-z0-9-]{1,62}$/;

export function idError() {
  return "The id must match [a-z0-9][a-z0-9-]{1,62}.";
}

/* ------------------------------------------------ add-backend flow (CP3) */

/* A user report: he could not see HOW to add a backend, especially where to
 * choose STT/LLM/TTS. The add flow is now role-first, with full labels and a
 * one-line explanation each. These helpers are pure so node tests can pin the
 * role-dependent field sets and the id slug/collision behaviour. */

export const BACKEND_ROLE_CHOICES = [
  {
    role: "stt",
    label: "Speech recognition (STT)",
    help: "Turns speech into text — the ears of the conversation.",
  },
  {
    role: "llm",
    label: "Reasoning (LLM)",
    help: "Reads the recognised text and writes the answer.",
  },
  {
    role: "tts",
    label: "Speech engine (TTS)",
    help: "Speaks the answer — the voice of the conversation.",
  },
];

/* Which connection fields apply to a role. `base` fields (name, id, endpoint,
 * key, model, timeout) are always shown; the rest are role-dependent. */
export function backendRoleFields(role) {
  const base = ["name", "id", "base_url", "api_key", "model", "timeout"];
  if (role === "stt") return [...base, "language"];
  if (role === "llm") return [...base, "thinking"];
  if (role === "tts") {
    return [...base, "language", "voice", "voices", "voice_cloning", "max_reference_ms", "native_trust"];
  }
  return base;
}

/* The three honest trust values for an engine's native voice labels. */
export const NATIVE_VOICE_LABEL_MODES = ["unknown", "honoured", "ignored"];

/* ------------------------------------------- endpoint / key (C3) -------- */

/* Saving a backend with a changed endpoint and a blank key must not silently
 * keep sending the stored secret to the new host. The server refuses with
 * `key_required_for_changed_endpoint` unless the edit form confirms. These
 * pure helpers carry the host shown to the operator and the exact ask. */

/** The host (+ optional port) an operator would actually send a key to. */
export function endpointHost(url) {
  try {
    const u = new URL(String(url || ""));
    return u.port ? `${u.hostname}:${u.port}` : u.hostname;
  } catch {
    return String(url || "").trim();
  }
}

/** The confirm-dialog message for a changed endpoint with a blank key. */
export function changedEndpointKeyPrompt(newUrl) {
  const host = endpointHost(newUrl) || "the new endpoint";
  return (
    `The endpoint changed. Enter the API key for ${host}, or keep sending the ` +
    "existing key to the new host."
  );
}

/** Confirm-dialog details explaining the two options plainly. */
export function changedEndpointKeyDetail() {
  return "Keeping the existing key sends the stored secret to the new endpoint. Only continue if you trust it.";
}

/* ---------------------------------------------- profile language (CP4) */

/* A user report: the profile editor showed "de" like a value (the input's
 * placeholder) and a one-line hint that did not explain precedence, defaults or
 * the STT/LLM/TTS effect. These pure helpers carry the honest, consistent
 * explanation and a placeholder that cannot be mistaken for a value. */

function suggestId(base, existing, baseline) {
  const stem = base.replace(/[._-]?\d+$/, "") || base || "backend";
  for (let n = 2; n < 1000; n++) {
    const candidate = `${stem}-${n}`.slice(0, 64);
    if (!existing.has(candidate) && !baseline.has(candidate)) return candidate;
  }
  return "";
}

/* ------------------------------------------------------------ audio levels */

/** Floor of the displayed level scale, in dBFS. Below this a signal counts as
 *  silence for DISPLAY purposes. -60 dBFS is far below usable speech, so real
 *  silence still renders as nothing. */
export const LEVEL_FLOOR_DBFS = -60;

/**
 * Map a linear audio amplitude (0..1) to a 0..1 DISPLAY level on a decibel
 * scale.
 *
 * Why this exists (F8): human loudness perception and audio metering are
 * logarithmic, but the UI used the raw linear amplitude as a bar height.
 * Normal speech peaking around -20 dBFS is 0.1 linear — a bar at 10 % that
 * looks flat — while only near-clipping speech moved the meter. Measured on a
 * real recording: peak -16.95 dBFS lit 2 of 15 bars.
 *
 * On this scale -20 dBFS -> 0.67, -40 dBFS -> 0.33, -60 dBFS and below -> 0.
 * Digital silence stays 0: the floor clamps, it does not lift quiet audio off
 * the ground. The underlying samples are never touched — this is display only.
 */
export function levelToDisplay(amplitude, floorDbfs = LEVEL_FLOOR_DBFS) {
  const a = Number(amplitude);
  if (!Number.isFinite(a) || a <= 0) return 0;
  const dbfs = 20 * Math.log10(Math.min(1, a));
  if (dbfs <= floorDbfs) return 0;
  return Math.min(1, (dbfs - floorDbfs) / -floorDbfs);
}

/** dBFS of a linear amplitude, for labels. Returns null for digital silence. */
export function amplitudeToDbfs(amplitude) {
  const a = Number(amplitude);
  if (!Number.isFinite(a) || a <= 0) return null;
  return 20 * Math.log10(Math.min(1, a));
}

/* ------------------------------------------------ reference length (F10) */

/** Seconds with one decimal, e.g. 12200 -> "12.2 s". */
export function fmtSeconds(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n <= 0) return "—";
  return `${(n / 1000).toFixed(1)} s`;
}

/**
 * Does a reference sample fit an engine's reference limit?
 *
 * `maxMs` 0/absent = the engine advertises no limit -> always fits.
 * `sampleMs` 0/absent = duration unknown on the client (e.g. a migrated
 * sample) -> reported as "unknown", never as fitting: the server measures the
 * real file and has the final word.
 *
 * @returns {{state: "fits"|"too_long"|"unknown"|"no_limit", sampleMs:number, maxMs:number, overMs:number}}
 */
export function referenceFit(sampleMs, maxMs) {
  const s = Math.max(0, Math.round(Number(sampleMs) || 0));
  const m = Math.max(0, Math.round(Number(maxMs) || 0));
  if (!m) return { state: "no_limit", sampleMs: s, maxMs: 0, overMs: 0 };
  if (!s) return { state: "unknown", sampleMs: 0, maxMs: m, overMs: 0 };
  if (s <= m) return { state: "fits", sampleMs: s, maxMs: m, overMs: 0 };
  return { state: "too_long", sampleMs: s, maxMs: m, overMs: s - m };
}

/** Plain-language explanation of a too-long reference (no jargon, no codes). */
export function referenceTooLongText(sampleMs, maxMs) {
  return (
    `The reference recording is ${fmtSeconds(sampleMs)}, but this speech engine uses at most ` +
    `${fmtSeconds(maxMs)} of a reference and would silently ignore the rest. ` +
    `Pick a ${fmtSeconds(maxMs)} excerpt instead — your original recording stays unchanged.`
  );
}

/** Route to the excerpt tool for a sample, pre-limited to an engine maximum. */
export function excerptRoute(voiceId, sampleId, maxMs, fromProfile) {
  const q = new URLSearchParams({ excerpt: sampleId });
  if (maxMs) q.set("max", String(Math.round(maxMs)));
  if (fromProfile) q.set("from", fromProfile);
  return `#/voices/${encodeURIComponent(voiceId)}?${q.toString()}`;
}

/** The reference_too_long issue from an ApiError / issues array, or null. */
export function referenceTooLongIssue(errOrIssues) {
  const list = Array.isArray(errOrIssues) ? errOrIssues : (errOrIssues && errOrIssues.issues) || [];
  return list.find((i) => i && i.code === "reference_too_long") || null;
}

/* ------------------------------------------- engine clone constraint (CP1) */

/** One-line constraint text for a cloning engine (design §6). Uses the
 *  honestly-advertised capability values; "unknown" never invents a limit. */
export function engineCloneConstraint(engine) {
  const e = engine || {};
  const maxMs = Number(e.max_reference_ms) || 0;
  if (e.reference_text_required) {
    return "uses the whole reference; a matching transcript is required";
  }
  if (maxMs > 0) {
    return `uses up to ${fmtSeconds(maxMs)} of a reference (longer audio is silently ignored)`;
  }
  return "uses the whole reference";
}

/* ----------------------------------------- engine model labels (CP1 fix R2) */

/* Engine options must be named after the ENGINE (a readable form of its model
 * id), never after a stored row's display_name: rows are named after speakers
 * ("Chatterbox / Clara") or migration artifacts ("Baseline STT", "... snapshot")
 * and that made the engine look like a different model. The functions below
 * derive a model-first label; a stored row's id is only ever SECONDARY info. */

/* Version/build noise that carries no engine identity. Dropping these from the
 * readable form is the whole "small transformation" — no per-model table. */
const ENGINE_MODEL_NOISE = new Set([
  "mlx", "community", "fp16", "fp32", "bf16", "it", "instruct", "default", "latest",
]);

function _isEngineAcronym(tok) {
  return /^[A-Z]{2,}$/.test(tok);
}
function _isEngineSize(tok) {
  return /^\d+(?:\.\d+)?[bB]$/.test(tok);
}
function _isEngineVersion(tok) {
  return /^v\d+$/.test(tok);
}
function _isEngineDateCode(tok) {
  return /^\d{4}$/.test(tok);
}
function _isEngineBit(tok) {
  return /^\d+bit$/i.test(tok);
}

/**
 * Readable engine name derived from a model id.
 *
 * The transformation is generic, not a per-model display table:
 *  - strip a vendor `foo--` prefix and version/build noise (`mlx`, `fp16`, a
 *    MoE active-params tag `a4b`, an instruction tag `it`, a `12Hz` frame rate);
 *  - a trailing build date (`2603` and anything after it) is dropped;
 *  - for LLM rows a quantization suffix (`4bit`) is noise (the size token
 *    `26B`/`14B` already names the model), while STT/TTS keep `8bit` because it
 *    is part of those published model names;
 *  - sizes keep an uppercase `B`; acronyms stay uppercase; the family token is
 *    capitalised.
 *
 * `role` is used only to decide whether a `Nbit` suffix is part of the name.
 * When the resulting name carries no number (e.g. plain "Chatterbox") the raw
 * model id is appended in parentheses so the identity is never lost.
 */
export function engineModelLabel(model, role = "") {
  let s = String(model == null ? "" : model).trim();
  if (!s) return "Unknown engine";
  // never surface an endpoint as a "model"
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) return "Unknown engine";
  // strip an HF-style vendor prefix
  if (s.includes("--")) s = s.split("--").pop();
  const raw = s;
  let tokens = s.replace(/[_\s]+/g, "-").split("-").filter(Boolean);
  const isLlm = String(role || "").toLowerCase() === "llm";

  // a build date code and everything after it is build metadata, not identity
  const dateAt = tokens.findIndex(_isEngineDateCode);
  if (dateAt > 0) tokens = tokens.slice(0, dateAt);

  const kept = tokens.filter((tok) => {
    const low = tok.toLowerCase();
    if (ENGINE_MODEL_NOISE.has(low)) return false;
    if (/^a\d+b$/i.test(tok)) return false;      // MoE active params (a4b)
    if (/^\d+hz$/i.test(tok)) return false;      // frame rate (12Hz)
    if (isLlm && _isEngineBit(tok)) return false; // LLM quant suffix is redundant
    return true;
  });
  if (!kept.length) return raw;

  const fixed = kept.map((tok) => (_isEngineSize(tok) ? tok.slice(0, -1) + "B" : tok));
  const rest = fixed.slice(1);
  const family = fixed[0].charAt(0).toUpperCase() + fixed[0].slice(1);

  let out = family;
  rest.forEach((tok, i) => {
    const prev = i === 0 ? family : rest[i - 1];
    // Qwen3-ASR / Qwen3-TTS stay hyphenated; v3 keeps its hyphen (large-v3-turbo)
    const hyphens = (i === 0 && _isEngineAcronym(tok)) || _isEngineVersion(tok) || _isEngineVersion(prev);
    out += (hyphens ? "-" : " ") + tok;
  });

  // a digitless friendly name (e.g. "Chatterbox") must carry the exact id
  if (!/\d/.test(out)) out += ` (${raw})`;
  return out;
}

/** Label for one engine option in the cloning screen: model + constraint, no
 *  internal host/port, no speaker/baseline row name. Installed variant ids are
 *  secondary information, clearly marked as `stored as …`. */
export function cloneEngineLabel(engine) {
  const e = engine || {};
  const role = e.role || "tts";
  const base = engineModelLabel(e.model || e.id, role);
  const parts = [base, engineCloneConstraint(e)];
  const variants = (e.variants || []).filter(Boolean);
  if (variants.length > 1) parts.push(`${variants.length} installed variants`);
  if (e.embeds_reference) parts.push("legacy profile with built-in reference — your sample overrides it");
  if (e.id) parts.push(`stored as ${e.id}`);
  return parts.join(" · ");
}

/** What a speaking model can do with voices, for its label and badge:
 *  "clones" (from a recording), "builtin" (its own named voices) or
 *  "fixed" (one configured voice). */
export function voiceAbility(group, caps) {
  const cloning = group ? group.voice_cloning : caps && caps.voice_cloning;
  if (cloning === "reference_audio") return "clones";
  const list = group ? group.native_voices || [] : (caps && caps.voices) || [];
  return list.length ? "builtin" : "fixed";
}

export const VOICE_ABILITY_LABEL = {
  clones: "clones voices",
  builtin: "built-in voices",
  fixed: "one voice",
};

/* ------------------------------------------------ model form help (Models) */

/** One plain sentence per field of the model form: what it does for you,
 *  the same for listening, thinking and speaking models. The single place
 *  these texts live, so the form and its tests cannot drift apart. */
export const MODEL_FIELD_HELP = {
  name: "What this model is called on this page and in sets, e.g. “Whisper large (studio)”.",
  base_url: "The address of the server that runs the model, usually ending in /v1. Any server that speaks the OpenAI API works: a local one or a cloud service.",
  api_key: "The key the server asks for; leave it empty if it needs none. It stays on this server and is never shown again.",
  api_key_edit: "Empty keeps the stored key. Type a new one to replace it.",
  model: "The model's name as that server knows it, e.g. whisper-large-v3-turbo. “Check connection” lists the names the server offers.",
  language_format: "Each call has the language of its set. This says how this model wants to be told it: most take a code (de), some a name (German; Qwen3-TTS does), some none at all.",
  thinking: "Thinking adds delay and uses the token budget. The voice never speaks it either way.",
  voice_mode: "What a set can choose for this model: one of your recorded voices, one of the model's own voices, or nothing.",
  voices: "The names of its own voices as the server knows them, separated by commas, e.g. ryan, serena. Sets offer them by name.",
  voice: {
    builtin: "The voice it uses when a set does not choose one of the voices above.",
    fixed: "The voice name sent to the server. Leave it as it is unless the server needs a particular name.",
    clones: "A name sent along with the recording. Most cloning servers ignore it.",
    unknown: "The voice name sent to the server when a set does not choose one.",
  },
  max_reference_s: "Some models use only the start of a recording and drop the rest (Chatterbox keeps 10 s). Enter that limit and the Voices page warns about longer recordings; 0 means no limit.",
  native_trust: "Whether a different voice name really gives a different voice. Only names the server honours are offered in sets as separate voices.",
  timeout: "How long to wait for the server before giving up, in seconds. Raise it for a slow model or one that needs time to load.",
  extra: {
    stt: "Fields sent with every request, for settings this form does not have, e.g. {\"temperature\": 0}. They override the fields above.",
    llm: "Fields sent with every request, for settings this form does not have, e.g. {\"temperature\": 0.7, \"max_tokens\": 300}. They override the fields above.",
    tts: "Fields sent with every request, for settings this form does not have, e.g. {\"speed\": 1.0}. They override the fields above.",
  },
  reference_fields:
    "The request fields that carry the voice recording and its text. Most cloning servers use ref_audio and ref_text; set them to what yours expects.",
  reference_encoding:
    "How the recording is put into its field: plain base64, or a data URL (data:audio/wav;base64,…) that some servers require.",
  audio_format:
    "What the server answers with. WAV works at any rate; raw PCM must be 16-bit mono at the rate below; MP3 is decoded on this server (needs ffmpeg).",
  start_buffer:
    "How much of each sentence is collected before it is spoken. Raise it (e.g. 700) for a model that sends its sound in bursts barely as fast as it is spoken, so words do not break up; 0 speaks at once.",
  pcm_sample_rate: "The rate of the raw PCM the server sends, e.g. 24000 or 44100.",
  ref_clear: "Older entries carried a voice recording of their own. Voices now live under Voices; tick this to remove the one stored in this model on save.",
};

/** The four answers to "what does it do with voices", as the form asks it. */
export const VOICE_MODE_CHOICES = [
  { value: "clones", text: "Clones a voice from a recording" },
  { value: "builtin", text: "Has its own voices to choose from" },
  { value: "fixed", text: "Speaks with one voice" },
  { value: "unknown", text: "Not known yet" },
];

/** The suggestion for "voice names are reliable" when the answer changes
 *  to this one: its own voices are meant to be chosen by name, one voice
 *  ignores the name. */
export const VOICE_MODE_TRUST = { builtin: "honoured", fixed: "ignored" };

/** The form's answer for stored capabilities. */
export function voiceModeFromCaps(caps) {
  const c = caps || {};
  if (c.voice_cloning === "reference_audio") return "clones";
  // not verified stays not verified, even with a list of voice names
  if (c.voice_cloning === "unknown" || !c.voice_cloning) return "unknown";
  return (c.voices || []).length ? "builtin" : "fixed";
}

/** Capabilities for a chosen answer. Whether voice names are reliable is
 *  the form's own field (VOICE_MODE_TRUST suggests it when the answer
 *  changes), never overridden here. */
export function capsForVoiceMode(mode, caps) {
  const out = { ...(caps || {}) };
  out.voice_cloning = mode === "clones" ? "reference_audio" : mode === "unknown" ? "unknown" : "none";
  if (mode === "fixed") out.voices = [];
  if (mode !== "clones") out.max_reference_ms = 0;
  return out;
}

/* ------------------------------------------------------- sets: language */

/** The languages a set can be set to: code and English name. A set's language
 *  is sent to each model in the form that model wants ("de" or "German"). */
export const LANGUAGES = [
  ["de", "German"], ["en", "English"], ["fr", "French"], ["es", "Spanish"],
  ["it", "Italian"], ["nl", "Dutch"], ["pl", "Polish"], ["pt", "Portuguese"],
  ["cs", "Czech"], ["da", "Danish"], ["sv", "Swedish"], ["no", "Norwegian"],
  ["fi", "Finnish"], ["hu", "Hungarian"], ["ro", "Romanian"], ["el", "Greek"],
  ["ru", "Russian"], ["uk", "Ukrainian"], ["tr", "Turkish"], ["ar", "Arabic"],
  ["he", "Hebrew"], ["hi", "Hindi"], ["ja", "Japanese"], ["ko", "Korean"],
  ["zh", "Chinese"], ["th", "Thai"], ["vi", "Vietnamese"], ["id", "Indonesian"],
];

export function languageLabel(code) {
  const c = String(code || "");
  const hit = LANGUAGES.find(([k]) => k === c.split("-")[0].toLowerCase());
  return hit ? (c.includes("-") ? `${hit[1]} (${c})` : hit[1]) : c || "—";
}

/** How a model is told the language of a call. */
export const LANGUAGE_FORMAT_CHOICES = [
  { value: "code", label: "As a code (de)" },
  { value: "name", label: "As a name (German)" },
  { value: "none", label: "Not at all" },
];

/* ---------------------------------------------------- sets: the voice */

function capsOf(row) {
  return (row && row.capabilities) || {};
}

function rowName(row) {
  return (row && (row.display_name || row.model || row.id)) || "";
}

/**
 * What a set can speak as, grouped for one picker:
 *  - "Your voices": every recorded voice, spoken by a model that clones;
 *  - "<model>'s own voices": the voices built into a model (only where its
 *    voice names are reliable);
 *  - "Models with one voice": a model that only speaks with its own voice.
 * Each item lists the speaking models that can speak it (`modelIds`).
 */
export function voiceChoices(ttsRows, voices) {
  const rows = (ttsRows || []).slice().sort((a, b) => rowName(a).localeCompare(rowName(b)));
  const cloning = rows.filter((r) => capsOf(r).voice_cloning === "reference_audio").map((r) => r.id);
  const groups = [];
  const own = (voices || []).slice().sort((a, b) => (a.display_name || a.id).localeCompare(b.display_name || b.id));
  if (own.length) {
    groups.push({
      label: "Your voices",
      items: own.map((v) => {
        const ref = (v.samples || []).find((s) => s.id === v.reference_sample_id);
        let note = "";
        if (!cloning.length) note = "no speaking model clones voices";
        else if (!ref) note = "no recording yet";
        else if (!ref.transcript_confirmed) note = "text of the recording not confirmed";
        return {
          key: `lib:${v.id}`, label: v.display_name || v.id, kind: "library",
          voiceId: v.id, modelIds: cloning, note,
        };
      }),
    });
  }
  const single = [];
  for (const r of rows) {
    const caps = capsOf(r);
    if (caps.voice_cloning === "reference_audio") continue;
    // named voices only where a different name really gives a different voice
    const builtIn = caps.native_voice_labels === "honoured" ? (caps.voices || []).filter(Boolean) : [];
    if (builtIn.length) {
      groups.push({
        label: `${rowName(r)}: its own voices`,
        items: builtIn.map((name) => ({
          key: `nat:${r.id}:${name}`, label: name, kind: "native", nativeVoice: name, modelIds: [r.id], note: "",
        })),
      });
    } else {
      single.push({ key: `own:${r.id}`, label: rowName(r), kind: "none", modelIds: [r.id], note: "" });
    }
  }
  if (single.length) groups.push({ label: "Models with one voice", items: single });
  return groups;
}

/** The picker key of a set's saved voice. */
export function voiceKeyFromSpec(spec) {
  const v = (spec && spec.voice) || {};
  if (v.kind === "library") return `lib:${v.voice_id || ""}`;
  if (v.kind === "native") return `nat:${(spec && spec.tts_profile) || ""}:${v.native_voice || ""}`;
  return `own:${(spec && spec.tts_profile) || ""}`;
}

/** The set's `voice` for a picker key. */
export function specVoiceFromKey(key) {
  const k = String(key || "");
  if (k.startsWith("lib:")) return { kind: "library", voice_id: k.slice(4), native_voice: "" };
  if (k.startsWith("nat:")) {
    const rest = k.slice(4);
    const i = rest.indexOf(":");
    return { kind: "native", voice_id: "", native_voice: i >= 0 ? rest.slice(i + 1) : "" };
  }
  return { kind: "none", voice_id: "", native_voice: "" };
}

/** The voice item for a key in voiceChoices() output, or null. */
export function findVoiceChoice(groups, key) {
  for (const g of groups || []) for (const it of g.items) if (it.key === key) return it;
  return null;
}

/** One sentence for what is wrong with a set (its first blocking issue), in
 *  plain words; "" when nothing is. */
export function setIssueText(issues) {
  const list = issues || [];
  if (!list.length) return "";
  const i = list[0];
  const text = {
    missing: "Something is not chosen yet",
    unknown_profile: "One of its models no longer exists",
    missing_reference: "The voice has no recording yet",
    transcript_missing: "The text of the voice's recording is empty",
    transcript_unconfirmed: "The text of the voice's recording is not confirmed yet",
    voice_incompatible: "The speaking model cannot clone a recorded voice",
    reference_too_long: "The voice's recording is too long for the speaking model",
    file_missing: "The voice's recording file is missing",
  }[i.code];
  return text || i.message || i.code || "";
}
