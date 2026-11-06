/* Test Console (screen 05).
 *
 * Profile picker (active profiles via the profiles API; drafts via a
 * profile-test ticket), Talk/Stop with explicit states
 * idle/connecting/route pending/listening/thinking/speaking, a "Running now"
 * panel populated ONLY from session.updated →
 * session.x_realtimevoice.route, yellow route warnings, the legacy_tts_only
 * notice, a two-column live transcript, and collapsed diagnostics.
 */

import { el, clear, waveform, errorBar } from "./dom.js";
import { routeWarningText, fmtMs, activeVoiceLabel, testConsoleOptionLabel, testConsoleRouteKind, languageLabel } from "./format.js";
import { RealtimeSession } from "./audio.js";
import { createMicPicker, watchDeviceChanges } from "./mic-devices.js";
import { profiles as profilesApi, voices as voicesApi } from "./api.js";

const STATE_TEXT = {
  idle: { label: "Idle", sub: "Press Talk to open a session.", color: "var(--text-1)" },
  connecting: { label: "Connecting", sub: "Opening the connection and the microphone.", color: "var(--text-1)" },
  "route-pending": { label: "Waiting for route", sub: "The profile is resolved as soon as the client picks one. This is expected, not a fault.", color: "var(--accent)" },
  listening: { label: "Listening", sub: "Speak now. A pause ends your turn.", color: "var(--accent)" },
  thinking: { label: "Thinking", sub: "The model is working on an answer.", color: "var(--warn)" },
  speaking: { label: "Speaking", sub: "The assistant is answering. Start talking any time to interrupt.", color: "var(--ready)" },
};

export async function renderTestConsole(inner, ctx, profileParam, ticketParam) {
  const requested = profileParam || "";
  const ticket = ticketParam || "";
  // C4: the visibilitychange handler is a per-mount resource. Declare it here
  // so the cleanup below (registered before `state` exists) can remove the
  // EXACT handler; otherwise every mount leaks one handler that keeps
  // refreshing a detached view.
  let visibilityHandler = null;
  // release the socket/mic (and the document listener) when the view is left
  if (ctx.setCleanup) ctx.setCleanup(() => {
    if (state.session) state.session.stop(true);
    if (state.profileId && ctx.markInUse) ctx.markInUse(state.profileId, false);
    if (countdownTimer) clearInterval(countdownTimer);
    if (visibilityHandler) {
      document.removeEventListener("visibilitychange", visibilityHandler);
      visibilityHandler = null;
    }
  });

  const state = {
    profileId: requested,
    ticket,
    use: ticket ? "editor" : "set",
    session: null,
    route: null,
    warnings: [],
    turns: [],
    events: [],
    metrics: {},
    stateName: "idle",
    micDeviceId: "",
    micLabel: "",
    micMissing: false,
  };

  // the sets list can change while the page is open (a set switched on or off
  // in another tab); keep it refreshable
  let active = ctx.profiles.filter((p) => p.enabled);
  let pickerRenderPending = false;
  /* A stable signature of what the picker actually shows: ids, active
   * save times, availability and voice labels. Used to avoid rebuilding
   * the <select> when nothing visible changed. */
  function activeSignature(list) {
    return (list || [])
      .map((p) => `${p.id}|${p.enabled ? 1 : 0}|${p.updated_at || 0}|${profileVoiceLabel(p)}`)
      .join("\n");
  }
  async function refreshProfiles() {
    const before = activeSignature(active);
    let changed = false;
    try {
      const list = await ctx.loadProfiles();
      active = (list || []).filter((p) => p.enabled);
      changed = activeSignature(active) !== before;
    } catch {
      /* keep the last known list */
    }
    // C3: never rebuild the options while the user is interacting with the
    // native select. Re-render on the next blur/change instead.
    if (changed) {
      if (document.activeElement === picker) pickerRenderPending = true;
      else renderPicker();
    }
    renderDraftBanner();
    renderStaleNote();
  }

  // the saved set's voice next to the picker option, from the sets list; the
  // RUNNING route is shown only from session.updated
  function findVoiceById(id) {
    return (ctx.voices || []).find((v) => v.id === id) || null;
  }
  function profileVoiceLabel(p) {
    return activeVoiceLabel(p, findVoiceById);
  }
  function profileById(id) {
    return active.find((p) => p.id === id) || null;
  }

  // ---- header
  const title = el("h1", { class: "h1", text: "Test" });
  const picker = el("select", { class: "inp", style: "max-width:320px", "aria-label": "Set" });
  picker.append(el("option", { value: "", text: "— choose a set —" }));
  for (const p of active) {
    const opt = el("option", { value: p.id, text: testConsoleOptionLabel(p, profileVoiceLabel(p)) });
    if (p.id === state.profileId) opt.selected = true;
    picker.append(opt);
  }

  const routeBox = el("div", { class: "route" });
  const talkState = el("div", { class: "talkstate" });
  const talkBtn = el("button", { class: "talkbtn", type: "button" });
  const pills = el("div", { class: "states" });
  const errorBox = el("div");

  function selectedModel() {
    if (state.ticket) return state.ticket; // "profile-test:<ticket>"
    if (state.profileId) return `profile:${state.profileId}`;
    return "";
  }

  function renderPicker() {
    clear(picker);
    picker.append(el("option", { value: "", text: "— choose a set —" }));
    for (const p of active) {
      const opt = el("option", { value: p.id, text: testConsoleOptionLabel(p, profileVoiceLabel(p)) });
      if (p.id === state.profileId && !state.ticket) opt.selected = true;
      picker.append(opt);
    }
    if (state.ticket) {
      // a test link carries what the set page showed (saved or not); the
      // picker's own change handler drops it when a set is chosen
      const p = profileById(state.profileId);
      const voice = p ? profileVoiceLabel(p) : "";
      const label = `${p ? p.display_name || p.id : state.profileId} · as on the set page${voice ? ` (saved voice ${voice})` : ""}`;
      picker.append(el("option", { value: state.profileId, text: label, selected: true }));
    }
  }

  picker.addEventListener("change", () => {
    state.profileId = picker.value;
    state.ticket = "";
    state.use = "set";
    if (state.session) state.session.stop(false);
    state.route = null;
    state.turns = [];
    state.events = [];
    renderRoute();
    renderTranscript();
    renderDiagnostics();
    renderState();
    renderDraftBanner();
    renderStaleNote();
  });

  // C3: do NOT refresh on focus. Focus fires when the user clicks the native
  // <select>; a rebuild ~100 ms later can close the popup or reset the
  // highlighted option. C1: do not refresh on any click path either. The only
  // remaining trigger is the page becoming visible, and its handler is removed
  // in the cleanup (C4).
  visibilityHandler = () => {
    if (document.visibilityState === "visible") refreshProfiles();
  };
  document.addEventListener("visibilitychange", visibilityHandler);

  // C3: if a background refresh changed the list while the picker was focused,
  // defer the rebuild until it loses focus (or the user picks an option).
  picker.addEventListener("blur", () => {
    if (pickerRenderPending) renderPicker();
  });

  // ---- route panel (ONLY from x_realtimevoice.route)
  function renderRoute() {
    clear(routeBox);
    const r = state.route;
    if (!r) {
      routeBox.append(
        el("div", { class: "routetop" }, [el("span", { class: "routelab", text: "Running now" })]),
        el("div", { class: "routepend" }, [
          el("span", { class: "mono", style: "color:var(--text-1);font-size:11px", text: "◌" }),
          el("div", { text: "Nothing running. Press Talk: the set's models and voice show here once the server confirms them." }),
        ]),
      );
      return;
    }
    routeBox.append(
      el("div", { class: "routetop" }, [
        el("span", { class: "routelab", text: "Running now" }),
        el("span", { class: "routeid mono", text: r.profile || r.requested || "—" }),
        el("span", { class: "routerev", text: r.language ? languageLabel(r.language) : "" }),
      ]),
    );
    const chain = el("div", { class: "routechain" });
    chain.append(rcs("Listens", r.stt_model));
    chain.append(el("span", { class: "rcarrow", "aria-hidden": "true", text: "→" }));
    chain.append(rcs("Thinks", r.llm_model));
    chain.append(el("span", { class: "rcarrow", "aria-hidden": "true", text: "→" }));
    const speaks = rcs("Speaks", r.tts_model);
    const voiceLabel = r.voice && r.voice.kind && r.voice.kind !== "none"
      ? (r.voice.kind === "library"
          ? (r.display_voice || r.voice.voice_id)
          : (r.voice.native_voice || r.display_voice))
      : "";
    if (voiceLabel) {
      speaks.append(el("span", { class: "rcvoice" }, [waveform(voiceLabel), el("span", { class: "rcmodel", text: voiceLabel })]));
    }
    chain.append(speaks);
    routeBox.append(chain);

    // warnings are yellow, never red (README §3)
    for (const w of state.warnings) {
      routeBox.append(
        errorBar([el("b", { text: routeWarningText(w) }), el("span", { class: "mono", style: "font-size:11px", text: typeof w === "string" ? w : w.code || "" })], "warn"),
      );
    }
  }

  function rcs(role, model) {
    return el("div", { class: "rcs" }, [el("span", { class: "rcrole", text: role }), el("span", { class: "rcmodel", text: model || "—" })]);
  }

  // ---- draft banner
  const draftBanner = el("div");
  let countdownTimer = null;
  function renderDraftBanner() {
    clear(draftBanner);
    const kind = testConsoleRouteKind(state);
    if (kind === "draft-test") {
      const expiresAt = Number(sessionStorage.getItem("rtv_ticket_exp")) || 0;
      const tick = () => {
        const left = Math.max(0, Math.round(expiresAt - Date.now() / 1000));
        draftBanner.textContent = "";
        draftBanner.append(
          errorBar(
            [
              el("b", { text: "Testing what the set page shows · " }),
              "saved or not, and also when the set is off. ",
              `This test link expires in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}.`,
            ],
            "note",
          ),
        );
        if (left <= 0) clearInterval(countdownTimer);
      };
      tick();
      clearInterval(countdownTimer);
      countdownTimer = setInterval(tick, 1000);
      return;
    }
    clearInterval(countdownTimer);
    if (kind === "active") {
      const p = profileById(state.profileId);
      const voice = p ? profileVoiceLabel(p) : "";
      draftBanner.append(
        el("div", { class: "fhint", "data-f12": "console-mode" }, [
          "The saved set",
          voice ? ` · voice ${voice}` : "",
          " — what somora gets.",
        ]),
      );
    }
  }

  /* A running call keeps what it started with. When the set was saved since,
   * say so: Stop and Talk starts a call with the saved set. */
  const staleNote = el("div");
  let startedAt = 0;
  function renderStaleNote() {
    clear(staleNote);
    const kind = testConsoleRouteKind(state);
    if (kind !== "active" || !state.route || !startedAt) return;
    const p = profileById(state.profileId);
    if (!p || !(Number(p.updated_at) > startedAt)) return;
    staleNote.append(
      errorBar(
        [
          el("b", { text: "The set was saved after this call started. " }),
          "A running call keeps what it started with — press Stop and Talk to use the saved set.",
        ],
        "warn",
      ),
    );
  }

  // ---- a failed model, named
  const MODEL_OF = { stt: "Listening model", llm: "Thinking model", tts: "Speaking model" };
  function showModelError(err) {
    const code = String(err.code || err.type || "");
    const role = code.split("_")[0];
    const who = MODEL_OF[role];
    if (!who && !/turn_queue|invalid|unknown|frozen/.test(code)) return;
    const slot = `model-error-${role || "other"}`;
    errorBox.querySelectorAll(`[data-slot="${slot}"]`).forEach((n) => n.remove());
    const bar = errorBar([el("b", { text: who ? `${who} failed. ` : "Problem. " }), err.message || code, " ", el("span", { class: "code mono", text: code })], "warn");
    bar.dataset.slot = slot;
    errorBox.append(bar);
  }

  // ---- talk state
  function renderState() {
    const info = STATE_TEXT[state.stateName] || STATE_TEXT.idle;
    clear(talkState);
    talkState.append(
      el("div", { class: "tsnow" }, [el("span", { class: "tsdot", style: `background:${info.color}` }), info.label]),
      el("div", { class: "tssub", text: info.sub }),
    );
    clear(talkBtn);
    const running = state.stateName !== "idle";
    talkBtn.className = "talkbtn " + (running ? "stop" : "");
    talkBtn.append(el("span", { class: "tb-main", text: running ? "Stop" : "Talk" }));
    talkBtn.setAttribute("aria-label", running ? "Stop" : "Talk");
    clear(pills);
    const pillLabels = [["idle", "Idle"], ["listening", "Listening"], ["thinking", "Thinking"], ["speaking", "Speaking"]];
    for (const [key, label] of pillLabels) {
      const on = state.stateName === key;
      pills.append(el("span", { class: "stpill " + (on ? "on" : "") }, [el("span", { class: "d" }), label]));
    }
    if (state.stateName === "route-pending") {
      pills.append(el("span", { class: "stpill on" }, [el("span", { class: "d" }), "Waiting for route"]));
    }
    if (state.stateName === "connecting") {
      pills.append(el("span", { class: "stpill on" }, [el("span", { class: "d" }), "Connecting"]));
    }
  }

  // ---- transcript
  const transcriptBox = el("div", { class: "tsgrid" });
  function renderTranscript() {
    clear(transcriptBox);
    const userCol = el("div", { class: "tscol" }, [el("div", { class: "tshead", text: "You" })]);
    const botCol = el("div", { class: "tscol" }, [el("div", { class: "tshead", text: "Assistant" })]);
    for (const turn of state.turns) {
      const col = turn.speaker === "user" ? userCol : botCol;
      const cls = "turn " + (turn.live ? "live" : "");
      const node = el("div", { class: cls }, [turn.text || ""]);
      if (turn.live) node.append(el("span", { class: "caret", "aria-hidden": "true" }));
      node.append(el("div", { class: "turnmeta", text: `${turn.time || ""} · ${turn.state || ""}` }));
      col.append(node);
    }
    transcriptBox.append(userCol, botCol);
  }

  function addTurn(turn) {
    state.turns.push(turn);
    renderTranscript();
  }

  // ---- diagnostics
  const diagBody = el("div", { class: "diagbody hidden" });
  const diagSum = el("div", { class: "diagsum" });
  function renderDiagnostics() {
    clear(diagBody);
    const cells = [
      ["Answer", state.metrics.llm],
      ["First audio", state.metrics.firstAudio],
    ];
    diagBody.append(
      el("div", { class: "mgrid" }, cells.map(([label, val]) => el("div", { class: "mcell" }, [el("div", { class: "mlab", text: label }), el("div", { class: "mval", text: val != null ? fmtMs(val) : "—" })]))),
      el("div", { class: "mnote", text: "Measured in this browser from the server's events — not what you hear through the speakers." }),
    );
    const evt = el("div", { class: "evt" });
    for (const e of state.events.slice(-30).reverse()) {
      evt.append(el("div", { class: "evtrow" }, [el("span", { class: "evtt mono", text: e.time }), el("span", { class: "evtn mono", text: e.type }), el("span", { class: "evtd", text: e.detail || "" })]));
    }
    diagBody.append(evt);
    if (state.route) {
      diagBody.append(
        el("div", { class: "mnote" }, ["Reference fingerpr.", " ", el("span", { class: "mono", text: (state.route.voice && state.route.voice.ref_sha256) || "—" })]),
      );
    }
    const errors = state.events.filter((e) => e.type === "error").length;
    diagSum.textContent = `${state.metrics.firstAudio != null ? "first audio " + fmtMs(state.metrics.firstAudio) + " · " : ""}${errors} error${errors === 1 ? "" : "s"}`;
  }

  const diagHead = el("button", { class: "diaghead", type: "button", "aria-expanded": "false" });
  const diagChev = el("span", { class: "diagchev", text: "▾" });
  diagHead.append(diagChev, el("span", { class: "diagtitle", text: "Diagnostics" }), diagSum);
  diagHead.addEventListener("click", () => {
    const open = diagBody.classList.toggle("hidden");
    diagChev.textContent = open ? "▾" : "▴";
    diagHead.setAttribute("aria-expanded", String(!open));
  });

  // ---- session lifecycle
  function ensureSession() {
    if (state.session) return state.session;
    const session = new RealtimeSession({
      token: sessionStorage.getItem("rtv_token") || "",
      deviceId: state.micDeviceId || "",
      onEvent: (msg) => {
        logEvent(msg);
        handleSessionEvent(msg);
      },
      onState: (name) => {
        state.stateName = name;
        renderState();
      },
      onRouteUpdate: (route, warnings) => {
        if (!state.route) startedAt = Date.now() / 1000;
        state.route = route;
        state.warnings = warnings || [];
        renderRoute();
        renderStaleNote();
        renderDiagnostics();
      },
      onError: (message) => {
        // surface exactly what failed and that nothing was sent
        clear(errorBox);
        errorBox.append(
          errorBar(
            [
              ...(/websocket|connection|closed/i.test(message)
                ? [
                    el("b", { text: "Could not connect. " }),
                    message,
                    " Check that the set is On (Sets) and its models answer (Models), then press Talk again.",
                  ]
                : [
                    el("b", { text: "Microphone/audio start failed. " }),
                    message,
                    el("b", { text: " No audio was sent." }),
                    " Allow microphone access, check the selected microphone, then press Talk again.",
                  ]),
            ],
            "err",
          ),
        );
      },
    });
    state.session = session;
    // diagnostics hook: the Test Console's own page may inspect/drive the live
    // session (used by the functional e2e for tool calls and barge-in)
    try {
      window.__rtvTestConsole = { session, state };
    } catch {
      /* ignore */
    }
    return session;
  }

  function logEvent(msg) {
    const t = new Date().toISOString().slice(11, 19);
    state.events.push({ time: t, type: msg.type || "?", detail: eventDetail(msg) });
    renderDiagnostics();
  }

  function eventDetail(msg) {
    switch (msg.type) {
      case "input_audio_buffer.speech_started":
        return "barge-in";
      case "ui.connection_closed":
        errorBox.append(errorBar([el("b", { text: "The server ended the call. " }), msg.reason || "No reason given.", " ", el("span", { class: "code mono", text: String(msg.code) })], "err"));
        break;
      case "response.output_audio.delta":
        return "chunk";
      case "response.function_call_arguments.done":
        return `tool ${msg.name || ""}`;
      case "session.updated":
        return msg.session && msg.session.x_realtimevoice && msg.session.x_realtimevoice.route ? `route resolved · ${msg.session.x_realtimevoice.route.profile}` : "";
      case "error":
        return (msg.error && msg.error.message) || "";
      default:
        return "";
    }
  }

  /* A deterministic test tool: the console installs it so a tool round trip
   * (arguments -> function_call_output -> follow-up response) can be proven. */
  const TEST_TOOL = {
    type: "function",
    name: "get_test_value",
    description: "Returns a fixed demo value for a label. Used to test tool calling.",
    parameters: {
      type: "object",
      properties: { label: { type: "string", description: "a label" } },
      required: ["label"],
    },
  };

  function installTestTool(session) {
    session.send({
      type: "session.update",
      session: { tools: [TEST_TOOL] },
    });
  }

  function handleToolCall(session, msg) {
    let args = {};
    try {
      args = JSON.parse(msg.arguments || "{}");
    } catch {
      args = {};
    }
    const output = JSON.stringify({ label: args.label || "demo", value: 42, source: "console-test-tool" });
    session.send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: msg.call_id, output },
    });
    session.send({ type: "response.create" });
    state.toolRoundTrips = (state.toolRoundTrips || 0) + 1;
  }

  function handleSessionEvent(msg) {
    const session = state.session;
    switch (msg.type) {
      case "input_audio_buffer.speech_started":
        state.metrics.endOfSpeech = null;
        break;
      case "conversation.item.input_audio_transcription.completed":
        addTurn({ speaker: "user", text: msg.transcript || "", state: "final", time: clock(), live: false });
        break;
      case "response.output_audio_transcript.delta":
        setLiveTurn("user", null);
        setLiveTurn("assistant", (getLiveTurn("assistant") || "") + (msg.delta || ""));
        break;
      case "response.output_audio_transcript.done":
        setLiveTurn("assistant", msg.transcript || getLiveTurn("assistant") || "", "done");
        state.metrics.llm = state.metrics.responseCreated != null ? performance.now() - state.metrics.responseCreated : state.metrics.llm;
        break;
      case "response.function_call_arguments.done":
        handleToolCall(session, msg);
        break;
      case "response.created":
        state.metrics.responseCreated = performance.now();
        // install the deterministic test tool once the session is open
        if (!state.testToolInstalled && session) {
          state.testToolInstalled = true;
          installTestTool(session);
        }
        break;
      case "session.updated":
        if (!state.testToolInstalled && session) {
          state.testToolInstalled = true;
          installTestTool(session);
        }
        break;
      case "response.output_audio.delta":
        if (state.metrics.firstAudio == null && state.metrics.responseCreated != null) {
          state.metrics.firstAudio = Math.round(performance.now() - state.metrics.responseCreated);
        }
        break;
      case "error":
        if (msg.error && msg.error.code === "voice_route_pending") {
          // calm, expected state — a pending line, not a red error
          renderState();
        } else if (msg.error && msg.error.fatal) {
          errorBox.append(errorBar([el("b", { text: "Session error. " }), msg.error.message], "err"));
        } else if (msg.error) {
          // A model that failed is said, with which one: the call goes on,
          // but the answer is missing. (Before: only a counter in the
          // folded diagnostics.)
          showModelError(msg.error);
        }
        break;
      default:
        break;
    }
  }

  function getLiveTurn(speaker) {
    const t = state.turns[state.turns.length - 1];
    return t && t.speaker === speaker && t.live ? t.text : null;
  }
  function setLiveTurn(speaker, text, stateName) {
    if (text == null) return;
    let t = state.turns[state.turns.length - 1];
    if (!t || t.speaker !== speaker || !t.live) {
      t = { speaker, text: "", state: stateName || "partial", time: clock(), live: true };
      state.turns.push(t);
    }
    t.text = text;
    if (stateName) t.state = stateName;
    renderTranscript();
  }
  function clock() {
    return new Date().toTimeString().slice(0, 8);
  }

  async function toggleTalk() {
    const session = ensureSession();
    if (state.stateName !== "idle") {
      session.stop(false);
      state.stateName = "idle";
      if (state.profileId && ctx.markInUse) ctx.markInUse(state.profileId, false);
      renderState();
      return;
    }
    const model = selectedModel();
    if (!model) {
      clear(errorBox);
      errorBox.append(errorBar([el("b", { text: "Choose a set first. " }), "Talk opens a call with the set you pick (only sets that are On are listed)."], "warn"));
      return;
    }
    clear(errorBox);
    if (state.profileId && ctx.markInUse) ctx.markInUse(state.profileId, true);
    // C1: NOTHING may be awaited between this click and session.start(). The
    // AudioContext + getUserMedia must stay inside the click's user activation
    // (Safari/iOS can otherwise refuse or suspend audio). The stale-revision
    // refresh runs fire-and-forget AFTER the session started, and again when
    // the picker gets focus or the page becomes visible — never awaited here.
    state.route = null;
    state.warnings = [];
    state.turns = [];
    state.events = [];
    state.metrics = {};
    state.testToolInstalled = false;
    state.toolRoundTrips = 0;
    renderRoute();
    renderTranscript();
    renderDiagnostics();
    try {
      await session.start(model);
      renderMicState();
      // C1: only AFTER start is in flight may the list be refreshed (the
      // note then reflects any concurrent activation). Fire-and-forget.
      Promise.resolve().then(() => refreshProfiles());
      // install the test tool right away so a tool round trip is possible
      if (!state.testToolInstalled) {
        state.testToolInstalled = true;
        installTestTool(session);
      }
    } catch (err) {
      if (!errorBox.childElementCount) {
        errorBox.append(errorBar([el("b", { text: "Could not start the session. " }), (err && err.message) || String(err), el("b", { text: " No audio was sent." }), " Check the connection and try again."], "err"));
      }
      session.stop(true);
      state.stateName = "idle";
      renderState();
    }
  }

  talkBtn.addEventListener("click", toggleTalk);

  // F2: microphone selection for the console
  const micHost = el("div", { class: "testmic" });
  let micPicker = null;
  async function mountMic() {
    try {
      micPicker = await createMicPicker({
        label: "Test console microphone",
        onChange: (id, missing) => {
          state.micDeviceId = id;
          state.micMissing = missing;
          renderMicState();
        },
      });
      clear(micHost);
      micHost.append(micPicker.node);
      state.micDeviceId = micPicker.value();
      watchDeviceChanges(() => micPicker.refresh().catch(() => {}));
    } catch {
      /* no device enumeration: system default is used */
    }
  }

  function renderMicState() {
    if (!micPicker) return;
    const existing = micHost.querySelector(".micsummary");
    if (existing) existing.remove();
    const session = state.session;
    const label = session && session.activeMicLabel ? session.activeMicLabel : micPicker.value() ? "selected device" : "System default";
    micHost.append(el("div", { class: "micsummary fhint", text: `Microphone: ${label}` }));
  }

  // ---- assemble
  const buildStamp = ctx.revision
    ? el("div", { class: "buildstamp mono", id: "test-build", text: "build " + ctx.revision })
    : el("div");
  inner.append(
    ctx.shell.breadcrumb([{ label: "Test" }]),
    el("div", { class: "topbar-editor" }, [
      el("div", {}, [title, el("div", { class: "h1sub", text: "Talk to a set the way somora does — over the realtime protocol." })]),
      el("div", { class: "head-actions" }, [el("label", { class: "sr-only", for: "test-profile", text: "Set" }), picker]),
    ]),
    draftBanner,
    routeBox,
    staleNote,
    el("div", { class: "talkrow" }, [talkBtn, talkState]),
    buildStamp,
    micHost,
    pills,
    errorBox,
    transcriptBox,
    el("div", { class: "diag" }, [diagHead, diagBody]),
  );

  renderPicker();
  renderRoute();
  renderState();
  renderTranscript();
  renderDiagnostics();
  renderDraftBanner();
  renderStaleNote();
  mountMic().then(renderMicState);

  if (ticket && !sessionStorage.getItem("rtv_ticket_exp")) {
    // best-effort: the Test Console cannot know expires_at unless the editor
    // passed it; keep a 15-minute default that matches the contract TTL
    sessionStorage.setItem("rtv_ticket_exp", String(Math.round(Date.now() / 1000) + 900));
  }
}

export function disposeTestConsole() {
  // retained for symmetry; the session closes on navigation via stop()
}