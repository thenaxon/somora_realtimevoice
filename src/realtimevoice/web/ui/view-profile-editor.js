/* Set page.
 *
 * A set is what an agent speaks with. The page reads as sentences:
 *   01 What it speaks with — listens with / thinks with / speaks as (the
 *      voice first; the speaking model list then offers only the models that
 *      can speak it) / language
 *   02 Try it — hear a sentence, or talk to it (with what is on screen)
 *   More — name, instructions, pacing, other apps, earlier versions, delete
 *
 * Saving is what calls use from the next call on. A set that is On must be
 * complete; a set that is Off can be saved half-done.
 */

import { el, clear, copyText, errorBar, spinner } from "./dom.js";
import {
  LANGUAGES,
  languageLabel,
  voiceChoices,
  voiceKeyFromSpec,
  specVoiceFromKey,
  findVoiceChoice,
  setIssueText,
  probeErrorText,
  fmtClock,
  referenceTooLongText,
  referenceTooLongIssue,
  excerptRoute,
} from "./format.js";
import { profiles as profilesApi, catalog as catalogApi } from "./api.js";
import { sharedSamplePlayer } from "./sample-player.js";
import { confirmDialog } from "./dialogs.js";

const ROLE_WORDS = { stt: "Listens with", llm: "Thinks with" };
const MODEL_WORDS = { stt: "the listening model", llm: "the thinking model", tts: "the speaking model" };

function rowName(row) {
  return (row && (row.display_name || row.model || row.id)) || "";
}

export async function renderProfileEditor(inner, ctx, profileId) {
  let profile;
  let server = {};
  try {
    [profile, server] = await Promise.all([profilesApi.get(profileId), catalogApi.settings().catch(() => ({}))]);
  } catch (err) {
    inner.append(
      ctx.shell.breadcrumb([{ label: "Sets", route: "#/profiles" }, { label: profileId }]),
      errorBar([el("b", { text: "Could not load the set. " }), err.message, " Go back to the list and try again."]),
    );
    return;
  }
  const voices = await ctx.loadVoices();
  const ttsRows = ctx.roleProfiles("tts");
  const choices = voiceChoices(ttsRows, voices);

  // what is on screen; Save stores it
  const state = {};
  function loadState(p) {
    const s = p.spec || {};
    Object.assign(state, {
      displayName: p.display_name || p.id,
      description: p.description || "",
      stt: s.stt_profile || "",
      llm: s.llm_profile || "",
      tts: s.tts_profile || "",
      voiceKey: s.tts_profile || (s.voice && s.voice.kind === "library") ? voiceKeyFromSpec(s) : "",
      language: s.language || "de",
      instructions: s.instructions_default || "",
      turnDetection: s.turn_detection_default || {},
      pacing: s.output_pacing || "off",
      enabled: !!p.enabled,
    });
  }
  loadState(profile);
  let saved = "";
  let testSentence = "";
  let hearUrl = "";

  function specFromState() {
    return {
      stt_profile: state.stt,
      llm_profile: state.llm,
      tts_profile: state.tts,
      voice: specVoiceFromKey(state.voiceKey),
      language: state.language,
      instructions_default: state.instructions,
      turn_detection_default: state.turnDetection,
      output_pacing: state.pacing,
    };
  }

  function payload() {
    return {
      display_name: state.displayName.trim() || profile.id,
      description: state.description,
      spec: specFromState(),
      enabled: state.enabled,
    };
  }

  function dirty() {
    return JSON.stringify(payload()) !== saved;
  }

  function markSaved() {
    saved = JSON.stringify(payload());
  }
  markSaved();

  /* ------------------------------------------------------------ header */

  const header = el("div", { class: "topbar-editor" });
  const status = el("div", { class: "statusline", role: "status" });
  const saveBtn = el("button", { class: "btn pri", type: "button", text: "Save", on: { click: () => save() } });
  const onSwitch = el("input", { type: "checkbox", "aria-label": "On", on: { change: () => toggleOn(onSwitch.checked) } });

  function renderHeader() {
    clear(header);
    onSwitch.checked = state.enabled;
    saveBtn.disabled = !dirty();
    const sub = state.enabled
      ? "On · somora can call it. Saving changes the next call; a running call keeps what it started with."
      : "Off · somora cannot call it. You can save it half-done and switch it on when it is complete.";
    header.append(
      el("div", {}, [
        el("h1", { class: "h1", text: state.displayName || profile.id }),
        el("div", { class: "h1sub", text: sub }),
      ]),
      el("div", { class: "head-actions" }, [
        el("label", { class: "onswitch", title: "Whether somora can call this set" }, [
          onSwitch,
          el("span", { text: "On" }),
        ]),
        saveBtn,
      ]),
    );
  }

  /* --------------------------------------------- 01 what it speaks with */

  const sentenceBox = el("div", { class: "card setcard" });
  const issueBox = el("div", { class: "setissues" });

  function roleSelect(role) {
    const rows = ctx.roleProfiles(role);
    const sel = el("select", { class: "inp", "aria-label": ROLE_WORDS[role] });
    sel.append(el("option", { value: "", text: "— choose —" }));
    for (const r of rows) sel.append(el("option", { value: r.id, text: rowName(r) }));
    sel.value = state[role];
    if (state[role] && sel.value !== state[role]) {
      sel.append(el("option", { value: state[role], text: `${state[role]} (no longer exists)` }));
      sel.value = state[role];
    }
    sel.addEventListener("change", () => {
      state[role] = sel.value;
      changed();
    });
    return sel;
  }

  function modelHint(role, id) {
    const row = ctx.roleProfiles(role).find((r) => r.id === id);
    if (!row) return "";
    return row.model && row.model !== rowName(row) ? row.model : "";
  }

  function voiceSelect() {
    const sel = el("select", { class: "inp", "aria-label": "Speaks as" });
    sel.append(el("option", { value: "", text: "— choose a voice —" }));
    for (const g of choices) {
      const og = el("optgroup", { label: g.label });
      for (const it of g.items) {
        og.append(el("option", { value: it.key, text: it.note ? `${it.label} (${it.note})` : it.label }));
      }
      sel.append(og);
    }
    sel.value = state.voiceKey;
    if (state.voiceKey && sel.value !== state.voiceKey) {
      sel.append(el("option", { value: state.voiceKey, text: "(the saved voice is no longer available)" }));
      sel.value = state.voiceKey;
    }
    sel.addEventListener("change", () => {
      state.voiceKey = sel.value;
      const item = findVoiceChoice(choices, state.voiceKey);
      // the speaking model follows the voice: keep it when it can speak the
      // voice, otherwise take the first one that can
      if (item && !item.modelIds.includes(state.tts)) state.tts = item.modelIds[0] || "";
      changed(true);
    });
    return sel;
  }

  function speakerControl() {
    const item = findVoiceChoice(choices, state.voiceKey);
    const ids = item ? item.modelIds : [];
    if (item && ids.length === 1 && state.tts === ids[0]) {
      return el("div", { class: "setval", text: rowName(ttsRows.find((r) => r.id === ids[0])) });
    }
    const sel = el("select", { class: "inp", "aria-label": "Speaking model" });
    sel.append(el("option", { value: "", text: item ? "— choose —" : "— choose a voice first —" }));
    for (const id of ids) sel.append(el("option", { value: id, text: rowName(ttsRows.find((r) => r.id === id)) }));
    sel.value = state.tts;
    if (state.tts && sel.value !== state.tts) {
      sel.append(el("option", { value: state.tts, text: `${state.tts} (cannot speak this voice)` }));
      sel.value = state.tts;
    }
    sel.disabled = !item;
    sel.addEventListener("change", () => {
      state.tts = sel.value;
      changed();
    });
    return sel;
  }

  function languageSelect() {
    const sel = el("select", { class: "inp", "aria-label": "Language" });
    for (const [code, name] of LANGUAGES) sel.append(el("option", { value: code, text: `${name} (${code})` }));
    if (state.language && !LANGUAGES.some(([c]) => c === state.language)) {
      sel.append(el("option", { value: state.language, text: languageLabel(state.language) }));
    }
    sel.value = state.language;
    sel.addEventListener("change", () => {
      state.language = sel.value;
      changed();
    });
    return sel;
  }

  function sentenceRow(label, control, hint) {
    return el("div", { class: "setrow" }, [
      el("div", { class: "setlab", text: label }),
      el("div", { class: "setctl" }, [control, hint ? el("div", { class: "fhint", text: hint }) : null]),
    ]);
  }

  function renderSentence() {
    clear(sentenceBox);
    const item = findVoiceChoice(choices, state.voiceKey);
    const voiceHint = {
      library: "Your recording, cloned by the speaking model. Change the recording in Voices.",
      native: "A voice built into the speaking model.",
      none: "This model speaks with its own single voice.",
    }[item ? item.kind : ""] || (choices.length ? "" : "Add a speaking model in Models first.");
    sentenceBox.append(
      sentenceRow("Listens with", roleSelect("stt"), modelHint("stt", state.stt)),
      sentenceRow("Thinks with", roleSelect("llm"), modelHint("llm", state.llm)),
      sentenceRow("Speaks as", voiceSelect(), voiceHint),
      sentenceRow("Speaking model", speakerControl(), modelHint("tts", state.tts)),
      sentenceRow("Language", languageSelect(), "The language you speak with it. Each model gets it in the form it needs."),
    );
  }

  let checkSeq = 0;
  async function renderIssues() {
    const seq = ++checkSeq;
    let res;
    try {
      res = await profilesApi.validate(profile.id, specFromState());
    } catch {
      return; // the check is informational
    }
    if (seq !== checkSeq) return; // a newer check is running
    clear(issueBox);
    const blocking = (res.issues || []).filter((i) => i.severity !== "warning");
    const failed = Object.entries(res.probes || {}).filter(([, p]) => p && !p.ok);
    if (!blocking.length && !failed.length) {
      issueBox.append(errorBar([el("b", { text: "Ready. " }), "All three models answered."], "ok"));
      return;
    }
    const tooLong = referenceTooLongIssue(blocking);
    if (tooLong) {
      issueBox.append(
        errorBar([
          el("b", { text: "Not ready. " }),
          referenceTooLongText(tooLong.sample_ms, tooLong.max_ms) + " ",
          el("a", { class: "linklike", href: excerptRoute(tooLong.voice_id, tooLong.sample_id, tooLong.max_ms, profile.id), text: "Choose an excerpt" }),
        ]),
      );
    } else if (blocking.length) {
      issueBox.append(errorBar([el("b", { text: "Not ready. " }), setIssueText(blocking) + "."]));
    }
    for (const [role, p] of failed) {
      const word = MODEL_WORDS[role];
      issueBox.append(
        errorBar([el("b", { text: `${word[0].toUpperCase()}${word.slice(1)} did not answer. ` }), probeErrorText(p.error_code, p)], "warn"),
      );
    }
  }

  let issueTimer = 0;
  function changed(rebuild = false) {
    if (rebuild) renderSentence();
    renderHeader();
    clearTimeout(issueTimer);
    issueTimer = setTimeout(() => renderIssues(), 400);
  }

  /* ---------------------------------------------------------- 02 try it */

  const testBox = el("div", { class: "card" });
  const testResult = el("div", { class: "resultbox hidden" });

  function renderTest() {
    clear(testBox);
    const text = el("textarea", { class: "inp", rows: 2, "aria-label": "Test sentence" });
    text.value = testSentence || "Guten Morgen, es ist Viertel nach acht und im Apfelbaum vor dem Fenster sitzen drei Vögel.";
    text.addEventListener("input", () => {
      testSentence = text.value;
    });
    testBox.append(
      el("div", { class: "testrow" }, [
        text,
        el("div", { class: "testactions" }, [
          el("button", { class: "btn pri", type: "button", text: "Hear a sentence", on: { click: () => hear(text.value) } }),
          el("button", { class: "btn", type: "button", text: "Talk to it", on: { click: () => talk() } }),
        ]),
      ]),
      el("div", { class: "fhint", text: "Uses what you see here, saved or not. Nothing is saved." }),
      testResult,
    );
  }

  function revokeHear() {
    if (!hearUrl) return;
    try {
      URL.revokeObjectURL(hearUrl);
    } catch {}
    hearUrl = "";
  }

  function testError(err, what) {
    const tooLong = referenceTooLongIssue(err);
    if (tooLong) {
      return errorBar([
        el("b", { text: `${what} ` }),
        referenceTooLongText(tooLong.sample_ms, tooLong.max_ms) + " ",
        el("a", { class: "linklike", href: excerptRoute(tooLong.voice_id, tooLong.sample_id, tooLong.max_ms, profile.id), text: "Choose an excerpt" }),
      ]);
    }
    const reason = (err.issues || []).length ? setIssueText(err.issues) + "." : err.message;
    return errorBar([el("b", { text: `${what} ` }), reason]);
  }

  async function hear(sentence) {
    clear(testResult);
    testResult.classList.remove("hidden");
    if (!(sentence || "").trim()) {
      testResult.append(errorBar([el("b", { text: "Nothing to speak. " }), "Type a sentence first."]));
      return;
    }
    testResult.append(spinner("Speaking the sentence…"));
    try {
      const resp = await profilesApi.testSpeech(profile.id, sentence, specFromState());
      const blob = await resp.blob();
      clear(testResult);
      revokeHear();
      hearUrl = URL.createObjectURL(blob);
      const audio = el("audio", { controls: true, preload: "none", src: hearUrl, class: "preview-audio", style: "width:100%;margin-top:8px" });
      audio.addEventListener("play", () => sharedSamplePlayer.stop());
      const removed = Number(resp.headers.get("X-Rtv-Emoji-Removed") || 0);
      testResult.append(
        audio,
        el("div", {
          class: "hintline",
          "data-f10": "hear-provenance",
          text:
            `${resp.headers.get("X-Rtv-Voice") || "voice"} · ${resp.headers.get("X-Rtv-Model") || ""}` +
            (removed ? ` · ${removed} emoji skipped, as in a call` : ""),
        }),
      );
    } catch (err) {
      clear(testResult);
      testResult.append(testError(err, err.code === "nothing_to_speak" ? "Nothing to speak." : "Could not speak it."));
    }
  }

  async function talk() {
    clear(testResult);
    testResult.classList.remove("hidden");
    testResult.append(spinner("Preparing a test call…"));
    try {
      const ticket = await profilesApi.testSession(profile.id, dirty() ? specFromState() : null);
      ctx.go(`#/test?profile=${encodeURIComponent(profile.id)}&ticket=${encodeURIComponent(ticket.model)}`);
    } catch (err) {
      clear(testResult);
      testResult.append(testError(err, "Cannot call it yet."));
    }
  }

  /* -------------------------------------------------------------- more */

  const moreBox = el("div", { class: "setmore" });

  function renderMore() {
    clear(moreBox);
    const nameInp = el("input", { class: "inp", value: state.displayName, "aria-label": "Display name" });
    nameInp.addEventListener("input", () => {
      state.displayName = nameInp.value;
      renderHeader();
    });
    const descInp = el("input", { class: "inp", value: state.description, "aria-label": "Description" });
    descInp.addEventListener("input", () => {
      state.description = descInp.value;
      renderHeader();
    });
    const instr = el("textarea", { class: "inp", rows: 3, "aria-label": "Instructions" });
    instr.value = state.instructions;
    instr.addEventListener("input", () => {
      state.instructions = instr.value;
      renderHeader();
    });
    const pacing = el("select", { class: "inp", "aria-label": "Pacing" }, [
      el("option", { value: "off", text: "As fast as it is made" }),
      el("option", { value: "realtime", text: "At speaking speed" }),
    ]);
    pacing.value = state.pacing;
    pacing.addEventListener("change", () => {
      state.pacing = pacing.value;
      renderHeader();
    });

    const isDefault = server.default_set === profile.id;
    const defaultBtn = el("button", {
      class: "btn",
      type: "button",
      text: isDefault ? "Stop using it for other apps" : "Use it for other apps",
      on: {
        click: async () => {
          try {
            server = await catalogApi.saveSettings({ default_set: isDefault ? "" : profile.id });
            renderMore();
          } catch (err) {
            showError(err, "Not changed");
          }
        },
      },
    });

    moreBox.append(
      details("Name and description", [
        field("Display name", nameInp, "Shown in this app."),
        field(
          "Name for somora",
          el("div", { class: "swrow" }, [
            el("span", { class: "code mono", text: profile.id }),
            el("button", { class: "btn", type: "button", text: "Copy", on: { click: (e) => copyText(profile.id, e.currentTarget) } }),
          ]),
          "In somora, give an agent this name as its voice. It cannot be changed.",
        ),
        field("Description", descInp),
      ]),
      details("Instructions and pacing", [
        field("Instructions", instr, "Used only when the app sends none. somora always sends its own, so for somora this stays unused."),
        field("Pacing", pacing, "At speaking speed, the voice is sent no faster than it is spoken, so an answer ends when you have heard it."),
      ]),
      details("Other apps", [
        el("p", { class: "fhint", text: "somora uses the name for somora. Another app can name this set in its model field:" }),
        el("div", { class: "swrow" }, [
          el("span", { class: "code mono", text: `profile:${profile.id}` }),
          el("button", { class: "btn", type: "button", text: "Copy", on: { click: (e) => copyText(`profile:${profile.id}`, e.currentTarget) } }),
        ]),
        el("p", {
          class: "fhint",
          text: isDefault
            ? "An app that names no set gets this one."
            : "An app that names no set gets the set chosen for other apps" +
              (server.default_set ? ` (now: ${server.default_set}).` : "; none is chosen, so such a call is refused."),
        }),
        defaultBtn,
      ]),
      details(`Earlier versions (${(profile.history || []).length})`, versionsList()),
      details("Delete", [
        el("p", { class: "fhint", text: state.enabled ? "Switch the set off first; somora can call it now." : "Deletes the set. Its voice and models stay." }),
        el("button", { class: "btn danger", type: "button", text: "Delete set", disabled: state.enabled, on: { click: () => removeSet() } }),
      ]),
    );
  }

  function versionsList() {
    const list = profile.history || [];
    if (!list.length) return [el("p", { class: "fhint", text: "Each save keeps the version before it here." })];
    return list.map((v, index) => {
      const s = v.spec || {};
      const voice = findVoiceChoice(choices, voiceKeyFromSpec(s));
      const what = [
        rowName(ctx.roleProfiles("stt").find((r) => r.id === s.stt_profile)) || s.stt_profile || "—",
        rowName(ctx.roleProfiles("llm").find((r) => r.id === s.llm_profile)) || s.llm_profile || "—",
        voice ? voice.label : "—",
        rowName(ttsRows.find((r) => r.id === s.tts_profile)) || s.tts_profile || "—",
        languageLabel(s.language),
      ].join(" · ");
      return el("div", { class: "rev" }, [
        el("div", { class: "revmain" }, [
          el("div", { class: "revmeta", text: `Saved ${fmtClock(v.saved_at)}` }),
          el("div", { class: "fhint", text: what }),
        ]),
        el("button", { class: "btn", type: "button", text: "Use this version", on: { click: () => restore(index) } }),
      ]);
    });
  }

  /* ---------------------------------------------------------- actions */

  function showResult(view, verb) {
    const failed = Object.entries(view.probes || {}).filter(([, p]) => p && !p.ok).map(([r]) => MODEL_WORDS[r]);
    status.textContent = failed.length ? `${verb}. Note: ${failed.join(", ")} did not answer.` : `${verb}.`;
    status.className = "statusline " + (failed.length ? "warn" : "ok");
  }

  function showError(err, verb) {
    const text = (err.issues || []).length ? setIssueText(err.issues) : err.message;
    status.textContent = `${verb}: ${text}. Nothing was changed.`;
    status.className = "statusline err";
  }

  async function save() {
    status.textContent = "Saving…";
    status.className = "statusline";
    try {
      const view = await profilesApi.save(profile.id, payload());
      profile = view;
      markSaved();
      showResult(view, state.enabled ? "Saved. The next call uses it" : "Saved");
      renderAll();
      return true;
    } catch (err) {
      showError(err, "Not saved");
      return false;
    }
  }

  async function toggleOn(on) {
    const before = state.enabled;
    state.enabled = on;
    // switching on or off saves what is on screen with it
    const ok = await save();
    if (!ok) {
      state.enabled = before;
      renderHeader();
    }
  }

  async function restore(index) {
    const ok = await confirmDialog({
      title: "Use this version?",
      message: "The set becomes this version again. What it is now goes to the earlier versions.",
      confirmLabel: "Use this version",
    });
    if (!ok) return;
    try {
      profile = await profilesApi.restore(profile.id, index);
      loadState(profile);
      markSaved();
      showResult(profile, "Earlier version in use");
      renderAll();
    } catch (err) {
      showError(err, "Not restored");
    }
  }

  async function removeSet() {
    const ok = await confirmDialog({
      title: `Delete ${state.displayName}?`,
      message: "The set is deleted. Its voice and models stay.",
      confirmLabel: "Delete set",
      danger: true,
    });
    if (!ok) return;
    try {
      await profilesApi.remove(profile.id);
      ctx.go("#/profiles");
    } catch (err) {
      showError(err, "Not deleted");
    }
  }

  function renderAll() {
    renderHeader();
    renderSentence();
    renderMore();
    renderIssues();
  }

  inner.append(
    ctx.shell.breadcrumb([{ label: "Sets", route: "#/profiles" }, { label: state.displayName }]),
    header,
    status,
    blockLabel("01", "What it speaks with"),
    sentenceBox,
    issueBox,
    blockLabel("02", "Try it"),
    testBox,
    blockLabel("", "More"),
    moreBox,
  );
  renderAll();
  renderTest();

  ctx.setCleanup(() => {
    revokeHear();
    clearTimeout(issueTimer);
    checkSeq += 1;
  });
}

function blockLabel(num, title, hint) {
  return el("div", { class: "blabel" }, [
    num ? el("span", { class: "bnum mono", text: num }) : null,
    el("span", { class: "btitle", text: title }),
    hint ? el("span", { class: "bhint", text: hint }) : null,
  ]);
}

function field(label, input, hint) {
  return el("div", { class: "fld" }, [el("label", { text: label }), input, hint ? el("div", { class: "fhint", text: hint }) : null]);
}

function details(title, children) {
  return el("details", { class: "fsection card" }, [el("summary", { class: "fsection-title", text: title }), ...children]);
}
