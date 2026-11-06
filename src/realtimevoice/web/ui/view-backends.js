/* Models: the model entries sets listen, think and speak with.
 *
 * A board with one column per kind of model, and a form per model (where it
 * runs, what it does, advanced). Edits use the masked-key semantics; a change
 * that would stop a switched-on set from working is refused by the server and
 * named here. Ids are made by the server and never shown.
 */

import { el, clear, errorBar } from "./dom.js";
import { catalog as catalogApi } from "./api.js";
import { formDialog, confirmDialog, noticeDialog } from "./dialogs.js";
import {
  thinkingModeFromExtra,
  extraWithThinking,
  THINKING_MODES,
  BACKEND_ROLE_CHOICES,
  NATIVE_VOICE_LABEL_MODES,
  changedEndpointKeyPrompt,
  changedEndpointKeyDetail,
  voiceAbility,
  VOICE_ABILITY_LABEL,
  MODEL_FIELD_HELP,
  VOICE_MODE_CHOICES,
  voiceModeFromCaps,
  capsForVoiceMode,
  VOICE_MODE_TRUST,
  LANGUAGE_FORMAT_CHOICES,
} from "./format.js";

const MASK = "********";
const SENSITIVE_EXTRA_RE = /(key|token|secret|password|authorization|credential|ref[_-]?audio|ref[_-]?wav|ref[_-]?text|ref[_-]?transcript|reference)/i;

export async function renderBackends(inner, ctx) {
  inner.append(
    ctx.shell.breadcrumb([{ label: "Models" }]),
    el("div", { class: "head" }, [
      el("div", {}, [
        el("h1", { class: "h1", text: "Models" }),
        el("p", { class: "lede", text: "The models your sets listen, think and speak with: each one a server address, a model name and its key. Keys stay on this server and are never shown." }),
      ]),
    ]),
  );

  const pane = el("div", { class: "pane" });
  inner.append(pane);
  await renderProfiles(pane, ctx);
}

/* --------------------------------------------------------------- profiles */

/* R2 / CP3: adding a backend must be understandable.
 *
 * The flow is role-first: "＋ Add backend" starts with three clearly explained
 * role choices, then a connection step whose fields depend on the role, then a
 * "Check connection" step that probes the UNSAVED form (endpoint + key + model)
 * and shows the reported model list, then save. Editing a saved backend is a
 * visibly separate mode ("Edit <name>", role fixed and shown, id read-only).
 * A duplicate id is refused with a suggestion and the existing row is never
 * overwritten — create is POST-only, never a silent PUT.
 */

async function renderProfiles(pane, ctx) {
  clear(pane);
  const status = el("div", { class: "statusline" });
  let role = "stt";
  let editing = null; // the saved profile being edited, or null for a new one
  let formMode = "none"; // "none" | "add" | "edit"
  let dirty = false;

  const roleSel = el("select", { class: "inp", style: "max-width:220px", "aria-label": "Role" }, [
    el("option", { value: "stt", text: "Speech recognition (STT)" }),
    el("option", { value: "llm", text: "Reasoning (LLM)" }),
    el("option", { value: "tts", text: "Speech engine (TTS)" }),
  ]);
  const profSel = el("select", { class: "inp", "aria-label": "Profile" });
  const addBtn = el("button", { class: "btn pri", type: "button", text: "＋ Add model" });
  const formTitle = el("div", { class: "ctitle", style: "margin-top:4px" });
  const formHost = el("div");

  function roleProfiles() {
    return ctx.roleProfiles(role);
  }
  function roleLabel() {
    return { stt: "STT", llm: "LLM", tts: "TTS" }[role];
  }
  function roleChoice(r) {
    return BACKEND_ROLE_CHOICES.find((c) => c.role === r) || BACKEND_ROLE_CHOICES[0];
  }
  function existingIds() {
    return ctx.roleProfiles(role).map((p) => p.id);
  }

  function fillProfileSelect(preferredId) {
    clear(profSel);
    const list = roleProfiles();
    for (const p of list) {
      profSel.append(el("option", { value: p.id, text: `${p.display_name || p.id} (${p.model || "?"})` }));
    }
    const wanted = preferredId || (editing && editing.id) || (list[0] && list[0].id) || "";
    const found = list.find((p) => p.id === wanted) || list[0] || null;
    editing = found;
    formMode = found ? "edit" : "none";
    dirty = false;
    if (found) profSel.value = found.id;
    fillForm();
  }

  function field(label, input, hint) {
    const help = typeof hint === "string" ? el("div", { class: "fhelp", text: hint }) : hint;
    return el("div", { class: "fld" }, [el("label", { text: label }), input, help || null]);
  }

  function setStatus(text, cls) {
    status.textContent = text;
    status.className = "statusline " + (cls || "");
  }

  /* Ask before throwing away unsaved edits. */
  async function confirmDiscard() {
    if (!dirty) return true;
    return await confirmDialog({
      title: "Discard your changes?",
      message: "This model has unsaved changes. Leaving now discards them.",
      confirmLabel: "Discard changes",
      cancelLabel: "Keep editing",
      danger: true,
    });
  }

  /* ------------------------------------------------------ role-first step */

  function renderRoleStep() {
    formMode = "add";
    editing = null;
    dirty = false;
    clear(formHost);
    formTitle.textContent = "Add a model — what should it do?";
    const cards = el("div", { class: "grid2", "data-cp3": "step-role", style: "margin-top:10px" });
    for (const choice of BACKEND_ROLE_CHOICES) {
      const card = el("button", {
        class: "btn",
        type: "button",
        "data-cp3": `role-${choice.role}`,
        style: "text-align:left;display:block;padding:12px 14px;line-height:1.5",
        on: {
          click: () => {
            role = choice.role;
            roleSel.value = choice.role;
            renderConnectionStep();
          },
        },
      }, [
        el("div", { style: "font-weight:600;font-size:14px", text: choice.label }),
        el("div", { style: "font-size:12px;color:var(--text-1);margin-top:2px", text: choice.help }),
      ]);
      cards.append(card);
    }
    formHost.append(
      el("div", { style: "font-size:12px;color:var(--text-1);margin-top:8px" }, [
        "Each model does one thing. Keys stay on this server and are never shown.",
      ]),
      cards,
    );
  }

  /* -------------------------------------------------- connection step (2) */

  function renderConnectionStep() {
    formMode = "add";
    editing = null;
    dirty = false;
    clear(formHost);
    const choice = roleChoice(role);
    formTitle.textContent = `Add a ${choice.label} model — where it runs`;
    formHost.append(
      el("div", { style: "font-size:12px;color:var(--text-1);margin:4px 0 8px" }, [
        el("button", {
          class: "btn sm", type: "button", text: "Change role",
          on: { click: async () => { if (await confirmDiscard()) renderRoleStep(); } },
        }),
      ]),
    );
    buildForm(null);
  }

  /* -------------------------------------------------------- edit step */

  function renderEditStep() {
    formMode = "edit";
    dirty = false;
    clear(formHost);
    formTitle.textContent = editing ? `Edit ${editing.display_name || editing.id}` : "Edit model";
    buildForm(editing);
  }

  /* ------------------------------------------------------------- form */

  function buildForm(p) {
    const creating = p == null;
    const role2 = role;
    const p0 = p || {
      role,
      id: "",
      adapter: "openai_http",
      base_url: "",
      model: "",
      language_format: "code",
      voice: "",
      request_timeout_s: 60,
      extra: {},
      capabilities: {},
    };

    const editBanner = creating
      ? null
      : el("div", { "data-cp3": "edit-mode", style: "margin:6px 0 8px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap" }, [
          el("span", {
            "data-cp3": "edit-role",
            "data-role": role2,
            style: "font-size:12px;color:var(--text-1)",
            text: `${roleChoice(role2).label} — what a model does cannot be changed`,
          }),
        ]);
    const addBanner = creating
      ? el("div", { "data-cp3": "add-role", "data-role": role2, style: "font-size:12px;color:var(--text-1);margin:6px 0 8px" }, [
          `${roleChoice(role2).label}`,
        ])
      : null;

    const nameInput = el("input", { class: "inp", value: p0.display_name || "", "aria-label": "Display name", placeholder: "e.g. Local reasoning model" });
    const urlInput = el("input", { class: "inp", value: p0.base_url || "", placeholder: "https://…/v1", "aria-label": "Base URL" });
    const urlHint = el("div", { "data-cp3": "url-hint", style: "font-size:11px;color:var(--text-1);margin-top:4px" }, [urlExample(role2)]);
    // write-only secret: never prefill the stored key
    const keyInput = el("input", { class: "inp", type: "password", autocomplete: "off", value: "", placeholder: creating ? "optional" : "unchanged", "aria-label": "API key" });
    const modelInput = el("input", { class: "inp", value: p0.model || "", "aria-label": "Model" });
    const timeoutInput = el("input", { class: "inp", type: "number", min: "1", step: "1", value: p0.request_timeout_s != null ? p0.request_timeout_s : 60, "aria-label": "Timeout seconds" });
    const langSel = el("select", { class: "inp", "aria-label": "Language as" },
      LANGUAGE_FORMAT_CHOICES.map((c) => el("option", { value: c.value, text: c.label })));
    langSel.value = p0.language_format || "code";
    const voiceInput = el("input", { class: "inp", value: p0.voice || "", "aria-label": "Voice (backend label)" });
    const caps = p0.capabilities || {};
    const voicesInput = el("input", { class: "inp", value: (caps.voices || []).join(", "), "aria-label": "Claimed voices" });
    const maxRefInput = el("input", { class: "inp", type: "number", min: "0", step: "1", value: (caps.max_reference_ms || 0) / 1000, "aria-label": "Longest recording seconds" });
    const storedMode = voiceModeFromCaps(caps);
    const modeSel = el("select", { class: "inp", "aria-label": "Voices" }, VOICE_MODE_CHOICES.map((c) => el("option", { value: c.value, text: c.text })));
    modeSel.value = storedMode;
    // R2 / CP3: whether this engine's native voice labels are a durable
    // identity. Replaces the old hard-coded model list with row data.
    const trustSel = el("select", { class: "inp", "aria-label": "Native voice labels" }, [
      el("option", { value: "unknown", text: "not checked (names are not offered)" }),
      el("option", { value: "honoured", text: "yes — another name, another voice" }),
      el("option", { value: "ignored", text: "no — the server ignores the name" }),
    ]);
    trustSel.value = caps.native_voice_labels || "unknown";
    // F13: model-thinking option (LLM rows only). It maps to
    // extra.chat_template_kwargs.enable_thinking; "backend default" stores no key.
    const thinkingSel = el("select", { class: "inp", "aria-label": "Model thinking" }, [
      el("option", { value: "default", text: "Model default" }),
      el("option", { value: "off", text: "Off (answer only; saves tokens)" }),
      el("option", { value: "on", text: "On (thinking; slower, uses budget)" }),
    ]);
    thinkingSel.value = thinkingModeFromExtra(p0.extra);
    const extras = el("textarea", { class: "inp", rows: 3, spellcheck: false, "aria-label": "Extra request fields", placeholder: "{ }" });
    extras.value = JSON.stringify(p0.extra || {}, null, 2);
    const refClear = el("input", { type: "checkbox" });
    const refAudioField = el("input", { class: "inp mono", value: p0.reference_audio_field || "ref_audio", "aria-label": "Recording field" });
    const refTextField = el("input", { class: "inp mono", value: p0.reference_text_field || "ref_text", "aria-label": "Recording text field" });
    const refEncoding = el("select", { class: "inp", "aria-label": "Recording sent as" }, [
      el("option", { value: "base64", text: "Plain base64" }),
      el("option", { value: "data_uri", text: "Data URL (data:audio/wav;base64,…)" }),
    ]);
    refEncoding.value = p0.reference_audio_encoding || "base64";
    const audioFormat = el("select", { class: "inp", "aria-label": "Answer format" }, [
      el("option", { value: "wav", text: "WAV" }),
      el("option", { value: "pcm", text: "Raw PCM (16-bit mono)" }),
      el("option", { value: "mp3", text: "MP3" }),
    ]);
    audioFormat.value = p0.audio_format || "wav";
    const headStart = el("input", { class: "inp", type: "number", min: "0", max: "5000", step: "50", value: p0.start_buffer_ms || 0, "aria-label": "Head start ms" });
    const pcmRate = el("input", { class: "inp", type: "number", min: "8000", step: "1", value: p0.pcm_sample_rate || 24000, "aria-label": "PCM sample rate" });

    const modelListBox = el("div", { class: "fld", style: "grid-column:1/-1" });

    // one section per question, the same for every kind of model:
    // where it runs, what it does, and what is rarely needed
    const section = (title, nodes, lede) =>
      el("div", { class: "fsection", style: "grid-column:1/-1" }, [
        el("div", { class: "fsection-title", text: title }),
        lede ? el("div", { class: "fhelp", text: lede }) : null,
        el("div", { class: "grid2" }, nodes.filter(Boolean)),
      ]);
    // a recording stored in the entry itself, under the usual or its own field names
    const storedRefNames = [...new Set(["ref_audio", "ref_text", p0.reference_audio_field, p0.reference_text_field].filter(Boolean))];
    const hasStoredRecording = !!(p0.extra && storedRefNames.some((n) => p0.extra[n]));

    // the voice fields that apply depend on the answer to "what does it do
    // with voices"; the others are hidden, not cleared
    const voicesField = field("Its own voices", voicesInput, MODEL_FIELD_HELP.voices);
    const voiceHelp = el("div", { class: "fhelp" });
    const voiceField = field("Default voice", voiceInput, voiceHelp);
    const maxRefField = field("Longest recording it uses (s)", maxRefInput, MODEL_FIELD_HELP.max_reference_s);
    const refFieldsField = field(
      "Recording fields",
      el("div", { style: "display:flex;gap:8px" }, [refAudioField, refTextField]),
      MODEL_FIELD_HELP.reference_fields,
    );
    const refEncodingField = field("Recording sent as", refEncoding, MODEL_FIELD_HELP.reference_encoding);
    const pcmRateField = field("PCM sample rate", pcmRate, MODEL_FIELD_HELP.pcm_sample_rate);
    function showPcmRate() {
      pcmRateField.classList.toggle("hidden", audioFormat.value !== "pcm");
    }
    audioFormat.addEventListener("change", () => { dirty = true; showPcmRate(); });
    showPcmRate();
    const trustField = field("Voice names are reliable", trustSel, MODEL_FIELD_HELP.native_trust);
    function showVoiceFields() {
      const mode = modeSel.value;
      voicesField.classList.toggle("hidden", mode !== "builtin" && mode !== "unknown");
      maxRefField.classList.toggle("hidden", mode !== "clones");
      refFieldsField.classList.toggle("hidden", mode !== "clones");
      refEncodingField.classList.toggle("hidden", mode !== "clones");
      trustField.classList.toggle("hidden", mode !== "builtin" && mode !== "unknown");
      voiceField.querySelector("label").textContent = mode === "builtin" ? "Default voice" : "Voice name";
      voiceHelp.textContent = MODEL_FIELD_HELP.voice[mode] || MODEL_FIELD_HELP.voice.unknown;
    }
    modeSel.addEventListener("change", () => {
      // a suggestion the person sees and can change under Advanced
      if (VOICE_MODE_TRUST[modeSel.value]) trustSel.value = VOICE_MODE_TRUST[modeSel.value];
      showVoiceFields();
    });
    showVoiceFields();

    const doesNodes =
      role2 === "stt"
        ? [field("Language as", langSel, MODEL_FIELD_HELP.language_format)]
        : role2 === "llm"
          ? [el("div", { "data-f13": "thinking-hint" }, [field("Model thinking", thinkingSel, MODEL_FIELD_HELP.thinking)])]
          : [
              field("Voices", modeSel, MODEL_FIELD_HELP.voice_mode),
              voicesField,
              voiceField,
              maxRefField,
              refFieldsField,
              refEncodingField,
              field("Language as", langSel, MODEL_FIELD_HELP.language_format),
            ];
    const doesTitle = { stt: "What it hears", llm: "How it answers", tts: "How it speaks" }[role2];

    const advanced = el("details", { class: "fsection", style: "grid-column:1/-1" }, [
      el("summary", { class: "fsection-title", text: "Advanced" }),
      el("div", { class: "grid2" }, [
        field("Timeout (s)", timeoutInput, MODEL_FIELD_HELP.timeout),
        role2 === "tts" ? trustField : null,
        role2 === "tts" ? field("Answer format", audioFormat, MODEL_FIELD_HELP.audio_format) : null,
        role2 === "tts" ? pcmRateField : null,
        role2 === "tts" ? field("Head start (ms)", headStart, MODEL_FIELD_HELP.start_buffer) : null,
        el("div", { class: "fld", style: "grid-column:1/-1" }, [
          el("label", { text: "Extra request fields (JSON)" }),
          extras,
          el("div", { class: "fhelp", text: MODEL_FIELD_HELP.extra[role2] }),
        ]),
        role2 === "tts" && hasStoredRecording
          ? el("div", { class: "fld", style: "grid-column:1/-1" }, [
              el("label", {
                style: "display:flex;align-items:center;gap:8px;text-transform:none;letter-spacing:0;font-size:12px",
              }, [refClear, "Remove the recording stored in this model on save"]),
              el("div", { class: "fhelp", text: MODEL_FIELD_HELP.ref_clear }),
            ])
          : null,
      ].filter(Boolean)),
    ]);

    const form = el("div", { class: "grid2", style: "margin-top:8px", "data-cp3": "connection-form" }, [
      editBanner,
      addBanner,
      section("Where it runs", [
        field("Display name", nameInput, MODEL_FIELD_HELP.name),
        field("Server address", urlInput, el("div", {}, [el("div", { class: "fhelp", text: MODEL_FIELD_HELP.base_url }), urlHint])),
        field(creating ? "Key" : "Key (empty keeps the stored one)", keyInput, creating ? MODEL_FIELD_HELP.api_key : MODEL_FIELD_HELP.api_key_edit),
        field("Model name", modelInput, MODEL_FIELD_HELP.model),
      ]),
      modelListBox,
      section(doesTitle, doesNodes),
      advanced,
    ]);

    nameInput.addEventListener("input", () => { dirty = true; });
    for (const node of [urlInput, keyInput, modelInput, voiceInput, voicesInput, extras, refAudioField, refTextField, pcmRate, headStart]) {
      node.addEventListener("input", () => { dirty = true; });
    }
    for (const node of [timeoutInput, maxRefInput]) node.addEventListener("input", () => { dirty = true; });
    for (const node of [modeSel, trustSel, thinkingSel, langSel, refEncoding]) node.addEventListener("change", () => { dirty = true; });

    // ---- step 3: check the UNSAVED form ----------------------------------
    const checkBtn = el("button", { class: "btn", type: "button", "data-cp3": "check", text: "Check connection" });
    const checkResult = el("div", { "data-cp3": "check-result", style: "font-size:12px;margin-top:6px;line-height:1.5" });
    checkResult.hidden = true;
    const modelListRow = el("div", { style: "margin-top:6px" });
    checkBtn.addEventListener("click", async () => {
      const profilePayload = collectPayload({ includeId: true });
      if (!profilePayload.base_url) {
        checkResult.hidden = false;
        checkResult.textContent = "Enter an endpoint first.";
        checkResult.style.color = "var(--err, #c33)";
        return;
      }
      checkResult.hidden = false;
      checkResult.textContent = "Checking the connection…";
      checkResult.style.color = "var(--text-1)";
      try {
        const res = await catalogApi.probe(role, creating ? "" : p0.id, profilePayload);
        clear(modelListRow);
        renderModelList(modelListRow, (res && res.models) || [], modelInput, checkResult, res);
        if (res && res.ok) {
          checkResult.textContent = `Reachable. ${((res.models || []).length)} model(s) reported${res.detail ? ` (${res.detail})` : ""}.`;
          checkResult.style.color = "var(--ok, #2a7)";
        } else if (res && res.error_code === "key_required_for_changed_endpoint") {
          // C1: the stored key is never sent to a changed/new endpoint. Say so
          // plainly and point at the API key field.
          checkResult.textContent =
            "The endpoint changed, so the stored API key was not sent. " +
            "Enter the API key to check this endpoint. You can still save — a failed check is not a block.";
          checkResult.style.color = "var(--err, #c33)";
        } else {
          const detail = (res && (res.detail || res.error_code)) || "unknown reason";
          checkResult.textContent = `Check failed: ${detail}. You can still save — a failed check is not a block.`;
          checkResult.style.color = "var(--err, #c33)";
        }
      } catch (err) {
        checkResult.textContent = `Check failed: ${err.message}. You can still save — a failed check is not a block.`;
        checkResult.style.color = "var(--err, #c33)";
      }
    });
    modelListBox.append(
      el("label", { text: "Check the connection" }),
      el("div", { style: "font-size:11px;color:var(--text-1);margin-bottom:4px" }, ["Asks the server with what is on this form, before you save: is it reachable, does the key work, which model names does it offer. Nothing is stored."]),
      el("div", { class: "inline-actions", style: "display:flex;gap:8px;flex-wrap:wrap" }, [checkBtn]),
      modelListRow,
      checkResult,
    );

    // ---- save ------------------------------------------------------------
    const saveError = el("div", { "data-cp3": "save-error", style: "font-size:12px;margin-top:6px" });
    saveError.hidden = true;
    const saveBtn = el("button", { class: "btn pri", type: "button", text: creating ? "Create backend" : "Save changes" });
    const cancelBtn = el("button", { class: "btn", type: "button", text: "Cancel" });
    const delBtn = creating ? null : el("button", { class: "btn ghost", type: "button", text: "Delete model" });
    const dupBtn = creating ? null : el("button", { class: "btn", type: "button", text: "Duplicate" });
    const actions = [saveBtn, cancelBtn];
    if (dupBtn) actions.push(dupBtn);
    if (delBtn) actions.push(delBtn);
    form.append(el("div", { class: "head-actions", style: "grid-column:1/-1" }, actions), saveError);

    function collectPayload({ includeId }) {
      let extra;
      try {
        extra = extras.value.trim() ? JSON.parse(extras.value) : {};
      } catch {
        throw new Error("Extra fields must be valid JSON.");
      }
      if (role2 === "llm") extra = extraWithThinking(extra, thinkingSel.value);
      const payload = {
        role,
        display_name: nameInput.value,
        base_url: urlInput.value,
        model: modelInput.value,
        language_format: langSel.value,
        voice: voiceInput.value,
        ...(role2 === "tts"
          ? {
              reference_audio_field: refAudioField.value.trim() || "ref_audio",
              reference_text_field: refTextField.value.trim() || "ref_text",
              reference_audio_encoding: refEncoding.value,
              audio_format: audioFormat.value,
              pcm_sample_rate: Math.round(Number(pcmRate.value) || 24000),
              start_buffer_ms: Math.max(0, Math.round(Number(headStart.value) || 0)),
            }
          : {}),
        request_timeout_s: Number(timeoutInput.value) || 60,
        extra,
        capabilities: capsForVoiceMode(modeSel.value, {
          voices: voicesInput.value.split(",").map((s) => s.trim()).filter(Boolean),
          max_reference_ms: Math.round((Number(maxRefInput.value) || 0) * 1000),
          native_voice_labels: trustSel.value,
        }),
      };
      if (!creating) payload.id = p0.id;
      if (keyInput.value !== "") payload.api_key = keyInput.value;
      if (refClear.checked) {
        payload.extra = { ...extra };
        for (const n of storedRefNames) payload.extra[n] = null;
      }
      return payload;
    }

    function showSaveError(text) {
      saveError.hidden = false;
      saveError.textContent = text;
      saveError.style.color = "var(--err, #c33)";
    }

    saveBtn.addEventListener("click", async () => {
      saveError.hidden = true;
      if (!nameInput.value.trim()) {
        showSaveError("A display name is required.");
        return;
      }
      if (!urlInput.value.trim()) {
        showSaveError("An endpoint / base URL is required.");
        return;
      }
      if (!modelInput.value.trim()) {
        showSaveError("A model is required.");
        return;
      }
      let payload;
      try {
        payload = collectPayload({ includeId: true });
      } catch (err) {
        showSaveError(err.message);
        return;
      }
      setStatus("Saving…", "");
      try {
        if (creating) {
          // create is POST-only; never a silent PUT
          const saved = await catalogApi.createProfile(role, payload);
          editing = saved;
          formMode = "edit";
          dirty = false;
          await ctx.refreshCatalog();
          fillProfileSelect(saved.id);
          setStatus("Model added. Sets can now use it.", "ok");
        } else {
          let saved;
          try {
            saved = await catalogApi.saveProfile(role, payload.id, payload);
          } catch (err) {
            // C3: a changed endpoint with a blank key is refused. Ask the
            // operator to either enter a key or explicitly keep the stored one
            // for the NEW endpoint. Never keep it silently.
            if (err.code !== "key_required_for_changed_endpoint") throw err;
            const keep = await confirmDialog({
              title: "The endpoint changed",
              message: changedEndpointKeyPrompt(payload.base_url),
              detail: changedEndpointKeyDetail(),
              confirmLabel: "Keep existing key",
              cancelLabel: "Enter a new key",
              danger: true,
            });
            if (!keep) {
              keyInput.focus();
              setStatus("Enter the API key for the new endpoint, then save again.", "err");
              return;
            }
            saved = await catalogApi.saveProfile(role, payload.id, {
              ...payload,
              keep_api_key_for_new_endpoint: true,
            });
          }
          editing = saved;
          dirty = false;
          await ctx.refreshCatalog();
          fillProfileSelect(saved.id);
          setStatus(
            (saved.used_by || []).length
              ? `Saved. ${(saved.used_by || []).length === 1 ? "The set that uses it" : "The sets that use it"} use the change from the next call.`
              : "Saved.",
            "ok",
          );
        }
      } catch (err) {
        if (err.code === "breaks_sets") {
          const names = (err.issues || []).map((b) => b.display_name || b.profile).join(", ");
          setStatus(`Not saved: ${names} would stop working with this change. Change those sets first, or switch them off.`, "err");
        } else {
          setStatus("Could not save: " + err.message, "err");
        }
      }
    });

    cancelBtn.addEventListener("click", async () => {
      if (!(await confirmDiscard())) return;
      dirty = false;
      fillProfileSelect();
      setStatus("", "");
    });

    if (dupBtn) {
      dupBtn.addEventListener("click", async () => {
        if (!editing) return;
        const values = await formDialog({
          title: `Duplicate ${editing.display_name || editing.model}`,
          description: "The copy gets the same server address, model and stored key server-side. Nothing is typed into the browser.",
          fields: [
            { name: "display_name", label: "Name of the copy", value: `${editing.display_name || editing.model} (copy)`, required: true },
          ],
          confirmLabel: "Duplicate",
          validate: (v) => (v.display_name.trim() ? "" : "A name is required."),
        });
        if (!values) return;
        setStatus("Duplicating…", "");
        try {
          const saved = await catalogApi.cloneProfile(role, editing.id, { display_name: values.display_name.trim() });
          editing = saved;
          dirty = false;
          await ctx.refreshCatalog();
          fillProfileSelect(saved.id);
          setStatus("Duplicated.", "ok");
        } catch (err) {
          setStatus("Could not duplicate: " + err.message, "err");
        }
      });
    }

    if (delBtn) {
      delBtn.addEventListener("click", async () => {
        if (!editing) return;
        const ok = await confirmDialog({
          title: `Delete the model ${editing.display_name || editing.model}?`,
          message: "This cannot be undone.",
          confirmLabel: "Delete model",
          danger: true,
        });
        if (!ok) return;
        try {
          await catalogApi.deleteProfile(role, editing.id);
          editing = null;
          dirty = false;
          await ctx.refreshCatalog();
          fillProfileSelect();
          setStatus("Deleted.", "ok");
        } catch (err) {
          setStatus("Could not delete: " + err.message, "err");
        }
      });
    }

    formHost.append(form);
  }

  /* Render the reported model list and let the user pick from it. */
  function renderModelList(host, models, modelInput, checkResult, res) {
    const list = el("select", { class: "inp", "aria-label": "Available models", style: "max-width:340px" });
    if (!models.length) {
      list.append(el("option", { value: "", text: res && res.detail ? `no models (${res.detail})` : "no models reported" }));
      host.append(el("div", { class: "inline-actions", style: "display:flex;gap:8px;margin-top:6px;flex-wrap:wrap" }, [list]));
      return;
    }
    list.append(el("option", { value: "", text: "Pick a model from the list…" }));
    for (const m of models) list.append(el("option", { value: m, text: m }));
    if (modelInput.value && models.includes(modelInput.value)) list.value = modelInput.value;
    list.addEventListener("change", () => {
      if (list.value) modelInput.value = list.value;
    });
    host.append(el("div", { class: "inline-actions", style: "display:flex;gap:8px;margin-top:6px;flex-wrap:wrap" }, [list]));
  }

  /* ----------------------------------------------------------- edit view */

  function fillForm() {
    clear(formHost);
    if (formMode === "add") {
      // caller already built the step
      return;
    }
    if (editing) {
      renderEditStep();
    } else {
      formTitle.textContent = "";
    }
  }

  roleSel.addEventListener("change", async () => {
    if (roleSel.value === role) return; // no-op: never prompt for the same role
    if (!(await confirmDiscard())) {
      roleSel.value = role;
      return;
    }
    role = roleSel.value;
    editing = null;
    formMode = "none";
    dirty = false;
    fillProfileSelect();
  });
  profSel.addEventListener("change", async () => {
    if (editing && profSel.value === editing.id) return; // no-op
    if (!(await confirmDiscard())) {
      if (editing) profSel.value = editing.id;
      return;
    }
    editing = roleProfiles().find((p) => p.id === profSel.value) || null;
    formMode = editing ? "edit" : "none";
    dirty = false;
    fillForm();
  });
  addBtn.addEventListener("click", async () => {
    if (!(await confirmDiscard())) return;
    renderRoleStep();
    setStatus("New backend — choose what it should do first.", "");
  });

  /* The board: one column per role, a card per model with what it can do,
   * where it runs, how many sets use it and whether it answers. A click opens
   * it in the form below; "＋ Add" starts the add steps for that role. */
  const board = el("div", { class: "mboard", "data-models": "board" });
  const health = new Map(); // "role/id" -> { ok, detail } | "checking"
  const ROLE_TITLES = { stt: ["Listens", "speech recognition"], llm: ["Thinks", "language model"], tts: ["Speaks", "speech output"] };
  function usedBy(r, id) {
    const key = `${r}_profile`;
    return (ctx.profiles || []).filter((pr) => pr.spec && pr.spec[key] === id).length;
  }
  function host(url) {
    try {
      return new URL(url).host;
    } catch {
      return url || "no address";
    }
  }
  function dot(r, id) {
    const h = health.get(`${r}/${id}`);
    if (h === "checking") return el("span", { class: "sdot dot-unknown", title: "Checking…" });
    if (!h) return el("span", { class: "sdot dot-unknown", title: "Not checked — press Check all" });
    return el("span", { class: "sdot " + (h.ok ? "dot-ready" : "dot-warn"), title: h.ok ? "Answers" : `Does not answer: ${h.detail || ""}` });
  }
  function card(r, p) {
    const n = usedBy(r, p.id);
    const badges = [];
    if (r === "tts") {
      // this row's own capabilities: rows of one engine may differ
      const ability = voiceAbility(null, p.capabilities || {});
      badges.push(el("span", { class: "mbadge " + ability, text: VOICE_ABILITY_LABEL[ability] }));
    }
    if (p.language_format === "name") badges.push(el("span", { class: "mbadge", text: "language as name" }));
    if (p.language_format === "none") badges.push(el("span", { class: "mbadge", text: "no language" }));
    const selected = editing && editing.id === p.id && role === r && formMode === "edit";
    const c = el("button", { class: "mcard" + (selected ? " on" : ""), type: "button", "data-model": `${r}/${p.id}` }, [
      el("div", { class: "mtop" }, [dot(r, p.id), el("span", { class: "mname", text: p.display_name || p.id })]),
      el("div", { class: "mmodel mono", text: p.model || "no model set" }),
      el("div", { class: "mmeta", text: `${host(p.base_url)} · ${n === 0 ? "not used" : n === 1 ? "used by 1 set" : `used by ${n} sets`}` }),
      badges.length ? el("div", { class: "mbadges" }, badges) : null,
    ]);
    if (r === "tts" && (p.capabilities || {}).voices && p.capabilities.voices.length && voiceAbility(null, p.capabilities) === "builtin") {
      c.title = "Voices: " + p.capabilities.voices.join(", ");
    }
    c.addEventListener("click", async () => {
      if (!(await confirmDiscard())) return;
      role = r;
      roleSel.value = r;
      fillProfileSelect(p.id);
      renderBoard();
      formTitle.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return c;
  }
  function renderBoard() {
    clear(board);
    for (const r of ["stt", "llm", "tts"]) {
      const [title, sub] = ROLE_TITLES[r];
      const col = el("div", { class: "mcol", "data-role": r });
      const add = el("button", { class: "btn sm", type: "button", text: "＋ Add", title: `Add a ${sub} model` });
      add.addEventListener("click", async () => {
        if (!(await confirmDiscard())) return;
        role = r;
        roleSel.value = r;
        renderConnectionStep();
        setStatus(`New ${sub} model — fill in where it runs, then check the connection.`, "");
        renderBoard();
        formTitle.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      col.append(el("div", { class: "mcolhead" }, [el("div", {}, [el("div", { class: "mcoltitle", text: title }), el("div", { class: "mcolsub", text: sub })]), add]));
      const rows = ctx.roleProfiles(r);
      if (!rows.length) col.append(el("div", { class: "fhint", text: "None yet." }));
      for (const p of rows) col.append(card(r, p));
      board.append(col);
    }
  }
  let checkGen = 0;
  async function checkAll() {
    // a newer check (after a save) wins: an older probe never paints over it
    const gen = ++checkGen;
    health.clear();
    const all = ["stt", "llm", "tts"].flatMap((r) => ctx.roleProfiles(r).map((p) => [r, p.id]));
    for (const [r, id] of all) health.set(`${r}/${id}`, "checking");
    renderBoard();
    await Promise.all(
      all.map(async ([r, id]) => {
        let result;
        try {
          const res = await catalogApi.probe(r, id);
          result = { ok: !!res.ok, detail: res.detail || res.error_code || "" };
        } catch (err) {
          result = { ok: false, detail: err.message };
        }
        if (gen !== checkGen) return;
        health.set(`${r}/${id}`, result);
        renderBoard();
      }),
    );
  }
  const checkBtn = el("button", { class: "btn", type: "button", text: "Check all", title: "Ask every model whether it answers" });
  checkBtn.addEventListener("click", () => checkAll());

  pane.append(
    el("div", { style: "display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap" }, [
      el("div", { style: "font-size:var(--fs-meta);color:var(--text-1);line-height:1.6" }, [
        "A speaking model either clones voices from a recording, has built-in voices, or speaks with one voice — the set picks the voice to match.",
      ]),
      checkBtn,
    ]),
    board,
    formTitle,
    formHost,
    status,
  );
  // nothing open until a card is clicked; the board checks every model at once
  clear(formHost);
  formTitle.textContent = "";
  editing = null;
  formMode = "none";
  renderBoard();
  void checkAll();
  // every save, duplicate or delete refreshes the catalog: redraw the board
  const refresh = ctx.refreshCatalog;
  ctx.refreshCatalog = async (...args) => {
    const out = await refresh(...args);
    // a saved, added or removed model is checked again, all of them with it
    renderBoard();
    void checkAll();
    return out;
  };
}

function urlExample(role) {
  if (role === "stt") return "Example: http://your-host:8000/v1";
  if (role === "llm") return "Example: http://your-host:8000/v1, or https://api.openai.com/v1";
  return "Example: http://your-host:8000/v1";
}
