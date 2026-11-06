/* Settings: what belongs to the server rather than to one set.
 *
 *  - the set an app gets that names none (another app sending its own model
 *    name); none = such a call is refused
 *  - the listening model that transcribes voice recordings
 *  - defaults for calls: used only when neither the app nor the set says
 *    otherwise (somora always sends its own)
 */

import { el, errorBar } from "./dom.js";
import { catalog as catalogApi } from "./api.js";

function numOr(value, fallback) {
  const t = String(value).trim();
  if (t === "") return fallback;
  const n = Number(t);
  return Number.isFinite(n) ? n : fallback;
}

const help = (text) => el("div", { class: "fhelp", text });
const fld = (label, input, text) => el("div", { class: "fld" }, [el("label", { text: label }), input, text ? help(text) : null]);
const check = (input, label, text) =>
  el("div", { class: "fld" }, [
    el("label", { style: "display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;font-size:12px" }, [input, label]),
    help(text),
  ]);

export async function renderSettings(inner, ctx) {
  inner.append(
    ctx.shell.breadcrumb([{ label: "Settings" }]),
    el("div", { class: "head" }, [
      el("div", {}, [
        el("h1", { class: "h1", text: "Settings" }),
        el("p", { class: "lede", text: "What belongs to this server rather than to one set." }),
      ]),
    ]),
  );
  let settings;
  try {
    settings = await catalogApi.settings();
  } catch (err) {
    inner.append(errorBar([el("b", { text: "Could not load the settings. " }), err.message]));
    return;
  }
  const status = el("div", { class: "statusline", role: "status" });
  const sets = await ctx.loadProfiles();

  async function save(body, what) {
    try {
      settings = await catalogApi.saveSettings(body);
      status.textContent = `${what} saved.`;
      status.className = "statusline ok";
    } catch (err) {
      status.textContent = `Not saved: ${err.message}`;
      status.className = "statusline err";
    }
  }

  /* -------------------------------------------- set for other apps */
  const defaultSel = el("select", { class: "inp", "aria-label": "Set for other apps" });
  defaultSel.append(el("option", { value: "", text: "None — refuse such a call" }));
  for (const p of sets) {
    defaultSel.append(el("option", { value: p.id, text: `${p.display_name || p.id}${p.enabled ? "" : " (off)"}` }));
  }
  defaultSel.value = settings.default_set || "";
  defaultSel.addEventListener("change", () => save({ default_set: defaultSel.value }, "Set for other apps"));

  /* ------------------------------------------- transcription model */
  const sttSel = el("select", { class: "inp", "aria-label": "Transcribes recordings" });
  sttSel.append(el("option", { value: "", text: "— choose —" }));
  for (const r of ctx.roleProfiles("stt")) sttSel.append(el("option", { value: r.id, text: r.display_name || r.model || r.id }));
  sttSel.value = (settings.voice_library || {}).transcription_stt_profile || "";
  sttSel.addEventListener("change", () =>
    save({ voice_library: { transcription_stt_profile: sttSel.value } }, "Model that transcribes recordings"),
  );

  /* ------------------------------------------------ call defaults */
  const session = settings.session || {};
  const td = session.turn_detection || {};
  const instructions = el("textarea", { class: "inp", rows: 5, "aria-label": "Default instructions" });
  instructions.value = session.instructions || "";
  const tdType = el("select", { class: "inp", "aria-label": "Turn detection type" }, [
    el("option", { value: "server_vad", text: "This server, when you pause" }),
    el("option", { value: "none", text: "The app, by a button" }),
  ]);
  tdType.value = td.type || "server_vad";
  const num = (label, value, step) =>
    ({ label, input: el("input", { class: "inp", type: "number", step: step || "1", value: value != null ? value : "", "aria-label": label }) });
  const threshold = num("How loud speech must be", td.threshold_rms, "10");
  const prefix = num("Audio kept before speech (ms)", td.prefix_padding_ms);
  const silence = num("Pause that ends your turn (ms)", td.silence_duration_ms, "50");
  const speechCheck = el("select", { class: "inp", "aria-label": "Telling speech from noise" }, [
    el("option", { value: "model", text: "A speech model and the loudness" }),
    el("option", { value: "loudness", text: "The loudness only" }),
  ]);
  speechCheck.value = td.speech_check || "model";
  const interrupt = el("input", { type: "checkbox" });
  interrupt.checked = !!td.interrupt_response;
  const auto = el("input", { type: "checkbox" });
  auto.checked = !!td.create_response;

  function saveDefaults() {
    return save(
      {
        session: {
          instructions: instructions.value,
          turn_detection: {
            type: tdType.value,
            threshold_rms: numOr(threshold.input.value, 350),
            prefix_padding_ms: numOr(prefix.input.value, 200),
            silence_duration_ms: numOr(silence.input.value, 420),
            interrupt_response: interrupt.checked,
            create_response: auto.checked,
            speech_check: speechCheck.value,
          },
        },
      },
      "Defaults for calls",
    );
  }

  const section = (title, lede, nodes) =>
    el("div", { class: "card", style: "margin-top:14px" }, [
      el("div", { class: "ctitle", text: title }),
      lede ? el("p", { class: "fhint", text: lede }) : null,
      ...nodes,
    ]);

  inner.append(
    status,
    section(
      "Set for other apps",
      "somora names its set (the agent's voice). Another app may name none, or send its own model name; it then gets this set. With none chosen, such a call is refused, so a typo never lands on the wrong set unnoticed.",
      [fld("Set", defaultSel)],
    ),
    section(
      "Transcribing recordings",
      "When you add a voice recording, this listening model writes down what is said in it; you then check the text.",
      [fld("Listening model", sttSel)],
    ),
    section(
      "Defaults for calls",
      "How a call behaves when neither the app nor the set says otherwise. somora sends its own instructions and turn detection with every call, so these matter for other apps and the Test page. A change applies to new calls.",
      [
        fld("Instructions", instructions, "What the model is told about its role before the conversation starts, when the app sends no instructions of its own."),
        el("div", { class: "grid2" }, [
          fld("Who decides you have finished", tdType, "Usually this server: it listens and answers after a pause. With the app, the app says when your turn is over."),
          fld(threshold.label, threshold.input, "A sound louder than this counts as speech (350 by default). Raise it if room noise interrupts the answers; lower it if quiet speech is missed."),
          fld(prefix.label, prefix.input, "How much audio before the detected start is kept, so the first syllable is not cut off. 200 ms is a good value."),
          fld(silence.label, silence.input, "How long you must be quiet before it answers. Shorter answers faster but may cut in during a breath; 420 ms is a short pause."),
          fld(
            "Telling speech from noise",
            speechCheck,
            "With a speech model, a cough, a knock or a chair is not taken for speech, so it neither interrupts an answer nor turns into a word like “ja”. It runs on this server's CPU. The loudness only is the old way.",
          ),
        ]),
        el("div", { class: "grid2" }, [
          check(interrupt, "Stop the answer when you speak", "When you talk while it is answering, it stops and listens."),
          check(auto, "Answer as soon as you finish", "Off means the app asks for each answer itself."),
        ]),
        el("div", { class: "head-actions" }, [el("button", { class: "btn pri", type: "button", text: "Save defaults", on: { click: saveDefaults } })]),
      ],
    ),
  );
}
