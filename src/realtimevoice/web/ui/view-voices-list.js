/* Voices: the user's recorded voices (cloned by a speaking model), and the
 * voices built into speaking models. A set picks one of either. */

import { el, waveform } from "./dom.js";
import { api, voices as voicesApi, getToken } from "./api.js";
import { sharedSamplePlayer, registerSampleSource, unregisterSampleSource } from "./sample-player.js";
import { formDialog, noticeDialog } from "./dialogs.js";
import { ID_PATTERN, idError } from "./format.js";

const PREVIEW_SENTENCE = "Guten Morgen, schön, dass du da bist.";

/** A play/pause button bound to the one shared playback controller (F6). */
function referencePlayButton(voice, ref, ctx) {
  const key = `${voice.id}:${ref.id}`;
  const btn = el("button", {
    class: "splay",
    type: "button",
    text: "▶",
    "aria-label": "Play reference sample",
    "aria-pressed": "false",
    "data-rtv-sample": key,
    title: ref ? "Play the reference sample" : "No sample to play",
  });
  if (!ref) {
    btn.disabled = true;
    return btn;
  }
  registerSampleSource(key, async () => {
    const resp = await fetch(voicesApi.sampleAudioUrl(voice.id, ref.id), {
      headers: { Authorization: "Bearer " + getToken() },
    });
    if (!resp.ok) throw new Error("audio not available");
    return resp.blob();
  });
  btn._rtvUpdate = () => {
    const playing = sharedSamplePlayer.stateFor(key) === "playing";
    btn.textContent = playing ? "❚❚" : "▶";
    btn.setAttribute("aria-pressed", String(playing));
    btn.setAttribute("aria-label", playing ? "Pause" : "Play reference sample");
  };
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await sharedSamplePlayer.toggle(key);
    document.querySelectorAll("button[data-rtv-sample]").forEach((b) => b._rtvUpdate && b._rtvUpdate());
  });
  btn.addEventListener("keydown", (e) => e.stopPropagation());
  return btn;
}

export async function renderVoiceLibrary(inner, ctx) {
  const voices = await ctx.loadVoices();
  // F6: stop any sample playback when leaving the list
  if (ctx.setCleanup) {
    ctx.setCleanup(() => {
      sharedSamplePlayer.stop();
      for (const voice of voices) {
        for (const s of voice.samples || []) unregisterSampleSource(`${voice.id}:${s.id}`);
      }
    });
  }

  inner.append(
    ctx.shell.breadcrumb([{ label: "Voices" }]),
    el("div", { class: "head" }, [
      el("div", {}, [
        el("h1", { class: "h1", text: "Voices" }),
        el("p", {
          class: "lede",
          text:
            "Your voices are recordings a speaking model clones. A voice has one recording in use; " +
            "every set that speaks with the voice uses it from the next call. Below them: the voices built into your speaking models.",
        }),
      ]),
      el("div", { class: "head-actions" }, [
        el("button", { class: "btn pri", type: "button", text: "＋ New voice", on: { click: () => newVoice(ctx) } }),
      ]),
    ]),
  );

  sharedSamplePlayer.onChange = () => {
    document.querySelectorAll("button[data-rtv-sample]").forEach((b) => b._rtvUpdate && b._rtvUpdate());
  };
  inner.append(el("div", { class: "seclabel" }, [el("span", { text: "Your voices" })]));
  if (voices.length === 0) {
    inner.append(
      el("div", { class: "empty" }, [
        el("div", { class: "et", text: "No voices yet" }),
        el("p", { class: "es", text: "Create a voice and record or upload a sample. The sample is transcribed automatically; you only check the text afterwards." }),
        el("button", { class: "btn pri", type: "button", text: "＋ New voice", on: { click: () => newVoice(ctx) } }),
      ]),
    );
  } else {
    inner.append(ownVoicesGrid(voices, ctx));
  }
  renderBuiltIn(inner, ctx);
}

function ownVoicesGrid(voices, ctx) {
  const grid = el("div", { class: "vgrid" });
  for (const voice of voices) {
    const samples = voice.samples || [];
    const ref = samples.find((s) => s.id === voice.reference_sample_id);
    const usedBy = (voice.used_by || []).map((u) => u.display_name || u.profile);
    const card = el("div", {
      class: "vcard",
      role: "button",
      tabindex: 0,
      on: {
        click: () => ctx.go(`#/voices/${encodeURIComponent(voice.id)}`),
        keydown: (e) => {
          if (e.key === "Enter" || e.key === " ") ctx.go(`#/voices/${encodeURIComponent(voice.id)}`);
        },
      },
    });
    const playBtn = referencePlayButton(voice, ref, ctx);
    card.append(
      el("div", { style: "display:flex;align-items:center;gap:10px" }, [
        playBtn,
        waveform(voice.id, 6, 18),
        el("div", {}, [
          el("div", { class: "pname", text: voice.display_name || voice.id }),
          el("div", {
            class: "pdesc",
            text: `${samples.length} recording${samples.length === 1 ? "" : "s"} · ${
              !ref ? "none in use yet" : ref.transcript_confirmed ? "one in use" : "text of the recording not confirmed"
            }`,
          }),
        ]),
      ]),
    );
    if (usedBy.length) {
      card.append(el("div", { class: "usedin" }, ["Sets: ", el("b", { text: [...new Set(usedBy)].join(", ") })]));
    }
    grid.append(card);
  }
  return grid;
}

/** A button that speaks a short sentence with a model's built-in voice. */
function builtInPlayButton(row, name) {
  const key = `builtin:${row.id}:${name}`;
  const btn = el("button", {
    class: "splay",
    type: "button",
    text: "▶",
    "aria-label": `Hear ${name}`,
    "aria-pressed": "false",
    "data-rtv-sample": key,
  });
  registerSampleSource(key, async () => {
    const resp = await api.raw("/api/tts-synthesis", {
      method: "POST",
      body: { profile_id: row.id, voice: name, text: PREVIEW_SENTENCE, language: "de" },
    });
    return resp.blob();
  });
  btn._rtvUpdate = () => {
    const playing = sharedSamplePlayer.stateFor(key) === "playing";
    btn.textContent = playing ? "❚❚" : "▶";
    btn.setAttribute("aria-pressed", String(playing));
  };
  btn.addEventListener("click", async () => {
    await sharedSamplePlayer.toggle(key);
    document.querySelectorAll("button[data-rtv-sample]").forEach((b) => b._rtvUpdate && b._rtvUpdate());
  });
  return btn;
}

function renderBuiltIn(inner, ctx) {
  const rows = ctx
    .roleProfiles("tts")
    .filter((r) => {
      const caps = r.capabilities || {};
      return caps.voice_cloning !== "reference_audio" && caps.native_voice_labels === "honoured" && (caps.voices || []).length;
    });
  if (!rows.length) return;
  inner.append(el("div", { class: "seclabel" }, [el("span", { text: "Built into a model" })]));
  for (const row of rows) {
    const grid = el("div", { class: "vgrid" });
    for (const name of row.capabilities.voices) {
      grid.append(
        el("div", { class: "vcard" }, [
          el("div", { style: "display:flex;align-items:center;gap:10px" }, [
            builtInPlayButton(row, name),
            waveform(`${row.id}:${name}`, 6, 18),
            el("div", {}, [
              el("div", { class: "pname", text: name }),
              el("div", { class: "pdesc", text: row.display_name || row.model }),
            ]),
          ]),
        ]),
      );
    }
    inner.append(grid);
  }
  inner.append(
    el("p", {
      class: "fhint",
      text:
        "The play button speaks a short German sentence with the voice. Which voices a model has is set on the model (Models › How it speaks).",
    }),
  );
}

async function newVoice(ctx) {
  const values = await formDialog({
    title: "New voice",
    description: "Give the voice an id and a display name. The sample is added on the next screen.",
    fields: [
      { name: "id", label: "Voice id", placeholder: "anna", required: true, hint: "lowercase letters, digits and dashes" },
      { name: "display_name", label: "Display name", placeholder: "Anna" },
    ],
    confirmLabel: "Create voice",
    validate: (v) => {
      if (!v.id.trim()) return "An id is required.";
      if (!ID_PATTERN.test(v.id.trim().toLowerCase())) return idError();
      return "";
    },
  });
  if (!values) return;
  const trimmed = values.id.trim().toLowerCase();
  const displayName = (values.display_name || "").trim() || trimmed;
  try {
    await voicesApi.create({ id: trimmed, display_name: displayName });
    await ctx.loadVoices();
    ctx.go(`#/voices/${encodeURIComponent(trimmed)}`);
  } catch (err) {
    await noticeDialog({ title: "Could not create the voice", message: err.message, detail: "Nothing was created." });
  }
}