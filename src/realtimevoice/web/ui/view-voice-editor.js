/* Voice editor (screen 03).
 *
 * 01 The recording — the reference the voice speaks with, and which sets
 *    that can be called still speak with an earlier one (a set keeps the
 *    recording it was saved with); one button brings them up to date.
 * 02 Replace the recording — free microphone recording (raw PCM via
 *    AudioWorklet, no DSP, level meter, clipping warning) OR WAV upload. Audio
 *    is stored at once; transcription never invents text. Confirming the
 *    text of the new recording makes it the one in use, in every live set.
 * 03 Other recordings — kept to switch back; remove (409 reasons shown, an
 *    earlier set version that speaks with it is removed only on request).
 * 04 Preview — against a TTS profile, shows the three server-side header
 *    measurements; last three previews stay comparable; never changes a
 *    set. A suggested reading text exists only as collapsed help and is
 *    never stored as a transcript.
 */

import { el, clear, errorBar, spinner } from "./dom.js";
import {
  fmtDuration,
  sampleMetaLine,
  sampleTitle,
  transcriptQuote,
  transcriptChips,
  transcriptionState,
  transcriptionErrorText,
  uploadErrorText,
  clippingPct,
  levelToDisplay,
  fmtSeconds,
  referenceFit,
  referenceTooLongText,
  referenceTooLongIssue,
  cloneEngineLabel,
  engineModelLabel,
  languageLabel,
  setIssueText,
} from "./format.js";
import { queryParam } from "./shell.js";
import { voices as voicesApi, ttsEngines as ttsEnginesApi, getToken } from "./api.js";
import { sharedSamplePlayer, registerSampleSource, unregisterSampleSource } from "./sample-player.js";
import { createMicPicker, watchDeviceChanges, openMic, stopStream } from "./mic-devices.js";
import { formDialog, confirmDialog, noticeDialog } from "./dialogs.js";

const SUGGESTIONS = [
  "Describe your commute to work.",
  "Tell the story of your last holiday.",
  "What did you cook this week?",
  "How was your day today?",
];

export async function renderVoiceEditor(inner, ctx, voiceId) {
  let voice;
  try {
    voice = await voicesApi.get(voiceId);
  } catch (err) {
    inner.append(
      ctx.shell.breadcrumb([{ label: "Voices", route: "#/voices" }, { label: voiceId }]),
      errorBar([el("b", { text: "Could not load the voice. " }), err.message, el("b", { text: " Nothing was changed." }), " Go back and try again."]),
    );
    return;
  }

  const usedByNames = () => [...new Set((voice.used_by || []).map((u) => u.display_name || u.profile))];

  const header = el("div", { class: "topbar-editor" });
  function renderHeader() {
    clear(header);
    const samples = voice.samples || [];
    const ref = samples.find((s) => s.id === voice.reference_sample_id);
    const refState = ref && ref.transcript_confirmed ? "recording ready" : "no confirmed recording";
    header.append(
      el("div", {}, [
        el("h1", { class: "h1", text: voice.display_name || voice.id }),
        el("div", { class: "h1sub", text: `${samples.length} recording${samples.length === 1 ? "" : "s"} · ${refState} · recorded in ${languageLabel(voice.language)}` }),
        usedByNames().length ? el("div", { class: "usedin" }, ["Sets: ", el("b", { text: usedByNames().join(", ") })]) : null,
      ]),
      el("div", { class: "head-actions" }, [
        el("button", { class: "btn ghost", type: "button", text: "Delete voice", on: { click: () => deleteVoice() } }),
        el("button", { class: "btn", type: "button", text: "Name & language…", on: { click: () => saveVoice() } }),
      ]),
    );
  }

  /* ---------------------------------------------------- block 01: record */

  const recordStage = el("div", { class: "stagebox hidden" });
  let recorder = null;

  function recordButtonCard() {
    const btn = el("button", { class: "recbtn idle", type: "button", "aria-label": "Start recording" });
    btn.append(el("i", { "aria-hidden": "true" }));
    btn.addEventListener("click", () => (recorder && recorder.recording ? recorder.stop() : startRecording()));
    return { btn };
  }
  const { btn: recBtn } = recordButtonCard();

  const micCard = el("div", { class: "inpbox mic" }, [
    recBtn,
    el("div", { class: "inpbtitle", text: "Speak freely" }),
    el("div", { class: "inpbsub", text: "Press record and talk about anything — your day, your work, a story. No script needed." }),
    el("div", { class: "limits", text: "5–30 seconds works best" }),
  ]);
  // F2: microphone selection (after permission labels become available)
  let selectedMic = "";
  let micPicker = null;

  function mountMicPicker() {
    createMicPicker({ label: "Recording microphone", onChange: (id) => { selectedMic = id; } }).then((picker) => {
      micPicker = picker;
      micCard.querySelector(".micpick")?.remove();
      micCard.append(picker.node);
      watchDeviceChanges(() => picker.refresh().catch(() => {}));
    });
  }

  // suggestions are collapsed help only — never a text to read, never stored
  const opt = el("details", { class: "opt" });
  opt.append(
    el("summary", { class: "optsum", text: "Need something to say?" }),
    el("ul", { style: "margin:8px 0 0;padding-left:18px;color:var(--text-1);font-size:12px;line-height:1.7" }, SUGGESTIONS.map((s) => el("li", { text: s }))),
  );
  micCard.append(opt);

  const fileInput = el("input", { type: "file", accept: "audio/wav,audio/x-wav,audio/*", class: "sr-only", id: "voice-file" });
  const uploadCard = el("div", { class: "inpbox" }, [
    el("div", { class: "dropglyph", "aria-hidden": "true", text: "+" }),
    el("div", { class: "inpbtitle", text: "Upload speech" }),
    el("div", { class: "inpbsub", text: "Drop a recording you already have, or click to browse." }),
    el("div", { class: "limits", text: "WAV · 1–30 s · 16–48 kHz · max 10 MB" }),
    fileInput,
  ]);
  uploadCard.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) uploadSample(fileInput.files[0]);
  });
  uploadCard.addEventListener("dragover", (e) => e.preventDefault());
  uploadCard.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) uploadSample(e.dataTransfer.files[0]);
  });

  /* -------------------------------------------------- transcription step */

  const transcriptStage = el("div");

  async function reloadVoice() {
    voice = await voicesApi.get(voiceId);
    await ctx.loadVoices();
    renderHeader();
    renderSamples();
    renderTranscriptStage();
    renderPreview();
  }

  // the recording being checked: the one just added on this page (until it
  // is the one in use), else the newest one whose text is not confirmed yet
  let checkingId = "";

  function renderTranscriptStage() {
    clear(transcriptStage);
    const samples = voice.samples || [];
    const isDone = (s) => s.id === voice.reference_sample_id && s.transcript_confirmed;
    let target = samples.find((s) => s.id === checkingId && !isDone(s));
    // the recording in use first: without a confirmed text no set can use it
    if (!target) target = samples.find((s) => s.id === voice.reference_sample_id && !s.transcript_confirmed);
    if (!target) target = samples.slice().reverse().find((s) => !s.transcript_confirmed && transcriptionState(s) !== "none");
    if (!target) return;
    transcriptStage.append(transcriptBlock(target));
  }

  function transcriptBlock(sample) {
    const state = transcriptionState(sample);
    const box = el("div", { class: "card" });
    // audio player
    box.append(
      el("div", { class: "aud" }, [
        playButton(sample, "play", "Play sample"),
        el("div", { class: "wave", "aria-hidden": "true" }, barRow(sample.sha256 || sample.id, sample.metrics && sample.metrics.waveform)),
        el("div", { class: "adur mono", text: fmtDuration(sample.duration_ms) }),
      ]),
    );

    if (state === "none") {
      box.append(el("div", { class: "tnote", text: "No transcript yet." }));
      box.append(
        el("div", { class: "trow" }, [
          el("button", { class: "btn pri sm", type: "button", text: "Transcribe", on: { click: () => transcribe(sample) } }),
          el("button", { class: "btn sm", type: "button", text: "Type it myself", on: { click: () => editManually(sample) } }),
        ]),
      );
      return box;
    }

    if (state === "running") {
      box.append(el("div", { class: "stagehead" }, [spinner("Transcribing…"), el("span", { class: "stagemetaR", text: (sample.transcription && sample.transcription.model) || "" })]));
      box.append(el("div", { class: "tnote", text: "Your recording is already saved. You can play it while the text is being produced." }));
      return box;
    }

    if (state === "failed") {
      box.style.borderColor = "rgba(255,107,107,0.35)";
      const code = (sample.transcription && sample.transcription.error_code) || "stt_failed";
      box.append(
        errorBar(
          [
            el("b", { text: "Could not transcribe this recording. " }),
            transcriptionErrorText(code) + " ",
            el("span", { class: "code mono", text: code }),
            el("b", { text: " Your audio is saved" }),
            " — nothing was lost and no text was invented. Try again, or type the transcript yourself.",
          ],
          "err",
        ),
      );
      box.append(
        el("div", { class: "trow" }, [
          el("button", { class: "btn pri sm", type: "button", text: "Try transcribing again", on: { click: () => transcribe(sample) } }),
          el("button", { class: "btn sm", type: "button", text: "Type it myself", on: { click: () => editManually(sample) } }),
        ]),
      );
      return box;
    }

    // done / edited / confirmed
    const chips = transcriptChips(sample);
    const chipEls = [];
    if (chips.includes("auto")) chipEls.push(el("span", { class: "chip auto", text: "Auto-transcribed" }));
    if (chips.includes("edit")) chipEls.push(el("span", { class: "chip edit", text: "Edited by you" }));
    chipEls.push(chips.includes("confirmed") ? el("span", { class: "chip conf", text: "Confirmed" }) : el("span", { class: "chip unconf", text: "Not confirmed" }));

    if (sample.transcript_confirmed) {
      box.append(
        el("div", { class: "tblock" }, [
          el("div", { class: "tlab" }, [el("span", { text: "Transcript" }), ...chipEls]),
          el("div", { class: "rowbox", style: "font-size:var(--fs-body);color:var(--text-0)", text: sample.transcript }),
          el("div", { class: "trow" }, [
            el("button", { class: "btn pri sm", type: "button", text: "Use this recording", on: { click: () => useRecording(sample) } }),
            el("button", { class: "btn sm", type: "button", text: "Edit again", on: { click: () => editManually(sample) } }),
          ]),
        ]),
      );
      return box;
    }

    const ta = el("textarea", { class: "inp", "aria-label": "Transcript" });
    ta.value = sample.transcript || "";
    const inUse = sample.id === voice.reference_sample_id;
    const confirmBtn = el("button", { class: "btn pri sm", type: "button", text: inUse ? "Confirm transcript" : "Confirm and use this recording", "data-rtv-confirm": "1" });
    confirmBtn.addEventListener("click", () => confirmTranscript(sample, ta.value));
    const meta = sample.transcription || {};
    box.append(
      el("div", { class: "tblock" }, [
        el("div", { class: "tlab" }, [el("span", { text: "Transcript" }), ...chipEls]),
        ta,
        el("div", { class: "tnote", text: "Speech recognition is not perfect with names, numbers and dialect. Correct anything that does not match what you actually said." }),
        el("div", { class: "trow" }, [
          confirmBtn,
          el("button", { class: "btn sm", type: "button", text: "Transcribe again", on: { click: () => transcribe(sample) } }),
          el("span", { style: "font-size:11.5px;color:var(--text-1)", text: `${meta.model || "STT"}${meta.duration_ms ? " · " + (meta.duration_ms / 1000).toFixed(1) + " s" : ""}` }),
        ]),
      ]),
    );
    return box;
  }

  /* ---------------------------------------------------- block 03: samples */

  const currentBox = el("div", { class: "card" });
  const samplesBox = el("div", { class: "card" });
  let samplesBlock = null;


  /* ------------------------------------------------ F10: excerpt tool */
  //
  // Cut a shorter reference out of an existing recording. The server creates
  // a NEW sample (original audio, metadata and transcript untouched) with an
  // empty, unconfirmed transcript, because the original text does not match
  // the cut audio. The excerpt is transcribed here and must be confirmed and
  // explicitly selected as the reference — nothing happens silently.
  const excerptStage = el("div", { class: "stagebox hidden", "data-f10": "excerpt-tool" });
  // persistent, hidden host: clear(excerptStage) must not orphan a playing element
  const excerptAudioHost = el("div", { hidden: true, "aria-hidden": "true" });
  let excerptAudio = null;
  let excerptAudioUrl = "";
  let excerptAudioFor = "";
  let excerptRaf = 0;

  function stopExcerptPlayback() {
    if (excerptRaf) cancelAnimationFrame(excerptRaf);
    excerptRaf = 0;
    if (excerptAudio) {
      try {
        excerptAudio.pause();
      } catch {
        /* ignore */
      }
    }
  }

  async function excerptAudioFor_(sample) {
    if (excerptAudio && excerptAudioFor === sample.id) return excerptAudio;
    stopExcerptPlayback();
    if (excerptAudioUrl) URL.revokeObjectURL(excerptAudioUrl);
    const resp = await fetch(voicesApi.sampleAudioUrl(voice.id, sample.id), {
      headers: { Authorization: "Bearer " + getToken() },
    });
    if (!resp.ok) throw new Error("the recording could not be loaded");
    excerptAudioUrl = URL.createObjectURL(await resp.blob());
    // F9 lesson: src set at construction
    if (excerptAudio) excerptAudio.remove();
    excerptAudio = el("audio", { preload: "auto", src: excerptAudioUrl, class: "excerpt-audio" });
    excerptAudioHost.append(excerptAudio);
    excerptAudioFor = sample.id;
    return excerptAudio;
  }

  function openExcerptTool(sample, maxMs) {
    stopExcerptPlayback();
    clear(excerptStage);
    excerptStage.classList.remove("hidden");
    const duration = Math.max(0, Math.round(Number(sample.duration_ms) || 0));
    const MIN = 1000;
    if (duration < MIN * 2) {
      excerptStage.append(errorBar([el("b", { text: "Too short to cut. " }), "This recording is shorter than 2 s."], "warn"));
      return;
    }
    const limit = maxMs ? Math.min(Math.round(maxMs), duration) : duration;
    let start = 0;
    let len = Math.min(limit, duration);

    const startIn = el("input", {
      type: "range", min: "0", max: String(duration - MIN), step: "100", value: "0",
      "aria-label": "Excerpt start", "data-f10": "excerpt-start", style: "width:100%",
    });
    const lenIn = el("input", {
      type: "range", min: String(MIN), max: String(limit), step: "100", value: String(len),
      "aria-label": "Excerpt length", "data-f10": "excerpt-length", style: "width:100%",
    });
    const startLab = el("b", { class: "mono" });
    const lenLab = el("b", { class: "mono" });
    const rangeLab = el("div", { class: "fhint mono", "data-f10": "excerpt-range" });
    const bars = el("div", { class: "wave", "aria-hidden": "true", style: "margin:10px 0" });
    const env = barRow(sample.sha256 || sample.id, sample.metrics && sample.metrics.waveform);
    bars.append(...env);

    const clamp = () => {
      start = Math.max(0, Math.min(start, duration - MIN));
      const maxLenHere = Math.min(limit, duration - start);
      lenIn.max = String(maxLenHere);
      len = Math.max(MIN, Math.min(len, maxLenHere));
      startIn.value = String(start);
      lenIn.value = String(len);
    };
    const paint = () => {
      startLab.textContent = fmtSeconds(start) === "—" ? "0.0 s" : fmtSeconds(start);
      lenLab.textContent = fmtSeconds(len);
      rangeLab.textContent = `Excerpt ${(start / 1000).toFixed(1)} s – ${((start + len) / 1000).toFixed(1)} s of ${fmtSeconds(duration)}`;
      env.forEach((b, i) => {
        const t0 = (i / env.length) * duration;
        const t1 = ((i + 1) / env.length) * duration;
        const inside = t1 > start && t0 < start + len;
        b.style.opacity = inside ? "1" : "0.25";
      });
    };
    startIn.addEventListener("input", () => { start = Number(startIn.value); clamp(); paint(); });
    lenIn.addEventListener("input", () => { len = Number(lenIn.value); clamp(); paint(); });
    clamp();
    paint();

    const playBtn = el("button", { class: "btn sm", type: "button", text: "▶ Play excerpt", "data-f10": "excerpt-play" });
    const status = el("div", { class: "statusline" });
    playBtn.addEventListener("click", async () => {
      if (excerptAudio && excerptAudioFor === sample.id && !excerptAudio.paused) {
        stopExcerptPlayback();
        playBtn.textContent = "▶ Play excerpt";
        return;
      }
      try {
        const audio = await excerptAudioFor_(sample);
        // F6: only one thing plays at a time
        sharedSamplePlayer.stop();
        document.querySelectorAll("audio.preview-audio").forEach((a) => {
          try { a.pause(); } catch { /* ignore */ }
        });
        audio.currentTime = start / 1000;
        await audio.play();
        playBtn.textContent = "❚❚ Stop";
        const end = (start + len) / 1000;
        const tick = () => {
          if (!excerptAudio || excerptAudio.paused || excerptAudio.currentTime >= end) {
            stopExcerptPlayback();
            playBtn.textContent = "▶ Play excerpt";
            return;
          }
          excerptRaf = requestAnimationFrame(tick);
        };
        excerptRaf = requestAnimationFrame(tick);
      } catch (err) {
        status.textContent = "Could not play the excerpt: " + err.message;
        status.className = "statusline err";
      }
    });

    const createBtn = el("button", {
      class: "btn pri", type: "button", "data-f10": "excerpt-create",
      text: "Create excerpt as a new sample",
    });
    const cancelBtn = el("button", { class: "btn ghost", type: "button", text: "Cancel" });
    cancelBtn.addEventListener("click", () => {
      stopExcerptPlayback();
      excerptStage.classList.add("hidden");
      clear(excerptStage);
    });
    createBtn.addEventListener("click", async () => {
      if (createBtn.disabled) return;
      createBtn.disabled = true;
      stopExcerptPlayback();
      status.textContent = "Creating the excerpt…";
      status.className = "statusline";
      let created;
      try {
        created = await voicesApi.excerpt(voice.id, sample.id, start, start + len);
      } catch (err) {
        createBtn.disabled = false;
        status.textContent = "Could not create the excerpt: " + err.message + " Your original recording is unchanged.";
        status.className = "statusline err";
        return;
      }
      checkingId = created.id;
      clear(excerptStage);
      const from = queryParam(location.hash, "from");
      excerptStage.append(
        errorBar(
          [
            el("b", { text: "Excerpt created as a new sample. " }),
            `${fmtSeconds(created.duration_ms)} from ${(start / 1000).toFixed(1)} s. Your original recording is unchanged. `,
            "Next: check its transcript below and confirm it to use the excerpt.",
            from
              ? el("span", {}, [" Then ", el("a", { href: `#/profiles/${encodeURIComponent(from)}`, text: `go back to profile ${from}` }), "."])
              : null,
          ],
          "ok",
        ),
      );
      // the excerpt needs its own transcript: transcribe it like a new recording
      await reloadVoice();
      const fresh = (voice.samples || []).find((x) => x.id === created.id) || created;
      await transcribe(fresh, true);
    });

    excerptStage.append(
      el("div", { class: "stagehead" }, [
        el("div", { class: "stagestate", text: `Excerpt of ${sampleTitle(sample)}` }),
        el("div", {
          class: "stagemetaR",
          text: maxMs
            ? `This speech engine uses at most ${fmtSeconds(maxMs)} of a reference`
            : "Pick the part that sounds most like you",
        }),
      ]),
      bars,
      el("label", { class: "fhint" }, ["Start ", startLab]),
      startIn,
      el("label", { class: "fhint" }, ["Length ", lenLab, maxMs ? ` (max ${fmtSeconds(Math.min(maxMs, duration))})` : ""]),
      lenIn,
      rangeLab,
      el("div", { style: "display:flex;gap:8px;flex-wrap:wrap;margin-top:10px" }, [playBtn, createBtn, cancelBtn]),
      el("div", {
        class: "fhint",
        text: "Creates a separate sample from this range. The original recording, its file and its transcript are not changed. The excerpt gets its own transcript, which you confirm before using it.",
      }),
      status,
    );
    excerptStage.scrollIntoView({ block: "nearest", behavior: "smooth" });
    startIn.focus();
  }

  function renderSamples() {
    renderCurrent();
    clear(samplesBox);
    const others = (voice.samples || []).filter((s) => s.id !== voice.reference_sample_id);
    samplesBlock.classList.toggle("hidden", others.length === 0);
    for (const sample of others.slice().reverse()) samplesBox.append(sampleRow(sample));
    if (others.length) {
      samplesBox.append(
        el("div", { class: "tnote refhint", text: "The voice speaks with one recording. These are kept so you can switch back; they are not mixed in." }),
      );
    }
  }

  /** Block 01: the recording the voice speaks with, and whether every set
   *  that can be called uses it. */
  function renderCurrent() {
    clear(currentBox);
    const ref = (voice.samples || []).find((s) => s.id === voice.reference_sample_id);
    if (!ref) {
      currentBox.append(el("div", { class: "tnote", text: "No recording yet. Record or upload one below." }));
      return;
    }
    const quote = transcriptQuote(ref);
    currentBox.append(
      el("div", { class: "aud" }, [
        playButton(ref, "play", "Play the recording in use"),
        el("div", { class: "wave", "aria-hidden": "true" }, barRow(ref.sha256 || ref.id, ref.metrics && ref.metrics.waveform)),
        el("div", { class: "adur mono", text: fmtDuration(ref.duration_ms) }),
      ]),
      el("div", { class: "sname", style: "margin-top:10px", text: sampleTitle(ref) }),
      el("div", { class: "stext", text: quote || "No transcript yet" }),
      el("div", { class: "smeta", text: sampleMetaLine(ref) }),
    );
    if (clippingPct(ref) > 0.1) {
      currentBox.append(el("div", { class: "tnote", text: `Clipping ${clippingPct(ref).toFixed(1)}%: parts are too loud and cut off. A quieter take usually clones better.` }));
    }
    const actions = [];
    if (ref.transcript_confirmed) {
      actions.push(el("button", { class: "btn sm", type: "button", text: "Edit text", on: { click: () => { editManually(ref); transcriptStage.scrollIntoView({ block: "center", behavior: "smooth" }); } } }));
      currentBox.append(el("div", { class: "tnote", text: "Ready to use as a cloning reference." }));
    } else {
      currentBox.append(el("div", { class: "tnote", text: "Its transcript is not confirmed yet: check it below before a set can speak with it." }));
    }
    if ((Number(ref.duration_ms) || 0) >= 2000) {
      actions.push(el("button", { class: "btn sm", type: "button", text: "Excerpt…", "data-f10": "excerpt-row", "aria-label": `Create an excerpt of ${sampleTitle(ref)}`, on: { click: () => openExcerptTool(ref, 0) } }));
    }
    if (actions.length) currentBox.append(el("div", { class: "trow" }, actions));

    // every set that speaks with the voice uses this recording from its next call
    const sets = usedByNames();
    currentBox.append(
      el("div", {
        class: "tnote",
        text: sets.length
          ? `${sets.join(", ")} ${sets.length === 1 ? "speaks" : "speak"} with this recording from the next call.`
          : "No set speaks with this voice yet. Choose it in a set under Sets.",
      }),
    );
  }

  function sampleRow(sample) {
    const row = el("div", { class: "srow", "data-sample-id": sample.id });
    row.append(playButton(sample, "splay", "Play"));
    const info = el("div", { class: "sinfo" }, [
      el("div", { class: "sname", text: sampleTitle(sample) }),
    ]);
    const quote = transcriptQuote(sample);
    info.append(quote ? el("div", { class: "stext", text: quote }) : el("div", { class: "stext", style: "font-style:normal;color:var(--text-1)", text: "No transcript yet" }));
    info.append(el("div", { class: "smeta", text: sampleMetaLine(sample) }));
    if (sample.derived_from && sample.derived_from.sample_id) {
      const parent = (voice.samples || []).find((x) => x.id === sample.derived_from.sample_id);
      info.append(
        el("div", {
          class: "smeta",
          "data-f10": "excerpt-origin",
          text: parent
            ? `Cut from “${sampleTitle(parent)}” — that recording is unchanged`
            : "Cut from a recording that has since been removed",
        }),
      );
    }
    row.append(info);

    if (clippingPct(sample) > 0.1) {
      const chip = el("span", { class: "chip clip", text: `Clipping ${clippingPct(sample).toFixed(1)}%`, title: "Parts of this recording are too loud and were cut off. A quieter take usually clones better." });
      row.append(chip);
    }
    if (sample.transcript_confirmed) {
      row.append(el("button", { class: "btn sm", type: "button", text: "Use this one", on: { click: () => useRecording(sample) } }));
    } else {
      row.append(el("span", { class: "chip unconf", text: "Not confirmed" }));
      row.append(
        el("button", {
          class: "btn sm",
          type: "button",
          text: "Check text",
          on: {
            click: () => {
              checkingId = sample.id;
              renderTranscriptStage();
              transcriptStage.scrollIntoView({ block: "center", behavior: "smooth" });
            },
          },
        }),
      );
    }

    // F10: cut a shorter reference from this recording (creates a new sample)
    if ((Number(sample.duration_ms) || 0) >= 2000) {
      row.append(
        el("button", {
          class: "sdel",
          type: "button",
          text: "Excerpt…",
          "data-f10": "excerpt-row",
          "aria-label": `Create an excerpt of ${sampleTitle(sample)}`,
          on: { click: () => openExcerptTool(sample, 0) },
        }),
      );
    }
    const removeBtn = el("button", { class: "sdel", type: "button", text: "Remove" });
    removeBtn.addEventListener("click", () => removeSample(sample));
    row.append(removeBtn);
    return row;
  }

  /* ---------------------------------------------------- block 04: preview */

  const previewBox = el("div", { class: "card" });
  const previews = [];
  let engines = [];
  let enginesLoaded = false;

  async function loadEngines() {
    try {
      const res = await ttsEnginesApi.list();
      engines = (res && res.engines) || [];
    } catch {
      engines = [];
    }
    enginesLoaded = true;
  }

  /** Engine label = model + clone constraint + honest legacy/variant notes.
   *  Internal hostnames/ports are never shown, and the engine is named after
   *  its MODEL — never after a stored row's display_name. */
  function engineLabel(e) {
    if (e.voice_cloning === "reference_audio") return cloneEngineLabel(e);
    const parts = [engineModelLabel(e.model, e.role || "tts")];
    if (e.voice_cloning === "unknown") {
      parts.push("· capability not verified");
    } else {
      parts.push("· preset voices only");
    }
    if (e.id) parts.push(`stored as ${e.id}`);
    return parts.join(" · ");
  }

  function renderPreview() {
    clear(previewBox);
    const ref = (voice.samples || []).find((s) => s.id === voice.reference_sample_id);
    const refConfirmed = !!(ref && (ref.transcript || "").trim() && ref.transcript_confirmed);
    const cloningEngines = engines.filter((e) => e.voice_cloning === "reference_audio");
    const unknownEngines = engines.filter((e) => e.voice_cloning === "unknown");
    const noneEngines = engines.filter((e) => e.voice_cloning === "none");

    const ta = el("textarea", { class: "inp", "aria-label": "Preview text", rows: 2 });
    ta.value = "Guten Morgen Alex, es ist Viertel nach acht und im Apfelbaum vor dem Fenster sitzen drei Vögel.";
    const sel = el("select", { class: "inp", style: "max-width:360px", "aria-label": "TTS profile" });
    // cloning engines only: unknown shown disabled, none never offered
    for (const e of cloningEngines) sel.append(el("option", { value: e.id, text: engineLabel(e) }));
    for (const e of unknownEngines) {
      const opt = el("option", { value: e.id, text: engineLabel(e), disabled: true });
      sel.append(opt);
    }
    const genBtn = el("button", { class: "btn pri", type: "button", text: "Generate preview" });

    if (!enginesLoaded) {
      previewBox.append(spinner("Loading speech engines…"));
      return;
    }
    if (engines.length === 0) {
      previewBox.append(errorBar([el("b", { text: "No speech engine configured. " }), "Add a TTS backend profile under Backends before previewing."], "warn"));
      return;
    }
    if (cloningEngines.length === 0) {
      previewBox.append(
        errorBar(
          [
            el("b", { text: "No cloning-capable engine. " }),
            "None of your speech engines advertises reference-audio cloning, so your recorded voice cannot be previewed. ",
            el("a", { href: "#/backends", text: "Add or edit a TTS backend" }),
            " with voice cloning set to reference_audio.",
          ],
          "warn",
        ),
      );
      return;
    }

    // Reference readiness gate: no 409 round trips.
    let gateReason = "";
    if (!ref) gateReason = "This voice has no recording yet.";
    else if (!(ref.transcript || "").trim()) gateReason = "The reference transcript is empty.";
    else if (!ref.transcript_confirmed) gateReason = "The reference transcript is not confirmed yet.";
    if (gateReason) {
      genBtn.disabled = true;
      genBtn.setAttribute("aria-disabled", "true");
      genBtn.title = gateReason;
    }

    // F10: reference length vs the selected engine, shown before any click
    // and re-evaluated when the engine changes. Too long = Generate is off
    // with the reason and a way to a fitting excerpt; never a silent cut.
    const fitLine = el("div", { class: "fitline", "data-f10": "preview-fit" });
    const updateFit = () => {
      clear(fitLine);
      if (gateReason || !ref) return;
      const engine = engines.find((e) => e.id === sel.value);
      const fit = referenceFit(ref.duration_ms, engine && engine.max_reference_ms);
      const tooLong = fit.state === "too_long";
      genBtn.disabled = tooLong;
      genBtn.setAttribute("aria-disabled", String(tooLong));
      genBtn.title = tooLong ? "Reference too long for this engine" : "";
      if (tooLong) {
        fitLine.append(
          errorBar(
            [
              el("b", { text: "Preview is off for this engine. " }),
              referenceTooLongText(fit.sampleMs, fit.maxMs) + " ",
              el("button", {
                class: "linklike",
                type: "button",
                "data-f10": "open-excerpt",
                text: `Choose a ${fmtSeconds(fit.maxMs)} excerpt`,
                on: { click: () => openExcerptTool(ref, fit.maxMs) },
              }),
            ],
            "warn",
          ),
        );
      } else if (fit.state === "fits") {
        fitLine.append(
          el("div", { class: "fhint", text: `Reference ${fmtSeconds(fit.sampleMs)} — fits this engine (uses up to ${fmtSeconds(fit.maxMs)}).` }),
        );
      }
    };
    sel.addEventListener("change", updateFit);

    previewBox.append(
      el("div", { class: "testrow" }, [
        ta,
        el("div", { style: "display:flex;flex-direction:column;gap:8px" }, [
          sel,
          genBtn,
        ]),
      ]),
      el("div", { class: "fhint", text: "Generates audio against the selected speech engine using the recording in use. It does not change any set." }),
      fitLine,
    );
    updateFit();
    if (gateReason) {
      previewBox.append(
        errorBar(
          [
            el("b", { text: "Preview is off. " }),
            gateReason + " ",
            // P1-1: the reference transcript lives on this same page; the hint
            // must bring the confirm action into view instead of linking to the
            // page we are already on.
            el("button", {
              class: "linklike",
              type: "button",
              text: "Confirm the reference transcript",
              on: { click: () => focusConfirmAction() },
            }),
            " first.",
          ],
          "warn",
        ),
      );
    }
    if (noneEngines.length) {
      previewBox.append(
        el("div", { class: "fhint", text: `${noneEngines.length} engine(s) use preset voices only and are hidden here.` }),
      );
    }
    genBtn.addEventListener("click", () => {
      if (genBtn.disabled) return;
      generatePreview(sel.value, ta.value, ref ? ref.id : "");
    });
    for (const p of previews) previewBox.append(p.node);
  }

  async function generatePreview(ttsProfile, text, sampleId) {
    if (!ttsProfile) return;
    const loading = spinner("Generating preview…");
    previewBox.append(loading);
    try {
      const resp = await voicesApi.preview(voice.id, ttsProfile, text, sampleId || undefined);
      const blob = await resp.blob();
      const headers = {
        first: resp.headers.get("X-Rtv-First-Pcm-Ms"),
        total: resp.headers.get("X-Rtv-Total-Ms"),
        audio: resp.headers.get("X-Rtv-Audio-Ms"),
        model: resp.headers.get("X-Rtv-Model"),
        profile: resp.headers.get("X-Rtv-Profile-Id"),
        voiceId: resp.headers.get("X-Rtv-Voice-Id"),
        sampleId: resp.headers.get("X-Rtv-Sample-Id"),
        sha8: resp.headers.get("X-Rtv-Reference-Sha8"),
      };
      const url = URL.createObjectURL(blob);
      // F9: the src must be present from the start. Assigning it only inside a
      // "play" listener was a deadlock: an <audio> with no media source has a
      // dead native Play button, so the listener could never fire and the user
      // saw a control bar that did nothing. `preload="none"` already gives the
      // lazy behaviour that was wanted — the browser fetches the blob on the
      // first play, not at render time.
      const audio = el("audio", {
        controls: true,
        preload: "none",
        src: url,
        class: "preview-audio",
        style: "flex:1;min-width:180px",
      });
      // F6: at most one sample/preview plays at a time — starting this
      // preview stops any other preview player on the page.
      audio.addEventListener("play", () => {
        sharedSamplePlayer.stop();
        document.querySelectorAll("audio.preview-audio").forEach((a) => {
          if (a !== audio) {
            try {
              a.pause();
            } catch {
              /* ignore */
            }
          }
        });
      });
      const node = el("div", { class: "stagebox" }, [
        el("div", { class: "aud" }, [
          audio,
          el("div", { class: "adur", text: headers.audio ? `audio ${(Number(headers.audio) / 1000).toFixed(2)} s` : "" }),
        ]),
        el("div", { class: "rlat", style: "display:flex;gap:16px;margin-top:10px;flex-wrap:wrap" }, [
          metric("First PCM", headers.first, "ms"),
          metric("Total", headers.total, "ms"),
          metric("Audio", headers.audio, "ms"),
        ]),
        el("div", { class: "provenance", style: "margin-top:10px;font-size:12px;line-height:1.7" }, [
          el("div", { class: "tlab", text: "Provenance (reported by the server)" }),
          kvLine("TTS model", headers.model || "—"),
          kvLine("Backend profile", headers.profile || "—"),
          kvLine("Voice", headers.voiceId || "—"),
          kvLine("Sample", `${headers.sampleId || "—"} · sha8 ${headers.sha8 || "—"}`),
        ]),
        el("div", { class: "tnote", text: "Server-side measurements from request to last PCM — not acoustic playback latency." }),
      ]);
      previews.unshift({ node, url });
      // F9 review: a replaced preview must release its blob URL, or every
      // generated preview leaks the audio until the tab is closed.
      while (previews.length > 3) revokePreview(previews.pop());
      loading.remove();
      renderPreview();
    } catch (err) {
      loading.remove();
      const code = err.code || "";
      let why = err.message;
      if (code === "voice_incompatible") why = "This engine does not support cloned voices.";
      else if (code === "transcript_not_confirmed") why = "The reference transcript must be confirmed before previewing.";
      const lengthIssue = referenceTooLongIssue(err);
      if (lengthIssue) {
        const src = (voice.samples || []).find((x) => x.id === lengthIssue.sample_id);
        previewBox.append(
          errorBar([
            el("b", { text: "Preview not generated. " }),
            referenceTooLongText(lengthIssue.sample_ms, lengthIssue.max_ms) + " ",
            src
              ? el("button", {
                  class: "linklike",
                  type: "button",
                  text: "Choose an excerpt",
                  on: { click: () => openExcerptTool(src, lengthIssue.max_ms) },
                })
              : null,
            el("b", { text: " No profile was changed." }),
          ]),
        );
        return;
      }
      previewBox.append(
        errorBar([
          el("b", { text: "Preview failed. " }),
          why,
          el("b", { text: " No profile was changed." }),
          " Check the speech engine and try again.",
        ]),
      );
    }
  }

  /** Release a preview's blob URL (evicted or on navigation). */
  function revokePreview(p) {
    if (!p || !p.url) return;
    try {
      URL.revokeObjectURL(p.url);
    } catch {
      /* ignore */
    }
    p.url = "";
  }

  function kvLine(k, v) {
    return el("div", { class: "kv" }, [el("span", { text: k }), el("b", { class: "mono", text: v })]);
  }

  function metric(label, value, unit) {
    return el("div", {}, [el("div", { class: "mlab", text: label }), el("div", { class: "mval", text: value != null ? `${value} ${unit}` : "—" })]);
  }

  /* -------------------------------------------------------------- actions */

  async function uploadSample(file) {
    const status = el("div");
    recordStage.classList.remove("hidden");
    clear(recordStage);
    recordStage.append(status, spinner(`Uploading ${file.name}…`));
    try {
      const blob = file;
      const sample = await voicesApi.uploadSample(voice.id, blob, file.name, "upload");
      checkingId = sample.id;
      clear(recordStage);
      recordStage.append(
        errorBar([el("b", { text: "Recording saved. " }), "It is being transcribed; check the text below, then use it."], "ok"),
      );
      await reloadVoice();
      await transcribe(sample, true);
    } catch (err) {
      clear(recordStage);
      const code = err.code || "";
      const uploadCodes = ["sample_too_short", "sample_too_long", "unsupported_format", "too_large"];
      if (uploadCodes.includes(code)) {
        recordStage.append(
          errorBar([
            el("b", { text: "Could not use this file. " }),
            uploadErrorText(code) + " ",
            el("span", { class: "code mono", text: code }),
            el("b", { text: " Nothing was saved." }),
            " Convert the file or record a sample with the microphone.",
          ]),
        );
      } else {
        recordStage.append(
          errorBar([
            el("b", { text: "The recording was saved, but the next step failed. " }),
            err.message || String(err),
            el("b", { text: " Your audio is saved." }),
            " Try transcribing again.",
          ]),
        );
      }
    }
  }

  function startRecording() {
    if (recorder && recorder.recording) return;
    recordStage.classList.remove("hidden");
    clear(recordStage);
    const stateLine = el("div", { class: "stagestate", style: "color:var(--danger)" }, [
      el("span", { class: "sdot dot-err" }),
      "Recording · 0:00",
    ]);
    const meter = el("div", { class: "meter" });
    const clipWarn = el("div", { class: "tnote hidden", text: "Clipping — the recording is too loud. Move back from the microphone." });
    const micLine = el("div", { class: "stagemetaR", text: "Raw audio — noise suppression and auto-gain are off on purpose" });
    const stopBtn = el("button", { class: "btn sm", type: "button", text: "Stop recording" });
    recordStage.append(
      el("div", { class: "stagehead" }, [stateLine, micLine]),
      meter,
      clipWarn,
      el("div", { style: "text-align:center;margin-top:12px" }, [stopBtn]),
    );
    recorder = new MicRecorder({
      deviceId: selectedMic,
      onLevel: (level, clipping) => {
        renderMeter(meter, level, clipping);
        clipWarn.classList.toggle("hidden", !clipping);
      },
      onTick: (ms) => {
        stateLine.lastChild.textContent = "Recording · " + fmtDuration(ms);
      },
      onStarted: (label, devId) => {
        micLine.textContent = label ? `Microphone: ${label}` : "Microphone: System default";
        // permission is now granted: labels become available
        micPicker && micPicker.refresh().catch(() => {});
      },
      onError: (err) => {
        clear(recordStage);
        recordStage.append(
          errorBar([el("b", { text: "Microphone unavailable. " }), err.message, el("b", { text: " Nothing was recorded." }), " Allow microphone access and try again."]),
        );
      },
    });
    recorder.start().catch(() => {});
    stopBtn.addEventListener("click", async () => {
      stopBtn.disabled = true;
      const result = await recorder.stop();
      recorder = null;
      await saveRecording(result);
    });
  }

  function renderMeter(meter, level, clipping) {
    clear(meter);
    const bars = 15;
    for (let i = 0; i < bars; i++) {
      const on = i / bars < level;
      const h = 6 + Math.round(Math.sin(i * 0.9) * 7 + 7);
      meter.append(el("i", { class: on && clipping ? "clip" : "", style: `height:${h}px;${on ? "" : "opacity:0.25"}` }));
    }
  }

  async function saveRecording({ blob, durationMs, sampleRate }) {
    recordStage.classList.remove("hidden");
    clear(recordStage);
    if (durationMs < 1000) {
      recordStage.append(
        errorBar([
          el("b", { text: "Could not save this recording. " }),
          `It is ${(durationMs / 1000).toFixed(1)} s. Samples need at least 1 second. `,
          el("span", { class: "code mono", text: "sample_too_short" }),
          el("b", { text: " Nothing was saved." }),
          " Record a longer sample and try again.",
        ]),
      );
      return;
    }
    if (durationMs > 30000) {
      recordStage.append(
        errorBar([
          el("b", { text: "Could not save this recording. " }),
          `It is ${Math.round(durationMs / 1000)} s. Samples can be at most 30 seconds. `,
          el("span", { class: "code mono", text: "sample_too_long" }),
          el("b", { text: " Nothing was saved." }),
          " Record a shorter sample and try again.",
        ]),
      );
      return;
    }
    recordStage.append(spinner(`Saving ${fmtDuration(durationMs)} recording…`));
    try {
      const sample = await voicesApi.uploadSample(voice.id, blob, "recording.wav", "mic");
      checkingId = sample.id;
      clear(recordStage);
      recordStage.append(errorBar([el("b", { text: "Recording saved. " }), "It is being transcribed; check the text below, then use it."], "ok"));
      await reloadVoice();
      await transcribe(sample, true);
    } catch (err) {
      clear(recordStage);
      const code = err.code || "unsupported_format";
      recordStage.append(
        errorBar([
          el("b", { text: "Could not save this recording. " }),
          uploadErrorText(code) + " ",
          el("span", { class: "code mono", text: code }),
          el("b", { text: " Nothing was saved." }),
          " Try again.",
        ]),
      );
    }
  }

  async function transcribe(sample, silent) {
    transcriptStage.dataset.running = sample.id;
    if (!silent) renderTranscriptStage();
    const running = { ...sample, transcription: { state: "running", model: (sample.transcription || {}).model } };
    clear(transcriptStage);
    transcriptStage.append(transcriptBlock(running));
    try {
      await voicesApi.transcribe(voice.id, sample.id);
    } catch (err) {
      // the endpoint returns 200 with a failed state for backend errors; a
      // non-200 is a client/protocol error and keeps the audio untouched
      if (err.status && err.status !== 200) {
        clear(transcriptStage);
        transcriptStage.append(
          errorBar([el("b", { text: "Transcription request failed. " }), err.message.replace(/[.\s]*$/, "."), el("b", { text: " Your audio is saved." }), " Try again or type the transcript yourself."]),
        );
        return;
      }
    }
    await reloadVoice();
  }

  async function confirmTranscript(sample, text) {
    if (!(text || "").trim()) {
      await noticeDialog({
        title: "The transcript is empty",
        message: "Type what was said before confirming.",
        detail: "Nothing was confirmed.",
      });
      return;
    }
    try {
      await voicesApi.patchSample(voice.id, sample.id, { transcript: text, transcript_confirmed: true });
    } catch (err) {
      await noticeDialog({ title: "Could not confirm the transcript", message: err.message, detail: "Nothing was confirmed." });
      return;
    }
    await useRecording(sample);
  }

  /** Bring the transcript confirm action into view and focus it (P1-1).
   *  Used by the preview gate hint instead of linking to the current page. */
  function focusConfirmAction() {
    const confirmBtn = document.querySelector("[data-rtv-confirm]");
    const ta = document.querySelector("textarea[aria-label='Transcript']");
    const target = confirmBtn || ta;
    if (target) {
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      try {
        target.focus({ preventScroll: true });
      } catch {
        target.focus();
      }
      return;
    }
    // no editor in the DOM (e.g. the block collapsed): re-render the pending
    // sample's check step, then focus it on the next frame.
    renderTranscriptStage();
    requestAnimationFrame(() => focusConfirmAction());
  }

  function editManually(sample) {
    const ta = el("textarea", { class: "inp", "aria-label": "Transcript" });
    ta.value = sample.transcript || "";
    // Primary: store the text AND confirm it in one step (voice-sample-flow
    // v1.2 §5.2/§5.4). The secondary keeps an unconfirmed draft, which the
    // sample list shows with a visible "Confirm transcript" action.
    const saveConfirm = el("button", { class: "btn pri sm", type: "button", text: "Save and Confirm" });
    if (sample.id !== voice.reference_sample_id && sample.id === checkingId) saveConfirm.textContent = "Save, confirm and use";
    const saveDraft = el("button", { class: "btn sm", type: "button", text: "Save without confirming" });
    const status = el("div", { class: "tnote" });
    saveConfirm.addEventListener("click", async () => {
      const text = ta.value.trim();
      if (!text) {
        status.className = "tnote err";
        status.textContent = "The transcript is empty. Type what was said before confirming.";
        return;
      }
      saveConfirm.disabled = true;
      saveDraft.disabled = true;
      try {
        await voicesApi.patchSample(voice.id, sample.id, { transcript: ta.value, transcript_confirmed: true });
      } catch (err) {
        saveConfirm.disabled = false;
        saveDraft.disabled = false;
        status.className = "tnote err";
        status.textContent = "Could not confirm the transcript: " + err.message;
        return;
      }
      // the recording in use, or the one just added: confirming means using it
      if (sample.id === voice.reference_sample_id || sample.id === checkingId) await useRecording(sample);
      else await reloadVoice();
    });
    saveDraft.addEventListener("click", async () => {
      saveConfirm.disabled = true;
      saveDraft.disabled = true;
      try {
        await voicesApi.patchSample(voice.id, sample.id, { transcript: ta.value });
        await reloadVoice();
      } catch (err) {
        saveConfirm.disabled = false;
        saveDraft.disabled = false;
        status.className = "tnote err";
        status.textContent = "Could not save the transcript: " + err.message;
      }
    });
    clear(transcriptStage);
    transcriptStage.append(
      el("div", { class: "card" }, [
        el("div", { class: "tlab", text: "Transcript" }),
        ta,
        el("div", { class: "tnote", text: "Confirm only when the text matches what you actually said — the cloning engine matches this text against your audio." }),
        el("div", { class: "trow" }, [saveConfirm, saveDraft]),
        status,
      ]),
    );
    ta.focus();
  }

  /** Make a recording the one the voice speaks with: every set that speaks
   *  with the voice uses it from its next call. */
  async function useRecording(sample) {
    try {
      voice = await voicesApi.setReference(voice.id, sample.id);
      if (checkingId === sample.id) checkingId = "";
      clear(recordStage);
      recordStage.classList.add("hidden");
      await ctx.loadVoices();
      renderHeader();
      renderSamples();
      renderTranscriptStage();
      // the preview gate depends on the reference sample: re-render it so the
      // "empty" vs "not confirmed" state is never stale
      renderPreview();
      currentBox.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch (err) {
      await noticeDialog({
        title: "Could not use this recording",
        message: breaksSetsText(err) || err.message,
        detail: "The voice still speaks with the recording it had.",
      });
    }
  }

  /** A refusal because a switched-on set could not use the change, in words. */
  function breaksSetsText(err) {
    if (err.code !== "breaks_sets") return "";
    const first = (err.issues || [])[0];
    if (!first) return err.message;
    const tooLong = (first.issues || []).find((i) => i.code === "reference_too_long");
    const what = tooLong
      ? referenceTooLongText(tooLong.sample_ms, tooLong.max_ms)
      : setIssueText(first.issues) + ".";
    return `${first.display_name || first.profile} could not speak with it: ${what}`;
  }

  async function removeSample(sample) {
    const ok = await confirmDialog({
      title: "Remove this sample?",
      message: "This recording is deleted from the voice.",
      confirmLabel: "Remove sample",
      danger: true,
    });
    if (!ok) return;
    // F6: switching/removing a sample stops its playback first
    sharedSamplePlayer.stopKey(`${voice.id}:${sample.id}`);
    try {
      await voicesApi.removeSample(voice.id, sample.id);
      unregisterSampleSource(`${voice.id}:${sample.id}`);
      await reloadVoice();
    } catch (err) {
      await noticeDialog({
        title: err.status === 409 ? "This recording is in use" : "Could not remove the sample",
        message: err.message,
        detail: err.status === 409 ? "Use another recording first. Nothing was removed." : "Nothing was removed.",
      });
    }
  }

  async function saveVoice() {
    const values = await formDialog({
      title: "Voice details",
      fields: [
        { name: "display_name", label: "Display name", value: voice.display_name || voice.id },
        {
          name: "language",
          label: "Language of the recordings",
          value: voice.language || "",
          placeholder: "de",
          hint: "The language spoken in the recordings, e.g. de or en; used to transcribe them. What a set speaks is set in the set.",
        },
      ],
      confirmLabel: "Save voice",
    });
    if (!values) return;
    try {
      voice = await voicesApi.save(voice.id, { display_name: values.display_name, language: values.language });
      await ctx.loadVoices();
      renderHeader();
    } catch (err) {
      await noticeDialog({ title: "Could not save the voice", message: err.message, detail: "Nothing was saved." });
    }
  }

  async function deleteVoice() {
    const ok = await confirmDialog({
      title: "Delete this voice?",
      message: "This cannot be undone.",
      confirmLabel: "Delete voice",
      danger: true,
    });
    if (!ok) return;
    try {
      await voicesApi.remove(voice.id);
      await ctx.loadVoices();
      ctx.go("#/voices");
    } catch (err) {
      if (err.status === 409) {
        await noticeDialog({
          title: "This voice is in use",
          message: err.message,
          detail: "Choose another voice in those sets first. Nothing was deleted.",
        });
      } else {
        await noticeDialog({ title: "Could not delete the voice", message: err.message, detail: "Nothing was deleted." });
      }
    }
  }

  /** A play/pause button bound to the shared controller (F6). */
  function playButton(sample, cls, label) {
    const key = `${voice.id}:${sample.id}`;
    const btn = el("button", { class: cls, type: "button", text: "▶", "aria-label": label || "Play", "aria-pressed": "false", "data-rtv-sample": key });
    btn._rtvKey = key;
    btn._rtvUpdate = () => {
      const st = sharedSamplePlayer.stateFor(key);
      const playing = st === "playing";
      btn.textContent = playing ? "❚❚" : "▶";
      btn.setAttribute("aria-pressed", String(playing));
      btn.setAttribute("aria-label", playing ? "Pause" : "Play");
      btn.classList.toggle("playing", playing);
    };
    registerSampleSource(key, async () => {
      const resp = await fetch(voicesApi.sampleAudioUrl(voice.id, sample.id), {
        headers: { Authorization: "Bearer " + getToken() },
      });
      if (!resp.ok) throw new Error("audio not available");
      return resp.blob();
    });
    btn.addEventListener("click", async () => {
      // F6: starting a sample stops any preview player, so never two at once
      document.querySelectorAll("audio.preview-audio").forEach((a) => {
        try {
          a.pause();
        } catch {
          /* ignore */
        }
      });
      await sharedSamplePlayer.toggle(key);
      refreshPlayButtons();
    });
    return btn;
  }

  function refreshPlayButtons() {
    document.querySelectorAll("button[data-rtv-sample]").forEach((b) => b._rtvUpdate && b._rtvUpdate());
  }

  /* -------------------------------------------------------------- assemble */

  renderHeader();
  inner.append(
    ctx.shell.breadcrumb([{ label: "Voices", route: "#/voices" }, { label: voice.display_name || voice.id }]),
    header,
    el("div", { class: "block" }, [
      blockLabel("01", "The recording", "What the voice sounds like"),
      currentBox,
      excerptStage,
      excerptAudioHost,
    ]),
    el("div", { class: "block" }, [
      blockLabel("02", (voice.samples || []).length ? "Replace the recording" : "Add a recording"),
      el("p", { class: "blede", text: "Just talk — say whatever you like, in your own words. Or upload speech you already have. It is transcribed; you check the text, confirm it, and the voice speaks with the new recording, in every set that uses it." }),
      el("div", { class: "two" }, [micCard, uploadCard]),
      recordStage,
      transcriptStage,
    ]),
    (samplesBlock = el("div", { class: "block" }, [blockLabel("03", "Other recordings", "Kept to switch back"), samplesBox])),
    el("div", { class: "block" }, [blockLabel("04", "Preview", "Does not change any set"), previewBox]),
  );
  // F6/F2: release playback + microphone on navigation
  if (ctx.setCleanup) {
    ctx.setCleanup(() => {
      stopExcerptPlayback();
      if (excerptAudioUrl) URL.revokeObjectURL(excerptAudioUrl);
      sharedSamplePlayer.stop();
      sharedSamplePlayer.onChange = () => {};
      document.querySelectorAll("audio.preview-audio").forEach((a) => {
        try {
          a.pause();
          a.removeAttribute("src");
        } catch {
          /* ignore */
        }
      });
      // F9 review: release every preview blob URL on navigation too
      while (previews.length) revokePreview(previews.pop());
      for (const s of voice.samples || []) unregisterSampleSource(`${voice.id}:${s.id}`);
      if (recorder) {
        try {
          recorder.recording = false;
          stopStream(recorder.stream);
          recorder.ctx && recorder.ctx.close().catch(() => {});
        } catch {
          /* ignore */
        }
        recorder = null;
      }
    });
  }
  renderSamples();
  // F6: reflect controller state (playing/paused/ended) on every button
  sharedSamplePlayer.onChange = () => refreshPlayButtons();
  renderTranscriptStage();
  loadEngines().then(() => renderPreview());
  renderPreview();
  mountMicPicker();
  // F10: arriving from a profile/preview with "?excerpt=<sample>&max=<ms>"
  // opens the excerpt tool on that sample, pre-limited to the engine maximum
  const wantExcerpt = queryParam(location.hash, "excerpt");
  if (wantExcerpt) {
    const target = (voice.samples || []).find((x) => x.id === wantExcerpt);
    if (target) {
      openExcerptTool(target, Number(queryParam(location.hash, "max")) || 0);
    } else {
      excerptStage.classList.remove("hidden");
      excerptStage.append(
        errorBar([el("b", { text: "That recording is no longer in this voice. " }), "Pick a sample below and use Excerpt…"], "warn"),
      );
    }
  }
}

/* --------------------------------------------------------------- helpers */

function blockLabel(num, title, hint) {
  return el("div", { class: "blabel" }, [
    el("span", { class: "bnum mono", text: num }),
    el("span", { class: "btitle", text: title }),
    hint ? el("span", { class: "bhint", text: hint }) : null,
  ]);
}

/** Absolute-scale waveform on a dB display scale (F8): each bar's height is the
 *  bucket's own peak relative to digital full scale, mapped logarithmically so
 *  normal speech is legible. The envelope values themselves are untouched —
 *  only their rendering changes. Digital silence still renders as the minimum
 *  bar, never lifted. Falls back to a neutral row when the server did not
 *  provide an envelope. */
function barRow(seed, waveform) {
  const bars = [];
  const env = Array.isArray(waveform) ? waveform : null;
  if (!env || env.length === 0) {
    // no envelope: do not fake amplitude (a flat, quiet line is honest)
    for (let i = 0; i < 24; i++) {
      bars.push(el("i", { style: "height:12%" }));
    }
    return bars;
  }
  const step = Math.max(1, Math.floor(env.length / 24));
  for (let i = 0; i < 24; i++) {
    const p = env[Math.min(env.length - 1, i * step)] || 0;
    const pct = Math.max(3, Math.min(100, Math.round(levelToDisplay(p) * 100)));
    bars.push(el("i", { class: pct > 45 ? "on" : "", style: `height:${pct}%` }));
  }
  return bars;
}

/* Raw PCM microphone recorder: AudioWorklet capture, no DSP, live level. */
class MicRecorder {
  constructor({ onLevel, onTick, onError, onStarted, deviceId }) {
    this.onLevel = onLevel;
    this.onTick = onTick;
    this.onError = onError;
    this.onStarted = onStarted;
    this.deviceId = deviceId || "";
    this.recording = false;
    this.chunks = [];
    this.frames = 0;
    this.sampleRate = 44100;
    this.ctx = null;
    this.stream = null;
    this.startedAt = 0;
    this.timer = null;
  }

  async start() {
    try {
      // F2: honour the selected deviceId and report the truthful active track
      const { stream, activeLabel, activeDeviceId } = await openMic(this.deviceId);
      this.stream = stream;
      this.activeLabel = activeLabel;
      this.activeDeviceId = activeDeviceId;
      this.onStarted && this.onStarted(activeLabel, activeDeviceId);
      const ctx = new AudioContext();
      this.ctx = ctx;
      this.sampleRate = ctx.sampleRate;
      await ctx.audioWorklet.addModule("../worklet.js");
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "capture-worklet");
      node.port.onmessage = (ev) => {
        const int16 = new Int16Array(ev.data);
        if (!this.recording) return;
        this.chunks.push(int16);
        this.frames += int16.length;
        let peak = 0;
        let sum = 0;
        for (let i = 0; i < int16.length; i++) {
          const a = Math.abs(int16[i]);
          if (a > peak) peak = a;
          sum += a;
        }
        // F8: the meter is a DISPLAY of loudness, so it uses a dB scale.
        // The linear peak drove it before, which left normal speech
        // (~-20 dBFS = 0.1 linear) at 1-2 of 15 bars. Clipping detection
        // stays on the raw linear peak.
        const level = levelToDisplay(peak / 32768);
        const clipping = peak >= 32700;
        this.onLevel && this.onLevel(level, clipping);
      };
      source.connect(node);
      // keep the graph alive without audible output
      node.connect(ctx.destination);
      this.recording = true;
      this.startedAt = performance.now();
      this.timer = setInterval(() => this.onTick && this.onTick(performance.now() - this.startedAt), 200);
    } catch (err) {
      this.onError && this.onError(err);
      throw err;
    }
  }

  async stop() {
    this.recording = false;
    if (this.timer) clearInterval(this.timer);
    const durationMs = performance.now() - this.startedAt;
    const total = this.frames;
    const merged = new Int16Array(total);
    let off = 0;
    for (const c of this.chunks) {
      merged.set(c, off);
      off += c.length;
    }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close().catch(() => {});
    const { encodeWav } = await import("./wav.js");
    return { blob: new Blob([encodeWav(merged, this.sampleRate)], { type: "audio/wav" }), durationMs: merged.length / this.sampleRate * 1000 || durationMs, sampleRate: this.sampleRate };
  }
}