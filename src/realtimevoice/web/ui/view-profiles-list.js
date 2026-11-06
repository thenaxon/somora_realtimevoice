/* Sets list: a set is what an agent in somora speaks with.
 *
 * Sections "On" (somora can call it) then "Off". Each card says in one line
 * what the set listens, thinks and speaks with, its voice and language, and
 * what (if anything) keeps it from working.
 */

import { el, waveform, copyText, errorBar } from "./dom.js";
import { ID_PATTERN, idError, languageLabel, setIssueText, voiceChoices, voiceKeyFromSpec, findVoiceChoice } from "./format.js";
import { profiles as profilesApi } from "./api.js";
import { formDialog, noticeDialog } from "./dialogs.js";

function rowName(row) {
  return (row && (row.display_name || row.model || row.id)) || "";
}

function stage(role, label, name) {
  return el("div", { class: "stage", dataset: { role } }, [
    el("div", { class: "stagerole", text: label }),
    el("div", { class: "stagemodel", text: name || "—" }),
  ]);
}

function profileCard(profile, ctx, choices) {
  const spec = profile.spec || {};
  const find = (role, id) => ctx.roleProfiles(role).find((p) => p.id === id);
  const voice = findVoiceChoice(choices, voiceKeyFromSpec(spec));
  const card = el("div", { class: "pcard", dataset: { profile: profile.id } });

  const speaks = stage("tts", "Speaks", rowName(find("tts", spec.tts_profile)) || spec.tts_profile);
  if (voice) speaks.append(el("div", { class: "vpill" }, [waveform(voice.label), voice.label]));

  const badges = el("div", { style: "display:flex;gap:7px;flex-wrap:wrap" });
  badges.append(
    el("span", { class: "badge " + (profile.enabled ? "avail" : "draft") }, [el("span", { class: "bd" }), profile.enabled ? "On" : "Off"]),
  );
  if (profile.is_default) badges.append(el("span", { class: "badge live" }, [el("span", { class: "bd" }), "For other apps"]));

  card.append(
    el("div", { class: "prow1" }, [
      el("div", {}, [
        el("div", { style: "display:flex;align-items:center;gap:10px;flex-wrap:wrap" }, [
          el("div", { class: "pname", text: profile.display_name || profile.id }),
          el("button", {
            class: "idchip mono",
            type: "button",
            title: "Copy — in somora, give an agent this name as its voice",
            text: profile.id,
            on: { click: (e) => copyText(profile.id, e.currentTarget) },
          }),
        ]),
        el("div", { class: "pdesc", text: [languageLabel(spec.language), profile.description].filter(Boolean).join(" · ") }),
      ]),
      badges,
    ]),
    el("div", { class: "chain" }, [
      stage("stt", "Listens", rowName(find("stt", spec.stt_profile)) || spec.stt_profile),
      el("div", { class: "arrow", "aria-hidden": "true", text: "→" }),
      stage("llm", "Thinks", rowName(find("llm", spec.llm_profile)) || spec.llm_profile),
      el("div", { class: "arrow", "aria-hidden": "true", text: "→" }),
      speaks,
    ]),
  );

  const issue = setIssueText(profile.issues);
  if (issue) {
    card.append(
      errorBar([el("b", { text: profile.enabled ? "Calls fail: " : "Not complete: " }), issue + "."], profile.enabled ? "err" : "warn"),
    );
  }

  card.append(
    el("div", { class: "prow3" }, [
      el("div", { class: "lat" }),
      el("div", { class: "acts" }, [
        el("button", { class: "tbtn", type: "button", text: "Test", on: { click: () => ctx.go(`#/test?profile=${encodeURIComponent(profile.id)}`) } }),
        el("button", { class: "tbtn", type: "button", text: "Edit", on: { click: () => ctx.go(`#/profiles/${encodeURIComponent(profile.id)}`) } }),
      ]),
    ]),
  );
  return card;
}

export async function renderProfilesList(inner, ctx) {
  const list = await ctx.loadProfiles();
  const choices = voiceChoices(ctx.roleProfiles("tts"), ctx.voices || []);

  inner.append(
    ctx.shell.breadcrumb([{ label: "Sets" }]),
    el("div", { class: "head" }, [
      el("div", {}, [
        el("h1", { class: "h1", text: "Sets" }),
        el("p", { class: "lede" }, [
          "A set is what an agent speaks with: the model that listens, the one that thinks, the voice and the model that speaks it. " +
            "In somora, give an agent the set's name as its voice. What you save is what the next call uses.",
        ]),
      ]),
      el("div", { class: "head-actions" }, [
        el("button", { class: "btn pri", type: "button", text: "＋ New set", on: { click: () => newProfile(ctx) } }),
      ]),
    ]),
  );

  if (list.length === 0) {
    inner.append(
      el("div", { class: "empty" }, [
        el("div", { class: "et", text: "No sets yet" }),
        el("p", { class: "es", text: "Add your models first (Models), then create a set: pick how it listens, thinks and speaks, and its voice." }),
        el("button", { class: "btn pri", type: "button", text: "＋ New set", on: { click: () => newProfile(ctx) } }),
      ]),
    );
    return;
  }

  const on = list.filter((p) => p.enabled);
  const off = list.filter((p) => !p.enabled);
  if (on.length) {
    inner.append(sectionLabel("On — somora can call these"));
    for (const p of on) inner.append(profileCard(p, ctx, choices));
  }
  if (off.length) {
    inner.append(sectionLabel("Off"));
    for (const p of off) inner.append(profileCard(p, ctx, choices));
  }
}

function sectionLabel(text) {
  return el("div", { class: "seclabel" }, [el("span", { text })]);
}

async function newProfile(ctx) {
  const values = await formDialog({
    title: "New set",
    description: "Its name is what you give an agent in somora as its voice. The models and the voice are chosen on the next screen.",
    fields: [
      { name: "id", label: "Name for somora", placeholder: "anna", required: true, hint: "lowercase letters, digits and dashes; cannot be changed later" },
      { name: "display_name", label: "Display name", placeholder: "Anna" },
    ],
    confirmLabel: "Create set",
    validate: (v) => {
      if (!v.id.trim()) return "A name is required.";
      if (!ID_PATTERN.test(v.id.trim().toLowerCase())) return idError();
      return "";
    },
  });
  if (!values) return;
  const id = values.id.trim().toLowerCase();
  const displayName = (values.display_name || "").trim() || id;
  try {
    await profilesApi.create({ id, display_name: displayName, spec: { language: "de" } });
    await ctx.loadProfiles();
    ctx.go(`#/profiles/${encodeURIComponent(id)}`);
  } catch (err) {
    await noticeDialog({ title: "Could not create the set", message: err.message, detail: "Nothing was created." });
  }
}
