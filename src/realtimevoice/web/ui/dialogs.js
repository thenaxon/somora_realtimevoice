/* In-app accessible dialogs (F3/P2).
 *
 * Replaces every native prompt()/confirm()/alert() in the web view so that
 * creation, confirmation and errors carry the app's palette, its structure
 * ("what did NOT happen") and an accessible modal pattern (role=dialog,
 * aria-modal, labelled title, focus trap, Escape/Cancel, initial focus,
 * focus restored on close).
 *
 * API (all return promises):
 *   formDialog({ title, description, fields, confirmLabel, validate })
 *       -> object of field values, or null when cancelled
 *   confirmDialog({ title, message, confirmLabel, danger, detail })
 *       -> true/false
 *   noticeDialog({ title, message, tone }) -> void
 *
 * No native dialog is ever invoked, so the functional e2e can assert that no
 * browser dialog event fires.
 */

import { el, clear } from "./dom.js";

let dialogSeq = 0;

function buildOverlay() {
  const overlay = el("div", { class: "dlg-overlay" });
  const dialog = el("div", {
    class: "dlg",
    role: "dialog",
    "aria-modal": "true",
    tabindex: "-1",
  });
  overlay.append(dialog);
  return { overlay, dialog };
}

function focusable(root) {
  return Array.from(
    root.querySelectorAll(
      'a[href],button:not([disabled]),input:not([disabled]):not([type=hidden]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])',
    ),
  ).filter((n) => n.offsetParent !== null || n === document.activeElement);
}

/** Mount a dialog; `render(body, api)` fills it. Returns a promise. */
function openDialog(render) {
  const { overlay, dialog } = buildOverlay();
  const previouslyFocused = document.activeElement;
  const titleId = `dlg-title-${++dialogSeq}`;
  dialog.setAttribute("aria-labelledby", titleId);

  let settled = false;
  let resolveOut;
  const done = (value) => {
    if (settled) return;
    settled = true;
    document.removeEventListener("keydown", onKey, true);
    overlay.remove();
    try {
      previouslyFocused && previouslyFocused.focus && previouslyFocused.focus();
    } catch {
      /* ignore */
    }
    resolveOut(value);
  };

  const api = {
    dialog,
    titleId,
    close: (value) => done(value),
    cancel: () => done(null),
    setError: (node, msg) => {
      if (!node) return;
      node.textContent = msg || "";
      node.classList.toggle("hidden", !msg);
    },
  };

  function onKey(ev) {
    if (ev.key === "Escape") {
      ev.preventDefault();
      // let the dialog decide (a busy state can veto); default is cancel
      if (!dialog.dataset.busy) done(null);
      return;
    }
    if (ev.key !== "Tab") return;
    const items = focusable(dialog);
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (ev.shiftKey && document.activeElement === first) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus();
    }
  }

  const body = el("div", { class: "dlg-body" });
  dialog.append(body);
  render(body, api);

  const promise = new Promise((resolve) => (resolveOut = resolve));
  overlay.addEventListener("mousedown", (ev) => {
    if (ev.target === overlay && !dialog.dataset.busy) done(null);
  });
  document.body.append(overlay);
  document.addEventListener("keydown", onKey, true);
  requestAnimationFrame(() => {
    const items = focusable(dialog);
    (items[0] || dialog).focus();
  });
  return promise;
}

function titleBar(title) {
  return el("div", { class: "dlg-title" });
}

/**
 * Modal form. `fields` is a list of
 *   { name, label, type, value, placeholder, hint, required, maxlength, pattern }
 * Returns { [name]: string } on confirm, or null on cancel.
 */
export function formDialog({ title, description, fields = [], confirmLabel = "Save", cancelLabel = "Cancel", validate } = {}) {
  return openDialog((body, api) => {
    const inputs = {};
    const errorLine = el("div", { class: "dlg-error hidden", role: "alert" });
    body.append(el("h2", { class: "dlg-title", id: api.titleId, text: title || "" }));
    if (description) body.append(el("p", { class: "dlg-desc", text: description }));

    for (const f of fields) {
      const input = el("input", {
        class: "inp",
        type: f.type || "text",
        value: f.value != null ? f.value : "",
        placeholder: f.placeholder || "",
        maxlength: f.maxlength,
        "aria-label": f.label,
        autocomplete: "off",
      });
      inputs[f.name] = input;
      body.append(
        el("div", { class: "fld" }, [
          el("label", { text: f.label + (f.required ? " (required)" : "") }),
          input,
          f.hint ? el("div", { class: "fhint", text: f.hint }) : null,
        ]),
      );
    }

    const confirm = el("button", { class: "btn pri", type: "button", text: confirmLabel });
    const cancel = el("button", { class: "btn", type: "button", text: cancelLabel });
    body.append(el("div", { class: "dlg-actions" }, [confirm, cancel]), errorLine);

    cancel.addEventListener("click", () => api.cancel());
    const submit = () => {
      const values = {};
      for (const f of fields) values[f.name] = inputs[f.name].value;
      const msg = validate ? validate(values) : "";
      if (msg) {
        api.setError(errorLine, msg);
        return;
      }
      api.close(values);
    };
    confirm.addEventListener("click", submit);
    body.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && ev.target.tagName !== "TEXTAREA") {
        ev.preventDefault();
        submit();
      }
    });
  });
}

/** Danger/normal confirmation. Returns true only on explicit confirm. */
export function confirmDialog({ title, message, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false, detail } = {}) {
  return openDialog((body, api) => {
    body.append(el("h2", { class: "dlg-title", id: api.titleId, text: title || "Confirm" }));
    if (message) body.append(el("p", { class: "dlg-desc", text: message }));
    if (detail) body.append(el("p", { class: "dlg-desc dlg-muted", text: detail }));
    const confirm = el("button", { class: "btn " + (danger ? "danger" : "pri"), type: "button", text: confirmLabel });
    const cancel = el("button", { class: "btn", type: "button", text: cancelLabel });
    body.append(el("div", { class: "dlg-actions" }, [confirm, cancel]));
    confirm.addEventListener("click", () => api.close(true));
    cancel.addEventListener("click", () => api.cancel());
  }).then((v) => v === true);
}

/** Informational error/notice. Resolves when dismissed. */
export function noticeDialog({ title, message, tone = "warn", detail } = {}) {
  return openDialog((body, api) => {
    body.append(el("h2", { class: `dlg-title dlg-${tone}`, id: api.titleId, text: title || "Notice" }));
    if (message) body.append(el("p", { class: "dlg-desc", text: message }));
    if (detail) body.append(el("p", { class: "dlg-desc dlg-muted", text: detail }));
    const ok = el("button", { class: "btn pri", type: "button", text: "OK" });
    ok.addEventListener("click", () => api.close(undefined));
    body.append(el("div", { class: "dlg-actions" }, [ok]));
  });
}