/* Tiny DOM helpers (no framework, no build step). */

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) node.addEventListener(ev, fn);
    else if (k in node && k !== "list" && typeof v !== "object") node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    if (child == null || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function qs(sel, root = document) {
  return root.querySelector(sel);
}

export function qsa(sel, root = document) {
  return Array.from(root.querySelectorAll(sel));
}

/** Build a CSS-shape waveform mark from a seed (deterministic, info-carrying). */
export function waveform(seed = "x", bars = 5, maxHeight = 14) {
  const wrap = el("span", { class: "wv", "aria-hidden": "true" });
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  for (let i = 0; i < bars; i++) {
    h = (h * 1103515245 + 12345) >>> 0;
    const height = 4 + (h % maxHeight);
    wrap.append(el("i", { style: `height:${height}px` }));
  }
  return wrap;
}

export function dot(colorClass) {
  return el("span", { class: "sdot " + (colorClass || ""), "aria-hidden": "true" });
}

/** Copy text to the clipboard with a visible, accessible fallback. */
export async function copyText(text, button) {
  const original = button ? button.textContent : "";
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = el("textarea", { value: text, style: "position:fixed;opacity:0" });
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    if (button) {
      button.textContent = "Copied";
      setTimeout(() => (button.textContent = original), 1400);
    }
    return true;
  } catch {
    return false;
  }
}

/** Standard error bar following: what failed → why (code) → NOT happened → next. */
export function errorBar(parts, variant = "err") {
  const glyph = variant === "ok" ? "✓" : variant === "warn" ? "▲" : variant === "note" ? "○" : "▲";
  return el("div", { class: "bar bar-" + variant, role: variant === "err" ? "alert" : undefined }, [
    el("span", { class: "g", "aria-hidden": "true", text: glyph }),
    el("div", {}, parts.map((p) => (p.nodeType ? p : el("span", {}, [p])))),
  ]);
}

export function spinner(label) {
  return el("span", { class: "stagestate", style: "display:inline-flex;align-items:center;gap:7px" }, [
    el("span", { class: "spin", "aria-hidden": "true" }),
    label || "Loading…",
  ]);
}