/* App shell + hash router.
 *
 * One shell for every view (app-shell-and-contrast-v1.4): persistent sidebar,
 * active nav state with three cues, breadcrumbs, and a sticky narrow-view
 * topbar with burger overlay + section chips. Views are mounted into
 * `.inner` by the router; a detail editor marks its list nav entry.
 */

import { el, clear, qs } from "./dom.js";

export const NAV = [
  { key: "profiles", glyph: "▤", label: "Sets", chip: "Sets", route: "#/profiles" },
  { key: "voices", glyph: "◍", label: "Voices", chip: "Voices", route: "#/voices" },
  { key: "test", glyph: "▶", label: "Test", chip: "Test", route: "#/test" },
  { key: "backends", glyph: "⚙", label: "Models", chip: "Models", route: "#/backends" },
  { key: "settings", glyph: "☰", label: "Settings", chip: "Settings", route: "#/settings" },
];

const ROUTES = [
  { re: /^#\/profiles\/([^/?]+)$/, view: "profile-editor", nav: "profiles" },
  { re: /^#\/profiles\/?$/, view: "profiles", nav: "profiles" },
  { re: /^#\/voices\/([^/?]+)$/, view: "voice-editor", nav: "voices" },
  { re: /^#\/voices\/?$/, view: "voices", nav: "voices" },
  { re: /^#\/test\/?/, view: "test", nav: "test" },
  { re: /^#\/backends\/?/, view: "backends", nav: "backends" },
  { re: /^#\/settings\/?/, view: "settings", nav: "settings" },
];

export function parseRoute(hash) {
  const h = hash || "#/profiles";
  for (const r of ROUTES) {
    const m = h.split("?")[0].match(r.re);
    if (m) return { view: r.view, nav: r.nav, param: m[1] ? decodeURIComponent(m[1]) : "", raw: h };
  }
  return { view: "profiles", nav: "profiles", param: "", raw: h };
}

export function queryParam(hash, name) {
  const q = (hash || "").split("?")[1];
  if (!q) return "";
  return new URLSearchParams(q).get(name) || "";
}

export class Shell {
  constructor({ onNavigate, footer, onSignOut }) {
    this.onNavigate = onNavigate;
    this.footerEl = footer || null;
    // account actions (Sign out) live INSIDE the shell
    // (sidebar footer on desktop, the same overlay on narrow view) — never a
    // position:fixed element floating over content.
    this.onSignOut = onSignOut || (() => {});
    this.root = null;
    this.sideEl = null;
    this.topbarEl = null;
    this.signoutEl = null;
    this.current = null;
  }

  mount(mountEl) {
    this.root = mountEl;
    this.render();
  }

  setFooter(footerProvider) {
    this.footerEl = footerProvider;
    this.renderFooter();
  }

  show(show) {
    if (this.root) this.root.classList.toggle("hidden", !show);
  }

  render() {
    if (!this.root) return;
    clear(this.root);
    const side = el("div", { class: "side", id: "shell-side" });
    side.append(
      el("div", { class: "brandmark" }, [
        el("div", { class: "dot9", "aria-hidden": "true" }),
        el("div", {}, [
          el("div", { class: "bname", text: "somora_realtimevoice" }),
          el("div", { class: "bsub mono", text: "voice for somora" }),
        ]),
      ]),
    );
    const nav = el("nav", { class: "nav", "aria-label": "Primary" });
    for (const item of NAV) {
      nav.append(
        el(
          "button",
          {
            class: "navi",
            dataset: { nav: item.key },
            type: "button",
            on: { click: () => this.go(item.route) },
          },
          [el("span", { class: "navglyph", "aria-hidden": "true", text: item.glyph }), item.label],
        ),
      );
    }
    side.append(nav);
    const sidefoot = el("div", { class: "sidefoot", id: "shell-foot" });
    side.append(sidefoot);

    const right = el("div", {});
    const topbar = el("div", { class: "topbar-m" });
    const burger = el("button", {
      class: "burger",
      type: "button",
      "aria-label": "Open navigation",
      on: {
        click: () => {
          const open = side.classList.toggle("open");
          burger.setAttribute("aria-expanded", String(open));
        },
      },
    });
    for (let i = 0; i < 3; i++) burger.append(el("i", { "aria-hidden": "true" }));
    const mtitle = el("div", { class: "mtitle", id: "shell-mtitle" });
    const mchips = el("div", { class: "mchips" });
    for (const item of NAV) {
      if (item.key === "backends") continue;
      mchips.append(
        el("button", {
          class: "mchip",
          dataset: { chip: item.key },
          type: "button",
          text: item.chip,
          on: { click: () => this.go(item.route) },
        }),
      );
    }
    topbar.append(burger, mtitle, mchips);

    const content = el("div", { class: "content" });
    const inner = el("div", { class: "inner", id: "shell-inner" });
    content.append(inner);
    right.append(topbar, content);
    this.root.append(el("div", { class: "shell" }, [side, right]));

    this.sideEl = side;
    this.topbarEl = topbar;
    this.mtitleEl = mtitle;
    this.innerEl = inner;
    this.renderFooter();
  }

  renderFooter() {
    const foot = qs("#shell-foot", this.root);
    if (!foot) return;
    clear(foot);
    const data = (this.footerEl && this.footerEl()) || {};
    if (typeof data.liveCalls === "number") {
      foot.append(
        el("div", { class: "sfrow" }, [
          el("span", { class: "sdot " + (data.liveCalls > 0 ? "dot-ready" : "dot-unknown") }),
          data.liveCalls === 1 ? "1 call running" : `${data.liveCalls} calls running`,
        ]),
      );
    }
    if (data.backends) {
      foot.append(
        el("div", { class: "sfrow", style: "margin-top:8px" }, [
          el("span", { class: "sdot " + (data.backends.ok ? "dot-ready" : "dot-warn") }),
          data.backends.text,
        ]),
      );
    }
    foot.append(this.accountActions());
  }

  accountActions() {
    const row = el("div", { class: "sfacts" });
    const data = (this.footerEl && this.footerEl()) || {};
    if (data.version || data.revision) {
      row.append(
        el("span", {
          class: "revstamp mono",
          id: "shell-rev",
          title: "Version and build revision",
          text: [data.version ? "v" + data.version : "", data.revision ? "build " + data.revision : ""].filter(Boolean).join(" · "),
        }),
      );
    }
    row.append(
      el("button", {
        class: "tbtn",
        type: "button",
        text: "Sign out",
        on: { click: () => this.onSignOut() },
      }),
    );
    return row;
  }

  go(route) {
    if (location.hash === route) {
      if (this.onNavigate) this.onNavigate(parseRoute(route));
      return;
    }
    location.hash = route;
  }

  setActive(navKey, { title, breadcrumb }) {
    for (const item of NAV) {
      qs(`.navi[data-nav="${item.key}"]`, this.root)?.classList.toggle("on", item.key === navKey);
      const chip = qs(`.mchip[data-chip="${item.key}"]`, this.root);
      if (chip) chip.classList.toggle("on", item.key === navKey);
    }
    if (this.mtitleEl) this.mtitleEl.textContent = title || "";
    this.sideEl?.classList.remove("open");
  }

  inner() {
    return this.innerEl;
  }

  /** Render a breadcrumb row. items: [{label, route?}] */
  breadcrumb(items) {
    const crumb = el("div", { class: "crumb" });
    items.forEach((item, i) => {
      if (i > 0) crumb.append(el("span", { class: "sep", "aria-hidden": "true", text: "›" }));
      if (item.route && i < items.length - 1) {
        crumb.append(el("a", { href: item.route, text: item.label }));
      } else {
        crumb.append(el("b", { text: item.label }));
      }
    });
    return crumb;
  }
}