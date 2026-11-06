/* realtimevoice management UI entry.
 *
 * Wires the shared shell, hash router, auth (token in sessionStorage) and the
 * view modules. Views receive a small context: {shell, catalog, refreshCatalog,
 * go, token}.
 */

import { el, clear, errorBar, qs } from "./dom.js";
import { api, ApiError, AuthError, getToken, setToken, profiles as profilesApi, voices as voicesApi, catalog as catalogApi } from "./api.js";
import { Shell, parseRoute, queryParam } from "./shell.js";
import { renderProfilesList } from "./view-profiles-list.js";
import { renderProfileEditor } from "./view-profile-editor.js";
import { renderVoiceLibrary } from "./view-voices-list.js";
import { renderVoiceEditor } from "./view-voice-editor.js";
import { renderTestConsole } from "./view-test.js";
import { renderBackends } from "./view-backends.js";
import { renderSettings } from "./view-settings.js";

const app = {
  shell: null,
  catalog: null,
  voices: [],
  profiles: [],
  backendHealth: null,
  revision: "",
  version: "",
  token: getToken(),
  // client-side view of which profile the in-page Test Console is currently
  // using; the backend offers no session-listing endpoint, so this is the
  // only honest source for an "In use now" badge in this UI
  inUseIds: new Set(),
};

/* --------------------------------------------------------------- catalog */

async function refreshCatalog() {
  const cat = await catalogApi.get();
  app.catalog = cat;
  app.shell.setFooter(renderFooter);
  return cat;
}

function roleProfiles(role) {
  const roles = (app.catalog && app.catalog.roles) || {};
  return ((roles[role] && roles[role].profiles) || []).slice();
}

function renderFooter() {
  let backends = null;
  if (app.backendHealth) backends = app.backendHealth;
  return { liveCalls: app.liveCalls, backends, revision: app.revision, version: app.version };
}

/** Calls running on the server now (any client), for the footer: read on
 *  every navigation and every 10 s while signed in. */
async function loadLiveCalls() {
  try {
    const n = (await api.get("/api/session-auth-check")).active_sessions;
    const changed = n !== app.liveCalls;
    app.liveCalls = n;
    if (changed && app.shell) app.shell.renderFooter();
  } catch {
    app.liveCalls = undefined;
  }
}
setInterval(() => {
  if (app.token && app.catalog && document.visibilityState === "visible") void loadLiveCalls();
}, 10_000);

async function loadProfiles() {
  const res = await profilesApi.list();
  app.profiles = res.profiles || [];
  return app.profiles;
}

async function loadVoices() {
  const res = await voicesApi.list();
  app.voices = res.voices || [];
  return app.voices;
}

/* ------------------------------------------------------------------ auth */

function showLogin(message) {
  app.shell.show(false);
  const login = qs("#login-view");
  login.classList.remove("hidden");
  qs("#login-status").textContent = message || "";
  qs("#login-status").className = "statusline " + (message ? "err" : "");
}

async function doLogin(token) {
  const value = (token || "").trim();
  if (!value) {
    qs("#login-status").textContent = "Enter an access token.";
    qs("#login-status").className = "statusline err";
    return false;
  }
  setToken(value);
  app.token = value;
  try {
    await api.get("/api/settings");
  } catch (err) {
    if (err instanceof AuthError) {
      setToken("");
      qs("#login-status").textContent = "Invalid token.";
      qs("#login-status").className = "statusline err";
      return false;
    }
    qs("#login-status").textContent = "Could not reach the server: " + err.message;
    qs("#login-status").className = "statusline err";
    return false;
  }
  qs("#login-view").classList.add("hidden");
  app.shell.show(true);
  await bootstrapData();
  if (!location.hash || location.hash === "#") location.hash = "#/profiles";
  await route();
  return true;
}

function logout() {
  setToken("");
  app.token = "";
  if (app.shell) app.shell.show(false);
  showLogin("Signed out.");
}

window.addEventListener("rtv:unauthorized", () => {
  setToken("");
  app.token = "";
  showLogin("Session expired — sign in again.");
});

/* --------------------------------------------------------------- routing */

// One navigation at a time: two at once (sign-in and a hash change) both
// cleared the page and both drew into it, so a list showed twice.
let routing = Promise.resolve();
function route() {
  routing = routing.then(routeNow, routeNow);
  return routing;
}

async function routeNow() {
  if (!app.token) {
    showLogin();
    return;
  }
  if (!app.catalog) {
    try {
      await bootstrapData();
    } catch (err) {
      if (err instanceof AuthError) return; // handler shows login
      renderFatal(err);
      return;
    }
  }
  const r = parseRoute(location.hash);
  const inner = app.shell.inner();
  // a view may hold live resources (the Test Console socket/mic); release them
  if (app.cleanup) {
    try {
      app.cleanup();
    } catch {}
    app.cleanup = null;
  }
  clear(inner);
  // refresh list data on every navigation so availability/status is current
  await Promise.all([loadProfiles(), loadVoices(), loadLiveCalls()]).catch(() => {});
  const ctx = {
    setCleanup: (fn) => {
      app.cleanup = fn;
    },
    inUseIds: app.inUseIds,
    markInUse: (id, on) => {
      if (on) app.inUseIds.add(id);
      else app.inUseIds.delete(id);
    },
    shell: app.shell,
    catalog: app.catalog,
    revision: app.revision,
    version: app.version,
    roleProfiles,
    profiles: app.profiles,
    voices: app.voices,
    refreshCatalog,
    loadProfiles,
    loadVoices,
    go: (hash) => app.shell.go(hash),
  };
  try {
    switch (r.view) {
      case "profiles":
        app.shell.setActive("profiles", { title: "Sets" });
        await renderProfilesList(inner, ctx);
        break;
      case "profile-editor":
        app.shell.setActive("profiles", { title: "Set" });
        await renderProfileEditor(inner, ctx, r.param);
        break;
      case "voices":
        app.shell.setActive("voices", { title: "Voices" });
        await renderVoiceLibrary(inner, ctx);
        break;
      case "voice-editor":
        app.shell.setActive("voices", { title: "Voice editor" });
        await renderVoiceEditor(inner, ctx, r.param);
        break;
      case "test":
        app.shell.setActive("test", { title: "Test" });
        await renderTestConsole(inner, ctx, queryParam(location.hash, "profile"), queryParam(location.hash, "ticket"));
        break;
      case "backends":
        app.shell.setActive("backends", { title: "Models" });
        await renderBackends(inner, ctx);
        break;
      case "settings":
        app.shell.setActive("settings", { title: "Settings" });
        await renderSettings(inner, ctx);
        break;
      default:
        app.shell.go("#/profiles");
    }
    app.shell.renderFooter();
  } catch (err) {
    if (err instanceof AuthError) return;
    renderFatal(err);
  }
}

function renderFatal(err) {
  const inner = app.shell.inner();
  clear(inner);
  inner.append(
    errorBar([
      el("b", { text: "Could not load this view. " }),
      err.message || String(err),
      el("b", { text: " Nothing was changed." }),
      " Check the connection and try again.",
    ]),
  );
}

async function bootstrapData() {
  await refreshCatalog();
  try {
    const v = await catalogApi.version();
    app.revision = v.revision || "";
    app.version = v.version || "";
  } catch {
    /* the stamp is informational; a failed fetch must not block the UI */
  }
  await Promise.all([loadProfiles(), loadVoices()]).catch(() => {});
}

/* ------------------------------------------------------------------ start */

function init() {
  const shell = new Shell({ onNavigate: () => route(), onSignOut: logout });
  app.shell = shell;
  shell.mount(qs("#shell-mount"));
  shell.show(false);

  qs("#btn-login").addEventListener("click", () => doLogin(qs("#token").value));
  qs("#token").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doLogin(qs("#token").value);
  });
  window.addEventListener("hashchange", route);

  if (app.token) {
    api
      .get("/api/settings")
      .then(async () => {
        qs("#login-view").classList.add("hidden");
        shell.show(true);
        await bootstrapData();
        await route();
      })
      .catch(() => showLogin());
  } else {
    showLogin();
  }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();