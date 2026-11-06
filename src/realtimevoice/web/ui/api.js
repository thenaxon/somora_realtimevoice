/* API client for the management REST endpoints.
 *
 * All calls carry `Authorization: Bearer <token>`. Errors are normalised into
 * an ApiError carrying the HTTP status, the machine-readable `code` and the
 * `issues` / `probes` payloads the contract returns (routes_profiles.py), so
 * views can render the mandated error pattern instead of a bare message.
 */

export const TOKEN_KEY = "rtv_token";

export class ApiError extends Error {
  constructor(status, code, message, body) {
    super(message || code || `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.code = code || "";
    this.body = body || {};
    this.issues = (body && body.issues) || [];
    this.probes = (body && body.probes) || null;
  }
}

export class AuthError extends ApiError {
  constructor() {
    super(401, "unauthorized", "Invalid or expired token");
    this.name = "AuthError";
  }
}

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  else sessionStorage.removeItem(TOKEN_KEY);
}

function emitUnauthorized() {
  window.dispatchEvent(new CustomEvent("rtv:unauthorized"));
}

async function parseError(resp) {
  let body = {};
  try {
    body = await resp.json();
  } catch {
    body = {};
  }
  const code = body.code || "";
  const message = body.detail || resp.statusText || `HTTP ${resp.status}`;
  return new ApiError(resp.status, code, message, body);
}

async function request(path, { method = "GET", body, headers, raw = false } = {}) {
  const h = { Authorization: "Bearer " + getToken(), ...(headers || {}) };
  const opts = { method, headers: h };
  if (body !== undefined) {
    if (body instanceof FormData) {
      opts.body = body;
    } else {
      h["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  const resp = await fetch(path, opts);
  if (resp.status === 401) {
    emitUnauthorized();
    throw new AuthError();
  }
  if (!resp.ok) throw await parseError(resp);
  if (raw) return resp;
  if (resp.status === 204) return null;
  return resp.json();
}

export const api = {
  get: (path, opts) => request(path, { ...opts, method: "GET" }),
  post: (path, body, opts) => request(path, { ...opts, method: "POST", body }),
  put: (path, body, opts) => request(path, { ...opts, method: "PUT", body }),
  patch: (path, body, opts) => request(path, { ...opts, method: "PATCH", body }),
  del: (path, opts) => request(path, { ...opts, method: "DELETE" }),
  raw: (path, opts) => request(path, { ...opts, raw: true }),
};

/* ---------------------------------- sets --------------------------------- */

export const profiles = {
  list: () => api.get("/api/conversation-profiles"),
  get: (id) => api.get(`/api/conversation-profiles/${encodeURIComponent(id)}`),
  create: (payload) => api.post("/api/conversation-profiles", payload),
  // saving is what calls use from now on; an enabled set must be valid
  save: (id, payload) => api.put(`/api/conversation-profiles/${encodeURIComponent(id)}`, payload),
  remove: (id) => api.del(`/api/conversation-profiles/${encodeURIComponent(id)}`),
  validate: (id, spec) =>
    api.post(`/api/conversation-profiles/${encodeURIComponent(id)}/validate`, spec ? { spec } : {}),
  // index into the set's earlier versions (0 = newest)
  restore: (id, index) => api.post(`/api/conversation-profiles/${encodeURIComponent(id)}/restore`, { index }),
  // `spec` is the editor's CURRENT (possibly unsaved) selection; the server
  // tests it instead of the saved set and persists nothing
  testSession: (id, spec) =>
    api.post(`/api/conversation-profiles/${encodeURIComponent(id)}/test-session`, spec ? { spec } : {}),
  // speak the typed sentence with the current (possibly unsaved) selection
  // through the conversation materialisation. Raw WAV response.
  testSpeech: (id, text, spec) =>
    api.raw(`/api/conversation-profiles/${encodeURIComponent(id)}/test-speech`, {
      method: "POST",
      body: { text, ...(spec ? { spec } : {}) },
    }),
};

/* --------------------------------- voices -------------------------------- */

export const voices = {
  list: () => api.get("/api/voices"),
  get: (id) => api.get(`/api/voices/${encodeURIComponent(id)}`),
  create: (payload) => api.post("/api/voices", payload),
  save: (id, payload) => api.patch(`/api/voices/${encodeURIComponent(id)}`, payload),
  remove: (id) => api.del(`/api/voices/${encodeURIComponent(id)}`),
  uploadSample: (id, blob, filename, source) => {
    const fd = new FormData();
    fd.append("file", blob, filename || "sample.wav");
    fd.append("source", source || "upload");
    return api.post(`/api/voices/${encodeURIComponent(id)}/samples`, fd);
  },
  transcribe: (id, sampleId, sttProfile) =>
    api.post(
      `/api/voices/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/transcribe`,
      sttProfile ? { stt_profile: sttProfile } : {},
    ),
  patchSample: (id, sampleId, payload) =>
    api.patch(`/api/voices/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}`, payload),
  removeSample: (id, sampleId) =>
    api.del(`/api/voices/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}`),
  // F10: cut [startMs, endMs) into a NEW sample; the original is never changed
  excerpt: (id, sampleId, startMs, endMs) =>
    api.post(`/api/voices/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/excerpt`, {
      start_ms: Math.round(startMs),
      end_ms: Math.round(endMs),
    }),
  // every set speaking with the voice uses it from the next call
  setReference: (id, sampleId) =>
    api.put(`/api/voices/${encodeURIComponent(id)}/reference`, { sample_id: sampleId }),
  sampleAudioUrl: (id, sampleId) =>
    `/api/voices/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/audio`,
  preview: (id, ttsProfile, text, sampleId) =>
    request(`/api/voices/${encodeURIComponent(id)}/preview`, {
      method: "POST",
      body: { tts_profile: ttsProfile, text, ...(sampleId ? { sample_id: sampleId } : {}) },
      raw: true,
    }),
};

/* ------------------------------- tts engines ----------------------------- */

export const ttsEngines = {
  list: () => api.get("/api/tts-engines"),
};

/* ------------------------------- catalog/settings ------------------------ */

export const catalog = {
  get: () => api.get("/api/catalog"),
  version: () => api.get("/api/version"),
  settings: () => api.get("/api/settings"),
  saveSettings: (payload) => api.put("/api/settings", payload),
  // R2/CP3: `profile` lets a form probe an UNSAVED row before saving. `id`
  // names the stored row that supplies the key when the form's key is blank.
  probe: (role, id, profile) =>
    api.post("/api/probe-profile", profile ? { role, id, profile } : { role, id }),
  saveProfile: (role, id, payload) =>
    api.put(`/api/profiles/${encodeURIComponent(role)}/${encodeURIComponent(id)}`, payload),
  createProfile: (role, payload) => api.post(`/api/profiles/${encodeURIComponent(role)}`, payload),
  cloneProfile: (role, id, payload) =>
    api.post(`/api/profiles/${encodeURIComponent(role)}/${encodeURIComponent(id)}/clone`, payload),
  deleteProfile: (role, id) =>
    api.del(`/api/profiles/${encodeURIComponent(role)}/${encodeURIComponent(id)}`),
  /* GET /v1/models (contract §7) is not present in this worktree yet; callers
     must catch and degrade. */
  models: () => api.get("/v1/models"),
};