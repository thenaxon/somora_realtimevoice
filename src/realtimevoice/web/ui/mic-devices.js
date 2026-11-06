/* Microphone device selection shared by the voice recorder and Test Console (F2).
 *
 * Behaviour:
 *   - after permission, enumerate audioinput devices with labels;
 *   - "System default" option (deviceId "");
 *   - the selected deviceId is used as {deviceId: {exact: id}};
 *   - the choice is remembered in localStorage;
 *   - `devicechange` refreshes the list;
 *   - if the remembered device is gone, a truthful notice is shown and the
 *     system default is used (never silently pretending);
 *   - callers stop all tracks when switching/stopping.
 *
 * The constraints builder and the "remembered device missing" decision are
 * pure functions so a node unit test can cover them without a browser.
 */

export const MIC_STORAGE_KEY = "rtv_mic_device_id";
export const SYSTEM_DEFAULT = "";

/** Build getUserMedia audio constraints for a deviceId (or system default). */
export function micConstraints(deviceId, extra) {
  const audio = { ...(extra || {}) };
  if (deviceId) audio.deviceId = { exact: deviceId };
  return { audio };
}

/**
 * Decide how a remembered device resolves against the current list.
 * Returns { deviceId, missing } where missing=true means the remembered id
 * is no longer present and the system default must be used + reported.
 */
export function resolveRememberedDevice(remembered, devices) {
  const inputs = (devices || []).filter((d) => d.kind === "audioinput");
  if (!remembered) return { deviceId: SYSTEM_DEFAULT, missing: false };
  const found = inputs.some((d) => d.deviceId === remembered);
  return found
    ? { deviceId: remembered, missing: false }
    : { deviceId: SYSTEM_DEFAULT, missing: true };
}

/** Read the remembered device id (empty string = system default). */
export function rememberedDeviceId() {
  try {
    return localStorage.getItem(MIC_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function rememberDeviceId(deviceId) {
  try {
    if (deviceId) localStorage.setItem(MIC_STORAGE_KEY, deviceId);
    else localStorage.removeItem(MIC_STORAGE_KEY);
  } catch {
    /* storage unavailable: selection still works for this session */
  }
}

/**
 * Enumerate audio input devices. Labels are empty until permission was
 * granted once; callers should request permission first.
 */
export async function listAudioInputs() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "audioinput");
  } catch {
    return [];
  }
}

/**
 * Obtain a microphone stream for a deviceId and report the ACTIVE track's
 * truthful label/deviceId (from track.label / getSettings().deviceId).
 */
export async function openMic(deviceId) {
  const stream = await navigator.mediaDevices.getUserMedia(
    micConstraints(deviceId, {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    }),
  );
  const track = stream.getAudioTracks()[0];
  const settings = (track && track.getSettings && track.getSettings()) || {};
  return {
    stream,
    activeLabel: (track && track.label) || "",
    activeDeviceId: settings.deviceId || deviceId || "",
  };
}

/** Stop every track on a stream (used on switch/stop). */
export function stopStream(stream) {
  if (!stream) return;
  try {
    stream.getTracks().forEach((t) => t.stop());
  } catch {
    /* ignore */
  }
}

/**
 * Create a microphone picker control. Returns { node, refresh, value() }.
 *
 * `enumerateDevices()` is called only lazily (when the operator opens the
 * picker) and after permission is granted — some embedded WebKit builds crash
 * when it is called before any device access, and labels are empty without
 * permission anyway. `onChange(deviceId, missing)` fires after a refresh.
 */
export async function createMicPicker({ onChange, label } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "micpick";
  const sel = document.createElement("select");
  sel.className = "inp";
  sel.setAttribute("aria-label", label || "Microphone");
  const notice = document.createElement("div");
  notice.className = "fhint";
  notice.setAttribute("role", "status");
  wrap.append(sel, notice);
  let missing = false;
  let loaded = false;

  function fill(devices, remembered) {
    sel.textContent = "";
    const sys = document.createElement("option");
    sys.value = SYSTEM_DEFAULT;
    sys.textContent = "System default";
    sel.append(sys);
    for (const d of devices) {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${sel.options.length}`;
      sel.append(opt);
    }
    const resolved = resolveRememberedDevice(remembered, devices);
    missing = resolved.missing;
    sel.value = resolved.deviceId;
    notice.textContent = missing ? "Selected microphone not available — using system default" : "";
    notice.style.display = missing ? "" : "none";
    onChange && onChange(sel.value, missing);
  }

  // safe initial render: no enumerateDevices() before permission
  fill([], rememberedDeviceId());

  async function refresh() {
    const devices = await listAudioInputs();
    loaded = true;
    fill(devices, rememberedDeviceId());
    return devices;
  }

  // lazily load the real device list the first time the picker is used
  const lazy = () => {
    if (!loaded) refresh().catch(() => {});
  };
  sel.addEventListener("pointerdown", lazy);
  sel.addEventListener("focus", lazy);

  sel.addEventListener("change", () => {
    rememberDeviceId(sel.value);
    missing = false;
    notice.style.display = "none";
    onChange && onChange(sel.value, false);
  });

  return { node: wrap, refresh, value: () => sel.value, select: sel, loaded: () => loaded };
}

/** Attach a `devicechange` listener that refreshes the list; safe to call. */
export function watchDeviceChanges(onChange) {
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", () => onChange());
    }
  } catch {
    /* ignored: the list can still be refreshed manually */
  }
}