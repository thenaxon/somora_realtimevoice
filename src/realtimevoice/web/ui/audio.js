/* Realtime audio session for the Test Console.
 *
 * Reuses the proven worklet pair (../worklet.js): capture-worklet posts mono
 * Int16 chunks, play-worklet buffers playback with explicit clear accounting.
 * This module keeps the two properties that were hard-won in the legacy
 * client and must not regress:
 *   - playback is flushed on stop and on barge-in (speech_started), so the
 *     stopped response's tail never keeps playing;
 *   - a stale socket/callback can never touch a newer session (generation
 *     fence on every async continuation and port message).
 *
 * The public protocol is /v1/realtime; auth travels as the
 * `rtv.bearer.<token>` WebSocket subprotocol.
 */

import { micConstraints } from "./mic-devices.js";

const SAMPLE_RATE = 24000;

/** Streaming linear resampler PCM16 in -> PCM16 out at the target rate.
 *  Keeps a fractional read position across chunks and one boundary sample so
 *  there is no click at chunk edges. */
export function makeResampler(inputRate, outputRate) {
  if (!inputRate || inputRate === outputRate) {
    return (samples) => samples;
  }
  const ratio = inputRate / outputRate;
  let pos = 0; // fractional index into the concatenated input stream
  let prev = 0; // last sample of the previous chunk
  return function resample(samples) {
    if (!samples.length) return samples;
    const n = samples.length;
    const out = [];
    while (pos < n - 1) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i < 0 ? prev : samples[i];
      const b = i + 1 < n ? samples[i + 1] : samples[n - 1];
      out.push((a + (b - a) * frac) | 0);
      pos += ratio;
    }
    // carry the fractional position and the last sample into the next chunk
    pos -= n;
    prev = samples[n - 1];
    return Int16Array.from(out);
  };
}

export const TALK_STATES = ["idle", "connecting", "route-pending", "listening", "thinking", "speaking"];

export class RealtimeSession {
  constructor({ token, onEvent, onState, onRouteUpdate, onError, deviceId }) {
    this.token = token;
    this.onEvent = onEvent || (() => {});
    this.onState = onState || (() => {});
    this.onRouteUpdate = onRouteUpdate || (() => {});
    this.onError = onError || (() => {});
    this.deviceId = deviceId || "";
    this.ws = null;
    this.wsModel = "";
    this.gen = 0;
    this.audioCtx = null;
    this.micStream = null;
    this.capture = null;
    this.playNode = null;
    this.recording = false;
    this.starting = false;
    this.state = "idle";
    this.activeResponseId = null;
    this.fenced = new Set();
    this.assistantText = "";
    this.metrics = {};
    this.routePending = false;
    this.routeResolved = false;
    this.activeMicLabel = "";
    this.activeMicDeviceId = "";
    this.nativeSampleRate = 0;
    this._resampler = null;
    // F11: playback lifecycle. `playbackActive` tracks whether the worklet ring
    // still holds audio; the UI only reports `listening` once it actually
    // drained (response.done arrives long before the browser finished playing).
    this.playbackActive = false;
    this._drainToListening = false;
    // Session turn-detection flag (baseline applyTurnDetection): cleared
    // playback on speech_started only when interrupt_response is on. Default
    // true when the session omits it, like the baseline.
    this.interruptResponse = true;
  }

  setState(next, detail) {
    if (this.state === next && !detail) return;
    this.state = next;
    this.onState(next, detail || "");
  }

  /** Create (once) and immediately resume the AudioContext on the gesture. */
  _ensureAudioContext() {
    if (this.audioCtx) return this.audioCtx;
    if (typeof AudioContext === "undefined") {
      throw new Error("this browser does not support Web Audio (AudioContext unavailable)");
    }
    const ctx = new AudioContext();
    this.audioCtx = ctx;
    this.nativeSampleRate = ctx.sampleRate;
    this._resampler = makeResampler(ctx.sampleRate, SAMPLE_RATE);
    // fire-and-forget resume: never await (Firefox can keep it suspended
    // until a later gesture; awaiting would hang the whole start)
    this._resumeContext();
    this._installGestureUnlock();
    return ctx;
  }

  _resumeContext() {
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      try {
        const p = this.audioCtx.resume();
        if (p && p.catch) p.catch(() => {});
      } catch {
        /* ignore */
      }
    }
  }

  /** Retry resume on the next user gesture while a session is starting. */
  _installGestureUnlock() {
    if (this._gestureUnlock) return;
    const unlock = () => {
      if (this.recording || this.starting) this._resumeContext();
    };
    this._gestureUnlock = unlock;
    try {
      window.addEventListener("pointerdown", unlock, { passive: true });
      window.addEventListener("keydown", unlock, { passive: true });
    } catch {
      /* ignore */
    }
  }

  _removeGestureUnlock() {
    if (!this._gestureUnlock) return;
    try {
      window.removeEventListener("pointerdown", this._gestureUnlock);
      window.removeEventListener("keydown", this._gestureUnlock);
    } catch {
      /* ignore */
    }
    this._gestureUnlock = null;
  }

  async connect(model) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/v1/realtime?model=${encodeURIComponent(model || "")}`;
    const ws = new WebSocket(url, ["rtv.bearer", "rtv.bearer." + this.token]);
    this.ws = ws;
    this.wsModel = model || "";
    const gen = ++this.gen;
    return new Promise((resolve, reject) => {
      let opened = false;
      ws.onopen = () => {
        opened = true;
        resolve(ws);
      };
      ws.onerror = () => {
        if (!opened) reject(new Error("WebSocket error"));
      };
      ws.onclose = (ev) => {
        // the server ended a running call (not our Stop): say how
        if (opened && this.ws === ws && ev.code !== 1000 && ev.code !== 1005) {
          this.onEvent({ type: "ui.connection_closed", code: ev.code, reason: ev.reason || "" });
        }
        if (this.ws === ws) {
          this.ws = null;
          this.wsModel = "";
          if (this.recording || this.starting) this.stop(true);
          this.setState("idle");
        }
        if (!opened) reject(new Error(ev.reason || "Connection closed before it opened"));
      };
      ws.onmessage = (ev) => {
        if (this.ws !== ws || gen !== this.gen) return;
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        this.handle(msg);
      };
    });
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
      return true;
    }
    return false;
  }

  sendAudio(samples) {
    const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 8192) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    }
    this.send({ type: "input_audio_buffer.append", audio: btoa(binary) });
  }

  decodePcm(base64) {
    const binary = atob(base64);
    const out = new Int16Array(binary.length / 2);
    for (let i = 0; i < out.length; i++) {
      out[i] = binary.charCodeAt(2 * i) | (binary.charCodeAt(2 * i + 1) << 8);
    }
    return out;
  }

  handle(msg) {
    const type = msg.type || "";
    this.onEvent(msg);
    switch (type) {
      case "session.created":
        this.applyRoute(msg.session);
        break;
      case "session.updated":
        this.applyRoute(msg.session);
        break;
      case "input_audio_buffer.speech_started":
        // Baseline 03cb679 app.js:320-338: with the session's interrupt flag
        // on, a speech_started UNCONDITIONALLY flushes playback — explicitly
        // including audio still buffered after a finished response
        // ("response.done can beat the last playback chunks"). With output
        // pacing off that tail is the common case: the server has no active
        // response to cancel, so the client itself must stop the assistant.
        // Echo protection is the microphone's AEC (see start()), not ignoring
        // this event. With interrupt_response false nothing is flushed.
        if (this.interruptResponse) {
          this.clearPlayback("speech_started");
          if (this.activeResponseId) this.fenced.add(this.activeResponseId);
          if (this.recording) this.setState("listening");
        }
        break;
      case "response.created":
        this.activeResponseId = msg.response && msg.response.id;
        this.assistantText = "";
        this.setState("thinking");
        break;
      case "response.output_audio_transcript.delta":
        if (this.isFenced(msg.response_id)) break;
        this.assistantText += msg.delta || "";
        break;
      case "response.output_audio.delta":
        if (this.isFenced(msg.response_id)) break;
        this.enqueueAudio(msg.delta);
        if (this.state !== "speaking") this.setState("speaking");
        break;
      case "response.done": {
        const rid = msg.response && msg.response.id;
        if (rid && this.activeResponseId && rid !== this.activeResponseId) break;
        const status = msg.response && msg.response.status;
        this.activeResponseId = null;
        if (status && status !== "completed") {
          // a genuine barge-in / cancel: flush the interrupted answer now and
          // fence it so no late delta can re-enter the ring
          if (rid) this.fenced.add(rid);
          this.clearPlayback(status);
          if (this.recording) this.setState("listening");
          else this.setState("idle");
        } else {
          // completed: the buffered tail deliberately keeps playing. Report
          // `listening` only once the worklet reports the ring drained; if
          // there is nothing queued, settle immediately.
          this._drainToListening = true;
          this._maybeFinishPlayback();
        }
        break;
      }
      case "error":
        if (msg.error && msg.error.code === "voice_route_pending") {
          this.routePending = true;
          this.setState("route-pending", msg.error.message || "");
        }
        break;
      default:
        break;
    }
  }

  isFenced(rid) {
    if (!rid) return false;
    return this.fenced.has(rid) || (this.activeResponseId && rid !== this.activeResponseId);
  }

  applyRoute(session) {
    if (!session) return;
    // track the session turn-detection flag (baseline applyTurnDetection):
    // default true when the session omits it
    const td = session.turn_detection;
    this.interruptResponse = !(td && td.interrupt_response === false);
    // x_realtimevoice is the contract field; realtimevoice is what the
    // backend in this worktree currently emits (documented API gap)
    const ext = session.x_realtimevoice || session.realtimevoice;
    if (ext && ext.route) {
      this.routePending = false;
      this.routeResolved = true;
      this.onRouteUpdate(ext.route, ext.warnings || []);
      // once a route resolves the session is listening (if recording)
      if (this.recording && this.state !== "speaking") this.setState("listening");
    } else if (this.recording) {
      this.routePending = true;
      this.setState("route-pending");
    }
  }

  /** Start talking. model is the client-facing string (profile:<id>,
   *  profile-test:<ticket>, voice-route:auto, ...). */
  async start(model) {
    if (this.starting || this.recording) return;
    this.starting = true;
    this.routeResolved = false;
    this.setState("connecting");
    // Firefox/Safari autoplay policy: the AudioContext must be created and
    // resumed INSIDE the user-gesture task. Do it synchronously before any
    // await so the click that started Talk still counts as the gesture.
    try {
      const ctx = this._ensureAudioContext();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("this browser does not provide microphone capture (getUserMedia unavailable)");
      }
      if (!ctx.audioWorklet || !ctx.audioWorklet.addModule) {
        throw new Error("this browser does not support the AudioWorklet capture path");
      }
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.wsModel !== (model || "")) {
        if (this.ws) {
          const old = this.ws;
          this.ws = null;
          try {
            old.close();
          } catch {}
        }
        await this.connect(model);
      }
      // connect() bumps this.gen; capture the CURRENT generation only now so
      // its internal fence can never invalidate this start (F4 root cause).
      const gen = this.gen;
      const stream = await navigator.mediaDevices.getUserMedia(
        micConstraints(this.deviceId, {
          // The LIVE conversation needs the browser's echo canceller: without
          // it the assistant's own speaker output re-enters the microphone and
          // fires a false `speech_started` during the playback tail (this cut
          // the answer short — F11). Same constraints as the proven baseline
          // (app.js 03cb679:783-784). The raw path is reserved for voice
          // samples for cloning (§8a, mic-devices.openMic), not the talk loop.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }),
      );
      if (gen !== this.gen) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.micStream = stream;
      const track = stream.getAudioTracks()[0];
      const settings = (track && track.getSettings && track.getSettings()) || {};
      this.activeMicLabel = (track && track.label) || "";
      this.activeMicDeviceId = settings.deviceId || this.deviceId || "";
      // the context was created on the gesture; make sure it is running now
      this._resumeContext();
      this.nativeSampleRate = ctx.sampleRate;
      this._resampler = makeResampler(ctx.sampleRate, SAMPLE_RATE);
      await ctx.audioWorklet.addModule("../worklet.js");
      if (gen !== this.gen) {
        this.releaseAudio();
        return;
      }
      const source = ctx.createMediaStreamSource(stream);
      const capture = new AudioWorkletNode(ctx, "capture-worklet");
      capture.port.onmessage = (ev) => {
        if (gen !== this.gen || !this.recording) return;
        const native = new Int16Array(ev.data);
        // resample native-rate PCM16 -> 24 kHz PCM16 before sending
        const resampled = this._resampler ? this._resampler(native) : native;
        if (resampled.length) this.sendAudio(resampled);
      };
      source.connect(capture);
      const playNode = new AudioWorkletNode(ctx, "play-worklet", { outputChannelCount: [1] });
      playNode.port.onmessage = (ev) => {
        if (gen !== this.gen) return;
        if (ev.data && ev.data.cmd === "overflow") {
          this.onEvent({ type: "ui.playback_overflow", dropped: ev.data.dropped });
        } else if (ev.data && ev.data.cmd === "idle") {
          // the playback ring actually drained: now the answer is really over
          this.playbackActive = false;
          this._maybeFinishPlayback();
        }
      };
      playNode.connect(ctx.destination);
      this.capture = capture;
      this.playNode = playNode;
      this.recording = true;
      // session.created may have resolved the route while we were still
      // wiring the audio graph (browser-dependent ordering); never clobber a
      // resolved route back to "waiting for route".
      if (this.routeResolved) {
        this.routePending = false;
        this.setState("listening");
      } else {
        this.routePending = true;
        this.setState("route-pending");
      }
      this.onEvent({
        type: "ui.mic_active",
        label: this.activeMicLabel,
        deviceId: this.activeMicDeviceId,
        sampleRate: this.nativeSampleRate,
      });
    } catch (err) {
      // surface the failure truthfully; never leave the UI stuck on
      // "connecting" with no explanation and no frames sent
      const message = (err && err.message) || String(err);
      this.onError(message);
      this.releaseAudio();
      this.recording = false;
      this.setState("idle");
      throw err;
    } finally {
      this.starting = false;
    }
  }

  enqueueAudio(base64) {
    if (!this.playNode) return;
    this.playbackActive = true;
    this.playNode.port.postMessage({ cmd: "feed", samples: this.decodePcm(base64).buffer });
  }

  /** Called whenever playback state changed: settle the completed-answer state
   *  to `listening`/`idle` only once the ring is really empty. */
  _maybeFinishPlayback() {
    if (!this._drainToListening) return;
    if (this.playbackActive) return;
    this._drainToListening = false;
    if (this.recording) this.setState("listening");
    else this.setState("idle");
  }

  clearPlayback(reason) {
    if (!this.playNode) return;
    this.playbackActive = false;
    this.playNode.port.postMessage({ cmd: "clear", reason });
  }

  releaseAudio() {
    this.clearPlayback("stop");
    this._removeGestureUnlock();
    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
      this.playNode = null;
      this.capture = null;
    }
    this._resampler = null;
    this.nativeSampleRate = 0;
  }

  stop(silent) {
    this.gen++;
    this.starting = false;
    this.recording = false;
    this.releaseAudio();
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      this.wsModel = "";
      try {
        ws.close();
      } catch {}
    }
    this.activeResponseId = null;
    this.routePending = false;
    this.routeResolved = false;
    if (!silent) this.setState("idle");
  }

  close() {
    this.stop(true);
  }
}