/* Shared sample/preview playback controller (F6).
 *
 * One controller instance for the whole UI guarantees:
 *   - at most one sample/preview plays at a time;
 *   - the SAME button toggles Play -> Pause -> Resume of the SAME audio
 *     element (position kept);
 *   - starting another sample stops the previous one;
 *   - clean reset to "Play" on ended/error;
 *   - stop on navigation (dispose) and on sample switch/delete;
 *   - rapid multiple clicks never create overlapping audio.
 *
 * The controller is transport-injectable so a node unit test can drive the
 * state machine with a fake audio element.
 */

export const PLAYER_STATES = ["idle", "loading", "playing", "paused", "error"];

export class SamplePlayer {
  /**
   * @param {object} opts
   * @param {(key:string)=>Promise<Blob|string>} opts.load  resolves audio for a key
   * @param {()=>object} [opts.createAudio]  audio factory (defaults to new Audio)
   * @param {(key:string, state:string, error?:string)=>void} [opts.onChange]
   */
  constructor({ load, createAudio, onChange } = {}) {
    this.load = load || (async () => null);
    this.createAudio = createAudio || (() => new Audio());
    this.onChange = onChange || (() => {});
    this.key = null;
    this.audio = null;
    this.state = "idle";
    this.error = "";
    this._tokens = new WeakMap();
    this._seq = 0;
  }

  _emit() {
    this.onChange(this.key, this.state, this.error);
  }

  _set(state, error) {
    this.state = state;
    if (error !== undefined) this.error = error || "";
    this._emit();
  }

  /** Current state for a given sample key (idle when not the active one). */
  stateFor(key) {
    return key === this.key ? this.state : "idle";
  }

  /** Toggle play/pause/resume for `key`. Rapid calls are serialised. */
  async toggle(key, target) {
    if (this.key === key && this.audio) {
      if (this.state === "playing") {
        try {
          this.audio.pause();
        } catch {
          /* ignore */
        }
        this._set("paused");
        return this.state;
      }
      if (this.state === "paused") {
        await this._resume();
        return this.state;
      }
      if (this.state === "loading") {
        // a second click while loading is a no-op (never a second element)
        return this.state;
      }
      // error/idle: fall through and restart from the beginning
    }
    // switching samples: stop the previous element first
    this._teardownCurrent();
    this.key = key;
    this.error = "";
    this._set("loading");
    const seq = ++this._seq;
    let src;
    try {
      src = await this.load(key);
    } catch (err) {
      src = null;
    }
    if (seq !== this._seq) return this.state; // superseded by a newer click
    if (!src) {
      this._set("error", "audio not available");
      return this.state;
    }
    const audio = this.createAudio();
    this.audio = audio;
    try {
      audio.src = typeof src === "string" ? src : URL.createObjectURL(src);
    } catch {
      /* ignore */
    }
    audio.addEventListener("ended", () => {
      if (this.audio !== audio) return;
      this._set("idle");
    });
    const onError = () => {
      if (this.audio !== audio) return;
      this._set("error", "playback failed");
    };
    audio.addEventListener("error", onError);
    try {
      await audio.play();
      if (this.audio !== audio) return this.state;
      this._set("playing");
    } catch (err) {
      if (this.audio !== audio) return this.state;
      this._set("error", (err && err.message) || "playback failed");
    }
    return this.state;
  }

  async _resume() {
    if (!this.audio) return;
    try {
      await this.audio.play();
      this._set("playing");
    } catch (err) {
      this._set("error", (err && err.message) || "playback failed");
    }
  }

  _teardownCurrent() {
    this._seq++;
    if (this.audio) {
      try {
        this.audio.pause();
      } catch {
        /* ignore */
      }
      const src = this.audio.src;
      try {
        this.audio.removeAttribute("src");
        this.audio.load && this.audio.load();
      } catch {
        /* ignore */
      }
      if (src && src.startsWith("blob:")) {
        try {
          URL.revokeObjectURL(src);
        } catch {
          /* ignore */
        }
      }
    }
    this.audio = null;
  }

  /** Stop everything (sample switch/delete, navigation, view unmount). */
  stop() {
    this._teardownCurrent();
    this.key = null;
    this.error = "";
    this._set("idle");
  }

  /** Convenience: pause the active key if it is the one playing. */
  stopKey(key) {
    if (this.key === key) this.stop();
  }
}

/** One process-wide controller used by every sample/preview button.
 *  Callers register an async resolver per key; the controller only owns the
 *  single-playback state machine. */
export const sharedSamplePlayer = new SamplePlayer({
  load: async (key) => {
    const resolver = sharedSamplePlayer.resolvers.get(key);
    if (!resolver) throw new Error("no audio source");
    return resolver();
  },
});
sharedSamplePlayer.resolvers = new Map();

/** Register how to load audio for a stable key (returns Blob or url string). */
export function registerSampleSource(key, resolver) {
  sharedSamplePlayer.resolvers.set(key, resolver);
}

export function unregisterSampleSource(key) {
  sharedSamplePlayer.resolvers.delete(key);
}