// Wire format of the realtime protocol: PCM16 mono at 24 kHz, both directions.
// The page runs a NATIVE AudioContext (44.1/48 kHz, device dependent), so both
// directions need a rate conversion:
//   capture : native -> 24 kHz, done in audio.js (makeResampler) before send
//   playback: 24 kHz -> native, done HERE when draining the ring
// Emitting one ring sample per output sample plays 24 kHz material at the
// native rate, i.e. 2.00x too fast at 48 kHz and 1.84x at 44.1 kHz — audibly
// a chipmunk. Regression cover: scripts/playback_rate_probe.py.
const WIRE_RATE = 24000;

// Capture: native-rate mono Float32 in, Int16 out (chunked), continuous including silence.
// (audio.js resamples these Int16 blocks to WIRE_RATE before they go on the wire.)
class CaptureWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(4800); // 200 ms
    this._fill = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      for (let i = 0; i < ch.length; i++) {
        this._buffer[this._fill++] = ch[i];
        if (this._fill === this._buffer.length) {
          this.flush();
        }
      }
    }
    return true;
  }
  flush() {
    const out = new Int16Array(this._fill);
    for (let i = 0; i < this._fill; i++) {
      let s = this._buffer[i];
      if (s > 1) s = 1; else if (s < -1) s = -1;
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this._fill = 0;
    this.port.postMessage(out.buffer, [out.buffer]);
  }
}
registerProcessor("capture-worklet", CaptureWorklet);

// Playback: bounded ring buffer with explicit overflow accounting.
class PlayWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    // 60 s of headroom: TTS backends may synthesize faster than realtime;
    // a short ring would silently chop mid-sentence audio.
    this.capacity = WIRE_RATE * 60;
    this.ring = new Float32Array(this.capacity);
    this.read = 0;
    this.filled = 0;
    this.droppedSamples = 0;
    // Ring holds WIRE_RATE samples; the output runs at the context's native
    // rate. Advance the read head by `step` per output frame and interpolate.
    // `sampleRate` is the AudioWorkletGlobalScope's native rate.
    this.step = WIRE_RATE / sampleRate;
    this.frac = 0; // sub-sample position between ring[read] and the next sample
    // metadata-only playback telemetry (sample counts, never content)
    this.fedSamples = 0;
    this.playedSamples = 0;
    this.nonZeroPlayed = 0;
    this.droppedTotal = 0;
    // FIFO ledger of fed CHUNKS in ring/play order: lets a clear report how
    // much of each item was actually HEARD (played) vs discarded, so the
    // server can truncate conversation history to the heard prefix instead
    // of the emitted one. NO eviction of pending chunks — every live chunk
    // owns its ring samples until fully played, so attribution always
    // follows the real playhead. Consecutive same-item chunks coalesce into
    // the last entry (the normal streaming case keeps the ledger small);
    // entry count is bounded by ring-buffered chunks, each owning >= 1
    // accepted sample.
    this.ledger = [];
    this.ledgerDropped = 0; // kept for ack contract stability (always 0: no eviction exists)
    // F11: notify the page once when the ring actually drains, so the UI can
    // tell "the answer finished playing" apart from "response.done arrived".
    this.wasFilled = false;
    // cumulative heard per item, keyed "gen:id" (LRU map, fixed cap): a
    // chunk that finished playing leaves the ledger but its heard samples
    // must still count toward the item's heard prefix on a later clear
    this.heardMap = new Map();
    this.heardMapCap = 16;
    this.port.onmessage = (ev) => {
      const data = ev.data;
      if (data.cmd === "feed") {
        const samples = new Int16Array(data.samples);
        // write loop first: only samples actually accepted by the ring may
        // enter the ledger — overflow rejects are never heard/discarded
        let written = 0;
        for (let i = 0; i < samples.length; i++) {
          if (this.filled >= this.capacity) {
            this.droppedSamples += samples.length - i;
            this.droppedTotal += samples.length - i;
            break;
          }
          this.ring[(this.read + this.filled) % this.capacity] = samples[i] / 32768;
          this.filled++;
          written++;
        }
        this.fedSamples += samples.length;
        // optional item tagging (backward compatible: absent -> id null, gen 0)
        const itemId = data.itemId ?? null;
        const itemGen = data.gen ?? 0;
        if (written > 0) {
          const last = this.ledger[this.ledger.length - 1];
          if (last && last.id === itemId && last.gen === itemGen) {
            // consecutive chunks of the same item coalesce into the last entry
            last.fed += written;
          } else {
            this.ledger.push({ id: itemId, gen: itemGen, fed: written, played: 0 });
          }
        }
        if (this.droppedSamples > 0) {
          this.port.postMessage({ cmd: "overflow", dropped: this.droppedSamples });
          this.droppedSamples = 0;
        }
      } else if (data.cmd === "clear") {
        // per-item heard-prefix report: history must reflect what was heard,
        // not what was emitted, so a barge-in can truncate transcripts.
        // Aggregated per item over the current chunk entries; heard comes
        // from heardMap so already-drained chunks still count.
        const agg = new Map();
        for (const e of this.ledger) {
          const key = e.gen + ":" + (e.id ?? "");
          const a = agg.get(key) || { id: e.id, gen: e.gen, discarded: 0 };
          a.discarded += e.fed - e.played;
          agg.set(key, a);
        }
        const items = [];
        for (const a of agg.values()) {
          if (a.discarded > 0) {
            items.push({
              id: a.id,
              gen: a.gen,
              heardSamples: this.heardMap.get(a.gen + ":" + (a.id ?? "")) || 0,
              discardedSamples: a.discarded,
            });
          }
        }
        // reply before wiping: tells the page exactly how much buffered
        // audio this clear discarded (field diagnostics for barge-in)
        this.port.postMessage({
          cmd: "cleared",
          bufferedSamples: this.filled,
          playedSamples: this.playedSamples,
          nonZeroPlayed: this.nonZeroPlayed,
          droppedTotal: this.droppedTotal,
          reason: typeof data.reason === "string" ? data.reason : "",
          ledgerDropped: this.ledgerDropped,
          items,
        });
        this.filled = 0;
        this.read = 0;
        this.frac = 0; // a stale sub-sample offset would skew the next turn
        this.droppedSamples = 0;
        this.wasFilled = false; // a cleared ring must not report a drain
        this.ledger = [];
        this.heardMap = new Map();
        this.ledgerDropped = 0;
      } else if (data.cmd === "stats") {
        this.port.postMessage({
          cmd: "stats",
          filled: this.filled,
          fedSamples: this.fedSamples,
          playedSamples: this.playedSamples,
          nonZeroPlayed: this.nonZeroPlayed,
          droppedTotal: this.droppedTotal,
        });
      }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      if (this.filled > 0) {
        // linear interpolation between the current ring sample and the next
        // one; with only a single sample left, hold it (no wrap-around read
        // into stale data)
        const a = this.ring[this.read];
        const b = this.filled > 1 ? this.ring[(this.read + 1) % this.capacity] : a;
        const v = a + (b - a) * this.frac;
        if (v !== 0) this.nonZeroPlayed++;
        out[i] = v;
        // advance the read head by step (24000/native) and consume WHOLE ring
        // samples as they are passed. All accounting below stays in WIRE_RATE
        // samples — the server truncates history by heard 24 kHz samples, so
        // counting output frames here would corrupt barge-in attribution.
        this.frac += this.step;
        while (this.frac >= 1 && this.filled > 0) {
          this.frac -= 1;
          this.read = (this.read + 1) % this.capacity;
          this.filled--;
          this.playedSamples++;
          this.advanceLedger();
        }
      } else {
        out[i] = 0;
      }
    }
    // ring just became empty after holding audio: report the real drain once
    if (this.wasFilled && this.filled === 0) {
      this.wasFilled = false;
      this.port.postMessage({ cmd: "idle" });
    } else if (this.filled > 0) {
      this.wasFilled = true;
    }
    return true;
  }

  // FIFO: playback consumes chunks in ring order — advance the FIRST entry
  // that is not fully heard yet; a fully heard chunk leaves the ledger
  // immediately (its heard already lives in heardMap, so removal is always
  // safe for cumulative attribution)
  advanceLedger() {
    for (const entry of this.ledger) {
      if (entry.played < entry.fed) {
        entry.played++;
        this.bumpHeard(entry.gen + ":" + (entry.id ?? ""));
        if (entry.played >= entry.fed) {
          this.ledger.splice(this.ledger.indexOf(entry), 1);
          // a drain can turn a key idle: keep the idle history cache
          // bounded at cap (its own key may be the one pruned)
          this.pruneIdleHeard();
        }
        break;
      }
    }
  }
  // bounded heardMap: record one heard sample for an item key with LRU
  // refresh (delete+re-set). The cap applies to the COMPLETED (idle)
  // history cache ONLY — live keys are inherently bounded by pending ring
  // entries, must ALL remain and are inserted unconditionally (the map may
  // then hold cap + live entries). For a new IDLE key at idle-cap, the
  // least recently used idle keys are evicted first. A bump is never
  // skipped: a heard count of zero must never be reported as fact.
  bumpHeard(key) {
    if (this.heardMap.has(key)) {
      const prev = this.heardMap.get(key);
      this.heardMap.delete(key);
      this.heardMap.set(key, prev + 1);
      return;
    }
    if (!this.isLiveKey(key)) {
      while (this.idleHeardCount() >= this.heardMapCap) {
        for (const k of this.heardMap.keys()) {
          if (!this.isLiveKey(k)) { this.heardMap.delete(k); break; }
        }
      }
    }
    this.heardMap.set(key, 1);
  }
  // after a drain frees its key (idle now), prune LRU idle keys down to cap
  pruneIdleHeard() {
    while (this.idleHeardCount() > this.heardMapCap) {
      for (const k of this.heardMap.keys()) {
        if (!this.isLiveKey(k)) { this.heardMap.delete(k); break; }
      }
    }
  }
  isLiveKey(key) {
    return this.ledger.some((e) => e.gen + ":" + (e.id ?? "") === key);
  }
  idleHeardCount() {
    let n = 0;
    for (const k of this.heardMap.keys()) {
      if (!this.isLiveKey(k)) n++;
    }
    return n;
  }
}
registerProcessor("play-worklet", PlayWorklet);
