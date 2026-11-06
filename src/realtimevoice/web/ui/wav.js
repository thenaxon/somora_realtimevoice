/* WAV encoding for raw microphone captures.
 *
 * The microphone path records raw Float32 PCM via the AudioWorklet
 * (echoCancellation/noiseSuppression/autoGainControl all off) and uploads a
 * canonical mono PCM16 WAV at the capture sample rate. There is no server-side
 * WebM/Opus decoding (contract §8a), so this encoder must be correct:
 * - 44-byte canonical RIFF/WAVE header, little-endian
 * - mono downmix of any channel count (average)
 * - Float32 [-1,1] -> Int16 with symmetric clipping and -32768 for -1.0
 * - PCM chunks already Int16 are passed through unchanged
 *
 * Pure and dependency-free so tests/ui_wav.mjs can run it under node.
 */

export const WAV_HEADER_BYTES = 44;

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

/** Interleave/downmix planar channel arrays to a single mono Float32 array. */
export function downmixToMono(channels) {
  if (!channels || channels.length === 0) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const len = channels[0].length;
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0;
    for (let c = 0; c < channels.length; c++) sum += channels[c][i] || 0;
    out[i] = sum / channels.length;
  }
  return out;
}

/** Float32 [-1,1] -> Int16. -1 maps to -32768, +1 maps to 32767. */
export function floatToPcm16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

/**
 * Encode mono PCM16 samples into a canonical RIFF/WAVE byte buffer.
 * @param {Int16Array} pcm16 mono samples
 * @param {number} sampleRate capture sample rate in Hz
 * @returns {ArrayBuffer}
 */
export function encodeWav(pcm16, sampleRate) {
  const dataBytes = pcm16.length * 2;
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true); // ChunkSize
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (PCM)
  view.setUint16(20, 1, true); // AudioFormat PCM
  view.setUint16(22, 1, true); // NumChannels mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // ByteRate = rate * channels * 2
  view.setUint16(32, 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  writeString(view, 36, "data");
  view.setUint32(40, dataBytes, true);
  new Int16Array(buffer, WAV_HEADER_BYTES).set(pcm16);
  return buffer;
}

/** Encode Float32 channels captured at sampleRate into mono PCM16 WAV bytes. */
export function encodeMonoWavFromFloat(channels, sampleRate) {
  const mono = downmixToMono(channels);
  return encodeWav(floatToPcm16(mono), sampleRate);
}

/** Parse just enough of a WAV to report channels/rate/frames (for the UI). */
export function inspectWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (arrayBuffer.byteLength < WAV_HEADER_BYTES) return null;
  const tag = (off) => String.fromCharCode(view.getUint8(off), view.getUint8(off + 1), view.getUint8(off + 2), view.getUint8(off + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  const channels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const bits = view.getUint16(34, true);
  const dataBytes = view.getUint32(40, true);
  const bytesPerFrame = Math.max(1, channels * (bits / 8));
  return { channels, sampleRate, bits, frames: Math.floor(dataBytes / bytesPerFrame) };
}