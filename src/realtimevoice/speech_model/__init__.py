"""Is this sound speech? A small speech classifier on the CPU.

Loudness alone lets a cough, a knock or a chair count as speech; the speech
recogniser then turns that noise into a word ("ja", "danke"). This asks a
small model trained to tell speech from other sound: Silero VAD v5 (MIT,
silero_vad.onnx in this directory, see LICENSE.silero-vad), about 2 MB, a
fraction of a millisecond per 32 ms of audio on one CPU core.

Without onnxruntime or the model file the server falls back to loudness
only and says so once in its log.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading

import numpy as np

from ..audio import resample_pcm16

logger = logging.getLogger("realtimevoice.speech_model")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
MODEL_RATE = 16_000
WINDOW = 512   # samples per model step at 16 kHz (32 ms)
CONTEXT = 64   # samples of the previous step the model sees with each one

_session = None
_session_lock = threading.Lock()
_unavailable_reason: str | None = None


def _load():
    """The shared model session, or None (with the reason logged once)."""
    global _session, _unavailable_reason
    if _session is not None or _unavailable_reason is not None:
        return _session
    with _session_lock:
        if _session is not None or _unavailable_reason is not None:
            return _session
        try:
            import onnxruntime as ort

            with open(MODEL_PATH, "rb") as fh:
                data = fh.read()
            if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
                raise ValueError("speech model file does not match its checksum")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(data, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 — any failure means: loudness only
            _unavailable_reason = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("speech model unavailable, detecting speech by loudness only (%s)", _unavailable_reason)
        return _session


def available() -> bool:
    return _load() is not None


class SpeechClassifier:
    """Per-call state: feed 16-bit mono PCM at any rate in any frame size;
    `probability` is the speech probability of the newest 32 ms."""

    def __init__(self, sample_rate: int):
        self._session = _load()
        if self._session is None:
            raise RuntimeError("speech model unavailable")
        self._rate = sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.probability = 0.0

    def feed(self, pcm: bytes) -> float:
        if self._rate != MODEL_RATE:
            pcm = resample_pcm16(pcm, self._rate, MODEL_RATE)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])
        while len(self._buf) >= WINDOW:
            chunk, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
            x = np.concatenate([self._context, chunk])[None, :]
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": np.array(MODEL_RATE, dtype=np.int64)}
            )
            self._context = chunk[-CONTEXT:]
            self.probability = float(out[0][0])
        return self.probability
