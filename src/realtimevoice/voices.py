"""Voice library: sample WAV validation, canonicalisation and file storage.

Security-relevant properties:
- Stored WAVs live OUTSIDE settings.json and Git, under
  ``<data_dir>/voices/<voice_id>/<sha256>.wav``; directories are 0700 and
  files 0600, written atomically (tmp + rename).
- Storage is content-addressed and immutable; the sample id is derived from
  the stored bytes so identical uploads deduplicate.
- Validation rejects anything that is not a bounded mono/stereo PCM16 or
  float32 RIFF/WAVE; the stored bytes are always canonical mono PCM16.
- Metrics are computed server-side and contain no audio content.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import time
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Voice, VoiceSample

MIN_SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 30.0
MIN_SAMPLE_RATE = 16_000
MAX_SAMPLE_RATE = 48_000
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
SAMPLE_ID_LEN = 16


class SampleValidationError(ValueError):
    """A rejected upload; ``code`` is a machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class CanonicalWav:
    data: bytes
    sha256: str
    duration_ms: int
    sample_rate: int
    channels: int
    metrics: dict[str, float]


def _dbfs(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return float(round(20.0 * np.log10(value / 32768.0), 2))


def _waveform_envelope(pcm16: np.ndarray, buckets: int = 48) -> list[float]:
    """Absolute-scale peak envelope: each value is |peak| / 32768 in [0, 1].

    NOT normalised to the sample's own maximum, so a quiet recording renders
    as a small waveform (truthful). Returns at most `buckets` values."""
    if pcm16.size == 0:
        return []
    n = min(buckets, pcm16.size)
    # split into n contiguous buckets and take the peak of each
    edges = np.linspace(0, pcm16.size, n + 1).astype(int)
    peaks = []
    for i in range(n):
        chunk = pcm16[edges[i] : edges[i + 1]]
        if chunk.size == 0:
            peaks.append(0.0)
        else:
            peaks.append(round(float(np.abs(chunk.astype(np.float64)).max()) / 32768.0, 4))
    return peaks


def canonicalise_wav(raw: bytes) -> CanonicalWav:
    """Validate and canonicalise an uploaded WAV to mono PCM16.

    Accepts RIFF/WAVE PCM16 or float32, mono or stereo, 16–48 kHz,
    1.0–30.0 s, <= 10 MB. Stereo is downmixed, float32 converted to PCM16;
    the sample rate is kept (no resample). Raises SampleValidationError with
    codes: too_large, unsupported_format, sample_too_short, sample_too_long,
    invalid_audio.
    """
    if len(raw) > MAX_SAMPLE_BYTES:
        raise SampleValidationError("too_large", f"sample exceeds {MAX_SAMPLE_BYTES} bytes")
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise SampleValidationError("unsupported_format", "not a RIFF/WAVE file")
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            rate = wav.getframerate()
            n_frames = wav.getnframes()
            frames = wav.readframes(n_frames)
    except wave.Error as exc:
        raise SampleValidationError("unsupported_format", f"cannot parse WAV: {exc}") from exc
    if channels not in (1, 2):
        raise SampleValidationError("unsupported_format", "only mono or stereo is supported")
    if sampwidth not in (2, 4):
        raise SampleValidationError("unsupported_format", "only 16-bit or 32-bit float WAV is supported")
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise SampleValidationError(
            "unsupported_format", f"sample rate {rate} Hz outside {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}"
        )
    if sampwidth == 2:
        x = np.frombuffer(frames[: (len(frames) // 2) * 2], dtype="<i2").astype(np.float64)
    else:
        # 32-bit: PCM32 or IEEE float. We only accept IEEE float (format tag 3)
        # or 32-bit PCM (tag 1); wave reports them identically, so decide by
        # value range: PCM32 values exceed the float [-1, 1] range.
        x32 = np.frombuffer(frames[: (len(frames) // 4) * 4], dtype="<f4").astype(np.float64)
        if x32.size and (np.abs(x32).max() > 1.5):
            x = np.frombuffer(frames[: (len(frames) // 4) * 4], dtype="<i4").astype(np.float64) / 65536.0
        else:
            x = np.clip(x32, -1.0, 1.0) * 32767.0
    if channels == 2:
        usable = (x.size // 2) * 2
        x = x[:usable].reshape(-1, 2).mean(axis=1)
    duration_s = x.size / float(rate) if rate else 0.0
    if duration_s < MIN_SAMPLE_SECONDS:
        raise SampleValidationError("sample_too_short", f"sample shorter than {MIN_SAMPLE_SECONDS}s")
    if duration_s > MAX_SAMPLE_SECONDS:
        raise SampleValidationError("sample_too_long", f"sample longer than {MAX_SAMPLE_SECONDS}s")
    pcm = np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()
    canonical = _wav_bytes(pcm, rate)
    sha = hashlib.sha256(canonical).hexdigest()
    peak = float(np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.float64)).max()) if pcm else 0.0
    rms_val = float(np.sqrt(np.mean(np.frombuffer(pcm, dtype="<i2").astype(np.float64) ** 2))) if pcm else 0.0
    clipping = (
        float(np.mean(np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.float64)) >= 32767.0) * 100.0)
        if pcm
        else 0.0
    )
    metrics = {
        "rms_dbfs": _dbfs(rms_val),
        "peak_dbfs": _dbfs(peak),
        "clipping_pct": round(clipping, 4),
    }
    # N-4: an absolute-scale waveform envelope (normalised 0..1 per bucket by
    # the digital full scale, NOT by the sample's own peak) so quiet samples
    # look quiet and loud samples look loud.
    metrics["waveform"] = _waveform_envelope(np.frombuffer(pcm, dtype="<i2"))
    return CanonicalWav(
        data=canonical,
        sha256=sha,
        duration_ms=int(round(duration_s * 1000)),
        sample_rate=rate,
        channels=1,
        metrics=metrics,
    )


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def sample_id_for(sha256: str) -> str:
    return f"smp_{sha256[:SAMPLE_ID_LEN]}"


def speech_duration_ms(wav_bytes: bytes, *, threshold_rms: float = 350.0, frame_ms: int = 20) -> float:
    """Estimate the voiced duration of a canonical mono PCM16 WAV.

    Energy-based VAD mirroring the live turn detector: frames whose RMS
    exceeds `threshold_rms` count as speech. Used as a pre-check so a silent
    sample never reaches an STT model (which can hallucinate text). Returns
    the summed duration of speech frames in milliseconds; 0 when silent.
    """
    if not wav_bytes or len(wav_bytes) < 44:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except wave.Error:
        return 0.0
    if rate <= 0 or not frames:
        return 0.0
    x = np.frombuffer(frames[: (len(frames) // 2) * 2], dtype="<i2").astype(np.float64)
    frame_len = max(1, int(rate * frame_ms / 1000))
    speech_frames = 0
    for start in range(0, x.size - frame_len + 1, frame_len):
        chunk = x[start : start + frame_len]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms >= threshold_rms:
            speech_frames += 1
    return speech_frames * frame_ms


def new_sample(
    canonical: CanonicalWav,
    *,
    source: str,
    original_filename: str = "",
    created_at: float | None = None,
) -> VoiceSample:
    return VoiceSample(
        id=sample_id_for(canonical.sha256),
        sha256=canonical.sha256,
        source=source,  # type: ignore[arg-type]
        original_filename=original_filename,
        duration_ms=canonical.duration_ms,
        sample_rate=canonical.sample_rate,
        channels=canonical.channels,
        size_bytes=len(canonical.data),
        created_at=time.time() if created_at is None else created_at,
        metrics=dict(canonical.metrics),
    )


class SampleStore:
    """Content-addressed sample file store under ``<data_dir>/voices``."""

    def __init__(self, data_dir: str):
        self.root = os.path.join(data_dir, "voices")

    def _voice_dir(self, voice_id: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", voice_id or ""):
            raise ValueError("invalid voice id for storage")
        return os.path.join(self.root, voice_id)

    def path_for(self, voice_id: str, sha256: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
            raise ValueError("invalid sample sha256")
        return os.path.join(self._voice_dir(voice_id), f"{sha256}.wav")

    def write(self, voice_id: str, sha256: str, data: bytes) -> str:
        """Atomically store bytes; the file is 0600 from creation on."""
        directory = self._voice_dir(voice_id)
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        for d in (self.root, directory):
            try:
                os.chmod(d, 0o700)
            except OSError:
                pass
        path = self.path_for(voice_id, sha256)
        if os.path.exists(path):
            return path
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def read(self, voice_id: str, sha256: str) -> bytes:
        with open(self.path_for(voice_id, sha256), "rb") as fh:
            return fh.read()

    def exists(self, voice_id: str, sha256: str) -> bool:
        try:
            return os.path.isfile(self.path_for(voice_id, sha256))
        except ValueError:
            return False

    def delete(self, voice_id: str, sha256: str) -> None:
        try:
            os.unlink(self.path_for(voice_id, sha256))
        except (OSError, ValueError):
            pass

    def remove_voice_dir(self, voice_id: str) -> None:
        directory = self._voice_dir(voice_id)
        try:
            for name in os.listdir(directory):
                try:
                    os.unlink(os.path.join(directory, name))
                except OSError:
                    pass
            os.rmdir(directory)
        except OSError:
            pass


def referenced_shas(settings: Any, voice_id: str) -> set[str]:
    """Shas still referenced by a sample of the voice. Used by the GC rule
    helper: only shas with NO reference may be deleted from disk (a running
    call holds its recording in memory)."""
    shas: set[str] = set()
    voice = (settings.voices or {}).get(voice_id)
    if voice is not None:
        for sample in voice.samples:
            shas.add(sample.sha256)
    return shas


def gc_eligible(settings: Any, voice_id: str, sha256: str, live_shas: set[str] | None = None) -> bool:
    """A sample file may be garbage-collected only when no voice sample and no
    live session references its sha."""
    if live_shas and sha256 in live_shas:
        return False
    return sha256 not in referenced_shas(settings, voice_id)


__all__ = [
    "CanonicalWav",
    "SampleStore",
    "SampleValidationError",
    "canonicalise_wav",
    "gc_eligible",
    "new_sample",
    "referenced_shas",
    "sample_id_for",
]