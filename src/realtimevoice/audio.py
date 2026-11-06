"""PCM16/WAV helpers. Everything CPU-heavy here is called via a thread executor."""

from __future__ import annotations

import base64
import binascii
import struct

import numpy as np

TARGET_SAMPLE_RATE = 24_000
DEFAULT_FRAME_MS = 20
MS_PER_FRAME = {}


def decode_base64_pcm16(data: str) -> bytes:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 audio: {exc}") from exc
    if len(raw) % 2:
        raw = raw[:-1]
    return raw


def encode_base64_pcm16(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def pcm16_to_numpy(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1 or not pcm:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    usable = (len(x) // channels) * channels
    x = x[:usable].reshape(-1, channels).astype(np.float64).mean(axis=1)
    return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler for mono PCM16."""
    if src_rate == dst_rate or not pcm:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    if x.size == 0:
        return b""
    n_out = int(round(x.size * dst_rate / src_rate))
    if n_out <= 0:
        return b""
    pos = np.linspace(0.0, x.size - 1.0, num=n_out)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, x.size - 1)
    frac = pos - i0
    y = x[i0] * (1.0 - frac) + x[i1] * frac
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


def wav_header(pcm_len: int, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1) -> bytes:
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_len = pcm_len
    riff_len = 36 + data_len
    return (
        b"RIFF"
        + struct.pack("<I", riff_len)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_len)
    )


def pcm16_to_wav(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1) -> bytes:
    return wav_header(len(pcm), sample_rate, channels) + pcm


def frame_ms(pcm_len: int, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    return pcm_len / 2 / sample_rate * 1000.0


class WavParser:
    """Incremental WAV parser.

    feed() accepts arbitrary network chunks and returns a list of
    (sample_rate, channels, pcm_bytes) tuples for completed payloads. The fmt
    and data chunk headers may arrive split across chunks. A data chunk with
    size 0 or 0xFFFFFFFF is treated as unbounded (stream until EOF).
    """

    def __init__(self, max_header: int = 1 << 16):
        self._buf = bytearray()
        self._fmt: tuple[int, int, int] | None = None
        self._data_left: int | None = None
        self._in_data = False
        self._riff_done = False
        self._max_header = max_header

    def feed(self, chunk: bytes) -> list[tuple[int, int, bytes]]:
        self._buf.extend(chunk)
        out: list[tuple[int, int, bytes]] = []
        if not self._in_data:
            if not self._riff_done:
                if len(self._buf) < 12:
                    return out
                if bytes(self._buf[:4]) != b"RIFF" or bytes(self._buf[8:12]) != b"WAVE":
                    raise ValueError("not a WAV stream (bad RIFF/WAVE header)")
                del self._buf[:12]
                self._riff_done = True
            self._consume_header(out)
        if self._in_data and self._data_left != 0:
            payload = bytes(self._buf)
            self._buf.clear()
            if self._data_left and self._data_left != 0xFFFFFFFF:
                if len(payload) > self._data_left:
                    extra = payload[self._data_left :]
                    payload = payload[: self._data_left]
                    self._data_left = 0
                    self._in_data = False
                    self._buf.extend(extra)
                    out.extend(self._trailer_pass())
                else:
                    self._data_left -= len(payload)
            if self._in_data and len(payload) % 2:
                # hold back a lone byte; PCM16 pairs must not be split
                self._buf.append(payload[-1])
                payload = payload[:-1]
            if payload:
                out.append(self._payload(payload))
        return out

    def flush(self) -> list[tuple[int, int, bytes]]:
        """Emit any held-back odd byte (zero padded) at end of stream."""
        if not self._buf:
            return []
        leftover = bytes(self._buf)
        self._buf.clear()
        if len(leftover) % 2:
            leftover += b"\x00"
        return [self._payload(leftover)]

    def _payload(self, pcm: bytes) -> tuple[int, int, bytes]:
        rate, channels, _bits = self._fmt or (TARGET_SAMPLE_RATE, 1, 16)
        return rate, channels, pcm

    def _consume_header(self, out: list[tuple[int, int, bytes]]) -> None:
        while not self._in_data:
            if len(self._buf) < 8:
                return
            chunk_id = bytes(self._buf[:4])
            (size,) = struct.unpack("<I", self._buf[4:8])
            if chunk_id == b"fmt ":
                if len(self._buf) < 8 + size:
                    return
                audio_format, channels, rate, _byte_rate, _align, bits = struct.unpack(
                    "<HHIIHH", bytes(self._buf[8 : 8 + 16])
                )
                if audio_format not in (1, 3) or bits != 16:
                    raise ValueError(f"unsupported WAV format (format={audio_format}, bits={bits})")
                self._fmt = (rate, channels, bits)
                del self._buf[: 8 + size + (size & 1)]
            elif chunk_id == b"data":
                del self._buf[:8]
                self._in_data = True
                self._data_left = 0 if size in (0, 0xFFFFFFFF) else size
            else:
                if size == 0 or 8 + size > self._max_header:
                    raise ValueError(f"unsupported WAV chunk {chunk_id!r}")
                if len(self._buf) < 8 + size + (size & 1):
                    return
                del self._buf[: 8 + size + (size & 1)]
        if self._fmt is None:
            raise ValueError("WAV stream missing fmt chunk before data")

    def _trailer_pass(self) -> list[tuple[int, int, bytes]]:
        out: list[tuple[int, int, bytes]] = []
        self._consume_header(out)
        if self._in_data and self._data_left != 0:
            payload = bytes(self._buf)
            self._buf.clear()
            if self._data_left and len(payload) > self._data_left:
                payload = payload[: self._data_left]
            if payload:
                out.append(self._payload(payload))
        return out


class PcmParser:
    """Raw little-endian 16-bit mono PCM at a known rate, in arbitrary
    network chunks. It is passed on in whole 20 ms blocks, so resampling each
    block neither drops samples nor depends on how the network cut it; the
    rest waits for the next chunk or the end."""

    def __init__(self, sample_rate: int):
        self._rate = int(sample_rate)
        self._block = max(2, (self._rate // 50) * 2)
        self._carry = b""

    @property
    def rate(self) -> int:
        return self._rate

    def feed(self, chunk: bytes) -> list[tuple[int, int, bytes]]:
        data = self._carry + chunk
        cut = len(data) - (len(data) % self._block)
        self._carry = data[cut:]
        return [(self._rate, 1, data[:cut])] if cut else []

    def flush(self) -> list[tuple[int, int, bytes]]:
        data, self._carry = self._carry, b""
        data = data[: len(data) - (len(data) % 2)]
        return [(self._rate, 1, data)] if data else []


class Mp3Decoder:
    """MP3 from a speaking model: collected, then decoded once by ffmpeg to
    mono PCM16 at the target rate (ffmpeg must be on the PATH)."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self._rate = sample_rate
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[int, int, bytes]]:
        self._buf.extend(chunk)
        return []

    def flush(self) -> list[tuple[int, int, bytes]]:
        if not self._buf:
            return []
        return [(self._rate, 1, decode_mp3(bytes(self._buf), self._rate))]


def decode_mp3(data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """MP3 bytes -> mono PCM16 at sample_rate, through ffmpeg."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("MP3 answers need ffmpeg on the server; choose WAV or PCM for this model")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        input=data, capture_output=True, timeout=60, check=False,
    )
    if proc.returncode != 0:
        raise ValueError("the MP3 answer could not be decoded: " + proc.stderr.decode("utf-8", "replace")[:200])
    return proc.stdout
