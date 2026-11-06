"""Test-only per-segment TTS capture (env-gated, inert by default).

When the environment variable RTV_TEST_SEGMENT_CAPTURE_DIR names a directory,
every segment's TTS input text and BOTH PCM streams of the SAME request are
written there for controlled reproduction runs on an ISOLATED test instance:

- backend PCM: the raw bytes as the TTS backend stream yielded them
- emitted PCM: the bytes actually emitted as WebSocket audio deltas

Comparing the two locates a backend->emit loss (chunking/staleness/queueing);
comparing emitted against the browser's fed/played streams locates a
transport loss. Nothing is resynthesized.

Safety contract:

- NEVER enable this on a shared live service: it would capture private
  conversations. Production runs with the variable unset (the default) and
  gets metadata-only lifecycle logs (segment index, durations, byte/char
  counts, cancel reason — never content).
- When disabled the module is completely inert: no bytes are accumulated, no
  file IO happens, no threads are created.
- When enabled everything is bounded: at most MAX_SEGMENTS_PER_RESPONSE
  segments and MAX_BYTES_PER_RESPONSE PCM bytes per response, and file IO
  runs on a dedicated single-thread executor (never on the event loop); a
  bounded pending queue drops rather than grows.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CAPTURE_ENV_VAR = "RTV_TEST_SEGMENT_CAPTURE_DIR"
# bounds, so even a runaway test response cannot fill the disk
MAX_SEGMENTS_PER_RESPONSE = 64
MAX_BYTES_PER_RESPONSE = 8 * 1024 * 1024
MAX_PENDING_WRITES = 32

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Single worker: serializes all capture file IO off the event loop."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtv-segcap")
    return _executor


def capture_dir() -> Path | None:
    """Active test capture directory, or None when capture is off (default)."""
    raw = os.environ.get(CAPTURE_ENV_VAR, "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


class SegmentCapture:
    """Collects one response's per-segment artifacts and a manifest.

    record() never blocks the caller: payloads are handed to the capture
    executor fire-and-forget (exceptions swallowed, bounded queue). When no
    capture dir is active the instance is inert and record() returns
    immediately without retaining any reference to the PCM buffers.
    """

    def __init__(self, response_id: str, turn_epoch: int):
        self.response_id = response_id
        self.turn_epoch = turn_epoch
        self.dir = capture_dir()
        self.enabled = self.dir is not None
        self.entries: list[dict] = []
        self.written_bytes = 0
        self.truncated = False
        self._pending = 0
        self.dropped_writes = 0

    def record(
        self,
        *,
        segment_index: int,
        text: str,
        backend_pcm: bytes | bytearray,
        emitted_pcm: bytes | bytearray,
        seg_start_ms: float,
        audio_ms: float,
        start_byte: int,
        end_byte: int,
        status: str,
        reason: str | None = None,
        spoken_text: str | None = None,
    ) -> None:
        if not self.enabled or self.truncated:
            return
        if segment_index >= MAX_SEGMENTS_PER_RESPONSE:
            self.truncated = True
            self._flush_manifest()
            return
        needed = len(backend_pcm) + len(emitted_pcm)
        if self.written_bytes + needed > MAX_BYTES_PER_RESPONSE:
            self.truncated = True
            self._flush_manifest()
            return
        entry: dict = {
            "response_id": self.response_id,
            "turn_epoch": self.turn_epoch,
            "segment_index": segment_index,
            "seg_start_ms": round(seg_start_ms, 1),
            "audio_ms": round(audio_ms, 1),
            "backend_bytes": len(backend_pcm),
            "emitted_bytes": len(emitted_pcm),
            "start_byte": start_byte,
            "end_byte": end_byte,
            "chars": len(text),
            "status": status,
        }
        if reason:
            entry["reason"] = reason
        self.written_bytes += needed
        self.entries.append(entry)
        entry["text_file"] = f"seg-{self.response_id}-{segment_index}.txt"
        if spoken_text is not None and spoken_text != text:
            # the SPOKEN text (emoji-stripped) differed from the original
            # segment text: capture both so runs stay exactly reproducible
            entry["spoken_text_file"] = f"seg-{self.response_id}-{segment_index}.spoken.txt"
        if backend_pcm:
            entry["backend_file"] = f"seg-{self.response_id}-{segment_index}.backend.pcm"
        if emitted_pcm:
            entry["emitted_file"] = f"seg-{self.response_id}-{segment_index}.emitted.pcm"
        dir_path = self.dir
        response_id = self.response_id
        index = segment_index
        self._submit(
            self._write_segment, dir_path, response_id, index, text,
            bytes(backend_pcm), bytes(emitted_pcm),
            spoken_text if (spoken_text is not None and spoken_text != text) else None,
        )
        self._submit(
            self._write_manifest, dir_path, response_id, self.turn_epoch,
            [dict(e) for e in self.entries], self.truncated,
        )

    def _submit(self, fn, *args) -> None:
        if self._pending >= MAX_PENDING_WRITES:
            self.dropped_writes += 1
            return
        self._pending += 1
        future = _get_executor().submit(fn, *args)

        def _done(fut) -> None:
            self._pending -= 1
            try:
                fut.result()
            except Exception:
                # capture is best-effort test tooling; never affect audio
                self.enabled = False

        future.add_done_callback(_done)

    @staticmethod
    def _write_segment(
        dir_path: Path, response_id: str, index: int, text: str, backend_pcm: bytes, emitted_pcm: bytes,
        spoken_text: str | None = None,
    ) -> None:
        (dir_path / f"seg-{response_id}-{index}.txt").write_text(text, encoding="utf-8")
        if spoken_text is not None:
            (dir_path / f"seg-{response_id}-{index}.spoken.txt").write_text(spoken_text, encoding="utf-8")
        if backend_pcm:
            (dir_path / f"seg-{response_id}-{index}.backend.pcm").write_bytes(backend_pcm)
        if emitted_pcm:
            (dir_path / f"seg-{response_id}-{index}.emitted.pcm").write_bytes(emitted_pcm)

    def _flush_manifest(self) -> None:
        self._submit(
            self._write_manifest, self.dir, self.response_id, self.turn_epoch, list(self.entries), self.truncated
        )

    @staticmethod
    def _write_manifest(dir_path: Path, response_id: str, turn_epoch: int, entries: list, truncated: bool) -> None:
        payload = {
            "response_id": response_id,
            "turn_epoch": turn_epoch,
            "capture_truncated": truncated,
            "segments": entries,
        }
        (dir_path / f"resp-{response_id}-manifest.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    def record_queued(self, *, queued_index: int, text: str, reason: str | None = None) -> None:
        """Record a segment that was discarded while still QUEUED (never
        reached synthesis). Text only — by definition it has no PCM."""
        if not self.enabled or self.truncated:
            return
        if len(self.entries) >= MAX_SEGMENTS_PER_RESPONSE:
            self.truncated = True
            self._flush_manifest()
            return
        entry: dict = {
            "response_id": self.response_id,
            "turn_epoch": self.turn_epoch,
            "queued_index": queued_index,
            "queued": True,
            "chars": len(text),
            "status": "queued_discarded",
        }
        if reason:
            entry["reason"] = reason
        self.entries.append(entry)
        entry["text_file"] = f"seg-{self.response_id}-q{queued_index}.txt"
        self._submit(self._write_queued_text, self.dir, self.response_id, queued_index, text)
        self._submit(
            self._write_manifest, self.dir, self.response_id, self.turn_epoch,
            [dict(e) for e in self.entries], self.truncated,
        )

    @staticmethod
    def _write_queued_text(dir_path: Path, response_id: str, index: int, text: str) -> None:
        (dir_path / f"seg-{response_id}-q{index}.txt").write_text(text, encoding="utf-8")

    def close(self) -> None:
        """Flush a final manifest (no-op when disabled)."""
        if self.enabled:
            self._flush_manifest()
