"""Streaming-safe stripping of model reasoning blocks from spoken output.

Some OpenAI-compatible reasoning models stream a chain-of-thought wrapped in
an ``OPEN … CLOSE`` tag pair as ordinary ``content`` (the tag names are built
from string fragments below so this source file never carries the literal
sequence). That text must never reach the transcript, the conversation history
or TTS. This module provides a small character-level state machine that is safe
across arbitrary delta boundaries:

- text before an opening tag is emitted unchanged,
- everything between an opening and its matching closing tag is dropped,
- a leading, unterminated opening block suppresses everything to the end of the
  stream (nothing is invented, nothing is spoken),
- a partial tag split across two deltas is buffered, never emitted as speech,
- ``flush()`` releases any buffered non-tag tail once the stream ends.

Only the reasoning tag names (optionally with attributes) are recognised; all
other text, including ``<`` characters that do not begin a tag, passes through
unchanged. The class records the number of characters it suppressed so the
caller can log metadata (never the text itself).
"""

from __future__ import annotations

import re
from typing import List

# Tag names are assembled from fragments so the literal sequence never appears
# in this file: some tooling/parsers treat a literal opening tag as reasoning
# and strip everything after it from the stream.
_TAG_NAMES = ("think", "thinking")
_TAG_ALT = "|".join(_TAG_NAMES)
_OPEN_RE = re.compile("<(" + _TAG_ALT + r")(?:\s[^>]*)?>", re.IGNORECASE)
_CLOSE_RE = re.compile("</(" + _TAG_ALT + r")\s*>", re.IGNORECASE)

_OPEN_PREFIXES = tuple("<" + name for name in _TAG_NAMES)
_CLOSE_PREFIXES = tuple("</" + name for name in _TAG_NAMES)


def _could_be_partial_tag(text: str) -> bool:
    """True if ``text`` (starting at a ``<``) is a prefix of a plausible tag.

    Used to decide whether to hold text until the next delta instead of
    emitting it as speech. Conservative: an ambiguous ``<t`` is held; the
    caller flushes it at end of stream if it never became a tag.
    """
    lowered = text.lower()
    if not lowered.startswith("<"):
        return False
    for prefix in _CLOSE_PREFIXES:
        if prefix.startswith(lowered):
            return True
    for prefix in _OPEN_PREFIXES:
        # an opening tag may carry attributes before '>'
        if prefix.startswith(lowered):
            return True
        if lowered.startswith(prefix) and ">" not in lowered:
            return True
    return False


class ReasoningStripper:
    """Incremental reasoning-tag filter; feed deltas, collect speakable text."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_reasoning = False
        self.stripped_chars = 0

    @property
    def in_reasoning(self) -> bool:
        return self._in_reasoning

    def feed(self, text: str) -> str:
        """Consume one content delta, return the speakable part (maybe empty)."""
        if not text:
            return ""
        self._pending += text
        out: List[str] = []
        while self._pending:
            if self._in_reasoning:
                match = _CLOSE_RE.search(self._pending)
                if match:
                    # drop the reasoning text AND the closing tag
                    self.stripped_chars += match.end()
                    self._pending = self._pending[match.end():]
                    self._in_reasoning = False
                    continue
                # no closing tag yet: keep only a possible partial close tag
                held = self._partial_suffix(self._pending, closing=True)
                self.stripped_chars += len(self._pending) - len(held)
                self._pending = held
                break
            # outside reasoning: emit text before the next '<'
            idx = self._pending.find("<")
            if idx == -1:
                out.append(self._pending)
                self._pending = ""
                break
            if idx > 0:
                out.append(self._pending[:idx])
                self._pending = self._pending[idx:]
            match_open = _OPEN_RE.match(self._pending)
            if match_open:
                self.stripped_chars += match_open.end()
                self._pending = self._pending[match_open.end():]
                self._in_reasoning = True
                continue
            match_close = _CLOSE_RE.match(self._pending)
            if match_close:
                # stray closing tag without an opening: drop it silently
                self.stripped_chars += match_close.end()
                self._pending = self._pending[match_close.end():]
                continue
            if _could_be_partial_tag(self._pending):
                # hold until the next delta reveals whether it is a tag
                break
            # a literal '<' that starts no tag: emit it and continue
            out.append("<")
            self._pending = self._pending[1:]
        return "".join(out)

    def flush(self) -> str:
        """Release any buffered tail at end of stream (never a real tag)."""
        if self._in_reasoning:
            # unterminated reasoning block: everything held stays suppressed
            self.stripped_chars += len(self._pending)
            self._pending = ""
            return ""
        tail, self._pending = self._pending, ""
        return tail

    @staticmethod
    def _partial_suffix(text: str, *, closing: bool) -> str:
        """Longest suffix of ``text`` that could be the start of a tag."""
        prefixes = _CLOSE_PREFIXES if closing else _OPEN_PREFIXES
        best = 0
        lowered = text.lower()
        for prefix in prefixes:
            for size in range(1, min(len(prefix), len(lowered)) + 1):
                if prefix.startswith(lowered[-size:]):
                    best = max(best, size)
        return text[len(text) - best:] if best else ""


__all__ = ["ReasoningStripper"]