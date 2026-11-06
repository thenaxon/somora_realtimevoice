"""TTS-input emoji hygiene: symbol-only segment skip + inline emoji strip.

Two complementary mechanisms (both display/chat-safe — the ORIGINAL text is
never modified anywhere; only the text handed to the TTS backend changes):

1. ``is_non_speakable_symbol_segment`` (standard library only, unchanged
   prior-lane behavior): a segment is skippable ONLY when it contains at
   least one recognized base emoji (SMILEYS block U+1F600–U+1F64F, category
   So, including the observed field symbol U+1F60A) and EVERY non-whitespace
   character is either such a base emoji or a recognized emoji sequence
   modifier (variation selectors U+FE00–U+FE0F, ZWJ U+200D, keycap U+20E3,
   skin tones U+1F3FB–U+1F3FF, tags U+E0020–E007F). Mixed text always passes.

2. ``strip_emoji_for_speech`` (maintained Unicode emoji data via the
   ``emoji`` package): removes emoji SEQUENCES from TTS input even when they
   sit inside or beside spoken text ("Hallo. 😊" -> "Hallo."), with an exact
   cleaned->original offset map so segment offsets and the audio->character
   truncation mapping stay consistent with the original display text.

   - REMOVED: sequences the emoji data marks as emoji (fully/minimally
     qualified) — ZWJ families, gender/skin-tone combinations, flag pairs,
     tag (subdivision) sequences, VS16-qualified singletons, keycaps.
   - KEPT: text-presentation forms (unqualified © ™ ❤ ⚠ ☎ ‼ … and everything
     followed by VS15), lone components (a bare skin-tone modifier), the
     meaning-carrying ✕ ✖ ➕ ➖ ➗ ⬅ ⬆ ⬇ symbols, and — untouched by matching
     altogether — letters of every script, combining marks, digits,
     punctuation, currency and mathematical symbols. No blanket
     math/currency/non-Latin-script stripping; no blanket variation-selector
     deletion (a selector stays attached to its kept base).
   - numeric keycaps normalize to their digit ("1️⃣" -> "1") in both the
     VS16-qualified and the bare form, so the digit meaning survives without
     the emoji enclosure.
   - invisible leftovers of a removal are dropped: a ZWJ, variation selector,
     keycap or skin-tone modifier whose ADJACENT original character was part
     of a removed sequence never survives (modifiers in intact non-emoji
     contexts — e.g. ZWJ/ZWNJ inside Brahmic scripts — are never touched).
   - removals never glue words: a removal that would put two alphanumeric
     runs into direct contact inserts one space; removal-adjacent whitespace
     collapses to a single space (edge whitespace is trimmed).

Honesty boundaries: the historical field finding proves only that a
symbol-only segment WAS synthesized (seconds of audio for one U+1F60A) — it
does NOT prove that this audio was noise/laughter, and nothing here claims to
fix word/sentence loss. The strip is a speakability optimization at the TTS
boundary; LLM compliance is a separate, independent runtime prompt boundary.
"""

from __future__ import annotations

import unicodedata

import emoji

_RECOGNIZED_BASE_EMOJI_RANGES = ((0x1F600, 0x1F64F),)  # smileys: uniformly emoji, incl. U+1F60A
_MODIFIER_RANGES = (
    (0xFE00, 0xFE0F),  # variation selectors
    (0x200D, 0x200D),  # zero-width joiner
    (0x20E3, 0x20E3),  # combining enclosing keycap
    (0x1F3FB, 0x1F3FF),  # skin tone modifiers
    (0xE0020, 0xE007F),  # emoji tag characters
)

# Meaning-carrying symbols that the Unicode emoji data fully qualifies but
# that must survive into speech (prior review decision, preserved): the
# multiplication crosses, the arithmetic operators and the block arrows.
_PRESERVED_SYMBOL_CPS = frozenset(
    {0x2715, 0x2716, 0x2795, 0x2796, 0x2797, 0x2B05, 0x2B06, 0x2B07}
)

_STATUS_FULLY_QUALIFIED = 2
_STATUS_MINIMALLY_QUALIFIED = 3
# statuses whose matches are kept as text even though the emoji data knows
# them: 1 = component (e.g. a lone skin-tone modifier), 4 = unqualified
# (text-presentation forms like © / ™ / ❤ without VS16)
_KEEP_STATUSES = frozenset({1, 4})


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def is_non_speakable_symbol_segment(text: str) -> bool:
    """True only for purely recognized-emoji segments (see policy)."""
    if not text or not text.strip():
        return False
    has_base_emoji = False
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if _in_ranges(cp, _MODIFIER_RANGES):
            continue
        if (
            _in_ranges(cp, _RECOGNIZED_BASE_EMOJI_RANGES)
            and unicodedata.category(ch) == "So"
        ):
            has_base_emoji = True
            continue
        return False
    return has_base_emoji


# --------------------------------------------------------------------------
# Inline emoji stripping for TTS input (display/chat text is NEVER touched).
#
# Policy (reviewed, deliberately conservative on the KEEP side):
# - maintained Unicode emoji data (the `emoji` package) finds the sequences:
#   ZWJ families, gender/skin-tone combinations, flag pairs, tag sequences
#   (subdivision flags), keycaps, VS16-qualified singletons — matched with
#   exact offsets, so the original-position mapping stays exact.
# - a matched sequence is REMOVED only when the data marks it as an emoji
#   sequence (fully/minimally qualified). Text-presentation forms (unqualified
#   © ™ ❤ ⚠ ☎ ‼ …), lone components (a bare skin-tone modifier) and the
#   preserved arithmetic/arrow/cross symbols above all SURVIVE — no blanket
#   math/currency/non-Latin-script stripping, no blanket variation-selector
#   deletion (a VS stays attached to its kept base).
# - numeric keycaps normalize to their digit ("1️⃣" -> "1") instead of being
#   lost; the digit keeps its original offset.
# - a variation selector left dangling by a removal (it selected the removed
#   base's presentation) is dropped; a VS after a kept base is never touched.
# - removals never glue words together: when a removal would put two
#   alphanumeric runs into direct contact, a single space is inserted;
#   whitespace runs that a removal cut open collapse to one space, and
#   edge whitespace that only existed around a removal is trimmed.
# - the caller gets the cleaned text plus, for every cleaned character, its
#   index in the ORIGINAL text (and an end sentinel), so segment offsets and
#   the audio->character truncation mapping stay consistent with the original
#   display text. The original text is never modified.


class StripResult:
    """Cleaned TTS text plus its mapping into the original string."""

    __slots__ = ("text", "offsets", "removed")

    def __init__(self, text: str, offsets: list[int], removed: int):
        self.text = text
        # offsets[i] = index of cleaned char i in the original text;
        # offsets[len(text)] (the sentinel) = insertion position at the end.
        self.offsets = offsets
        self.removed = removed  # count of removed/normalized emoji sequences

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StripResult)
            and self.text == other.text
            and self.offsets == other.offsets
            and self.removed == other.removed
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StripResult(text={self.text!r}, offsets={self.offsets!r}, removed={self.removed})"


def _match_status(seq: str) -> int:
    entry = emoji.EMOJI_DATA.get(seq)
    if not isinstance(entry, dict):
        return _STATUS_FULLY_QUALIFIED  # unknown to the data: treat as emoji
    status = entry.get("status")
    return status if isinstance(status, int) else _STATUS_FULLY_QUALIFIED


def _is_keycap(seq: str) -> bool:
    return len(seq) >= 2 and seq[-1] == "\u20E3" and seq[0] in "0123456789#*"


def _preserves_meaning(seq: str) -> bool:
    return any(ord(ch) in _PRESERVED_SYMBOL_CPS for ch in seq)


def strip_emoji_for_speech(text: str) -> StripResult:
    """Strip emoji sequences from TTS input; map every kept character back
    to its original index. The original string is never modified."""
    if not text:
        return StripResult("", [0], 0)

    removed_spans: list[tuple[int, int]] = []
    actions: list[tuple[str, int, int]] = []  # ("keep"|"drop"|"keycap", start, end)
    removed = 0
    pos = 0
    for match in emoji.emoji_list(text):
        start, end, seq = match["match_start"], match["match_end"], match["emoji"]
        if start > pos:
            actions.append(("keep", pos, start))
        # keycaps normalize FIRST: both "1️⃣" (VS16-qualified) and the bare
        # "1⃣" (unqualified, would otherwise survive the status keep) lose
        # the emoji enclosure while the digit keeps its meaning and offset
        if _is_keycap(seq):
            actions.append(("keycap", start, end))
            removed += 1
            removed_spans.append((start, end))
        elif _preserves_meaning(seq) or _match_status(seq) in _KEEP_STATUSES:
            actions.append(("keep", start, end))
        else:
            actions.append(("drop", start, end))
            removed += 1
            removed_spans.append((start, end))
        pos = end
    if pos < len(text):
        actions.append(("keep", pos, len(text)))

    if removed == 0:
        # byte-identical identity mapping (no re-mapping side effects)
        return StripResult(text, list(range(len(text) + 1)), 0)

    dropped_indices: set[int] = set()
    for s, e in removed_spans:
        dropped_indices.update(range(s, e))
    dropped_start = {s for s, _ in removed_spans}
    dropped_end = {e for _, e in removed_spans}

    chars: list[str] = []
    offsets: list[int] = []
    boundary: list[bool] = []  # whitespace touched by a removal (collapsible)

    def emit(ch: str, idx: int, is_boundary: bool = False) -> None:
        chars.append(ch)
        offsets.append(idx)
        boundary.append(is_boundary)

    for kind, start, end in actions:
        if kind == "keep":
            for i in range(start, end):
                ch = text[i]
                if _in_ranges(ord(ch), _MODIFIER_RANGES) and (
                    (i - 1) in dropped_indices or (i + 1) in dropped_indices
                ):
                    # invisible leftover of a removal: a variation selector,
                    # ZWJ, keycap or skin-tone modifier whose ADJACENT original
                    # character was part of the removed sequence (joiners and
                    # selectors inside intact non-emoji scripts are untouched —
                    # their neighbours are letters, never removed spans)
                    continue
                touching = (
                    (i - 1) in dropped_end
                    or (i - 1) in dropped_indices
                    or (i + 1) in dropped_start
                )
                emit(ch, i, touching and ch.isspace())
        elif kind == "keycap":
            emit(text[start], start)  # the digit keeps its meaning + offset
        # "drop": nothing emitted here; the unglue pass below keeps the
        # surrounding words apart (it sees the next KEPT character, so a
        # dangling joiner between the removal and the next word cannot mask
        # the contact)

    # unglue pass: a removal (or its dropped invisible leftovers) sitting
    # between two alphanumeric runs would otherwise fuse the words
    glued: list[str] = []
    glued_offsets: list[int] = []
    glued_boundary: list[bool] = []
    for k, ch in enumerate(chars):
        if (
            k > 0
            and ch.isalnum()
            and chars[k - 1].isalnum()
            and offsets[k] > offsets[k - 1] + 1
        ):
            glued.append(" ")
            glued_offsets.append(offsets[k - 1] + 1)
            glued_boundary.append(True)
        glued.append(ch)
        glued_offsets.append(offsets[k])
        glued_boundary.append(boundary[k])
    chars, offsets, boundary = glued, glued_offsets, glued_boundary

    # collapse whitespace runs that a removal cut open to a single space;
    # edge whitespace that only existed around a removal is trimmed; runs the
    # removal never touched are copied verbatim (no unrelated side effects)
    cleaned = ""
    final_offsets: list[int] = []
    i = 0
    n = len(chars)
    while i < n:
        if chars[i].isspace():
            j = i
            while j < n and chars[j].isspace():
                j += 1
            if any(boundary[k] for k in range(i, j)):
                if i != 0 and j != n:  # interior run cut open by a removal
                    cleaned += " "
                    final_offsets.append(offsets[i])
                # leading/trailing removal-adjacent whitespace: trimmed
            else:
                for k in range(i, j):
                    cleaned += chars[k]
                    final_offsets.append(offsets[k])
            i = j
        else:
            cleaned += chars[i]
            final_offsets.append(offsets[i])
            i += 1
    final_offsets.append(len(text))
    return StripResult(cleaned, final_offsets, removed)
