"""Streaming-safe reasoning-block stripping (content-level filter).

Covers the delta-boundary cases: a full block inline, tags split across
chunks, an unterminated leading block, stray close tags, literal '<' text and
attribute/whitespace tag variants.

Tag literals are assembled from fragments so this test file never contains the
raw sequence (some parsers treat a literal opening tag as reasoning).
"""

from realtimevoice.reasoning import ReasoningStripper

_OPEN = "<" + "think" + ">"
_CLOSE = "</" + "think" + ">"


def run(chunks):
    stripper = ReasoningStripper()
    out = "".join(stripper.feed(c) for c in chunks) + stripper.flush()
    return out, stripper.stripped_chars


def test_normal_text_is_unaffected():
    out, stripped = run(["Guten ", "Tag, ", "wie geht es dir?"])
    assert out == "Guten Tag, wie geht es dir?"
    assert stripped == 0


def test_full_inline_block_is_removed():
    out, stripped = run(["Hallo ", _OPEN, "geheim", _CLOSE, " Welt"])
    assert out == "Hallo  Welt"
    assert stripped == len(_OPEN) + len("geheim") + len(_CLOSE)


def test_tags_split_across_chunks():
    out, _ = run(["Hi <thi", "nk>hidden", " more</thin", "k> done"])
    assert out == "Hi  done"


def test_opening_tag_split_character_by_character():
    out, _ = run(list(_OPEN + "x" + _CLOSE + "answer"))
    assert out == "answer"


def test_unterminated_leading_block_suppresses_everything():
    out, stripped = run(["<thi" + "nk", "ing>only reasoning, no closing tag"])
    assert out == ""
    assert stripped > 0


def test_unterminated_inline_block_keeps_earlier_text():
    out, _ = run(["Der Preis ist ", _OPEN + "ich rechne noch"])
    assert out == "Der Preis ist "


def test_stray_closing_tag_is_dropped():
    out, _ = run(["keep", _CLOSE, "more"])
    assert out == "keepmore"


def test_literal_less_than_is_preserved():
    out, stripped = run(["a < b and c <= d"])
    assert out == "a < b and c <= d"
    assert stripped == 0


def test_tag_with_attributes_and_spaced_close():
    open_with_attr = "<" + "think" + " foo='1'" + ">"
    spaced_close = "</" + "think" + " >"
    out, _ = run([open_with_attr + "z" + spaced_close + "tail"])
    assert out == "tail"


def test_multiple_blocks():
    out, _ = run(["A" + _OPEN + "1" + _CLOSE + "B" + _OPEN + "2" + _CLOSE + "C"])
    assert out == "ABC"


def test_partial_tag_tail_is_flushed_as_text():
    # a lone '<' at end of stream that never became a tag must not be lost
    out, stripped = run(["hello <"])
    assert out == "hello <"
    assert stripped == 0