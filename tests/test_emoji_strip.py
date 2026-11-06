"""Inline emoji strip at the TTS boundary (visible-emoji lane).

The field evidence: emoji sat INLINE in spoken segments ("Hallo. 😊") and
emoji-only segments were synthesized into seconds of audio (one U+1F60A ->
3.08 s). The strip removes emoji SEQUENCES from the SPOKEN text only; the
display/transcript text and the original-offset/truncation mapping keep the
original bytes. Preserved explicitly: text-presentation forms (©, ™, ❤︎),
math/currency, digits (numeric keycaps normalize to the digit), letters of
every script, combining marks, joiners/selectors in intact non-emoji
contexts. No claim is made that emoji audio was noise — only that it was
synthesized and now is not.
"""

import asyncio
import base64
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.segment_capture import CAPTURE_ENV_VAR
from realtimevoice.session import (
    SPEECH_OUTPUT_DIRECTIVE,
    RealtimeSession,
    ResponseContext,
    SentenceBuffer,
)
from realtimevoice.symbol_filter import strip_emoji_for_speech


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


# ------------------------------------------------------------------ unit policy

STRIP_CASES = [
    # exact brief examples: emoji adjacent to speech, both placements
    ("Hallo. 😊", "Hallo."),
    ("Hallo 😊", "Hallo"),
    # glue guard (incl. a dangling ZWJ that must not mask the word contact)
    ("Hallo😊Welt", "Hallo Welt"),
    ("abc😊\u200ddef", "abc def"),
    # multi-sequence inline text
    ("Hallo. 😊 Wie gehts? 👨‍👩‍👧‍👦 gut", "Hallo. Wie gehts? gut"),
    # emoji-only -> empty (segment skip decision happens in the session)
    ("😊", ""),
    ("🇩🇪", ""),
    ("👨‍👩‍👧", ""),
    ("👍🏽", ""),
    # keycaps normalize to the digit in BOTH forms (VS16-qualified and bare)
    ("1️⃣", "1"),
    ("1⃣", "1"),
    ("2️⃣ Warten", "2 Warten"),
    # text presentation and meaning-carrying symbols survive
    ("©", "©"),
    ("™", "™"),
    ("Preis: 42 €", "Preis: 42 €"),
    ("2+2=4", "2+2=4"),
    ("∑", "∑"),
    ("2 × 3 = 6", "2 × 3 = 6"),
    ("✕", "✕"),
    ("➕", "➕"),
    ("⬅", "⬅"),
    ("⬅️", "⬅️"),
    ("❤︎", "❤︎"),
    ("😊✕", "✕"),
    # VS16-qualified emoji forms are removed
    ("©️", ""),
    ("™️", ""),
    ("❤️", ""),
    ("⚠️", ""),
    # scripts, combining marks, punctuation untouched
    ("مرحبا 你好 नमस्ते é", "مرحبا 你好 नमस्ते é"),
    ("Ärger?", "Ärger?"),
    # joiners/selectors inside intact non-emoji scripts are untouched
    ("क्\u200dष", "क्\u200dष"),
    # modifier-only leftovers pass (no blanket modifier deletion)
    ("🏽", "🏽"),
    ("‍", "‍"),
    ("️", "️"),
]


@pytest.mark.parametrize("source,expected", STRIP_CASES)
def test_strip_cases(source, expected):
    result = strip_emoji_for_speech(source)
    assert result.text == expected


def test_identity_without_emoji_is_byte_exact():
    text = "Ganz normaler Text, 2+2=4, 42 €, Ärger?"
    result = strip_emoji_for_speech(text)
    assert result.text == text
    assert result.removed == 0
    assert result.offsets == list(range(len(text) + 1))


def test_original_string_never_modified():
    text = "Hallo. 😊 Wie gehts? 👨‍👩‍👧‍👦"
    before = text
    strip_emoji_for_speech(text)
    assert text == before


@pytest.mark.parametrize(
    "source",
    ["Hallo. 😊", "Hallo😊Welt", "abc😊\u200ddef", "😊 Hallo 😊", "1️⃣0 Dinge", "a  b 😊 c", "🇩🇪🇩🇪 x"],
)
def test_offsets_map_kept_characters_to_original_positions(source):
    result = strip_emoji_for_speech(source)
    # every cleaned char maps back to the same char in the original — except
    # the synthetic unglue space, which maps INTO the removed span it separates
    for cleaned_idx, orig_idx in enumerate(result.offsets[:-1]):
        assert (
            source[orig_idx] == result.text[cleaned_idx]
            or (result.text[cleaned_idx] == " " and source[orig_idx] != " ")
        ), (source, cleaned_idx, orig_idx)
    # offsets are strictly monotonic and end at the original length
    assert result.offsets == sorted(result.offsets)
    assert result.offsets[-1] == len(source)
    # the map is complete: every non-removed original char is reachable
    assert len(result.text) == len(result.offsets) - 1


def test_dangling_modifiers_never_survive_a_removal():
    # a ZWJ, VS, keycap or skin-tone modifier directly adjacent to a removed
    # sequence must not remain as invisible junk after the strip
    for source in ("😊\u200d", "x😊\u200dy", "😊️", "😊🏽", "😊⃣"):
        result = strip_emoji_for_speech(source)
        for ch in result.text:
            assert ch not in ("\u200d", "\uFE0E", "\uFE0F", "\u20E3", "🏽"), (source, result.text)


def test_keycap_digit_keeps_meaning_not_enclosure():
    for source in ("1️⃣", "1⃣", "1️⃣0 Dinge", "2️⃣ Warten"):
        result = strip_emoji_for_speech(source)
        assert result.text and result.text[0].isdigit()
        assert "\u20E3" not in result.text and "\uFE0F" not in result.text


# ------------------------------------------------- SentenceBuffer offsets intact


def test_sentence_buffer_offsets_with_inline_emoji():
    full = "Erster Satz. Zweiter Satz, 😊 oder? Dritter 👨‍👩‍👧 Satz."
    buf = SentenceBuffer()
    segments = []
    for ch in full:
        segments.extend(buf.push(ch))
    segments.extend(buf.flush())
    joined = "".join(s.text for s in segments)
    for seg in segments:
        # start_char always points at the segment text inside the full stream
        assert full[seg.start_char : seg.start_char + len(seg.text)] == seg.text
    # every non-whitespace display char is covered by exactly one segment
    covered = "".join(
        full[seg.start_char : seg.start_char + len(seg.text)] for seg in segments
    )
    assert "".join(covered.split()) == "".join(full.split())


# ------------------------------------------------------- e2e: inline strip + map


async def test_inline_emoji_stripped_from_tts_display_unchanged(make_session, lifecycle, monkeypatch, tmp_path):
    """Inline emoji in SPOKEN segments reaches the TTS backend cleaned while
    the display transcript keeps the original bytes; lifecycle logs carry the
    removed count only (never content), and the milestone map keeps the
    ORIGINAL offsets so a full truncate keeps the whole display text."""
    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    llm = FakeLLM(script=[{"text": "Hallo. 😊 Wie gehts? 👨‍👩‍👧 Gut."}])
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        await settle(100)

        original = "Hallo. 😊 Wie gehts? 👨‍👩‍👧 Gut."
        # the SPOKEN text reached the backend cleaned, emoji gone, words intact
        assert [c["text"] for c in tts.calls] == [
            "Hallo. Wie gehts? Gut.",
        ] or all("😊" not in c["text"] and "👨" not in c["text"] for c in tts.calls)
        for c in tts.calls:
            assert "😊" not in c["text"] and "\u200d" not in c["text"]

        # the DISPLAY text (transcript, deltas, item) keeps the original bytes
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        assert item["content"][0]["transcript"] == original
        transcript_deltas = [
            e["delta"] for e in driver.of_type("response.output_audio_transcript.delta")
        ]
        assert "".join(transcript_deltas) == original

        # metadata-only logs: removed counts present where emoji sat, content never
        segs = [r for r in lifecycle() if r["action"] == "tts_segment" and r.get("status") == "ok"]
        assert segs and any(s.get("emoji_removed", 0) >= 1 for s in segs)
        for record in lifecycle():
            assert "😊" not in json.dumps(record, ensure_ascii=False)
            assert "👨" not in json.dumps(record, ensure_ascii=False)

        # capture: original + spoken text files differ, spoken has no emoji
        manifest = json.loads(
            next(tmp_path.glob("resp-*-manifest.json")).read_text(encoding="utf-8")
        )
        spoken_files = [s["spoken_text_file"] for s in manifest["segments"] if "spoken_text_file" in s]
        assert spoken_files
        for f in spoken_files:
            spoken = (tmp_path / f).read_text(encoding="utf-8")
            assert "😊" not in spoken and "👨" not in spoken

        # a FULL client truncate keeps the whole ORIGINAL display text
        session.handle_event(
            {"type": "conversation.item.truncate", "item_id": item["id"], "content_index": 0, "audio_end_ms": 10_000}
        )
        await settle(50)
        assert item["content"][0]["transcript"] == original
        assert not driver.of_type("error")
    finally:
        await driver.stop()


async def test_mid_segment_truncate_maps_to_original_text(make_session, lifecycle):
    """A partial audio_end_ms inside an emoji-stripped segment cuts the
    ORIGINAL transcript at the position that was actually heard (via the
    cleaned->original offset map), never beyond it and never off-text."""
    llm = FakeLLM()
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        resp = ResponseContext(
            id=session._next_id("resp"),
            gen=session._gen,
            instructions=None,
            tts_queue=asyncio.Queue(),
            item={
                "id": session._next_id("item"),
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [{"type": "audio", "transcript": ""}],
                "segments": [],
            },
            turn_epoch=session._turn_epoch,
        )
        original = "Hallo 😊 Welt"
        resp.tts_queue.put_nowait(ResponseSegmentShim(original, 0))
        resp.tts_queue.put_nowait(None)
        await session._tts_loop(resp)

        # the backend got the cleaned text
        assert [c["text"] for c in tts.calls] == ["Hallo Welt"]
        # the milestone carries the original offsets (5th element)
        assert len(resp.milestones) == 1
        milestone = resp.milestones[0]
        assert milestone[1] == 0 and milestone[3] == len(original)
        offsets = milestone[4]
        assert offsets[:6] == [0, 1, 2, 3, 4, 5]  # "Hallo " untouched
        assert offsets[6] == 8  # after the strip, the next char is 'W' at 8
        assert offsets[-1] == len(original)

        # half the audio -> half the SPOKEN text -> original position mapping
        resp.item["segments"] = [list(m) for m in resp.milestones]
        item = {"segments": resp.item["segments"]}
        half = RealtimeSession._cut_chars_for_audio(None, item, 100.0)
        assert original[:half] == "Hallo " or original[:half] == "Hallo"
        # full audio -> the whole ORIGINAL text (display completeness)
        full = RealtimeSession._cut_chars_for_audio(None, item, 10_000.0)
        assert full == len(original)
    finally:
        await driver.stop()


class ResponseSegmentShim:
    """Minimal stand-in for a Segment with a set start_char."""

    def __init__(self, text: str, start_char: int):
        self.text = text
        self.start_char = start_char
        self.text_len = len(text)


async def test_multi_segment_offsets_stay_aligned_across_skips(make_session, lifecycle):
    """Mixed stream: a stripped MIDDLE segment must not shift the following
    spoken segment's original start_char, and a mid-segment truncate of the
    LAST segment still lands inside the original transcript."""
    llm = FakeLLM()
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        resp = ResponseContext(
            id=session._next_id("resp"),
            gen=session._gen,
            instructions=None,
            tts_queue=asyncio.Queue(),
            item={
                "id": session._next_id("item"),
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [{"type": "audio", "transcript": ""}],
                "segments": [],
            },
            turn_epoch=session._turn_epoch,
        )
        full = "Erster Satz. 😊 Dritter Satz, 👍🏽 klar?"
        resp.tts_queue.put_nowait(ResponseSegmentShim("Erster Satz.", 0))
        resp.tts_queue.put_nowait(ResponseSegmentShim("😊", 13))
        resp.tts_queue.put_nowait(ResponseSegmentShim("Dritter Satz, 👍🏽 klar?", 15))
        resp.tts_queue.put_nowait(None)
        await session._tts_loop(resp)

        assert [c["text"] for c in tts.calls] == ["Erster Satz.", "Dritter Satz, klar?"]
        milestones = resp.milestones
        assert [m[1] for m in milestones] == [0, 15]
        assert milestones[1][3] == len("Dritter Satz, 👍🏽 klar?")
        third_offsets = milestones[1][4]
        # offsets are relative to the SEGMENT text; "Dritter Satz," (13 chars)
        # survives byte-exact at its own start (start_char=15 absolute)
        assert third_offsets[:13] == list(range(0, 13))
        # the emoji span (relative 14-15) is skipped; the removal-adjacent
        # whitespace around it collapses to ONE space (kept at its first
        # original position), so "klar?" resumes at original index 17
        assert third_offsets[13] == 13 and third_offsets[14] == 17
        # recompute honestly: offsets must map back into the original text
        original_third = full[15:]
        for cleaned_idx, orig_idx in enumerate(third_offsets[:-1]):
            cleaned = "Dritter Satz, klar?"[cleaned_idx]
            assert original_third[orig_idx] == cleaned or (cleaned == " " and original_third[orig_idx] != " ")

        resp.item["segments"] = [list(m) for m in resp.milestones]
        item = {"segments": resp.item["segments"]}
        full_cut = RealtimeSession._cut_chars_for_audio(None, item, 10_000.0)
        assert full_cut == len(full)
        # the skip log keeps metadata-only semantics
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert [s["status"] for s in segs] == ["ok", "skipped", "ok"]
        assert segs[1]["reason"] == "non_speakable_symbol"
        for record in lifecycle():
            assert "😊" not in json.dumps(record, ensure_ascii=False)
    finally:
        await driver.stop()


# --------------------------------------------------- LLM prompt boundary


async def test_speech_output_directive_at_runtime_boundary(make_session):
    """The spoken-output hygiene boundary is appended server-side to every
    LLM prompt: after the operator's instructions, before the conversation —
    user content and tools pass through untouched and the STORED settings
    are never overwritten."""
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        stored_instructions = session._cfg["instructions"]
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        await settle(100)
        messages = llm.calls[-1]["messages"]
        # boundary present exactly once, as its own system message
        directive_msgs = [m for m in messages if m["role"] == "system" and m["content"] == SPEECH_OUTPUT_DIRECTIVE]
        assert len(directive_msgs) == 1
        # operator instructions still first, byte-identical; user content intact
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == stored_instructions
        assert stored_instructions == session.settings.session.instructions
        assert any(m["role"] == "user" for m in messages)
        # the stored settings object was not rewritten
        assert SPEECH_OUTPUT_DIRECTIVE not in session.settings.session.instructions
    finally:
        await driver.stop()


async def test_directive_cowrites_with_one_off_instructions(make_session):
    """A client one-off instruction keeps its own system message; the
    boundary is appended after it, not merged into it."""
    llm = FakeLLM()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create", "response": {"instructions": "Antworte kurz."}})
        await driver.wait_for("response.created")
        await settle(100)
        messages = llm.calls[-1]["messages"]
        assert messages[0] == {"role": "system", "content": "Antworte kurz."}
        assert messages[1] == {"role": "system", "content": SPEECH_OUTPUT_DIRECTIVE}
    finally:
        await driver.stop()


async def test_keycap_segment_is_spoken_as_digit_not_skipped(make_session, lifecycle):
    """Numeric keycaps normalize to the digit: the segment is SPOKEN (not
    skipped) and the digit meaning reaches the backend without the enclosure."""
    llm = FakeLLM(script=[{"text": "2️⃣ Warten bitte."}])
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        assert tts.calls and tts.calls[0]["text"].startswith("2 ")
        assert "\u20E3" not in tts.calls[0]["text"]
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        assert item["content"][0]["transcript"] == "2️⃣ Warten bitte."
    finally:
        await driver.stop()


async def test_ws_audio_of_stripped_response_is_pcm_complete(make_session, lifecycle):
    """Byte accounting: a response whose segments were emoji-stripped still
    streams exactly the spoken segments' PCM (strip changes text, never the
    audio pipeline)."""
    llm = FakeLLM(script=[{"text": "Hallo. 😊"}])
    tts = FakeTTS(chunks=2, chunk_ms=100)
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        deltas = [d for d in driver.of_type("response.output_audio.delta")]
        ws_bytes = sum(len(base64.b64decode(d["delta"])) for d in deltas)
        assert ws_bytes == 200 * 48  # one spoken segment, 2 x 100 ms
        assert [c["text"] for c in tts.calls] == ["Hallo."]
    finally:
        await driver.stop()
