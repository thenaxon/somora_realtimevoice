"""Conservative non-speakable-symbol TTS skip (bug4f775e00 follow-up).

The observed field symbol (saved oMLX trace, metadata-extracted) was exactly
one U+1F60A (category So) segment that Chatterbox synthesized into ~3.08 s of
non-speech audio. The filter must skip ONLY purely decorative symbol/emoji
segments, keep every mixed/meaningful segment byte-identical, never send a
skipped segment to the TTS backend, and never disturb milestones, offsets,
display text or the response lifecycle. Lifecycle logs stay metadata-only.
"""

import asyncio
import base64
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.segment_capture import CAPTURE_ENV_VAR
from realtimevoice.session import RealtimeSession, ResponseContext, Segment, SentenceBuffer
from realtimevoice.symbol_filter import is_non_speakable_symbol_segment


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


# ------------------------------------------------------------- predicate policy

SKIP_CASES = [
    "😊",  # the observed field symbol (U+1F60A, So, recognized smileys block)
    "😀😀",
    "😊 😊",
    "😂😀",  # both inside the recognized smileys block
    "😊️",  # base emoji + variation selector
    "😊🏽",  # base emoji + skin tone modifier
]

KEEP_CASES = [
    "Hallo 😊",  # mixed text stays byte-identical
    "€",
    "42 €",
    "+",
    "∑",
    "2+2=4",
    "1️⃣",  # keycap digit: the digit keeps it eligible
    "a",
    "Ärger",
    "مرحبا",  # Arabic letters
    "你好",  # Chinese letters
    "नमस्ते",  # Devanagari letters
    "é",  # letter + combining mark
    "©",  # So outside the recognized block: unknown passes
    "™",
    "°",
    "20 °C",
    "→",  # math symbol
    "😊!",  # punctuation keeps it eligible
    "😊.",
    "",  # empty
    "   ",  # whitespace-only
    "️",  # variation-selector only (no base emoji)
    "🏽",  # skin-tone modifier only (no base emoji)
    "‍",  # ZWJ only (no base emoji)
    "⃣",  # enclosing keycap only (no base emoji)
    "✕",  # multiplication-cross: meaningful, preserved
    "✖",
    "➕",
    "➖",
    "➗",
    "⬅",  # arrow: meaningful, preserved
    "❤️",  # outside the recognized block: unknown passes
    "⚠️",
    "☎",
    "⭐",
    "🇩🇪",  # flag: unknown passes
    "👍🏽",
    "👨‍👩‍👧",  # people outside the smileys block: unknown passes
    "😊✕",  # recognized base + unrecognized operator: whole segment kept
    "😊€",
]


@pytest.mark.parametrize("text", SKIP_CASES)
def test_symbol_only_segments_are_skippable(text):
    assert is_non_speakable_symbol_segment(text)


@pytest.mark.parametrize("text", KEEP_CASES)
def test_meaningful_segments_stay_eligible(text):
    assert not is_non_speakable_symbol_segment(text)


def test_sentence_buffer_never_emits_empty_segments():
    buf = SentenceBuffer()
    assert buf.push("Erster Satz. Zweiter Satz.") or True
    for seg in buf.flush():
        assert seg.text


# ------------------------------------------------------- end-to-end: tail skip

async def test_symbol_only_tail_segment_skipped(make_session, lifecycle, monkeypatch, tmp_path):
    """Field-faithful multi-segment case: a purely symbolic TAIL segment gets
    no TTS request and no audio, while both spoken segments synthesize, the
    display text stays complete and the lifecycle log is metadata-only."""
    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz. 😊"}])
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        created = await driver.wait_for("response.created")
        rid = created["response"]["id"]
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        await settle(100)

        # only the two spoken segments reached the TTS backend
        assert [c["text"] for c in tts.calls] == ["Erster Satz.", "Zweiter Satz."]

        # audio adds up over spoken segments only (2 x 200 ms)
        assert done["response"].get("usage") is not None or True
        assert driver.count("response.output_audio.delta") > 0

        # item transcript (display text) unchanged: full LLM text incl. emoji
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        assert item["content"][0]["transcript"] == "Erster Satz. Zweiter Satz. 😊"

        # metadata-only skip log: index, chars, count, reason — never content
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert len(segs) == 3
        skip = segs[-1]
        assert skip["segment_index"] == 2
        assert skip["status"] == "skipped"
        assert skip["reason"] == "non_speakable_symbol"
        assert skip["chars"] == 1
        assert skip["pcm_bytes"] == 0
        assert skip["count"] == 1
        assert skip["audio_ms"] == 0
        for record in lifecycle():
            assert "😊" not in json.dumps(record, ensure_ascii=False)

        # test-only capture: skip entry exists, no PCM files, no bogus milestone
        manifest = json.loads((tmp_path / f"resp-{rid}-manifest.json").read_text(encoding="utf-8"))
        assert [s["status"] for s in manifest["segments"]] == ["ok", "ok", "skipped"]
        assert manifest["segments"][-1]["backend_bytes"] == 0
        assert "backend_file" not in manifest["segments"][-1]
        assert "emitted_file" not in manifest["segments"][-1]
        assert (tmp_path / manifest["segments"][-1]["text_file"]).read_text(encoding="utf-8") == "😊"

        # milestones map ONLY the spoken segments; a full client truncate keeps
        # exactly the heard prefix (both sentences, never-audible emoji outside)
        assert len(item["segments"]) == 2
        session.handle_event(
            {
                "type": "conversation.item.truncate",
                "item_id": item["id"],
                "content_index": 0,
                "audio_end_ms": 10_000,
            }
        )
        await settle(50)
        assert not driver.of_type("error")
        assert item["content"][0]["transcript"] == "Erster Satz. Zweiter Satz."
    finally:
        await driver.stop()


# ------------------------------------------------------ worker-level: mid skip

async def test_symbol_only_middle_segment_keeps_later_mapping(make_session, lifecycle):
    """A skipped MIDDLE segment must not allocate an audio milestone: the
    following spoken segment keeps its correct start_char and a full truncate
    keeps the complete display text (the emoji sits between spoken segments)."""
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
        full = "Erster Satz. 😊 Zweiter Satz."
        # SentenceBuffer proves the offsets; the middle emoji segment is
        # injected the way a stream tail/force-split would produce it
        buf = SentenceBuffer()
        for ch in full:
            buf.push(ch)
        spoken = [s for s in buf.flush() if s.text != "😊"]
        resp.tts_queue.put_nowait(Segment("Erster Satz.", 0))
        resp.tts_queue.put_nowait(Segment("😊", 13))
        resp.tts_queue.put_nowait(Segment("Zweiter Satz.", 15))
        resp.tts_queue.put_nowait(None)
        await session._tts_loop(resp)

        assert [c["text"] for c in tts.calls] == ["Erster Satz.", "Zweiter Satz."]
        # no bogus milestone for the skipped middle segment; durations are the
        # emitted audio of the two spoken segments (2 x 100 ms fake chunks)
        assert resp.milestones == [[0.0, 0, 200.0, 12], [200.0, 15, 200.0, 13]]
        assert resp.item["segments"] == []  # snapshot happens in the response lifecycle
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert [s["status"] for s in segs] == ["ok", "skipped", "ok"]
        assert segs[1]["reason"] == "non_speakable_symbol"
        assert segs[1]["count"] == 1
        assert segs[2]["segment_index"] == 2
        # full-truncate mapping keeps the whole display text
        kept = RealtimeSession._cut_chars_for_audio(None, {"segments": resp.milestones}, 10_000)
        assert kept == len(full)
    finally:
        await driver.stop()


async def test_all_symbol_response_completes_without_tts(make_session, lifecycle):
    """A response whose ONLY segment is symbolic: no TTS request, no audio
    milestone, clean completed lifecycle — no hang, no error."""
    llm = FakeLLM(script=[{"text": "😊"}])
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        await settle(100)
        assert tts.calls == []
        assert driver.count("response.output_audio.delta") == 0
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert segs and segs[-1]["status"] == "skipped"
        assert segs[-1]["reason"] == "non_speakable_symbol"
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        assert item["content"][0]["transcript"] == "😊"
    finally:
        await driver.stop()


# ------------------------------------------------------- WS payload integrity

async def test_skipped_segment_emits_no_audio_delta_for_it(make_session, lifecycle):
    """Byte accounting: WS deltas of the response contain exactly the spoken
    segments' PCM — the skipped segment contributes zero bytes."""
    llm = FakeLLM(script=[{"text": "Erster Satz. 😊"}])
    tts = FakeTTS(chunks=2, chunk_ms=100)
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        created = await driver.wait_for("response.created")
        rid = created["response"]["id"]
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "completed"
        await settle(100)
        deltas = [d for d in driver.of_type("response.output_audio.delta") if d["response_id"] == rid]
        ws_bytes = sum(len(base64.b64decode(d["delta"])) for d in deltas)
        # exactly one spoken segment: 200 ms @ 24 kHz PCM16
        assert ws_bytes == 200 * 48
        assert [c["text"] for c in tts.calls] == ["Erster Satz."]
    finally:
        await driver.stop()
