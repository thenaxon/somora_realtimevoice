"""Per-segment TTS telemetry: metadata logs, cancel reasons, env-gated
same-request capture, and the truncation milestone mapping.

The tts_segment lifecycle log must carry segment id/count/status/duration/
cancel-reason metadata (never content). The env-gated capture (segment_capture)
must record the exact TTS input text and the exact emitted PCM of the SAME
request, be inert by default, and stay bounded. The milestone mapping must cut
a barge-in-interrupted transcript to exactly the audible segment boundary.
"""

import asyncio
import base64
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends
from realtimevoice.backends.base import Finish, TextDelta
from realtimevoice.segment_capture import CAPTURE_ENV_VAR, SegmentCapture, capture_dir
from realtimevoice.session import RealtimeSession


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


def b64_pcm(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def cut_chars(item, audio_end_ms) -> int:
    # unbound call: the mapping only depends on item/audio_end_ms
    return RealtimeSession._cut_chars_for_audio(None, item, audio_end_ms)  # type: ignore[arg-type]


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


async def test_tts_segment_metadata_logs(make_session, lifecycle):
    """Each segment gets one tts_segment record: index, status, durations,
    byte/char counts — and never the segment text."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    tts = FakeTTS()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        segments = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert len(segments) == 2, segments
        for idx, seg in enumerate(segments):
            assert seg["segment_index"] == idx
            assert seg["status"] == "ok"
            assert seg["chars"] > 0
            assert seg["pcm_bytes"] > 0
            assert seg["audio_ms"] == pytest.approx(200, abs=1)
            assert seg["response_id"]
            assert "turn_epoch" in seg
        # audio adds up across segments exactly as emitted
        done = next(r for r in lifecycle() if r["action"] == "response_done")
        assert done["audio_ms"] == pytest.approx(400, abs=2)
        # content never leaks: no segment text in any lifecycle record
        for record in lifecycle():
            assert "Erster" not in json.dumps(record)
            assert "Zweiter" not in json.dumps(record)
        # the same request really had two TTS calls with the split sentences
        assert [c["text"] for c in tts.calls] == ["Erster Satz.", "Zweiter Satz."]
    finally:
        await driver.stop()


async def test_queued_segments_discarded_on_cancel_visible(make_session, lifecycle, tmp_path, monkeypatch):
    """Segments still waiting in the TTS queue at cancel time must be visible:
    a tts_queue_discarded metadata log (count + reason) and, in capture mode,
    manifest entries with the queued text — the missing-rest queue must not
    stay invisible. Scheduling semantics unchanged."""
    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    # five sentences -> five segments; the first blocks synthesis so the
    # remaining four sit in the tts queue when the cancel arrives
    llm = FakeLLM(script=[{"text": "Eins eins. Zwei zwei. Drei drei. Vier vier. Fünf fünf."}])

    class BlockFirstSegment(FakeTTS):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seg_gate = asyncio.Event()  # set => blocked

        async def stream_speech(self, text, *, voice=None, language=None):
            self.calls.append({"text": text, "voice": voice, "language": language})
            await self.seg_gate.wait()
            async for chunk in super().stream_speech(text, voice=voice, language=language):
                yield chunk

    tts = BlockFirstSegment()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        created = await driver.wait_for("response.created")
        rid = created["response"]["id"]
        await settle(200)  # LLM stream done -> queue holds segments, tts blocked
        assert tts.calls  # first segment reached the TTS backend
        session.interrupt("barge_in")
        await driver.wait_for("response.done")
        await settle(150)
        discarded_logs = [r for r in lifecycle() if r["action"] == "tts_queue_discarded"]
        assert discarded_logs, "queued-discard must be logged"
        assert discarded_logs[-1]["count"] == 4
        assert discarded_logs[-1]["reason"] == "barge_in"
        assert discarded_logs[-1]["response_id"] == rid
        # capture manifest: 4 queued entries with their text, no audio claims
        manifest = json.loads((tmp_path / f"resp-{rid}-manifest.json").read_text(encoding="utf-8"))
        queued = [s for s in manifest["segments"] if s.get("queued")]
        assert len(queued) == 4
        assert all(s["status"] == "queued_discarded" for s in queued)
        assert all(s["reason"] == "barge_in" for s in queued)
        queued_texts = [(tmp_path / s["text_file"]).read_text(encoding="utf-8") for s in queued]
        assert queued_texts == ["Zwei zwei.", "Drei drei.", "Vier vier.", "Fünf fünf."]
        # the synthesized-and-cancelled first segment is still logged separately
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert segs and segs[-1]["status"] == "cancelled"
        # no content in any lifecycle record
        for record in lifecycle():
            assert "Zwei zwei" not in json.dumps(record)
    finally:
        await driver.stop()


async def test_truncate_mapping_fresh_after_drain(make_session):
    """Regression: the item's segment snapshot is taken while asynchronous TTS
    is still draining later segments; the snapshot must be refreshed after the
    drain, otherwise a later client truncate maps audio to the wrong chars
    (history metadata only — no audible-path claim)."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    tts = FakeTTS(inter_chunk_delay=0.05)  # real awaits: deltas flow during drain
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        done = await driver.wait_for("response.done")
        rid = done["response"]["id"]
        await settle(100)
        item = next(
            i for i in session._conversation if i.get("role") == "assistant"
        )
        total_ms = item.get("truncated_audio_ms", 0) or 400
        # client truncate covering the whole response: every segment is audible
        session.handle_event(
            {
                "type": "conversation.item.truncate",
                "item_id": item["id"],
                "content_index": 0,
                "audio_end_ms": 10_000,
            }
        )
        await settle(50)
        # fresh snapshot: both segments mapped -> full text kept
        assert item["content"][0]["transcript"] == "Erster Satz. Zweiter Satz."
        assert len(item["segments"]) == 2
    finally:
        await driver.stop()


async def test_interrupt_truncation_uses_live_milestones(make_session):
    """Regression: interrupt while the LLM stream is still open (segments not
    yet snapshotted) must still map audible audio to text — the history item
    keeps the audible prefix, not an empty transcript."""
    llm_script_text = "Erster Satz. Zweiter Satz."

    class GateMidStreamLLM(FakeLLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seg_gate = asyncio.Event()  # set => blocked

        async def stream_chat(self, messages, tools=None):
            self.calls.append({"messages": messages, "tools": tools})
            behavior = self.script.pop(0) if self.script else {"text": self.default_text}
            text = behavior.get("text", "")
            for i in range(0, len(text), 8):
                if i >= 16:
                    await self.seg_gate.wait()
                yield TextDelta(text[i : i + 8])
                await asyncio.sleep(0)
            yield Finish("stop")

    llm = GateMidStreamLLM(script=[{"text": llm_script_text}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        await driver.wait_for("response.output_audio.delta")
        await settle(120)  # first segment fully audible (200 ms fake TTS)
        session.interrupt("barge_in")
        truncated = await driver.wait_for("conversation.item.truncated")
        item = session._items_by_id[truncated["item_id"]]
        # audible segment 1 must survive in the history metadata
        assert item["content"][0]["transcript"] == "Erster Satz."
        assert item.get("truncated_audio_ms") == 200
    finally:
        await driver.stop()


async def test_next_turn_history_excludes_never_heard_words(make_session):
    """The client's barge-in truncate (heard prefix, audio_end_ms = actually
    played position) must remove never-heard words from the conversation
    history so the next turn's LLM context does not believe they were said.
    Server audio_ms is the EMITTED prefix; the heard prefix can be smaller."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        assert "Zweiter" in item["content"][0]["transcript"]
        total_audio_ms = item.get("truncated_audio_ms", 0) or 400
        assert total_audio_ms == pytest.approx(400, abs=2)
        # client heard only the first half of segment 1 (100 of 400 emitted ms)
        session.handle_event(
            {
                "type": "conversation.item.truncate",
                "item_id": item["id"],
                "content_index": 0,
                "audio_end_ms": 100,
            }
        )
        await settle(50)
        kept = item["content"][0]["transcript"]
        assert kept.startswith("Erster")
        assert "Zweiter" not in kept
        # the next turn's LLM context contains only the heard prefix
        messages = session._build_messages(None)
        assistant = [m for m in messages if m["role"] == "assistant" and m.get("content")]
        assert len(assistant) == 1
        assert "Zweiter" not in assistant[0]["content"]
        assert assistant[0]["content"] == kept
    finally:
        await driver.stop()


async def test_bargein_truncate_server_ordering_active_item(make_session, lifecycle):
    """Client barge-in truncate arrives AFTER the server's interrupt: the
    server registers the item synchronously in interrupt() (before the
    speech_started event even reaches the client), so the client's truncate
    for the ACTIVE item must find it — no item_not_found, refined to the
    heard prefix. Proves the ordering instead of guessing a deferral."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])

    class BlockSecondSegment(FakeTTS):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seg_gate = asyncio.Event()  # set => blocked

        async def stream_speech(self, text, *, voice=None, language=None):
            self.calls.append({"text": text, "voice": voice, "language": language})
            if len(self.calls) >= 2:
                await self.seg_gate.wait()
            async for chunk in super().stream_speech(text, voice=voice, language=language):
                yield chunk

    tts = BlockSecondSegment()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        await driver.wait_for("response.output_audio.delta")
        await settle(120)  # segment 1 fully emitted (200 ms), segment 2 blocked
        session.interrupt("barge_in")
        done = await driver.wait_for("response.done")
        assert done["response"]["status"] == "cancelled"
        await settle(100)
        # client truncate for the interrupted item, heard 100 ms < emitted 200 ms
        session.handle_event(
            {
                "type": "conversation.item.truncate",
                "item_id": driver.of_type("conversation.item.truncated")[-1]["item_id"],
                "content_index": 0,
                "audio_end_ms": 100,
            }
        )
        await settle(100)
        errors = [e for e in driver.of_type("error")]
        assert not any("item" in (e.get("error", {}).get("code") or "") for e in errors), errors
        acks = driver.of_type("conversation.item.truncated")
        assert acks and acks[-1]["audio_end_ms"] == 100
        item = session._items_by_id[acks[-1]["item_id"]]
        # heard prefix wins over the emitted prefix the interrupt kept;
        # truncated_audio_ms now carries the latest (heard) cursor
        assert item["content"][0]["transcript"] == "Erster"
        assert item["truncated_audio_ms"] == 100
    finally:
        await driver.stop()


async def test_bargein_truncate_completed_tail_no_item_not_found(make_session):
    """Barge-in during a COMPLETED response's buffered tail: the item was
    registered before response.done, so the client's late truncate must be
    accepted (truncated ack, no item_not_found error) and the heard prefix
    (not the emitted prefix) must survive in history."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        item = next(i for i in session._conversation if i.get("role") == "assistant")
        session.handle_event(
            {
                "type": "conversation.item.truncate",
                "item_id": item["id"],
                "content_index": 0,
                "audio_end_ms": 100,
            }
        )
        await settle(100)
        assert not driver.of_type("error"), driver.of_type("error")
        acks = driver.of_type("conversation.item.truncated")
        assert acks and acks[-1]["item_id"] == item["id"] and acks[-1]["audio_end_ms"] == 100
        assert item["content"][0]["transcript"] == "Erster"
    finally:
        await driver.stop()


async def test_truncate_unknown_item_is_rejected(make_session):
    """The guard the client relies on: a truncate for an unregistered item is
    a clean item_not_found error, never a crash or a wrong-item cut."""
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event(
            {"type": "conversation.item.truncate", "item_id": "item_missing", "content_index": 0, "audio_end_ms": 100}
        )
        err = await driver.wait_for("error")
        assert err["error"]["code"] == "item_not_found"
    finally:
        await driver.stop()


async def test_cancelled_segment_carries_reason(make_session, lifecycle):
    """A barge-in during synthesis logs the segment as cancelled with the
    interrupt reason, and no stale audio is emitted afterwards."""
    from fake_backends import silence_pcm

    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])

    class GateFirstSegment(FakeTTS):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # NOTE: must not be named `gate` — FakeTTS.stream_speech would
            # treat it as its own blocking gate
            self.seg_gate = asyncio.Event()  # set => blocked

        async def stream_speech(self, text, *, voice=None, language=None):
            self.calls.append({"text": text, "voice": voice, "language": language})
            await self.seg_gate.wait()
            async for chunk in super().stream_speech(text, voice=voice, language=language):
                yield chunk

    tts = GateFirstSegment()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        await settle(150)
        assert driver.count("response.output_audio.delta") == 0
        session.interrupt("barge_in")
        await driver.wait_for("response.done")
        await settle(100)
        cancelled = [r for r in lifecycle() if r["action"] == "response_cancelled"]
        assert cancelled and cancelled[-1]["reason"] == "barge_in"
        segs = [r for r in lifecycle() if r["action"] == "tts_segment"]
        assert segs, "interrupted synthesis must still log its segment metadata"
        assert segs[-1]["status"] == "cancelled"
        assert segs[-1]["reason"] == "barge_in"
        assert driver.count("response.output_audio.delta") == 0
    finally:
        await driver.stop()


async def test_segment_capture_same_request(monkeypatch, make_session, tmp_path):
    """With the env var set, capture records the exact TTS input text and the
    exact emitted PCM of the same request; manifest fields are consistent."""
    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    assert capture_dir() == tmp_path
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        created = await driver.wait_for("response.created")
        rid = created["response"]["id"]
        await driver.wait_for("response.done")
        await settle(100)

        manifest = json.loads((tmp_path / f"resp-{rid}-manifest.json").read_text(encoding="utf-8"))
        segs = manifest["segments"]
        assert [s["segment_index"] for s in segs] == [0, 1]
        # sent == the exact TTS input of the same request
        texts = [(tmp_path / s["text_file"]).read_text(encoding="utf-8") for s in segs]
        assert texts == ["Erster Satz.", "Zweiter Satz."]
        # raw == the exact PCM emitted on the WebSocket for that response
        deltas = [d for d in driver.of_type("response.output_audio.delta") if d["response_id"] == rid]
        ws_pcm = b"".join(base64.b64decode(d["delta"]) for d in deltas)
        raw = b"".join((tmp_path / s["emitted_file"]).read_bytes() for s in segs)
        assert raw == ws_pcm
        # backend stream == emitted stream for the fake (no chunking loss);
        # both are captured separately so a real backend->emit loss would show
        for s in segs:
            backend = (tmp_path / s["backend_file"]).read_bytes()
            emitted = (tmp_path / s["emitted_file"]).read_bytes()
            assert backend == emitted
        for s in segs:
            assert s["end_byte"] - s["start_byte"] == s["emitted_bytes"]
            assert s["status"] == "ok"
            assert s["chars"] == len(texts[s["segment_index"]])
        assert manifest["turn_epoch"] == 1  # one speech turn happened
    finally:
        await driver.stop()


def test_segment_capture_inert_by_default(monkeypatch):
    monkeypatch.delenv(CAPTURE_ENV_VAR, raising=False)
    assert capture_dir() is None


async def test_segment_capture_bounded(monkeypatch, tmp_path):
    """A runaway response cannot create more than MAX_SEGMENTS_PER_RESPONSE
    artifact sets."""
    from realtimevoice.segment_capture import MAX_SEGMENTS_PER_RESPONSE

    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    cap = SegmentCapture("resp_test", 0)
    for i in range(MAX_SEGMENTS_PER_RESPONSE + 5):
        cap.record(
            segment_index=i,
            text="x",
            backend_pcm=b"\x00\x00",
            emitted_pcm=b"\x00\x00",
            seg_start_ms=0,
            audio_ms=1,
            start_byte=i * 2,
            end_byte=i * 2 + 2,
            status="ok",
        )
    assert len(cap.entries) == MAX_SEGMENTS_PER_RESPONSE
    assert cap.truncated


def test_segment_capture_byte_budget(monkeypatch, tmp_path):
    from realtimevoice.segment_capture import MAX_BYTES_PER_RESPONSE

    monkeypatch.setenv(CAPTURE_ENV_VAR, str(tmp_path))
    cap = SegmentCapture("resp_budget", 0)
    big = bytes(MAX_BYTES_PER_RESPONSE)
    cap.record(segment_index=0, text="a", backend_pcm=big, emitted_pcm=b"",
               seg_start_ms=0, audio_ms=1, start_byte=0, end_byte=0, status="ok")
    assert len(cap.entries) == 1
    cap.record(segment_index=1, text="b", backend_pcm=b"\x00", emitted_pcm=b"",
               seg_start_ms=1, audio_ms=1, start_byte=0, end_byte=0, status="ok")
    assert len(cap.entries) == 1  # budget exhausted, nothing more recorded
    assert cap.truncated


def test_sentence_buffer_start_char_offsets():
    """Regression: Segment.start_char must point at the segment's first char
    in the FULL LLM stream (captured before _advance), otherwise the
    truncation milestone mapping keeps unheard text in the history."""
    from realtimevoice.session import SentenceBuffer

    full = "Erster Satz. Zweiter Satz."
    buf = SentenceBuffer()
    segs = []
    for ch in full:
        segs += buf.push(ch)
    segs += buf.flush()
    assert [s.text for s in segs] == ["Erster Satz.", "Zweiter Satz."]
    assert [s.start_char for s in segs] == [0, 13]
    # every segment's [start_char, start_char+len) span yields its own text
    # (inter-segment whitespace belongs to no stripped segment)
    for s in segs:
        assert full[s.start_char : s.start_char + len(s.text)] == s.text

    # leading whitespace and a forced split at max_chars keep true offsets
    full2 = "  " + "Wort " * 60  # no boundary within max_chars? has none -> force split
    buf2 = SentenceBuffer()
    segs2 = []
    for ch in full2:
        segs2 += buf2.push(ch)
    segs2 += buf2.flush()
    for s in segs2:
        assert full2[s.start_char : s.start_char + len(s.text)] == s.text


def test_cut_chars_for_audio_milestone_mapping():
    """The truncation mapping keeps exactly the chars of fully audible
    segments and a proportional part of a partially audible one."""
    item = {"segments": [[0, 0, 1000, 12], [1000, 12, 1000, 13]]}
    assert cut_chars(item, 0) == 0
    assert cut_chars(item, 1000) == 12
    assert cut_chars(item, 2000) == 25
    assert cut_chars(item, 1500) == 12 + 6
    assert cut_chars(item, 3000) == 25  # beyond the last milestone
    assert cut_chars({}, 1000) == 0
    assert cut_chars({"segments": [[500, 7, 100, 5]]}, 100) == 0


async def test_barge_in_truncates_transcript_to_audible_text(make_session, lifecycle):
    """Interrupt between segments: the conversation item keeps exactly the
    text whose audio was actually emitted — the LLM context must not believe
    more was said than the user could hear."""
    llm = FakeLLM(script=[{"text": "Erster Satz. Zweiter Satz."}])

    class BlockSecondSegment(FakeTTS):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # NOTE: must not be named `gate` — FakeTTS.stream_speech would
            # treat it as its own blocking gate
            self.seg_gate = asyncio.Event()  # set => blocked

        async def stream_speech(self, text, *, voice=None, language=None):
            self.calls.append({"text": text, "voice": voice, "language": language})
            if len(self.calls) >= 2:
                await self.seg_gate.wait()
            async for chunk in super().stream_speech(text, voice=voice, language=language):
                yield chunk

    tts = BlockSecondSegment()
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, tts))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.created")
        # wait until the first segment is fully emitted (one 100 ms chunk)
        await driver.wait_for("response.output_audio.delta")
        await settle(120)
        session.interrupt("barge_in")
        truncated = await driver.wait_for("conversation.item.truncated")
        item_id = truncated["item_id"]
        item = session._items_by_id.get(item_id)
        assert item is not None
        # milestone mapping: segment 1 (12 chars) fully audible, segment 2 not
        assert item["content"][0]["transcript"] == "Erster Satz."
        assert item in session._conversation
    finally:
        await driver.stop()
