"""B1 lifecycle diagnostics: metadata-only instrumentation + the H7 VAD fix.

Covers the new log fields (vad_burst / vad_burst_discarded / response
provenance / llm_request shape / similarity ratios / audio timing / vad_reset)
and proves that no transcript or instruction text ever reaches the lifecycle
log. All synthetic content; no real audio.
"""

import asyncio
import base64
import json
import logging

import pytest

from fake_backends import FakeLLM, FakeSTT, FakeTTS, make_backends, silence_pcm, tone_pcm
from realtimevoice.config import Settings
from realtimevoice.vad import BurstDiscarded, SpeechStopped, TurnDetector
from realtimevoice.vad import SpeechStarted


async def settle(ms=80):
    await asyncio.sleep(ms / 1000.0)


@pytest.fixture
def lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="realtimevoice.lifecycle")
    return lambda: [json.loads(r.message) for r in caplog.records if r.name == "realtimevoice.lifecycle"]


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


# ------------------------------------------------------------------ VAD unit

def test_vad_burst_stats_stopped():
    d = TurnDetector(silence_duration_ms=300, prefix_padding_ms=200, threshold_rms=350.0)
    events = d.process(tone_pcm(400) + silence_pcm(500))
    stopped = [e for e in events if isinstance(e, SpeechStopped)]
    assert len(stopped) == 1
    st = stopped[0].stats
    assert st is not None
    assert st.outcome == "stopped"
    assert st.speech_ms >= 380
    assert st.peak_rms > 350
    assert st.mean_rms > 0
    # gap is unset before any previous stop
    assert st.gap_ms_since_prev_stop == -1.0


def test_vad_burst_stats_discarded_short():
    d = TurnDetector(min_speech_ms=100, silence_duration_ms=100, threshold_rms=350.0)
    events = d.process(tone_pcm(40) + silence_pcm(400))
    discarded = [e for e in events if isinstance(e, BurstDiscarded)]
    assert len(discarded) == 1
    assert discarded[0].stats.outcome == "discarded_short"
    assert not any(isinstance(e, SpeechStopped) for e in events)


def test_vad_gap_since_prev_stop_measured():
    d = TurnDetector(silence_duration_ms=200, threshold_rms=350.0)
    first = [e for e in d.process(tone_pcm(300) + silence_pcm(400)) if isinstance(e, SpeechStopped)]
    assert first and first[0].stats.gap_ms_since_prev_stop == -1.0
    second = [
        e
        for e in d.process(silence_pcm(200) + tone_pcm(300) + silence_pcm(400))
        if isinstance(e, SpeechStopped)
    ]
    assert second
    # the second burst's gap is the fed audio since the first stop; positive
    assert second[0].stats.gap_ms_since_prev_stop > 0


def test_vad_max_len_outcome():
    d = TurnDetector(max_utterance_ms=1000, threshold_rms=350.0)
    events = d.process(tone_pcm(2500))
    stopped = [e for e in events if isinstance(e, SpeechStopped)]
    assert stopped
    assert stopped[0].stats.outcome == "max_len"


# --------------------------------------------------------------- session logs

async def test_vad_burst_and_speech_metadata(make_session, lifecycle):
    stt = FakeSTT(text="Hallo.")
    session, driver = make_session(backends=make_backends(stt, FakeLLM(), FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(100)
        records = lifecycle()
        bursts = [r for r in records if r["action"] == "vad_burst"]
        assert bursts, "vad_burst must be logged per kept burst"
        b = bursts[0]
        assert b["outcome"] == "stopped"
        assert b["speech_ms"] > 0
        assert b["peak_rms"] > 0
        assert b["mean_rms"] > 0
        # gap is simply absent on the very first burst (None is dropped)
        started = [r for r in records if r["action"] == "speech_started"]
        assert started
        # first speech of the session: no prior response, tail False and the
        # ms-since-done field legitimately absent (None is dropped)
        assert started[-1]["during_estimated_playback_tail"] is False
        assert "ms_since_last_response_done" not in started[-1]
    finally:
        await driver.stop()


async def test_vad_burst_discarded_soft_barge_in_rejects(make_session, lifecycle):
    # short noise burst during an active response: with the H2 soft barge-in
    # default the response is NOT killed; the burst is discarded and rejected,
    # the response survives with all its audio and no new turn appears
    settings = Settings()
    settings.session.turn_detection.silence_duration_ms = 100
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        # 60 ms tone then silence: enough to trigger speech_started, too short
        # to be kept (min_speech_ms default 100)
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(60))})
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(300))})
        await driver.wait_for("input_audio_buffer.speech_started")
        await settle(150)
        discarded = [r for r in lifecycle() if r["action"] == "vad_burst_discarded"]
        assert discarded, "a discarded burst must be logged"
        assert discarded[-1]["response_cancelled_by_burst"] is False
        assert discarded[-1]["outcome"] == "discarded_short"
        rejected = [r for r in lifecycle() if r["action"] == "barge_in_rejected"]
        assert rejected, "the unconfirmed burst must be logged as rejected"
        assert rejected[-1]["speech_ms"] < 250
        # no cancel: release the gated LLM and the response completes
        llm.gate.set()
        await driver.wait_for("response.done")
        assert driver.of_type("response.done")[-1]["response"]["status"] == "completed"
    finally:
        await driver.stop()


async def test_vad_burst_discarded_immediate_mode_cancels(make_session, lifecycle):
    # barge_in_confirm_ms=0 restores the old immediate behaviour: the short
    # burst cancels the active response but supplies no new turn
    settings = Settings()
    settings.session.turn_detection.silence_duration_ms = 100
    settings.session.turn_detection.barge_in_confirm_ms = 0
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(settings=settings, backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(60))})
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(300))})
        await driver.wait_for("input_audio_buffer.speech_started")
        await settle(150)
        discarded = [r for r in lifecycle() if r["action"] == "vad_burst_discarded"]
        assert discarded, "a discarded burst must be logged"
        assert discarded[-1]["response_cancelled_by_burst"] is True
        assert discarded[-1]["outcome"] == "discarded_short"
        assert driver.count("response.done") == 1
    finally:
        await driver.stop()


async def test_response_provenance_back_to_back_no_new_items(make_session, lifecycle):
    llm = FakeLLM(script=[{"text": "Erste Antwort."}, {"text": "Zweite Antwort."}])
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create", "event_id": "evt_1"})
        await driver.wait_for("response.done")
        await settle(80)
        # second create with NO new items in between (H4)
        session.handle_event({"type": "response.create", "event_id": "evt_2"})
        await driver.wait_next("response.done", 1)
        await settle(80)
        reserved = [r for r in lifecycle() if r["action"] == "response_reserved"]
        assert len(reserved) == 2
        second = reserved[1]
        assert second["new_items_since_last_response"] == 1  # the assistant item
        assert second["new_items_by_type"] == '{"message:assistant":1}'
        assert second["client_event_id"] == "evt_2"
        assert isinstance(second["ms_since_last_response_done"], (int, float))
        assert second["has_instructions"] is False
    finally:
        await driver.stop()


async def test_response_provenance_instructions_sha8_only(make_session, lifecycle):
    secret = "SYSTEMPROMPT-GEHEIM-123456"
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create", "response": {"instructions": secret}})
        await driver.wait_for("response.done")
        await settle(80)
        reserved = [r for r in lifecycle() if r["action"] == "response_reserved"][-1]
        assert reserved["has_instructions"] is True
        # only the sha256 prefix, never the text
        import hashlib

        assert reserved["instructions_sha8"] == hashlib.sha256(secret.encode()).hexdigest()[:8]
        raw = json.dumps(lifecycle())
        assert secret not in raw
    finally:
        await driver.stop()


async def test_llm_request_shape_and_finish(make_session, lifecycle):
    llm = FakeLLM(script=[{"text": "Antwort."}])
    session, driver = make_session(backends=make_backends(FakeSTT(text="Frage."), llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(80)
        req = [r for r in lifecycle() if r["action"] == "llm_request"]
        assert req
        # s*2 = stored instructions + runtime boundary (both system), u = user;
        # the assistant answer is produced by this very response
        assert req[0]["role_sequence"] == "s*2,u"
        assert req[0]["message_count"] >= 3
        assert req[0]["tools_count"] == 0
        finish = [r for r in lifecycle() if r["action"] == "llm_finish"]
        assert finish
        assert finish[0]["finish_reason"] == "stop"
        assert finish[0]["output_chars"] == len("Antwort.")
        assert finish[0]["tool_calls_count"] == 0
    finally:
        await driver.stop()


async def test_llm_request_role_sequence_marks_tool_calls(make_session, lifecycle):
    llm = FakeLLM(
        script=[
            {"text": "", "tool_calls": [{"id": "call_1", "name": "get_w", "arguments": "{}"}]},
            {"text": "Fertig."},
        ]
    )
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            }
        )
        session.handle_event({"type": "response.create"})
        await driver.wait_next("response.done", 1)
        await settle(80)
        reqs = [r for r in lifecycle() if r["action"] == "llm_request"]
        # the second request must contain a tool-call assistant turn then tool
        assert any("a[tc]" in r["role_sequence"] and "t" in r["role_sequence"] for r in reqs)
    finally:
        await driver.stop()


async def test_similarity_ratios_numbers_only(make_session, lifecycle):
    repeated = "Der Zug fährt um sieben Uhr ab."
    llm = FakeLLM(script=[{"text": repeated}, {"text": repeated}])
    # STT echoes the assistant text (H3)
    stt = FakeSTT(text=repeated)
    settings = Settings()
    session, driver = make_session(settings=settings, backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        await settle(80)
        # turn 2: user speech transcribed (echo) + repeated answer
        await driver.append_speech_turn()
        await driver.wait_next("response.done", 1)
        await settle(100)
        completed = [r for r in lifecycle() if r["action"] == "stt_completed"]
        assert completed
        assert completed[-1]["stt_echo_ratio"] == pytest.approx(1.0, abs=0.01)
        dones = [r for r in lifecycle() if r["action"] == "response_done"]
        # second response repeats the first assistant transcript exactly
        assert dones[-1]["assistant_repeat_ratio"] == pytest.approx(1.0, abs=0.01)
        # only rounded numbers, never the text
        raw = json.dumps(lifecycle())
        assert repeated not in raw
    finally:
        await driver.stop()


async def test_audio_timing_and_playback_tail(make_session, lifecycle):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS(chunks=2, chunk_ms=100)))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.done")
        await settle(80)
        done = [r for r in lifecycle() if r["action"] == "response_done"][-1]
        assert done["first_audio_delta_at"] is not None
        assert done["last_audio_delta_at"] is not None
        assert done["last_audio_delta_at"] >= done["first_audio_delta_at"]
        assert done["audio_ms"] > 0
        # server streams much faster than real time here: a tail is estimated
        assert done["estimated_playback_tail_ms"] is not None
        assert done["estimated_playback_tail_ms"] >= 0
    finally:
        await driver.stop()


async def test_speech_started_during_estimated_tail(make_session, lifecycle):
    # a gate on the LLM makes the response take a while, then TTS emits audio
    # quickly relative to its duration: a speech_started right after done must
    # be flagged as during the estimated playback tail
    llm = FakeLLM(gate=asyncio.Event())
    session, driver = make_session(backends=make_backends(FakeSTT(), llm, FakeTTS(chunks=10, chunk_ms=100)))
    try:
        await settle()
        session.handle_event({"type": "response.create"})
        await driver.wait_for("response.created")
        llm.gate.set()
        await driver.wait_for("response.done")
        # speech starts immediately after response.done: still in the tail
        session._on_speech_started()
        await settle(50)
        started = [r for r in lifecycle() if r["action"] == "speech_started"]
        assert started[-1]["during_estimated_playback_tail"] is True
    finally:
        await driver.stop()


async def test_diagnostics_never_leak_transcript_or_instructions(make_session, lifecycle):
    secret_transcript = "GEHEIMERINHALT-XYZZY-42"
    secret_instr = "GEHEIMEPROMPT-ABC-99"
    stt = FakeSTT(text=secret_transcript)
    llm = FakeLLM(script=[{"text": "GEHEIMEANTWORT-QQ-77"}])
    session, driver = make_session(backends=make_backends(stt, llm, FakeTTS()))
    try:
        await settle()
        await driver.append_speech_turn()
        await driver.wait_for("response.done")
        await settle(120)
        raw = json.dumps(lifecycle())
        for leak in (secret_transcript, secret_instr, "GEHEIMEANTWORT-QQ-77"):
            assert leak not in raw
    finally:
        await driver.stop()


# ------------------------------------------------------------------- H7 fix

async def test_session_update_instructions_does_not_reset_vad(make_session, lifecycle):
    """H7: a session.update with instructions/voice/tools must NOT rebuild the
    VAD detector, so an utterance in progress is not dropped."""
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        # put the detector mid-speech
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(200))})
        await settle(60)
        assert session._detector.speaking is True
        detector_before = session._detector
        session.handle_event({"type": "session.update", "session": {"instructions": "neu"}})
        await settle(30)
        assert session._detector is detector_before, "instructions update must not reset VAD"
        assert session._detector.speaking is True
        resets = [r for r in lifecycle() if r["action"] == "vad_reset"]
        assert resets == []
    finally:
        await driver.stop()


async def test_session_update_turn_detection_resets_vad_with_log(make_session, lifecycle):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(200))})
        await settle(60)
        assert session._detector.speaking is True
        session.handle_event(
            {"type": "session.update", "session": {"turn_detection": {"threshold": 0.7}}}
        )
        await settle(30)
        resets = [r for r in lifecycle() if r["action"] == "vad_reset"]
        assert resets, "a real turn-detection change must log vad_reset"
        assert resets[-1]["speaking_before"] is True
        assert resets[-1]["reason"] == "session_update"
    finally:
        await driver.stop()


async def test_session_update_tools_and_voice_keep_detector(make_session):
    session, driver = make_session(backends=make_backends(FakeSTT(), FakeLLM(), FakeTTS()))
    try:
        await settle()
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(tone_pcm(200))})
        await settle(60)
        before = session._detector
        session.handle_event(
            {
                "type": "session.update",
                "session": {
                    "voice": "v2",
                    "tools": [
                        {"type": "function", "name": "f", "parameters": {"type": "object"}}
                    ],
                },
            }
        )
        await settle(30)
        assert session._detector is before
        # the utterance must survive: silence then closes it normally
        session.handle_event({"type": "input_audio_buffer.append", "audio": b64(silence_pcm(600))})
        started = await driver.wait_for("input_audio_buffer.speech_started")
        assert started is not None
        stopped = await driver.wait_for("input_audio_buffer.speech_stopped")
        assert stopped is not None
    finally:
        await driver.stop()