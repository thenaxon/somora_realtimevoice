"""B3: diag_scenarios.py — deterministic synthetic hypothesis measurements.

Each scenario drives the public WS API against fake backends and asserts the
MECHANISM it is meant to measure is reproduced. These are mechanism checks,
not root-cause claims. All audio/content synthetic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from diag_scenarios import (  # noqa: E402
    SCENARIOS,
    close_all,
    run_scenarios,
    scenario_h1,
    scenario_h1_pacing,
    scenario_h2,
    scenario_h4,
    scenario_h5a,
    scenario_h5b,
    scenario_h7,
)


def test_all_scenarios_registered():
    assert set(SCENARIOS) == {"H1", "H1p", "H2", "H4", "H5a", "H5b", "H7"}


def test_h4_back_to_back_create_without_items():
    m = scenario_h4()
    # two responses on the SAME context (no user/tool item in between)
    assert m["llm_calls"] == 2
    assert m["responses_done"] == 2
    assert m["duplicated_answer"] is True
    assert m["no_new_items"] >= 1
    # the busy guard rejects a create fired while one is active
    assert m["busy_error_code"] == "already_active_response"
    # H4 visibility: the post-done create is flagged as a duplicate suspect
    assert m["duplicate_request_suspect"][-1] is True


def test_h1_bargein_after_done_hits_estimated_tail():
    m = scenario_h1()
    assert m["response_done_status"] == "completed"
    assert m["estimated_playback_tail_ms"] is not None
    assert m["estimated_playback_tail_ms"] > 0
    assert m["barge_in_during_tail"] is True
    # nothing is active server-side anymore: no cancel happens
    assert m["server_side_cancel_after_done"] is False


def test_h1_pacing_reduces_estimated_tail():
    m = scenario_h1_pacing()
    assert m["on"]["status"] == "completed"
    # pacing keeps the server response open until audio is nearly due, so the
    # estimate of still-unplayed audio at response.done collapses
    assert m["tail_reduced"] is True
    assert m["on"]["estimated_playback_tail_ms"] < m["off"]["estimated_playback_tail_ms"]


def test_h2_short_burst_kills_response_without_new_turn():
    m = scenario_h2()
    # BEFORE (barge_in_confirm_ms=0): the burst cancelled the active response
    # but supplied no new turn
    assert m["off"]["burst_discarded"] is True
    assert m["off"]["burst_outcome"] == "discarded_short"
    assert m["off"]["cancelled_response_by_burst"] is True
    assert m["off"]["response_done_status"] == "cancelled"
    assert m["off"]["new_turn_stt_calls"] == 0
    assert m["off"]["kept_bursts"] == 0
    # ... and the question is answered again once the burst proves to be noise
    assert m["off"]["answered_again"] is True
    # AFTER (soft barge-in default): the response survives with its audio
    assert m["on"]["barge_in_pending"] is True
    assert m["on"]["barge_in_rejected"] is True
    assert m["on"]["cancelled_response_by_burst"] is False
    assert m["on"]["response_done_status"] == "completed"
    assert m["fix_active"] is True


def test_h7_update_mid_utterance_preserves_detector():
    m = scenario_h7()
    assert m["detector_survived_instructions_update"] is True
    assert m["detector_still_speaking"] is True
    assert m["vad_reset_logged_on_td_change"] is True
    assert m["fix_active"] is True


def test_h5a_tool_call_ordering_in_history():
    m = scenario_h5a()
    assert m["tool_call_before_tool_result"] is True
    assert m["assistant_text_kept_in_context"] is True
    # H5a fix: text and tool call live in ONE assistant message (content +
    # tool_calls), followed by the tool result; no separate text message
    assert m["merged_assistant_message"] is True
    assert "assistant_text" not in m["history_shape"]
    assert "assistant_tool_call+text" in m["history_shape"]
    assert "tool_result" in m["history_shape"]


def test_h5b_one_off_instructions_replace_for_one_response_only():
    m = scenario_h5b()
    assert m["first_has_oneoff"] is True
    assert m["first_has_session_instr"] is False
    assert m["second_has_session_instr"] is True
    assert m["second_has_oneoff"] is False
    assert m["replaces_for_one_response_only"] is True
    # only a hash is logged, never the instruction text
    assert m["oneoff_sha8_present"] is True
    assert m["oneoff_text_logged"] is False


def test_run_scenarios_all_and_no_handler_leak():
    import logging

    logger = logging.getLogger("realtimevoice.lifecycle")
    close_all()
    before = list(logger.handlers)
    report = run_scenarios(list(SCENARIOS))
    assert len(report["scenarios"]) == 7
    # every ScenarioClient removed its capture handler
    assert logger.handlers == before