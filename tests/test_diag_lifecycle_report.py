"""B2: diag_lifecycle_report.py — synthetic lifecycle log parsing/metrics.

Unit-tests the parser and metric mapping with synthetic metadata-only lines
(no content), including syslog-prefixed lines and malformed input.
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from diag_lifecycle_report import (  # noqa: E402
    aggregate,
    build_report,
    main,
    parse_line,
    parse_lines,
    render_markdown,
    session_metrics,
)


def rec(action, session="sess_1", **fields):
    return {"session_id": session, "action": action, "generation": 0, "turn_epoch": 0, **fields}


def test_parse_line_plain_json():
    line = json.dumps(rec("speech_started", audio_ms=100))
    assert parse_line(line)["action"] == "speech_started"


def test_parse_line_syslog_prefixed():
    line = "2026-09-15T10:00:00+02:00 host realtimevoice[1]: " + json.dumps(rec("response_done"))
    assert parse_line(line)["action"] == "response_done"


def test_parse_line_malformed_returns_none():
    assert parse_line("") is None
    assert parse_line("no json here") is None
    assert parse_line("{not json}") is None
    # JSON but not a lifecycle record
    assert parse_line(json.dumps({"hello": "world"})) is None


def test_parse_lines_filters():
    lines = [
        json.dumps(rec("session_open")),
        "garbage",
        json.dumps(rec("response_done")),
    ]
    assert [r["action"] for r in parse_lines(lines)] == ["session_open", "response_done"]


def test_h1_playback_tail_metrics():
    records = [
        rec("speech_started", during_estimated_playback_tail=True, ms_since_last_response_done=120.0),
        rec("speech_started", during_estimated_playback_tail=False, ms_since_last_response_done=5000.0),
        rec("response_done", status="completed", estimated_playback_tail_ms=900.0),
        rec("response_done", status="cancelled"),
    ]
    m = session_metrics(records)["h1"]
    assert m["speech_started"] == 2
    assert m["during_estimated_tail"] == 1
    assert m["response_done_completed"] == 1
    assert m["response_cancelled"] == 1
    assert m["estimated_playback_tail_ms"]["max"] == 900.0


def test_h2_vad_metrics():
    records = [
        rec("vad_burst", outcome="stopped", speech_ms=300, gap_ms_since_prev_stop=-1),
        rec("vad_burst", outcome="max_len", speech_ms=60000),
        rec("vad_burst_discarded", outcome="discarded_short", response_cancelled_by_burst=True),
        rec("vad_burst_discarded", outcome="discarded_short", response_cancelled_by_burst=False),
        rec("speech_started"),
    ]
    m = session_metrics(records)["h2"]
    assert m["vad_burst"] == 2
    assert m["outcomes"] == {"stopped": 1, "max_len": 1}
    assert m["burst_discarded"] == 2
    assert m["discarded_cancelled_response"] == 1
    assert m["speech_started_without_stopped"] == 1


def test_h3_echo_metrics():
    records = [
        rec("stt_completed", stt_echo_ratio=0.95),
        rec("stt_completed", stt_echo_ratio=0.1),
        rec("stt_completed"),  # ratio None dropped from the log
    ]
    m = session_metrics(records)["h3"]
    assert m["stt_completed"] == 3
    assert m["high_echo_ge_0_6"] == 1
    assert m["stt_echo_ratio"]["count"] == 2


def test_h4_multiplication_metrics():
    records = [
        rec("response_reserved", source="client", new_items_since_last_response=0, ms_since_last_response_done=50),
        rec("response_reserved", source="client", new_items_since_last_response=0, ms_since_last_response_done=60),
        rec("response_done", assistant_repeat_ratio=0.9),
    ]
    m = session_metrics(records)["h4"]
    assert m["response_reserved"] == 2
    assert m["no_new_items"] == 2
    assert m["high_repeat_ge_0_8"] == 1
    assert m["sources"] == {"client": 2}


def test_h5_context_shape_metrics():
    records = [
        rec("response_reserved", has_instructions=True, instructions_sha8="abcd1234"),
        rec("llm_request", role_sequence="s*2,a[tc],a,t", message_count=4, tools_count=1, max_tokens=256),
        rec("llm_finish", finish_reason="tool_calls", output_chars=0, tool_calls_count=1),
    ]
    m = session_metrics(records)["h5"]
    assert m["one_off_instructions"] == 1
    assert m["instruction_hashes"] == ["abcd1234"]
    assert m["tool_call_before_assistant_text"] == 1
    assert m["max_tokens"] == [256]
    assert m["finish_reasons"] == {"tool_calls": 1}
    assert m["tool_calls_count"] == 1


def test_h7_vad_reset_metrics():
    records = [
        rec("vad_reset", speaking_before=True, reason="session_update"),
        rec("vad_reset", speaking_before=False, reason="session_update"),
    ]
    m = session_metrics(records)["h7"]
    assert m["vad_reset"] == 2
    assert m["resets_speaking_before"] == 1
    assert m["reasons"] == {"session_update": 2}


def test_build_report_groups_sessions_and_aggregates():
    records = [
        rec("vad_burst", session="s1", outcome="stopped"),
        rec("vad_burst_discarded", session="s1", response_cancelled_by_burst=True),
        rec("vad_burst", session="s2", outcome="stopped"),
    ]
    report = build_report(records)
    assert report["session_count"] == 2
    assert set(report["sessions"]) == {"s1", "s2"}
    agg = report["aggregate"]
    assert agg["sessions"] == 2
    assert agg["H2"]["vad_burst"] == 2
    assert agg["H2"]["burst_discarded"] == 1
    assert agg["H2"]["discarded_cancelled_response"] == 1


def test_render_markdown_mentions_hypotheses():
    report = build_report([rec("speech_started", during_estimated_playback_tail=True)])
    md = render_markdown(report)
    for hyp in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        assert hyp in md
    assert "during_estimated_tail" in md


def test_report_output_has_no_transcript_content():
    # even if a malicious line carried content fields, the report only reads
    # allowlisted metric fields; the rendered output must not echo them
    records = [
        rec("response_reserved", transcript="GEHEIM", instructions="SECRETPROMPT"),
        rec("llm_request", role_sequence="s*2,u", tools_count=0),
    ]
    report = build_report(records)
    payload = json.dumps(report)
    assert "GEHEIM" not in payload
    assert "SECRETPROMPT" not in payload


def test_main_reads_stdin_and_writes_json(tmp_path, monkeypatch, capsys):
    log = tmp_path / "life.log"
    log.write_text(
        "\n".join(
            [
                json.dumps(rec("session_open")),
                "syslog 2026-09-15 prefix " + json.dumps(rec("response_done", status="completed")),
            ]
        )
        + "\n"
    )
    out = tmp_path / "report.json"
    rc = main([str(log), "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["record_count"] == 2


def test_main_from_stdin(monkeypatch, capsys):
    payload = json.dumps(rec("vad_burst", outcome="stopped"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload + "\n"))
    rc = main(["-", "--json", ""])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out)["record_count"] == 1