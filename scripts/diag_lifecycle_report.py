#!/usr/bin/env python3
"""Lifecycle diagnostics report: map metadata-only lifecycle logs to H1-H7.

Reads `realtimevoice.lifecycle` JSON lines (a file, a list of files, or
stdin — e.g. `journalctl --user -u <unit> -o cat`) and produces per-session
and aggregate metrics for the interruption/duplication hypotheses H1-H7.
Output is JSON + a short markdown summary.

Privacy contract: only the fields the lifecycle allowlist already writes are
read; the report never contains transcripts, instructions, tool arguments,
audio, tokens or secrets. Its own output is numbers/ids/enums only.

Usage:
  scripts/diag_lifecycle_report.py journal.log --json out.json --md out.md
  journalctl --user -u realtimevoice.service -o cat | scripts/diag_lifecycle_report.py -
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

# hypothesis id -> human label (used in the markdown and JSON keys)
HYPOTHESES = {
    "H1": "playback tail: barge-in after response.done while audio would still play",
    "H2": "VAD sensitivity: short bursts / discarded bursts / epoch churn",
    "H3": "echo: TTS re-enters mic and is transcribed",
    "H4": "client response multiplication: back-to-back response.create",
    "H5": "LLM context shape: tool ordering / one-off instructions / max_tokens",
    "H6": "TTS repetition inside one segment",
    "H7": "session.update rebuilt the VAD detector mid-utterance",
}


def parse_line(line: str) -> dict[str, Any] | None:
    """Parse one log line into a lifecycle record.

    Accepts both pure JSON and syslog/timestamp-prefixed lines (the deployed
    runtime prefixes the JSON payload with a timestamp/level). Returns None
    for non-lifecycle or malformed lines.
    """
    line = line.strip()
    if not line:
        return None
    idx = line.find("{")
    if idx < 0:
        return None
    try:
        record = json.loads(line[idx:])
    except ValueError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("action"), str):
        return None
    return record


def parse_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    return [record for record in (parse_line(line) for line in lines) if record is not None]


def _stats(values: list[float]) -> dict[str, Any]:
    values = [float(v) for v in values if isinstance(v, (int, float))]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 1)

    return {
        "count": n,
        "min": round(ordered[0], 1),
        "max": round(ordered[-1], 1),
        "mean": round(sum(ordered) / n, 1),
        "p50": pct(0.5),
        "p90": pct(0.9),
    }


def _by_session(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sid = record.get("session_id")
        if isinstance(sid, str) and sid:
            grouped[sid].append(record)
        else:
            grouped.setdefault("_unknown", []).append(record)
    return grouped


def session_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-session metric bundle mapped to H1-H7 (numbers/ids only)."""
    actions = Counter(r.get("action", "") for r in records)
    m: dict[str, Any] = {
        "events": len(records),
        "actions": dict(actions),
        "routes": sorted(
            {
                str(r.get("route_mode"))
                for r in records
                if r.get("action") in ("route_resolved", "backend_bound") and r.get("route_mode")
            }
        ),
        "models": sorted(
            {
                str(r.get("model"))
                for r in records
                if r.get("action") in ("llm_request", "llm_finish") and r.get("model")
            }
        ),
    }

    # --- H1 playback tail: barge-ins during an estimated tail
    speech = [r for r in records if r.get("action") == "speech_started"]
    m["h1"] = {
        "speech_started": len(speech),
        "during_estimated_tail": sum(1 for r in speech if r.get("during_estimated_playback_tail") is True),
        "ms_since_last_response_done": _stats(
            [r["ms_since_last_response_done"] for r in speech if isinstance(r.get("ms_since_last_response_done"), (int, float))]
        ),
        "estimated_playback_tail_ms": _stats(
            [r["estimated_playback_tail_ms"] for r in records if isinstance(r.get("estimated_playback_tail_ms"), (int, float))]
        ),
        "response_done_completed": sum(
            1
            for r in records
            if r.get("action") == "response_done" and r.get("status") == "completed"
        ),
        "response_cancelled": sum(
            1 for r in records if r.get("action") == "response_done" and r.get("status") == "cancelled"
        ),
    }

    # --- H2 VAD sensitivity / endpointing
    bursts = [r for r in records if r.get("action") == "vad_burst"]
    discarded = [r for r in records if r.get("action") == "vad_burst_discarded"]
    m["h2"] = {
        "vad_burst": len(bursts),
        "outcomes": dict(Counter(str(r.get("outcome")) for r in bursts)),
        "speech_ms": _stats([r.get("speech_ms") for r in bursts]),
        "gap_ms_since_prev_stop": _stats(
            [r["gap_ms_since_prev_stop"] for r in bursts if isinstance(r.get("gap_ms_since_prev_stop"), (int, float))]
        ),
        "burst_discarded": len(discarded),
        "burst_discarded_outcomes": dict(Counter(str(r.get("outcome")) for r in discarded)),
        "discarded_cancelled_response": sum(
            1 for r in discarded if r.get("response_cancelled_by_burst") is True
        ),
        "speech_started_without_stopped": max(
            0, len(speech) - actions.get("speech_stopped", 0)
        ),
    }

    # --- H3 echo: STT transcript resembling the previous assistant output
    echo = [r.get("stt_echo_ratio") for r in records if isinstance(r.get("stt_echo_ratio"), (int, float))]
    m["h3"] = {
        "stt_completed": actions.get("stt_completed", 0),
        "stt_echo_ratio": _stats(echo),
        "high_echo_ge_0_6": sum(1 for v in echo if v >= 0.6),
    }

    # --- H4 response multiplication: reservations without new items
    reserved = [r for r in records if r.get("action") == "response_reserved"]
    new_items = [r.get("new_items_since_last_response", 0) for r in reserved]
    m["h4"] = {
        "response_reserved": len(reserved),
        "sources": dict(Counter(str(r.get("source")) for r in reserved)),
        "no_new_items": sum(1 for r in reserved if r.get("new_items_since_last_response") == 0),
        "duplicate_request_suspect": sum(
            1 for r in reserved if r.get("duplicate_request_suspect") is True
        ),
        "new_items_since_last_response": _stats(new_items),
        "ms_since_last_response_done": _stats(
            [r["ms_since_last_response_done"] for r in reserved if isinstance(r.get("ms_since_last_response_done"), (int, float))]
        ),
        "assistant_repeat_ratio": _stats(
            [r["assistant_repeat_ratio"] for r in records if isinstance(r.get("assistant_repeat_ratio"), (int, float))]
        ),
        "high_repeat_ge_0_8": sum(
            1
            for r in records
            if isinstance(r.get("assistant_repeat_ratio"), (int, float))
            and r["assistant_repeat_ratio"] >= 0.8
        ),
    }

    # --- H5 LLM context shape
    llm_req = [r for r in records if r.get("action") == "llm_request"]
    llm_fin = [r for r in records if r.get("action") == "llm_finish"]
    nonstandard_sequences: list[str] = []
    for r in llm_req:
        seq = str(r.get("role_sequence", ""))
        # a function_call BEFORE the assistant text of the same response:
        # 'a[tc]' followed later by a plain 'a' (a[tc] then a) is the H5a shape
        if "a[tc]" in seq and seq.split("a[tc]", 1)[-1].lstrip(",").startswith("a"):
            nonstandard_sequences.append(seq)
    m["h5"] = {
        "llm_request": len(llm_req),
        "role_sequences": dict(Counter(str(r.get("role_sequence")) for r in llm_req)),
        "tool_call_before_assistant_text": len(nonstandard_sequences),
        "one_off_instructions": sum(1 for r in reserved if r.get("has_instructions") is True),
        "instruction_hashes": sorted(
            {str(r.get("instructions_sha8")) for r in reserved if r.get("instructions_sha8")}
        ),
        "max_tokens": sorted(
            {r.get("max_tokens") for r in llm_req if r.get("max_tokens") is not None}
        ),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in llm_fin)),
        "output_chars": _stats([r.get("output_chars") for r in llm_fin]),
        "tool_calls_count": sum(
            int(r.get("tool_calls_count") or 0)
            for r in llm_fin
            if isinstance(r.get("tool_calls_count"), (int, float))
        ),
    }

    # --- H6 TTS repetition inside one segment
    segments = [r for r in records if r.get("action") == "tts_segment"]
    m["h6"] = {
        "tts_segment": len(segments),
        "statuses": dict(Counter(str(r.get("status")) for r in segments)),
        "segment_audio_ms": _stats([r.get("audio_ms") for r in segments]),
    }

    # --- H7 VAD reset
    resets = [r for r in records if r.get("action") == "vad_reset"]
    m["h7"] = {
        "vad_reset": len(resets),
        "resets_speaking_before": sum(1 for r in resets if r.get("speaking_before") is True),
        "reasons": dict(Counter(str(r.get("reason")) for r in resets)),
    }

    return m


def aggregate(session_bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-session metrics into one cross-session bundle."""
    total: dict[str, Any] = {"sessions": len(session_bundles)}
    for hyp in HYPOTHESES:
        hyp_metrics: dict[str, Any] = defaultdict(int)
        for bundle in session_bundles.values():
            for key, value in (bundle.get(hyp.lower()) or {}).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    hyp_metrics[key] += value
        total[hyp] = dict(hyp_metrics)
    return total


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _by_session(records)
    sessions = {sid: session_metrics(recs) for sid, recs in sorted(grouped.items())}
    return {
        "record_count": len(records),
        "session_count": len([s for s in sessions if s != "_unknown"]),
        "sessions": sessions,
        "aggregate": aggregate({k: v for k, v in sessions.items() if k != "_unknown"}),
        "hypotheses": HYPOTHESES,
    }


def _fmt_stats(stats: dict[str, Any]) -> str:
    if not stats or not stats.get("count"):
        return "n/a"
    return f"min={stats['min']} p50={stats['p50']} p90={stats['p90']} max={stats['max']} n={stats['count']}"


def render_markdown(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    lines: list[str] = []
    lines.append("# Lifecycle diagnostics report")
    lines.append("")
    lines.append(f"- records: {report['record_count']}")
    lines.append(f"- sessions: {report['session_count']}")
    lines.append("")
    lines.append("## Aggregate per hypothesis")
    lines.append("")
    lines.append("| Hypothesis | signal |")
    lines.append("|---|---|")
    for hyp, label in HYPOTHESES.items():
        lines.append(f"| {hyp} | {label} |")
    lines.append("")
    lines.append("### Key counters")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for hyp in HYPOTHESES:
        for key in sorted(agg.get(hyp, {})):
            lines.append(f"| {hyp}.{key} | {agg[hyp][key]} |")
    lines.append("")
    lines.append("## Per session")
    lines.append("")
    for sid, bundle in report["sessions"].items():
        lines.append(f"### {sid}")
        lines.append("")
        lines.append(f"- events: {bundle['events']}")
        for hyp in HYPOTHESES:
            m = bundle.get(hyp.lower()) or {}
            compact = ", ".join(
                f"{k}={v if not isinstance(v, dict) else json.dumps(v, separators=(',', ':'))}"
                for k, v in m.items()
            )
            lines.append(f"- {hyp}: {compact}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def read_records(paths: list[str]) -> list[dict[str, Any]]:
    if not paths or paths == ["-"]:
        return parse_lines(sys.stdin)
    records: list[dict[str, Any]] = []
    for path in paths:
        if path == "-":
            records.extend(parse_lines(sys.stdin))
        else:
            with open(path, "r", encoding="utf-8") as fh:
                records.extend(parse_lines(fh))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="*", default=["-"], help="log file(s); '-' or none = stdin")
    parser.add_argument("--json", dest="json_out", default="", help="write JSON report here")
    parser.add_argument("--md", dest="md_out", default="", help="write markdown report here")
    args = parser.parse_args(argv)

    records = read_records(args.inputs)
    report = build_report(records)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    else:
        print(payload)
    markdown = render_markdown(report)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(markdown)
    else:
        sys.stderr.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())