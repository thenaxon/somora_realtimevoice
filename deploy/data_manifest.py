#!/usr/bin/env python3
"""data_manifest.py — secret-free inventory of a realtimevoice data directory.

Usage:
    data_manifest.py <data_dir>
    data_manifest.py <data_dir> --compare <manifest.json>

Prints a JSON summary of the catalog in ``<data_dir>/settings.json`` and the
content-addressed voice samples under ``<data_dir>/voices``. It NEVER emits a
credential, token, transcript or URL value — only counts, public ids, sample
metadata and hashes. That makes the manifest safe to store next to a backup.

Exit status:
    0  manifest produced and (with --compare) identical to the reference
    1  usage / unreadable data dir / settings.json
    2  a referenced sample file is missing, its content hash does not match
       its filename, or --compare found a difference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAMPLE_ID_RE = re.compile(r"smp_[0-9a-f]{16}")


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_settings(data_dir: str) -> tuple[dict, str]:
    path = os.path.join(data_dir, "settings.json")
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _count_backend_rows(settings: dict) -> dict:
    profiles = settings.get("profiles") or {}
    counts = {}
    for role in ("stt", "llm", "tts"):
        rows = profiles.get(role) or {}
        counts[role] = len(rows) if isinstance(rows, dict) else 0
    return counts


def build_manifest(data_dir: str) -> tuple[dict, list[str]]:
    """Return (manifest, problems). ``problems`` is non-empty when a sample is
    missing or its hash does not match its filename."""
    settings, settings_sha = _load_settings(data_dir)
    problems: list[str] = []

    profiles = settings.get("conversation_profiles") or {}
    enabled_sets: list[str] = []
    set_history: dict[str, int] = {}
    if isinstance(profiles, dict):
        for pid, profile in sorted(profiles.items()):
            if not isinstance(profile, dict):
                continue
            if profile.get("enabled") is True:
                enabled_sets.append(pid)
            history = profile.get("history")
            set_history[pid] = len(history) if isinstance(history, list) else 0
    default_set = settings.get("default_set")
    voice_library = settings.get("voice_library") or {}
    transcription_model = (
        voice_library.get("transcription_stt_profile") if isinstance(voice_library, dict) else ""
    )

    voices = settings.get("voices") or {}

    samples: list[dict] = []
    total_bytes = 0
    voice_count = 0
    if isinstance(voices, dict):
        for voice_id, voice in sorted(voices.items()):
            voice_count += 1
            voice_samples = (voice or {}).get("samples") or []
            if not isinstance(voice_samples, list):
                continue
            for sample in voice_samples:
                if not isinstance(sample, dict):
                    continue
                sha = sample.get("sha256", "")
                entry = {
                    "voice_id": voice_id,
                    "sample_id": sample.get("id", ""),
                    "sha256": sha if SHA256_RE.fullmatch(sha or "") else "",
                }
                if not SHA256_RE.fullmatch(sha or ""):
                    entry["present"] = False
                    entry["sha256_matches_filename"] = False
                    entry["bytes"] = 0
                    problems.append(f"{voice_id}: invalid or missing sha256 for a sample")
                    samples.append(entry)
                    continue
                path = os.path.join(data_dir, "voices", voice_id, f"{sha}.wav")
                if not os.path.isfile(path):
                    entry["present"] = False
                    entry["sha256_matches_filename"] = False
                    entry["bytes"] = 0
                    problems.append(f"{voice_id}: missing sample file {sha}.wav")
                    samples.append(entry)
                    continue
                actual, size = _sha256_file(path)
                matches = actual == sha
                entry["present"] = True
                entry["sha256_matches_filename"] = matches
                entry["bytes"] = size
                total_bytes += size
                if not matches:
                    problems.append(
                        f"{voice_id}: hash mismatch for {sha}.wav (content {actual[:12]}...)"
                    )
                samples.append(entry)

    manifest = {
        "settings_version": settings.get("version"),
        "settings_sha256": settings_sha,
        # public ids only: the set for other apps and the transcribing model
        "default_set": default_set if isinstance(default_set, str) else "",
        "transcription_stt_profile": transcription_model if isinstance(transcription_model, str) else "",
        "counts": {
            "conversation_profiles": len(profiles) if isinstance(profiles, dict) else 0,
            "enabled_sets": enabled_sets,
            "set_history": set_history,
            "backend_rows": _count_backend_rows(settings),
            "voices": voice_count,
            "samples": len(samples),
        },
        "samples": samples,
        "total_bytes": total_bytes,
    }
    return manifest, problems


def _diff(reference: dict, current: dict) -> list[str]:
    """Structural comparison over the secret-free manifest only."""
    differences: list[str] = []
    for key in (
        "settings_version",
        "settings_sha256",
        "default_set",
        "transcription_stt_profile",
        "counts",
        "total_bytes",
    ):
        if reference.get(key) != current.get(key):
            differences.append(
                f"{key}: reference={reference.get(key)!r} current={current.get(key)!r}"
            )
    ref_samples = {
        (s.get("voice_id"), s.get("sample_id")): s for s in reference.get("samples", [])
    }
    cur_samples = {
        (s.get("voice_id"), s.get("sample_id")): s for s in current.get("samples", [])
    }
    for key in sorted(set(ref_samples) | set(cur_samples)):
        if ref_samples.get(key) != cur_samples.get(key):
            differences.append(f"sample {key}: differs")
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="runtime data directory to inspect")
    parser.add_argument(
        "--compare",
        default="",
        help="path to a previously produced manifest; exit 2 on any difference",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.data_dir):
        print(f"not a directory: {args.data_dir}", file=sys.stderr)
        return 1
    try:
        manifest, problems = build_manifest(args.data_dir)
    except (OSError, ValueError) as exc:
        print(f"cannot read data dir: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    if problems:
        exit_code = 2

    if args.compare:
        try:
            with open(args.compare, encoding="utf-8") as fh:
                reference = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"cannot read reference manifest: {exc}", file=sys.stderr)
            return 2
        for diff in _diff(reference, manifest):
            problems.append(f"compare: {diff}")
        if problems:
            exit_code = 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
