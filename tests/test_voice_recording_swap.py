"""Replacing a voice's recording reaches the sets that speak with it.

A call resolves the voice's CURRENT reference recording, so choosing a new
reference on the voice page (or correcting its transcript) reaches every set
that speaks with the voice from its next call, without saving the set again.
A running call keeps the recording it started with.

What a switched-on set could not use is refused instead of silently breaking
it (409 `breaks_sets`, naming the set and the issue): an unconfirmed
recording, one too long for the set's speaking model, a transcript edit that
is not confirmed, a re-transcription of the reference.
"""

import base64

from realtimevoice.config import reference_fingerprint

from test_set_save_is_live import (
    LIMIT_MS,
    TTS_ROW,
    _make_cloning_row,
    _make_set,
    _make_voice,
    _resolve,
    _resolve_fp,
)
from test_profiles_api import AUTH, env, sine_pcm, wav_bytes  # noqa: F401 (fixture)


def _upload(tc, vid, ms, *, confirm=True):
    up = tc.post(
        f"/api/voices/{vid}/samples",
        headers=AUTH,
        files={"file": ("new.wav", wav_bytes(sine_pcm(ms=ms), rate=24000), "audio/wav")},
        data={"source": "upload"},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    if confirm:
        patch = {"transcript": "the new recording", "transcript_confirmed": True}
        assert tc.patch(f"/api/voices/{vid}/samples/{sid}", headers=AUTH, json=patch).status_code == 200
    return sid


def _fp_of_sample(tc, vid, sid):
    audio = tc.get(f"/api/voices/{vid}/samples/{sid}/audio", headers=AUTH).content
    return reference_fingerprint({"ref_audio": base64.b64encode(audio).decode()})


def _live_set(tc, *, enabled=True):
    _make_cloning_row(tc)
    old = _make_voice(tc, "anna", ms=3_000)
    _make_set(tc, "anna-set", "anna", enabled=enabled)
    return old


def _reference(tc):
    return tc.get("/api/voices/anna", headers=AUTH).json()["reference_sample_id"]


def test_a_new_reference_reaches_the_sets_next_call(env):
    tc, _, _ = env
    old = _live_set(tc)
    running, _ = _resolve(tc, "anna-set")
    new = _upload(tc, "anna", 4_000)
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["reference_sample_id"] == new
    assert view["used_by"] == [{"profile": "anna-set", "display_name": "anna-set", "enabled": True}]
    for gone in ("live_sets", "applied_to", "not_applied"):
        assert gone not in view

    fp, route = _resolve_fp(tc, "anna-set")
    assert fp == _fp_of_sample(tc, "anna", new)
    assert route["voice"]["sample_id"] == new
    selection, _ = _resolve(tc, "anna-set")
    assert selection.settings.tts.extra["ref_text"] == "the new recording"
    # the set itself was not saved again
    profile = tc.get("/api/conversation-profiles/anna-set", headers=AUTH).json()
    assert profile["history"] == []
    # a call that was already running keeps the old recording
    assert reference_fingerprint(running.settings.tts.extra) == _fp_of_sample(tc, "anna", old)


def test_an_unconfirmed_recording_is_refused_for_a_switched_on_set(env):
    tc, _, _ = env
    old = _live_set(tc)
    new = _upload(tc, "anna", 4_000, confirm=False)
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "breaks_sets"
    assert body["issues"][0]["profile"] == "anna-set"
    assert [i["code"] for i in body["issues"][0]["issues"]] == ["transcript_missing"]
    assert _reference(tc) == old
    fp, _ = _resolve_fp(tc, "anna-set")
    assert fp == _fp_of_sample(tc, "anna", old)


def test_a_recording_too_long_for_the_engine_is_refused_for_a_switched_on_set(env):
    tc, _, _ = env
    old = _live_set(tc)
    new = _upload(tc, "anna", LIMIT_MS + 2_000)
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new})
    assert r.status_code == 409, r.text
    issue = r.json()["issues"][0]["issues"][0]
    assert issue["code"] == "reference_too_long" and issue["max_ms"] == LIMIT_MS
    assert _reference(tc) == old


def test_a_switched_off_set_does_not_hold_the_voice_back(env):
    tc, _, _ = env
    _live_set(tc, enabled=False)
    new = _upload(tc, "anna", 4_000, confirm=False)
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new})
    assert r.status_code == 200, r.text
    assert r.json()["used_by"][0]["enabled"] is False
    # the set now reports what keeps it from being switched on
    issues = tc.get("/api/conversation-profiles/anna-set", headers=AUTH).json()["issues"]
    assert [i["code"] for i in issues] == ["transcript_missing"]


def test_the_limit_is_the_speaking_models_current_one(env):
    tc, _, _ = env
    _live_set(tc)
    # the speaking model now allows longer recordings: the set uses the
    # entry as it is now, so the longer recording fits
    r = tc.put(f"/api/profiles/tts/{TTS_ROW}", headers=AUTH, json={"capabilities": {
        "voice_cloning": "reference_audio", "max_reference_ms": 60_000, "reference_text_required": False,
    }})
    assert r.status_code == 200, r.text
    new = _upload(tc, "anna", LIMIT_MS + 2_000)
    r = tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new})
    assert r.status_code == 200, r.text
    fp, _ = _resolve_fp(tc, "anna-set")
    assert fp == _fp_of_sample(tc, "anna", new)


def test_a_confirmed_transcript_edit_reaches_the_next_call(env):
    tc, _, _ = env
    old = _live_set(tc)
    patch = {"transcript": "corrected text", "transcript_confirmed": True}
    r = tc.patch(f"/api/voices/anna/samples/{old}", headers=AUTH, json=patch)
    assert r.status_code == 200, r.text
    selection, _ = _resolve(tc, "anna-set")
    assert selection.settings.tts.extra["ref_text"] == "corrected text"


def test_an_unconfirmed_transcript_edit_of_the_reference_is_refused(env):
    tc, _, _ = env
    old = _live_set(tc)
    r = tc.patch(f"/api/voices/anna/samples/{old}", headers=AUTH, json={"transcript": "half done"})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "breaks_sets"
    assert r.json()["issues"][0]["issues"][0]["code"] == "transcript_unconfirmed"
    r = tc.patch(f"/api/voices/anna/samples/{old}", headers=AUTH, json={"transcript_confirmed": False})
    assert r.status_code == 409 and r.json()["code"] == "breaks_sets"
    sample = tc.get("/api/voices/anna", headers=AUTH).json()["samples"][0]
    assert sample["transcript"] == "reference anna" and sample["transcript_confirmed"] is True
    # another recording of the voice can be edited freely
    other = _upload(tc, "anna", 4_000, confirm=False)
    r = tc.patch(f"/api/voices/anna/samples/{other}", headers=AUTH, json={"transcript": "draft"})
    assert r.status_code == 200, r.text


def test_retranscribing_the_reference_of_a_switched_on_set_is_refused(env):
    """A new transcription is unconfirmed: on the reference of a switched-on
    set it would stop the set, so it is refused and nothing changes."""
    tc, _, _ = env
    old = _live_set(tc)
    r = tc.post(f"/api/voices/anna/samples/{old}/transcribe", headers=AUTH, json={})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "breaks_sets"
    sample = tc.get("/api/voices/anna", headers=AUTH).json()["samples"][0]
    assert sample["transcript"] == "reference anna" and sample["transcript_confirmed"] is True


def test_the_old_recording_can_be_removed_after_the_swap(env):
    tc, _, _ = env
    old = _live_set(tc)
    new = _upload(tc, "anna", 4_000)
    assert tc.put("/api/voices/anna/reference", headers=AUTH, json={"sample_id": new}).status_code == 200
    r = tc.delete(f"/api/voices/anna/samples/{old}", headers=AUTH)
    assert r.status_code == 200, r.text
    assert [s["id"] for s in tc.get("/api/voices/anna", headers=AUTH).json()["samples"]] == [new]
    fp, _ = _resolve_fp(tc, "anna-set")
    assert fp == _fp_of_sample(tc, "anna", new)


def test_the_recording_a_set_speaks_with_cannot_be_removed(env):
    tc, _, _ = env
    old = _live_set(tc, enabled=False)
    # also for a switched-off set: it would lose its voice silently
    r = tc.delete(f"/api/voices/anna/samples/{old}", headers=AUTH)
    assert r.status_code == 409
    assert r.json()["code"] == "sample_is_reference"
