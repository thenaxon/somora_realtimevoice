"""A saved set is what the next call uses (server proof).

A user report: changing the voice of a set that was already in use, saving it
and reconnecting still used the OLD voice; only deactivate -> save -> activate
helped. Sets no longer have a draft and an active revision: saving a
switched-on set is live at once, and a call resolves the set against the
CURRENT model entries and the voice's CURRENT reference recording.

This module proves the RESOLUTION semantics, with two clearly distinct
library voices A/B (synthetic WAVs generated here, never user data):

  * set on with voice A -> ``profile:<id>`` resolves fingerprint(A);
  * save voice B (a plain PUT) -> the next ``profile:<id>`` resolves B;
  * a selection resolved BEFORE the save keeps A (a running call keeps its
    copy), also when a model entry changes afterwards;
  * a changed model entry reaches the next call without saving the set;
  * ``restore`` -> the next call resolves A again;
  * a switched-off set is refused for calls but can still be tested.

Assertions are on ``reference_fingerprint`` / voice ids — never on audio
content. No real endpoint is called; the fake HTTP backends capture only.
"""

import base64

import pytest

from realtimevoice.config import ProfileSpec, reference_fingerprint
from realtimevoice.profiles import ProfileResolutionError, resolve_profile, resolve_test

from test_profiles_api import AUTH, env, sine_pcm, wav_bytes  # noqa: F401 (fixture)

LIMIT_MS = 10_000
TTS_ROW = "chatterbox-f12"


def _make_cloning_row(tc):
    r = tc.post(
        "/api/profiles/tts",
        headers=AUTH,
        json={
            "id": TTS_ROW,
            "display_name": "Chatterbox",
            "base_url": "http://cap.test/v1",
            "model": "tts-alt",
            "voice": "clara-voice",
            "capabilities": {
                "voice_cloning": "reference_audio",
                "max_reference_ms": LIMIT_MS,
                "reference_text_required": False,
                "voices": ["default", "clara"],
            },
            "extra": {},
        },
    )
    assert r.status_code == 200, r.text


def _make_voice(tc, vid, ms):
    """A distinct library voice: a different synthetic WAV => different sha256."""
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": vid.upper()}).status_code == 201
    up = tc.post(
        f"/api/voices/{vid}/samples",
        headers=AUTH,
        files={"file": ("ref.wav", wav_bytes(sine_pcm(ms=ms), rate=24000), "audio/wav")},
        data={"source": "mic"},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    patch = {"transcript": f"reference {vid}", "transcript_confirmed": True}
    assert tc.patch(f"/api/voices/{vid}/samples/{sid}", headers=AUTH, json=patch).status_code == 200
    assert tc.put(f"/api/voices/{vid}/reference", headers=AUTH, json={"sample_id": sid}).status_code == 200
    return sid


def _spec(voice_id):
    return {
        "stt_profile": "alt",
        "llm_profile": "alt",
        "tts_profile": TTS_ROW,
        "language": "de",
        "voice": {"kind": "library", "voice_id": voice_id},
    }


def _make_set(tc, pid, voice_id, *, enabled=True):
    r = tc.post("/api/conversation-profiles", headers=AUTH,
                json={"id": pid, "spec": _spec(voice_id), "enabled": enabled})
    assert r.status_code == 201, r.text


def _settings(tc):
    return tc.app.state.settings_store._settings  # noqa: SLF001 (test)


def _resolve(tc, pid):
    """What the next `profile:<id>` call gets: (selection, route)."""
    return resolve_profile(
        _settings(tc),
        pid,
        sample_store=tc.app.state.sample_store,
        b64_cache=tc.app.state.b64_cache,
    )


def _resolve_fp(tc, pid):
    selection, route = _resolve(tc, pid)
    return reference_fingerprint(selection.settings.tts.extra), route


def _resolve_ticket_fp(tc, pid, spec=None):
    selection, route = resolve_test(
        _settings(tc),
        pid,
        sample_store=tc.app.state.sample_store,
        b64_cache=tc.app.state.b64_cache,
        spec_override=None if spec is None else ProfileSpec.model_validate(spec),
    )
    return reference_fingerprint(selection.settings.tts.extra), route


def _sha_of(tc, vid):
    voice = tc.get(f"/api/voices/{vid}", headers=AUTH).json()
    return voice["samples"][0]["sha256"]


def _fingerprint_of(tc, vid):
    """reference_fingerprint() of the materialised reference for a library
    voice: sha256 over the base64 of the stored sample bytes (the convention
    fixed 2026-09-12)."""
    voice = tc.get(f"/api/voices/{vid}", headers=AUTH).json()
    sample = voice["samples"][0]
    audio = tc.get(f"/api/voices/{vid}/samples/{sample['id']}/audio", headers=AUTH).content
    ref_b64 = base64.b64encode(audio).decode()
    return reference_fingerprint({"ref_audio": ref_b64})


def test_saving_a_switched_on_set_is_live_for_the_next_call(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_voice(tc, "voice-b", ms=5_000)
    _make_set(tc, "alex", "voice-a")

    fp_a, route_a = _resolve_fp(tc, "alex")
    assert route_a["voice"]["voice_id"] == "voice-a"
    assert fp_a == _fingerprint_of(tc, "voice-a"), "the set resolves the A reference"
    assert route_a["voice"]["ref_sha256"] == _sha_of(tc, "voice-a")
    assert fp_a != _fingerprint_of(tc, "voice-b")

    # a plain save with B, no other step: the next call speaks with B
    put = tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"spec": _spec("voice-b")})
    assert put.status_code == 200, put.text
    assert put.json()["enabled"] is True
    fp_b, route_b = _resolve_fp(tc, "alex")
    assert route_b["voice"]["voice_id"] == "voice-b"
    assert fp_b == _fingerprint_of(tc, "voice-b")


def test_a_running_call_keeps_what_it_started_with(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_voice(tc, "voice-b", ms=5_000)
    _make_set(tc, "alex", "voice-a")

    # a call "opened" before the changes
    running, _ = _resolve(tc, "alex")

    # save B and change the speaking model's entry
    assert tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"spec": _spec("voice-b")}).status_code == 200
    assert tc.put(f"/api/profiles/tts/{TTS_ROW}", headers=AUTH, json={"model": "tts-alt-v2"}).status_code == 200

    new, _ = _resolve(tc, "alex")
    assert reference_fingerprint(new.settings.tts.extra) == _fingerprint_of(tc, "voice-b")
    assert new.settings.tts.model == "tts-alt-v2"
    # the running call's copy is untouched by either change
    assert reference_fingerprint(running.settings.tts.extra) == _fingerprint_of(tc, "voice-a")
    assert running.settings.tts.model == "tts-alt"


def test_a_changed_model_entry_reaches_the_next_call_without_resaving(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_set(tc, "alex", "voice-a")
    updated_before = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()["updated_at"]

    r = tc.put(f"/api/profiles/tts/{TTS_ROW}", headers=AUTH,
               json={"model": "tts-alt-v2", "extra": {"temperature": 0.3}})
    assert r.status_code == 200, r.text
    r = tc.put("/api/profiles/llm/alt", headers=AUTH, json={"model": "llm-alt-v2"})
    assert r.status_code == 200, r.text

    selection, route = _resolve(tc, "alex")
    assert selection.settings.tts.model == "tts-alt-v2"
    assert selection.settings.tts.extra["temperature"] == 0.3
    assert selection.settings.llm.model == "llm-alt-v2"
    assert route["tts_model"] == "tts-alt-v2" and route["llm_model"] == "llm-alt-v2"
    # the reference still comes from the voice, never from the entry
    assert reference_fingerprint(selection.settings.tts.extra) == _fingerprint_of(tc, "voice-a")
    # the set itself was not touched
    after = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()
    assert after["updated_at"] == updated_before and after["history"] == []


def test_restore_returns_to_the_previous_voice(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_voice(tc, "voice-b", ms=5_000)
    _make_set(tc, "alex", "voice-a")

    assert tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"spec": _spec("voice-b")}).status_code == 200
    fp_b, _ = _resolve_fp(tc, "alex")
    assert fp_b == _fingerprint_of(tc, "voice-b")

    r = tc.post("/api/conversation-profiles/alex/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 200, r.text
    fp_a, route_a = _resolve_fp(tc, "alex")
    assert fp_a == _fingerprint_of(tc, "voice-a")
    assert route_a["voice"]["voice_id"] == "voice-a"
    assert r.json()["history"][0]["spec"]["voice"]["voice_id"] == "voice-b"


def test_a_test_ticket_tries_an_unsaved_spec_while_calls_keep_the_saved_one(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_voice(tc, "voice-b", ms=5_000)
    _make_set(tc, "alex", "voice-a")

    fp_ticket, route = _resolve_ticket_fp(tc, "alex", _spec("voice-b"))
    assert fp_ticket == _fingerprint_of(tc, "voice-b")
    assert route["requested"] == "profile-test"
    fp_saved, _ = _resolve_fp(tc, "alex")
    assert fp_saved == _fingerprint_of(tc, "voice-a")


def test_a_switched_off_set_is_refused_for_calls_but_can_be_tested(env):
    tc, _, _ = env
    _make_cloning_row(tc)
    _make_voice(tc, "voice-a", ms=3_000)
    _make_set(tc, "alex", "voice-a", enabled=False)

    with pytest.raises(ProfileResolutionError) as exc:
        _resolve(tc, "alex")
    assert exc.value.code == "profile_not_available"
    fp, _ = _resolve_ticket_fp(tc, "alex")
    assert fp == _fingerprint_of(tc, "voice-a")
    r = tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH, json={})
    assert r.status_code == 200, r.text
    assert r.json()["model"].startswith("profile-test:")
