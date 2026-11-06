"""F10: set preview — hidden save requirement and reference length.

Alex on /profiles/alex: picked a selection, "hear it" failed until he saved;
after saving he saw `reference_length: sample 12200 ms exceeds max 10000 ms`.

Established (server log + adapter code), not assumed:
  * the test session validated the SAVED draft, never the on-screen selection
    -> hidden save requirement;
  * the length check existed only as a draft-save WARNING; voice preview, test
    sessions and activation never enforced it, so a 12.2 s reference went to
    chatterbox, which keeps only DEC_COND_LEN = 10 s and silently drops the rest.

Covered here:
  * one limit decision, enforced (blocking) for test sessions, switching a
    set on, restoring a version of a switched-on set and voice preview,
    reported (non-blocking) when a switched-off set is saved; a model edit
    that would make an enabled set's reference too long is refused;
  * unknown stored duration (migrated samples carry duration_ms=0) is measured
    from the real WAV instead of silently passing;
  * a test session can test the UNSAVED selection and persists nothing;
  * an excerpt creates a NEW sample, never touches the original, carries an
    empty unconfirmed transcript and provenance, and is not auto-selected.

Synthetic audio only; capturing fake HTTP backends — never a real endpoint.
"""

import io
import json
import wave

from realtimevoice.config import Settings
from realtimevoice.profiles import ProfileSpec, ProfileTestStore, reference_length_issue, sample_duration_ms

from test_profiles_api import AUTH, env, sine_pcm, wav_bytes  # noqa: F401 (fixture)

LIMIT_MS = 10_000


def _upload_ms(tc, vid, ms, rate=24000):
    r = tc.post(
        f"/api/voices/{vid}/samples",
        headers=AUTH,
        files={"file": ("long.wav", wav_bytes(sine_pcm(ms=ms, rate=rate), rate=rate), "audio/wav")},
        data={"source": "mic"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _confirm(tc, vid, sid, text="Hallo Welt"):
    r = tc.patch(
        f"/api/voices/{vid}/samples/{sid}",
        headers=AUTH,
        json={"transcript": text, "transcript_confirmed": True},
    )
    assert r.status_code == 200, r.text


def _voice_with_sample(tc, vid="alex-t", ms=12_200):
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": "R"}).status_code == 201
    sample = _upload_ms(tc, vid, ms)
    _confirm(tc, vid, sample["id"])
    return sample


def _limit_tts(tc, limit=LIMIT_MS):
    """Give the fixture's cloning engine the chatterbox-style limit."""
    r = tc.put("/api/profiles/tts/alt", headers=AUTH, json={"capabilities": {
        "voice_cloning": "reference_audio", "max_reference_ms": limit, "reference_text_required": False,
    }})
    assert r.status_code == 200, r.text


def _spec(vid="alex-t"):
    return {
        "stt_profile": "alt",
        "llm_profile": "alt",
        "tts_profile": "alt",
        "language": "de",
        "voice": {"kind": "library", "voice_id": vid},
    }


# ------------------------------------------------------------- limit decision

def test_long_reference_blocks_test_session_with_numbers(env):
    tc, _, _ = env
    _limit_tts(tc)
    sample = _voice_with_sample(tc, ms=12_200)
    assert tc.post("/api/conversation-profiles", headers=AUTH,
                   json={"id": "alex", "spec": _spec()}).status_code == 201

    r = tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH, json={})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "set_incomplete"
    issue = next(i for i in body["issues"] if i["code"] == "reference_too_long")
    # the UI needs these to offer a fitting excerpt
    assert issue["sample_ms"] == 12_200 and issue["max_ms"] == LIMIT_MS
    assert issue["sample_id"] == sample["id"] and issue["voice_id"] == "alex-t"
    assert issue["tts_profile"] == "alt"


def test_long_reference_blocks_switching_on_and_names_the_reason(env):
    tc, _, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=12_200)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec()})
    r = tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"enabled": True})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "set_invalid"
    # no longer a bare "validation issues": the actual reason and numbers
    assert "reference sample" in body["detail"] and "10.0 s" in body["detail"]
    assert any(i["code"] == "reference_too_long" for i in body["issues"])


def test_long_reference_blocks_voice_preview_before_any_backend_call(env):
    tc, cap, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=12_200)
    calls_before = [c for c in cap.calls if c["path"].endswith("/audio/speech")]
    r = tc.post("/api/voices/alex-t/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "reference_too_long"
    assert r.json()["issues"][0]["sample_ms"] == 12_200
    # the engine was never asked: nothing was silently truncated
    assert [c for c in cap.calls if c["path"].endswith("/audio/speech")] == calls_before


def test_saving_a_switched_off_set_is_allowed_and_reports_length(env):
    tc, _, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=12_200)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": {}})
    r = tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"spec": _spec()})
    assert r.status_code == 200, r.text  # a switched-off set is never refused
    assert any("reference_length" in w for w in r.json()["compatibility_warnings"])
    assert any(i["code"] == "reference_too_long" for i in r.json()["issues"])


def test_fitting_reference_passes_everywhere(env):
    tc, _, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=4_700)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec()})
    assert tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                   json={}).status_code == 200
    r = tc.post("/api/voices/alex-t/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 200, r.text


def test_engine_without_limit_is_never_checked(env):
    tc, _, _ = env
    _limit_tts(tc, limit=0)  # e.g. Qwen3-TTS Base: no truncation in the adapter
    _voice_with_sample(tc, ms=25_000)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec()})
    assert tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                   json={}).status_code == 200


def test_unknown_stored_duration_is_measured_from_the_file(tmp_path):
    """Migrated samples carry duration_ms=0; the check must read the WAV."""
    from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample

    store = SampleStore(str(tmp_path))
    canonical = canonicalise_wav(wav_bytes(sine_pcm(ms=12_200)))
    sample = new_sample(canonical, source="mic")
    store.write("vo", canonical.sha256, canonical.data)
    sample.duration_ms = 0  # as written by the profile migration
    assert sample_duration_ms("vo", sample, store) == 12_200
    assert sample_duration_ms("vo", sample, None) == 0  # no store: honestly unknown


def test_reference_length_issue_uses_measured_duration(tmp_path):
    from realtimevoice.config import BackendProfile, Voice
    from realtimevoice.voices import SampleStore, canonicalise_wav, new_sample

    store = SampleStore(str(tmp_path))
    canonical = canonicalise_wav(wav_bytes(sine_pcm(ms=12_200)))
    sample = new_sample(canonical, source="migration")
    sample.duration_ms = 0
    store.write("vo", canonical.sha256, canonical.data)
    s = Settings()
    s.voices["vo"] = Voice(id="vo", display_name="vo", samples=[sample], reference_sample_id=sample.id,
                           created_at=0, updated_at=0)
    tts = BackendProfile(id="cb", role="tts", base_url="http://x/v1", model="chatterbox-fp16")
    tts.capabilities.voice_cloning = "reference_audio"
    tts.capabilities.max_reference_ms = LIMIT_MS
    s.profiles["tts"]["cb"] = tts
    spec = ProfileSpec.model_validate({"tts_profile": "cb", "voice": {"kind": "library", "voice_id": "vo"}})
    assert reference_length_issue(s, spec) is None  # metadata alone: unknown
    issue = reference_length_issue(s, spec, sample_store=store)
    assert issue is not None and issue["sample_ms"] == 12_200


# -------------------------------------------------- unsaved selection testing

def test_test_session_tests_the_unsaved_selection_and_persists_nothing(env):
    tc, _, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=4_700)
    # the saved set is EMPTY — this is exactly what made Alex's first attempt 409
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": {}})
    saved_before = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()["spec"]

    assert tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                   json={}).status_code == 409  # without a spec: the saved set

    r = tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                json={"spec": _spec()})
    assert r.status_code == 200, r.text
    ticket = r.json()["model"].split(":", 1)[1]
    snap = tc.app.state.profile_tests.spec_for(ticket)
    assert snap is not None and snap.tts_profile == "alt" and snap.voice.voice_id == "alex-t"

    # nothing was saved
    saved_after = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()["spec"]
    assert saved_after == saved_before


def test_unsaved_selection_is_still_length_checked(env):
    tc, _, _ = env
    _limit_tts(tc)
    _voice_with_sample(tc, ms=12_200)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": {}})
    r = tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                json={"spec": _spec()})
    assert r.status_code == 409
    assert any(i["code"] == "reference_too_long" for i in r.json()["issues"])


def test_ticket_spec_is_a_snapshot_and_expires_with_the_ticket():
    store = ProfileTestStore(ttl_s=10)
    spec = ProfileSpec.model_validate({"tts_profile": "a", "voice": {"kind": "library", "voice_id": "v"}})
    t = store.create("p", now=0, spec=spec)
    spec.tts_profile = "changed-later"
    assert store.spec_for(t).tts_profile == "a"  # later edits do not leak in
    assert store.get(t, now=11) is None
    assert store.spec_for(t) is None
    # a ticket without a spec tests the saved set
    t2 = store.create("p", now=0)
    assert store.get(t2, now=1) == ("p", 10) and store.spec_for(t2) is None


# ------------------------------------------------------------------- excerpt

def test_excerpt_creates_new_sample_and_never_touches_the_original(env):
    tc, _, tmp = env
    original = _voice_with_sample(tc, ms=12_200)
    voice_before = tc.get("/api/voices/alex-t", headers=AUTH).json()
    orig_before = next(s for s in voice_before["samples"] if s["id"] == original["id"])
    orig_file = tmp / "voices" / "alex-t" / f"{original['sha256']}.wav"
    orig_bytes = orig_file.read_bytes()

    r = tc.post(f"/api/voices/alex-t/samples/{original['id']}/excerpt", headers=AUTH,
                json={"start_ms": 1_000, "end_ms": 11_000})
    assert r.status_code == 200, r.text
    ex = r.json()
    assert ex["id"] != original["id"] and ex["sha256"] != original["sha256"]
    assert abs(ex["duration_ms"] - 10_000) <= 1
    assert ex["transcript"] == "" and ex["transcript_confirmed"] is False  # must be re-transcribed
    assert ex["derived_from"] == {
        "sample_id": original["id"], "sha256": original["sha256"], "start_ms": 1_000, "end_ms": 11_000,
    }
    assert ex["source"] == original["source"]  # no new literal: rollback-safe

    voice_after = tc.get("/api/voices/alex-t", headers=AUTH).json()
    orig_after = next(s for s in voice_after["samples"] if s["id"] == original["id"])
    assert orig_after == orig_before, "original sample metadata must be unchanged"
    assert orig_file.read_bytes() == orig_bytes, "original audio file must be byte-identical"
    assert voice_after["reference_sample_id"] == original["id"], "excerpt is never auto-selected"
    assert len(voice_after["samples"]) == 2

    # the excerpt audio is the requested slice of the original
    with wave.open(io.BytesIO(orig_bytes)) as w:
        rate = w.getframerate()
        w.setpos(int(1.0 * rate))
        expected = w.readframes(int(10.0 * rate))
    ex_bytes = (tmp / "voices" / "alex-t" / f"{ex['sha256']}.wav").read_bytes()
    with wave.open(io.BytesIO(ex_bytes)) as w:
        got = w.readframes(w.getnframes())
    assert got == expected


def test_excerpt_then_confirm_then_use_as_reference_fits(env):
    tc, _, _ = env
    _limit_tts(tc)
    original = _voice_with_sample(tc, ms=12_200)
    ex = tc.post(f"/api/voices/alex-t/samples/{original['id']}/excerpt", headers=AUTH,
                 json={"start_ms": 0, "end_ms": 9_500}).json()
    _confirm(tc, "alex-t", ex["id"], text="Ausschnitt")
    assert tc.put("/api/voices/alex-t/reference", headers=AUTH,
                  json={"sample_id": ex["id"]}).status_code == 200
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec()})
    assert tc.post("/api/conversation-profiles/alex/test-session", headers=AUTH,
                   json={}).status_code == 200
    r = tc.post("/api/voices/alex-t/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 200, r.text


def test_excerpt_rejects_bad_ranges(env):
    tc, _, _ = env
    original = _voice_with_sample(tc, ms=12_200)
    url = f"/api/voices/alex-t/samples/{original['id']}/excerpt"
    for body, code in (
        ({"start_ms": 5_000, "end_ms": 5_000}, "invalid_range"),
        ({"start_ms": 6_000, "end_ms": 5_000}, "invalid_range"),
        ({"start_ms": -1, "end_ms": 5_000}, "invalid_range"),
        ({"start_ms": 0, "end_ms": 99_000}, "invalid_range"),
        ({"start_ms": 0, "end_ms": 500}, "excerpt_too_short"),
        ({"start_ms": "x", "end_ms": 5_000}, "invalid_range"),
    ):
        r = tc.post(url, headers=AUTH, json=body)
        assert r.status_code == 400, (body, r.text)
        assert r.json()["code"] == code, (body, r.json())
    assert len(tc.get("/api/voices/alex-t", headers=AUTH).json()["samples"]) == 1


def test_unknown_duration_sample_is_measured_and_refused_by_preview(env):
    """F10 gap B5: a migrated sample carries duration_ms=0. The server must
    measure the real WAV, refuse the too-long reference with numbers, and never
    call the engine; the client-side helper must not claim it fits."""
    tc, cap, _ = env
    _limit_tts(tc)
    sample = _voice_with_sample(tc, ms=12_200)

    # simulate the migration: the stored duration metadata is unknown (0)
    async def _apply() -> None:
        store = tc.app.state.settings_store
        async with store._lock:
            store._settings.voices["alex-t"].samples[0].duration_ms = 0

    tc.portal.call(_apply)

    before = [c for c in cap.calls if c["path"].endswith("/audio/speech")]
    r = tc.post("/api/voices/alex-t/preview", headers=AUTH, json={"tts_profile": "alt", "text": "Hallo"})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "reference_too_long"
    assert body["issues"][0]["sample_ms"] == 12_200 and body["issues"][0]["max_ms"] == LIMIT_MS
    # the engine was never asked: no silent truncation for unknown durations
    assert [c for c in cap.calls if c["path"].endswith("/audio/speech")] == before


def test_restoring_a_too_long_reference_is_blocked_and_explained(env):
    """F10 gap: restoring an earlier version whose reference exceeds the
    (now correct) engine limit must not silently put a truncated reference
    into calls. For a switched-on set it is refused with the same numbers +
    a way to a fitting excerpt, exactly like switching on."""
    tc, _, _ = env
    # a 12.2 s reference with no advertised limit: the set is switched on
    _limit_tts(tc, limit=0)
    sample = _voice_with_sample(tc, ms=12_200)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec(), "enabled": True})
    # it now speaks with a built-in voice; the recording version is history
    native = {**_spec(), "voice": {"kind": "native", "native_voice": "ryan"}}
    assert tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"spec": native}).status_code == 200

    # operator corrects the engine limit (chatterbox keeps 10 s): restoring
    # would put the 12.2 s reference back into calls and silently truncate.
    _limit_tts(tc, limit=LIMIT_MS)

    r = tc.post("/api/conversation-profiles/alex/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "set_invalid"
    issue = next(i for i in body["issues"] if i["code"] == "reference_too_long")
    assert issue["sample_ms"] == 12_200 and issue["max_ms"] == LIMIT_MS
    assert issue["voice_id"] == "alex-t" and issue["sample_id"] == sample["id"]
    # the set is untouched
    after = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()
    assert after["spec"]["voice"]["kind"] == "native" and len(after["history"]) == 1


def test_a_model_limit_that_would_cut_an_enabled_sets_reference_is_refused(env):
    """Lowering the speaking model's limit below the reference of a switched-on
    set would silently truncate its next call: refused with 409 breaks_sets,
    naming the set and the numbers."""
    tc, _, _ = env
    _limit_tts(tc, limit=0)
    _voice_with_sample(tc, ms=12_200)
    r = tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec(), "enabled": True})
    assert r.status_code == 201, r.text
    r = tc.put("/api/profiles/tts/alt", headers=AUTH, json={"capabilities": {
        "voice_cloning": "reference_audio", "max_reference_ms": LIMIT_MS, "reference_text_required": False,
    }})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "breaks_sets"
    assert body["issues"][0]["profile"] == "alex"
    issue = body["issues"][0]["issues"][0]
    assert issue["code"] == "reference_too_long" and issue["sample_ms"] == 12_200
    stored = tc.app.state.settings_store._settings.profiles["tts"]["alt"]  # noqa: SLF001 (test)
    assert stored.capabilities.max_reference_ms == 0, "the refused edit changed nothing"
    # a switched-off set does not hold the model back
    assert tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"enabled": False}).status_code == 200
    _limit_tts(tc, limit=LIMIT_MS)


def test_voice_sample_ignores_unknown_fields_so_rollback_stays_loadable():
    """Rollback safety rests on one property: VoiceSample must NOT forbid extra
    fields. Then an older build (without derived_from) simply ignores the new
    key instead of refusing settings.json. The real old-code check is done
    separately against the previous commit (see F10 evidence)."""
    from realtimevoice.config import VoiceSample

    assert VoiceSample.model_config.get("extra", "ignore") != "forbid"
    loaded = VoiceSample.model_validate({
        "id": "smp_x", "sha256": "0" * 64, "source": "mic", "duration_ms": 1000,
        "sample_rate": 24000, "size_bytes": 10, "created_at": 0,
        "derived_from": {"sample_id": "smp_y", "start_ms": 0, "end_ms": 1000},
        "some_future_field": 1,
    })
    assert loaded.source == "mic" and loaded.derived_from["sample_id"] == "smp_y"
