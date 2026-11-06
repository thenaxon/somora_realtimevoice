"""Restore of an earlier version: the same rule as saving.

`POST …/restore` makes an earlier spec the set again. For a switched-on set
the restored spec must be valid: it is refused (409 `set_invalid`) for
**every blocking `validate_spec` error**, not only F10's
`reference_too_long`; warnings never block; backend reachability is not a
gate (probes run and only report). A switched-off set restores any version.

Synthetic audio only; capturing fake HTTP backends.

Run: pytest -q tests/test_restore_scope.py
"""

from test_profiles_api import AUTH, env  # noqa: F401 (fixture)

from test_voice_reference_length import _spec, _voice_with_sample


def _profile(tc, pid, spec, *, enabled=True):
    r = tc.post("/api/conversation-profiles", headers=AUTH, json={"id": pid, "spec": spec, "enabled": enabled})
    assert r.status_code == 201, r.text


def _save(tc, pid, spec):
    r = tc.put(f"/api/conversation-profiles/{pid}", headers=AUTH, json={"spec": spec})
    assert r.status_code == 200, r.text
    return r.json()


def _native_spec(native="ryan"):
    # native voice against a reference_audio cloning engine: warning-only
    return {
        "stt_profile": "alt",
        "llm_profile": "alt",
        "tts_profile": "alt",
        "language": "de",
        "voice": {"kind": "native", "native_voice": native},
    }


def test_restore_allows_warning_only_target(env):
    """`native_on_cloning_engine` is severity=warning and must never block."""
    tc, _, _ = env
    _profile(tc, "warned", _native_spec("ryan"))
    _save(tc, "warned", _native_spec("clara"))

    r = tc.post("/api/conversation-profiles/warned/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 200, r.text
    assert r.json()["spec"]["voice"]["native_voice"] == "ryan"


def test_restore_allows_unreachable_backend_and_reports_probe(env):
    """The target validates but its backend no longer answers for the model:
    restore still succeeds and the failed probe is reported in the response
    (reachability is not a gate)."""
    tc, _, _ = env
    _voice_with_sample(tc, vid="alex-t")
    _profile(tc, "alex", _spec())
    _save(tc, "alex", {**_spec(), "language": "en"})

    # the backend stops advertising the model (a model edit that keeps the
    # set valid is accepted)
    r = tc.put("/api/profiles/tts/alt", headers=AUTH, json={"model": "tts-gone"})
    assert r.status_code == 200, r.text

    r = tc.post("/api/conversation-profiles/alex/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["spec"]["language"] == "de"
    assert body["probes"]["tts"]["ok"] is False
    assert body["probes"]["tts"]["error_code"] == "model_not_listed"


def test_restore_refuses_non_length_blocking_error(env):
    """An unconfirmed reference transcript is a blocking, non-length error:
    restoring a version that speaks with that voice is refused with 409
    `set_invalid` and the set is unchanged."""
    tc, _, _ = env
    sample = _voice_with_sample(tc, vid="alex-t")
    _profile(tc, "alex", _spec())
    # the set now speaks with a built-in voice; the recording version is history
    before = _save(tc, "alex", _native_spec())

    # nobody switched on speaks with alex-t any more: unconfirming is allowed
    r = tc.patch(
        "/api/voices/alex-t/samples/" + sample["id"],
        headers=AUTH,
        json={"transcript_confirmed": False},
    )
    assert r.status_code == 200, r.text

    r = tc.post("/api/conversation-profiles/alex/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "set_invalid"
    assert any(i["code"] == "transcript_unconfirmed" for i in body["issues"])

    after = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()
    assert after["spec"] == before["spec"]
    assert after["history"] == before["history"]


def test_a_switched_off_set_restores_any_version(env):
    tc, _, _ = env
    sample = _voice_with_sample(tc, vid="alex-t")
    _profile(tc, "alex", _spec(), enabled=False)
    _save(tc, "alex", _native_spec())
    tc.patch("/api/voices/alex-t/samples/" + sample["id"], headers=AUTH, json={"transcript_confirmed": False})

    r = tc.post("/api/conversation-profiles/alex/restore", headers=AUTH, json={"index": 0})
    assert r.status_code == 200, r.text
    assert r.json()["spec"]["voice"]["voice_id"] == "alex-t"
    assert any(i["code"] == "transcript_unconfirmed" for i in r.json()["issues"])
    # switching it on is what needs a valid set
    r = tc.put("/api/conversation-profiles/alex", headers=AUTH, json={"enabled": True})
    assert r.status_code == 409 and r.json()["code"] == "set_invalid"
