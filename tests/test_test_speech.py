"""R2 / CP2: the audible "test this sentence" path.

A user report: typing a test sentence in block 03 and pressing Run test only
offered a link to the Test Console. It must play the typed sentence directly,
with the current (possibly unsaved) TTS + voice selection, using EXACTLY the
conversation materialisation — no LLM may reword it, nothing is saved.

Facts this pins (all verified against the code before the fix):
  * the old voice-area preview (`/api/voices/{id}/preview`) synthesises a
    LIBRARY voice's sample; it can never reproduce "Engine default" on a
    Chatterbox row that carries an embedded `extra.ref_audio` — reusing
    `/api/tts-synthesis` for that spec would play Clara's stored reference
    while the real conversation plays the engine default. The audible sample
    would lie.

Covered here:
  * the exact typed text reaches the fake TTS byte-for-byte (after the same
    symbol filter the conversation applies, and nothing else);
  * no LLM and no STT request is made;
  * parity: library / native / none on a Chatterbox row with an embedded
    reference produce the SAME TTS request (reference material + voice) as a
    conversation resolved through `resolve_test` with the same spec;
  * an unsaved spec is used and the saved set is unchanged afterwards;
  * `reference_too_long` is refused before any TTS call;
  * the endpoint only ever calls TTS.

Synthetic audio only; capturing fake HTTP backends — never a real endpoint.
"""

import base64
import json

from realtimevoice.backends.openai_http import _tts_payload
from realtimevoice.config import ProfileSpec
from realtimevoice.profiles import resolve_test

from test_profiles_api import AUTH, env, sine_pcm, wav_bytes  # noqa: F401 (fixture)

LIMIT_MS = 10_000
SENTENCE = "Guten Morgen Alex, es ist Viertel nach acht."


def _setup_voice(tc, vid="alex-t", ms=4_700, transcript="Hallo Welt", confirmed=True):
    assert tc.post("/api/voices", headers=AUTH, json={"id": vid, "display_name": "R"}).status_code == 201
    up = tc.post(
        f"/api/voices/{vid}/samples",
        headers=AUTH,
        files={"file": ("ref.wav", wav_bytes(sine_pcm(ms=ms), rate=24000), "audio/wav")},
        data={"source": "mic"},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    patch = {"transcript": transcript, "transcript_confirmed": confirmed}
    assert tc.patch(f"/api/voices/{vid}/samples/{sid}", headers=AUTH, json=patch).status_code == 200
    return sid


def _make_chatterbox_row(tc, rid="chatterbox-clara", embedded=True):
    """A Chatterbox row exactly like the field one: it carries an embedded
    legacy reference (`extra.ref_audio`/`ref_text`) that the conversation
    strips and replaces with the library sample (or drops for native/none)."""
    extra = {"ref_audio": "CLARA-EMBEDDED-REF", "ref_text": "clara text"} if embedded else {}
    r = tc.post(
        "/api/profiles/tts",
        headers=AUTH,
        json={
            "id": rid,
            "display_name": "Chatterbox (Clara)",
            "base_url": "http://cap.test/v1",
            "model": "chatterbox-clara",
            "voice": "clara-voice",
            "capabilities": {
                "voice_cloning": "reference_audio",
                "max_reference_ms": LIMIT_MS,
                "reference_text_required": False,
                "voices": ["default", "clara"],
            },
            "extra": extra,
        },
    )
    assert r.status_code == 200, r.text
    return rid


def _spec(rid, kind="library", vid="alex-t", native=""):
    return {
        "stt_profile": "alt",
        "llm_profile": "alt",
        "tts_profile": rid,
        "language": "de",
        "voice": {"kind": kind, "voice_id": vid if kind == "library" else "", "native_voice": native},
    }


def _speech_calls(cap, since=0):
    return [c for c in cap.calls[since:] if c["path"].endswith("/audio/speech")]


def _last_speech_body(cap, since=0):
    calls = _speech_calls(cap, since)
    assert calls, "no /audio/speech request was made"
    return json.loads(calls[-1]["content"].decode("utf-8"))


def _conversation_payload(tc, spec, text):
    """What a conversation with the same spec would send, via the REAL
    conversation resolution (resolve_test + the adapter's own payload builder),
    with the same unsaved spec the endpoint is given."""
    settings = tc.app.state.settings_store._settings  # noqa: SLF001 (test)
    selection, _route = resolve_test(
        settings,
        "alex",
        sample_store=tc.app.state.sample_store,
        b64_cache=tc.app.state.b64_cache,
        spec_override=ProfileSpec.model_validate(spec),
    )
    voice = selection.settings.session.voice or None
    language = selection.settings.session.language or None
    return _tts_payload(selection.settings.tts, text, voice, language)


# ------------------------------------------------------- text + isolation

def test_test_speech_sends_the_typed_text_verbatim(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec("chatterbox-clara")})

    before = len(cap.calls)
    r = tc.post(
        "/api/conversation-profiles/alex/test-speech",
        headers=AUTH,
        json={"text": SENTENCE},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    body = _last_speech_body(cap, before)
    assert body["input"] == SENTENCE, body.get("input")

    # no LLM, no STT was involved
    paths = [c["path"] for c in cap.calls[before:]]
    assert not any(p.endswith("/chat/completions") for p in paths), paths
    assert not any(p.endswith("/audio/transcriptions") for p in paths), paths


def test_test_speech_applies_the_same_symbol_filter_as_the_conversation(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec("chatterbox-clara")})

    before = len(cap.calls)
    r = tc.post(
        "/api/conversation-profiles/alex/test-speech",
        headers=AUTH,
        json={"text": "Hallo. 😊"},
    )
    assert r.status_code == 200, r.text
    body = _last_speech_body(cap, before)
    # the emoji is stripped (spoken text), but the words survive verbatim
    assert body["input"] == "Hallo.", body.get("input")


def test_test_speech_rejects_empty_text(env):
    tc, _, _ = env
    _make_chatterbox_row(tc, embedded=False)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec("chatterbox-clara")})
    r = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": "   "})
    assert r.status_code == 422, r.text


def test_test_speech_refuses_a_symbol_only_sentence_without_calling_tts(env):
    # the conversation would skip such a segment entirely; the sample must be
    # honest about that instead of synthesising noise
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": _spec("chatterbox-clara")})
    before = len(cap.calls)
    r = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": "😊"})
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "nothing_to_speak"
    assert _speech_calls(cap, before) == []


# ---------------------------------------------------------------- parity

def test_parity_library_native_none_on_chatterbox_with_embedded_reference(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=True)
    _setup_voice(tc, ms=4_700, transcript="hallo welt")

    # library -----------------------------------------------------------------
    lib = _spec("chatterbox-clara", "library", vid="alex-t")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": lib})
    before = len(cap.calls)
    assert tc.post(
        "/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": lib}
    ).status_code == 200, "library test-speech failed"
    got = _last_speech_body(cap, before)
    want = _conversation_payload(tc, lib, SENTENCE)
    assert got == want, {"got": got, "want": want}
    # the embedded Clara reference never reaches the engine
    assert "CLARA-EMBEDDED-REF" not in json.dumps(got), "embedded reference leaked"
    assert got["ref_text"] == "hallo welt"

    # native ------------------------------------------------------------------
    native = _spec("chatterbox-clara", "native", native="clara")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": native})
    before = len(cap.calls)
    assert tc.post(
        "/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": native}
    ).status_code == 200, "native test-speech failed"
    got = _last_speech_body(cap, before)
    want = _conversation_payload(tc, native, SENTENCE)
    assert got == want, {"got": got, "want": want}
    assert "ref_audio" not in got and "ref_text" not in got
    assert "CLARA-EMBEDDED-REF" not in json.dumps(got)

    # none (engine default) ---------------------------------------------------
    none = _spec("chatterbox-clara", "none")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": none})
    before = len(cap.calls)
    assert tc.post(
        "/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": none}
    ).status_code == 200, "none test-speech failed"
    got = _last_speech_body(cap, before)
    want = _conversation_payload(tc, none, SENTENCE)
    assert got == want, {"got": got, "want": want}
    assert "ref_audio" not in got and "ref_text" not in got
    assert "CLARA-EMBEDDED-REF" not in json.dumps(got), "Engine default played Clara's embedded reference"
    # exactly what a conversation with kind=none sends (the session default
    # voice, never the library label and never the row's embedded reference)
    assert got["voice"] == want["voice"]


def test_parity_reference_sha_header_is_the_materialised_sample(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=True)
    sample_id = _setup_voice(tc, ms=4_700, transcript="hallo welt")
    spec = _spec("chatterbox-clara", "library", vid="alex-t")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": spec})
    voice = tc.get("/api/voices/alex-t", headers=AUTH).json()
    sample = next(s for s in voice["samples"] if s["id"] == sample_id)
    r = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": spec})
    assert r.status_code == 200, r.text
    assert "CLARA-EMBEDDED-REF" not in r.headers.get("X-Rtv-Reference-Sha8", "")
    assert r.headers["X-Rtv-Reference-Sha8"] == sample["sha256"][:8]


# ------------------------------------------------------------- unsaved spec

def test_unsaved_spec_is_used_and_the_saved_set_is_unchanged(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": {}})
    saved_before = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()["spec"]

    # the saved set is empty -> without a spec the endpoint must refuse it
    refused = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE})
    assert refused.status_code == 409 and refused.json()["code"] == "set_incomplete"

    before = len(cap.calls)
    r = tc.post(
        "/api/conversation-profiles/alex/test-speech",
        headers=AUTH,
        json={"text": SENTENCE, "spec": _spec("chatterbox-clara")},
    )
    assert r.status_code == 200, r.text
    assert _last_speech_body(cap, before)["input"] == SENTENCE
    saved_after = tc.get("/api/conversation-profiles/alex", headers=AUTH).json()["spec"]
    assert saved_after == saved_before


def test_the_saved_set_is_spoken_without_a_spec_even_when_switched_off(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    spec = _spec("chatterbox-clara")
    r = tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": spec})
    assert r.status_code == 201 and r.json()["enabled"] is False
    before = len(cap.calls)
    r = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE})
    assert r.status_code == 200, r.text
    assert _last_speech_body(cap, before) == _conversation_payload(tc, spec, SENTENCE)


# ----------------------------------------------------- reference too long

def test_reference_too_long_is_refused_before_any_tts_call(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=12_200)  # > LIMIT_MS
    spec = _spec("chatterbox-clara", "library", vid="alex-t")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": spec})
    before = len(cap.calls)
    r = tc.post(
        "/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": spec}
    )
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["code"] == "set_incomplete"
    issue = next(i for i in body["issues"] if i["code"] == "reference_too_long")
    assert issue["sample_ms"] == 12_200 and issue["max_ms"] == LIMIT_MS
    assert _speech_calls(cap, before) == [], "TTS was called despite a blocking length issue"


# --------------------------------------------------------------- provenance

def test_provenance_headers_are_safe_and_kind_is_reported(env):
    tc, cap, _ = env
    _make_chatterbox_row(tc, embedded=False)
    _setup_voice(tc, ms=4_700)
    spec = _spec("chatterbox-clara", "library", vid="alex-t")
    tc.post("/api/conversation-profiles", headers=AUTH, json={"id": "alex", "spec": spec})
    r = tc.post("/api/conversation-profiles/alex/test-speech", headers=AUTH, json={"text": SENTENCE, "spec": spec})
    assert r.status_code == 200, r.text
    assert r.headers["X-Rtv-Model"] == "chatterbox-clara"
    assert r.headers["X-Rtv-Voice-Kind"] == "library"
    assert r.headers["X-Rtv-Sample-Sha8"]
    # never a host or a secret
    flat = json.dumps(dict(r.headers))
    assert "cap.test" not in flat and "CLARA-EMBEDDED-REF" not in flat