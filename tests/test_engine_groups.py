"""Round 2 / CP1 (Option A): engine grouping — pure unit tests.

The brief's catalog shape (9 TTS rows -> 5 engines, 4 STT -> 3, 3 LLM -> 2) is
reconstructed with synthetic fields. These tests would fail before CP1 because
`engine_key`/`group_engines` did not exist and the presentation iterated stored
rows.

No services, no real endpoints, synthetic audio only.
"""

import base64

from realtimevoice.config import BackendProfile, ProfileCapabilities
from realtimevoice.engine_groups import (
    EngineGroup,
    engine_id,
    engine_key,
    group_engines,
    group_engine_view,
    native_presets_trustworthy,
    primary_row,
)

# --------------------------------------------------------------- fixtures

OLLAMA = "http://mac.local:8080/v1"


def _ref(tag: str) -> str:
    return base64.b64encode(f"RIFF-{tag}".encode()).decode()


def _tts(pid: str, model: str, *, voice="", ref="", ref_text="", caps=None, base_url=OLLAMA, extra=None
         ) -> BackendProfile:
    raw_extra = {"temperature": 0.8}
    if ref:
        raw_extra["ref_audio"] = ref
        raw_extra["ref_text"] = ref_text
    if extra:
        raw_extra.update(extra)
    p = BackendProfile(
        id=pid, role="tts", display_name=pid, base_url=base_url, model=model,
        voice=voice, extra=raw_extra,
        capabilities=caps or {},
    )
    return p


def brief_catalog() -> dict[str, list[BackendProfile]]:
    """The exact field-diff shape from the brief (synthetic values)."""
    stt = [
        BackendProfile(id="stt-qwen3-asr-0.6b", role="stt", base_url=OLLAMA, model="Qwen3-ASR-0.6B-8bit"),
        BackendProfile(id="stt-qwen3-asr-0.6b-8bit-snapshot", role="stt", base_url=OLLAMA, model="Qwen3-ASR-0.6B-8bit"),
        BackendProfile(id="stt-qwen3-asr-1.7b", role="stt", base_url=OLLAMA, model="Qwen3-ASR-1.7B-8bit"),
        BackendProfile(id="whisper-existing", role="stt", base_url=OLLAMA, model="whisper-large-v3-turbo"),
    ]
    llm = [
        BackendProfile(id="llm-gemma", role="llm", base_url=OLLAMA, model="gemma-4-26b-a4b-it-4bit"),
        BackendProfile(id="llm-gemma-4-26b-a4b-it-4bit-snapshot", role="llm", base_url=OLLAMA, model="gemma-4-26b-a4b-it-4bit"),
        BackendProfile(id="qwen14b-existing", role="llm", base_url=OLLAMA, model="qwen3-14b-4bit"),
    ]
    cb = {"voice_cloning": "reference_audio", "max_reference_ms": 10000}
    base_caps = {"voice_cloning": "reference_audio", "max_reference_ms": 0, "reference_text_required": True}
    cv_caps = {"voice_cloning": "none", "voices": ["ryan", "serena"]}
    vox_caps = {"voice_cloning": "none", "voices": ["de_male", "de_female", "casual_male"]}
    fish_caps = {"voice_cloning": "none", "voices": ["native"]}
    tts = [
        _tts("chatterbox-clara", "chatterbox-fp16", voice="clara-reference", ref=_ref("k"), ref_text="clara", caps=cb),
        _tts("chatterbox-jonas", "chatterbox-fp16", voice="jonas-reference", ref=_ref("t"), ref_text="jonas", caps=cb),
        _tts("tts-qwen3-base-clone", "Qwen3-TTS-12Hz-1.7B-Base-8bit", caps=base_caps),
        _tts("qwen-ryan", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", voice="ryan", caps={**cv_caps, "evidence": "probe"}),
        _tts("qwen-serena", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", voice="serena", caps={**cv_caps, "evidence": "probe"}),
        _tts("qwen-default", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", caps={**cv_caps, "evidence": "unverified"}),
        _tts("voxtral-casual", "mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", voice="casual_male", caps=vox_caps),
        _tts("voxtral-de-female", "mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", voice="de_female", caps=vox_caps),
        _tts("fish-native", "fish-audio-s2-pro-8bit", voice="native", caps=fish_caps),
    ]
    return {"stt": stt, "llm": llm, "tts": tts}


# ------------------------------------------------------------ engine_key

def test_engine_key_groups_identical_rows():
    cat = brief_catalog()
    rows = {p.id: p for p in cat["stt"]}
    assert engine_key(rows["stt-qwen3-asr-0.6b"]) == engine_key(rows["stt-qwen3-asr-0.6b-8bit-snapshot"])
    tts = {p.id: p for p in cat["tts"]}
    assert engine_key(tts["qwen-ryan"]) == engine_key(tts["qwen-serena"]) == engine_key(tts["qwen-default"])
    assert engine_key(tts["chatterbox-clara"]) == engine_key(tts["chatterbox-jonas"])
    # a real variant never merges
    assert engine_key(tts["tts-qwen3-base-clone"]) != engine_key(tts["qwen-ryan"])


def test_engine_key_excludes_secret_voice_capability_but_includes_endpoint_model_extra():
    base = _tts("a", "m", voice="v", caps={"voices": ["x"]}, extra={"temperature": 0.1})
    base.api_key = "k1"
    other = _tts("b", "m", voice="w", caps={"voices": ["y"]}, extra={"temperature": 0.1})
    other.api_key = "k2"
    assert engine_key(base) == engine_key(other), "api_key/voice/capabilities must not split the engine"

    ref_a = _tts("a", "m", ref=_ref("a"), caps={"voice_cloning": "reference_audio"})
    ref_b = _tts("b", "m", ref=_ref("b"), caps={"voice_cloning": "reference_audio"})
    assert engine_key(ref_a) == engine_key(ref_b), "embedded legacy reference must not split the engine"

    changed_endpoint = _tts("c", "m", base_url="http://other.local:9000/v1")
    changed_model = _tts("d", "m2")
    changed_extra = _tts("e", "m", extra={"temperature": 0.9})
    changed_language_format = _tts("f", "m", voice="v", caps={"voices": ["x"]}, extra={"temperature": 0.1})
    changed_language_format.language_format = "name"
    assert engine_key(base) != engine_key(changed_endpoint)
    assert engine_key(base) != engine_key(changed_model)
    assert engine_key(base) != engine_key(changed_extra)
    # the way the model is told the language is part of the engine
    assert engine_key(base) != engine_key(changed_language_format)


# ---------------------------------------------------------- group counts

def test_group_counts_match_the_brief():
    cat = brief_catalog()
    assert len(group_engines(cat["stt"])) == 3, "4 STT rows -> 3 engines"
    assert len(group_engines(cat["llm"])) == 2, "3 LLM rows -> 2 engines"
    tts_groups = group_engines(cat["tts"])
    assert len(tts_groups) == 5, "9 TTS rows -> 5 engines"
    cb = next(g for g in tts_groups if g.model == "chatterbox-fp16")
    assert set(cb.ids) == {"chatterbox-clara", "chatterbox-jonas"}


# --------------------------------------------------------- primary row

def test_primary_row_prefers_referenced_then_smallest_id():
    rows = [
        BackendProfile(id="b-engine", role="stt", base_url=OLLAMA, model="m"),
        BackendProfile(id="c-engine", role="stt", base_url=OLLAMA, model="m"),
        BackendProfile(id="a-engine", role="stt", base_url=OLLAMA, model="m"),
    ]
    group = EngineGroup(engine_key(rows[0]), rows)
    # (a) a referenced id wins
    assert primary_row(group, referenced="c-engine").id == "c-engine"
    # (b) otherwise the smallest id
    assert primary_row(group).id == "a-engine"


def test_catalog_primary_row_never_borrows_sibling_metadata():
    """C4: the group's metadata comes from the primary (used) row."""
    plain = BackendProfile(id="a-row", role="tts", base_url=OLLAMA, model="m",
                           capabilities={"evidence": "unverified", "voices": ["a"]})
    probed = BackendProfile(id="b-row", role="tts", base_url=OLLAMA, model="m",
                            capabilities={"evidence": "probe", "voices": ["a"]})
    group = EngineGroup(engine_key(plain), [probed, plain])
    view = group_engine_view(group, role="tts")
    # primary = smallest id, so evidence must be "unverified", NOT "probe"
    assert view["primary_id"] == "a-row"
    assert view["evidence"] == "unverified", "must not borrow the sibling's better evidence"


# ---------------------------------------------------- C2: no fake presets

def test_c2_legacy_speaker_rows_offer_no_native_presets():
    """The embedded reference is stripped in the profile path, so a native
    label from a legacy speaker-named row would silently produce the engine's
    default voice. Clara/Jonas are library voices, never presets."""
    cat = brief_catalog()
    tts = {g.model: g for g in group_engines(cat["tts"])}
    cb = group_engine_view(tts["chatterbox-fp16"], role="tts")
    assert cb["native_voices"] == [], "no fake native presets from legacy rows"
    assert cb["embeds_reference"] is True


def test_c2_trustworthy_native_presets_are_offered_customvoice_and_voxtral_only():
    cat = brief_catalog()
    tts = {g.model: g for g in group_engines(cat["tts"])}
    cv = group_engine_view(tts["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"], role="tts")
    assert sorted(cv["native_voices"]) == ["ryan", "serena"]
    vox = group_engine_view(tts["mlx-community--Voxtral-4B-TTS-2603-mlx-4bit"], role="tts")
    assert sorted(vox["native_voices"]) == ["casual_male", "de_female", "de_male"]
    fish = group_engine_view(tts["fish-audio-s2-pro-8bit"], role="tts")
    assert fish["native_voices"] == [], "Fish's label is not a stable identity"
    assert native_presets_trustworthy(tts["fish-audio-s2-pro-8bit"].rows[0]) is False


def test_engine_id_is_safe_and_stable():
    key = engine_key(BackendProfile(id="x", role="tts", base_url="http://secret-host.local:9000/v1", model="m"))
    eid = engine_id(key)
    assert "secret-host" not in eid and "/" not in eid
    assert eid == engine_id(key)