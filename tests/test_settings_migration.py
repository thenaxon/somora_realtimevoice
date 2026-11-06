"""Settings migration (v1 -> v3, v2 -> v3), sticky load errors,
commit-after-write.

Regression focus:
- exact private 0600 backup of the original before the first migration
  write, never overwriting a previously saved backup of the same version,
- v2 -> v3: model ids are rewritten from the model name and every reference
  (sets, their versions, the transcription model) follows; a model's stored
  language becomes its language_format; a set is its active spec (else its
  draft) and is on only when it was available to clients with an active
  revision; presets, voice-routing mappings and the legacy backend triple are
  dropped,
- migration/update failures keep memory and disk consistent (no silent
  empty defaults; sticky SettingsLoadError on repeated get; the original
  file stays untouched).
"""

import asyncio
import json
import os
import stat

import pytest

import realtimevoice.config as config_module
from realtimevoice.config import (
    SETTINGS_V1_BACKUP_NAME,
    SETTINGS_V2_BACKUP_NAME,
    SETTINGS_VERSION,
    BackendProfile,
    Settings,
    SettingsStore,
    _model_id,
    language_for_model,
    migrate_v2_data,
)

V1 = {
    "version": 1,
    "stt": {
        "base_url": "http://old-stt.test/v1",
        "api_key": "sk-v1-stt-key",
        "model": "whisper-v1",
        "language": "de",
    },
    "llm": {
        "base_url": "http://old-llm.test/v1",
        "api_key": "sk-v1-llm-key",
        "model": "qwen3-4b-instruct-2507",
    },
    "tts": {
        "base_url": "http://old-tts.test/v1",
        "api_key": "sk-v1-tts-key",
        "model": "qwen3-tts-1.7b",
        "voice": "ryan",
    },
    "session": {"language": "de", "voice": "ryan"},
}


def _row(pid, role, model, **extra):
    return {"id": pid, "role": role, "base_url": "http://backend.test/v1", "model": model, **extra}


def _spec(stt, llm, tts, **extra):
    return {
        "stt_profile": stt, "llm_profile": llm, "tts_profile": tts,
        "language": "de", "voice": {"kind": "native", "native_voice": "serena"}, **extra,
    }


def _revision(n, spec, at):
    return {"revision": n, "activated_at": at, "spec": spec, "stt": {}, "llm": {}, "tts": {}}


V2 = {
    "version": 2,
    # the stored legacy triple: dropped
    "stt": {"base_url": "http://legacy.test/v1", "model": "legacy-stt", "api_key": "sk-legacy"},
    "llm": {"base_url": "http://legacy.test/v1", "model": "legacy-llm"},
    "tts": {"base_url": "http://legacy.test/v1", "model": "legacy-tts"},
    "session": {"language": "de", "voice": "serena"},
    "profiles": {
        "stt": {
            "stt-baseline": _row("stt-baseline", "stt", "whisper-large-v3", language="de", api_key="sk-stt"),
            "stt-alt": _row("stt-alt", "stt", "Parakeet/TDT 0.6b", language="de-AT"),
        },
        "llm": {
            "llm-baseline": _row("llm-baseline", "llm", "qwen3-4b", language=""),
        },
        "tts": {
            # two rows of the same model: the second gets a suffix
            "chatterbox-clara": _row("chatterbox-clara", "tts", "chatterbox-fp16", language="de"),
            "chatterbox-jonas": _row("chatterbox-jonas", "tts", "chatterbox-fp16", language="auto"),
            "qwen-tts": _row(
                "qwen-tts", "tts", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", language="German",
                voice="serena", capabilities={"voices": ["serena", "ryan"]},
            ),
            # no model name: the id comes from the display name
            "tts-baseline": _row("tts-baseline", "tts", "", display_name="Old Default", language=""),
        },
    },
    "presets": {
        "baseline": {"id": "baseline", "stt": "stt-baseline", "llm": "llm-baseline", "tts": "tts-baseline"},
    },
    "default_preset": "baseline",
    "voice_routing": {"mappings": {"clara": "chatterbox-clara"}},
    "voice_library": {"transcription_stt_profile": ""},
    "conversation_profiles": {
        # active + available: on, speaks with the ACTIVE spec, not the draft
        "live-set": {
            "id": "live-set",
            "display_name": "Live",
            "draft": _spec("stt-alt", "llm-baseline", "chatterbox-jonas"),
            "draft_updated_at": 300.0,
            "active": _revision(2, _spec("stt-baseline", "llm-baseline", "qwen-tts"), 200.0),
            "available_to_clients": True,
            "history": [_revision(1, _spec("stt-alt", "llm-baseline", "chatterbox-clara"), 100.0)],
            "created_at": 50.0,
        },
        # active but not available: off
        "parked-set": {
            "id": "parked-set",
            "display_name": "Parked",
            "draft": _spec("stt-baseline", "llm-baseline", "qwen-tts"),
            "draft_updated_at": 300.0,
            "active": _revision(1, _spec("stt-baseline", "llm-baseline", "qwen-tts"), 200.0),
            "available_to_clients": False,
            "history": [],
            "created_at": 50.0,
        },
        # never activated: the draft is the set, off
        "draft-set": {
            "id": "draft-set",
            "display_name": "Draft only",
            "draft": _spec("stt-alt", "llm-baseline", "chatterbox-clara"),
            "draft_updated_at": 300.0,
            "active": None,
            "available_to_clients": False,
            "history": [],
            "created_at": 50.0,
        },
    },
}


def write_settings(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def store_for(tmp_path):
    return SettingsStore(data_dir=str(tmp_path))


# --------------------------------------------------------------------- v1


def test_v1_migrates_to_v3_with_exact_private_backup(tmp_path):
    write_settings(tmp_path / "settings.json", V1)
    settings = asyncio.run(store_for(tmp_path).get())

    assert settings.version == SETTINGS_VERSION == 3
    assert settings.session.language == "de"
    # the single backend per role became one ordinary model entry, its id
    # derived from the model name; key and voice carried over losslessly
    assert set(settings.profiles["stt"]) == {"whisper-v1"}
    assert set(settings.profiles["llm"]) == {"qwen3-4b-instruct-2507"}
    assert set(settings.profiles["tts"]) == {"qwen3-tts-1.7b"}
    assert settings.profiles["stt"]["whisper-v1"].api_key == "sk-v1-stt-key"
    assert settings.profiles["stt"]["whisper-v1"].language_format == "code"
    assert settings.profiles["tts"]["qwen3-tts-1.7b"].api_key == "sk-v1-tts-key"
    assert settings.profiles["tts"]["qwen3-tts-1.7b"].voice == "ryan"
    # recordings are transcribed with the old listening model
    assert settings.voice_library.transcription_stt_profile == "whisper-v1"
    assert settings.default_set == ""
    # backup: exact bytes, private mode
    backup_path = tmp_path / SETTINGS_V1_BACKUP_NAME
    assert json.loads(backup_path.read_text()) == V1
    assert stat.S_IMODE(os.stat(backup_path).st_mode) & 0o077 == 0
    # the migrated file is v3 and stores no legacy triple
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["version"] == 3
    assert not {"stt", "llm", "tts", "presets", "default_preset"} & set(stored)


def test_v1_migration_backup_never_overwritten(tmp_path):
    write_settings(tmp_path / "settings.json", V1)
    original_backup = tmp_path / SETTINGS_V1_BACKUP_NAME
    original_backup.write_text('{"keep": "first-original"}')
    asyncio.run(store_for(tmp_path).get())
    assert json.loads(original_backup.read_text()) == {"keep": "first-original"}


# --------------------------------------------------------------------- v2


def _migrated(tmp_path):
    write_settings(tmp_path / "settings.json", V2)
    return asyncio.run(store_for(tmp_path).get())


def test_v2_model_ids_follow_model_names_and_references_follow(tmp_path):
    settings = _migrated(tmp_path)
    assert set(settings.profiles["stt"]) == {"whisper-large-v3", "parakeet-tdt-0.6b"}
    assert set(settings.profiles["llm"]) == {"qwen3-4b"}
    # same model twice: sorted by the old id, the second one gets "-2"
    assert set(settings.profiles["tts"]) == {
        "chatterbox-fp16", "chatterbox-fp16-2", "qwen3-tts-12hz-1.7b-customvoice-8bit", "old-default",
    }
    for role, rows in settings.profiles.items():
        for pid, row in rows.items():
            assert row.id == pid and row.role == role
    # an entry keeps everything but its id
    assert settings.profiles["stt"]["whisper-large-v3"].api_key == "sk-stt"
    qwen = settings.profiles["tts"]["qwen3-tts-12hz-1.7b-customvoice-8bit"]
    assert qwen.voice == "serena" and qwen.capabilities.voices == ["serena", "ryan"]
    # the old trusted-model list still marks its native labels honoured
    assert qwen.capabilities.native_voice_labels == "honoured"

    live = settings.conversation_profiles["live-set"]
    assert live.spec.stt_profile == "whisper-large-v3"
    assert live.spec.llm_profile == "qwen3-4b"
    assert live.spec.tts_profile == "qwen3-tts-12hz-1.7b-customvoice-8bit"
    # earlier revisions became Restore versions, their references renamed too
    assert [v.spec.tts_profile for v in live.history] == ["chatterbox-fp16"]
    assert [v.spec.stt_profile for v in live.history] == ["parakeet-tdt-0.6b"]
    assert live.history[0].saved_at == 100.0
    assert settings.conversation_profiles["draft-set"].spec.tts_profile == "chatterbox-fp16"
    # the old baseline listening model transcribes recordings
    assert settings.voice_library.transcription_stt_profile == "whisper-large-v3"


def test_v2_language_becomes_language_format(tmp_path):
    settings = _migrated(tmp_path)
    tts = settings.profiles["tts"]
    assert settings.profiles["stt"]["whisper-large-v3"].language_format == "code"   # "de"
    assert settings.profiles["stt"]["parakeet-tdt-0.6b"].language_format == "code"  # "de-AT"
    assert settings.profiles["llm"]["qwen3-4b"].language_format == "code"           # ""
    assert tts["chatterbox-fp16-2"].language_format == "code"                       # "auto"
    assert tts["qwen3-tts-12hz-1.7b-customvoice-8bit"].language_format == "name"   # "German"
    stored = json.loads((tmp_path / "settings.json").read_text())
    for rows in stored["profiles"].values():
        for row in rows.values():
            assert "language" not in row


def test_v2_sets_take_active_spec_else_draft_and_enabled_needs_both(tmp_path):
    sets = _migrated(tmp_path).conversation_profiles
    assert set(sets) == {"live-set", "parked-set", "draft-set"}
    # the active revision wins over a newer draft
    assert sets["live-set"].spec.stt_profile == "whisper-large-v3"
    assert sets["live-set"].enabled is True
    assert sets["parked-set"].enabled is False
    assert sets["draft-set"].enabled is False
    assert sets["draft-set"].spec.stt_profile == "parakeet-tdt-0.6b"
    assert sets["draft-set"].history == []
    assert sets["live-set"].display_name == "Live"
    assert sets["live-set"].updated_at == 200.0


def test_v2_presets_mappings_and_legacy_triple_dropped(tmp_path):
    _migrated(tmp_path)
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["version"] == 3
    for gone in ("presets", "default_preset", "voice_routing", "stt", "llm", "tts"):
        assert gone not in stored, gone
    assert stored["default_set"] == ""
    assert "sk-legacy" not in json.dumps(stored)
    for cp in stored["conversation_profiles"].values():
        assert not {"draft", "active", "available_to_clients", "draft_updated_at"} & set(cp)


def test_v2_backup_written_once_and_exact(tmp_path):
    write_settings(tmp_path / "settings.json", V2)
    original = (tmp_path / "settings.json").read_text()
    asyncio.run(store_for(tmp_path).get())
    backup = tmp_path / SETTINGS_V2_BACKUP_NAME
    assert backup.read_text() == original
    assert stat.S_IMODE(os.stat(backup).st_mode) & 0o077 == 0
    # the migrated v3 file loads again unchanged (no second migration)
    migrated = (tmp_path / "settings.json").read_text()
    asyncio.run(store_for(tmp_path).get())
    assert (tmp_path / "settings.json").read_text() == migrated
    # another v2 file later (e.g. a restored copy) never overwrites the
    # first backup
    write_settings(tmp_path / "settings.json", dict(V2, session={"language": "en"}))
    asyncio.run(store_for(tmp_path).get())
    assert backup.read_text() == original


def test_v2_transcription_model_kept_when_chosen_else_first_listening_model():
    data = json.loads(json.dumps(V2))
    data["voice_library"] = {"transcription_stt_profile": "stt-alt"}
    assert migrate_v2_data(data)["voice_library"]["transcription_stt_profile"] == "parakeet-tdt-0.6b"
    data = json.loads(json.dumps(V2))
    data["profiles"]["stt"].pop("stt-baseline")
    data["conversation_profiles"] = {}
    # no choice and no old baseline: the first listening model
    assert migrate_v2_data(data)["voice_library"]["transcription_stt_profile"] == "parakeet-tdt-0.6b"
    data["profiles"]["stt"] = {}
    assert migrate_v2_data(data)["voice_library"]["transcription_stt_profile"] == ""


def test_v2_migration_that_cannot_validate_leaves_file_untouched(tmp_path):
    bad = json.loads(json.dumps(V2))
    bad["profiles"]["llm"]["llm-baseline"]["base_url"] = "ftp://not-http"
    write_settings(tmp_path / "settings.json", bad)
    original = (tmp_path / "settings.json").read_text()
    store = store_for(tmp_path)
    for _ in range(2):  # sticky, never silent defaults
        with pytest.raises(Exception) as excinfo:
            asyncio.run(store.get())
        assert "migration produced invalid settings" in str(excinfo.value)
    assert (tmp_path / "settings.json").read_text() == original
    # the original is also kept as the backup
    assert (tmp_path / SETTINGS_V2_BACKUP_NAME).read_text() == original


def test_v2_migration_write_failure_keeps_original(tmp_path):
    write_settings(tmp_path / "settings.json", V2)
    original = (tmp_path / "settings.json").read_text()
    real_atomic = config_module._atomic_write_private

    def failing_write(path, payload):
        if path.endswith("settings.json"):
            raise OSError("disk full")
        return real_atomic(path, payload)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(config_module, "_atomic_write_private", failing_write)
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(store_for(tmp_path).get())
        assert "migration write failed" in str(excinfo.value)
    finally:
        monkey.undo()
    assert (tmp_path / "settings.json").read_text() == original


# ---------------------------------------------------------- id + language


def test_model_id_from_model_name_with_collision_suffix():
    taken: set[str] = set()
    assert _model_id("Qwen/Qwen3 TTS", "", taken) == "qwen-qwen3-tts"
    assert _model_id("Qwen/Qwen3 TTS", "", taken) == "qwen-qwen3-tts-2"
    assert _model_id("Qwen/Qwen3 TTS", "", taken) == "qwen-qwen3-tts-3"
    assert _model_id("", "My Speaker", taken) == "my-speaker"
    assert _model_id("", "", taken) == "model"
    assert len(_model_id("x" * 80, "", taken)) <= 64
    assert len(_model_id("x" * 80, "", taken)) <= 64
    for pid in taken:
        BackendProfile(id=pid)  # every generated id is a valid model id


@pytest.mark.parametrize(
    "code,fmt,expected",
    [
        ("de", "code", "de"),
        ("de-AT", "code", "de"),
        ("de-AT", "name", "German"),
        ("en", "name", "English"),
        ("xx", "name", "xx"),   # unknown code: sent as given
        ("de", "none", ""),
        ("", "code", ""),
        ("auto", "name", ""),
    ],
)
def test_language_for_model(code, fmt, expected):
    assert language_for_model(code, fmt) == expected


# --------------------------------------------------------- load + update


def test_malformed_settings_is_sticky_error_not_defaults(tmp_path):
    (tmp_path / "settings.json").write_text("{not json at all")
    store = store_for(tmp_path)
    for _ in range(3):
        with pytest.raises(Exception) as excinfo:
            asyncio.run(store.get())
        assert "malformed" in str(excinfo.value)


def test_empty_settings_file_errors(tmp_path):
    (tmp_path / "settings.json").write_text("")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(store_for(tmp_path).get())
    assert "empty" in str(excinfo.value)


def test_newer_settings_version_never_silently_defaults(tmp_path):
    write_settings(tmp_path / "settings.json", {"version": 4, "session": {}})
    store = store_for(tmp_path)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(store.get())
    assert "version 4" in str(excinfo.value)
    with pytest.raises(Exception):  # sticky
        asyncio.run(store.get())


def test_v1_migration_write_failure_keeps_original_and_sticky_error(tmp_path):
    write_settings(tmp_path / "settings.json", V1)
    original = (tmp_path / "settings.json").read_text()
    store = store_for(tmp_path)
    real_atomic = config_module._atomic_write_private
    calls = {"n": 0}

    def failing_write(path, payload):
        calls["n"] += 1
        if path.endswith("settings.json"):
            raise OSError("disk full")
        return real_atomic(path, payload)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(config_module, "_atomic_write_private", failing_write)
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(store.get())
        assert "migration write failed" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo2:
            asyncio.run(store.get())
        assert "migration write failed" in str(excinfo2.value)
    finally:
        monkey.undo()
    assert (tmp_path / "settings.json").read_text() == original
    assert calls["n"] >= 2


def test_update_save_failure_leaves_memory_unchanged(tmp_path):
    store = store_for(tmp_path)
    asyncio.run(store.get())

    def failing_write(path, payload):
        if path.endswith("settings.json"):
            raise OSError("disk full")
        raise AssertionError("unexpected write target")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(config_module, "_atomic_write_private", failing_write)
    try:
        with pytest.raises(OSError):
            asyncio.run(store.update({"session": {"language": "en"}}))
    finally:
        monkey.undo()
    assert asyncio.run(store.get()).session.language == "de"
    asyncio.run(store.update({"session": {"language": "en"}}))
    assert asyncio.run(store_for(tmp_path).get()).session.language == "en"


def test_model_mutation_save_failure_leaves_memory_unchanged(tmp_path):
    store = store_for(tmp_path)
    asyncio.run(store.get())

    def add_profile(settings):
        settings.profiles["tts"]["tts-x"] = BackendProfile(id="tts-x", model="m")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        config_module, "_atomic_write_private",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )
    try:
        with pytest.raises(OSError):
            asyncio.run(store.mutate(add_profile))
    finally:
        monkey.undo()
    assert "tts-x" not in asyncio.run(store.get()).profiles["tts"]


def test_update_rejects_version_change(tmp_path):
    store = store_for(tmp_path)
    asyncio.run(store.get())
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(store.update({"version": 2}))
    assert "version" in str(excinfo.value)


def test_fresh_start_has_defaults(tmp_path):
    settings = asyncio.run(store_for(tmp_path).get())
    assert settings.version == 3
    assert settings.session.model_dump() == Settings().session.model_dump()
    assert settings.profiles == {"stt": {}, "llm": {}, "tts": {}}
    assert settings.default_set == ""
