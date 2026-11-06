"""compare_profiles.py URL construction regressions.

Live incident (2026-09-12): --url with a trailing slash produced
wss://host//v1/realtime (and, without the fix, the WS path was missing
entirely), so the socket fell through to the StaticFiles mount, whose
scope-type assert turned every WS handshake into HTTP 500. The browser
works because app.js builds the URL itself. The app must NOT need a reroute
workaround — these tests pin the runner-side normalization.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from compare_profiles import (  # noqa: E402
    WS_PATH,
    check_realtimevoice,
    expected_for_set,
    normalize_base_url,
    refusal_code,
    ws_url_for,
)


def test_trailing_slash_stripped():
    assert normalize_base_url("https://host:8443/") == "https://host:8443"
    assert normalize_base_url("https://host:8443///") == "https://host:8443"
    assert normalize_base_url(" http://host:8765/ ") == "http://host:8765"


def test_ws_url_trailing_slash_vs_plain_are_identical():
    a = ws_url_for("https://host:8443/", "profile:anna")
    b = ws_url_for("https://host:8443", "profile:anna")
    assert a == b


def test_ws_url_contains_path_exactly_once():
    url = ws_url_for("https://host:8443/", "")
    assert url == f"wss://host:8443{WS_PATH}?model="
    url2 = ws_url_for("https://host:8443", "profile:x")
    assert url2 == f"wss://host:8443{WS_PATH}?model=profile%3Ax"


def test_ws_url_appends_path_when_missing():
    assert ws_url_for("http://127.0.0.1:8765", "").startswith(f"ws://127.0.0.1:8765{WS_PATH}")


def test_ws_url_does_not_double_append_path():
    url = ws_url_for("https://host:8443/v1/realtime", "profile:x")
    assert url.count(WS_PATH) == 1


def test_ws_url_keeps_base_path_no_double_slash():
    url = ws_url_for("https://host:8443/rtv/", "profile:x")
    assert url == "wss://host:8443/rtv/v1/realtime?model=profile%3Ax"
    assert "//" not in url.split("wss://")[1]


def test_ws_url_model_quoting_no_token_ever():
    url = ws_url_for("https://host:8443/", "profile-test:a1b2c3d4e5f6")
    assert "token" not in url
    assert url.endswith("model=profile-test%3Aa1b2c3d4e5f6")


def _catalog_models():
    return {
        "stt": {"anna-stt": {"model": "asr-1"}},
        "llm": {"anna-llm": {"model": "llm-1"}},
        "tts": {"anna-tts": {"model": "tts-1"}},
    }


def _set(language="de", voice=None):
    return {
        "id": "anna",
        "enabled": True,
        "spec": {
            "stt_profile": "anna-stt", "llm_profile": "anna-llm", "tts_profile": "anna-tts",
            "language": language, "voice": voice or {"kind": "none"},
        },
    }


def test_expected_call_is_derived_from_the_set_and_the_catalog():
    expected = expected_for_set(
        _set(voice={"kind": "native", "native_voice": "ryan"}), _catalog_models(), {"session": {"language": "en"}}
    )
    assert expected["set"] == "anna"
    assert expected["roles"]["tts"] == {"profile": "anna-tts", "model": "tts-1"}
    assert expected["language"] == "de"
    assert expected["voice"] == "ryan"
    # no set language: the server's call default
    assert expected_for_set(_set(language=""), _catalog_models(), {"session": {"language": "en"}})["language"] == "en"


def test_realtimevoice_block_must_name_the_set_and_its_models():
    expected = expected_for_set(_set(), _catalog_models(), {})
    good = {
        "set": "anna",
        "stt": {"profile": "anna-stt", "model": "asr-1"},
        "llm": {"profile": "anna-llm", "model": "llm-1"},
        "tts": {"profile": "anna-tts", "model": "tts-1"},
        "language": "de",
    }
    assert check_realtimevoice(good, expected)[0] is True
    ok, detail = check_realtimevoice({**good, "set": "bert"}, expected)
    assert ok is False and "set" in detail
    ok, detail = check_realtimevoice({**good, "tts": {"profile": "other", "model": "tts-1"}}, expected)
    assert ok is False and "tts profile" in detail


def test_refusal_code_is_read_from_the_handshake_body():
    class Response:
        body = bytearray(b'{"error":{"code":"no_set_selected","message":"x"}}')

    class Refused(Exception):
        response = Response()

    assert refusal_code(Refused()) == "no_set_selected"
    assert refusal_code(RuntimeError("closed")) == ""
